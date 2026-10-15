//! The laws of the gadget builder.
//!
//! - **one term, two models**: the witness satisfies the circuit
//!   (completeness), the circuit does not depend on the witness, and
//!   `evaluate` is the function the constraints define;
//! - **soundness, sampled**: changing a constrained cell breaks the circuit;
//! - **composition**: `evaluate(Chain(a, b)) = evaluate(b) . evaluate(a)`,
//!   `Repeat<g, 3>` is `g` three times, and the empty repetition is the
//!   identity;
//! - **the builder's errors**, one gadget each: a constant that varies
//!   between instances, a cell written twice, a false constraint, a gadget
//!   taking more rows than it declares, a copy of an expression, a copy
//!   outside the permuted columns, an unplaced input, and a circuit too
//!   large for `N` or `D`.

use arrabbiata_field::Field;
use arrabbiata_gadget::{
    Array, Cell, Chain, Env, Error, Layout, Of, Repeat, Scalar, TypedGadget, circuit, evaluate,
    witness,
};
use arrabbiata_pasta::Fq as F;
use proptest::prelude::*;

const LAYOUT: Layout = Layout {
    witness: 3,
    fixed: 1,
    permuted: 2,
};
const N: usize = 8;
const D: usize = 3;

/// `x -> x^2 + c`, `x` on column 0, the result on column 0 of the next row.
#[derive(Clone, Copy, Debug)]
struct SquarePlus {
    c: F,
}

impl TypedGadget<F> for SquarePlus {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "square-plus";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let c = env.fixed(Cell::new(0, 0), self.c);
        env.define(Cell::new(0, 1), x.clone() * x + c)
    }
}

/// A gadget placing its private input at column 0 of its first row, then
/// running `inner`, then making the result public.
#[derive(Clone, Copy, Debug)]
struct Program<G> {
    inner: G,
}

impl<G> TypedGadget<F> for Program<G>
where
    G: TypedGadget<F, Input = Scalar, Output = Scalar>,
{
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "program";
    const ROWS: usize = G::ROWS + 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        let y = env.call(&self.inner, x);
        env.public(&y);
        y
    }
}

fn square_plus(x: F, c: F) -> F {
    x * x + c
}

fn arb_f() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(F::from_u64)
}

type Three = Chain<SquarePlus, Chain<SquarePlus, SquarePlus>>;

fn three(c: [F; 3]) -> Program<Three> {
    Program {
        inner: Chain {
            first: SquarePlus { c: c[0] },
            second: Chain {
                first: SquarePlus { c: c[1] },
                second: SquarePlus { c: c[2] },
            },
        },
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// The witness satisfies the circuit, the circuit is the one built
    /// without a witness, and the output is the composite function.
    #[test]
    fn one_term_two_models(x in arb_f(), c in prop::array::uniform3(arb_f())) {
        let gadget = three(c);
        let (compiled, output) = witness::<F, _, N, D>(&gadget, LAYOUT, &x).unwrap();
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        prop_assert_eq!(&compiled.circuit, &circuit::<F, _, N, D>(&gadget, LAYOUT).unwrap());
        let expected = square_plus(square_plus(square_plus(x, c[0]), c[1]), c[2]);
        prop_assert_eq!(output, expected);
        prop_assert_eq!(evaluate(&gadget, LAYOUT, &x).unwrap(), expected);
        prop_assert_eq!(compiled.public, vec![expected]);
    }

    /// Changing any cell the gadget wrote breaks the circuit.
    #[test]
    fn a_changed_cell_breaks_the_circuit(x in arb_f(), c in prop::array::uniform3(arb_f()), row in 0usize..4, delta in 1u64..1000) {
        let gadget = three(c);
        let (compiled, _) = witness::<F, _, N, D>(&gadget, LAYOUT, &x).unwrap();
        let mut columns = compiled.columns.clone();
        columns[0][row] = columns[0][row] + F::from_u64(delta);
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }

    /// Chain is composition, Repeat is iteration.
    #[test]
    fn composition(x in arb_f(), c in arb_f()) {
        // Each gadget runs inside a program that places its input first: a
        // gadget given a raw input is rejected (see `misuses_are_rejected`).
        let g = SquarePlus { c };
        let once = |v: F| evaluate(&Program { inner: g }, LAYOUT, &v).unwrap();
        let chained = Program { inner: Chain { first: g, second: g } };
        prop_assert_eq!(evaluate(&chained, LAYOUT, &x).unwrap(), once(once(x)));
        let repeated = Program { inner: Repeat::<_, 3> { gadget: g } };
        prop_assert_eq!(evaluate(&repeated, LAYOUT, &x).unwrap(), once(once(once(x))));
        let none = Program { inner: Repeat::<_, 0> { gadget: g } };
        prop_assert_eq!(evaluate(&none, LAYOUT, &x).unwrap(), x);
    }
}

/// A round with its constant in the constraint instead of a fixed column.
#[derive(Clone, Copy, Debug)]
struct BakedConstant {
    c: F,
}

impl TypedGadget<F> for BakedConstant {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "baked";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let c = env.constant(self.c);
        env.define(Cell::new(0, 1), x + c)
    }
}

#[test]
fn a_varying_constant_is_rejected() {
    let gadget = Program {
        inner: Chain {
            first: BakedConstant { c: F::ONE },
            second: BakedConstant { c: F::from_u64(2) },
        },
    };
    let found = circuit::<F, _, N, D>(&gadget, LAYOUT).unwrap_err();
    assert_eq!(found, Error::Inconsistent { gadget: "baked" });
    // The same constant in every instance is fine.
    let same = Program {
        inner: Chain {
            first: BakedConstant { c: F::ONE },
            second: BakedConstant { c: F::ONE },
        },
    };
    assert!(circuit::<F, _, N, D>(&same, LAYOUT).is_ok());
}

/// Gadgets misusing the builder, one way each.
#[derive(Clone, Copy, Debug)]
enum Misuse {
    Overwrite,
    False,
    CopyExpression,
    CopyUnpermuted,
    Unplaced,
    Cubic,
}

impl TypedGadget<F> for Misuse {
    type Input = Scalar;
    type Output = Unit2;
    const NAME: &'static str = "misuse";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Unit2, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        match self {
            Self::Overwrite => {
                env.define(Cell::new(0, 0), x.clone());
            }
            Self::False => {
                let one = env.constant(F::ONE);
                env.assert_equal(x.clone(), x.clone() + one);
            }
            Self::CopyExpression => {
                env.copy(Cell::new(1, 0), &(x.clone() + x.clone()));
            }
            Self::CopyUnpermuted => {
                env.copy(Cell::new(2, 0), &x);
            }
            Self::Unplaced => env.assert_zero(input.clone() - x.clone()),
            Self::Cubic => {
                env.define(
                    Cell::new(1, 0),
                    x.clone() * x.clone() * x.clone() * x.clone(),
                );
            }
        }
        [x.clone(), x]
    }
}

type Unit2 = Array<2>;

#[test]
fn misuses_are_rejected() {
    let input = F::from_u64(5);
    let cases = [
        (
            Misuse::Overwrite,
            Error::Overwrite {
                gadget: "misuse",
                column: 0,
                row: 0,
            },
        ),
        (
            Misuse::False,
            Error::Unsatisfied {
                gadget: "misuse",
                row: 0,
            },
        ),
        (Misuse::CopyExpression, Error::NotACell { gadget: "misuse" }),
        (
            Misuse::CopyUnpermuted,
            Error::NotPermuted {
                gadget: "misuse",
                column: 2,
            },
        ),
        (Misuse::Unplaced, Error::Unplaced { gadget: "misuse" }),
        (
            Misuse::Cubic,
            Error::Degree {
                gadget: "misuse",
                degree: 4,
            },
        ),
    ];
    for (gadget, expected) in cases {
        let found = witness::<F, _, N, D>(&gadget, LAYOUT, &input).unwrap_err();
        assert_eq!(found, expected, "{gadget:?}");
    }
}

/// A gadget declaring one row while its sub-gadgets take two.
#[derive(Clone, Copy, Debug)]
struct TooFewRows;

impl TypedGadget<F> for TooFewRows {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "too-few-rows";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let g = SquarePlus { c: F::ONE };
        let y = env.call(&g, x);
        env.call(&g, y)
    }
}

#[test]
fn rows_are_checked() {
    let gadget = Program { inner: TooFewRows };
    let found = circuit::<F, _, N, D>(&gadget, LAYOUT).unwrap_err();
    assert_eq!(
        found,
        Error::Rows {
            gadget: "too-few-rows"
        }
    );
}

#[test]
fn too_many_variables() {
    let gadget = three([F::ONE; 3]);
    // The gate reads two witness cells and one fixed cell.
    let found = circuit::<F, _, 2, D>(&gadget, LAYOUT).unwrap_err();
    assert_eq!(found, Error::Variables { needed: 3 });
}

/// `place` copies a permuted cell and defines anything else.
#[derive(Clone, Copy, Debug)]
struct Placing;

impl TypedGadget<F> for Placing {
    type Input = Scalar;
    type Output = Array<3>;
    const NAME: &'static str = "placing";
    const ROWS: usize = 2;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        input: Of<Scalar, E::Var>,
    ) -> Of<Array<3>, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        // A cell of a permuted column into a permuted column: a copy.
        let copied = env.place(Cell::new(1, 1), &x);
        // An expression: a definition.
        let doubled = env.place(Cell::new(2, 1), &(x.clone() + x));
        env.public(&copied.clone());
        [copied.clone(), doubled, copied]
    }
}

#[test]
fn place_copies_or_defines() {
    let input = F::from_u64(7);
    // Making column 1 public is not allowed: only the first column.
    let found = witness::<F, _, N, D>(&Placing, LAYOUT, &input).unwrap_err();
    assert_eq!(found, Error::NotACell { gadget: "placing" });
}

/// The same, made public correctly, and checked on the witness.
#[derive(Clone, Copy, Debug)]
struct PlacingPublic;

impl TypedGadget<F> for PlacingPublic {
    type Input = Scalar;
    type Output = Array<2>;
    const NAME: &'static str = "placing-public";
    const ROWS: usize = 2;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        input: Of<Scalar, E::Var>,
    ) -> Of<Array<2>, E::Var> {
        let x = env.advice(Cell::new(1, 0), [&input], |[v]| v);
        let copied = env.place(Cell::new(0, 1), &x);
        let doubled = env.place(Cell::new(2, 1), &(x.clone() + x));
        env.public(&copied);
        [copied, doubled]
    }
}

#[test]
fn a_copy_is_enforced_by_the_permutation() {
    let input = F::from_u64(7);
    let (compiled, [copied, doubled]) =
        witness::<F, _, N, D>(&PlacingPublic, LAYOUT, &input).unwrap();
    assert_eq!((copied, doubled), (input, input + input));
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
    // The copy has no gate: only the definition does.
    assert_eq!(compiled.circuit.gates().len(), 1);
    // Move the copied cell, and the public input with it: no gate reads
    // the cell, so only the permutation catches it.
    let mut columns = compiled.columns.clone();
    let moved = columns[0][1] + F::ONE;
    columns[0][1] = moved;
    assert!(!compiled.circuit.is_satisfied(&columns, &[moved]));
}

/// A gadget placing a cell where it already is, twice.
#[derive(Clone, Copy, Debug)]
struct PlaceInPlace;

impl TypedGadget<F> for PlaceInPlace {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "place-in-place";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        let once = env.place(Cell::new(0, 0), &x);
        let twice = env.place(Cell::new(0, 0), &once);
        env.public(&twice);
        twice
    }
}

/// Placing is idempotent: a cell placed where it is is not written again.
#[test]
fn placing_in_place_is_the_identity() {
    let input = F::from_u64(11);
    let (compiled, output) = witness::<F, _, N, D>(&PlaceInPlace, LAYOUT, &input).unwrap();
    assert_eq!(output, input);
    assert!(compiled.circuit.gates().is_empty());
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
}

/// A gadget copying a cell onto itself, in a column the permutation does
/// not cover.
#[derive(Clone, Copy, Debug)]
struct CopyInPlace;

impl TypedGadget<F> for CopyInPlace {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "copy-in-place";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let x = env.advice(Cell::new(2, 0), [&input], |[v]| v);
        env.copy(Cell::new(2, 0), &x)
    }
}

/// A copy onto the same cell is that cell: no copy, so no permutation needed.
#[test]
fn copying_in_place_is_the_identity() {
    let input = F::from_u64(13);
    assert_eq!(evaluate(&CopyInPlace, LAYOUT, &input).unwrap(), input);
}
