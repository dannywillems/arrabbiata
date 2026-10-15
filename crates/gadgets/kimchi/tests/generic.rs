//! Kimchi's Generic gate against Kimchi.
//!
//! - Kimchi's own test circuit (`generic::testing::create_circuit` and
//!   `fill_in_witness`): public inputs, 10 rows of `Add` (right
//!   coefficient 3) with `Mul` (multiplication coefficient 2), and 10 rows
//!   of `Const(3)` with `Const(5)`, on Kimchi's witness. The coefficients
//!   are the ones `create_generic_gadget` sets, the witness satisfies the
//!   circuit, and changing an output breaks it;
//! - the typed operations compute what their half-gate pins;
//! - a proof of knowledge of a root: `x^3 + x + 5 = 35`.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit, evaluate, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    generic::{Generic, GenericPair, GenericRow, GenericSpec},
};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

/// The column variables of the gate: columns 0-5 and coefficients 0-9.
const VARIABLES: usize = 16;
/// `c3 w0 w1`.
const DEGREE: usize = 3;

/// The public inputs of Kimchi's test circuit.
const PUBLIC: usize = 3;
/// Its rows of each kind.
const REPEATS: usize = 10;

/// The Add and Mul half-gates of Kimchi's test circuit.
fn add_mul() -> GenericRow<Fp> {
    let add = GenericSpec::Add {
        left: None,
        right: Some(Fp::from_u64(3)),
        output: None,
    };
    let mul = GenericSpec::Mul {
        output: None,
        mul: Some(Fp::from_u64(2)),
    };
    GenericRow::new(add, Some(mul))
}

/// Its two constants.
fn consts() -> GenericRow<Fp> {
    GenericRow::new(
        GenericSpec::Const(Fp::from_u64(3)),
        Some(GenericSpec::Const(Fp::from_u64(5))),
    )
}

/// Kimchi's `fill_in_witness` on the Add and Mul rows, then on the Const
/// rows.
fn kimchi_witness() -> ([u64; 6], [u64; 6]) {
    let add_mul = [11, 23, 11 + 23 * 3, 11, 23, 11 * 23 * 2];
    let consts = [3, 0, 0, 5, 0, 0];
    (add_mul, consts)
}

/// A public input on column 0 of its row.
#[derive(Clone, Copy, Debug)]
struct PublicRow;

impl TypedGadget<Fp> for PublicRow {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "public-row";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, p: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let cell = env.advice(Cell::new(0, 0), [&p], |[v]| v);
        env.public(&cell);
        cell
    }
}

/// A Generic row on given values, as `fill_in_witness` writes them.
#[derive(Clone, Copy, Debug)]
struct WitnessRow {
    gate: GenericRow<Fp>,
    values: [Fp; 6],
}

impl TypedGadget<Fp> for WitnessRow {
    type Input = Unit;
    type Output = Unit;
    const NAME: &'static str = "witness-row";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, (): Of<Unit, E::Var>) -> Of<Unit, E::Var> {
        let cells = core::array::from_fn(|c| {
            let value = self.values[c];
            env.advice(Cell::new(c, 0), [], |[]| value)
        });
        env.call(&self.gate, cells);
    }
}

/// Kimchi's test circuit: the public rows, then the gate rows.
#[derive(Clone, Debug)]
struct KimchiTest {
    rows: Vec<WitnessRow>,
}

impl TypedGadget<Fp> for KimchiTest {
    type Input = Array<PUBLIC>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-generic-test";
    const ROWS: usize = PUBLIC + 2 * REPEATS;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        public: Of<Array<PUBLIC>, E::Var>,
    ) -> Of<Unit, E::Var> {
        for p in public {
            env.call(&PublicRow, p);
        }
        for row in &self.rows {
            env.call(row, ());
        }
    }
}

fn kimchi_test() -> KimchiTest {
    let (add_mul_values, const_values) = kimchi_witness();
    let mut rows = Vec::new();
    for _ in 0..REPEATS {
        let gate = add_mul();
        let values = add_mul_values.map(Fp::from_u64);
        rows.push(WitnessRow { gate, values });
    }
    for _ in 0..REPEATS {
        let gate = consts();
        let values = const_values.map(Fp::from_u64);
        rows.push(WitnessRow { gate, values });
    }
    KimchiTest { rows }
}

#[test]
fn kimchis_coefficients() {
    let one = Fp::ONE;
    let zero = Fp::ZERO;
    let three = Fp::from_u64(3);
    let two = Fp::from_u64(2);
    let five = Fp::from_u64(5);
    assert_eq!(
        add_mul().coefficients,
        [one, three, -one, zero, zero, zero, zero, -one, two, zero]
    );
    assert_eq!(
        consts().coefficients,
        [one, zero, zero, zero, -three, one, zero, zero, zero, -five]
    );
    let plus = GenericSpec::Plus(five).coefficients();
    assert_eq!(plus, [one, zero, -one, zero, five]);
}

#[test]
fn kimchis_test_circuit() {
    let gadget = kimchi_test();
    let public = [Fp::from_u64(7), Fp::from_u64(8), Fp::from_u64(9)];
    let (compiled, ()) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &public).unwrap();
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
    assert_eq!(compiled.public, public.to_vec());
    // One selector, on the 20 gate rows, and the coefficients of each row.
    let c = &compiled.circuit;
    assert_eq!(c.selectors().len(), 1);
    let selected: Vec<usize> = (0..c.num_rows())
        .filter(|r| c.selectors()[0][*r] == Fp::ONE)
        .collect();
    let gate_rows: Vec<usize> = (PUBLIC..PUBLIC + 2 * REPEATS).collect();
    assert_eq!(selected, gate_rows);
    for (i, witness_row) in gadget.rows.iter().enumerate() {
        let row = PUBLIC + i;
        let found: [Fp; 10] = core::array::from_fn(|k| c.fixed()[k][row]);
        assert_eq!(found, witness_row.gate.coefficients);
    }
    // Changing the output of an Add or of a Mul breaks the circuit.
    for column in [2, 5] {
        let mut columns = compiled.columns.clone();
        columns[column][PUBLIC] = columns[column][PUBLIC] + Fp::ONE;
        assert!(!c.is_satisfied(&columns, &compiled.public));
    }
}

/// One operation of a program placing its inputs at columns 0 and 1.
#[derive(Clone, Copy, Debug)]
struct One {
    spec: GenericSpec<Fp>,
}

impl TypedGadget<Fp> for One {
    type Input = Array<2>;
    type Output = Scalar;
    const NAME: &'static str = "one";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&input[i]], |[v]| v));
        env.call(&Generic { spec: self.spec }, cells)
    }
}

/// Two operations of a program placing their inputs at columns 0, 1, 3, 4.
#[derive(Clone, Copy, Debug)]
struct Two {
    pair: GenericPair<Fp>,
}

impl TypedGadget<Fp> for Two {
    type Input = Array<4>;
    type Output = Array<2>;
    const NAME: &'static str = "two";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<4>, E::Var>,
    ) -> Of<Array<2>, E::Var> {
        let at = [0, 1, 3, 4];
        let cells: [E::Var; 4] =
            core::array::from_fn(|i| env.advice(Cell::new(at[i], 0), [&input[i]], |[v]| v));
        let [a, b, c, d] = cells;
        env.call(&self.pair, ([a, b], [c, d]))
    }
}

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<u64>().prop_map(Fp::from_u64)
}

fn arb_spec() -> impl Strategy<Value = GenericSpec<Fp>> {
    let some = || prop::option::of(arb_fp().prop_filter("non-zero", |x| *x != Fp::ZERO));
    prop_oneof![
        (some(), some(), some()).prop_map(|(left, right, output)| GenericSpec::Add {
            left,
            right,
            output
        }),
        (some(), some()).prop_map(|(output, mul)| GenericSpec::Mul { output, mul }),
        arb_fp().prop_map(GenericSpec::Plus),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// An operation's output is the solution of its half-gate, and the
    /// witness satisfies the circuit.
    #[test]
    fn operations_pin_their_output(spec in arb_spec(), a in arb_fp(), b in arb_fp()) {
        let gadget = One { spec };
        let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &[a, b]).unwrap();
        let [l, r, o, m, c] = spec.coefficients();
        prop_assert_eq!(l * a + r * b + o * out + m * a * b + c, Fp::ZERO);
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
    }

    /// The two halves of a row are independent operations.
    #[test]
    fn a_pair_is_two_operations(first in arb_spec(), second in arb_spec(), x in prop::array::uniform4(arb_fp())) {
        let pair = GenericPair { first, second };
        let [p, q] = evaluate(&Two { pair }, LAYOUT, &x).unwrap();
        let alone = |spec, a, b| evaluate(&One { spec }, LAYOUT, &[a, b]).unwrap();
        prop_assert_eq!(p, alone(first, x[0], x[1]));
        prop_assert_eq!(q, alone(second, x[2], x[3]));
    }
}

#[test]
fn a_constant_checks_its_input() {
    let five = Fp::from_u64(5);
    let gadget = One {
        spec: GenericSpec::Const(five),
    };
    assert_eq!(evaluate(&gadget, LAYOUT, &[five, Fp::ZERO]).unwrap(), five);
    assert!(evaluate(&gadget, LAYOUT, &[five + Fp::ONE, Fp::ZERO]).is_err());
}

/// `x^3 + x + 5`, made public, for a private `x`.
#[derive(Clone, Copy, Debug)]
struct Cubic;

impl TypedGadget<Fp> for Cubic {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "cubic";
    const ROWS: usize = 5;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let mul = Generic {
            spec: GenericSpec::Mul {
                output: None,
                mul: None,
            },
        };
        let add = Generic {
            spec: GenericSpec::Add {
                left: None,
                right: None,
                output: None,
            },
        };
        let plus_five = Generic {
            spec: GenericSpec::Plus(Fp::from_u64(5)),
        };
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        let square = env.call(&mul, [x.clone(), x.clone()]);
        let cube = env.call(&mul, [square, x.clone()]);
        let sum = env.call(&add, [cube, x]);
        let y = env.call(&plus_five, [sum.clone(), sum]);
        // Public inputs are cells of column 0: copy the result there.
        let result = env.copy(Cell::new(0, 4), &y);
        env.public(&result);
        result
    }
}

fn ipa(size: usize) -> Ipa<Vesta, PoseidonSponge<Fp>> {
    let mut generators = vesta_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = pallas_sponge(IPA_DOMAIN);
    Ipa::<Vesta, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_root_is_proved() {
    let root = Fp::from_u64(3);
    let (compiled, y) = witness::<Fp, _, VARIABLES, DEGREE>(&Cubic, LAYOUT, &root).unwrap();
    let thirty_five = Fp::from_u64(35);
    assert_eq!(y, thirty_five);
    let pcs = ipa(max_coefficients(&compiled.circuit));
    let (prover_key, verifier_key) = setup(&compiled.circuit, &pcs).unwrap();
    let (proof, _) = prove(
        &prover_key,
        &pcs,
        pallas_sponge(PROOF_DOMAIN),
        &compiled.public,
        &compiled.columns,
    )
    .unwrap();
    let claimed = [thirty_five];
    assert!(
        verify(
            &verifier_key,
            &pcs,
            pallas_sponge(PROOF_DOMAIN),
            &claimed,
            proof.clone()
        )
        .is_ok()
    );
    let another = [thirty_five + Fp::ONE];
    assert!(
        verify(
            &verifier_key,
            &pcs,
            pallas_sponge(PROOF_DOMAIN),
            &another,
            proof
        )
        .is_err()
    );
}
