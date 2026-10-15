//! Each gate's figures against Kimchi's.
//!
//! For every gate, the gadget's gate kind (one selector) has Kimchi's
//! figures:
//!
//! - the number of constraints, Kimchi's `CONSTRAINTS` constant of the
//!   gate's `Argument` (quoted with its file and line in the local checkout
//!   of o1-labs/proof-systems, read 2026-10-15);
//! - the degree of each constraint, in Kimchi's order, as Kimchi's
//!   `Expr::degree(1, 0)` counts it (a cell has degree 1, a constant 0):
//!   derived by reading each gate's `constraint_checks`, to be confirmed by
//!   running Kimchi;
//! - the cells the constraints read: the `witness_curr` and `witness_next`
//!   columns, and the coefficient columns, that `constraint_checks` reads.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Cell, Env, Of, TypedGadget, Unit, figures};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    complete_add::CompleteAdd,
    generic::{GenericRow, GenericSpec},
    poseidon::PoseidonPermutation,
    varbasemul::VarBaseMul,
};
use arrabbiata_pasta::Fp;
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, pallas_base};

/// A gate's figures in Kimchi.
struct Kimchi {
    /// The gadget's name here.
    gadget: &'static str,
    /// `CONSTRAINTS`, and where Kimchi declares it.
    constraints: usize,
    source: &'static str,
    /// The degree of each constraint.
    degrees: Vec<usize>,
    /// The columns read on the gate's row, and on the next.
    current: Vec<usize>,
    next: Vec<usize>,
    /// The coefficient columns read.
    coefficients: Vec<usize>,
}

fn range(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn kimchi() -> Vec<Kimchi> {
    let varbasemul_degrees = core::iter::once(1)
        .chain((0..5).flat_map(|_| [2, 2, 6, 4]))
        .collect();
    vec![
        Kimchi {
            gadget: "kimchi-generic",
            constraints: 2,
            source: "kimchi/src/circuits/polynomials/generic.rs:81",
            degrees: vec![3, 3],
            current: range(6),
            next: vec![],
            coefficients: range(10),
        },
        Kimchi {
            gadget: "kimchi-poseidon",
            constraints: 15,
            source: "kimchi/src/circuits/polynomials/poseidon.rs:349",
            degrees: vec![7; 15],
            current: range(15),
            next: range(3),
            coefficients: range(15),
        },
        Kimchi {
            gadget: "kimchi-complete-add",
            constraints: 7,
            source: "kimchi/src/circuits/polynomials/complete_add.rs:101",
            degrees: vec![2, 2, 3, 2, 2, 2, 2],
            current: range(11),
            next: vec![],
            coefficients: vec![],
        },
        Kimchi {
            gadget: "kimchi-varbasemul",
            constraints: 21,
            source: "kimchi/src/circuits/polynomials/varbasemul.rs:417",
            degrees: varbasemul_degrees,
            current: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14].to_vec(),
            next: range(12),
            coefficients: vec![],
        },
    ]
}

/// Inputs in cells, then the gate under test.
#[derive(Clone, Copy, Debug)]
enum Harness<'a> {
    Generic,
    Poseidon(&'a KimchiPoseidon<Fp>),
    CompleteAdd,
    VarBaseMul,
}

impl TypedGadget<Fp> for Harness<'_> {
    type Input = Unit;
    type Output = Unit;
    const NAME: &'static str = "harness";
    const ROWS: usize = 11;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, (): Of<Unit, E::Var>) -> Of<Unit, E::Var> {
        let mut cell = |column: usize| env.advice(Cell::new(column, 0), [], |[]| Fp::ZERO);
        match *self {
            Self::Generic => {
                let cells: [E::Var; 6] = core::array::from_fn(&mut cell);
                let gate = GenericRow::new(GenericSpec::Const(Fp::ONE), None);
                env.call(&gate, cells);
            }
            Self::Poseidon(poseidon) => {
                let cells: [E::Var; 3] = core::array::from_fn(&mut cell);
                env.call(&PoseidonPermutation { poseidon }, cells);
            }
            Self::CompleteAdd => {
                let cells: [E::Var; 4] = core::array::from_fn(&mut cell);
                env.call(&CompleteAdd, cells);
            }
            Self::VarBaseMul => {
                let cells: [E::Var; 4] = core::array::from_fn(&mut cell);
                let bits: [E::Var; 5] = core::array::from_fn(|_| env.constant(Fp::ZERO));
                let [xt, yt, x0, y0] = cells;
                env.call(&VarBaseMul::<5>, ([xt, yt], [x0, y0], bits));
            }
        }
    }
}

#[test]
fn every_gate_has_kimchis_figures() {
    let poseidon = pallas_base();
    let harnesses = [
        Harness::Generic,
        Harness::Poseidon(&poseidon),
        Harness::CompleteAdd,
        Harness::VarBaseMul,
    ];
    for (harness, expected) in harnesses.iter().zip(kimchi()) {
        let f = figures(harness, LAYOUT).unwrap();
        let kinds = f.gates_of(expected.gadget);
        let [gate] = kinds[..] else {
            panic!(
                "{}: {} gate kinds, Kimchi has one selector",
                expected.gadget,
                kinds.len()
            );
        };
        assert_eq!(gate.offset, 0, "{}", expected.gadget);
        assert_eq!(
            gate.constraints, expected.constraints,
            "{} ({})",
            expected.gadget, expected.source
        );
        assert_eq!(gate.degrees, expected.degrees, "{}", expected.gadget);
        let witness: Vec<(usize, usize)> = expected
            .current
            .iter()
            .map(|c| (*c, 0))
            .chain(expected.next.iter().map(|c| (*c, 1)))
            .collect();
        let mut found = gate.witness_cells.clone();
        found.sort_unstable();
        let mut witness = witness;
        witness.sort_unstable();
        assert_eq!(found, witness, "{}", expected.gadget);
        let coefficients: Vec<(usize, usize)> =
            expected.coefficients.iter().map(|c| (*c, 0)).collect();
        assert_eq!(gate.fixed_cells, coefficients, "{}", expected.gadget);
    }
}

/// The figures of the Poseidon permutation: 11 instances of the gate, on
/// rows 0-10, the output on row 11.
#[test]
fn the_poseidon_permutation() {
    let poseidon = pallas_base();
    let f = figures(&Harness::Poseidon(&poseidon), LAYOUT).unwrap();
    let [gate] = f.gates_of("kimchi-poseidon")[..] else {
        panic!("one kind")
    };
    assert_eq!(gate.instances, 11);
    assert_eq!(f.rows, 12);
}
