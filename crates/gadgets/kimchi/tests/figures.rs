//! Each gate's figures against Kimchi's.
//!
//! For every gate, the gadget's gate kind (one selector) has Kimchi's
//! figures:
//!
//! - the number of constraints and the degree of each, in Kimchi's order,
//!   as Kimchi computes them: [`KIMCHI`] is the output of running each
//!   gate's `Argument::constraints` and `Expr::degree(1, 0)` (a cell has
//!   degree 1, a constant 0) on o1-labs/proof-systems at `9227444890`
//!   (master, 2026-09-24), with its locked dependencies; the gate's
//!   `CONSTRAINTS` constant agrees with the number returned, and is quoted
//!   with its file and line;
//! - the cells the constraints read: the `witness_curr` and `witness_next`
//!   columns, and the coefficient columns, that `constraint_checks` reads.

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::Field;
use arrabbiata_gadget::{Cell, Env, Of, TypedGadget, Unit, figures};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    complete_add::CompleteAdd,
    endomul::EndoMul,
    endomul_scalar::EndoMulScalar,
    foreign_field_add::{FFOps, ForeignFieldAddGate},
    foreign_field_mul::ForeignFieldMulGate,
    generic::{GenericRow, GenericSpec},
    poseidon::PoseidonPermutation,
    range_check::{MultiRangeCheck, range_check_table},
    rot::{Rot64, RotMode},
    varbasemul::VarBaseMul,
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::{Fp, Pallas};
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
    /// The cells Kimchi looks up (its lookup patterns), as (column,
    /// rotation).
    lookups: Vec<(usize, usize)>,
}

fn range(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Kimchi's figures for every gate with constraints of its own, as printed
/// by running Kimchi: the gate, the number of constraints, their degrees.
/// The gates not yet written here are the targets of the ones to come.
const KIMCHI: [(&str, usize, &[usize]); 12] = [
    ("Generic", 2, &[3, 3]),
    ("Poseidon", 15, &[7; 15]),
    ("CompleteAdd", 7, &[2, 2, 3, 2, 2, 2, 2]),
    (
        "VarBaseMul",
        21,
        &[
            1, 2, 2, 6, 4, 2, 2, 6, 4, 2, 2, 6, 4, 2, 2, 6, 4, 2, 2, 6, 4,
        ],
    ),
    ("EndoMul", 12, &[2, 2, 2, 2, 3, 4, 4, 3, 4, 4, 1, 3]),
    ("EndoMulScalar", 11, &[1, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),
    ("RangeCheck0", 10, &[4, 4, 4, 4, 4, 4, 4, 4, 1, 2]),
    (
        "RangeCheck1",
        21,
        &[
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1,
        ],
    ),
    ("ForeignFieldAdd", 4, &[2, 3, 2, 2]),
    ("ForeignFieldMul", 11, &[4, 4, 2, 2, 2, 4, 4, 4, 2, 2, 1]),
    ("Xor16", 3, &[1, 1, 1]),
    ("Rot64", 11, &[4, 4, 4, 4, 4, 4, 4, 4, 2, 1, 1]),
];

/// Kimchi's degrees for `gate`.
fn degrees(gate: &str) -> Vec<usize> {
    KIMCHI
        .iter()
        .find(|(name, _, _)| *name == gate)
        .map(|(_, _, d)| d.to_vec())
        .unwrap_or_default()
}

fn kimchi() -> Vec<Kimchi> {
    vec![
        Kimchi {
            gadget: "kimchi-generic",
            constraints: 2,
            source: "kimchi/src/circuits/polynomials/generic.rs:81",
            degrees: degrees("Generic"),
            current: range(6),
            next: vec![],
            coefficients: range(10),
            lookups: vec![],
        },
        Kimchi {
            gadget: "kimchi-poseidon",
            constraints: 15,
            source: "kimchi/src/circuits/polynomials/poseidon.rs:349",
            degrees: degrees("Poseidon"),
            current: range(15),
            next: range(3),
            coefficients: range(15),
            lookups: vec![],
        },
        Kimchi {
            gadget: "kimchi-complete-add",
            constraints: 7,
            source: "kimchi/src/circuits/polynomials/complete_add.rs:101",
            degrees: degrees("CompleteAdd"),
            current: range(11),
            next: vec![],
            coefficients: vec![],
            lookups: vec![],
        },
        Kimchi {
            gadget: "kimchi-endomul",
            constraints: 12,
            source: "kimchi/src/circuits/polynomials/endosclmul.rs:473",
            degrees: degrees("EndoMul"),
            current: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].to_vec(),
            next: vec![4, 5, 6],
            coefficients: vec![],
            lookups: vec![],
        },
        Kimchi {
            gadget: "kimchi-endomul-scalar",
            constraints: 11,
            source: "kimchi/src/circuits/polynomials/endomul_scalar.rs:172",
            degrees: degrees("EndoMulScalar"),
            current: range(14),
            next: vec![],
            coefficients: vec![],
            lookups: vec![],
        },
        Kimchi {
            gadget: "kimchi-range-check-0",
            constraints: 10,
            source: "kimchi/src/circuits/polynomials/range_check/circuitgates.rs:177",
            degrees: degrees("RangeCheck0"),
            current: range(15),
            next: vec![0, 1],
            coefficients: vec![0],
            // The RangeCheck pattern on its row: columns 3-6.
            lookups: (3..=6).map(|c| (c, 0)).collect(),
        },
        Kimchi {
            gadget: "kimchi-range-check-1",
            constraints: 21,
            source: "kimchi/src/circuits/polynomials/range_check/circuitgates.rs:281",
            degrees: degrees("RangeCheck1"),
            current: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].to_vec(),
            next: [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14].to_vec(),
            coefficients: vec![],
            // The RangeCheck pattern on its row and the next: columns 3-6.
            lookups: (3..=6).flat_map(|c| [(c, 0), (c, 1)]).collect(),
        },
        Kimchi {
            gadget: "kimchi-xor16",
            constraints: 3,
            source: "kimchi/src/circuits/polynomials/xor.rs:156",
            degrees: degrees("Xor16"),
            current: range(15),
            next: range(3),
            coefficients: vec![],
            // The Xor pattern: (3 + k, 7 + k, 11 + k) for k < 4.
            lookups: (3..15).map(|c| (c, 0)).collect(),
        },
        Kimchi {
            gadget: "kimchi-foreign-field-add",
            constraints: 4,
            source: "kimchi/src/circuits/polynomials/foreign_field_add/circuitgates.rs:147",
            degrees: degrees("ForeignFieldAdd"),
            current: range(8),
            next: range(3),
            coefficients: range(4),
            lookups: vec![],
        },
        Kimchi {
            gadget: "kimchi-foreign-field-mul",
            constraints: 11,
            source: "kimchi/src/circuits/polynomials/foreign_field_mul/circuitgates.rs:199",
            degrees: degrees("ForeignFieldMul"),
            current: range(15),
            next: range(12),
            coefficients: range(4),
            // The ForeignFieldMul pattern: columns 7-10 of both rows.
            lookups: (7..=10).flat_map(|c| [(c, 0), (c, 1)]).collect(),
        },
        Kimchi {
            gadget: "kimchi-rot64",
            constraints: 11,
            source: "kimchi/src/circuits/polynomials/rot.rs:226",
            degrees: degrees("Rot64"),
            current: range(15),
            next: vec![0],
            coefficients: vec![0],
            // The RangeCheck pattern on its row: columns 3-6.
            lookups: (3..=6).map(|c| (c, 0)).collect(),
        },
        Kimchi {
            gadget: "kimchi-varbasemul",
            constraints: 21,
            source: "kimchi/src/circuits/polynomials/varbasemul.rs:417",
            degrees: degrees("VarBaseMul"),
            current: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14].to_vec(),
            next: range(12),
            coefficients: vec![],
            lookups: vec![],
        },
    ]
}

/// Inputs in cells, then the gate under test.
#[derive(Clone, Copy, Debug)]
enum Harness<'a> {
    Generic,
    Poseidon(&'a KimchiPoseidon<Fp>),
    CompleteAdd,
    EndoMul,
    EndoMulScalar,
    ForeignFieldAdd,
    ForeignFieldMul,
    RangeCheck,
    Rot,
    VarBaseMul,
    Xor,
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
            Self::EndoMul => {
                let t: [E::Var; 2] = core::array::from_fn(&mut cell);
                let p: [E::Var; 2] = core::array::from_fn(|i| cell(4 + i));
                let bits: [E::Var; 4] = core::array::from_fn(|_| env.constant(Fp::ZERO));
                env.call(
                    &EndoMul::<Fp, 4> {
                        endo: Pallas::beta(),
                    },
                    (t, p, bits),
                );
            }
            Self::EndoMulScalar => {
                let crumbs: [E::Var; 8] = core::array::from_fn(|_| env.constant(Fp::ZERO));
                env.call(&EndoMulScalar::<8>, crumbs);
            }
            Self::RangeCheck => {
                let values: [E::Var; 3] =
                    core::array::from_fn(|i| env.advice(Cell::new(0, i), [], |[]| Fp::ZERO));
                let gadget = MultiRangeCheck {
                    table: range_check_table(),
                };
                env.call(&gadget, values);
            }
            Self::ForeignFieldAdd => {
                let left: [E::Var; 3] = core::array::from_fn(&mut cell);
                let right: [E::Var; 3] = core::array::from_fn(|i| cell(3 + i));
                let modulus = [1, 2, 3, 0];
                let gate = ForeignFieldAddGate {
                    modulus,
                    op: FFOps::Add,
                };
                env.call(&gate, (left, right));
            }
            Self::ForeignFieldMul => {
                let left: [E::Var; 3] = core::array::from_fn(&mut cell);
                let right: [E::Var; 3] = core::array::from_fn(|i| cell(3 + i));
                let gate = ForeignFieldMulGate {
                    modulus: [5, 0, 0, 0],
                    table: range_check_table(),
                };
                env.call(&gate, (left, right));
            }
            Self::Rot => {
                let word = cell(0);
                let gadget = Rot64 {
                    table: range_check_table(),
                    rot: 3,
                    side: RotMode::Left,
                };
                env.call(&gadget, word);
            }
            Self::Xor => {
                let words: [E::Var; 2] = core::array::from_fn(&mut cell);
                env.call(&Xor::<Fp, 16> { table: xor_table() }, words);
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
    for expected in kimchi() {
        let harness = match expected.gadget {
            "kimchi-generic" => Harness::Generic,
            "kimchi-poseidon" => Harness::Poseidon(&poseidon),
            "kimchi-complete-add" => Harness::CompleteAdd,
            "kimchi-endomul" => Harness::EndoMul,
            "kimchi-endomul-scalar" => Harness::EndoMulScalar,
            "kimchi-range-check-0" | "kimchi-range-check-1" => Harness::RangeCheck,
            "kimchi-xor16" => Harness::Xor,
            "kimchi-rot64" => Harness::Rot,
            "kimchi-foreign-field-add" => Harness::ForeignFieldAdd,
            "kimchi-foreign-field-mul" => Harness::ForeignFieldMul,
            _ => Harness::VarBaseMul,
        };
        let f = figures(&harness, LAYOUT).unwrap();
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
        let mut looked_up = gate.read_cells.clone();
        looked_up.sort_unstable();
        let mut lookups = expected.lookups.clone();
        lookups.sort_unstable();
        assert_eq!(looked_up, lookups, "{}", expected.gadget);
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

/// Kimchi's table is consistent: each gate's count is the number of its
/// degrees, and the counts are Kimchi's `CONSTRAINTS`.
#[test]
fn kimchis_table() {
    for (gate, constraints, degrees) in KIMCHI {
        assert_eq!(degrees.len(), constraints, "{gate}");
    }
    let total: usize = KIMCHI.iter().map(|(_, c, _)| c).sum();
    assert_eq!(total, 2 + 15 + 7 + 21 + 12 + 11 + 10 + 21 + 4 + 11 + 3 + 11);
}
