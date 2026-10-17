//! Each of Kimchi's gates in a harness: its inputs in cells of row 0, then
//! one instance of the gadget, so that the builder meets the gate's kind
//! once ([`arrabbiata_gadget::gates`] gives its constraints, `figures` its
//! figures). The tests check the figures and the constraints against
//! Kimchi's; the Kimchi prover evaluates the constraints ([`gate_kinds`]).

use alloc::vec::Vec;

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Cell, Env, Error, GateConstraints, Of, TypedGadget, Unit, gates};

use crate::{
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
use arrabbiata_poseidon::kimchi::KimchiPoseidon;

/// Inputs in cells, then the gate under test.
#[derive(Clone, Copy, Debug)]
pub enum Harness<'a, F> {
    /// A Generic row.
    Generic,
    /// A Poseidon permutation (11 rows), with Kimchi's parameters.
    Poseidon(&'a KimchiPoseidon<F>),
    /// A CompleteAdd row.
    CompleteAdd,
    /// An EndoMul row, with the curve's GLV `beta` (Kimchi's endomorphism
    /// coefficient).
    EndoMul(F),
    /// An EndoMulScalar row.
    EndoMulScalar,
    /// A ForeignFieldAdd row.
    ForeignFieldAdd,
    /// A ForeignFieldMul row.
    ForeignFieldMul,
    /// A multi-range check: RangeCheck0 rows then RangeCheck1.
    RangeCheck,
    /// A Rot64 row.
    Rot,
    /// A VarBaseMul pair of rows.
    VarBaseMul,
    /// A Xor16 row.
    Xor,
}

impl<F> TypedGadget<F> for Harness<'_, F>
where
    F: PrimeField,
{
    type Input = Unit;
    type Output = Unit;
    const NAME: &'static str = "harness";
    const ROWS: usize = 11;

    fn synthesize<E: Env<F>>(&self, env: &mut E, (): Of<Unit, E::Var>) -> Of<Unit, E::Var> {
        let mut cell = |column: usize| env.advice(Cell::new(column, 0), [], |[]| F::ZERO);
        match *self {
            Self::Generic => {
                let cells: [E::Var; 6] = core::array::from_fn(&mut cell);
                let gate = GenericRow::new(GenericSpec::Const(F::ONE), None);
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
            Self::EndoMul(endo) => {
                let t: [E::Var; 2] = core::array::from_fn(&mut cell);
                let p: [E::Var; 2] = core::array::from_fn(|i| cell(4 + i));
                let bits: [E::Var; 4] = core::array::from_fn(|_| env.constant(F::ZERO));
                env.call(&EndoMul::<F, 4> { endo }, (t, p, bits));
            }
            Self::EndoMulScalar => {
                let crumbs: [E::Var; 8] = core::array::from_fn(|_| env.constant(F::ZERO));
                env.call(&EndoMulScalar::<8>, crumbs);
            }
            Self::RangeCheck => {
                let values: [E::Var; 3] =
                    core::array::from_fn(|i| env.advice(Cell::new(0, i), [], |[]| F::ZERO));
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
                env.call(&Xor::<F, 16> { table: xor_table() }, words);
            }
            Self::VarBaseMul => {
                let cells: [E::Var; 4] = core::array::from_fn(&mut cell);
                let bits: [E::Var; 5] = core::array::from_fn(|_| env.constant(F::ZERO));
                let [xt, yt, x0, y0] = cells;
                env.call(&VarBaseMul::<5>, ([xt, yt], [x0, y0], bits));
            }
        }
    }
}

/// The harness of the gadget named `gadget`, with Kimchi's Poseidon
/// parameters and endomorphism coefficient `endo`.
#[must_use]
pub fn of<'a, F>(gadget: &str, poseidon: &'a KimchiPoseidon<F>, endo: F) -> Harness<'a, F> {
    match gadget {
        "kimchi-generic" => Harness::Generic,
        "kimchi-poseidon" => Harness::Poseidon(poseidon),
        "kimchi-complete-add" => Harness::CompleteAdd,
        "kimchi-endomul" => Harness::EndoMul(endo),
        "kimchi-endomul-scalar" => Harness::EndoMulScalar,
        "kimchi-range-check-0" | "kimchi-range-check-1" => Harness::RangeCheck,
        "kimchi-xor16" => Harness::Xor,
        "kimchi-rot64" => Harness::Rot,
        "kimchi-foreign-field-add" => Harness::ForeignFieldAdd,
        "kimchi-foreign-field-mul" => Harness::ForeignFieldMul,
        _ => Harness::VarBaseMul,
    }
}

/// Our gadget name of each of Kimchi's gates with constraints, by Kimchi's
/// gate name.
pub const NAMES: [(&str, &str); 12] = [
    ("Generic", "kimchi-generic"),
    ("Poseidon", "kimchi-poseidon"),
    ("CompleteAdd", "kimchi-complete-add"),
    ("VarBaseMul", "kimchi-varbasemul"),
    ("EndoMul", "kimchi-endomul"),
    ("EndoMulScalar", "kimchi-endomul-scalar"),
    ("RangeCheck0", "kimchi-range-check-0"),
    ("RangeCheck1", "kimchi-range-check-1"),
    ("ForeignFieldAdd", "kimchi-foreign-field-add"),
    ("ForeignFieldMul", "kimchi-foreign-field-mul"),
    ("Xor16", "kimchi-xor16"),
    ("Rot64", "kimchi-rot64"),
];

/// The constraints of each of Kimchi's gates, by Kimchi's gate name, in
/// Kimchi's order: the gate kind of the gadget at its first row.
///
/// # Errors
///
/// An [`Error`] of the builder, which a harness does not raise.
pub fn gate_kinds<F>(
    poseidon: &KimchiPoseidon<F>,
    endo: F,
) -> Result<Vec<(&'static str, GateConstraints<F>)>, Error>
where
    F: PrimeField,
{
    let mut kinds = Vec::with_capacity(NAMES.len());
    for (kimchi, ours) in NAMES {
        let harness = of(ours, poseidon, endo);
        let found = gates::<F, _>(&harness, LAYOUT)?
            .into_iter()
            .find(|k| k.gadget == ours && k.offset == 0);
        match found {
            Some(kind) => kinds.push((kimchi, kind)),
            None => unreachable!("every harness meets its gate at its first row"),
        }
    }
    Ok(kinds)
}
