//! The gate under test, each in a harness: its inputs in cells of row 0,
//! then one instance of the gadget. Shared by `figures.rs` and
//! `differential.rs`, so both test the same gate kinds.

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::Field;
use arrabbiata_gadget::{Cell, Env, Of, TypedGadget, Unit};
use arrabbiata_kimchi_gadgets::{
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
use arrabbiata_poseidon::kimchi::KimchiPoseidon;

/// Inputs in cells, then the gate under test.
#[derive(Clone, Copy, Debug)]
pub enum Harness<'a> {
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

/// The harness of the gadget named `gadget`.
pub fn of<'a>(gadget: &str, poseidon: &'a KimchiPoseidon<Fp>) -> Harness<'a> {
    match gadget {
        "kimchi-generic" => Harness::Generic,
        "kimchi-poseidon" => Harness::Poseidon(poseidon),
        "kimchi-complete-add" => Harness::CompleteAdd,
        "kimchi-endomul" => Harness::EndoMul,
        "kimchi-endomul-scalar" => Harness::EndoMulScalar,
        "kimchi-range-check-0" | "kimchi-range-check-1" => Harness::RangeCheck,
        "kimchi-xor16" => Harness::Xor,
        "kimchi-rot64" => Harness::Rot,
        "kimchi-foreign-field-add" => Harness::ForeignFieldAdd,
        "kimchi-foreign-field-mul" => Harness::ForeignFieldMul,
        _ => Harness::VarBaseMul,
    }
}
