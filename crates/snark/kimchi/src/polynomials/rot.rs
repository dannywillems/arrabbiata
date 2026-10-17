//! Kimchi's rotation gadget (`polynomials/rot.rs`): a Rot64 row and two
//! RangeCheck0 rows, the high limbs of the two range checks wired to a zero
//! cell of another row.

use alloc::{vec, vec::Vec};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Cell, Env, Error, Of, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::{
    range_check::{RangeCheck0Gate, range_check_table},
    rot::{Rot64Gate, WORD_BITS},
    zero::Zero,
};

pub use arrabbiata_kimchi_gadgets::rot::RotMode;

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect, GateType, Wire},
    polynomials::columns,
};

/// `2^k` in `F`.
fn two_pow<F: Field>(k: u32) -> F {
    F::from_u64(2).pow(u64::from(k))
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// A left rotation by `rot` from row `new_row`: Rot64 with coefficient
    /// `2^rot`, then the RangeCheck0 rows of `shifted` and `excess` (Kimchi's
    /// `create_rot64`).
    #[must_use]
    pub fn create_rot64(new_row: usize, rot: u32) -> Vec<Self> {
        vec![
            Self::new(GateType::Rot64, Wire::for_row(new_row), vec![two_pow(rot)]),
            Self::new(
                GateType::RangeCheck0,
                Wire::for_row(new_row + 1),
                vec![F::ZERO],
            ),
            Self::new(
                GateType::RangeCheck0,
                Wire::for_row(new_row + 2),
                vec![F::ZERO],
            ),
        ]
    }

    /// A rotation by `rot` to `side` from row `new_row`, a right one by
    /// `rot` being a left one by `64 - rot`: the row after it and its rows
    /// (Kimchi's `create_rot`).
    #[must_use]
    pub fn create_rot(new_row: usize, rot: u32, side: RotMode) -> (usize, Vec<Self>) {
        let left = match side {
            RotMode::Left => rot,
            RotMode::Right => WORD_BITS - rot,
        };
        let gates = Self::create_rot64(new_row, left);
        (new_row + gates.len(), gates)
    }

    /// `gates` extended by a rotation whose range checks' high limbs are
    /// wired to `(zero_row, 0)` and whose `excess` is wired to the second
    /// range check: the number of rows after it (Kimchi's `extend_rot`).
    pub fn extend_rot(gates: &mut Vec<Self>, rot: u32, side: RotMode, zero_row: usize) -> usize {
        let (new_row, mut rot) = Self::create_rot(gates.len(), rot, side);
        gates.append(&mut rot);
        gates.connect_64bit(zero_row, new_row - 2);
        gates.connect_64bit(zero_row, new_row - 1);
        gates.connect_cell_pair((new_row - 3, 2), (new_row - 1, 0));
        gates.len()
    }
}

/// A zero row, then the rotation's three rows with their high limbs wired
/// to the zero cell: the gadget whose last three rows are Kimchi's
/// `extend_rot` witness.
#[derive(Clone, Debug)]
struct AfterZero<F> {
    rot: u32,
    table: arrabbiata_gadget::Rom<F>,
}

impl<F> TypedGadget<F> for AfterZero<F>
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-rot-after-zero";
    const ROWS: usize = 4;

    fn synthesize<E: Env<F>>(&self, env: &mut E, word: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        env.call(&Zero, ());
        let zero = env.advice(Cell::new(0, 0), [], |[]| F::ZERO);
        let word = env.advice(Cell::new(0, 1), [&word], |[w]| w);
        let gate = Rot64Gate {
            table: self.table.clone(),
            rot: self.rot,
        };
        let [rotated, excess, shifted] = env.call(&gate, word);
        let check = RangeCheck0Gate {
            table: self.table.clone(),
            compact: false,
        };
        let [s0, s1] = env.call(&check, shifted);
        let [e0, e1] = env.call(&check, excess);
        for high in [s0, s1, e0, e1] {
            env.connect(&zero, &high);
        }
        rotated
    }
}

/// The column variables of the rotation's rows.
const VARIABLES: usize = 40;
/// The degree of the crumb checks.
const DEGREE: usize = 4;

/// `witness` extended by the three rows of a rotation of `word` by `rot`
/// to `side` (Kimchi's `rot::extend_rot`), computed by the rotation gates
/// of `arrabbiata-kimchi-gadgets`.
///
/// # Errors
///
/// The gadget's [`Error`]; Kimchi panics when `rot > 64`.
pub fn extend_rot<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    word: u64,
    rot: u32,
    side: RotMode,
) -> Result<(), Error> {
    if rot > WORD_BITS {
        return Err(Error::Rows {
            gadget: "kimchi-rot",
        });
    }
    let left = match side {
        RotMode::Left => rot,
        RotMode::Right => WORD_BITS - rot,
    };
    let gadget = AfterZero {
        rot: left,
        table: range_check_table(),
    };
    let rows = columns::<F, _, VARIABLES, DEGREE>(&gadget, &F::from_u64(word), 4)?;
    for (column, new) in witness.iter_mut().zip(rows) {
        column.extend(new.into_iter().skip(1));
    }
    Ok(())
}
