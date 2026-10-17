//! Kimchi's rotation gadget (`polynomials/rot.rs`): a Rot64 row and two
//! RangeCheck0 rows, the high limbs of the two range checks wired to a zero
//! cell of another row. The rows and the witness are derived from our
//! rotation gates.

use alloc::vec::Vec;

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Cell, Env, Error, Of, Rom, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::{
    range_check::{RangeCheck0Gate, range_check_table},
    rot::{Rot64Gate, WORD_BITS},
};

pub use arrabbiata_kimchi_gadgets::rot::RotMode;

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect},
    polynomials::{DeriveError, Rows, derive},
};

/// The left offset of a rotation by `rot` to `side`: a right rotation by
/// `rot` is a left one by `64 - rot`.
const fn left_offset(rot: u32, side: RotMode) -> u32 {
    match side {
        RotMode::Left => rot,
        RotMode::Right => WORD_BITS - rot,
    }
}

/// A rotation's three rows: the word in column 0 of row 0, Rot64 with
/// coefficient `2^rot`, then the RangeCheck0 rows of `shifted` and of
/// `excess` (copied from the Rot64 row), the four high limbs of the two
/// range checks on one cycle, which [`CircuitGate::extend_rot`] joins to a
/// zero cell.
#[derive(Clone, Debug)]
struct RotRows<F> {
    rot: u32,
    table: Rom<F>,
}

impl<F> TypedGadget<F> for RotRows<F>
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-rot-rows";
    const ROWS: usize = 3;

    fn synthesize<E: Env<F>>(&self, env: &mut E, word: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let word = env.advice(Cell::new(0, 0), [&word], |[w]| w);
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
        for high in [s1, e0, e1] {
            env.connect(&s0, &high);
        }
        rotated
    }
}

/// The column variables of the rotation's rows.
const VARIABLES: usize = 40;
/// The degree of the crumb checks.
const DEGREE: usize = 4;

/// The rows and witness of [`RotRows`] of `word` by `rot` to the left, from
/// row `offset`.
fn rot_rows<F: PrimeField>(word: F, rot: u32, offset: usize) -> Result<Rows<F>, DeriveError> {
    let gadget = RotRows {
        rot,
        table: range_check_table(),
    };
    derive::<F, _, VARIABLES, DEGREE>(&gadget, &word, RotRows::<F>::ROWS, offset)
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// A left rotation by `rot` from row `new_row`, derived from our
    /// rotation gates: Rot64 with coefficient `2^rot`, then the RangeCheck0
    /// rows of `shifted` and `excess` (Kimchi's `create_rot64`). Unlike
    /// Kimchi's, which leaves the wiring to its caller, the rows carry the
    /// gadget's own: `excess` copied to the second range check, and the four
    /// high limbs on one cycle.
    ///
    /// # Panics
    ///
    /// If `rot` is above 64.
    #[must_use]
    pub fn create_rot64(new_row: usize, rot: u32) -> Vec<Self> {
        match rot_rows(F::ZERO, rot, new_row) {
            Ok(rows) => rows.gates,
            Err(e) => panic!("no rotation by {rot}: {e:?}"),
        }
    }

    /// A rotation by `rot` to `side` from row `new_row`, a right one by
    /// `rot` being a left one by `64 - rot`: the row after it and its rows
    /// (Kimchi's `create_rot`).
    ///
    /// # Panics
    ///
    /// As [`CircuitGate::create_rot64`].
    #[must_use]
    pub fn create_rot(new_row: usize, rot: u32, side: RotMode) -> (usize, Vec<Self>) {
        let gates = Self::create_rot64(new_row, left_offset(rot, side));
        (new_row + gates.len(), gates)
    }

    /// `gates` extended by a rotation whose range checks' four high limbs
    /// are joined to `(zero_row, 0)`: the number of rows after it (Kimchi's
    /// `extend_rot`). Kimchi wires them with `connect_64bit`, which leaves
    /// one limb unwired (see [`Connect::connect_64bit`]); here the join is
    /// the union of the cycles, so all four are bounded by the zero cell.
    ///
    /// # Panics
    ///
    /// As [`CircuitGate::create_rot64`], or if `zero_row` is not a row of
    /// `gates`.
    pub fn extend_rot(gates: &mut Vec<Self>, rot: u32, side: RotMode, zero_row: usize) -> usize {
        let new_row = gates.len();
        let (_, mut rot) = Self::create_rot(new_row, rot, side);
        gates.append(&mut rot);
        gates.join((zero_row, 0), (new_row + 1, 1));
        gates.len()
    }
}

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
    let wide = Error::Rows {
        gadget: "kimchi-rot",
    };
    if rot > WORD_BITS {
        return Err(wide);
    }
    let rows = rot_rows(F::from_u64(word), left_offset(rot, side), 0).map_err(|_| wide)?;
    for (column, new) in witness.iter_mut().zip(rows.witness) {
        column.extend(new);
    }
    Ok(())
}
