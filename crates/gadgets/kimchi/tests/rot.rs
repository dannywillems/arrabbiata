//! Kimchi's 64-bit rotation (Rot64, two RangeCheck0, a zero row) against
//! Kimchi.
//!
//! - for random words and offsets, the rows are Kimchi's
//!   `rot::extend_rot`, cell for cell (the Rot64 row, the RangeCheck0 rows
//!   of `shifted` and `excess`), and the output is `u64::rotate_left`, or
//!   `rotate_right` for a right rotation;
//! - the split of `word 2^rot` is unique: moving `2^64` from `shifted` to
//!   `excess` keeps the two equations and is caught by the range checks;
//! - a wrong `rotated` is caught.
//!
//! No proof here: the 12-bit table makes the circuit 8192 rows, as for
//! RangeCheck, whose proof is the slow test of `tests/range_check.rs`.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Cell, Env, Of, Scalar, TypedGadget, witness};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    range_check::range_check_table,
    rot::{Rot64, RotMode},
};
use arrabbiata_pasta::Fp;
use proptest::prelude::*;

/// The column variables: every cell of a row, those of the next, and the
/// coefficients.
const VARIABLES: usize = 40;
/// A crumb's check.
const DEGREE: usize = 4;

/// The word in column 0 of the first row, then the rotation.
#[derive(Clone, Debug)]
struct Rotate {
    rot: u32,
    side: RotMode,
}

impl TypedGadget<Fp> for Rotate {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "rotate";
    const ROWS: usize = 4;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, word: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let cell = env.advice(Cell::new(0, 0), [&word], |[v]| v);
        let gadget = Rot64 {
            table: range_check_table(),
            rot: self.rot,
            side: self.side,
        };
        env.call(&gadget, cell)
    }
}

/// The bits `start .. end` of `v`.
fn range(v: u128, start: usize, end: usize) -> u64 {
    ((v >> start) & ((1u128 << (end - start)) - 1)) as u64
}

fn fp(v: u128) -> Fp {
    let two_64 = Fp::from_u64(1 << 32) * Fp::from_u64(1 << 32);
    Fp::from_u64(v as u64) + Fp::from_u64((v >> 64) as u64) * two_64
}

/// A RangeCheck0 row of `x`: `x`, then its bit ranges.
fn range_check_0_row(x: u128) -> [Fp; 15] {
    let ranges = [
        (76, 88),
        (64, 76),
        (52, 64),
        (40, 52),
        (28, 40),
        (16, 28),
        (14, 16),
        (12, 14),
        (10, 12),
        (8, 10),
        (6, 8),
        (4, 6),
        (2, 4),
        (0, 2),
    ];
    let mut row = [Fp::ZERO; 15];
    row[0] = fp(x);
    for (i, (s, e)) in ranges.iter().enumerate() {
        row[1 + i] = Fp::from_u64(range(x, *s, *e));
    }
    row
}

/// Kimchi's `extend_rot(word, rot, side)`: the Rot64 row and the two
/// RangeCheck0 rows.
fn kimchi_rows(word: u64, rot: u32, side: RotMode) -> [[Fp; 15]; 3] {
    let rot = if side == RotMode::Right {
        64 - rot
    } else {
        rot
    };
    let shifted = (u128::from(word) << rot) % (1u128 << 64);
    let excess = u128::from(word) >> (64 - rot);
    let rotated = shifted + excess;
    let bound = excess + (1u128 << 64) - (1u128 << rot);
    let ranges = [
        (52, 64),
        (40, 52),
        (28, 40),
        (16, 28),
        (14, 16),
        (12, 14),
        (10, 12),
        (8, 10),
        (6, 8),
        (4, 6),
        (2, 4),
        (0, 2),
    ];
    let mut r0 = [Fp::ZERO; 15];
    r0[0] = Fp::from_u64(word);
    r0[1] = fp(rotated);
    r0[2] = fp(excess);
    for (i, (s, e)) in ranges.iter().enumerate() {
        r0[3 + i] = Fp::from_u64(range(bound, *s, *e));
    }
    [r0, range_check_0_row(shifted), range_check_0_row(excess)]
}

fn arb_side() -> impl Strategy<Value = RotMode> {
    prop_oneof![Just(RotMode::Left), Just(RotMode::Right)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn kimchis_layout(word in any::<u64>(), rot in 0u32..=64, side in arb_side()) {
        let gadget = Rotate { rot, side };
        let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &Fp::from_u64(word)).unwrap();
        let expected = match side {
            RotMode::Left => word.rotate_left(rot),
            RotMode::Right => word.rotate_right(rot),
        };
        prop_assert_eq!(out, Fp::from_u64(expected));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (r, row) in kimchi_rows(word, rot, side).iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][r]);
            prop_assert_eq!(&found, row, "row {}", r);
        }
    }

    /// `excess + 1, shifted - 2^64, rotated + 1 - 2^64` keeps both
    /// equations, and `bound + 1` keeps its recombination; what remains
    /// wrong is that `shifted` is negative and the copied `excess` no longer
    /// matches its limbs, which only the RangeCheck0 rows see.
    #[test]
    fn the_split_is_unique(word in any::<u64>(), rot in 1u32..64) {
        let gadget = Rotate { rot, side: RotMode::Left };
        let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &Fp::from_u64(word)).unwrap();
        let two_64 = fp(1u128 << 64);
        let mut columns = compiled.columns.clone();
        // The lowest crumb of `bound` (row 0, column 14) absorbs the + 1.
        prop_assume!(columns[14][0] != Fp::from_u64(3));
        columns[14][0] = columns[14][0] + Fp::ONE;
        // Row 0: excess (column 2); row 1: shifted (column 0); row 2: the
        // copy of excess.
        columns[2][0] = columns[2][0] + Fp::ONE;
        columns[1][0] = columns[1][0] + Fp::ONE - two_64;
        columns[0][1] = columns[0][1] - two_64;
        columns[0][2] = columns[2][0];
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }

    #[test]
    fn a_wrong_rotation_is_caught(word in any::<u64>(), rot in 0u32..=64) {
        let gadget = Rotate { rot, side: RotMode::Left };
        let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &Fp::from_u64(word)).unwrap();
        let mut columns = compiled.columns.clone();
        columns[1][0] = columns[1][0] + Fp::ONE;
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }
}
