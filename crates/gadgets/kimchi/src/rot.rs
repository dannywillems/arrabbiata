//! Kimchi's Rot64 gate: a 64-bit word rotated by a known offset `rot`.
//!
//! The rotation is a multiplication modulo `2^64` split into its two
//! parts: `word 2^rot = excess 2^64 + shifted`, then `rotated = shifted +
//! excess`. Soundness needs `shifted < 2^64` and `excess < 2^rot`. The
//! first, and `excess < 2^64`, are two RangeCheck0 rows whose two highest
//! limbs are wired to a zero cell; the second is `bound = excess - 2^rot +
//! 2^64 < 2^64`, decomposed on the Rot64 row itself (Kimchi's
//! `rot.rs`).
//!
//! The rows, as Kimchi lays them out:
//!
//! | row | gate        | 0              | 1         | 2        | 3-6            | 7-14            |
//! | --- | ----------- | -------------- | --------- | -------- | -------------- | --------------- |
//! | 0   | Rot64       | `word`         | `rotated` | `excess` | `bound` limbs  | `bound` crumbs  |
//! | 1   | RangeCheck0 | `shifted`      | 0         | 0        | limbs          | crumbs          |
//! | 2   | RangeCheck0 | `excess` (copy)| 0         | 0        | limbs          | crumbs          |
//! | 3   | Generic     | 0              |           |          |                |                 |
//!
//! Columns 3-6 of the Rot64 row are looked up in the 12-bit table. The
//! coefficient 0 of the Rot64 row is `2^rot`; a right rotation by `r` is a
//! left rotation by `64 - r`. The witness is Kimchi's `rot::extend_rot`.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, Scalar, TypedGadget};

use crate::{
    generic::{GenericRow, GenericSpec},
    range_check::{RangeCheck0Gate, bits, crumb, recombine},
};

/// The width of the words.
pub const WORD_BITS: u32 = 64;

/// The bit ranges of the `bound` cells of the Rot64 row, columns 3-14.
const BOUND: [(usize, usize); 12] = [
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

/// The 12-bit limbs Kimchi looks up on the Rot64 row.
const LOOKED_UP: [usize; 4] = [3, 4, 5, 6];

/// The side of a rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotMode {
    /// Towards the most significant bits: the new low bits are the old high
    /// ones.
    Left,
    /// Towards the least significant bits.
    Right,
}

/// `2^k` in `F`, for `k <= 64`.
fn two_pow<F: Field>(k: u32) -> F {
    let half = F::from_u64(1 << (k / 2));
    let rest = F::from_u64(1 << (k - k / 2));
    half * rest
}

/// The low 64 bits of `x`, as an integer.
fn word<F: PrimeField>(x: F) -> u128 {
    u128::from(x.limb(0))
}

/// `x` below `2^128` in `F`.
fn from_u128<F: Field>(x: u128) -> F {
    F::from_u64(x as u64) + F::from_u64((x >> 64) as u64) * two_pow::<F>(64)
}

/// The Rot64 row, a left rotation by `rot` (`rot <= 64`): input `word`,
/// wired by copy to column 0; output `[rotated, excess, shifted]`, the
/// first two in columns 1 and 2, `shifted` in column 0 of the next row,
/// where the RangeCheck0 checking it reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rot64Gate<F> {
    /// The 12-bit table.
    pub table: Rom<F>,
    /// The offset of the left rotation.
    pub rot: u32,
}

impl<F> TypedGadget<F> for Rot64Gate<F>
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Array<3>;
    const NAME: &'static str = "kimchi-rot64";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, word_in: Of<Scalar, E::Var>) -> Of<Array<3>, E::Var>
    where
        E: Env<F>,
    {
        let rot = self.rot;
        let w = env.copy(Cell::new(0, 0), &word_in);
        let rotated = env.advice(Cell::new(1, 0), [&w], |[x]| {
            let shifted = (word(x) << rot) % (1u128 << 64);
            let excess = word(x) >> (WORD_BITS - rot);
            from_u128(shifted + excess)
        });
        let excess = env.advice(Cell::new(2, 0), [&w], |[x]| {
            from_u128(word(x) >> (WORD_BITS - rot))
        });
        let bound_value = |x: F| -> F {
            let excess = word(x) >> (WORD_BITS - rot);
            from_u128(excess + (1u128 << 64) - (1u128 << rot))
        };
        let cells: alloc::vec::Vec<E::Var> = BOUND
            .iter()
            .enumerate()
            .map(|(i, (start, end))| {
                env.advice(Cell::new(3 + i, 0), [&w], |[x]| {
                    bits(bound_value(x), *start, *end)
                })
            })
            .collect();
        let shifted = env.advice(Cell::new(0, 1), [&w], |[x]| {
            from_u128((word(x) << rot) % (1u128 << 64))
        });
        // C1-C8: the crumbs, columns 7-14.
        for c in &cells[4..] {
            let check = crumb(env, c);
            env.assert_zero(check);
        }
        let two_to_64 = env.constant(two_pow::<F>(64));
        let two_to_rot = env.fixed(Cell::new(0, 0), two_pow::<F>(rot));
        // C9: word 2^rot = excess 2^64 + shifted.
        env.assert_zero(
            w * two_to_rot.clone() - (excess.clone() * two_to_64.clone() + shifted.clone()),
        );
        // C10: rotated = shifted + excess.
        env.assert_zero(rotated.clone() - (shifted.clone() + excess.clone()));
        // C11: bound = excess - 2^rot + 2^64, bound from the crumbs (columns
        // 14 down to 7) then the limbs (6 down to 3).
        let ordered: alloc::vec::Vec<(E::Var, u64)> = (7..=14)
            .rev()
            .map(|c| (cells[c - 3].clone(), 4))
            .chain((3..=6).rev().map(|c| (cells[c - 3].clone(), 4096)))
            .collect();
        let bound = recombine(env, &ordered);
        env.assert_zero(bound - (excess.clone() - two_to_rot + two_to_64));
        for c in LOOKED_UP {
            env.read(&self.table, &[cells[c - 3].clone()]);
        }
        [rotated, excess, shifted]
    }
}

/// A rotation of a 64-bit word by `rot` to `side`, on four rows: Rot64,
/// RangeCheck0 of `shifted`, RangeCheck0 of `excess`, and a Generic row
/// whose `w0 = 0` the four high limbs of the two range checks are wired
/// to (Kimchi's `extend_rot` with `connect_64bit`). Input: the word, a cell
/// of a copied column; output: the rotated word. The word must be 64 bits,
/// which the caller ensures (Kimchi's assumption too).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rot64<F> {
    /// The 12-bit table.
    pub table: Rom<F>,
    /// The offset.
    pub rot: u32,
    /// The side.
    pub side: RotMode,
}

impl<F> TypedGadget<F> for Rot64<F>
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-rot";
    const ROWS: usize = 4;

    fn synthesize<E>(&self, env: &mut E, word: Of<Scalar, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let rot = match self.side {
            RotMode::Left => self.rot,
            RotMode::Right => WORD_BITS - self.rot,
        };
        let gate = Rot64Gate {
            table: self.table.clone(),
            rot,
        };
        let [rotated, excess, shifted] = env.call(&gate, word);
        let check = RangeCheck0Gate {
            table: self.table.clone(),
        };
        let [s0, s1] = env.call(&check, shifted);
        let [e0, e1] = env.call(&check, excess);
        let zero: [E::Var; 6] =
            core::array::from_fn(|c| env.advice(Cell::new(c, 3), [], |[]| F::ZERO));
        let row = GenericRow::new(GenericSpec::Const(F::ZERO), None);
        let first = zero[0].clone();
        env.call(&row, zero);
        for high in [s0, s1, e0, e1] {
            env.connect(&first, &high);
        }
        rotated
    }
}
