//! Foreign field elements as Kimchi writes them: three 88-bit limbs, least
//! significant first (`ForeignElement<F, 88, 3>`), and the integer
//! arithmetic the witness of the foreign field gates needs.
//!
//! The values are below `2^264` (or `2^265` for a sum before reduction),
//! so an integer here is four limbs of base `2^88` in `u128`s, normalized
//! (each limb below `2^88`). This is witness plumbing: the constraints are
//! over the native field, and these helpers only compute the values the
//! prover fills in.

use core::cmp::Ordering;

use arrabbiata_field::{Field, PrimeField};

/// The bits of a limb.
pub const LIMB_BITS: u32 = 88;

/// `2^88`.
const BASE: u128 = 1 << LIMB_BITS;

/// An integer below `2^352`, in base `2^88`, least significant first.
pub type Limbs = [u128; 4];

/// The integer of three field limbs (each below `2^88`, or `2^89` for the
/// high limb of the bound's right input, which is `2^88`).
#[must_use]
pub fn from_field<F: PrimeField>(limbs: [F; 3]) -> Limbs {
    let to_u128 = |x: F| u128::from(x.limb(0)) | (u128::from(x.limb(1)) << 64);
    normalize([to_u128(limbs[0]), to_u128(limbs[1]), to_u128(limbs[2]), 0])
}

/// The three limbs, as field elements, the fourth folded into the high
/// one (`hi + 2^88 top`): a value of `2^264` or more keeps its high limb of
/// `2^88` or more, which a range check then rejects, instead of being
/// silently truncated.
#[must_use]
pub fn to_field<F: Field>(x: &Limbs) -> [F; 3] {
    let base: F = field_of(BASE);
    [
        field_of(x[0]),
        field_of(x[1]),
        field_of::<F>(x[2]) + field_of::<F>(x[3]) * base,
    ]
}

/// A `u128` in `F`.
#[must_use]
pub fn field_of<F: Field>(x: u128) -> F {
    let two_32 = F::from_u64(1 << 32);
    F::from_u64(x as u64) + F::from_u64((x >> 64) as u64) * two_32 * two_32
}

/// Carry each limb's excess over `2^88` into the next.
#[must_use]
pub fn normalize(mut x: Limbs) -> Limbs {
    let mut i = 0;
    while i < 3 {
        x[i + 1] += x[i] >> LIMB_BITS;
        x[i] &= BASE - 1;
        i += 1;
    }
    x
}

/// `a + b`.
#[must_use]
pub fn add(a: &Limbs, b: &Limbs) -> Limbs {
    normalize([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
}

/// `a - b`, for `a >= b`.
#[must_use]
pub fn sub(a: &Limbs, b: &Limbs) -> Limbs {
    let mut out = [0u128; 4];
    let mut borrow = 0u128;
    let mut i = 0;
    while i < 4 {
        let need = b[i] + borrow;
        if a[i] >= need {
            out[i] = a[i] - need;
            borrow = 0;
        } else {
            out[i] = a[i] + BASE - need;
            borrow = 1;
        }
        i += 1;
    }
    out
}

/// The order of two normalized integers.
#[must_use]
pub fn compare(a: &Limbs, b: &Limbs) -> Ordering {
    let mut i = 4;
    while i > 0 {
        i -= 1;
        match a[i].cmp(&b[i]) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

/// `2^264`: the right input of the bound addition, `(0, 0, 0, 1)`.
#[must_use]
pub const fn two_to_264() -> Limbs {
    [0, 0, 0, 1]
}
