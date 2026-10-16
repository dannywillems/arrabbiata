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

/// `2^k` in `F`.
#[must_use]
pub fn two_pow<F: Field>(k: u32) -> F {
    let mut x = F::ONE;
    let two_32 = F::from_u64(1 << 32);
    for _ in 0..k / 32 {
        x = x * two_32;
    }
    x * F::from_u64(1 << (k % 32))
}

/// `2^264`: the right input of the bound addition, `(0, 0, 0, 1)`.
#[must_use]
pub const fn two_to_264() -> Limbs {
    [0, 0, 0, 1]
}

/// A natural number, in 64-bit words, least significant first, for the
/// products of foreign field multiplication (up to 528 bits).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Nat(pub alloc::vec::Vec<u64>);

impl Nat {
    /// Without leading zero words.
    fn trimmed(mut self) -> Self {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
        self
    }

    /// `x`.
    #[must_use]
    pub fn from_u128(x: u128) -> Self {
        Self(alloc::vec![x as u64, (x >> 64) as u64]).trimmed()
    }

    /// The number of the limbs (base `2^88`), `limbs[0]` least significant.
    #[must_use]
    pub fn from_limbs(limbs: &[u128]) -> Self {
        let mut x = Self::default();
        for limb in limbs.iter().rev() {
            x = x.shl(LIMB_BITS).add(&Self::from_u128(*limb));
        }
        x
    }

    /// Bit `i`.
    fn bit(&self, i: usize) -> bool {
        self.0.get(i / 64).is_some_and(|w| (w >> (i % 64)) & 1 == 1)
    }

    /// The number of bits.
    fn bits(&self) -> usize {
        match self.0.last() {
            Some(w) => 64 * (self.0.len() - 1) + (64 - w.leading_zeros() as usize),
            None => 0,
        }
    }

    /// `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.0.len().max(other.0.len()) + 1;
        let mut out = alloc::vec![0u64; n];
        let mut carry = 0u128;
        for (i, slot) in out.iter_mut().enumerate() {
            let sum = u128::from(*self.0.get(i).unwrap_or(&0))
                + u128::from(*other.0.get(i).unwrap_or(&0))
                + carry;
            *slot = sum as u64;
            carry = sum >> 64;
        }
        Self(out).trimmed()
    }

    /// `self - other`, for `self >= other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut out = alloc::vec![0u64; self.0.len()];
        let mut borrow = 0i128;
        for (i, slot) in out.iter_mut().enumerate() {
            let d = i128::from(self.0[i]) - i128::from(*other.0.get(i).unwrap_or(&0)) - borrow;
            if d < 0 {
                *slot = (d + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                *slot = d as u64;
                borrow = 0;
            }
        }
        Self(out).trimmed()
    }

    /// `self other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = alloc::vec![0u64; self.0.len() + other.0.len() + 1];
        for (i, a) in self.0.iter().enumerate() {
            let mut carry = 0u128;
            for (j, b) in other.0.iter().enumerate() {
                let t = u128::from(*a) * u128::from(*b) + u128::from(out[i + j]) + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
            let mut k = i + other.0.len();
            while carry > 0 {
                let t = u128::from(out[k]) + carry;
                out[k] = t as u64;
                carry = t >> 64;
                k += 1;
            }
        }
        Self(out).trimmed()
    }

    /// `self 2^k`.
    #[must_use]
    pub fn shl(&self, k: u32) -> Self {
        let (words, bits) = ((k / 64) as usize, k % 64);
        let mut out = alloc::vec![0u64; self.0.len() + words + 1];
        for (i, w) in self.0.iter().enumerate() {
            out[i + words] |= w << bits;
            if bits > 0 {
                out[i + words + 1] |= w >> (64 - bits);
            }
        }
        Self(out).trimmed()
    }

    /// `floor(self / 2^k)`.
    #[must_use]
    pub fn shr(&self, k: u32) -> Self {
        let (words, bits) = ((k / 64) as usize, k % 64);
        let mut out = alloc::vec![0u64; self.0.len().saturating_sub(words)];
        for (i, slot) in out.iter_mut().enumerate() {
            let low = self.0[i + words] >> bits;
            let high = if bits > 0 {
                self.0.get(i + words + 1).map_or(0, |w| w << (64 - bits))
            } else {
                0
            };
            *slot = low | high;
        }
        Self(out).trimmed()
    }

    /// `self mod 2^k`.
    #[must_use]
    pub fn low(&self, k: u32) -> Self {
        self.sub(&self.shr(k).shl(k))
    }

    /// The order of two numbers.
    #[must_use]
    pub fn compare(&self, other: &Self) -> Ordering {
        let (a, b) = (self.clone().trimmed(), other.clone().trimmed());
        match a.0.len().cmp(&b.0.len()) {
            Ordering::Equal => {
                let mut i = a.0.len();
                while i > 0 {
                    i -= 1;
                    match a.0[i].cmp(&b.0[i]) {
                        Ordering::Equal => {}
                        other => return other,
                    }
                }
                Ordering::Equal
            }
            other => other,
        }
    }

    /// `(self / d, self mod d)`, by binary long division, for `d > 0`.
    #[must_use]
    pub fn divrem(&self, d: &Self) -> (Self, Self) {
        let mut quotient = Self::default();
        let mut remainder = Self::default();
        let mut i = self.bits();
        while i > 0 {
            i -= 1;
            remainder = remainder.shl(1);
            if self.bit(i) {
                remainder = remainder.add(&Self(alloc::vec![1]));
            }
            if remainder.compare(d) != Ordering::Less {
                remainder = remainder.sub(d);
                quotient = quotient.add(&Self(alloc::vec![1]).shl(i as u32));
            }
        }
        (quotient, remainder)
    }

    /// The three 88-bit limbs, for a number below `2^264`.
    #[must_use]
    pub fn limbs(&self) -> [u128; 3] {
        let word = |x: &Self| {
            u128::from(*x.0.first().unwrap_or(&0)) | (u128::from(*x.0.get(1).unwrap_or(&0)) << 64)
        };
        let lo = self.low(LIMB_BITS);
        let mi = self.shr(LIMB_BITS).low(LIMB_BITS);
        let hi = self.shr(2 * LIMB_BITS);
        [word(&lo), word(&mi), word(&hi)]
    }

    /// The number in `F` (reduced modulo the native field).
    #[must_use]
    pub fn to_field<F: Field>(&self) -> F {
        let base: F = field_of(1u128 << 64);
        let mut x = F::ZERO;
        for w in self.0.iter().rev() {
            x = x * base + F::from_u64(*w);
        }
        x
    }
}
