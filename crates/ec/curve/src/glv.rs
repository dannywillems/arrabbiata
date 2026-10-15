//! GLV scalar multiplication (Gallant, Lambert, Vanstone, *Faster point
//! multiplication on elliptic curves with efficient endomorphisms*, CRYPTO
//! 2001).
//!
//! On a curve `y^2 = x^3 + b` over a field with a primitive cube root of
//! unity `beta`, the map `phi(x, y) = (beta x, y)` is an endomorphism, and on
//! a group of prime order `r` it acts as multiplication by a cube root of
//! unity `lambda` of the scalars: `phi(P) = [lambda] P`. A scalar then splits
//! as `k = k1 + k2 lambda (mod r)` with `k1`, `k2` of about half its bits, and
//! `[k] P = [k1] P + [k2] phi(P)` costs one doubling per bit of the halves,
//! with both halves added in the same loop.
//!
//! The split rounds `k` against a short basis `(a1, b1), (a2, b2)` of the
//! lattice `{(a, b) : a + b lambda = 0 (mod r)}` (Hankerson, Menezes,
//! Vanstone, *Guide to Elliptic Curve Cryptography*, Algorithm 3.74):
//! `c1 = round(b2 k / r)`, `c2 = round(-b1 k / r)`, and
//! `k1 = k - c1 a1 - c2 a2`, `k2 = -c1 b1 - c2 b2`, so that
//! `k1 + k2 lambda = k - c1 (a1 + b1 lambda) - c2 (a2 + b2 lambda) = k`.
//! The divisions are precomputed: `g1 = round(2^256 b2 / r)`,
//! `g2 = round(-2^256 b1 / r)`, and `c_i = floor(k g_i / 2^256)` (with the
//! sign of `g_i`). Everything else is exact arithmetic modulo `2^256`, which
//! is exact here because the results have at most 128 bits.

use crate::{Curve, Projective};
use arrabbiata_field::PrimeField;

/// An integer of at most 256 bits, with a sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Signed {
    /// Whether it is negative.
    pub negative: bool,
    /// Its absolute value, as little-endian limbs.
    pub magnitude: [u64; 4],
}

impl Signed {
    /// The value modulo `2^256`, in two's complement.
    const fn wrapping(&self) -> [u64; 4] {
        if self.negative {
            negate(&self.magnitude)
        } else {
            self.magnitude
        }
    }
}

/// The decomposition constants of a curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Basis {
    /// The first basis vector `(a1, b1)`.
    pub a1: Signed,
    /// See [`Basis::a1`].
    pub b1: Signed,
    /// The second basis vector `(a2, b2)`.
    pub a2: Signed,
    /// See [`Basis::a2`].
    pub b2: Signed,
    /// `round(2^256 b2 / r)`.
    pub g1: Signed,
    /// `round(-2^256 b1 / r)`.
    pub g2: Signed,
}

/// A curve with the endomorphism `phi(x, y) = (beta x, y)`.
pub trait Glv: Curve {
    /// `beta`, a primitive cube root of unity of the base field.
    fn beta() -> Self::Base;

    /// `lambda`, the cube root of unity of the scalars with
    /// `phi(P) = [lambda] P`.
    fn lambda() -> Self::Scalar;

    /// The decomposition constants for `lambda`.
    const BASIS: Basis;
}

const fn negate(a: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut carry = 1u64;
    let mut i = 0;
    while i < 4 {
        let (sum, overflow) = (!a[i]).overflowing_add(carry);
        out[i] = sum;
        carry = overflow as u64;
        i += 1;
    }
    out
}

const fn add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut carry = 0u128;
    let mut i = 0;
    while i < 4 {
        let sum = a[i] as u128 + b[i] as u128 + carry;
        out[i] = sum as u64;
        carry = sum >> 64;
        i += 1;
    }
    out
}

/// The full product.
const fn mul_wide(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    let mut out = [0u64; 8];
    let mut i = 0;
    while i < 4 {
        let mut carry = 0u128;
        let mut j = 0;
        while j < 4 {
            let t = a[i] as u128 * b[j] as u128 + out[i + j] as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
            j += 1;
        }
        out[i + 4] = carry as u64;
        i += 1;
    }
    out
}

/// The product modulo `2^256`.
const fn mul_low(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let wide = mul_wide(a, b);
    [wide[0], wide[1], wide[2], wide[3]]
}

/// `sign(g) floor(k |g| / 2^256)`, modulo `2^256`.
const fn rounded(k: &[u64; 4], g: &Signed) -> [u64; 4] {
    let wide = mul_wide(k, &g.magnitude);
    let quotient = [wide[4], wide[5], wide[6], wide[7]];
    if g.negative {
        negate(&quotient)
    } else {
        quotient
    }
}

/// A two's complement value of at most 128 bits, as a sign and a magnitude.
///
/// # Panics
///
/// If the value does not fit, which the basis rules out.
const fn small(v: &[u64; 4]) -> (bool, u128) {
    let negative = v[3] >> 63 == 1;
    let magnitude = if negative { negate(v) } else { *v };
    assert!(
        magnitude[2] == 0 && magnitude[3] == 0,
        "a GLV half has more than 128 bits"
    );
    (
        negative,
        magnitude[0] as u128 | ((magnitude[1] as u128) << 64),
    )
}

/// The split `k = k1 + k2 lambda (mod r)`, each half as a sign and a
/// magnitude of at most 128 bits.
///
/// # Panics
///
/// If a half has more than 128 bits, which the basis of each curve here
/// rules out (its halves have at most 128 bits).
#[must_use]
pub fn decompose<C>(k: &C::Scalar) -> ((bool, u128), (bool, u128))
where
    C: Glv,
{
    let basis = C::BASIS;
    let limbs = [k.limb(0), k.limb(1), k.limb(2), k.limb(3)];
    let c1 = rounded(&limbs, &basis.g1);
    let c2 = rounded(&limbs, &basis.g2);
    let c1_a1 = mul_low(&c1, &basis.a1.wrapping());
    let c2_a2 = mul_low(&c2, &basis.a2.wrapping());
    let c1_b1 = mul_low(&c1, &basis.b1.wrapping());
    let c2_b2 = mul_low(&c2, &basis.b2.wrapping());
    let k1 = add(&add(&limbs, &negate(&c1_a1)), &negate(&c2_a2));
    let k2 = add(&negate(&c1_b1), &negate(&c2_b2));
    (small(&k1), small(&k2))
}

impl<C> Projective<C>
where
    C: Glv,
{
    /// `phi(P) = (beta x, y)`.
    #[must_use]
    pub fn endomorphism(&self) -> Self {
        Self {
            x: self.x * C::beta(),
            y: self.y,
            z: self.z,
        }
    }

    /// `[k] P`, as `[k1] P + [k2] phi(P)` for the split of `k`: about 128
    /// doublings instead of 256.
    #[must_use]
    pub fn mul_glv(&self, k: &C::Scalar) -> Self {
        let ((negative1, k1), (negative2, k2)) = decompose::<C>(k);
        let p1 = if negative1 { -*self } else { *self };
        let phi = self.endomorphism();
        let p2 = if negative2 { -phi } else { phi };
        let both = p1 + p2;
        let mut acc = Self::identity();
        let mut bit = 128;
        while bit > 0 {
            bit -= 1;
            acc = acc.double();
            match ((k1 >> bit) & 1 == 1, (k2 >> bit) & 1 == 1) {
                (true, true) => acc = acc + both,
                (true, false) => acc = acc + p1,
                (false, true) => acc = acc + p2,
                (false, false) => {}
            }
        }
        acc
    }
}

/// The field element of canonical little-endian limbs `limbs`: how the
/// curves write their GLV constants.
///
/// # Panics
///
/// If `limbs` is not below the modulus.
#[must_use]
pub fn canonical<F>(limbs: [u64; 4]) -> F
where
    F: PrimeField,
{
    match F::from_limbs(&limbs) {
        Some(x) => x,
        None => panic!("a GLV constant is below the modulus"),
    }
}
