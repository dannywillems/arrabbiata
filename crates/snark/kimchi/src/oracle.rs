//! Kimchi's Fiat-Shamir: two sponges and a challenge map
//! (`mina_poseidon::sponge` and `kimchi::plonk_sponge`).
//!
//! A Kimchi proof over a curve `C` has commitments in `C` (coordinates in
//! the base field) and evaluations in the scalar field, so it runs two
//! sponges:
//!
//! - [`FqSponge`], over the base field: it absorbs points (their affine
//!   coordinates, the identity as `(0, 0)`), base-field elements, and
//!   scalars embedded into the base field; its challenges are 128 bits, the
//!   two low limbs of a squeezed element, read as a scalar; its digest is a
//!   squeezed element read as a scalar (zero when it does not fit);
//! - [`FrSponge`], over the scalar field: it absorbs evaluations; its
//!   challenges are 128-bit [`ScalarChallenge`]s.
//!
//! Each squeezed element yields two 64-bit limbs, buffered: a challenge
//! takes two, so every challenge is one fresh squeeze, and any absorption
//! empties the buffer.
//!
//! [`ScalarChallenge::to_field`] maps 128 bits `r` to `a lambda + b`, with
//! `a, b` accumulated two bits at a time (Kimchi's `to_field_with_length`):
//! this is the scalar whose multiplication EndoMul computes from the same
//! bits, which is why the in-circuit verifier wants it.
//!
//! Both sponges are generic over the underlying sponge (any
//! [`Transcript`]); Kimchi's is `arrabbiata_poseidon::kimchi::KimchiSponge`
//! with its `fq_kimchi` and `fp_kimchi` permutations. The tests check every
//! operation against Kimchi's output.

use alloc::{vec, vec::Vec};
use core::cmp::Ordering;

use arrabbiata_curve::{Curve, Projective};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_transcript::Transcript;

/// The limbs of a challenge (Kimchi's `CHALLENGE_LENGTH_IN_LIMBS`).
pub const CHALLENGE_LIMBS: usize = 2;

/// The limbs kept from each squeezed element (Kimchi's
/// `HIGH_ENTROPY_LIMBS`).
const KEPT_LIMBS: usize = 2;

/// The element of `F` with the given limbs, which must fit.
fn pack<F: PrimeField>(limbs: &[u64]) -> F {
    match F::from_limbs(limbs) {
        Some(x) => x,
        None => unreachable!("128 bits fit in the field"),
    }
}

/// The canonical limbs of `x`.
fn limbs<F: PrimeField>(x: F) -> Vec<u64> {
    (0..F::LIMBS).map(|i| x.limb(i)).collect()
}

/// The order of the moduli of `A` and `B`, compared through `p - 1`.
fn compare_moduli<A: PrimeField, B: PrimeField>() -> Ordering {
    let (a, b) = (limbs(-A::ONE), limbs(-B::ONE));
    let n = a.len().max(b.len());
    let at = |v: &[u64], i: usize| v.get(i).copied().unwrap_or(0);
    let mut i = n;
    while i > 0 {
        i -= 1;
        match at(&a, i).cmp(&at(&b, i)) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

/// A 128-bit challenge, before the endomorphism map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarChallenge<F>(pub F);

impl<F> ScalarChallenge<F>
where
    F: PrimeField,
{
    /// The challenge of the given two limbs, least significant first.
    #[must_use]
    pub fn from_limbs(limbs: [u64; CHALLENGE_LIMBS]) -> Self {
        Self(pack(&limbs))
    }

    /// The challenge as `a lambda + b`, from its low `bits` bits: from `a = b
    /// = 2`, for each pair of bits `(r_(2i+1), r_(2i))` from the top, double
    /// both and add `+1` (if `r_(2i)` is set) or `-1` to `a` (if `r_(2i+1)`
    /// is set) or to `b` (Kimchi's `to_field_with_length`).
    #[must_use]
    pub fn to_field_with_length(&self, bits: usize, lambda: F) -> F {
        let two = F::ONE + F::ONE;
        let (mut a, mut b) = (two, two);
        let bit = |i: usize| (self.0.limb(i / 64) >> (i % 64)) & 1 == 1;
        let mut i = bits / 2;
        while i > 0 {
            i -= 1;
            a = a + a;
            b = b + b;
            let s = if bit(2 * i) { F::ONE } else { -F::ONE };
            if bit(2 * i + 1) {
                a = a + s;
            } else {
                b = b + s;
            }
        }
        a * lambda + b
    }

    /// [`ScalarChallenge::to_field_with_length`] on the 128 bits.
    #[must_use]
    pub fn to_field(&self, lambda: F) -> F {
        self.to_field_with_length(64 * CHALLENGE_LIMBS, lambda)
    }
}

/// Kimchi's `DefaultFqSponge` for the curve `C`, over the sponge `S` on its
/// base field.
#[derive(Clone, Debug)]
pub struct FqSponge<C, S>
where
    C: Curve,
{
    sponge: S,
    buffer: Vec<u64>,
    curve: core::marker::PhantomData<fn() -> C>,
}

impl<C, S> FqSponge<C, S>
where
    C: Curve,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    /// A sponge over `sponge`, nothing buffered.
    #[must_use]
    pub const fn new(sponge: S) -> Self {
        Self {
            sponge,
            buffer: Vec::new(),
            curve: core::marker::PhantomData,
        }
    }

    /// Absorb points: `x` then `y`, the identity as `(0, 0)`.
    pub fn absorb_g(&mut self, points: &[Projective<C>]) {
        self.buffer.clear();
        for p in points {
            let a = p.to_affine();
            let (x, y) = match (a.x(), a.y()) {
                (Some(x), Some(y)) => (x, y),
                _ => (C::Base::ZERO, C::Base::ZERO),
            };
            self.sponge.absorb(x);
            self.sponge.absorb(y);
        }
    }

    /// Absorb base-field elements.
    pub fn absorb_fq(&mut self, values: &[C::Base]) {
        self.buffer.clear();
        for v in values {
            self.sponge.absorb(*v);
        }
    }

    /// Absorb scalars: as they are when the scalar modulus is the smaller,
    /// otherwise as their high bits then their low bit.
    pub fn absorb_fr(&mut self, values: &[C::Scalar]) {
        self.buffer.clear();
        let fits = compare_moduli::<C::Scalar, C::Base>() == Ordering::Less;
        for v in values {
            let l = limbs(*v);
            if fits {
                self.sponge.absorb(pack(&l));
            } else {
                let low = if l[0] & 1 == 1 {
                    C::Base::ONE
                } else {
                    C::Base::ZERO
                };
                let mut high = vec![0u64; l.len()];
                for i in 0..l.len() {
                    high[i] = (l[i] >> 1) | l.get(i + 1).map_or(0, |w| w << 63);
                }
                self.sponge.absorb(pack(&high));
                self.sponge.absorb(low);
            }
        }
    }

    /// `n` limbs: from the buffer, refilled with the two low limbs of a
    /// fresh squeeze when it runs short.
    fn squeeze_limbs(&mut self, n: usize) -> Vec<u64> {
        while self.buffer.len() < n {
            let x = self.sponge.squeeze();
            self.buffer.extend((0..KEPT_LIMBS).map(|i| x.limb(i)));
        }
        self.buffer.drain(..n).collect()
    }

    /// A 128-bit challenge, as a scalar.
    pub fn challenge(&mut self) -> C::Scalar {
        pack(&self.squeeze_limbs(CHALLENGE_LIMBS))
    }

    /// A full base-field element.
    pub fn challenge_fq(&mut self) -> C::Base {
        self.buffer.clear();
        self.sponge.squeeze()
    }

    /// The digest as a scalar: a squeezed element read as a scalar, zero
    /// when it does not fit.
    #[must_use]
    pub fn digest(mut self) -> C::Scalar {
        let x = self.challenge_fq();
        C::Scalar::from_limbs(&limbs(x)).unwrap_or(C::Scalar::ZERO)
    }

    /// The digest in the base field.
    #[must_use]
    pub fn digest_fq(mut self) -> C::Base {
        self.challenge_fq()
    }
}

/// Kimchi's `DefaultFrSponge` over the sponge `S` on `F`.
#[derive(Clone, Debug)]
pub struct FrSponge<F, S> {
    sponge: S,
    buffer: Vec<u64>,
    field: core::marker::PhantomData<fn() -> F>,
}

impl<F, S> FrSponge<F, S>
where
    F: PrimeField,
    S: Transcript<F>,
{
    /// A sponge over `sponge`, nothing buffered.
    #[must_use]
    pub const fn new(sponge: S) -> Self {
        Self {
            sponge,
            buffer: Vec::new(),
            field: core::marker::PhantomData,
        }
    }

    /// Absorb one element.
    pub fn absorb(&mut self, value: F) {
        self.buffer.clear();
        self.sponge.absorb(value);
    }

    /// Absorb elements, in order.
    pub fn absorb_multiple(&mut self, values: &[F]) {
        self.buffer.clear();
        for v in values {
            self.sponge.absorb(*v);
        }
    }

    /// A 128-bit challenge.
    pub fn challenge(&mut self) -> ScalarChallenge<F> {
        while self.buffer.len() < CHALLENGE_LIMBS {
            let x = self.sponge.squeeze();
            self.buffer.extend((0..KEPT_LIMBS).map(|i| x.limb(i)));
        }
        let taken: Vec<u64> = self.buffer.drain(..CHALLENGE_LIMBS).collect();
        ScalarChallenge(pack(&taken))
    }

    /// The digest: a squeezed element.
    #[must_use]
    pub fn digest(mut self) -> F {
        self.sponge.squeeze()
    }
}
