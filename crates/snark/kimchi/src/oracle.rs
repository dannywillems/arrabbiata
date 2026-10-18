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
//! Kimchi buffers two 64-bit limbs of each squeezed element and a
//! challenge takes two, so the buffer is empty between draws: every
//! challenge is the lowest 128 bits of one fresh squeeze, which is how both
//! sponges compute it, without a buffer. Both run in a model of the field
//! signature too (the field itself by default, a circuit for Pickles), so
//! each message and each draw is defined once.
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
use core::{cell::RefCell, cmp::Ordering};

use arrabbiata_curve::{Curve, Projective};
use arrabbiata_field::{Field, FieldModel, Native, PrimeField};
use arrabbiata_poseidon::kimchi::{KimchiDuplex, WIDTH};
use arrabbiata_transcript::{PermutationIn, Transcript};

/// The limbs of a challenge (Kimchi's `CHALLENGE_LENGTH_IN_LIMBS`).
pub const CHALLENGE_LIMBS: usize = 2;

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
pub(crate) fn compare_moduli<A: PrimeField, B: PrimeField>() -> Ordering {
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

/// The operations of Kimchi's 128-bit challenges beyond the field
/// signature, in a model of it ([`FieldModel`]): what the scalar sponge
/// needs to draw a challenge, and what a verifier needs to use one.
pub trait ChallengeModel<F>: FieldModel<F>
where
    F: Field,
{
    /// The lowest 128 bits of `x`: the challenge a squeeze gives (the two
    /// low limbs of [`FrSponge::challenge`]). A circuit writes `x = lo +
    /// 2^128 hi`, checks `hi` to 128 bits, and leaves `lo` to the use that
    /// checks it ([`ChallengeModel::endo`]), as Pickles'
    /// `lowest_128_bits`.
    fn lowest_128_bits(&mut self, x: &Self::Value) -> Self::Value;

    /// The scalar `a lambda + b` of the challenge `chal`
    /// ([`ScalarChallenge::to_field`]); a circuit also checks that `chal`
    /// has 128 bits, as Pickles' `to_field_checked`.
    fn endo(&mut self, chal: &Self::Value, lambda: F) -> Self::Value;
}

impl<F> ChallengeModel<F> for Native
where
    F: PrimeField,
{
    fn lowest_128_bits(&mut self, x: &F) -> F {
        pack(&[x.limb(0), x.limb(1)])
    }

    fn endo(&mut self, chal: &F, lambda: F) -> F {
        ScalarChallenge(*chal).to_field(lambda)
    }
}

impl<F, M> ChallengeModel<F> for &mut M
where
    F: Field,
    M: ChallengeModel<F>,
{
    fn lowest_128_bits(&mut self, x: &Self::Value) -> Self::Value {
        (**self).lowest_128_bits(x)
    }

    fn endo(&mut self, chal: &Self::Value, lambda: F) -> Self::Value {
        (**self).endo(chal, lambda)
    }
}

/// Kimchi's `DefaultFqSponge` for the curve `C`: the sponge `S` on its base
/// field, with its arithmetic where `H` says, the field itself by default.
/// As for [`FrSponge`], Kimchi's limb buffer never holds anything between
/// draws (two limbs per squeeze, two per challenge, and a full squeeze
/// empties it), so a challenge is the lowest 128 bits of one squeeze. How a
/// draw is read is the model's ([`BaseReading`]): natively as a scalar
/// (Kimchi's), in a circuit as the base-field value it is (Pickles').
#[derive(Clone, Debug)]
pub struct FqSponge<C, S, H = Native>
where
    C: Curve,
{
    model: H,
    sponge: S,
    curve: core::marker::PhantomData<fn() -> C>,
}

/// How a model reads the base sponge's draws, a challenge (the lowest 128
/// bits of a squeeze) and the digest (a squeeze): natively as scalars of
/// `C`, Kimchi's reading; in a shared model as the base-field values they
/// are, the reading of a circuit over the base field, which hands them to
/// the other field later.
pub trait BaseReading<C>: ModelHandle<C::Base>
where
    C: Curve,
{
    /// A draw, as read.
    type Read;

    /// The 128-bit challenge of the squeeze `x`.
    fn read_challenge(&self, x: &Value<C::Base, Self>) -> Self::Read;

    /// The digest `x`.
    fn read_digest(&self, x: &Value<C::Base, Self>) -> Self::Read;
}

impl<C> BaseReading<C> for Native
where
    C: Curve,
    C::Base: PrimeField,
{
    type Read = C::Scalar;

    fn read_challenge(&self, x: &C::Base) -> C::Scalar {
        pack(&[x.limb(0), x.limb(1)])
    }

    /// A squeezed element read as a scalar, zero when it does not fit.
    fn read_digest(&self, x: &C::Base) -> C::Scalar {
        C::Scalar::from_limbs(&limbs(*x)).unwrap_or(C::Scalar::ZERO)
    }
}

impl<C, M> BaseReading<C> for &RefCell<M>
where
    C: Curve,
    M: ChallengeModel<C::Base>,
{
    type Read = M::Value;

    fn read_challenge(&self, x: &M::Value) -> M::Value {
        self.borrow_mut().lowest_128_bits(x)
    }

    fn read_digest(&self, x: &M::Value) -> M::Value {
        x.clone()
    }
}

impl<C, S, H> FqSponge<C, S, H>
where
    C: Curve,
    H: SpongeIn<C::Base, S>,
{
    /// A sponge over `sponge`, in the model `model` handles.
    pub const fn with_model(model: H, sponge: S) -> Self {
        Self {
            model,
            sponge,
            curve: core::marker::PhantomData,
        }
    }

    /// Absorb `value`.
    pub fn absorb_value(&mut self, value: &Value<C::Base, H>) {
        self.model.absorb_into(&mut self.sponge, value);
    }

    /// Squeeze an element.
    pub fn squeeze_value(&mut self) -> Value<C::Base, H> {
        self.model.squeeze_from(&mut self.sponge)
    }

    /// A 128-bit challenge, as the model reads it.
    pub fn challenge_read(&mut self) -> H::Read
    where
        H: BaseReading<C>,
    {
        let x = self.squeeze_value();
        self.model.read_challenge(&x)
    }

    /// The digest, as the model reads it.
    #[must_use]
    pub fn digest_read(mut self) -> H::Read
    where
        H: BaseReading<C>,
    {
        let x = self.squeeze_value();
        self.model.read_digest(&x)
    }

    /// The model's handle.
    pub fn model(&self) -> &H {
        &self.model
    }
}

impl<C, S> FqSponge<C, S>
where
    C: Curve,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    /// A sponge over `sponge`, in the field itself.
    #[must_use]
    pub const fn new(sponge: S) -> Self {
        Self::with_model(Native, sponge)
    }

    /// Absorb points: `x` then `y`, the identity as `(0, 0)`.
    pub fn absorb_g(&mut self, points: &[Projective<C>]) {
        for p in points {
            let (x, y) = coordinates(p);
            self.absorb_value(&x);
            self.absorb_value(&y);
        }
    }

    /// Absorb base-field elements.
    pub fn absorb_fq(&mut self, values: &[C::Base]) {
        for v in values {
            self.absorb_value(v);
        }
    }

    /// Absorb scalars, each as [`scalar_elements`].
    pub fn absorb_fr(&mut self, values: &[C::Scalar]) {
        for v in values {
            for e in scalar_elements::<C>(*v) {
                self.absorb_value(&e);
            }
        }
    }

    /// A 128-bit challenge, as a scalar.
    pub fn challenge(&mut self) -> C::Scalar {
        self.challenge_read()
    }

    /// A full base-field element.
    pub fn challenge_fq(&mut self) -> C::Base {
        self.squeeze_value()
    }

    /// The digest as a scalar: a squeezed element read as a scalar, zero
    /// when it does not fit.
    #[must_use]
    pub fn digest(self) -> C::Scalar {
        self.digest_read()
    }

    /// The digest in the base field.
    #[must_use]
    pub fn digest_fq(mut self) -> C::Base {
        self.squeeze_value()
    }
}

impl<'m, C, M, P> FqSponge<C, Duplexed<'m, M::Value, P>, &'m RefCell<M>>
where
    C: Curve,
    M: FieldModel<C::Base>,
    P: PermutationIn<M, C::Base, WIDTH>,
{
    /// Kimchi's base sponge in `model`, over `permutation`, fresh.
    pub fn in_model(model: &'m RefCell<M>, permutation: &'m P) -> Self {
        let duplex = Duplexed::new(&mut *model.borrow_mut(), permutation);
        Self::with_model(model, duplex)
    }
}

/// The base-field elements a scalar of `C` is absorbed as: itself when the
/// scalar modulus is the smaller, otherwise its high bits then its low bit
/// (Kimchi's `absorb_fr`). A circuit over the base field holds a scalar in
/// this form already, so it absorbs what it holds.
#[must_use]
pub fn scalar_elements<C>(x: C::Scalar) -> Vec<C::Base>
where
    C: Curve,
    C::Base: PrimeField,
{
    let l = limbs(x);
    if compare_moduli::<C::Scalar, C::Base>() == Ordering::Less {
        return vec![pack(&l)];
    }
    let low = if l[0] & 1 == 1 {
        C::Base::ONE
    } else {
        C::Base::ZERO
    };
    let mut high = vec![0u64; l.len()];
    for i in 0..l.len() {
        high[i] = (l[i] >> 1) | l.get(i + 1).map_or(0, |w| w << 63);
    }
    vec![pack(&high), low]
}

/// The affine coordinates of `p` as the base sponge absorbs them, the
/// identity as `(0, 0)`.
#[must_use]
pub fn coordinates<C>(p: &Projective<C>) -> (C::Base, C::Base)
where
    C: Curve,
{
    let a = p.to_affine();
    match (a.x(), a.y()) {
        (Some(x), Some(y)) => (x, y),
        _ => (C::Base::ZERO, C::Base::ZERO),
    }
}

/// Where a sponge does its arithmetic: the field itself ([`Native`]), or a
/// model shared with the rest of a circuit (`&RefCell<M>`, so that a copy of
/// the sponge, which a peek draws from, writes into the same circuit).
pub trait ModelHandle<F>: Clone
where
    F: Field,
{
    /// The model.
    type Model: FieldModel<F>;

    /// Run `f` in the model.
    fn with<R>(&self, f: impl FnOnce(&mut Self::Model) -> R) -> R;
}

impl<F> ModelHandle<F> for Native
where
    F: Field,
{
    type Model = Self;

    fn with<R>(&self, f: impl FnOnce(&mut Self) -> R) -> R {
        f(&mut Self)
    }
}

impl<F, M> ModelHandle<F> for &RefCell<M>
where
    F: Field,
    M: FieldModel<F>,
{
    type Model = M;

    fn with<R>(&self, f: impl FnOnce(&mut M) -> R) -> R {
        f(&mut self.borrow_mut())
    }
}

/// An element of the model `H` handles.
pub type Value<F, H> = <<H as ModelHandle<F>>::Model as FieldModel<F>>::Value;

/// How the model a handle points to runs the sponge `S`: in the field
/// itself, any [`Transcript`]; in a shared model, Kimchi's duplex over a
/// permutation in that model ([`Duplexed`]). Implemented on the handle, so
/// the two cases are two types and never overlap.
pub trait SpongeIn<F, S>: ModelHandle<F>
where
    F: Field,
{
    /// Absorb `value` into `sponge`.
    fn absorb_into(&self, sponge: &mut S, value: &Value<F, Self>);

    /// Squeeze an element of `sponge`.
    fn squeeze_from(&self, sponge: &mut S) -> Value<F, Self>;
}

impl<F, S> SpongeIn<F, S> for Native
where
    F: Field,
    S: Transcript<F>,
{
    fn absorb_into(&self, sponge: &mut S, value: &F) {
        sponge.absorb(*value);
    }

    fn squeeze_from(&self, sponge: &mut S) -> F {
        sponge.squeeze()
    }
}

/// Kimchi's duplex ([`KimchiDuplex`]) over a permutation in a model.
#[derive(Debug)]
pub struct Duplexed<'p, V, P> {
    duplex: KimchiDuplex<V>,
    permutation: &'p P,
}

impl<V, P> Clone for Duplexed<'_, V, P>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            duplex: self.duplex.clone(),
            permutation: self.permutation,
        }
    }
}

impl<'p, V, P> Duplexed<'p, V, P>
where
    V: Clone,
{
    /// A fresh duplex over `permutation`, its state the model's zero.
    pub fn new<F, M>(model: &mut M, permutation: &'p P) -> Self
    where
        F: Field,
        M: FieldModel<F, Value = V>,
    {
        Self {
            duplex: KimchiDuplex::new(model.constant(F::ZERO)),
            permutation,
        }
    }
}

impl<F, M, P> SpongeIn<F, Duplexed<'_, M::Value, P>> for &RefCell<M>
where
    F: Field,
    M: FieldModel<F>,
    P: PermutationIn<M, F, WIDTH>,
{
    fn absorb_into(&self, sponge: &mut Duplexed<'_, M::Value, P>, value: &M::Value) {
        let mut model = self.borrow_mut();
        sponge.duplex.absorb(&mut *model, sponge.permutation, value);
    }

    fn squeeze_from(&self, sponge: &mut Duplexed<'_, M::Value, P>) -> M::Value {
        let mut model = self.borrow_mut();
        sponge.duplex.squeeze(&mut *model, sponge.permutation)
    }
}

/// Kimchi's `DefaultFrSponge`: the sponge `S` with its arithmetic where
/// `H` says, the field itself by default. A challenge is the lowest 128
/// bits of one squeeze ([`ChallengeModel::lowest_128_bits`]): Kimchi buffers
/// two limbs per squeeze and a challenge takes two, so nothing is left in
/// the buffer between challenges. One sponge, so the messages of
/// `transcript::Evaluation` are written once, natively and in a circuit.
#[derive(Clone, Debug)]
pub struct FrSponge<F, S, H = Native> {
    model: H,
    sponge: S,
    field: core::marker::PhantomData<fn() -> F>,
}

impl<F, S, H> FrSponge<F, S, H>
where
    F: Field,
    H: SpongeIn<F, S>,
{
    /// A sponge over `sponge`, in the model `model` handles.
    pub const fn with_model(model: H, sponge: S) -> Self {
        Self {
            model,
            sponge,
            field: core::marker::PhantomData,
        }
    }

    /// Absorb `value`.
    pub fn absorb_value(&mut self, value: &Value<F, H>) {
        self.model.absorb_into(&mut self.sponge, value);
    }

    /// Squeeze an element.
    pub fn squeeze_value(&mut self) -> Value<F, H> {
        self.model.squeeze_from(&mut self.sponge)
    }

    /// A 128-bit challenge: the lowest 128 bits of a squeeze.
    pub fn challenge_value(&mut self) -> ScalarChallenge<Value<F, H>>
    where
        H::Model: ChallengeModel<F>,
    {
        let x = self.squeeze_value();
        ScalarChallenge(self.model.with(|m| m.lowest_128_bits(&x)))
    }

    /// The model's handle.
    pub fn model(&self) -> &H {
        &self.model
    }
}

impl<F, S> FrSponge<F, S>
where
    F: PrimeField,
    S: Transcript<F>,
{
    /// A sponge over `sponge`, in the field itself.
    #[must_use]
    pub const fn new(sponge: S) -> Self {
        Self::with_model(Native, sponge)
    }

    /// Absorb one element.
    pub fn absorb(&mut self, value: F) {
        self.absorb_value(&value);
    }

    /// Absorb elements, in order.
    pub fn absorb_multiple(&mut self, values: &[F]) {
        for v in values {
            self.absorb_value(v);
        }
    }

    /// A 128-bit challenge.
    pub fn challenge(&mut self) -> ScalarChallenge<F> {
        self.challenge_value()
    }

    /// The digest: a squeezed element.
    #[must_use]
    pub fn digest(mut self) -> F {
        self.squeeze_value()
    }
}

impl<'m, F, M, P> FrSponge<F, Duplexed<'m, M::Value, P>, &'m RefCell<M>>
where
    F: Field,
    M: FieldModel<F>,
    P: PermutationIn<M, F, WIDTH>,
{
    /// Kimchi's sponge in `model`, over `permutation`, fresh.
    pub fn in_model(model: &'m RefCell<M>, permutation: &'m P) -> Self {
        let duplex = Duplexed::new(&mut *model.borrow_mut(), permutation);
        Self::with_model(model, duplex)
    }
}
