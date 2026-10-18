//! Kimchi's Fiat-Shamir transcript as a type: every message and every
//! challenge of the protocol, in Kimchi's order, so that the prover and the
//! verifier, which both follow [`Piop`] and [`Evaluation`], cannot absorb or
//! squeeze anything else, or in another order, and still compile.
//!
//! Kimchi runs two sponges. The base-field sponge ([`FqSponge`]) absorbs the
//! statement and the commitments and squeezes the PIOP's challenges, as
//! 128-bit scalar challenges:
//!
//! 1. statement: the verifier index's digest, the previous accumulators'
//!    commitments, the commitment to the public input;
//! 2. the witness columns' commitments;
//! 3. with lookups ([`Branch`] on the index): the runtime table's
//!    commitment when there is one (another branch), the joint combiner when
//!    a lookup is joint (another branch, the combiner is zero otherwise), the
//!    sorted columns' commitments;
//! 4. `beta`, `gamma`; with lookups, the aggregation's commitment;
//! 5. the permutation's commitment `z`, then `alpha`;
//! 6. the quotient's commitment `t`, then `zeta`;
//! 7. its digest, from a copy ([`Peek`]): the sponge itself goes on into
//!    the opening.
//!
//! The scalar-field sponge ([`FrSponge`]) absorbs that digest and the
//! previous accumulators' digest (the digest of a sponge that absorbed their
//! challenges, [`PreviousChallenges`]), then `ft_eval1` and the evaluations,
//! and squeezes `v` (polyscale) and `u` (evalscale), 128-bit challenges too.
//!
//! The opening, on the base-field sponge after step 7, is the inner-product
//! argument's transcript, typed in its crate
//! ([`arrabbiata_ipa::generic::OpeningTranscript`]): for Kimchi's choices,
//! the shifted combined inner product, the base-field element hashed to the
//! value base, the rounds' `(L, R)` and challenges, then `delta` and `c`.
//!
//! The tapes start [`Tape::undescribed`]: Kimchi's transcript absorbs no
//! description of its pattern, and ours must be Kimchi's bit for bit.

use core::{cell::RefCell, marker::PhantomData};

use alloc::vec::Vec;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, FieldModel, Native, PrimeField};
use arrabbiata_poseidon::kimchi::WIDTH;
use arrabbiata_transcript::{
    PermutationIn, Transcript,
    typed::{Branch, Draw, End, Label, Peek, Send, Squeeze, Statement, Tape, Write},
};

use crate::{
    constraints::COLUMNS,
    oracle::{
        BaseReading, ChallengeModel, Duplexed, FqSponge, FrSponge, ScalarChallenge, SpongeIn,
        Value, coordinates,
    },
    prover::ProofEvaluations,
    srs::PolyComm,
};

/// A value of the protocol, named by `Name`: what the type of a step says
/// is written there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message<T, Name>(pub T, pub PhantomData<Name>);

impl<T, Name> Message<T, Name> {
    /// The message `value`.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self(value, PhantomData)
    }
}

impl<T, Name> Label for Message<T, Name>
where
    Name: Label,
{
    const LABEL: u64 = Name::LABEL;
}

/// A challenge of the protocol, named by `Name`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Challenge<Name>(PhantomData<Name>);

impl<Name> Label for Challenge<Name>
where
    Name: Label,
{
    const LABEL: u64 = Name::LABEL;
}

/// A step's name, with a label for its pattern's description.
macro_rules! name {
    ($(#[$doc:meta])* $name:ident = $label:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;

        impl Label for $name {
            const LABEL: u64 = $label;
        }
    };
}

name!(
    /// The verifier index's digest.
    VerifierIndexDigest = 0x6b01
);
name!(
    /// The previous accumulators' commitments.
    PreviousCommitments = 0x6b02
);
name!(
    /// The commitment to the public input.
    PublicInput = 0x6b03
);
name!(
    /// The witness columns' commitments.
    WitnessColumns = 0x6b04
);
name!(
    /// The runtime table's commitment.
    RuntimeTable = 0x6b05
);
name!(
    /// The joint combiner of the lookups.
    JointCombiner = 0x6b06
);
name!(
    /// The sorted columns' commitments.
    SortedColumns = 0x6b07
);
name!(
    /// The permutation's `beta`.
    Beta = 0x6b08
);
name!(
    /// The permutation's `gamma`.
    Gamma = 0x6b09
);
name!(
    /// The lookup aggregation's commitment.
    Aggregation = 0x6b0a
);
name!(
    /// The permutation's commitment `z`.
    Permutation = 0x6b0b
);
name!(
    /// The constraints' combiner `alpha`.
    Alpha = 0x6b0c
);
name!(
    /// The quotient's commitment `t`.
    Quotient = 0x6b0d
);
name!(
    /// The evaluation point `zeta`.
    Zeta = 0x6b0e
);
name!(
    /// The base-field sponge's digest, handed to the scalar-field sponge.
    Digest = 0x6b0f
);
name!(
    /// The previous accumulators' challenges.
    PreviousChallenges = 0x6b10
);
name!(
    /// `ft(zeta w)`.
    FtEval1 = 0x6b11
);
name!(
    /// The evaluations at `zeta` and `zeta w`, the public input's first.
    Evaluations = 0x6b12
);
name!(
    /// The polynomials' combiner `v`.
    Polyscale = 0x6b13
);
name!(
    /// The points' combiner `u`.
    Evalscale = 0x6b14
);

/// One base-field element, as the base-field sponge absorbs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Base<F>(pub F);

/// One scalar, as the scalar-field sponge absorbs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar<F>(pub F);

/// A point as a circuit holds it: its affine coordinates, values of a
/// model. The base sponge absorbs `x` then `y`, as it does a native point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point<V> {
    /// The first coordinate.
    pub x: V,
    /// The second coordinate.
    pub y: V,
}

impl<C> From<&Projective<C>> for Point<C::Base>
where
    C: arrabbiata_curve::Curve,
{
    /// The coordinates the base sponge absorbs: the identity as `(0, 0)`.
    fn from(p: &Projective<C>) -> Self {
        let (x, y) = coordinates(p);
        Self { x, y }
    }
}

/// What a message of the base sponge is: a sequence of base-field
/// elements, the ones it absorbs, in order. A base element is one; a point
/// is its coordinates, `x` then `y` (the identity `(0, 0)`); a commitment
/// is its chunks; a list or an array is its elements in turn. So one writer
/// serves every message of the base sponge, natively and in a model.
pub trait BaseElements<V> {
    /// Call `f` on each element, in order.
    fn each_element(&self, f: &mut impl FnMut(&V));
}

impl<V> BaseElements<V> for Base<V> {
    fn each_element(&self, f: &mut impl FnMut(&V)) {
        f(&self.0);
    }
}

impl<V> BaseElements<V> for Point<V> {
    fn each_element(&self, f: &mut impl FnMut(&V)) {
        f(&self.x);
        f(&self.y);
    }
}

impl<C> BaseElements<C::Base> for Projective<C>
where
    C: arrabbiata_curve::Curve,
{
    fn each_element(&self, f: &mut impl FnMut(&C::Base)) {
        let (x, y) = coordinates(self);
        f(&x);
        f(&y);
    }
}

impl<C> BaseElements<C::Base> for PolyComm<C>
where
    C: arrabbiata_curve::Curve,
{
    fn each_element(&self, f: &mut impl FnMut(&C::Base)) {
        self.chunks.each_element(f);
    }
}

impl<V, T> BaseElements<V> for Vec<T>
where
    T: BaseElements<V>,
{
    fn each_element(&self, f: &mut impl FnMut(&V)) {
        for t in self {
            t.each_element(f);
        }
    }
}

impl<V, T, const N: usize> BaseElements<V> for [T; N]
where
    T: BaseElements<V>,
{
    fn each_element(&self, f: &mut impl FnMut(&V)) {
        for t in self {
            t.each_element(f);
        }
    }
}

/// Every message of the base sponge: its elements, absorbed in order.
impl<C, S, H, T, Name> Write<FqSponge<C, S, H>> for Message<T, Name>
where
    C: arrabbiata_curve::Curve,
    H: SpongeIn<C::Base, S>,
    T: BaseElements<Value<C::Base, H>>,
{
    fn write(&self, sponge: &mut FqSponge<C, S, H>) {
        self.0.each_element(&mut |v| sponge.absorb_value(v));
    }
}

/// A 128-bit challenge of the base-field sponge, as its model reads it
/// ([`BaseReading`]): natively the scalar its limbs are (the caller maps it
/// through the endomorphism), in a circuit the base-field value.
impl<C, S, H, Name> Draw<FqSponge<C, S, H>> for Challenge<Name>
where
    C: arrabbiata_curve::Curve,
    H: SpongeIn<C::Base, S> + BaseReading<C>,
{
    type Value = H::Read;

    fn draw(sponge: &mut FqSponge<C, S, H>) -> H::Read {
        sponge.challenge_read()
    }
}

/// The base-field sponge's digest, read as a scalar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DigestOf<Name>(PhantomData<Name>);

impl<Name> Label for DigestOf<Name>
where
    Name: Label,
{
    const LABEL: u64 = Name::LABEL;
}

/// The base-field sponge's digest, from a copy, as its model reads it.
impl<C, S, H, Name> Draw<FqSponge<C, S, H>> for DigestOf<Name>
where
    C: arrabbiata_curve::Curve,
    H: SpongeIn<C::Base, S> + BaseReading<C>,
    S: Clone,
{
    type Value = H::Read;

    fn draw(sponge: &mut FqSponge<C, S, H>) -> H::Read {
        sponge.clone().digest_read()
    }
}

impl<F, S, H, Name> Write<FrSponge<F, S, H>> for Message<Scalar<Value<F, H>>, Name>
where
    F: Field,
    H: SpongeIn<F, S>,
{
    fn write(&self, sponge: &mut FrSponge<F, S, H>) {
        sponge.absorb_value(&self.0.0);
    }
}

impl<F, S, H, Name> Write<FrSponge<F, S, H>> for Message<Vec<Vec<Value<F, H>>>, Name>
where
    F: Field,
    H: SpongeIn<F, S>,
{
    fn write(&self, sponge: &mut FrSponge<F, S, H>) {
        for chals in &self.0 {
            for c in chals {
                sponge.absorb_value(c);
            }
        }
    }
}

impl<F, S, H, Name> Write<FrSponge<F, S, H>> for Message<ProofEvaluations<Value<F, H>>, Name>
where
    F: Field,
    H: SpongeIn<F, S>,
{
    fn write(&self, sponge: &mut FrSponge<F, S, H>) {
        let e = &self.0;
        for p in core::iter::once(&e.public).chain(e.absorbed()) {
            for v in p.zeta.iter().chain(&p.zeta_omega) {
                sponge.absorb_value(v);
            }
        }
    }
}

/// A 128-bit challenge of the scalar-field sponge: the lowest 128 bits of
/// a squeeze, in whichever model the sponge computes.
impl<F, S, H, Name> Draw<FrSponge<F, S, H>> for Challenge<Name>
where
    F: Field,
    H: SpongeIn<F, S>,
    H::Model: ChallengeModel<F>,
{
    type Value = ScalarChallenge<Value<F, H>>;

    fn draw(sponge: &mut FrSponge<F, S, H>) -> Self::Value {
        sponge.challenge_value()
    }
}

/// The scalar-field sponge's digest: a squeeze of a copy.
impl<F, S, H, Name> Draw<FrSponge<F, S, H>> for DigestOf<Name>
where
    F: Field,
    H: SpongeIn<F, S>,
    S: Clone,
{
    type Value = Value<F, H>;

    fn draw(sponge: &mut FrSponge<F, S, H>) -> Self::Value {
        sponge.clone().squeeze_value()
    }
}

/// A commitment message.
pub type Commitment<C, Name> = Message<PolyComm<C>, Name>;

// The base-field sponge's patterns, written once over the carrier `G` of a
// commitment (`PolyComm<C>` natively, its chunks `Vec<Point<V>>` in a
// model); the native names below are their readings at `PolyComm<C>`.

/// After the sorted columns: `beta`, `gamma`, the aggregation, then
/// [`RestOf`].
pub type WithLookupOf<G> =
    Squeeze<Challenge<Beta>, Squeeze<Challenge<Gamma>, Send<Message<G, Aggregation>, RestOf<G>>>>;

/// After the witness columns, without lookups: `beta`, `gamma`, then
/// [`RestOf`].
pub type WithoutLookupOf<G> = Squeeze<Challenge<Beta>, Squeeze<Challenge<Gamma>, RestOf<G>>>;

/// From `t`: `zeta`, and the digest from a copy.
pub type AfterAlphaOf<G> =
    Send<Message<G, Quotient>, Squeeze<Challenge<Zeta>, Peek<DigestOf<Digest>, End>>>;

/// From `z`: `alpha`, then [`AfterAlphaOf`].
pub type RestOf<G> = Send<Message<G, Permutation>, Squeeze<Challenge<Alpha>, AfterAlphaOf<G>>>;

/// The sorted columns, then [`WithLookupOf`].
pub type SortedOf<G> = Send<Message<Vec<G>, SortedColumns>, WithLookupOf<G>>;

/// The joint combiner when a lookup is joint, or none: then [`SortedOf`].
pub type JointOf<G> = Branch<Squeeze<Challenge<JointCombiner>, SortedOf<G>>, SortedOf<G>>;

/// The lookups: the runtime table's commitment or none, then [`JointOf`].
pub type LookupsOf<G> = Branch<Send<Message<G, RuntimeTable>, JointOf<G>>, JointOf<G>>;

/// The base-field sponge's part of Kimchi's PIOP, at the carriers `B` of a
/// base element, `P` of a point and `G` of a commitment.
pub type PiopOf<B, P, G> = Statement<
    Message<Base<B>, VerifierIndexDigest>,
    Statement<
        Message<Vec<P>, PreviousCommitments>,
        Statement<
            Message<G, PublicInput>,
            Send<Message<[G; COLUMNS], WitnessColumns>, Branch<WithoutLookupOf<G>, LookupsOf<G>>>,
        >,
    >,
>;

/// [`WithLookupOf`] natively.
pub type WithLookup<C> = WithLookupOf<PolyComm<C>>;

/// [`WithoutLookupOf`] natively.
pub type WithoutLookup<C> = WithoutLookupOf<PolyComm<C>>;

/// [`AfterAlphaOf`] natively.
pub type AfterAlpha<C> = AfterAlphaOf<PolyComm<C>>;

/// [`RestOf`] natively.
pub type Rest<C> = RestOf<PolyComm<C>>;

/// [`SortedOf`] natively.
pub type Sorted<C> = SortedOf<PolyComm<C>>;

/// [`JointOf`] natively.
pub type Joint<C> = JointOf<PolyComm<C>>;

/// [`LookupsOf`] natively.
pub type Lookups<C> = LookupsOf<PolyComm<C>>;

/// [`PiopOf`] natively: base elements, points and commitments of `C`.
pub type Piop<C> = PiopOf<<C as arrabbiata_curve::Curve>::Base, Projective<C>, PolyComm<C>>;

/// [`PiopOf`] in a model: its values `V`, points their coordinates, a
/// commitment its chunks.
pub type PiopIn<V> = PiopOf<V, Point<V>, Vec<Point<V>>>;

/// The base-field sponge's tape after `gamma`: at `z` without lookups, at
/// the aggregation's commitment with them.
#[derive(Debug)]
pub enum AfterGamma<C, S>
where
    C: arrabbiata_curve::Curve,
{
    /// Without lookups.
    WithoutLookup(FqTape<C, S, Rest<C>>),
    /// With lookups.
    WithLookup(FqTape<C, S, Send<Commitment<C, Aggregation>, Rest<C>>>),
}

/// The digest of the previous accumulators' challenges, on its own sponge.
pub type PreviousChallengesDigest<F> =
    Statement<Message<Vec<Vec<F>>, PreviousChallenges>, Peek<DigestOf<PreviousChallenges>, End>>;

/// The scalar-field sponge's part: the two digests, `ft(zeta w)` and the
/// evaluations, then `v` and `u`.
pub type Evaluation<F> = Statement<
    Message<Scalar<F>, Digest>,
    Statement<
        Message<Scalar<F>, PreviousChallenges>,
        Send<
            Message<Scalar<F>, FtEval1>,
            Send<
                Message<ProofEvaluations<F>, Evaluations>,
                Squeeze<Challenge<Polyscale>, Squeeze<Challenge<Evalscale>, End>>,
            >,
        >,
    >,
>;

/// A tape of the base-field sponge at the step `P`, in the field itself.
pub type FqTape<C, S, P> = FqTapeOn<C, S, Native, P>;

/// A tape of the base-field sponge in the model `H` handles, at the step
/// `P`.
pub type FqTapeOn<C, S, H, P> = Tape<<C as arrabbiata_curve::Curve>::Base, FqSponge<C, S, H>, P>;

/// A tape of Kimchi's base-field sponge in `M`, over the permutation `P`,
/// at the step `Pattern`.
pub type FqTapeIn<'m, C, M, P, Pattern> = FqTapeOn<
    C,
    Duplexed<'m, <M as FieldModel<<C as arrabbiata_curve::Curve>::Base>>::Value, P>,
    &'m RefCell<M>,
    Pattern,
>;

/// A tape of the scalar-field sponge at the step `P`, in the field itself.
pub type FrTape<F, S, P> = FrTapeOn<F, S, Native, P>;

/// A tape of the scalar-field sponge in the model `H` handles, at the step
/// `P`.
pub type FrTapeOn<F, S, H, P> = Tape<F, FrSponge<F, S, H>, P>;

/// The base-field sponge's tape at the start of [`Piop`].
#[must_use]
pub fn piop<C, S>(sponge: S) -> FqTape<C, S, Piop<C>>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    piop_on(FqSponge::new(sponge))
}

/// The tape of `sponge` at the start of [`PiopOf`], at the carriers `P` of
/// a point and `G` of a commitment: the one definition [`piop`] and
/// [`piop_in`] read.
#[must_use]
#[allow(clippy::type_complexity)]
pub fn piop_on<C, S, H, P, G>(
    sponge: FqSponge<C, S, H>,
) -> FqTapeOn<C, S, H, PiopOf<Value<C::Base, H>, P, G>>
where
    C: arrabbiata_curve::Curve,
    H: SpongeIn<C::Base, S>,
{
    Tape::undescribed(sponge)
}

/// The tape of Kimchi's base-field sponge in `model`, over `permutation`,
/// at the start of [`PiopIn`]: [`piop`] at the model's carrier.
#[must_use]
pub fn piop_in<'m, C, M, P>(
    model: &'m RefCell<M>,
    permutation: &'m P,
) -> FqTapeIn<'m, C, M, P, PiopIn<M::Value>>
where
    C: arrabbiata_curve::Curve,
    M: FieldModel<C::Base>,
    P: PermutationIn<M, C::Base, WIDTH>,
{
    piop_on(FqSponge::in_model(model, permutation))
}

/// The scalar-field sponge's tape at the start of [`Evaluation`].
#[must_use]
pub fn evaluation<F, S>(sponge: S) -> FrTape<F, S, Evaluation<F>>
where
    F: PrimeField,
    S: Transcript<F>,
{
    evaluation_on(FrSponge::new(sponge))
}

/// The digest of the previous accumulators' challenges, on a fresh
/// scalar-field sponge.
#[must_use]
pub fn previous_challenges_digest<F, S>(sponge: S, challenges: Vec<Vec<F>>) -> F
where
    F: PrimeField,
    S: Transcript<F> + Clone,
{
    previous_challenges_digest_on(FrSponge::new(sponge), challenges)
}

/// The tape of `sponge` at the start of [`Evaluation`], at the carrier of
/// its model: the one definition [`evaluation`] and [`evaluation_in`] read.
#[must_use]
pub fn evaluation_on<F, S, H>(
    sponge: FrSponge<F, S, H>,
) -> FrTapeOn<F, S, H, Evaluation<Value<F, H>>>
where
    F: Field,
    H: SpongeIn<F, S>,
{
    Tape::undescribed(sponge)
}

/// The digest of the previous accumulators' challenges on `sponge`, fresh:
/// the one definition [`previous_challenges_digest`] and
/// [`previous_challenges_digest_in`] read.
pub fn previous_challenges_digest_on<F, S, H>(
    sponge: FrSponge<F, S, H>,
    challenges: Vec<Vec<Value<F, H>>>,
) -> Value<F, H>
where
    F: Field,
    H: SpongeIn<F, S>,
    S: Clone,
{
    let tape: FrTapeOn<F, S, H, PreviousChallengesDigest<Value<F, H>>> = Tape::undescribed(sponge);
    let tape = tape.statement(&Message::new(challenges));
    let (digest, _) = tape.peek();
    digest
}

/// Kimchi's scalar-field sponge in a model of the field signature:
/// [`FrSponge`] over Kimchi's duplex ([`Duplexed`]) over a permutation in
/// the model, its challenges the lowest 128 bits of a squeeze
/// ([`ChallengeModel`]). It is the native sponge at another model, so the
/// patterns of this module ([`Evaluation`], [`PreviousChallengesDigest`])
/// and their messages' writers are the same: a verifier in a circuit runs
/// the transcript the native verifier runs, by the same definitions.
///
/// The model is shared ([`RefCell`]), so the sponge is a value a tape can
/// hold and copy (a [`Peek`] draws from a copy).
pub type FrSpongeIn<'m, F, M, P> =
    FrSponge<F, Duplexed<'m, <M as FieldModel<F>>::Value, P>, &'m RefCell<M>>;

/// A tape of the scalar-field sponge in a model, at the step `P`.
pub type FrTapeIn<'m, F, M, P, Pattern> = Tape<F, FrSpongeIn<'m, F, M, P>, Pattern>;

/// The tape of the scalar-field sponge in `model` at the start of
/// [`Evaluation`], at the model's carrier.
#[must_use]
pub fn evaluation_in<'m, F, M, P>(
    model: &'m RefCell<M>,
    permutation: &'m P,
) -> FrTapeIn<'m, F, M, P, Evaluation<M::Value>>
where
    F: Field,
    M: FieldModel<F>,
    P: PermutationIn<M, F, WIDTH>,
{
    evaluation_on(FrSponge::in_model(model, permutation))
}

/// The digest of the previous accumulators' challenges in `model`, on a
/// fresh sponge: [`previous_challenges_digest`] at the model's carrier.
pub fn previous_challenges_digest_in<F, M, P>(
    model: &RefCell<M>,
    permutation: &P,
    challenges: Vec<Vec<M::Value>>,
) -> M::Value
where
    F: Field,
    M: FieldModel<F>,
    P: PermutationIn<M, F, WIDTH>,
{
    previous_challenges_digest_on(FrSponge::in_model(model, permutation), challenges)
}
