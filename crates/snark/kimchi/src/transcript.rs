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
//! argument's own transcript ([`arrabbiata_ipa::generic`]).
//!
//! The tapes start [`Tape::undescribed`]: Kimchi's transcript absorbs no
//! description of its pattern, and ours must be Kimchi's bit for bit.

use core::marker::PhantomData;

use alloc::vec::Vec;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::PrimeField;
use arrabbiata_transcript::{
    Transcript,
    typed::{Branch, Draw, End, Label, Peek, Send, Squeeze, Statement, Tape, Write},
};

use crate::{
    constraints::COLUMNS,
    oracle::{FqSponge, FrSponge, ScalarChallenge},
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

impl<C, S, Name> Write<FqSponge<C, S>> for Message<Base<C::Base>, Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn write(&self, sponge: &mut FqSponge<C, S>) {
        sponge.absorb_fq(&[self.0.0]);
    }
}

impl<C, S, Name> Write<FqSponge<C, S>> for Message<Vec<Projective<C>>, Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn write(&self, sponge: &mut FqSponge<C, S>) {
        for p in &self.0 {
            sponge.absorb_g(&[*p]);
        }
    }
}

impl<C, S, Name> Write<FqSponge<C, S>> for Message<PolyComm<C>, Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn write(&self, sponge: &mut FqSponge<C, S>) {
        sponge.absorb_g(&self.0.chunks);
    }
}

impl<C, S, Name> Write<FqSponge<C, S>> for Message<Vec<PolyComm<C>>, Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn write(&self, sponge: &mut FqSponge<C, S>) {
        for c in &self.0 {
            sponge.absorb_g(&c.chunks);
        }
    }
}

impl<C, S, Name> Write<FqSponge<C, S>> for Message<[PolyComm<C>; COLUMNS], Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn write(&self, sponge: &mut FqSponge<C, S>) {
        for c in &self.0 {
            sponge.absorb_g(&c.chunks);
        }
    }
}

/// A 128-bit challenge of the base-field sponge, as the scalar its limbs
/// are (the caller maps it through the endomorphism).
impl<C, S, Name> Draw<FqSponge<C, S>> for Challenge<Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    type Value = C::Scalar;

    fn draw(sponge: &mut FqSponge<C, S>) -> C::Scalar {
        sponge.challenge()
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

impl<C, S, Name> Draw<FqSponge<C, S>> for DigestOf<Name>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base> + Clone,
{
    type Value = C::Scalar;

    fn draw(sponge: &mut FqSponge<C, S>) -> C::Scalar {
        sponge.clone().digest()
    }
}

impl<F, S, Name> Write<FrSponge<F, S>> for Message<Scalar<F>, Name>
where
    F: PrimeField,
    S: Transcript<F>,
{
    fn write(&self, sponge: &mut FrSponge<F, S>) {
        sponge.absorb(self.0.0);
    }
}

impl<F, S, Name> Write<FrSponge<F, S>> for Message<Vec<Vec<F>>, Name>
where
    F: PrimeField,
    S: Transcript<F>,
{
    fn write(&self, sponge: &mut FrSponge<F, S>) {
        for chals in &self.0 {
            sponge.absorb_multiple(chals);
        }
    }
}

impl<F, S, Name> Write<FrSponge<F, S>> for Message<ProofEvaluations<F>, Name>
where
    F: PrimeField,
    S: Transcript<F>,
{
    fn write(&self, sponge: &mut FrSponge<F, S>) {
        sponge.absorb_multiple(&self.0.public.zeta);
        sponge.absorb_multiple(&self.0.public.zeta_omega);
        for e in self.0.absorbed() {
            sponge.absorb_multiple(&e.zeta);
            sponge.absorb_multiple(&e.zeta_omega);
        }
    }
}

/// A 128-bit challenge of the scalar-field sponge.
impl<F, S, Name> Draw<FrSponge<F, S>> for Challenge<Name>
where
    F: PrimeField,
    S: Transcript<F>,
{
    type Value = ScalarChallenge<F>;

    fn draw(sponge: &mut FrSponge<F, S>) -> ScalarChallenge<F> {
        sponge.challenge()
    }
}

/// The scalar-field sponge's digest.
impl<F, S, Name> Draw<FrSponge<F, S>> for DigestOf<Name>
where
    F: PrimeField,
    S: Transcript<F> + Clone,
{
    type Value = F;

    fn draw(sponge: &mut FrSponge<F, S>) -> F {
        sponge.clone().digest()
    }
}

/// A commitment message.
pub type Commitment<C, Name> = Message<PolyComm<C>, Name>;

/// After the sorted columns: `beta`, `gamma`, the aggregation, then
/// [`Rest`].
pub type WithLookup<C> =
    Squeeze<Challenge<Beta>, Squeeze<Challenge<Gamma>, Send<Commitment<C, Aggregation>, Rest<C>>>>;

/// After the witness columns, without lookups: `beta`, `gamma`, then
/// [`Rest`].
pub type WithoutLookup<C> = Squeeze<Challenge<Beta>, Squeeze<Challenge<Gamma>, Rest<C>>>;

/// From `t`: `zeta`, and the digest from a copy.
pub type AfterAlpha<C> =
    Send<Commitment<C, Quotient>, Squeeze<Challenge<Zeta>, Peek<DigestOf<Digest>, End>>>;

/// From `z`: `alpha`, then [`AfterAlpha`].
pub type Rest<C> = Send<Commitment<C, Permutation>, Squeeze<Challenge<Alpha>, AfterAlpha<C>>>;

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

/// The sorted columns, then [`WithLookup`].
pub type Sorted<C> = Send<Message<Vec<PolyComm<C>>, SortedColumns>, WithLookup<C>>;

/// The joint combiner when a lookup is joint, or none: then [`Sorted`].
pub type Joint<C> = Branch<Squeeze<Challenge<JointCombiner>, Sorted<C>>, Sorted<C>>;

/// The lookups: the runtime table's commitment or none, then [`Joint`].
pub type Lookups<C> = Branch<Send<Commitment<C, RuntimeTable>, Joint<C>>, Joint<C>>;

/// The base-field sponge's part of Kimchi's PIOP.
pub type Piop<C> = Statement<
    Message<Base<<C as arrabbiata_curve::Curve>::Base>, VerifierIndexDigest>,
    Statement<
        Message<Vec<Projective<C>>, PreviousCommitments>,
        Statement<
            Commitment<C, PublicInput>,
            Send<
                Message<[PolyComm<C>; COLUMNS], WitnessColumns>,
                Branch<WithoutLookup<C>, Lookups<C>>,
            >,
        >,
    >,
>;

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

/// A tape of the base-field sponge at the step `P`.
pub type FqTape<C, S, P> = Tape<<C as arrabbiata_curve::Curve>::Base, FqSponge<C, S>, P>;

/// A tape of the scalar-field sponge at the step `P`.
pub type FrTape<F, S, P> = Tape<F, FrSponge<F, S>, P>;

/// The base-field sponge's tape at the start of [`Piop`].
#[must_use]
pub fn piop<C, S>(sponge: S) -> FqTape<C, S, Piop<C>>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    Tape::undescribed(FqSponge::new(sponge))
}

/// The scalar-field sponge's tape at the start of [`Evaluation`].
#[must_use]
pub fn evaluation<F, S>(sponge: S) -> FrTape<F, S, Evaluation<F>>
where
    F: PrimeField,
    S: Transcript<F>,
{
    Tape::undescribed(FrSponge::new(sponge))
}

/// The digest of the previous accumulators' challenges, on a fresh
/// scalar-field sponge.
#[must_use]
pub fn previous_challenges_digest<F, S>(sponge: S, challenges: Vec<Vec<F>>) -> F
where
    F: PrimeField,
    S: Transcript<F> + Clone,
{
    let tape: FrTape<F, S, PreviousChallengesDigest<F>> = Tape::undescribed(FrSponge::new(sponge));
    let tape = tape.statement(&Message::new(challenges));
    let (digest, _) = tape.peek();
    digest
}
