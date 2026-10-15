//! Reductions of knowledge, as types.
//!
//! A *relation* `R` is a set of pairs (instance, witness), given here by a
//! check ([`Relation`]). A *reduction of knowledge* from `R1` to `R2`
//! (Kothapalli and Parno, *Algebraic reductions of knowledge*, CRYPTO
//! 2023, Definition 1) is an interactive protocol between a prover holding
//! `(u1, w1)` in `R1` and a verifier holding `u1`, at the end of which
//! both hold an instance `u2` and the prover a witness `w2`, such that:
//!
//! - *completeness*: an honest run from `(u1, w1)` in `R1` ends with
//!   `(u2, w2)` in `R2`;
//! - *knowledge soundness*: from any prover making the verifier output
//!   `u2` together with a `w2` such that `(u2, w2)` is in `R2`, a `w1` with
//!   `(u1, w1)` in `R1` can be extracted.
//!
//! Here the protocols are non-interactive, on a Fiat-Shamir transcript
//! ([`Reduction`]), and *publicly reducible*: the verifier computes the
//! output instance itself from the input instance and the proof.
//!
//! # The category
//!
//! Relations and reductions form a symmetric monoidal category (Kothapalli,
//! PhD thesis, chapter 8, Construction 8.1 and Theorem 8.1):
//!
//! | category | here |
//! | --- | --- |
//! | objects | [`Relation`]s |
//! | morphisms `R1 -> R2` | [`Reduction`]s with `Source = R1`, `Target = R2` |
//! | identity | [`Identity`] |
//! | composition (Theorem 1) | [`Then`] |
//! | monoidal product (Theorem 2) | [`Product`] of relations, [`Parallel`] of reductions |
//! | unit | [`Unit`] |
//!
//! Two special shapes are the ones accumulation is built from:
//!
//! - an *argument of knowledge* for `R` is a reduction `R -> R_top`, with
//!   `R_top = {(true, bottom)}` ([`Top`]; Kothapalli and Parno,
//!   Definition 3), so verifying it is the whole check ([`Argument`]);
//! - an *`l`-folding scheme* for `R` is a reduction `R^l -> R`
//!   (Definition 4).
//!
//! An accumulation scheme for a relation `R` is then a relaxation `R'`, an
//! embedding `R -> R'`, a fold `R' x R' -> R'` and a decider `R' -> R_top`.
//! The decider composed with the embedding is an argument for `R`: that is
//! [`Then`] of two morphisms, and it is why a decider only needs to know
//! `R'`.
//!
//! A worked example, the folding of linear relations, is in
//! `tests/category.rs`.

#![no_std]

use core::{convert::Infallible, fmt};

use arrabbiata_field::Field;
use arrabbiata_transcript::Transcript;

/// A relation: the pairs (instance, witness) that pass [`Relation::check`].
///
/// A value of a type implementing it carries the parameters the relation
/// depends on (a circuit, a commitment key): it is a handle on one
/// relation of a family, cheap to clone.
pub trait Relation {
    /// An instance: what the verifier knows.
    type Instance;

    /// A witness: what the prover knows in addition.
    type Witness;

    /// Why a pair is not in the relation.
    type Error;

    /// `Ok(())` exactly when `(instance, witness)` is in the relation.
    ///
    /// # Errors
    ///
    /// Why the pair is not in the relation.
    fn check(&self, instance: &Self::Instance, witness: &Self::Witness) -> Result<(), Self::Error>;
}

/// `R_top = {(true, bottom)}`: the relation with one pair, the target of an
/// argument of knowledge. Reducing to it leaves nothing to check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Top;

impl Relation for Top {
    type Instance = ();
    type Witness = ();
    type Error = Infallible;

    fn check(&self, (): &(), (): &()) -> Result<(), Infallible> {
        Ok(())
    }
}

/// `R_I = {(bottom, bottom)}`, the unit of the monoidal product: `R x R_I`
/// is `R` up to the isomorphism forgetting the second component.
///
/// As a relation it is the same set as [`Top`]; the two names record the
/// two roles (terminal target of arguments, unit of products).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unit;

impl Relation for Unit {
    type Instance = ();
    type Witness = ();
    type Error = Infallible;

    fn check(&self, (): &(), (): &()) -> Result<(), Infallible> {
        Ok(())
    }
}

/// One of two things: which side of a product or a composition failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Either<A, B> {
    /// The first (left) component.
    Left(A),
    /// The second (right) component.
    Right(B),
}

impl<A, B> fmt::Display for Either<A, B>
where
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left(a) => write!(f, "left: {a}"),
            Self::Right(b) => write!(f, "right: {b}"),
        }
    }
}

impl<A, B> core::error::Error for Either<A, B>
where
    A: fmt::Debug + fmt::Display,
    B: fmt::Debug + fmt::Display,
{
}

/// The product `R1 x R2`: pairs of instances and pairs of witnesses, each
/// component in its relation (Kothapalli and Parno, Theorem 2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Product<R1, R2>(pub R1, pub R2);

impl<R1, R2> Relation for Product<R1, R2>
where
    R1: Relation,
    R2: Relation,
{
    type Instance = (R1::Instance, R2::Instance);
    type Witness = (R1::Witness, R2::Witness);
    type Error = Either<R1::Error, R2::Error>;

    fn check(&self, instance: &Self::Instance, witness: &Self::Witness) -> Result<(), Self::Error> {
        self.0
            .check(&instance.0, &witness.0)
            .map_err(Either::Left)?;
        self.1.check(&instance.1, &witness.1).map_err(Either::Right)
    }
}

/// What an honest prover ends with: the output pair, the proof, and the
/// transcript, so that a caller can continue the protocol on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output<I, W, P, T> {
    /// The output instance, which the verifier also computes.
    pub instance: I,
    /// The output witness.
    pub witness: W,
    /// The proof the verifier reads.
    pub proof: P,
    /// The transcript after the protocol.
    pub transcript: T,
}

/// The prover's output for the reduction `R`, on the transcript `T`.
pub type OutputOf<F, R, T> = Output<
    <<R as Reduction<F>>::Target as Relation>::Instance,
    <<R as Reduction<F>>::Target as Relation>::Witness,
    <R as Reduction<F>>::Proof,
    T,
>;

/// A non-interactive, publicly reducible reduction of knowledge from
/// [`Reduction::Source`] to [`Reduction::Target`], on a Fiat-Shamir
/// transcript over `F`.
///
/// Laws (checked by [`laws::complete`] on a given pair):
///
/// - the verifier's output instance is the prover's;
/// - completeness: from a pair in the source, the prover's output pair is
///   in the target.
///
/// Knowledge soundness is a property of the protocol, not of the code; its
/// executable shadow is that a corrupted proof or input leads to a target
/// pair that fails, which each scheme tests.
pub trait Reduction<F>
where
    F: Field,
{
    /// The relation reduced from.
    type Source: Relation;

    /// The relation reduced to.
    type Target: Relation;

    /// What the prover sends.
    type Proof;

    /// Why proving or verifying failed.
    type Error;

    /// The relation reduced from.
    fn source(&self) -> Self::Source;

    /// The relation reduced to.
    fn target(&self) -> Self::Target;

    /// Run the prover on `(instance, witness)`.
    ///
    /// # Errors
    ///
    /// When the prover cannot run on its input (a witness of the wrong
    /// shape); a scheme may also refuse a pair outside the source.
    #[allow(clippy::type_complexity)]
    fn prove<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, T>, Self::Error>
    where
        T: Transcript<F>;

    /// Run the verifier on `instance` and `proof`: the output instance, and
    /// the transcript after the protocol.
    ///
    /// # Errors
    ///
    /// When the proof is rejected outright (a reduction may also accept any
    /// proof and leave the check to the target relation).
    #[allow(clippy::type_complexity)]
    fn verify<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, T), Self::Error>
    where
        T: Transcript<F>;
}

/// A reference to a reduction is the same reduction, so composites can
/// borrow a reduction that owns keys instead of moving it.
impl<F, R> Reduction<F> for &R
where
    F: Field,
    R: Reduction<F>,
{
    type Source = R::Source;
    type Target = R::Target;
    type Proof = R::Proof;
    type Error = R::Error;

    fn source(&self) -> Self::Source {
        (*self).source()
    }

    fn target(&self) -> Self::Target {
        (*self).target()
    }

    fn prove<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, T>, Self::Error>
    where
        T: Transcript<F>,
    {
        (*self).prove(transcript, instance, witness)
    }

    fn verify<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, T), Self::Error>
    where
        T: Transcript<F>,
    {
        (*self).verify(transcript, instance, proof)
    }
}

/// An argument of knowledge: a reduction to [`Top`]. Verifying it is the
/// whole check, since the target relation holds for its only instance.
pub trait Argument<F>: Reduction<F, Target = Top>
where
    F: Field,
{
}

impl<F, R> Argument<F> for R
where
    F: Field,
    R: Reduction<F, Target = Top>,
{
}

/// The identity reduction `R -> R`: the prover outputs its input, the proof
/// is empty, the transcript is untouched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Identity<R>(pub R);

impl<F, R> Reduction<F> for Identity<R>
where
    F: Field,
    R: Relation + Clone,
    R::Instance: Clone,
    R::Witness: Clone,
{
    type Source = R;
    type Target = R;
    type Proof = ();
    type Error = Infallible;

    fn source(&self) -> R {
        self.0.clone()
    }

    fn target(&self) -> R {
        self.0.clone()
    }

    fn prove<T>(
        &self,
        transcript: T,
        instance: &R::Instance,
        witness: &R::Witness,
    ) -> Result<OutputOf<F, Self, T>, Infallible>
    where
        T: Transcript<F>,
    {
        Ok(Output {
            instance: instance.clone(),
            witness: witness.clone(),
            proof: (),
            transcript,
        })
    }

    fn verify<T>(
        &self,
        transcript: T,
        instance: &R::Instance,
        (): (),
    ) -> Result<(R::Instance, T), Infallible>
    where
        T: Transcript<F>,
    {
        Ok((instance.clone(), transcript))
    }
}

/// Sequential composition `B . A : R1 -> R3` of `A : R1 -> R2` and
/// `B : R2 -> R3` (Kothapalli and Parno, Theorem 1): run `A`, then `B` on
/// its output, on the same transcript. The proof is the pair of proofs.
///
/// That the target of `A` is the source of `B` is a bound of the
/// implementation, so composing mismatched reductions does not compile.
///
/// The target of the first must be the source of the second. `Top` and
/// `Unit` are the same set but not the same relation, so this does not
/// compile:
///
/// ```compile_fail
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_poseidon::vesta_sponge;
/// use arrabbiata_reduction::{Identity, Reduction, Then, Top, Unit};
/// let mismatched = Then(Identity(Top), Identity(Unit));
/// let _ = Reduction::<F>::prove(&mismatched, vesta_sponge(0), &(), &());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Then<A, B>(pub A, pub B);

impl<F, A, B> Reduction<F> for Then<A, B>
where
    F: Field,
    A: Reduction<F>,
    B: Reduction<F, Source = A::Target>,
{
    type Source = A::Source;
    type Target = B::Target;
    type Proof = (A::Proof, B::Proof);
    type Error = Either<A::Error, B::Error>;

    fn source(&self) -> Self::Source {
        self.0.source()
    }

    fn target(&self) -> Self::Target {
        self.1.target()
    }

    fn prove<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, T>, Self::Error>
    where
        T: Transcript<F>,
    {
        let first = self
            .0
            .prove(transcript, instance, witness)
            .map_err(Either::Left)?;
        let second = self
            .1
            .prove(first.transcript, &first.instance, &first.witness)
            .map_err(Either::Right)?;
        Ok(Output {
            instance: second.instance,
            witness: second.witness,
            proof: (first.proof, second.proof),
            transcript: second.transcript,
        })
    }

    fn verify<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, T), Self::Error>
    where
        T: Transcript<F>,
    {
        let (middle, transcript) = self
            .0
            .verify(transcript, instance, proof.0)
            .map_err(Either::Left)?;
        self.1
            .verify(transcript, &middle, proof.1)
            .map_err(Either::Right)
    }
}

/// Parallel composition `A x B : R1 x R3 -> R2 x R4` of `A : R1 -> R2` and
/// `B : R3 -> R4` (Kothapalli and Parno, Theorem 2), run one after the
/// other on the same transcript.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Parallel<A, B>(pub A, pub B);

impl<F, A, B> Reduction<F> for Parallel<A, B>
where
    F: Field,
    A: Reduction<F>,
    B: Reduction<F>,
{
    type Source = Product<A::Source, B::Source>;
    type Target = Product<A::Target, B::Target>;
    type Proof = (A::Proof, B::Proof);
    type Error = Either<A::Error, B::Error>;

    fn source(&self) -> Self::Source {
        Product(self.0.source(), self.1.source())
    }

    fn target(&self) -> Self::Target {
        Product(self.0.target(), self.1.target())
    }

    fn prove<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, T>, Self::Error>
    where
        T: Transcript<F>,
    {
        let left = self
            .0
            .prove(transcript, &instance.0, &witness.0)
            .map_err(Either::Left)?;
        let right = self
            .1
            .prove(left.transcript, &instance.1, &witness.1)
            .map_err(Either::Right)?;
        Ok(Output {
            instance: (left.instance, right.instance),
            witness: (left.witness, right.witness),
            proof: (left.proof, right.proof),
            transcript: right.transcript,
        })
    }

    fn verify<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, T), Self::Error>
    where
        T: Transcript<F>,
    {
        let (left, transcript) = self
            .0
            .verify(transcript, &instance.0, proof.0)
            .map_err(Either::Left)?;
        let (right, transcript) = self
            .1
            .verify(transcript, &instance.1, proof.1)
            .map_err(Either::Right)?;
        Ok(((left, right), transcript))
    }
}

/// Executable laws of a reduction, for tests.
pub mod laws {
    use super::{Either, Reduction, Relation, Transcript};
    use arrabbiata_field::Field;

    /// How a run of [`complete`] failed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Failure<E, S, T> {
        /// The input pair is not in the source relation.
        Source(S),
        /// The prover or the verifier failed.
        Reduction(E),
        /// The verifier's output instance is not the prover's.
        Disagree,
        /// The output pair is not in the target relation.
        Target(T),
    }

    /// Run the reduction on a pair in the source relation, on two copies of
    /// the same transcript, and check the laws: the pair is in the source,
    /// the verifier's instance is the prover's, and the output pair is in
    /// the target. Returns the prover's output on success.
    ///
    /// # Errors
    ///
    /// The first law that fails.
    #[allow(clippy::type_complexity)]
    pub fn complete<F, R, T>(
        reduction: &R,
        transcript: impl Fn() -> T,
        instance: &<R::Source as Relation>::Instance,
        witness: &<R::Source as Relation>::Witness,
    ) -> Result<
        super::OutputOf<F, R, T>,
        Failure<R::Error, <R::Source as Relation>::Error, <R::Target as Relation>::Error>,
    >
    where
        F: Field,
        R: Reduction<F>,
        R::Proof: Clone,
        <R::Target as Relation>::Instance: PartialEq,
        T: Transcript<F>,
    {
        reduction
            .source()
            .check(instance, witness)
            .map_err(Failure::Source)?;
        let out = reduction
            .prove(transcript(), instance, witness)
            .map_err(Failure::Reduction)?;
        let (verified, _) = reduction
            .verify(transcript(), instance, out.proof.clone())
            .map_err(Failure::Reduction)?;
        if verified != out.instance {
            return Err(Failure::Disagree);
        }
        reduction
            .target()
            .check(&out.instance, &out.witness)
            .map_err(Failure::Target)?;
        Ok(out)
    }

    /// The two sides of a law that relates two reductions on the same
    /// input, as the instances their verifiers output: equal when the law
    /// holds. Used for the category laws (`Then(Identity, A) = A`, ...).
    ///
    /// # Errors
    ///
    /// Which side failed to verify.
    #[allow(clippy::type_complexity)]
    pub fn same_instance<F, A, B, T>(
        a: &A,
        b: &B,
        transcript: impl Fn() -> T,
        instance: &<A::Source as Relation>::Instance,
        proof_a: A::Proof,
        proof_b: B::Proof,
    ) -> Result<bool, Either<A::Error, B::Error>>
    where
        F: Field,
        A: Reduction<F>,
        B: Reduction<
                F,
                Source = A::Source,
                Target: Relation<Instance = <A::Target as Relation>::Instance>,
            >,
        <A::Target as Relation>::Instance: PartialEq,
        T: Transcript<F>,
    {
        let (ua, _) = a
            .verify(transcript(), instance, proof_a)
            .map_err(Either::Left)?;
        let (ub, _) = b
            .verify(transcript(), instance, proof_b)
            .map_err(Either::Right)?;
        Ok(ua == ub)
    }
}
