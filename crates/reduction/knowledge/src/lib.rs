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
//! # The transcript is a type
//!
//! Each reduction declares its Fiat-Shamir steps as a pattern of the typed
//! transcript ([`Reduction::Pattern`], an `arrabbiata_transcript::typed`
//! sequence). It runs on a tape at its pattern followed by whatever comes
//! next, and hands back the tape at what comes next, so its steps happen in
//! the order its pattern says or it does not compile. The combinators build
//! the composite's pattern from their parts': [`Then`] and [`Parallel`] the
//! sequence of the two, [`Each`] the repetition, the structural maps
//! ([`Identity`], [`Single`], [`Concat`]) the empty pattern. So the pattern is
//! a strict monoidal functor from reductions under composition to patterns
//! under concatenation, and the type of a composite's whole tape is computed
//! by the type checker. [`tape`] starts that tape, absorbing the
//! composite's description once.
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
//! | finite powers `R^*` (lists) | [`Batch`], with [`Single`], [`Concat`] |
//! | `A^* : R1^* -> R2^*` | [`Each`] |
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

extern crate alloc;

use alloc::{vec, vec::Vec};
use core::{convert::Infallible, fmt};

use arrabbiata_field::Field;
use arrabbiata_transcript::{
    Transcript,
    typed::{End, Pattern, Rounds, Seq, Sequence, Tail, Tape},
};

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

    /// Every Fiat-Shamir step the reduction takes, in order: the pattern of
    /// its tape.
    type Pattern: Sequence;

    /// The relation reduced from.
    fn source(&self) -> Self::Source;

    /// The relation reduced to.
    fn target(&self) -> Self::Target;

    /// Run the prover on `(instance, witness)`, on a tape at this
    /// reduction's pattern followed by `R`: the output, with the tape at
    /// `R`.
    ///
    /// # Errors
    ///
    /// When the prover cannot run on its input (a witness of the wrong
    /// shape); a scheme may also refuse a pair outside the source.
    #[allow(clippy::type_complexity)]
    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Self::Error>
    where
        T: Transcript<F>;

    /// Run the verifier on `instance` and `proof`, on a tape at this
    /// reduction's pattern followed by `R`: the output instance, and the
    /// tape at `R`.
    ///
    /// # Errors
    ///
    /// When the proof is rejected outright (a reduction may also accept any
    /// proof and leave the check to the target relation).
    #[allow(clippy::type_complexity)]
    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, Tape<F, T, R>), Self::Error>
    where
        T: Transcript<F>;

    /// [`Reduction::prove`] as the whole protocol on `transcript`: on the
    /// tape [`tape`] starts, run to its end.
    ///
    /// # Errors
    ///
    /// As [`Reduction::prove`].
    #[allow(clippy::type_complexity)]
    fn prove_on<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, End>>, Self::Error>
    where
        Self: Sized,
        T: Transcript<F>,
        Tail<Self::Pattern, End>: Pattern,
    {
        self.prove(Tape::start(transcript), instance, witness)
    }

    /// [`Reduction::verify`] as the whole protocol on `transcript`: on the
    /// tape [`tape`] starts, run to its end.
    ///
    /// # Errors
    ///
    /// As [`Reduction::verify`].
    #[allow(clippy::type_complexity)]
    fn verify_on<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, Tape<F, T, End>), Self::Error>
    where
        Self: Sized,
        T: Transcript<F>,
        Tail<Self::Pattern, End>: Pattern,
    {
        self.verify(Tape::start(transcript), instance, proof)
    }
}

/// The tape of `R`'s whole protocol on `transcript`, its description
/// absorbed first: the domain separation of the protocol by its own pattern.
/// A composite's description is its parts' in sequence, so two different
/// composites diverge from their first step.
#[must_use]
pub fn tape<F, R, T>(transcript: T) -> Tape<F, T, Tail<R::Pattern, End>>
where
    F: Field,
    R: Reduction<F>,
    T: Transcript<F>,
    Tail<R::Pattern, End>: Pattern,
{
    Tape::start(transcript)
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
    type Pattern = R::Pattern;

    fn source(&self) -> Self::Source {
        (*self).source()
    }

    fn target(&self) -> Self::Target {
        (*self).target()
    }

    fn prove<T, Rest>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, Rest>>,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Self::Error>
    where
        T: Transcript<F>,
    {
        (*self).prove(tape, instance, witness)
    }

    fn verify<T, Rest>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, Rest>>,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, Tape<F, T, Rest>), Self::Error>
    where
        T: Transcript<F>,
    {
        (*self).verify(tape, instance, proof)
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
    type Pattern = End;

    fn source(&self) -> R {
        self.0.clone()
    }

    fn target(&self) -> R {
        self.0.clone()
    }

    fn prove<T, Rest>(
        &self,
        tape: Tape<F, T, Rest>,
        instance: &R::Instance,
        witness: &R::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Infallible>
    where
        T: Transcript<F>,
    {
        Ok(Output {
            instance: instance.clone(),
            witness: witness.clone(),
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T, Rest>(
        &self,
        tape: Tape<F, T, Rest>,
        instance: &R::Instance,
        (): (),
    ) -> Result<(R::Instance, Tape<F, T, Rest>), Infallible>
    where
        T: Transcript<F>,
    {
        Ok((instance.clone(), tape))
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
    type Pattern = Seq<A::Pattern, B::Pattern>;

    fn source(&self) -> Self::Source {
        self.0.source()
    }

    fn target(&self) -> Self::Target {
        self.1.target()
    }

    fn prove<T, Rest>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, Rest>>,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Self::Error>
    where
        T: Transcript<F>,
    {
        let first = self
            .0
            .prove(tape, instance, witness)
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

    fn verify<T, Rest>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, Rest>>,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, Tape<F, T, Rest>), Self::Error>
    where
        T: Transcript<F>,
    {
        let (middle, tape) = self
            .0
            .verify(tape, instance, proof.0)
            .map_err(Either::Left)?;
        self.1.verify(tape, &middle, proof.1).map_err(Either::Right)
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
    type Pattern = Seq<A::Pattern, B::Pattern>;

    fn source(&self) -> Self::Source {
        Product(self.0.source(), self.1.source())
    }

    fn target(&self) -> Self::Target {
        Product(self.0.target(), self.1.target())
    }

    fn prove<T, Rest>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, Rest>>,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Self::Error>
    where
        T: Transcript<F>,
    {
        let left = self
            .0
            .prove(tape, &instance.0, &witness.0)
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

    fn verify<T, Rest>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, Rest>>,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, Tape<F, T, Rest>), Self::Error>
    where
        T: Transcript<F>,
    {
        let (left, tape) = self
            .0
            .verify(tape, &instance.0, proof.0)
            .map_err(Either::Left)?;
        let (right, tape) = self
            .1
            .verify(tape, &instance.1, proof.1)
            .map_err(Either::Right)?;
        Ok(((left, right), tape))
    }
}

/// Why a list failed: lengths that do not match, or the element at an
/// index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Indexed<E> {
    /// Two lists that must have the same length do not.
    Length,
    /// The element at this index failed.
    At(usize, E),
}

impl<E> fmt::Display for Indexed<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length => write!(f, "lengths do not match"),
            Self::At(i, e) => write!(f, "at {i}: {e}"),
        }
    }
}

impl<E> core::error::Error for Indexed<E> where E: fmt::Debug + fmt::Display {}

/// The finite powers of a relation, all at once: `R^*`, lists of pairs of
/// `R` (the free monoid on `R` under the monoidal product, up to the
/// associativity and unit isomorphisms). A list is in it when each pair is
/// in `R`.
///
/// It is what an accumulation scheme consumes: any number of accumulators,
/// not a fixed `R x R`. [`Single`] and [`Concat`] are its unit and
/// multiplication, and [`Each`] its action on reductions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Batch<R>(pub R);

impl<R> Relation for Batch<R>
where
    R: Relation,
{
    type Instance = Vec<R::Instance>;
    type Witness = Vec<R::Witness>;
    type Error = Indexed<R::Error>;

    fn check(&self, instance: &Self::Instance, witness: &Self::Witness) -> Result<(), Self::Error> {
        if instance.len() != witness.len() {
            return Err(Indexed::Length);
        }
        let mut i = 0;
        while i < instance.len() {
            self.0
                .check(&instance[i], &witness[i])
                .map_err(|e| Indexed::At(i, e))?;
            i += 1;
        }
        Ok(())
    }
}

/// `A^* : R1^* -> R2^*`, the reduction `A` run on each element of a list,
/// in order, on one transcript. The proof is the list of proofs.
///
/// `Each` is a functor: `Each(Identity) = Identity` and `Each(Then(A, B))
/// = Then(Each(A), Each(B))`, both up to the shape of the proofs, which the
/// laws compare by the instances the verifiers output. It is a functor on
/// reductions, not on their tapes: its pattern is the repetition of `A`'s
/// (`Rounds`), and `(A;B)*` and `A*;B*` are different patterns, so the two
/// sides of the second law are different protocols under Fiat-Shamir. They
/// agree on their instances when their absorptions coincide (when `A` takes
/// no step, on undescribed tapes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Each<A>(pub A);

impl<F, A> Reduction<F> for Each<A>
where
    F: Field,
    A: Reduction<F>,
{
    type Source = Batch<A::Source>;
    type Target = Batch<A::Target>;
    type Proof = Vec<A::Proof>;
    type Error = Indexed<A::Error>;
    type Pattern = Rounds<A::Pattern, End>;

    fn source(&self) -> Self::Source {
        Batch(self.0.source())
    }

    fn target(&self) -> Self::Target {
        Batch(self.0.target())
    }

    fn prove<T, Rest>(
        &self,
        mut tape: Tape<F, T, Rounds<A::Pattern, Rest>>,
        instance: &Vec<<A::Source as Relation>::Instance>,
        witness: &Vec<<A::Source as Relation>::Witness>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Self::Error>
    where
        T: Transcript<F>,
    {
        if instance.len() != witness.len() {
            return Err(Indexed::Length);
        }
        let mut instances = Vec::with_capacity(instance.len());
        let mut witnesses = Vec::with_capacity(instance.len());
        let mut proofs = Vec::with_capacity(instance.len());
        let mut i = 0;
        while i < instance.len() {
            let out = self
                .0
                .prove(tape.round(), &instance[i], &witness[i])
                .map_err(|e| Indexed::At(i, e))?;
            instances.push(out.instance);
            witnesses.push(out.witness);
            proofs.push(out.proof);
            tape = out.transcript;
            i += 1;
        }
        Ok(Output {
            instance: instances,
            witness: witnesses,
            proof: proofs,
            transcript: tape.done(),
        })
    }

    fn verify<T, Rest>(
        &self,
        mut tape: Tape<F, T, Rounds<A::Pattern, Rest>>,
        instance: &Vec<<A::Source as Relation>::Instance>,
        proof: Vec<A::Proof>,
    ) -> Result<(Vec<<A::Target as Relation>::Instance>, Tape<F, T, Rest>), Self::Error>
    where
        T: Transcript<F>,
    {
        if instance.len() != proof.len() {
            return Err(Indexed::Length);
        }
        let mut instances = Vec::with_capacity(instance.len());
        for (i, (u, p)) in instance.iter().zip(proof).enumerate() {
            let (v, t) = self
                .0
                .verify(tape.round(), u, p)
                .map_err(|e| Indexed::At(i, e))?;
            instances.push(v);
            tape = t;
        }
        Ok((instances, tape.done()))
    }
}

/// `R -> R^*`, a pair as a list of one: the unit of [`Batch`]. No messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Single<R>(pub R);

impl<F, R> Reduction<F> for Single<R>
where
    F: Field,
    R: Relation + Clone,
    R::Instance: Clone,
    R::Witness: Clone,
{
    type Source = R;
    type Target = Batch<R>;
    type Proof = ();
    type Error = Infallible;
    type Pattern = End;

    fn source(&self) -> R {
        self.0.clone()
    }

    fn target(&self) -> Batch<R> {
        Batch(self.0.clone())
    }

    fn prove<T, Rest>(
        &self,
        tape: Tape<F, T, Rest>,
        instance: &R::Instance,
        witness: &R::Witness,
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Infallible>
    where
        T: Transcript<F>,
    {
        Ok(Output {
            instance: vec![instance.clone()],
            witness: vec![witness.clone()],
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T, Rest>(
        &self,
        tape: Tape<F, T, Rest>,
        instance: &R::Instance,
        (): (),
    ) -> Result<(Vec<R::Instance>, Tape<F, T, Rest>), Infallible>
    where
        T: Transcript<F>,
    {
        Ok((vec![instance.clone()], tape))
    }
}

/// `R^* x R^* -> R^*`, two lists as one, the first first: the
/// multiplication of [`Batch`]. It is associative, with the empty list as
/// unit, which is what makes `Batch` the free monoid. No messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Concat<R>(pub R);

impl<F, R> Reduction<F> for Concat<R>
where
    F: Field,
    R: Relation + Clone,
    R::Instance: Clone,
    R::Witness: Clone,
{
    type Source = Product<Batch<R>, Batch<R>>;
    type Target = Batch<R>;
    type Proof = ();
    type Error = Infallible;
    type Pattern = End;

    fn source(&self) -> Self::Source {
        Product(Batch(self.0.clone()), Batch(self.0.clone()))
    }

    fn target(&self) -> Batch<R> {
        Batch(self.0.clone())
    }

    fn prove<T, Rest>(
        &self,
        tape: Tape<F, T, Rest>,
        instance: &(Vec<R::Instance>, Vec<R::Instance>),
        witness: &(Vec<R::Witness>, Vec<R::Witness>),
    ) -> Result<OutputOf<F, Self, Tape<F, T, Rest>>, Infallible>
    where
        T: Transcript<F>,
    {
        let mut instances = instance.0.clone();
        instances.extend_from_slice(&instance.1);
        let mut witnesses = witness.0.clone();
        witnesses.extend_from_slice(&witness.1);
        Ok(Output {
            instance: instances,
            witness: witnesses,
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T, Rest>(
        &self,
        tape: Tape<F, T, Rest>,
        instance: &(Vec<R::Instance>, Vec<R::Instance>),
        (): (),
    ) -> Result<(Vec<R::Instance>, Tape<F, T, Rest>), Infallible>
    where
        T: Transcript<F>,
    {
        let mut instances = instance.0.clone();
        instances.extend_from_slice(&instance.1);
        Ok((instances, tape))
    }
}

/// Executable laws of a reduction, for tests.
pub mod laws {
    use super::{Either, Reduction, Relation, Transcript, tape};
    use arrabbiata_field::Field;
    use arrabbiata_transcript::typed::{End, Pattern, Tail, Tape};

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

    /// Run the reduction on a pair in the source relation, on two tapes of
    /// its whole protocol over fresh copies of the transcript ([`tape`]),
    /// and check the laws: the pair is in the source, the verifier's
    /// instance is the prover's, and the output pair is in the target.
    /// Returns the prover's output on success, its tape at the end.
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
        super::OutputOf<F, R, Tape<F, T, End>>,
        Failure<R::Error, <R::Source as Relation>::Error, <R::Target as Relation>::Error>,
    >
    where
        F: Field,
        R: Reduction<F>,
        R::Proof: Clone,
        <R::Target as Relation>::Instance: PartialEq,
        T: Transcript<F>,
        Tail<R::Pattern, End>: Pattern,
    {
        reduction
            .source()
            .check(instance, witness)
            .map_err(Failure::Source)?;
        let out = reduction
            .prove(tape::<F, R, T>(transcript()), instance, witness)
            .map_err(Failure::Reduction)?;
        let (verified, _) = reduction
            .verify(tape::<F, R, T>(transcript()), instance, out.proof.clone())
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
    /// input, as the instances their verifiers output, each on a tape of its
    /// own protocol: equal when the law holds. Used for the category laws
    /// (`Then(Identity, A) = A`, ...).
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
        Tail<A::Pattern, End>: Pattern,
        Tail<B::Pattern, End>: Pattern,
    {
        let (ua, _) = a
            .verify(tape::<F, A, T>(transcript()), instance, proof_a)
            .map_err(Either::Left)?;
        let (ub, _) = b
            .verify(tape::<F, B, T>(transcript()), instance, proof_b)
            .map_err(Either::Right)?;
        Ok(ua == ub)
    }
}
