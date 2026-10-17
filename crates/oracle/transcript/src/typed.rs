//! Typed Fiat-Shamir: the protocol is the type of the transcript.
//!
//! An interactive protocol is a shared tape on which the prover writes
//! messages and the verifier writes challenges. Fiat-Shamir replaces the
//! verifier by a sponge, and its classic failure is a mismatch between the
//! tape the security proof assumes and the tape the code actually writes: a
//! message not absorbed, absorbed after the challenge it should bind, or a
//! sponge state reused. This module makes the tape a type.
//!
//! # Patterns
//!
//! A *pattern* is a type-level list:
//!
//! - [`Public<M, Next>`]: both parties absorb a public value of type `M`
//!   (the statement). It is not part of the proof, since the verifier
//!   already knows it, but it is part of the tape, so a protocol must
//!   declare its statement to be well-typed;
//! - [`Message<M, Next>`]: the prover sends a value of type `M`;
//! - [`Challenge<Next>`]: the verifier sends one challenge;
//! - [`End`]: the protocol is over.
//!
//! The nodes above absorb and squeeze through [`Absorb`] and
//! [`Transcript::squeeze`]. A protocol whose sponge has more structure (a
//! sponge that buffers the limbs of its challenges, absorbs points and
//! scalars by its own rules, and draws challenges of several kinds) uses the
//! nodes that write and draw through the transcript type itself:
//!
//! - [`Statement<M, Next>`]: both parties write a public `M` ([`Write`]);
//! - [`Send<M, Next>`]: the prover sends `M` ([`Write`]);
//! - [`Squeeze<K, Next>`]: one challenge of the kind `K` ([`Draw`]), so a
//!   128-bit scalar challenge and a base-field element are different steps;
//! - [`Peek<K, Next>`]: a draw from a copy of the transcript, which leaves
//!   it as it was (a digest handed to another sponge);
//! - [`Rounds<Body, Next>`]: `Body` as many times as the parties agree,
//!   then `Next` (the rounds of an inner-product argument, as many as the
//!   logarithm of the key's size);
//! - [`Branch<A, B>`]: `A` or `B`, as the public statement decides (a
//!   protocol with optional rounds).
//!
//! A protocol reproducing an existing transcript bit for bit starts its
//! tape with [`Tape::undescribed`], which absorbs no description.
//!
//! Patterns compose sequentially with [`Then`]: `Then<P, Q>` is the pattern
//! that runs `P`, then `Q` (type-level list concatenation, so it is
//! associative with [`End`] as unit).
//!
//! # Tapes
//!
//! A [`Tape<F, T, P>`] is a transcript `T` that still has to follow the
//! pattern `P`. Each step consumes the tape and returns it at the rest of
//! the pattern, so:
//!
//! - steps happen in the order of the pattern, or the code does not compile;
//! - an old state cannot be reused (moves are affine), so a sponge cannot be
//!   forked by accident;
//! - only a tape at [`End`] can be finished, so a protocol cannot stop early
//!   without saying so.
//!
//! On creation, the tape absorbs a description of its pattern (the label of
//! every message and the position of every challenge). Two parties running
//! different protocols therefore diverge from the first challenge, which is
//! the domain separation of the protocol itself.
//!
//! Squeezing before the commitment is sent does not compile:
//!
//! ```compile_fail
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_poseidon::{goldilocks, goldilocks_sponge};
//! use arrabbiata_transcript::{typed::{Challenge, End, Message, Tape}};
//! type P = Message<F, Challenge<End>>;
//! let tape = Tape::<F, _, P>::start(goldilocks_sponge(0));
//! let (_c, _tape) = tape.challenge();
//! ```
//!
//! Neither does reusing a state:
//!
//! ```compile_fail
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_poseidon::{goldilocks, goldilocks_sponge};
//! use arrabbiata_transcript::{typed::{End, Message, Tape}};
//! type P = Message<F, End>;
//! let tape = Tape::<F, _, P>::start(goldilocks_sponge(0));
//! let _a = tape.send(&F::new(1));
//! let _b = tape.send(&F::new(2));
//! ```

use core::marker::PhantomData;

use arrabbiata_field::Field;

use crate::{
    Absorb, Transcript,
    hlist::{HCons, HList, HNil, Reverse},
    tag,
};

/// A message type's label, absorbed into the pattern description.
///
/// Distinct message types of one protocol should have distinct labels, so
/// that patterns differing only in which message is sent are separated.
pub trait Label {
    /// The label.
    const LABEL: u64;
}

impl<F> Label for F
where
    F: Field,
{
    const LABEL: u64 = 0x4645;
}

/// A value of type `T` with a protocol-specific label.
///
/// Protocols often send several messages of the same Rust type (three
/// commitments in a row); wrapping each in its own label keeps them
/// distinguishable in the pattern, so a transcript that swaps two of them
/// is a different protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Labeled<T, const LABEL: u64>(pub T);

impl<T, const L: u64> Label for Labeled<T, L> {
    const LABEL: u64 = L;
}

impl<F, T, const L: u64> Absorb<F> for Labeled<T, L>
where
    F: Field,
    T: Absorb<F>,
{
    fn absorb_into<Tr>(&self, transcript: &mut Tr)
    where
        Tr: Transcript<F>,
    {
        transcript.absorb(F::from_u64(L));
        self.0.absorb_into(transcript);
    }
}

/// The end of a protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct End;

/// The prover sends a value of type `M`, then the protocol continues as
/// `Next`.
#[derive(Debug)]
pub struct Message<M, Next>(PhantomData<(M, Next)>);

/// The verifier sends one challenge, then the protocol continues as `Next`.
#[derive(Debug)]
pub struct Challenge<Next>(PhantomData<Next>);

/// Both parties absorb a public value of type `M`, then the protocol
/// continues as `Next`. Public values are not part of the proof.
#[derive(Debug)]
pub struct Public<M, Next>(PhantomData<(M, Next)>);

/// A protocol pattern: a type-level list of messages and challenges.
pub trait Pattern {
    /// The number of prover messages.
    const MESSAGES: usize;

    /// The number of verifier challenges.
    const CHALLENGES: usize;

    /// The proof: the list of the prover's messages, in order. Challenges
    /// are not part of it, since the verifier recomputes them.
    type Proof: HList;

    /// Absorb a description of the pattern: one tag per step, and the label
    /// of every message.
    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>;
}

impl Pattern for End {
    const MESSAGES: usize = 0;
    const CHALLENGES: usize = 0;

    type Proof = HNil;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_END));
    }
}

impl<M, Next> Pattern for Message<M, Next>
where
    M: Label,
    Next: Pattern,
{
    const MESSAGES: usize = 1 + Next::MESSAGES;
    const CHALLENGES: usize = Next::CHALLENGES;

    type Proof = HCons<M, <Next as Pattern>::Proof>;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_MESSAGE));
        transcript.absorb(F::from_u64(M::LABEL));
        Next::describe(transcript);
    }
}

impl<M, Next> Pattern for Public<M, Next>
where
    M: Label,
    Next: Pattern,
{
    const MESSAGES: usize = Next::MESSAGES;
    const CHALLENGES: usize = Next::CHALLENGES;

    type Proof = <Next as Pattern>::Proof;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_PUBLIC));
        transcript.absorb(F::from_u64(M::LABEL));
        Next::describe(transcript);
    }
}

impl<Next> Pattern for Challenge<Next>
where
    Next: Pattern,
{
    const MESSAGES: usize = Next::MESSAGES;
    const CHALLENGES: usize = 1 + Next::CHALLENGES;

    type Proof = <Next as Pattern>::Proof;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_CHALLENGE));
        Next::describe(transcript);
    }
}

/// Sequential composition of patterns: type-level list concatenation.
pub trait Concat<Q> {
    /// The pattern `Self` followed by `Q`.
    type Output;
}

impl<Q> Concat<Q> for End {
    type Output = Q;
}

impl<M, Next, Q> Concat<Q> for Message<M, Next>
where
    Next: Concat<Q>,
{
    type Output = Message<M, <Next as Concat<Q>>::Output>;
}

impl<M, Next, Q> Concat<Q> for Public<M, Next>
where
    Next: Concat<Q>,
{
    type Output = Public<M, <Next as Concat<Q>>::Output>;
}

impl<Next, Q> Concat<Q> for Challenge<Next>
where
    Next: Concat<Q>,
{
    type Output = Challenge<<Next as Concat<Q>>::Output>;
}

/// The pattern `P` followed by `Q`.
pub type Then<P, Q> = <P as Concat<Q>>::Output;

/// A transcript `T` over `F` that still has to follow the pattern `P`.
#[derive(Debug)]
pub struct Tape<F, T, P> {
    transcript: T,
    marker: PhantomData<(F, P)>,
}

impl<F, T, P> Tape<F, T, P>
where
    F: Field,
    T: Transcript<F>,
    P: Pattern,
{
    /// Start a protocol following `P` on `transcript`, after absorbing the
    /// description of `P`.
    #[must_use]
    pub fn start(mut transcript: T) -> Self {
        P::describe(&mut transcript);
        Self {
            transcript,
            marker: PhantomData,
        }
    }

    const fn advance<Next>(transcript: T) -> Tape<F, T, Next> {
        Tape {
            transcript,
            marker: PhantomData,
        }
    }
}

impl<F, T, M, Next> Tape<F, T, Message<M, Next>>
where
    F: Field,
    T: Transcript<F>,
    M: Absorb<F> + Label,
    Next: Pattern,
{
    /// The prover writes `message` on the tape.
    #[must_use]
    pub fn send(mut self, message: &M) -> Tape<F, T, Next> {
        message.absorb_into(&mut self.transcript);
        Tape::<F, T, Message<M, Next>>::advance_to(self.transcript)
    }

    /// The verifier reads `message` from the tape. Identical to
    /// [`Tape::send`]; the two names keep protocol code readable on both
    /// sides.
    #[must_use]
    pub fn receive(self, message: &M) -> Tape<F, T, Next> {
        self.send(message)
    }

    const fn advance_to(transcript: T) -> Tape<F, T, Next> {
        Tape {
            transcript,
            marker: PhantomData,
        }
    }
}

impl<F, T, M, Next> Tape<F, T, Public<M, Next>>
where
    F: Field,
    T: Transcript<F>,
    M: Absorb<F> + Label,
    Next: Pattern,
{
    /// Absorb the public value `value`.
    #[must_use]
    pub fn public(mut self, value: &M) -> Tape<F, T, Next> {
        value.absorb_into(&mut self.transcript);
        Tape {
            transcript: self.transcript,
            marker: PhantomData,
        }
    }
}

impl<F, T, Next> Tape<F, T, Challenge<Next>>
where
    F: Field,
    T: Transcript<F>,
    Next: Pattern,
{
    /// The verifier writes a challenge on the tape.
    #[must_use]
    pub fn challenge(mut self) -> (F, Tape<F, T, Next>) {
        let challenge = self.transcript.squeeze();
        (
            challenge,
            Tape::<F, T, Challenge<Next>>::advance(self.transcript),
        )
    }
}

impl<F, T> Tape<F, T, End>
where
    F: Field,
    T: Transcript<F>,
{
    /// The protocol is over; recover the transcript, for instance to run a
    /// subsequent protocol on the same sponge.
    #[must_use]
    pub fn finish(self) -> T {
        self.transcript
    }
}

/// The prover's side of a protocol `Full`, currently at the suffix `Rest`,
/// having written the messages `Written` (newest first).
///
/// [`Prover::finish`] only exists once `Rest` is [`End`], and returns the
/// proof as a value of exactly `Full::Proof`: the type checker verifies that
/// the messages written are the messages the protocol declares.
#[derive(Debug)]
pub struct Prover<F, T, Full, Rest, Written> {
    tape: Tape<F, T, Rest>,
    written: Written,
    full: PhantomData<Full>,
}

impl<F, T, Full> Prover<F, T, Full, Full, HNil>
where
    F: Field,
    T: Transcript<F>,
    Full: Pattern,
{
    /// Start proving `Full` on `transcript`.
    #[must_use]
    pub fn start(transcript: T) -> Self {
        Self {
            tape: Tape::start(transcript),
            written: HNil,
            full: PhantomData,
        }
    }
}

impl<F, T, Full, M, Next, Written> Prover<F, T, Full, Message<M, Next>, Written>
where
    F: Field,
    T: Transcript<F>,
    M: Absorb<F> + Label,
    Next: Pattern,
{
    /// Absorb `message` and record it in the proof.
    #[must_use]
    pub fn send(self, message: M) -> Prover<F, T, Full, Next, HCons<M, Written>> {
        let tape = self.tape.send(&message);
        Prover {
            tape,
            written: HCons::new(message, self.written),
            full: PhantomData,
        }
    }
}

impl<F, T, Full, M, Next, Written> Prover<F, T, Full, Public<M, Next>, Written>
where
    F: Field,
    T: Transcript<F>,
    M: Absorb<F> + Label,
    Next: Pattern,
{
    /// Absorb the public value `value`; it is not recorded in the proof.
    #[must_use]
    pub fn public(self, value: &M) -> Prover<F, T, Full, Next, Written> {
        Prover {
            tape: self.tape.public(value),
            written: self.written,
            full: PhantomData,
        }
    }
}

impl<F, T, Full, Next, Written> Prover<F, T, Full, Challenge<Next>, Written>
where
    F: Field,
    T: Transcript<F>,
    Next: Pattern,
{
    /// Squeeze the next challenge.
    #[must_use]
    pub fn challenge(self) -> (F, Prover<F, T, Full, Next, Written>) {
        let (challenge, tape) = self.tape.challenge();
        let prover = Prover {
            tape,
            written: self.written,
            full: PhantomData,
        };
        (challenge, prover)
    }
}

impl<F, T, Full, Written> Prover<F, T, Full, End, Written>
where
    F: Field,
    T: Transcript<F>,
    Full: Pattern,
    Written: Reverse<HNil, Output = <Full as Pattern>::Proof>,
{
    /// The transcript and the proof, in protocol order.
    #[must_use]
    pub fn finish(self) -> (T, <Full as Pattern>::Proof) {
        (self.tape.finish(), self.written.reverse_onto(HNil))
    }
}

/// The verifier's side of a protocol, currently at the suffix `Rest`, with
/// the messages of `Rest` still to read.
///
/// The unread part of the proof is a value of type `Rest::Proof`, so
/// [`Verifier::read`] returns a message of exactly the declared type, and
/// [`Verifier::finish`] (only at [`End`]) certifies that the whole proof was
/// consumed: a proof cannot carry messages the verifier ignores.
pub struct Verifier<F, T, Rest>
where
    Rest: Pattern,
{
    tape: Tape<F, T, Rest>,
    unread: <Rest as Pattern>::Proof,
}

impl<F, T, Full> Verifier<F, T, Full>
where
    F: Field,
    T: Transcript<F>,
    Full: Pattern,
{
    /// Start verifying `proof` against `Full` on `transcript`.
    #[must_use]
    pub fn start(transcript: T, proof: <Full as Pattern>::Proof) -> Self {
        Self {
            tape: Tape::start(transcript),
            unread: proof,
        }
    }
}

impl<F, T, M, Next> Verifier<F, T, Message<M, Next>>
where
    F: Field,
    T: Transcript<F>,
    M: Absorb<F> + Label,
    Next: Pattern,
{
    /// Read and absorb the next message.
    #[must_use]
    pub fn read(self) -> (M, Verifier<F, T, Next>) {
        let HCons { head, tail } = self.unread;
        let tape = self.tape.receive(&head);
        (head, Verifier { tape, unread: tail })
    }
}

impl<F, T, M, Next> Verifier<F, T, Public<M, Next>>
where
    F: Field,
    T: Transcript<F>,
    M: Absorb<F> + Label,
    Next: Pattern,
{
    /// Absorb the public value `value`, which the verifier supplies.
    #[must_use]
    pub fn public(self, value: &M) -> Verifier<F, T, Next> {
        Verifier {
            tape: self.tape.public(value),
            unread: self.unread,
        }
    }
}

impl<F, T, Next> Verifier<F, T, Challenge<Next>>
where
    F: Field,
    T: Transcript<F>,
    Next: Pattern,
{
    /// Squeeze the next challenge.
    #[must_use]
    pub fn challenge(self) -> (F, Verifier<F, T, Next>) {
        let (challenge, tape) = self.tape.challenge();
        (
            challenge,
            Verifier {
                tape,
                unread: self.unread,
            },
        )
    }
}

impl<F, T> Verifier<F, T, End>
where
    F: Field,
    T: Transcript<F>,
{
    /// The protocol is over and the proof fully read.
    #[must_use]
    pub fn finish(self) -> T {
        self.tape.finish()
    }
}

/// A non-interactive protocol, identified by its transcript.
///
/// The transcript pattern is the single source of truth shared by the two
/// parties: [`prover_tape`] and [`verifier_tape`] both start from
/// `Self::Transcript`, and the proof is a [`ProofOf<Self>`]. A prover and a
/// verifier written against the same `Protocol` therefore agree, by
/// construction, on the order of the tape, on the statement, and on the type
/// of every message; if they did not, they would not be running the same
/// SNARK, and they would not type-check against each other.
pub trait Protocol {
    /// The transcript both parties follow.
    type Transcript: Pattern;
}

/// The proof of the protocol `P`: the list of its prover messages.
pub type ProofOf<P> = <<P as Protocol>::Transcript as Pattern>::Proof;

/// The prover's tape for the protocol `P`.
#[must_use]
pub fn prover_tape<F, T, P>(
    transcript: T,
) -> Prover<F, T, <P as Protocol>::Transcript, <P as Protocol>::Transcript, HNil>
where
    F: Field,
    T: Transcript<F>,
    P: Protocol,
{
    Prover::start(transcript)
}

/// The verifier's tape for the protocol `P`, reading `proof`.
#[must_use]
pub fn verifier_tape<F, T, P>(
    transcript: T,
    proof: ProofOf<P>,
) -> Verifier<F, T, <P as Protocol>::Transcript>
where
    F: Field,
    T: Transcript<F>,
    P: Protocol,
{
    Verifier::start(transcript, proof)
}

/// How a value of type `Self` is written to the transcript `T`: the
/// absorptions of a [`Statement`] or a [`Send`] step.
pub trait Write<T> {
    /// Write `self` to `transcript`.
    fn write(&self, transcript: &mut T);
}

/// A kind of challenge drawn from the transcript `T`: the squeezes of a
/// [`Squeeze`] or a [`Peek`] step.
pub trait Draw<T> {
    /// The challenge.
    type Value;
    /// Draw one from `transcript`.
    fn draw(transcript: &mut T) -> Self::Value;
}

/// Both parties write a public `M` (not part of the proof), then `Next`.
#[derive(Debug)]
pub struct Statement<M, Next>(PhantomData<(M, Next)>);

/// The prover sends `M`, written by [`Write`], then `Next`.
#[derive(Debug)]
pub struct Send<M, Next>(PhantomData<(M, Next)>);

/// One challenge of the kind `K`, drawn by [`Draw`], then `Next`.
#[derive(Debug)]
pub struct Squeeze<K, Next>(PhantomData<(K, Next)>);

/// One challenge of the kind `K` drawn from a copy of the transcript, which
/// is left as it was, then `Next`.
#[derive(Debug)]
pub struct Peek<K, Next>(PhantomData<(K, Next)>);

/// `Body` (a pattern ending in [`End`]) any number of times, then `Next`.
#[derive(Debug)]
pub struct Rounds<Body, Next>(PhantomData<(Body, Next)>);

/// `A` or `B`, as the statement decides.
#[derive(Debug)]
pub struct Branch<A, B>(PhantomData<(A, B)>);

/// The proof of a [`Branch`]: the proof of the branch taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Either<A, B> {
    /// The first branch.
    Left(A),
    /// The second branch.
    Right(B),
}

impl<M, Next> Pattern for Statement<M, Next>
where
    M: Label,
    Next: Pattern,
{
    const MESSAGES: usize = Next::MESSAGES;
    const CHALLENGES: usize = Next::CHALLENGES;
    type Proof = <Next as Pattern>::Proof;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_PUBLIC));
        transcript.absorb(F::from_u64(M::LABEL));
        Next::describe(transcript);
    }
}

impl<M, Next> Pattern for Send<M, Next>
where
    M: Label,
    Next: Pattern,
{
    const MESSAGES: usize = 1 + Next::MESSAGES;
    const CHALLENGES: usize = Next::CHALLENGES;
    type Proof = HCons<M, <Next as Pattern>::Proof>;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_MESSAGE));
        transcript.absorb(F::from_u64(M::LABEL));
        Next::describe(transcript);
    }
}

impl<K, Next> Pattern for Squeeze<K, Next>
where
    K: Label,
    Next: Pattern,
{
    const MESSAGES: usize = Next::MESSAGES;
    const CHALLENGES: usize = 1 + Next::CHALLENGES;
    type Proof = <Next as Pattern>::Proof;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_SQUEEZE));
        transcript.absorb(F::from_u64(K::LABEL));
        Next::describe(transcript);
    }
}

impl<K, Next> Pattern for Peek<K, Next>
where
    K: Label,
    Next: Pattern,
{
    const MESSAGES: usize = Next::MESSAGES;
    const CHALLENGES: usize = 1 + Next::CHALLENGES;
    type Proof = <Next as Pattern>::Proof;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_PEEK));
        transcript.absorb(F::from_u64(K::LABEL));
        Next::describe(transcript);
    }
}

/// The counts of a [`Rounds`] are those of `Next`: how many rounds run is
/// agreed at run time, so its messages are a vector of round proofs.
impl<Body, Next> Pattern for Rounds<Body, Next>
where
    Body: Pattern,
    Next: Pattern,
{
    const MESSAGES: usize = Next::MESSAGES;
    const CHALLENGES: usize = Next::CHALLENGES;
    type Proof = HCons<alloc::vec::Vec<<Body as Pattern>::Proof>, <Next as Pattern>::Proof>;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_ROUNDS));
        Body::describe(transcript);
        Next::describe(transcript);
    }
}

/// The counts of a [`Branch`] are the smaller of its branches'.
impl<A, B> Pattern for Branch<A, B>
where
    A: Pattern,
    B: Pattern,
{
    const MESSAGES: usize = if A::MESSAGES < B::MESSAGES {
        A::MESSAGES
    } else {
        B::MESSAGES
    };
    const CHALLENGES: usize = if A::CHALLENGES < B::CHALLENGES {
        A::CHALLENGES
    } else {
        B::CHALLENGES
    };
    type Proof = HCons<Either<<A as Pattern>::Proof, <B as Pattern>::Proof>, HNil>;

    fn describe<F, T>(transcript: &mut T)
    where
        F: Field,
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::PATTERN_BRANCH));
        A::describe(transcript);
        B::describe(transcript);
    }
}

impl<M, Next, Q> Concat<Q> for Statement<M, Next>
where
    Next: Concat<Q>,
{
    type Output = Statement<M, <Next as Concat<Q>>::Output>;
}

impl<M, Next, Q> Concat<Q> for Send<M, Next>
where
    Next: Concat<Q>,
{
    type Output = Send<M, <Next as Concat<Q>>::Output>;
}

impl<K, Next, Q> Concat<Q> for Squeeze<K, Next>
where
    Next: Concat<Q>,
{
    type Output = Squeeze<K, <Next as Concat<Q>>::Output>;
}

impl<K, Next, Q> Concat<Q> for Peek<K, Next>
where
    Next: Concat<Q>,
{
    type Output = Peek<K, <Next as Concat<Q>>::Output>;
}

impl<Body, Next, Q> Concat<Q> for Rounds<Body, Next>
where
    Next: Concat<Q>,
{
    type Output = Rounds<Body, <Next as Concat<Q>>::Output>;
}

impl<A, B, Q> Concat<Q> for Branch<A, B>
where
    A: Concat<Q>,
    B: Concat<Q>,
{
    type Output = Branch<<A as Concat<Q>>::Output, <B as Concat<Q>>::Output>;
}

impl<F, T, P> Tape<F, T, P> {
    /// Start a protocol following `P` on `transcript` without absorbing
    /// its description: for a protocol whose transcript is fixed elsewhere
    /// and must be reproduced bit for bit, which then has no domain
    /// separation of its own pattern beyond what its steps absorb.
    #[must_use]
    pub const fn undescribed(transcript: T) -> Self {
        Self {
            transcript,
            marker: PhantomData,
        }
    }

    /// The tape at another point of a pattern, the transcript moved.
    fn moved<Q>(self) -> Tape<F, T, Q> {
        Tape {
            transcript: self.transcript,
            marker: PhantomData,
        }
    }
}

impl<F, T, M, Next> Tape<F, T, Statement<M, Next>>
where
    M: Write<T>,
{
    /// Write the public `value`.
    #[must_use]
    pub fn statement(mut self, value: &M) -> Tape<F, T, Next> {
        value.write(&mut self.transcript);
        self.moved()
    }
}

impl<F, T, M, Next> Tape<F, T, Send<M, Next>>
where
    M: Write<T>,
{
    /// The prover sends `message`.
    #[must_use]
    pub fn send(mut self, message: &M) -> Tape<F, T, Next> {
        message.write(&mut self.transcript);
        self.moved()
    }

    /// The verifier reads `message`: the same absorption as [`Self::send`].
    #[must_use]
    pub fn receive(self, message: &M) -> Tape<F, T, Next> {
        self.send(message)
    }
}

impl<F, T, K, Next> Tape<F, T, Squeeze<K, Next>>
where
    K: Draw<T>,
{
    /// Draw the challenge.
    #[must_use]
    pub fn squeeze(mut self) -> (K::Value, Tape<F, T, Next>) {
        let value = K::draw(&mut self.transcript);
        (value, self.moved())
    }
}

impl<F, T, K, Next> Tape<F, T, Peek<K, Next>>
where
    T: Clone,
    K: Draw<T>,
{
    /// Draw the challenge from a copy, leaving the transcript as it was.
    #[must_use]
    pub fn peek(self) -> (K::Value, Tape<F, T, Next>) {
        let value = K::draw(&mut self.transcript.clone());
        (value, self.moved())
    }
}

impl<F, T, Body, Next> Tape<F, T, Rounds<Body, Next>>
where
    Body: Concat<Rounds<Body, Next>>,
{
    /// One more round: the tape at `Body`, then back at these rounds.
    #[must_use]
    pub fn round(self) -> Tape<F, T, Then<Body, Rounds<Body, Next>>> {
        self.moved()
    }

    /// No more round.
    #[must_use]
    pub fn done(self) -> Tape<F, T, Next> {
        self.moved()
    }
}

impl<F, T, A, B> Tape<F, T, Branch<A, B>> {
    /// Take the first branch.
    #[must_use]
    pub fn left(self) -> Tape<F, T, A> {
        self.moved()
    }

    /// Take the second branch.
    #[must_use]
    pub fn right(self) -> Tape<F, T, B> {
        self.moved()
    }
}

impl<F, T> Tape<F, T, End> {
    /// The protocol is over: the transcript, whatever its kind.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.transcript
    }
}
