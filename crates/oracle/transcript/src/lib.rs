//! Fiat-Shamir transcripts, untyped and typed.
//!
//! Two layers:
//!
//! - [`Transcript`] is the untyped interface: absorb a field element,
//!   squeeze a challenge. [`Duplex`] implements it as a duplex sponge over
//!   any [`Permutation`]. This layer is small on purpose; it is what the
//!   Lean side models as a random oracle.
//! - [`typed`] puts the *protocol* in the type. A transcript in state
//!   `Message<M, Next>` can only absorb an `M`, after which it is in state
//!   `Next`; in state `Challenge<Next>` it can only squeeze. Steps consume
//!   the state, so an old state cannot be reused. A verifier that squeezes
//!   before absorbing the statement does not compile, which rules out the
//!   weak Fiat-Shamir (Frozen Heart) class of bugs by construction.
//!
//! The design follows the "typed Fiat-Shamir" experiment of
//! o1-labs/proof-systems#2094: the transcript is a shared tape, a
//! heterogeneous list of prover messages and verifier challenges whose shape
//! is known at the type level, and the permutation is a parameter.
//!
//! Protocols instantiate the sponge with Poseidon (`arrabbiata-poseidon`).

#![no_std]

pub mod hlist;
pub mod typed;

use arrabbiata_field::Field;

/// A Fiat-Shamir transcript over the field `F`.
pub trait Transcript<F>
where
    F: Field,
{
    /// Absorb one field element.
    fn absorb(&mut self, value: F);

    /// Squeeze one challenge. The challenge depends on everything absorbed
    /// and squeezed so far.
    fn squeeze(&mut self) -> F;
}

/// A value that can be absorbed into a transcript over `F`.
///
/// Implementations must be injective on the values of one type: two
/// different values must absorb different sequences. Prefixing variable
/// length data with its length is the usual way to guarantee it.
pub trait Absorb<F>
where
    F: Field,
{
    /// Absorb `self` into `transcript`.
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>;
}

impl<F> Absorb<F> for F
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(*self);
    }
}

/// Absorb a length, so that variable-length sequences absorb injectively.
pub fn absorb_length<F, T>(transcript: &mut T, length: usize)
where
    F: Field,
    T: Transcript<F>,
{
    transcript.absorb(F::from_u64(length as u64));
}

/// Absorb a slice of absorbable values, prefixed with its length.
pub fn absorb_slice<F, A, T>(transcript: &mut T, values: &[A])
where
    F: Field,
    A: Absorb<F>,
    T: Transcript<F>,
{
    absorb_length(transcript, values.len());
    let mut i = 0;
    while i < values.len() {
        values[i].absorb_into(transcript);
        i += 1;
    }
}

/// Domain-separation tags.
pub mod tag {
    /// A constraint-system description.
    pub const CONSTRAINT_SYSTEM: u64 = 0x4353;
    /// A relaxed instance.
    pub const RELAXED_INSTANCE: u64 = 0x5249;
    /// A folding proof.
    pub const FOLDING_PROOF: u64 = 0x4650;
    /// The combiner challenge of a fresh instance.
    pub const COMBINER: u64 = 0x434f;
    /// A prover message in a protocol pattern.
    pub const PATTERN_MESSAGE: u64 = 0x504d;
    /// A verifier challenge in a protocol pattern.
    pub const PATTERN_CHALLENGE: u64 = 0x5043;
    /// The end of a protocol pattern.
    pub const PATTERN_END: u64 = 0x5045;
    /// A public input in a protocol pattern.
    pub const PATTERN_PUBLIC: u64 = 0x5050;
}

/// A permutation of `F^WIDTH`, the core of a sponge.
///
/// Poseidon, Poseidon2, Rescue, Griffin or Anemoi are instances; the sponge
/// is generic over the choice.
pub trait Permutation<F, const WIDTH: usize>
where
    F: Field,
{
    /// Apply the permutation in place.
    fn permute(&self, state: &mut [F; WIDTH]);
}

/// A duplex sponge over the permutation `P`, with a state of `WIDTH`
/// elements of which the first `RATE` are the rate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Duplex<F, P, const WIDTH: usize, const RATE: usize> {
    permutation: P,
    state: [F; WIDTH],
    position: usize,
}

impl<F, P, const WIDTH: usize, const RATE: usize> Duplex<F, P, WIDTH, RATE>
where
    F: Field,
    P: Permutation<F, WIDTH>,
{
    /// A fresh sponge, separated by `domain` in the capacity.
    #[must_use]
    pub fn new(permutation: P, domain: u64) -> Self {
        const {
            assert!(RATE > 0, "the rate must be non-zero");
            assert!(RATE < WIDTH, "the capacity must be non-zero");
        };
        let mut state = [F::ZERO; WIDTH];
        state[WIDTH - 1] = F::from_u64(domain);
        Self {
            permutation,
            state,
            position: 0,
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> Transcript<F> for Duplex<F, P, WIDTH, RATE>
where
    F: Field,
    P: Permutation<F, WIDTH>,
{
    fn absorb(&mut self, value: F) {
        if self.position == RATE {
            self.permutation.permute(&mut self.state);
            self.position = 0;
        }
        self.state[self.position] = self.state[self.position] + value;
        self.position += 1;
    }

    fn squeeze(&mut self) -> F {
        self.permutation.permute(&mut self.state);
        self.position = 0;
        self.state[0]
    }
}
