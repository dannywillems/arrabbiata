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
//! [`ToyPermutation`] is a deterministic stand-in for tests. It is NOT
//! cryptographically secure.

#![no_std]

pub mod hlist;
pub mod typed;

use arrabbiata_field::Field;

/// A Fiat-Shamir transcript over the field `F`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Transcript, toy_sponge};
/// fn challenge_after<T>(t: &mut T, x: F) -> F
/// where
///     T: Transcript<F>,
/// {
///     t.absorb(x);
///     t.squeeze()
/// }
/// let c = challenge_after(&mut toy_sponge(0), F::new(1));
/// assert_eq!(c, challenge_after(&mut toy_sponge(0), F::new(1)));
/// ```
pub trait Transcript<F>
where
    F: Field,
{
    /// Absorb one field element.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_transcript::{Transcript, toy_sponge};
    /// let mut t = toy_sponge::<F>(0);
    /// t.absorb(F::new(1));
    /// ```
    fn absorb(&mut self, value: F);

    /// Squeeze one challenge. The challenge depends on everything absorbed
    /// and squeezed so far.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_transcript::{Transcript, toy_sponge};
    /// let mut t = toy_sponge::<F>(0);
    /// let a = t.squeeze();
    /// let b = t.squeeze();
    /// assert_ne!(a, b);
    /// ```
    fn squeeze(&mut self) -> F;
}

/// A value that can be absorbed into a transcript over `F`.
///
/// Implementations must be injective on the values of one type: two
/// different values must absorb different sequences. Prefixing variable
/// length data with its length is the usual way to guarantee it.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Absorb, Transcript, toy_sponge};
/// struct Point { x: F, y: F }
/// impl Absorb<F> for Point {
///     fn absorb_into<T>(&self, t: &mut T)
///     where
///         T: Transcript<F>,
///     {
///         t.absorb(self.x);
///         t.absorb(self.y);
///     }
/// }
/// let mut t = toy_sponge::<F>(0);
/// Point { x: F::ONE, y: F::ZERO }.absorb_into(&mut t);
/// ```
pub trait Absorb<F>
where
    F: Field,
{
    /// Absorb `self` into `transcript`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_transcript::{Absorb, Transcript, toy_sponge};
    /// let mut t = toy_sponge::<F>(0);
    /// F::new(5).absorb_into(&mut t);
    /// let _ = t.squeeze();
    /// ```
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
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Transcript, absorb_length, toy_sponge};
/// let mut t = toy_sponge::<F>(0);
/// absorb_length(&mut t, 3);
/// ```
pub fn absorb_length<F, T>(transcript: &mut T, length: usize)
where
    F: Field,
    T: Transcript<F>,
{
    transcript.absorb(F::from_u64(length as u64));
}

/// Absorb a slice of absorbable values, prefixed with its length.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Transcript, absorb_slice, toy_sponge};
/// let mut t = toy_sponge::<F>(0);
/// absorb_slice(&mut t, &[F::ONE, F::ONE]);
/// let _ = t.squeeze();
/// ```
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
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Duplex, Permutation, Transcript};
/// /// Rotate the state: a (useless) permutation of F^2.
/// struct Swap;
/// impl Permutation<F, 2> for Swap {
///     fn permute(&self, state: &mut [F; 2]) {
///         state.swap(0, 1);
///     }
/// }
/// let mut sponge = Duplex::<F, Swap, 2, 1>::new(Swap, 0);
/// sponge.absorb(F::new(3));
/// assert_eq!(sponge.squeeze(), F::ZERO);
/// ```
pub trait Permutation<F, const WIDTH: usize>
where
    F: Field,
{
    /// Apply the permutation in place.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_transcript::{Permutation, ToyPermutation};
    /// let mut state = [F::ZERO; 3];
    /// ToyPermutation.permute(&mut state);
    /// assert_ne!(state, [F::ZERO; 3]);
    /// ```
    fn permute(&self, state: &mut [F; WIDTH]);
}

/// A duplex sponge over the permutation `P`, with a state of `WIDTH`
/// elements of which the first `RATE` are the rate.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Duplex, ToyPermutation, Transcript};
/// let mut sponge = Duplex::<F, _, 3, 2>::new(ToyPermutation, 42);
/// sponge.absorb(F::ONE);
/// let _challenge = sponge.squeeze();
/// ```
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
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_transcript::{Duplex, ToyPermutation, Transcript};
    /// let mut a = Duplex::<F, _, 3, 2>::new(ToyPermutation, 1);
    /// let mut b = Duplex::<F, _, 3, 2>::new(ToyPermutation, 2);
    /// assert_ne!(a.squeeze(), b.squeeze());
    /// ```
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

/// A toy permutation of `F^3` for tests. NOT cryptographically secure, and
/// not even a permutation unless `gcd(7, p - 1) = 1`.
///
/// Eight rounds of: add the round index plus one, raise to the 7th power,
/// mix with the matrix `[[2,1,1],[1,2,1],[1,1,2]]`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Permutation, ToyPermutation};
/// let mut s = [F::ONE, F::new(2), F::new(3)];
/// ToyPermutation.permute(&mut s);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToyPermutation;

impl<F> Permutation<F, 3> for ToyPermutation
where
    F: Field,
{
    fn permute(&self, state: &mut [F; 3]) {
        let mut round: u64 = 0;
        while round < 8 {
            let c = F::from_u64(round + 1);
            let mut i = 0;
            while i < 3 {
                state[i] = (state[i] + c).pow(7);
                i += 1;
            }
            let sum = state[0] + state[1] + state[2];
            let mut j = 0;
            while j < 3 {
                state[j] = state[j] + sum;
                j += 1;
            }
            round += 1;
        }
    }
}

/// The toy test transcript: a rate-2 duplex over [`ToyPermutation`].
/// NOT cryptographically secure.
pub type ToySponge<F> = Duplex<F, ToyPermutation, 3, 2>;

/// A fresh [`ToySponge`] separated by `domain`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_transcript::{Transcript, toy_sponge};
/// let mut t = toy_sponge::<F>(7);
/// let _ = t.squeeze();
/// ```
#[must_use]
pub fn toy_sponge<F>(domain: u64) -> ToySponge<F>
where
    F: Field,
{
    Duplex::new(ToyPermutation, domain)
}

#[cfg(test)]
mod tests {
    use arrabbiata_field::Goldilocks;

    use super::*;

    fn challenge(domain: u64, values: &[u64]) -> Goldilocks {
        let mut t = toy_sponge::<Goldilocks>(domain);
        for v in values {
            t.absorb(Goldilocks::new(*v));
        }
        t.squeeze()
    }

    #[test]
    fn deterministic() {
        assert_eq!(challenge(1, &[5]), challenge(1, &[5]));
    }

    #[test]
    fn sensitive_to_input_domain_and_order() {
        let base = challenge(1, &[1, 2, 3]);
        assert_ne!(base, challenge(2, &[1, 2, 3]));
        assert_ne!(base, challenge(1, &[1, 2, 4]));
        assert_ne!(base, challenge(1, &[2, 1, 3]));
        assert_ne!(base, challenge(1, &[1, 2]));
    }

    #[test]
    fn slices_absorb_with_length_prefix() {
        let mut a = toy_sponge::<Goldilocks>(0);
        absorb_slice(&mut a, &[Goldilocks::new(1)]);
        absorb_slice(&mut a, &[Goldilocks::new(2)]);
        let mut b = toy_sponge::<Goldilocks>(0);
        absorb_slice(&mut b, &[Goldilocks::new(1), Goldilocks::new(2)]);
        assert_ne!(a.squeeze(), b.squeeze());
    }
}
