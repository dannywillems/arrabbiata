//! Homomorphic vector commitments.
//!
//! Folding only needs one algebraic property of the commitment scheme: that
//! committing is a morphism of `F`-modules,
//!
//! ```text
//! commit(a + s * b) = add(commit(a), scale(commit(b), s)),
//! ```
//!
//! so the verifier can fold commitments exactly as the prover folds the
//! committed vectors. Pedersen and the IPA commitment over a curve of order
//! `|F|` satisfy it; so does [`IdentityCommitment`], the trivial test scheme
//! that "commits" to a vector by publishing it.
//!
//! Security assumptions (binding, hiding) are properties of a concrete
//! scheme and are stated by each implementation, not by the trait.

#![no_std]

extern crate alloc;

pub mod pcs;

pub use pcs::{
    Multilinear, NoProof, PolynomialCommitment, PolynomialFamily, RevealCommitment, Revealed,
    Univariate,
};

use alloc::vec::Vec;
use core::fmt::Debug;

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, Transcript, absorb_slice};

/// A commitment scheme for vectors over `F` whose commitments form an
/// `F`-module and for which `commit` is a module morphism.
///
/// Laws, for vectors `a`, `b` of equal length and scalars `s`, `t`:
///
/// - `commit(a + b) = add(commit(a), commit(b))`;
/// - `commit(s * a) = scale(commit(a), s)`;
/// - `add` is associative and commutative, `scale` is an action:
///   `scale(scale(c, s), t) = scale(c, s * t)`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
/// let ck = IdentityCommitment;
/// let (a, b, s) = ([F::new(1), F::new(2)], [F::new(3), F::new(4)], F::new(5));
/// let folded = [a[0] + s * b[0], a[1] + s * b[1]];
/// // The verifier folds commitments as the prover folds vectors.
/// assert_eq!(ck.commit(&folded), ck.add(&ck.commit(&a), &ck.scale(&ck.commit(&b), s)));
/// ```
pub trait HomomorphicCommitment<F>
where
    F: Field,
{
    /// A commitment.
    type Commitment: Clone + Debug + PartialEq + Absorb<F>;

    /// Commit to `values`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
    /// let c = IdentityCommitment.commit(&[F::ONE]);
    /// assert_eq!(c.values(), &[F::ONE]);
    /// ```
    fn commit(&self, values: &[F]) -> Self::Commitment;

    /// The sum of two commitments.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
    /// let ck = IdentityCommitment;
    /// let c = ck.add(&ck.commit(&[F::ONE]), &ck.commit(&[F::ONE]));
    /// assert_eq!(c, ck.commit(&[F::new(2)]));
    /// ```
    fn add(&self, a: &Self::Commitment, b: &Self::Commitment) -> Self::Commitment;

    /// A commitment scaled by `s`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
    /// let ck = IdentityCommitment;
    /// assert_eq!(ck.scale(&ck.commit(&[F::new(3)]), F::new(2)), ck.commit(&[F::new(6)]));
    /// ```
    fn scale(&self, a: &Self::Commitment, s: F) -> Self::Commitment;
}

/// The identity "commitment": the commitment to a vector is the vector.
///
/// Perfectly binding, not hiding, not succinct. It exists so that the
/// folding scheme can be tested, and extracted to Lean, against the
/// algebraic interface alone. Do not use it in a deployed protocol.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
/// let c = IdentityCommitment.commit(&[F::new(7), F::ZERO]);
/// assert_eq!(c.values(), &[F::new(7)]);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityCommitment;

/// A vector published in full, as committed by [`IdentityCommitment`].
///
/// Vectors are compared as finitely supported sequences: trailing zeros are
/// dropped, so the module operations are total on vectors of any length.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
/// let c = IdentityCommitment.commit(&[F::ZERO, F::ONE, F::ZERO]);
/// assert_eq!(c.values(), &[F::ZERO, F::ONE]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opened<F>(Vec<F>);

impl<F> Opened<F>
where
    F: Field,
{
    fn normalized(mut values: Vec<F>) -> Self {
        while let Some(last) = values.last() {
            if last.is_zero() {
                values.pop();
            } else {
                break;
            }
        }
        Self(values)
    }

    /// The committed vector, without trailing zeros.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
    /// assert!(IdentityCommitment.commit(&[F::ZERO]).values().is_empty());
    /// ```
    #[must_use]
    pub fn values(&self) -> &[F] {
        &self.0
    }
}

impl<F> Absorb<F> for Opened<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.0);
    }
}

impl<F> HomomorphicCommitment<F> for IdentityCommitment
where
    F: Field,
{
    type Commitment = Opened<F>;

    fn commit(&self, values: &[F]) -> Opened<F> {
        Opened::normalized(values.to_vec())
    }

    fn add(&self, a: &Opened<F>, b: &Opened<F>) -> Opened<F> {
        let len = if a.0.len() >= b.0.len() {
            a.0.len()
        } else {
            b.0.len()
        };
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while i < len {
            let x = a.0.get(i).copied().unwrap_or(F::ZERO);
            let y = b.0.get(i).copied().unwrap_or(F::ZERO);
            out.push(x + y);
            i += 1;
        }
        Opened::normalized(out)
    }

    fn scale(&self, a: &Opened<F>, s: F) -> Opened<F> {
        let mut out = Vec::with_capacity(a.0.len());
        let mut i = 0;
        while i < a.0.len() {
            out.push(a.0[i] * s);
            i += 1;
        }
        Opened::normalized(out)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use arrabbiata_field::Goldilocks as F;
    use proptest::prelude::*;
    use std::vec::Vec;

    use super::*;

    fn arb_vec(len: usize) -> impl Strategy<Value = Vec<F>> {
        prop::collection::vec(
            prop_oneof![Just(F::ZERO), any::<u64>().prop_map(F::new)],
            len,
        )
    }

    proptest! {
        /// commit is a module morphism: commit(a + s b) = commit(a) + s commit(b).
        #[test]
        fn commit_is_a_morphism(
            (a, b) in (0usize..6).prop_flat_map(|n| (arb_vec(n), arb_vec(n))),
            s in any::<u64>().prop_map(F::new),
        ) {
            let ck = IdentityCommitment;
            let folded: Vec<F> = a.iter().zip(&b).map(|(x, y)| *x + s * *y).collect();
            prop_assert_eq!(
                ck.commit(&folded),
                ck.add(&ck.commit(&a), &ck.scale(&ck.commit(&b), s))
            );
        }

        /// scale is an action of the multiplicative monoid.
        #[test]
        fn scale_is_an_action(
            a in arb_vec(4),
            s in any::<u64>().prop_map(F::new),
            t in any::<u64>().prop_map(F::new),
        ) {
            let ck = IdentityCommitment;
            let c = ck.commit(&a);
            prop_assert_eq!(ck.scale(&ck.scale(&c, s), t), ck.scale(&c, s * t));
        }
    }
}
