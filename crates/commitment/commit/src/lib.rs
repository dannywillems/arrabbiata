//! Commitment schemes: the abstractions.
//!
//! Two traits, for two different needs:
//!
//! - [`HomomorphicCommitment`]: vectors over `F`, with commitments forming an
//!   `F`-module and committing a module morphism. Folding needs exactly this:
//!
//!   ```text
//!   commit(a + s * b) = add(commit(a), scale(commit(b), s)),
//!   ```
//!
//!   so the verifier folds commitments as the prover folds vectors.
//! - [`PolynomialCommitment`]: polynomials of a [`PolynomialFamily`]
//!   (univariate or multilinear), with evaluation proofs. SNARKs compiled
//!   from polynomial IOPs need this.
//!
//! The schemes live in sibling crates (`arrabbiata-ipa`, `arrabbiata-kzg`).
//! Security assumptions (binding, hiding) are properties of a concrete
//! scheme and are stated by each implementation, not by the traits.

#![no_std]

extern crate alloc;

pub mod pcs;

pub use pcs::{Multilinear, PolynomialCommitment, PolynomialFamily, Univariate};

use core::fmt::Debug;

use arrabbiata_field::Field;
use arrabbiata_transcript::Absorb;

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
/// use arrabbiata_commit::HomomorphicCommitment;
/// use arrabbiata_field::Field;
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// // Pedersen vector commitments on Pallas (the commitment of IPA).
/// let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// let (a, b, s) = ([Fq::from_u64(1), Fq::from_u64(2)], [Fq::from_u64(3), Fq::from_u64(4)], Fq::from_u64(5));
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
    /// use arrabbiata_commit::HomomorphicCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// // Pedersen vector commitments on Pallas (the commitment of IPA).
    /// let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// assert_eq!(ck.commit(&[Fq::ONE]), ck.commit(&[Fq::ONE, Fq::ZERO]));
    /// ```
    fn commit(&self, values: &[F]) -> Self::Commitment;

    /// The sum of two commitments.
    ///
    /// ```
    /// use arrabbiata_commit::HomomorphicCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// // Pedersen vector commitments on Pallas (the commitment of IPA).
    /// let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// let c = ck.add(&ck.commit(&[Fq::ONE]), &ck.commit(&[Fq::ONE]));
    /// assert_eq!(c, ck.commit(&[Fq::from_u64(2)]));
    /// ```
    fn add(&self, a: &Self::Commitment, b: &Self::Commitment) -> Self::Commitment;

    /// A commitment scaled by `s`.
    ///
    /// ```
    /// use arrabbiata_commit::HomomorphicCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// // Pedersen vector commitments on Pallas (the commitment of IPA).
    /// let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// let c = ck.scale(&ck.commit(&[Fq::from_u64(3)]), Fq::from_u64(2));
    /// assert_eq!(c, ck.commit(&[Fq::from_u64(6)]));
    /// ```
    fn scale(&self, a: &Self::Commitment, s: F) -> Self::Commitment;
}
