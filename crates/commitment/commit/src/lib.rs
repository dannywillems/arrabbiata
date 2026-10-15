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
//! [`BatchOpening`] opens many polynomials at one point, as protocols need.
//! Every scheme has one ([`Unbatched`], a proof per polynomial); a
//! [`LinearCommitment`], whose commitments combine linearly, has one with
//! a single proof ([`open_combined`], [`verify_combined`]).
//!
//! The schemes live in sibling crates (`arrabbiata-ipa`, `arrabbiata-kzg`).
//! Security assumptions (binding, hiding) are properties of a concrete
//! scheme and are stated by each implementation, not by the traits.

#![no_std]

extern crate alloc;

pub mod batch;
pub mod pcs;

pub use batch::{
    BatchOpening, LinearCommitment, Point, Proofs, Unbatched, combine_polynomials, combine_values,
    open_combined, verify_combined,
};
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
pub trait HomomorphicCommitment<F>
where
    F: Field,
{
    /// A commitment.
    type Commitment: Clone + Debug + PartialEq + Absorb<F>;

    /// Commit to `values`.
    fn commit(&self, values: &[F]) -> Self::Commitment;

    /// The sum of two commitments.
    fn add(&self, a: &Self::Commitment, b: &Self::Commitment) -> Self::Commitment;

    /// A commitment scaled by `s`.
    fn scale(&self, a: &Self::Commitment, s: F) -> Self::Commitment;
}
