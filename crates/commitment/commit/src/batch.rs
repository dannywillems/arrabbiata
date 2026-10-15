//! Batched openings: many polynomials opened at one point.
//!
//! [`BatchOpening`] is what a protocol asks of its scheme: open
//! `p_0, ..., p_(k-1)` at one point `x`, for a challenge `nu` drawn after
//! the values, and check it. It is an argument for the product of `k`
//! evaluation claims, and every scheme has one: [`Unbatched`] proves the
//! claims one by one. What differs between schemes is its cost.
//!
//! A [`LinearCommitment`] has a cheap one: the prover opens
//! `p = sum_i nu^i p_i` at `x`; the verifier forms `C = sum_i nu^i C_i` and
//! `v = sum_i nu^i v_i` and checks the one opening ([`open_combined`],
//! [`verify_combined`]). If some `v_j` is not `p_j(x)`, the claimed `v`
//! differs from `p(x)` except for the at most `k - 1` values of `nu` that
//! are roots of `sum_i nu^i (v_i - p_i(x))`, a nonzero polynomial of degree
//! below `k` (Schwartz-Zippel). That is a reduction of `k` claims to one,
//! the same random linear combination as folding.
//!
//! A scheme whose commitments do not combine (FRI's Merkle roots) can
//! still batch, by combining its codewords where the verifier queries them,
//! with its own `BatchProof`.

use alloc::vec::Vec;
use core::fmt::Debug;

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, Transcript, absorb_slice, typed::Label};

use crate::pcs::{PolynomialCommitment, PolynomialFamily};

/// The point of evaluation of `PC`'s family.
pub type Point<F, PC> = <<PC as PolynomialCommitment<F>>::Family as PolynomialFamily<F>>::Point;

/// A polynomial commitment that opens many polynomials at one point.
///
/// Laws, for `(values, proof) = open_batch(polynomials, x, nu)`:
///
/// - `values[i] = Family::evaluate(polynomials[i], x)`;
/// - `verify_batch(commit(polynomials), x, values, nu, proof)` holds
///   (completeness);
/// - it fails, except with negligible probability over `nu`, when some
///   value is not the committed polynomial's, or the counts differ
///   (soundness).
pub trait BatchOpening<F>: PolynomialCommitment<F>
where
    F: Field,
{
    /// A proof of all the values.
    type BatchProof: Clone + Debug + PartialEq + Absorb<F>;

    /// The values of `polynomials` at `point`, and one proof of them, for
    /// the challenge `nu`.
    fn open_batch(
        &self,
        polynomials: &[&[F]],
        point: &Point<F, Self>,
        nu: F,
    ) -> (Vec<F>, Self::BatchProof);

    /// Whether `proof` shows that the polynomials committed to by
    /// `commitments` take `values` at `point`, for the challenge `nu`.
    fn verify_batch(
        &self,
        commitments: &[&Self::Commitment],
        point: &Point<F, Self>,
        values: &[F],
        nu: F,
        proof: &Self::BatchProof,
    ) -> bool;
}

/// A polynomial commitment whose commitments combine linearly: the
/// commitment to `sum_i s_i p_i` is `sum_i s_i C_i`.
///
/// Law: `combine(&[(s_i, commit(p_i))]) = commit(sum_i s_i p_i)`.
pub trait LinearCommitment<F>: PolynomialCommitment<F>
where
    F: Field,
{
    /// `sum_i s_i C_i` for the terms `(s_i, C_i)`.
    fn combine(&self, terms: &[(F, &Self::Commitment)]) -> Self::Commitment;
}

/// `sum_i nu^i p_i`, as coefficients (the longest length).
#[must_use]
pub fn combine_polynomials<F>(polynomials: &[&[F]], nu: F) -> Vec<F>
where
    F: Field,
{
    let mut length = 0;
    let mut i = 0;
    while i < polynomials.len() {
        if polynomials[i].len() > length {
            length = polynomials[i].len();
        }
        i += 1;
    }
    let mut combined = alloc::vec![F::ZERO; length];
    let mut power = F::ONE;
    let mut i = 0;
    while i < polynomials.len() {
        let p = polynomials[i];
        let mut j = 0;
        while j < p.len() {
            combined[j] = combined[j] + power * p[j];
            j += 1;
        }
        power = power * nu;
        i += 1;
    }
    combined
}

/// `sum_i nu^i v_i`.
#[must_use]
pub fn combine_values<F>(values: &[F], nu: F) -> F
where
    F: Field,
{
    let mut combined = F::ZERO;
    let mut power = F::ONE;
    let mut i = 0;
    while i < values.len() {
        combined = combined + power * values[i];
        power = power * nu;
        i += 1;
    }
    combined
}

/// The values of `polynomials` at `point`, and the opening of their
/// combination by powers of `nu`: [`BatchOpening::open_batch`] for a
/// linear scheme.
pub fn open_combined<F, PC>(
    pcs: &PC,
    polynomials: &[&[F]],
    point: &Point<F, PC>,
    nu: F,
) -> (Vec<F>, PC::Proof)
where
    F: Field,
    PC: LinearCommitment<F>,
{
    let mut values = Vec::with_capacity(polynomials.len());
    let mut i = 0;
    while i < polynomials.len() {
        values.push(<PC::Family as PolynomialFamily<F>>::evaluate(
            polynomials[i],
            point,
        ));
        i += 1;
    }
    let combined = combine_polynomials(polynomials, nu);
    let (_, proof) = pcs.open(&combined, point);
    (values, proof)
}

/// Whether `proof` opens the combination by powers of `nu` of
/// `commitments` to that of `values`: [`BatchOpening::verify_batch`] for a
/// linear scheme.
pub fn verify_combined<F, PC>(
    pcs: &PC,
    commitments: &[&PC::Commitment],
    point: &Point<F, PC>,
    values: &[F],
    nu: F,
    proof: &PC::Proof,
) -> bool
where
    F: Field,
    PC: LinearCommitment<F>,
{
    if commitments.len() != values.len() {
        return false;
    }
    let mut terms = Vec::with_capacity(commitments.len());
    let mut power = F::ONE;
    let mut i = 0;
    while i < commitments.len() {
        terms.push((power, commitments[i]));
        power = power * nu;
        i += 1;
    }
    let commitment = pcs.combine(&terms);
    pcs.verify(&commitment, point, combine_values(values, nu), proof)
}

/// Proofs, one per claim, absorbed with their count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proofs<P>(pub Vec<P>);

impl<F, P> Absorb<F> for Proofs<P>
where
    F: Field,
    P: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.0);
    }
}

/// Any polynomial commitment, batching by proving each claim on its own:
/// the batch proof is the `k` proofs, and `nu` is not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unbatched<PC>(pub PC);

impl<PC> Label for Unbatched<PC>
where
    PC: Label,
{
    /// The scheme's label, marked as unbatched: the two produce different
    /// proofs, so they must not share a transcript pattern.
    const LABEL: u64 = PC::LABEL ^ 0x7562_0000_0000_0000;
}

impl<F, PC> PolynomialCommitment<F> for Unbatched<PC>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    type Family = PC::Family;
    type Commitment = PC::Commitment;
    type Proof = PC::Proof;

    fn commit(&self, polynomial: &[F]) -> Self::Commitment {
        self.0.commit(polynomial)
    }

    fn open(&self, polynomial: &[F], point: &Point<F, PC>) -> (F, Self::Proof) {
        self.0.open(polynomial, point)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &Point<F, PC>,
        value: F,
        proof: &Self::Proof,
    ) -> bool {
        self.0.verify(commitment, point, value, proof)
    }
}

impl<F, PC> BatchOpening<F> for Unbatched<PC>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    type BatchProof = Proofs<PC::Proof>;

    fn open_batch(
        &self,
        polynomials: &[&[F]],
        point: &Point<F, PC>,
        _nu: F,
    ) -> (Vec<F>, Proofs<PC::Proof>) {
        let mut values = Vec::with_capacity(polynomials.len());
        let mut proofs = Vec::with_capacity(polynomials.len());
        let mut i = 0;
        while i < polynomials.len() {
            let (value, proof) = self.0.open(polynomials[i], point);
            values.push(value);
            proofs.push(proof);
            i += 1;
        }
        (values, Proofs(proofs))
    }

    fn verify_batch(
        &self,
        commitments: &[&Self::Commitment],
        point: &Point<F, PC>,
        values: &[F],
        _nu: F,
        proof: &Proofs<PC::Proof>,
    ) -> bool {
        if commitments.len() != values.len() || proof.0.len() != values.len() {
            return false;
        }
        let mut i = 0;
        while i < values.len() {
            if !self.0.verify(commitments[i], point, values[i], &proof.0[i]) {
                return false;
            }
            i += 1;
        }
        true
    }
}
