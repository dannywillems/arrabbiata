//! The KZG polynomial commitment (Kate, Zaverucha, Goldberg, ASIACRYPT
//! 2010), over any pairing.
//!
//! A structured reference string holds `[tau^i] G` in `G1` for `i < n`, and
//! `H, [tau] H` in `G2`, for a secret `tau`. A polynomial `p` of degree below
//! `n` is committed as `C = [p(tau)] G`. To open at `z` to `v = p(z)`, the
//! prover sends `pi = [q(tau)] G` for the quotient `q = (p - v) / (X - z)`,
//! and the verifier checks
//!
//! ```text
//! e(C - [v] G + [z] pi, H) = e(pi, [tau] H),
//! ```
//!
//! which holds because `p(tau) - v = q(tau) (tau - z)`. Binding rests on the
//! `n`-strong Diffie-Hellman assumption in the pairing groups, and on nobody
//! knowing `tau`.
//!
//! The commitment is a module morphism, so [`Kzg`] is also a
//! [`HomomorphicCommitment`] (a vector is committed as the coefficients of a
//! polynomial).

#![no_std]

extern crate alloc;

pub mod accumulation;

use alloc::vec::Vec;

use arrabbiata_commit::{
    BatchOpening, HomomorphicCommitment, LinearCommitment, PolynomialCommitment, Univariate,
    open_combined, verify_combined,
};
use arrabbiata_curve::{Curve, Pairing, Projective, msm};
use arrabbiata_field::{Field, PrimeField, powers};
use arrabbiata_transcript::typed::Label;

pub use crate::accumulation::PairingAccumulator;

/// The scalar field of the pairing groups.
type Scalar<E> = <<E as Pairing>::G1 as Curve>::Scalar;

/// The KZG scheme over the pairing `E`.
#[derive(Clone, Debug)]
pub struct Kzg<E>
where
    E: Pairing,
{
    powers: Vec<Projective<E::G1>>,
    h: Projective<E::G2>,
    tau_h: Projective<E::G2>,
}

impl<E> Label for Kzg<E>
where
    E: Pairing,
{
    const LABEL: u64 = 0x6b7a_6700;
}

impl<E> Kzg<E>
where
    E: Pairing,
{
    /// A reference string for polynomials of degree below `size`, computed
    /// from a known `tau`.
    ///
    /// INSECURE: whoever knows `tau` can open a commitment to any value. A
    /// deployment must use a string from a multi-party ceremony, loaded with
    /// [`Kzg::from_powers`]. This constructor exists for tests and examples.
    #[must_use]
    pub fn insecure_setup(size: usize, tau: Scalar<E>) -> Self {
        let g = Projective::<E::G1>::generator();
        let h = Projective::<E::G2>::generator();
        let taus = powers(tau, size);
        let mut points = Vec::with_capacity(size);
        let mut i = 0;
        while i < size {
            points.push(g * taus[i]);
            i += 1;
        }
        Self {
            powers: points,
            h,
            tau_h: h * tau,
        }
    }

    /// A scheme from a reference string `[tau^i] G` (with `powers[0]` the
    /// generator of `G1`), `H` and `[tau] H`.
    #[must_use]
    pub const fn from_powers(
        powers: Vec<Projective<E::G1>>,
        h: Projective<E::G2>,
        tau_h: Projective<E::G2>,
    ) -> Self {
        Self { powers, h, tau_h }
    }

    /// The number of powers: polynomials must have fewer coefficients.
    #[must_use]
    pub fn size(&self) -> usize {
        self.powers.len()
    }

    /// Whether `e(P, H) = e(W, [tau] H)`: the pairing check every KZG
    /// verification ends with, on the pair `(P, W)`. For an opening of `C`
    /// at `z` to `v` with proof `pi`, `P = C - [v] G + [z] pi` and `W = pi`.
    /// The pairs it accepts are the kernel of a group homomorphism `G1^2
    /// -> GT`, so a subgroup: this is what makes KZG accumulators fold by
    /// random linear combinations.
    #[must_use]
    pub fn decide_pair(&self, p: &Projective<E::G1>, w: &Projective<E::G1>) -> bool {
        E::pairing_product_is_one(&[(*p, self.h), (-*w, self.tau_h)])
    }

    fn commit_coefficients(&self, coefficients: &[Scalar<E>]) -> Projective<E::G1> {
        assert!(
            coefficients.len() <= self.size(),
            "more coefficients than powers in the reference string"
        );
        msm(&self.powers, coefficients)
    }
}

/// `p(z)` and the coefficients of `(p - p(z)) / (X - z)`, by synthetic
/// division.
fn divide_by_linear<F>(coefficients: &[F], z: F) -> (F, Vec<F>)
where
    F: Field,
{
    let n = coefficients.len();
    if n == 0 {
        return (F::ZERO, Vec::new());
    }
    let mut quotient = alloc::vec![F::ZERO; n - 1];
    let mut acc = F::ZERO;
    let mut i = n;
    while i > 0 {
        i -= 1;
        acc = acc * z + coefficients[i];
        if i > 0 {
            quotient[i - 1] = acc;
        }
    }
    (acc, quotient)
}

impl<E> PolynomialCommitment<Scalar<E>> for Kzg<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
{
    type Family = Univariate;
    type Commitment = Projective<E::G1>;
    type Proof = Projective<E::G1>;

    /// # Panics
    ///
    /// If the polynomial has more coefficients than the reference string has
    /// powers.
    fn commit(&self, coefficients: &[Scalar<E>]) -> Projective<E::G1> {
        self.commit_coefficients(coefficients)
    }

    fn open(
        &self,
        coefficients: &[Scalar<E>],
        point: &Scalar<E>,
    ) -> (Scalar<E>, Projective<E::G1>) {
        let (value, quotient) = divide_by_linear(coefficients, *point);
        (value, self.commit_coefficients(&quotient))
    }

    fn verify(
        &self,
        commitment: &Projective<E::G1>,
        point: &Scalar<E>,
        value: Scalar<E>,
        proof: &Projective<E::G1>,
    ) -> bool {
        let g = Projective::<E::G1>::generator();
        let p = *commitment - g * value + *proof * *point;
        self.decide_pair(&p, proof)
    }
}

impl<E> HomomorphicCommitment<Scalar<E>> for Kzg<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
{
    type Commitment = Projective<E::G1>;

    /// # Panics
    ///
    /// If the vector is longer than the reference string.
    fn commit(&self, values: &[Scalar<E>]) -> Projective<E::G1> {
        self.commit_coefficients(values)
    }

    fn add(&self, a: &Projective<E::G1>, b: &Projective<E::G1>) -> Projective<E::G1> {
        *a + *b
    }

    fn scale(&self, a: &Projective<E::G1>, s: Scalar<E>) -> Projective<E::G1> {
        *a * s
    }
}

impl<E> LinearCommitment<Scalar<E>> for Kzg<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
{
    fn combine(&self, terms: &[(Scalar<E>, &Projective<E::G1>)]) -> Projective<E::G1> {
        let mut sum = Projective::identity();
        for (s, c) in terms {
            sum = sum + **c * *s;
        }
        sum
    }
}

/// One opening of the combination by powers of `nu`.
impl<E> BatchOpening<Scalar<E>> for Kzg<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
{
    type BatchProof = <Self as PolynomialCommitment<Scalar<E>>>::Proof;

    fn open_batch(
        &self,
        polynomials: &[&[Scalar<E>]],
        point: &Scalar<E>,
        nu: Scalar<E>,
    ) -> (Vec<Scalar<E>>, Self::BatchProof) {
        open_combined(self, polynomials, point, nu)
    }

    fn verify_batch(
        &self,
        commitments: &[&Self::Commitment],
        point: &Scalar<E>,
        values: &[Scalar<E>],
        nu: Scalar<E>,
        proof: &Self::BatchProof,
    ) -> bool {
        verify_combined(self, commitments, point, values, nu, proof)
    }
}
