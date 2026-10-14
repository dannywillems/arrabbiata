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
//!
//! ```
//! use arrabbiata_bn254::{Bn254, Fr};
//! use arrabbiata_commit::PolynomialCommitment;
//! use arrabbiata_field::Field;
//! use arrabbiata_kzg::Kzg;
//!
//! // A test setup whose secret is known: never use it in a deployment.
//! let kzg = Kzg::<Bn254>::insecure_setup(8, Fr::from_u64(123_456_789));
//! let p = [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
//! let c = kzg.commit(&p);
//! let (v, proof) = kzg.open(&p, &Fr::from_u64(2));
//! assert_eq!(v, Fr::from_u64(17));
//! assert!(kzg.verify(&c, &Fr::from_u64(2), v, &proof));
//! ```

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment, Univariate};
use arrabbiata_curve::{Curve, Pairing, Projective, msm};
use arrabbiata_field::{Field, PrimeField, powers};
use arrabbiata_transcript::typed::Label;

/// The scalar field of the pairing groups.
type Scalar<E> = <<E as Pairing>::G1 as Curve>::Scalar;

/// The KZG scheme over the pairing `E`.
///
/// ```
/// use arrabbiata_bn254::{Bn254, Fr};
/// use arrabbiata_field::Field;
/// use arrabbiata_kzg::Kzg;
/// assert_eq!(Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(5)).size(), 4);
/// ```
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
    ///
    /// ```
    /// use arrabbiata_bn254::{Bn254, Fr};
    /// use arrabbiata_field::Field;
    /// use arrabbiata_kzg::Kzg;
    /// let _ = Kzg::<Bn254>::insecure_setup(2, Fr::from_u64(42));
    /// ```
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
    ///
    /// ```
    /// use arrabbiata_bn254::{Bn254, Fr, G1, G2};
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_kzg::Kzg;
    /// let tau = Fr::from_u64(9);
    /// let (g, h) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    /// let kzg = Kzg::<Bn254>::from_powers(vec![g, g * tau], h, h * tau);
    /// assert_eq!(kzg.size(), 2);
    /// ```
    #[must_use]
    pub const fn from_powers(
        powers: Vec<Projective<E::G1>>,
        h: Projective<E::G2>,
        tau_h: Projective<E::G2>,
    ) -> Self {
        Self { powers, h, tau_h }
    }

    /// The number of powers: polynomials must have fewer coefficients.
    ///
    /// ```
    /// use arrabbiata_bn254::{Bn254, Fr};
    /// use arrabbiata_field::Field;
    /// use arrabbiata_kzg::Kzg;
    /// assert_eq!(Kzg::<Bn254>::insecure_setup(16, Fr::ONE + Fr::ONE).size(), 16);
    /// ```
    #[must_use]
    pub fn size(&self) -> usize {
        self.powers.len()
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
        let lhs = *commitment - g * value + *proof * *point;
        E::pairing_product_is_one(&[(lhs, self.h), (-*proof, self.tau_h)])
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
