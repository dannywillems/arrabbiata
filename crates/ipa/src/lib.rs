//! The inner-product argument (IPA) polynomial commitment, over any curve.
//!
//! A polynomial `a(X) = sum_i a_i X^i` of degree below `n = 2^k` is
//! committed as the Pedersen vector commitment `C = sum_i a_i G_i`, for
//! generators `G_i` with no known discrete-log relations. To open at `z`,
//! the prover shows `<a, b> = v` for `b = (1, z, ..., z^(n-1))` with the
//! inner-product argument of Bulletproofs (Bunz, Bootle, Boneh, Poelstra,
//! Wuille, Maxwell, S&P 2018), in the form used by Halo (Bowe, Grigg,
//! Hopwood, ePrint 2019/1021): `k` rounds, each halving the vectors and
//! sending two points `L, R`, then one scalar. No trusted setup; the
//! verifier is linear in `n`.
//!
//! The value `v` is bound into the statement through `U' = xi U`, where
//! `xi` is a challenge derived from `(C, z, v)`: the argument proves
//! `C + v U' = <a, G> + <a, b> U'`.
//!
//! This implementation is not hiding (there is no blinding term), so it is
//! not zero-knowledge on its own.
//!
//! The commitment is a module morphism, so [`Ipa`] is also the
//! [`HomomorphicCommitment`] that folding needs.
//!
//! ```
//! use arrabbiata_commit::PolynomialCommitment;
//! use arrabbiata_field::Field;
//! use arrabbiata_ipa::Ipa;
//! use arrabbiata_pasta::{Fq, Pallas};
//! use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
//!
//! // 8 generators on Pallas from Poseidon over Fp; challenges from Poseidon
//! // over Fq, the field the polynomials live in.
//! let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
//! let p = [Fq::from_u64(1), Fq::from_u64(2), Fq::from_u64(3)]; // 1 + 2X + 3X^2
//! let c = ipa.commit(&p);
//! let (v, proof) = ipa.open(&p, &Fq::from_u64(2));
//! assert_eq!(v, Fq::from_u64(17));
//! assert!(ipa.verify(&c, &Fq::from_u64(2), v, &proof));
//! assert!(!ipa.verify(&c, &Fq::from_u64(2), v + Fq::ONE, &proof));
//! ```

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment, Univariate};
use arrabbiata_curve::{Curve, Projective, absorb_point, msm, points_from_transcript};
use arrabbiata_field::{Field, PrimeField, Sqrt};
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, typed::Label};

/// Domain separator absorbed at the start of every opening.
const DOMAIN: u64 = 0x6970_615f_6f70_656e;

/// The IPA commitment scheme over the curve `C`, with polynomials over its
/// scalar field and Fiat-Shamir challenges from a clone of `T`.
///
/// ```
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::Pallas;
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// assert_eq!(ipa.size(), 4);
/// ```
#[derive(Clone, Debug)]
pub struct Ipa<C, T>
where
    C: Curve,
{
    generators: Vec<Projective<C>>,
    u: Projective<C>,
    transcript: T,
}

/// An IPA opening proof: the `(L, R)` pairs of the `k` rounds, and the
/// final scalar.
///
/// ```
/// use arrabbiata_commit::PolynomialCommitment;
/// use arrabbiata_field::Field;
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
/// let (_, proof) = ipa.open(&[Fq::ONE], &Fq::ZERO);
/// assert_eq!(proof.rounds().len(), 3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpaProof<C>
where
    C: Curve,
{
    rounds: Vec<(Projective<C>, Projective<C>)>,
    a: C::Scalar,
}

impl<C> IpaProof<C>
where
    C: Curve,
{
    /// The `(L, R)` points, one pair per round.
    ///
    /// ```
    /// use arrabbiata_commit::PolynomialCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    /// assert_eq!(ipa.open(&[Fq::ONE], &Fq::ONE).1.rounds().len(), 1);
    /// ```
    #[must_use]
    pub fn rounds(&self) -> &[(Projective<C>, Projective<C>)] {
        &self.rounds
    }

    /// A proof with the given parts, for tests that tamper with proofs.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_ipa::IpaProof;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// let proof = IpaProof::<Pallas>::from_parts(vec![], Fq::ONE);
    /// assert!(proof.rounds().is_empty());
    /// ```
    #[must_use]
    pub const fn from_parts(rounds: Vec<(Projective<C>, Projective<C>)>, a: C::Scalar) -> Self {
        Self { rounds, a }
    }

    /// The final scalar.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_ipa::IpaProof;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// assert_eq!(IpaProof::<Pallas>::from_parts(vec![], Fq::ONE).scalar(), Fq::ONE);
    /// ```
    #[must_use]
    pub const fn scalar(&self) -> C::Scalar {
        self.a
    }
}

impl<C> Absorb<C::Scalar> for IpaProof<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<C::Scalar>,
    {
        absorb_length(transcript, self.rounds.len());
        let mut i = 0;
        while i < self.rounds.len() {
            absorb_point(&self.rounds[i].0, transcript);
            absorb_point(&self.rounds[i].1, transcript);
            i += 1;
        }
        transcript.absorb(self.a);
    }
}

impl<C, T> Label for Ipa<C, T>
where
    C: Curve,
{
    const LABEL: u64 = 0x6970_6100;
}

fn inner_product<F>(a: &[F], b: &[F]) -> F
where
    F: Field,
{
    let mut acc = F::ZERO;
    let mut i = 0;
    while i < a.len() && i < b.len() {
        acc = acc + a[i] * b[i];
        i += 1;
    }
    acc
}

impl<C, T> Ipa<C, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar> + Clone,
{
    /// A scheme for polynomials of degree below `size` (a power of two),
    /// with `size + 1` generators derived from `points` (a transcript over
    /// the base field, see [`points_from_transcript`]), and challenges from
    /// clones of `transcript`.
    ///
    /// # Panics
    ///
    /// If `size` is not a power of two.
    ///
    /// ```
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::Pallas;
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(7), 16, vesta_sponge(1));
    /// assert_eq!(ipa.size(), 16);
    /// ```
    #[must_use]
    pub fn setup<P>(points: &mut P, size: usize, transcript: T) -> Self
    where
        P: Transcript<C::Base>,
        C::Base: Sqrt,
    {
        assert!(size.is_power_of_two(), "the size must be a power of two");
        let mut generators = points_from_transcript::<C, P>(points, size + 1);
        let u = match generators.pop() {
            Some(u) => u,
            None => unreachable!("size + 1 points were generated"),
        };
        Self {
            generators,
            u,
            transcript,
        }
    }

    /// The number of generators, an upper bound on the number of
    /// coefficients.
    ///
    /// ```
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::Pallas;
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// assert_eq!(Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1)).size(), 2);
    /// ```
    #[must_use]
    pub fn size(&self) -> usize {
        self.generators.len()
    }

    /// The coefficients padded with zeros to the number of generators.
    fn padded(&self, coefficients: &[C::Scalar]) -> Vec<C::Scalar> {
        assert!(
            coefficients.len() <= self.size(),
            "more coefficients than generators"
        );
        let mut a = coefficients.to_vec();
        a.resize(self.size(), C::Scalar::ZERO);
        a
    }

    /// The transcript of an opening of `commitment` at `point` to `value`,
    /// and the challenge `xi` binding them: `U' = xi U`.
    fn start(
        &self,
        commitment: &Projective<C>,
        point: C::Scalar,
        value: C::Scalar,
    ) -> (T, Projective<C>) {
        let mut transcript = self.transcript.clone();
        transcript.absorb(C::Scalar::from_u64(DOMAIN));
        absorb_point(commitment, &mut transcript);
        transcript.absorb(point);
        transcript.absorb(value);
        let xi = transcript.squeeze();
        (transcript, self.u * xi)
    }
}

/// `(1, z, z^2, ..., z^(n-1))`.
fn powers<F>(z: F, n: usize) -> Vec<F>
where
    F: Field,
{
    let mut out = Vec::with_capacity(n);
    let mut acc = F::ONE;
    let mut i = 0;
    while i < n {
        out.push(acc);
        acc = acc * z;
        i += 1;
    }
    out
}

impl<C, T> PolynomialCommitment<C::Scalar> for Ipa<C, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar> + Clone,
{
    type Family = Univariate;
    type Commitment = Projective<C>;
    type Proof = IpaProof<C>;

    /// # Panics
    ///
    /// If the polynomial has more coefficients than generators.
    fn commit(&self, coefficients: &[C::Scalar]) -> Projective<C> {
        msm(&self.generators, &self.padded(coefficients))
    }

    fn open(&self, coefficients: &[C::Scalar], point: &C::Scalar) -> (C::Scalar, IpaProof<C>) {
        let mut a = self.padded(coefficients);
        let mut b = powers(*point, self.size());
        let value = inner_product(&a, &b);
        let commitment = msm(&self.generators, &a);
        let (mut transcript, u) = self.start(&commitment, *point, value);
        let mut g = self.generators.clone();
        let mut rounds = Vec::new();
        while a.len() > 1 {
            let half = a.len() / 2;
            let (a_lo, a_hi) = a.split_at(half);
            let (b_lo, b_hi) = b.split_at(half);
            let (g_lo, g_hi) = g.split_at(half);
            let l = msm(g_hi, a_lo) + u * inner_product(a_lo, b_hi);
            let r = msm(g_lo, a_hi) + u * inner_product(a_hi, b_lo);
            absorb_point(&l, &mut transcript);
            absorb_point(&r, &mut transcript);
            let x = transcript.squeeze();
            let x_inv = match x.inverse() {
                Some(inverse) => inverse,
                // Probability 1 / |F|; the verifier rejects such a proof.
                None => C::Scalar::ONE,
            };
            let mut next_a = Vec::with_capacity(half);
            let mut next_b = Vec::with_capacity(half);
            let mut next_g = Vec::with_capacity(half);
            let mut i = 0;
            while i < half {
                next_a.push(a_lo[i] * x + a_hi[i] * x_inv);
                next_b.push(b_lo[i] * x_inv + b_hi[i] * x);
                next_g.push(g_lo[i] * x_inv + g_hi[i] * x);
                i += 1;
            }
            rounds.push((l, r));
            a = next_a;
            b = next_b;
            g = next_g;
        }
        (value, IpaProof { rounds, a: a[0] })
    }

    fn verify(
        &self,
        commitment: &Projective<C>,
        point: &C::Scalar,
        value: C::Scalar,
        proof: &IpaProof<C>,
    ) -> bool {
        let n = self.size();
        let k = n.trailing_zeros() as usize;
        if proof.rounds.len() != k {
            return false;
        }
        let (mut transcript, u) = self.start(commitment, *point, value);
        let mut challenges = Vec::with_capacity(k);
        let mut inverses = Vec::with_capacity(k);
        let mut p = *commitment + u * value;
        let mut j = 0;
        while j < k {
            let (l, r) = proof.rounds[j];
            absorb_point(&l, &mut transcript);
            absorb_point(&r, &mut transcript);
            let x = transcript.squeeze();
            let Some(x_inv) = x.inverse() else {
                return false;
            };
            p = p + l * (x * x) + r * (x_inv * x_inv);
            challenges.push(x);
            inverses.push(x_inv);
            j += 1;
        }
        // s_i = prod_j (x_j if bit j of i, from the top, is 1, else x_j^-1):
        // the coefficient of G_i, and of z^i, after all the rounds.
        let mut s = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut acc = C::Scalar::ONE;
            let mut j = 0;
            while j < k {
                acc = acc
                    * if (i >> (k - 1 - j)) & 1 == 1 {
                        challenges[j]
                    } else {
                        inverses[j]
                    };
                j += 1;
            }
            s.push(acc);
            i += 1;
        }
        let g = msm(&self.generators, &s);
        let b = inner_product(&s, &powers(*point, n));
        p == g * proof.a + u * (proof.a * b)
    }
}

impl<C, T> HomomorphicCommitment<C::Scalar> for Ipa<C, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar> + Clone,
{
    type Commitment = Projective<C>;

    /// # Panics
    ///
    /// If the vector is longer than the number of generators.
    fn commit(&self, values: &[C::Scalar]) -> Projective<C> {
        msm(&self.generators, &self.padded(values))
    }

    fn add(&self, a: &Projective<C>, b: &Projective<C>) -> Projective<C> {
        *a + *b
    }

    fn scale(&self, a: &Projective<C>, s: C::Scalar) -> Projective<C> {
        *a * s
    }
}
