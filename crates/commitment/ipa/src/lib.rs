//! The inner-product argument (IPA) polynomial commitment, over any curve.
//!
//! A polynomial `a(X) = sum_i a_i X^i` of degree below `n = 2^k` is
//! committed as the Pedersen vector commitment `C = sum_i a_i G_i`, for
//! generators `G_i` with no known discrete-log relations. To open, the
//! prover shows `<a, b> = v` with the inner-product argument of
//! Bulletproofs (Bunz, Bootle, Boneh, Poelstra, Wuille, Maxwell, S&P 2018),
//! in the form used by Halo (Bowe, Grigg, Hopwood, ePrint 2019/1021) and
//! Kimchi: `k` rounds, each halving the vectors and sending two points
//! `L, R`, then one scalar and the folded generator `sg`. No trusted setup.
//!
//! As in Kimchi, `b` need not be the powers of one point: for points
//! `x_0, ..., x_(m-1)` and a scale `evalscale`, `b = sum_j evalscale^j (1,
//! x_j, x_j^2, ...)` ([`evaluation_vector`]), so one proof shows the
//! combination `sum_j evalscale^j a(x_j)`. With the polynomials themselves
//! combined by `polyscale` first, that is one proof for many polynomials at
//! many points ([`combined_inner_product`]).
//!
//! The value `v` is bound into the statement through `U' = xi U`, where
//! `xi` is a challenge derived from the [`Statement`]: the argument proves
//! `C + v U' = <a, G> + <a, b> U'`.
//!
//! The verifier comes in two halves, which is what Halo's accumulation is
//! made of: [`Ipa::succinct`] checks everything but `sg = <s, G>` in `O(k)`
//! and returns the claim as a [`RecursionChallenge`]; [`Ipa::decide`] (or
//! [`Ipa::decide_all`], for many) checks it with one MSM of size `n`.
//! [`PolynomialCommitment::verify`] is the two in sequence.
//!
//! This implementation is not hiding (there is no blinding term), so it is
//! not zero-knowledge on its own.
//!
//! The commitment is a module morphism, so [`Ipa`] is also the
//! [`HomomorphicCommitment`] that folding needs.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use arrabbiata_commit::{
    BatchOpening, HomomorphicCommitment, LinearCommitment, PolynomialCommitment, PolynomialFamily,
    Univariate, open_combined, verify_combined,
};
use arrabbiata_curve::{Curve, Projective, absorb_point, msm, points_from_transcript};
use arrabbiata_field::{Field, PrimeField, Sqrt, inner_product};
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, absorb_slice, typed::Label};

pub mod accumulation;
pub mod challenge;

pub use accumulation::{Accumulators, Lift};
pub use challenge::{
    RecursionChallenge, b_poly, b_poly_coefficients, combined_inner_product, evaluation_vector,
};

/// Domain separator absorbed at the start of every opening.
const DOMAIN: u64 = 0x6970_615f_6f70_656e;

/// The IPA commitment scheme over the curve `C`, with polynomials over its
/// scalar field and Fiat-Shamir challenges from a clone of `T`.
#[derive(Clone, Debug)]
pub struct Ipa<C, T>
where
    C: Curve,
{
    generators: Vec<Projective<C>>,
    u: Projective<C>,
    transcript: T,
}

/// An IPA opening proof, Kimchi's `OpeningProof` without the blinding
/// (`delta`, `z2`): the `(L, R)` pairs of the `k` rounds, the final scalar
/// `a` (Kimchi's `z1` for a zero blinder), and `sg`, the folded generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpaProof<C>
where
    C: Curve,
{
    rounds: Vec<(Projective<C>, Projective<C>)>,
    a: C::Scalar,
    sg: Projective<C>,
}

impl<C> IpaProof<C>
where
    C: Curve,
{
    /// The `(L, R)` points, one pair per round.
    #[must_use]
    pub fn rounds(&self) -> &[(Projective<C>, Projective<C>)] {
        &self.rounds
    }

    /// A proof with the given parts, for tests that tamper with proofs.
    #[must_use]
    pub const fn from_parts(
        rounds: Vec<(Projective<C>, Projective<C>)>,
        a: C::Scalar,
        sg: Projective<C>,
    ) -> Self {
        Self { rounds, a, sg }
    }

    /// The final scalar.
    #[must_use]
    pub const fn scalar(&self) -> C::Scalar {
        self.a
    }

    /// The folded generator: the claimed commitment to the challenge
    /// polynomial.
    #[must_use]
    pub const fn sg(&self) -> Projective<C> {
        self.sg
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
        absorb_point(&self.sg, transcript);
    }
}

impl<C, T> Label for Ipa<C, T>
where
    C: Curve,
{
    const LABEL: u64 = 0x6970_6100;
}

/// The statement of an opening: a commitment, the points, the scale
/// combining them, and the value of the polynomial on the combined
/// evaluation vector.
#[derive(Clone, Copy, Debug)]
pub struct Statement<'a, C>
where
    C: Curve,
{
    /// The commitment.
    pub commitment: &'a Projective<C>,
    /// The points.
    pub points: &'a [C::Scalar],
    /// The scale combining the points (`evalscale`).
    pub evalscale: C::Scalar,
    /// `sum_j evalscale^j p(points[j])`.
    pub value: C::Scalar,
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

    /// A scheme on given generators: `generators` (a power of two of them)
    /// for the coefficients and `u` for the value, with challenges from
    /// clones of `transcript`. For a key derived elsewhere, such as
    /// Kimchi's SRS, so that accumulators made against it are decided and
    /// folded by this scheme. The generators must have no known discrete-log
    /// relation, `u` included.
    ///
    /// # Panics
    ///
    /// If the number of generators is not a power of two.
    #[must_use]
    pub fn from_generators(
        generators: Vec<Projective<C>>,
        u: Projective<C>,
        transcript: T,
    ) -> Self {
        assert!(
            generators.len().is_power_of_two(),
            "the size must be a power of two"
        );
        Self {
            generators,
            u,
            transcript,
        }
    }

    /// The number of generators, an upper bound on the number of
    /// coefficients.
    #[must_use]
    pub fn size(&self) -> usize {
        self.generators.len()
    }

    /// The number of rounds of an opening, `log2` of the size.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.size().trailing_zeros() as usize
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

    /// Absorb the statement and derive `U' = xi U`, which binds the value:
    /// the argument proves `C + v U' = <a, G> + <a, b> U'`.
    fn start<Tr>(&self, transcript: &mut Tr, statement: &Statement<'_, C>) -> Projective<C>
    where
        Tr: Transcript<C::Scalar>,
    {
        transcript.absorb(C::Scalar::from_u64(DOMAIN));
        absorb_point(statement.commitment, transcript);
        absorb_slice(transcript, statement.points);
        transcript.absorb(statement.evalscale);
        transcript.absorb(statement.value);
        let xi = transcript.squeeze();
        self.u * xi
    }

    /// Prove that `coefficients`, committed to by `statement.commitment`,
    /// takes `statement.value` on the evaluation vector of
    /// `statement.points` combined by `statement.evalscale`, on
    /// `transcript`.
    ///
    /// The rounds are Kimchi's (`SRS::open`): `L = <a_hi, G_lo> + <a_hi,
    /// b_lo> U'`, `R = <a_lo, G_hi> + <a_lo, b_hi> U'`, then with the round
    /// challenge `u`, `a <- a_lo + u^-1 a_hi`, `b <- b_lo + u b_hi`,
    /// `G <- G_lo + u G_hi`. Returns the proof and its accumulator, the
    /// round challenges with `sg`, which the verifier's [`Ipa::succinct`]
    /// outputs too.
    ///
    /// # Panics
    ///
    /// If the polynomial has more coefficients than generators.
    pub fn prove<Tr>(
        &self,
        transcript: &mut Tr,
        statement: &Statement<'_, C>,
        coefficients: &[C::Scalar],
    ) -> (IpaProof<C>, RecursionChallenge<C>)
    where
        Tr: Transcript<C::Scalar>,
    {
        let mut a = self.padded(coefficients);
        let mut b = evaluation_vector(statement.points, statement.evalscale, self.size());
        let u = self.start(transcript, statement);
        let mut g = self.generators.clone();
        let mut rounds = Vec::with_capacity(self.rounds());
        let mut chals = Vec::with_capacity(self.rounds());
        while a.len() > 1 {
            let half = a.len() / 2;
            let (a_lo, a_hi) = a.split_at(half);
            let (b_lo, b_hi) = b.split_at(half);
            let (g_lo, g_hi) = g.split_at(half);
            let l = msm(g_lo, a_hi) + u * inner_product(a_hi, b_lo);
            let r = msm(g_hi, a_lo) + u * inner_product(a_lo, b_hi);
            absorb_point(&l, transcript);
            absorb_point(&r, transcript);
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
                next_a.push(a_lo[i] + a_hi[i] * x_inv);
                next_b.push(b_lo[i] + b_hi[i] * x);
                next_g.push(g_lo[i] + g_hi[i] * x);
                i += 1;
            }
            rounds.push((l, r));
            chals.push(x);
            a = next_a;
            b = next_b;
            g = next_g;
        }
        let proof = IpaProof {
            rounds,
            a: a[0],
            sg: g[0],
        };
        let accumulator = RecursionChallenge {
            chals,
            comm: proof.sg,
        };
        (proof, accumulator)
    }

    /// The succinct half of the verifier, on `transcript`: the round
    /// challenges, and the check `Q = a sg + a b0 U'` with `Q = C + v U' +
    /// sum_j (u_j^-1 L_j + u_j R_j)` and `b0 = sum_j evalscale^j b(x_j)`.
    /// It costs `O(k)` group operations, and trusts `sg`: the accumulator
    /// it returns is the claim that `sg` commits to `b`, which [`Ipa::decide`]
    /// checks.
    pub fn succinct<Tr>(
        &self,
        transcript: &mut Tr,
        statement: &Statement<'_, C>,
        proof: &IpaProof<C>,
    ) -> Option<RecursionChallenge<C>>
    where
        Tr: Transcript<C::Scalar>,
    {
        let k = self.rounds();
        if proof.rounds.len() != k {
            return None;
        }
        let u = self.start(transcript, statement);
        let mut chals = Vec::with_capacity(k);
        let mut q = *statement.commitment + u * statement.value;
        let mut j = 0;
        while j < k {
            let (l, r) = proof.rounds[j];
            absorb_point(&l, transcript);
            absorb_point(&r, transcript);
            let x = transcript.squeeze();
            let x_inv = x.inverse()?;
            q = q + l * x_inv + r * x;
            chals.push(x);
            j += 1;
        }
        let mut b0 = C::Scalar::ZERO;
        let mut scale = C::Scalar::ONE;
        let mut i = 0;
        while i < statement.points.len() {
            b0 = b0 + scale * b_poly(&chals, statement.points[i]);
            scale = scale * statement.evalscale;
            i += 1;
        }
        if q == proof.sg * proof.a + u * (proof.a * b0) {
            Some(RecursionChallenge {
                chals,
                comm: proof.sg,
            })
        } else {
            None
        }
    }

    /// The deferred half of the verifier: whether `accumulator.comm` is the
    /// commitment to the challenge polynomial, `<s, G>`, an MSM of the size
    /// of the key.
    #[must_use]
    pub fn decide(&self, accumulator: &RecursionChallenge<C>) -> bool {
        accumulator.chals.len() == self.rounds()
            && accumulator.comm == msm(&self.generators, &accumulator.coefficients())
    }

    /// [`Ipa::decide`] for many accumulators with one MSM: with `r` drawn
    /// after them, `sum_i r^i comm_i = <sum_i r^i s_i, G>`, which a false
    /// accumulator fails except for at most `m - 1` values of `r`
    /// (Kimchi's `sg_rand_base`).
    #[must_use]
    pub fn decide_all(&self, accumulators: &[RecursionChallenge<C>], r: C::Scalar) -> bool {
        let mut scalars = alloc::vec![C::Scalar::ZERO; self.size()];
        let mut combined = Projective::identity();
        let mut power = C::Scalar::ONE;
        let mut i = 0;
        while i < accumulators.len() {
            if accumulators[i].chals.len() != self.rounds() {
                return false;
            }
            let s = accumulators[i].coefficients();
            let mut j = 0;
            while j < s.len() {
                scalars[j] = scalars[j] + power * s[j];
                j += 1;
            }
            combined = combined + accumulators[i].comm * power;
            power = power * r;
            i += 1;
        }
        combined == msm(&self.generators, &scalars)
    }
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

    /// An opening at one point: [`Ipa::prove`] on a clone of the scheme's
    /// transcript.
    fn open(&self, coefficients: &[C::Scalar], point: &C::Scalar) -> (C::Scalar, IpaProof<C>) {
        let value = <Univariate as PolynomialFamily<C::Scalar>>::evaluate(coefficients, point);
        let commitment = msm(&self.generators, &self.padded(coefficients));
        let points = [*point];
        let statement = Statement {
            commitment: &commitment,
            points: &points,
            evalscale: C::Scalar::ONE,
            value,
        };
        let mut transcript = self.transcript.clone();
        let (proof, _) = self.prove(&mut transcript, &statement, coefficients);
        (value, proof)
    }

    /// [`Ipa::succinct`] then [`Ipa::decide`].
    fn verify(
        &self,
        commitment: &Projective<C>,
        point: &C::Scalar,
        value: C::Scalar,
        proof: &IpaProof<C>,
    ) -> bool {
        let points = [*point];
        let statement = Statement {
            commitment,
            points: &points,
            evalscale: C::Scalar::ONE,
            value,
        };
        let mut transcript = self.transcript.clone();
        match self.succinct(&mut transcript, &statement, proof) {
            Some(accumulator) => self.decide(&accumulator),
            None => false,
        }
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

impl<C, T> LinearCommitment<C::Scalar> for Ipa<C, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar> + Clone,
{
    fn combine(&self, terms: &[(C::Scalar, &Projective<C>)]) -> Projective<C> {
        let mut sum = Projective::identity();
        for (s, c) in terms {
            sum = sum + **c * *s;
        }
        sum
    }
}

/// One opening of the combination by powers of `nu`.
impl<C, T> BatchOpening<C::Scalar> for Ipa<C, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar> + Clone,
{
    type BatchProof = <Self as PolynomialCommitment<C::Scalar>>::Proof;

    fn open_batch(
        &self,
        polynomials: &[&[C::Scalar]],
        point: &C::Scalar,
        nu: C::Scalar,
    ) -> (Vec<C::Scalar>, Self::BatchProof) {
        open_combined(self, polynomials, point, nu)
    }

    fn verify_batch(
        &self,
        commitments: &[&Self::Commitment],
        point: &C::Scalar,
        values: &[C::Scalar],
        nu: C::Scalar,
        proof: &Self::BatchProof,
    ) -> bool {
        verify_combined(self, commitments, point, values, nu, proof)
    }
}
