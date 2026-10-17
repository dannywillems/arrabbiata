//! Kimchi's verifier (`kimchi/src/verifier.rs`: `oracles`, `to_batch`, and
//! `SRS::verify` of `poly-commitment`), for proofs without lookups.
//!
//! [`oracles`] replays the transcript: the base sponge over the index
//! digest, the previous accumulators, the public commitment (computed from
//! the public inputs, `h` alone when there are none), the witness, `z` and
//! `t` commitments, then the scalar sponge over the evaluations; it gives
//! the challenges, `ft(zeta)` and the combined inner product. `ft(zeta)`
//! (Kimchi's `ft_eval0`) is the permutation identity at `zeta`, minus the
//! public polynomial, plus the two boundary terms, minus every gate's
//! constraints at the evaluations (the constant term of Kimchi's
//! linearization: every column but the last sigma is evaluated, so it is
//! all of them).
//!
//! [`verify`] then checks the opening, as `SRS::verify` does for a batch
//! of one: with the round challenges `u_j` replayed, `s` the coefficients
//! of `b(X) = prod_j (1 + u_(k-j) X^(2^j))`, `b0 = b(zeta) + u b(zeta w)`,
//! `C` the commitments combined by powers of `v` (chunk by chunk, in the
//! order of the opening; `ft`'s commitment is `lin(f) - (zeta^n - 1)
//! lin(t)` with `f = scalar sigma_6`), random `r` and `r'`:
//!
//! `r (c (C + cip U + sum_j (u_j^(-1) L_j + u_j R_j)) + delta - z1 (sg +
//! b0 U) - z2 H) + r' (<s, G> - sg) = 0`.
//!
//! The first part is the IPA's Schnorr-like check, the second the check
//! that `sg` is the commitment of `b`, which Halo's accumulation defers;
//! `r` and `r'` (from the [`Sampler`]) keep the two from cancelling.

use alloc::{vec, vec::Vec};
use core::fmt;

use arrabbiata_curve::{Curve, Projective, glv::Glv, msm};
use arrabbiata_field::{Field, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_ipa::{b_poly, b_poly_coefficients};
use arrabbiata_poly::Domain;
use arrabbiata_transcript::Transcript;

use crate::{
    constraints::GateType,
    gates::Gates,
    oracle::{FqSponge, FrSponge, ScalarChallenge},
    permutation::PERMUTS,
    prover::{
        CONSTRAINED, PERMUTATION_ALPHA, PointEvaluations, ProverProof, combine, shift_scalar,
    },
    rng::Sampler,
    srs::{GroupMap, PolyComm, Srs},
    verifier_index::VerifierIndex,
};

/// Why a proof is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// The proof carries another number of previous accumulators than the
    /// index expects.
    IncorrectPrevChallengesLength,
    /// Another number of public inputs than the index expects.
    IncorrectPublicInputLength,
    /// A commitment or an evaluation with the wrong number of chunks.
    IncorrectLength,
    /// The index has lookups, not supported yet.
    LookupsUnsupported,
    /// The opening has another number of rounds than the SRS.
    IncorrectRounds,
    /// The opening equation does not hold.
    OpeningProof,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncorrectPrevChallengesLength => {
                write!(f, "wrong number of previous accumulators")
            }
            Self::IncorrectPublicInputLength => write!(f, "wrong number of public inputs"),
            Self::IncorrectLength => write!(f, "a commitment or evaluation of the wrong size"),
            Self::LookupsUnsupported => write!(f, "lookups are not supported yet"),
            Self::IncorrectRounds => write!(f, "the opening has the wrong number of rounds"),
            Self::OpeningProof => write!(f, "the opening does not verify"),
        }
    }
}

/// An opened polynomial: its commitment and its evaluations (one per chunk)
/// at `zeta` and at `zeta w`.
pub type Claim<C> = (PolyComm<C>, [Vec<<C as Curve>::Scalar>; 2]);

/// The verifier's view of the transcript.
#[derive(Clone, Debug)]
pub struct Oracles<C, S>
where
    C: Curve,
{
    /// The first permutation challenge.
    pub beta: C::Scalar,
    /// The second permutation challenge.
    pub gamma: C::Scalar,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha_chal: ScalarChallenge<C::Scalar>,
    /// The constraint combiner.
    pub alpha: C::Scalar,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta_chal: ScalarChallenge<C::Scalar>,
    /// The evaluation point.
    pub zeta: C::Scalar,
    /// `zeta w`.
    pub zeta_omega: C::Scalar,
    /// The 128-bit challenge `v` comes from.
    pub v_chal: ScalarChallenge<C::Scalar>,
    /// The polynomial combiner.
    pub v: C::Scalar,
    /// The 128-bit challenge `u` comes from.
    pub u_chal: ScalarChallenge<C::Scalar>,
    /// The point combiner.
    pub u: C::Scalar,
    /// `ft(zeta)`, as the verifier computes it.
    pub ft_eval0: C::Scalar,
    /// The combined inner product the opening must prove.
    pub combined_inner_product: C::Scalar,
    /// The base-field digest the scalar sponge starts from.
    pub digest: C::Scalar,
    /// The public commitment.
    pub public_comm: PolyComm<C>,
    /// The base sponge after `zeta`, where the opening continues.
    pub fq_sponge: FqSponge<C, S>,
    /// Each opened polynomial's commitment and evaluations at `zeta` and
    /// `zeta w`, in the order of the opening.
    pub evaluations: Vec<Claim<C>>,
}

/// The chunks `n / max_poly_size`, at least one.
fn chunks<C: Curve>(index: &VerifierIndex<C>) -> usize {
    if index.domain_size < index.max_poly_size {
        1
    } else {
        index.domain_size / index.max_poly_size
    }
}

/// `sum_i y^i c_i`, for commitments.
fn chunk_commitment<C: Curve>(c: &PolyComm<C>, y: C::Scalar) -> Projective<C> {
    c.chunks
        .iter()
        .rev()
        .fold(Projective::identity(), |acc, p| acc * y + *p)
}

/// The two evaluations of `e`.
fn pair<F: Clone>(e: &PointEvaluations<F>) -> [Vec<F>; 2] {
    [e.zeta.clone(), e.zeta_omega.clone()]
}

/// The transcript of `proof` against `index` and `public`, the gates'
/// constraints given by `gates` (see the module documentation).
///
/// # Errors
///
/// A [`VerifyError`] for a proof of the wrong shape.
///
/// # Panics
///
/// If the index's domain is not one of the field's.
pub fn oracles<C, S, T, G>(
    index: &VerifierIndex<C>,
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
    gates: &G,
    sponges: (&S, &T),
) -> Result<Oracles<C, S>, VerifyError>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
{
    let (base_sponge, scalar_sponge) = sponges;
    if index.lookup_index.is_some() {
        return Err(VerifyError::LookupsUnsupported);
    }
    if proof.prev_challenges.len() != index.prev_challenges {
        return Err(VerifyError::IncorrectPrevChallengesLength);
    }
    if public.len() != index.public {
        return Err(VerifyError::IncorrectPublicInputLength);
    }
    let num_chunks = chunks(index);
    let d1 = match Domain::<C::Scalar>::containing(index.domain_size) {
        Some(d) => d,
        None => panic!("the index's domain is not one of the field's"),
    };
    let n = d1.size();
    let m = index.max_poly_size;
    let e = &proof.evals;
    let all_evaluations = core::iter::once(&e.public)
        .chain(e.w.iter())
        .chain(core::iter::once(&e.z))
        .chain(e.s.iter())
        .chain(e.coefficients.iter())
        .chain([
            &e.generic_selector,
            &e.poseidon_selector,
            &e.complete_add_selector,
            &e.mul_selector,
            &e.emul_selector,
            &e.endomul_scalar_selector,
        ])
        .chain(e.optional_selectors.iter().map(|(_, p)| p));
    for p in all_evaluations {
        if p.zeta.len() != num_chunks || p.zeta_omega.len() != num_chunks {
            return Err(VerifyError::IncorrectLength);
        }
    }
    if proof.t_comm.chunks.len() > 7 * num_chunks {
        return Err(VerifyError::IncorrectLength);
    }

    // The public commitment: the Lagrange bases of the public rows, or h.
    let public_comm = if public.is_empty() {
        PolyComm {
            chunks: vec![srs.h; num_chunks],
        }
    } else {
        let mut negated = vec![C::Scalar::ZERO; n];
        for (slot, p) in negated.iter_mut().zip(public) {
            *slot = -*p;
        }
        let coefficients = match d1.interpolate(&negated) {
            Some(p) => p,
            None => unreachable!("one value per row"),
        };
        srs.mask_fixed(srs.commit_non_hiding(coefficients.coefficients(), n))
    };

    let mut fq_sponge = FqSponge::<C, S>::new(base_sponge.clone());
    fq_sponge.absorb_fq(&[index.digest(base_sponge.clone())]);
    for previous in &proof.prev_challenges {
        fq_sponge.absorb_g(&[previous.comm]);
    }
    fq_sponge.absorb_g(&public_comm.chunks);
    for c in &proof.w_comm {
        fq_sponge.absorb_g(&c.chunks);
    }
    let beta = fq_sponge.challenge();
    let gamma = fq_sponge.challenge();
    fq_sponge.absorb_g(&proof.z_comm.chunks);
    let alpha_chal = ScalarChallenge(fq_sponge.challenge());
    let alpha = alpha_chal.to_field(C::lambda());
    fq_sponge.absorb_g(&proof.t_comm.chunks);
    let zeta_chal = ScalarChallenge(fq_sponge.challenge());
    let zeta = zeta_chal.to_field(C::lambda());

    let digest = fq_sponge.clone().digest();
    let mut fr_sponge = FrSponge::<C::Scalar, T>::new(scalar_sponge.clone());
    fr_sponge.absorb(digest);
    let mut challenges = FrSponge::<C::Scalar, T>::new(scalar_sponge.clone());
    for previous in &proof.prev_challenges {
        challenges.absorb_multiple(&previous.chals);
    }
    fr_sponge.absorb(challenges.digest());

    let zeta1 = zeta.pow(n as u64);
    let zetaw = zeta * d1.generator();
    let zeta_m = zeta.pow(m as u64);
    let zetaw_m = zetaw.pow(m as u64);
    fr_sponge.absorb(proof.ft_eval1);
    fr_sponge.absorb_multiple(&e.public.zeta);
    fr_sponge.absorb_multiple(&e.public.zeta_omega);
    for p in e.absorbed() {
        fr_sponge.absorb_multiple(&p.zeta);
        fr_sponge.absorb_multiple(&p.zeta_omega);
    }
    let v_chal = fr_sponge.challenge();
    let v = v_chal.to_field(C::lambda());
    let u_chal = fr_sponge.challenge();
    let u = u_chal.to_field(C::lambda());

    // The evaluations, chunks combined.
    let at = |p: &PointEvaluations<C::Scalar>| {
        (combine(&p.zeta, zeta_m), combine(&p.zeta_omega, zetaw_m))
    };
    let mut alphas = [C::Scalar::ONE; PERMUTATION_ALPHA + 3];
    for k in 1..alphas.len() {
        alphas[k] = alphas[k - 1] * alpha;
    }
    let (alpha0, alpha1, alpha2) = (
        alphas[PERMUTATION_ALPHA],
        alphas[PERMUTATION_ALPHA + 1],
        alphas[PERMUTATION_ALPHA + 2],
    );
    let zk_rows = index.zk_rows;
    let w_end = d1.element(n - zk_rows);
    let vanishing =
        (zeta - w_end) * (zeta - d1.element(n - zk_rows + 1)) * (zeta - d1.element(n - 1));
    let z = at(&e.z);
    let mut ft_eval0 = (at(&e.w[PERMUTS - 1]).0 + gamma) * z.1 * alpha0 * vanishing;
    for i in 0..PERMUTS - 1 {
        ft_eval0 = ft_eval0 * (beta * at(&e.s[i]).0 + at(&e.w[i]).0 + gamma);
    }
    ft_eval0 = ft_eval0 - combine(&e.public.zeta, zeta_m);
    let mut shifted = alpha0 * vanishing * z.0;
    for i in 0..PERMUTS {
        shifted = shifted * (gamma + beta * zeta * index.shift[i] + at(&e.w[i]).0);
    }
    ft_eval0 = ft_eval0 - shifted;
    let zeta1m1 = zeta1 - C::Scalar::ONE;
    let numerator = (zeta1m1 * alpha1 * (zeta - w_end)
        + zeta1m1 * alpha2 * (zeta - C::Scalar::ONE))
        * (C::Scalar::ONE - z.0);
    let denominator = match ((zeta - w_end) * (zeta - C::Scalar::ONE)).inverse() {
        Some(x) => x,
        None => unreachable!("zeta is in the domain with negligible probability"),
    };
    ft_eval0 = ft_eval0 + numerator * denominator;
    // The gates: each selector times its constraints weighted by alpha^i.
    let witness = |c: usize, r: usize| if r == 0 { at(&e.w[c]).0 } else { at(&e.w[c]).1 };
    let coefficient = |c: usize, _: usize| at(&e.coefficients[c]).0;
    let selector = |typ: GateType| match typ {
        GateType::Generic => Some(&e.generic_selector),
        GateType::Poseidon => Some(&e.poseidon_selector),
        GateType::CompleteAdd => Some(&e.complete_add_selector),
        GateType::VarBaseMul => Some(&e.mul_selector),
        GateType::EndoMul => Some(&e.emul_selector),
        GateType::EndoMulScalar => Some(&e.endomul_scalar_selector),
        _ => e
            .optional_selectors
            .iter()
            .find(|(t, _)| *t == typ)
            .map(|(_, p)| p),
    };
    for typ in CONSTRAINED {
        let Some(s) = selector(typ) else {
            continue;
        };
        let values = gates.evaluate(typ, &witness, &coefficient);
        let combined = values
            .iter()
            .enumerate()
            .fold(C::Scalar::ZERO, |acc, (i, c)| acc + alphas[i] * *c);
        ft_eval0 = ft_eval0 - at(s).0 * combined;
    }

    // ft's commitment: lin(scalar sigma_6) - (zeta^n - 1) lin(t).
    let mut scalar = z.1 * beta * alpha0 * vanishing;
    for i in 0..PERMUTS - 1 {
        scalar = scalar * (gamma + beta * at(&e.s[i]).0 + at(&e.w[i]).0);
    }
    let scalar = -scalar;
    let f_comm = chunk_commitment(&index.sigma_comm[PERMUTS - 1], zeta_m) * scalar;
    let ft_comm = f_comm - chunk_commitment(&proof.t_comm, zeta_m) * zeta1m1;

    let mut evaluations: Vec<Claim<C>> = proof
        .prev_challenges
        .iter()
        .map(|previous| {
            let comm = PolyComm {
                chunks: vec![previous.comm],
            };
            (
                comm,
                [
                    vec![b_poly(&previous.chals, zeta)],
                    vec![b_poly(&previous.chals, zetaw)],
                ],
            )
        })
        .collect();
    evaluations.push((public_comm.clone(), pair(&e.public)));
    evaluations.push((
        PolyComm {
            chunks: vec![ft_comm],
        },
        [vec![ft_eval0], vec![proof.ft_eval1]],
    ));
    evaluations.push((proof.z_comm.clone(), pair(&e.z)));
    for (c, p) in [
        (&index.generic_comm, &e.generic_selector),
        (&index.psm_comm, &e.poseidon_selector),
        (&index.complete_add_comm, &e.complete_add_selector),
        (&index.mul_comm, &e.mul_selector),
        (&index.emul_comm, &e.emul_selector),
        (&index.endomul_scalar_comm, &e.endomul_scalar_selector),
    ] {
        evaluations.push((c.clone(), pair(p)));
    }
    for (c, p) in proof.w_comm.iter().zip(&e.w) {
        evaluations.push((c.clone(), pair(p)));
    }
    for (c, p) in index.coefficients_comm.iter().zip(&e.coefficients) {
        evaluations.push((c.clone(), pair(p)));
    }
    for (c, p) in index.sigma_comm.iter().zip(&e.s) {
        evaluations.push((c.clone(), pair(p)));
    }
    for (typ, comm) in [
        (GateType::RangeCheck0, &index.range_check0_comm),
        (GateType::RangeCheck1, &index.range_check1_comm),
        (GateType::ForeignFieldAdd, &index.foreign_field_add_comm),
        (GateType::ForeignFieldMul, &index.foreign_field_mul_comm),
        (GateType::Xor16, &index.xor_comm),
        (GateType::Rot64, &index.rot_comm),
    ] {
        if let Some(c) = comm {
            let p = selector(typ).ok_or(VerifyError::IncorrectLength)?;
            evaluations.push((c.clone(), pair(p)));
        }
    }
    // Kimchi's combined_inner_product: chunk i of each polynomial at the
    // next power of v, its two points combined by u.
    let mut combined_inner_product = C::Scalar::ZERO;
    let mut scale = C::Scalar::ONE;
    for (_, [at_zeta, at_zetaw]) in &evaluations {
        for (a, b) in at_zeta.iter().zip(at_zetaw) {
            combined_inner_product = combined_inner_product + scale * (*a + u * *b);
            scale = scale * v;
        }
    }
    Ok(Oracles {
        beta,
        gamma,
        alpha_chal,
        alpha,
        zeta_chal,
        zeta,
        zeta_omega: zetaw,
        v_chal,
        v,
        u_chal,
        u,
        ft_eval0,
        combined_inner_product,
        digest,
        public_comm,
        fq_sponge,
        evaluations,
    })
}

/// Verify `proof` (see the module documentation).
///
/// # Errors
///
/// A [`VerifyError`]; [`VerifyError::OpeningProof`] when the final check
/// fails.
pub fn verify<C, S, T, G, R>(
    index: &VerifierIndex<C>,
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
    gates: &G,
    sponges: (&S, &T),
    rng: &mut R,
) -> Result<(), VerifyError>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
    R: Sampler<C::Scalar>,
{
    let oracles = oracles(index, srs, proof, public, gates, sponges)?;
    let opening = &proof.proof;
    let size = srs.g.len();
    if !size.is_power_of_two() || opening.lr.len() != size.trailing_zeros() as usize {
        return Err(VerifyError::IncorrectRounds);
    }
    let mut sponge = oracles.fq_sponge.clone();
    let cip = oracles.combined_inner_product;
    sponge.absorb_fr(&[shift_scalar::<C>(cip)]);
    let group_map = GroupMap::<C>::setup();
    let u_base = group_map.to_group(sponge.challenge_fq()).to_projective();
    let mut chals = Vec::with_capacity(opening.lr.len());
    for (l, r) in &opening.lr {
        sponge.absorb_g(&[*l]);
        sponge.absorb_g(&[*r]);
        chals.push(ScalarChallenge(sponge.challenge()).to_field(C::lambda()));
    }
    sponge.absorb_g(&[opening.delta]);
    let c = ScalarChallenge(sponge.challenge()).to_field(C::lambda());
    let b0 = b_poly(&chals, oracles.zeta) + oracles.u * b_poly(&chals, oracles.zeta_omega);
    let s = b_poly_coefficients(&chals);
    let rand_base: C::Scalar = rng.sample();
    let sg_rand_base: C::Scalar = rng.sample();

    // C: every chunk at the next power of v.
    let mut combined = Projective::<C>::identity();
    let mut scale = C::Scalar::ONE;
    for (comm, _) in &oracles.evaluations {
        for chunk in &comm.chunks {
            combined = combined + *chunk * scale;
            scale = scale * oracles.v;
        }
    }
    let mut rounds = Projective::<C>::identity();
    for ((l, r), u) in opening.lr.iter().zip(&chals) {
        let u_inv = u.inverse().ok_or(VerifyError::OpeningProof)?;
        rounds = rounds + *l * u_inv + *r * *u;
    }
    let ipa = (combined + u_base * cip + rounds) * c + opening.delta
        - (opening.sg + u_base * b0) * opening.z1
        - srs.h * opening.z2;
    let sg_check = msm(&srs.g, &s) - opening.sg;
    let total = ipa * rand_base + sg_check * sg_rand_base;
    if total.is_identity() {
        Ok(())
    } else {
        Err(VerifyError::OpeningProof)
    }
}
