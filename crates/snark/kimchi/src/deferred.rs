//! Kimchi's succinct verifier factored in two halves, the way Pickles runs
//! it across a cycle of curves (`pickles/composition_types`,
//! `step_verifier.ml`, `wrap_verifier.ml`).
//!
//! A proof over `C` has points over the base field and scalars in the
//! scalar field. The succinct verifier ([`crate::verifier::succinct`]) does
//! both kinds of work: group operations, cheap in a circuit over the base
//! field, and scalar arithmetic, cheap in a circuit over the scalar field.
//! It factors through the values Pickles calls *deferred*:
//!
//! ```text
//! succinct = scalar_half . group_half
//! ```
//!
//! - [`group_half`] replays the base sponge over the commitments, checks
//!   that the challenges it draws are the deferred ones, builds `ft`'s
//!   commitment from the deferred `perm`, `zeta^n` and `zeta^m`, combines
//!   the commitments by the deferred `xi`, and checks the opening with the
//!   deferred `combined_inner_product` and `b`. It reads no evaluation and
//!   does no arithmetic in the scalar field but scaling points. It outputs
//!   a [`Deferred`] instance: the values, the digest of the base sponge, and
//!   the opening's challenges and `sg`.
//! - [`scalar_half`] replays the scalar sponge from that digest over the
//!   evaluations and checks every deferred value against its definition:
//!   `xi`, the linearization's `perm`, `zeta^n`, `zeta^m`, `ft(zeta)` inside
//!   the combined inner product, and `b` from the opening's challenges. It
//!   does no group operation, and outputs the accumulator `(u, sg)`.
//!
//! A deferred value is a computation `f(x)` the first half cannot afford,
//! replaced by a claimed `y` together with the obligation `y = f(x)`, which
//! the second half discharges. Composed, the two halves accept exactly what
//! the succinct verifier accepts and output the same accumulator.
//!
//! Pickles' own finalizer (`Plonk_checks.checked`) compares only `perm`
//! with its recomputation; [`scalar_half`] also compares `zeta^n` and
//! `zeta^m`, which [`group_half`] scales `t`'s and the last sigma's
//! commitments by.

use alloc::vec::Vec;
use core::fmt;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_ipa::{RecursionChallenge, Statement, folded_evaluation};
use arrabbiata_poly::Domain;
use arrabbiata_transcript::Transcript;

use crate::{
    gates::Gates,
    ipa::kimchi_ipa,
    oracle::ScalarChallenge,
    prover::{ProofEvaluations, ProverProof},
    srs::{PolyComm, Srs},
    verifier::{
        Challenges, VerifyError, base_transcript, combined_commitment, combined_inner_product,
        commitments, evaluation_shape, evaluations, ft_comm, linearization, oracles,
        scalar_transcript,
    },
    verifier_index::VerifierIndex,
};

/// The deferred values of the linearization (Pickles' `Plonk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plonk<F> {
    /// The 128-bit challenge `alpha` comes from.
    pub alpha: ScalarChallenge<F>,
    /// The first permutation challenge.
    pub beta: F,
    /// The second permutation challenge.
    pub gamma: F,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta: ScalarChallenge<F>,
    /// With lookups, the 128-bit challenge the joint combiner comes from.
    pub joint_combiner: Option<ScalarChallenge<F>>,
    /// `zeta^m`, the chunk size `m`.
    pub zeta_to_srs_length: F,
    /// `zeta^n`, the domain's size `n`.
    pub zeta_to_domain_size: F,
    /// The scalar of the last sigma's commitment in `ft`'s.
    pub perm: F,
}

/// The values the group half takes on trust and the scalar half checks
/// (Pickles' `Deferred_values`, without the branch data, which is
/// Pickles').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredValues<F> {
    /// The linearization's.
    pub plonk: Plonk<F>,
    /// The value the opening proves.
    pub combined_inner_product: F,
    /// The opening's folded evaluation vector, `b(zeta) + r b(zeta w)`.
    pub b: F,
    /// The 128-bit challenge the polynomial combiner `xi` comes from.
    pub xi: ScalarChallenge<F>,
    /// The opening's round challenges.
    pub bulletproof_challenges: Vec<F>,
}

/// What the group half leaves to the scalar half: the deferred values, the
/// base sponge's digest the scalar sponge starts from, the previous
/// accumulators, and the opening's `sg`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deferred<C>
where
    C: Glv,
{
    /// The deferred values.
    pub values: DeferredValues<C::Scalar>,
    /// The base sponge's digest after `zeta` (Pickles'
    /// `sponge_digest_before_evaluations`).
    pub digest: C::Scalar,
    /// The previous accumulators the proof carries.
    pub prev_challenges: Vec<RecursionChallenge<C>>,
    /// The opening's folded generator.
    pub sg: Projective<C>,
}

/// What the scalar half reads: the proof's evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evaluations<F> {
    /// The evaluations at `zeta` and `zeta w`.
    pub evals: ProofEvaluations<F>,
    /// `ft(zeta w)`.
    pub ft_eval1: F,
}

impl<C> From<&ProverProof<C>> for Evaluations<C::Scalar>
where
    C: Glv,
{
    fn from(proof: &ProverProof<C>) -> Self {
        Self {
            evals: proof.evals.clone(),
            ft_eval1: proof.ft_eval1,
        }
    }
}

/// A deferred value, as the obligation `y = f(x)` it stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Obligation {
    /// `alpha`.
    Alpha,
    /// `beta`.
    Beta,
    /// `gamma`.
    Gamma,
    /// `zeta`.
    Zeta,
    /// The joint combiner.
    JointCombiner,
    /// `perm`.
    Perm,
    /// `zeta^n`.
    ZetaToDomainSize,
    /// `zeta^m`.
    ZetaToSrsLength,
    /// `xi`.
    Xi,
    /// The combined inner product.
    CombinedInnerProduct,
    /// `b`.
    B,
    /// The opening's challenges.
    BulletproofChallenges,
}

/// Why a half refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeferredError {
    /// What the succinct verifier refuses too.
    Verify(VerifyError),
    /// A deferred value is not what it claims to be.
    Mismatch(Obligation),
}

impl fmt::Display for DeferredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verify(e) => write!(f, "{e}"),
            Self::Mismatch(v) => write!(f, "the deferred {v:?} is not its definition"),
        }
    }
}

impl From<VerifyError> for DeferredError {
    fn from(e: VerifyError) -> Self {
        Self::Verify(e)
    }
}

/// `Ok` when `claimed` is `actual`, else the mismatch of `value`.
fn expect<X: PartialEq>(
    claimed: X,
    actual: X,
    obligation: Obligation,
) -> Result<(), DeferredError> {
    if claimed == actual {
        Ok(())
    } else {
        Err(DeferredError::Mismatch(obligation))
    }
}

/// The deferred values of an honest run on `proof`: everything the
/// verifier derives (Pickles' prover computes them out of circuit, its
/// verifier as `expand_deferred`).
///
/// # Errors
///
/// A [`VerifyError`] for a proof of the wrong shape, or an opening with a
/// zero challenge.
///
/// # Panics
///
/// If the index's domain is not one of the field's.
pub fn expand_deferred<C, S, T, G>(
    index: &VerifierIndex<C>,
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
    gates: &G,
    sponges: (&S, &T),
) -> Result<DeferredValues<C::Scalar>, VerifyError>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
{
    let o = oracles(index, srs, proof, public, gates, sponges)?;
    let base = base_transcript(index, srs, proof, public, sponges.0)?;
    let lookup = evaluation_shape(index, &proof.evals)?;
    let lin = linearization(index, &proof.evals, lookup, &base.challenges(), gates);
    let comms: Vec<PolyComm<C>> = o.evaluations.iter().map(|(c, _)| c.clone()).collect();
    let combined = combined_commitment(&comms, o.v);
    let points = [o.zeta, o.zeta_omega];
    let statement = Statement {
        commitment: &combined,
        points: &points,
        evalscale: o.u,
        value: o.combined_inner_product,
    };
    let mut sponge = o.fq_sponge.clone();
    let (accumulator, _) = kimchi_ipa(srs)
        .check(&mut sponge, &statement, &(&proof.proof).into())
        .ok_or(VerifyError::OpeningProof)?;
    Ok(DeferredValues {
        plonk: Plonk {
            alpha: o.alpha_chal,
            beta: o.beta,
            gamma: o.gamma,
            zeta: o.zeta_chal,
            joint_combiner: base.joint_combiner_chal,
            zeta_to_srs_length: lin.zeta_to_srs_length,
            zeta_to_domain_size: lin.zeta_to_domain_size,
            perm: lin.perm,
        },
        combined_inner_product: o.combined_inner_product,
        b: folded_evaluation(&accumulator.chals, &points, o.u),
        xi: o.v_chal,
        bulletproof_challenges: accumulator.chals,
    })
}

/// The group half: the commitments and the opening of `proof` checked up
/// to `values` (see the module documentation). It reads neither the
/// proof's evaluations nor `ft(zeta w)`.
///
/// # Errors
///
/// A [`DeferredError`]: the commitments' shape, a challenge that is not
/// the deferred one, or an opening that does not hold with the deferred
/// values.
///
/// # Panics
///
/// If the index's domain is not one of the field's.
pub fn group_half<C, S>(
    index: &VerifierIndex<C>,
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
    values: &DeferredValues<C::Scalar>,
    base_sponge: &S,
) -> Result<Deferred<C>, DeferredError>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
{
    let base = base_transcript(index, srs, proof, public, base_sponge)?;
    let plonk = &values.plonk;
    expect(plonk.alpha, base.alpha_chal, Obligation::Alpha)?;
    expect(plonk.beta, base.beta, Obligation::Beta)?;
    expect(plonk.gamma, base.gamma, Obligation::Gamma)?;
    expect(plonk.zeta, base.zeta_chal, Obligation::Zeta)?;
    expect(
        plonk.joint_combiner,
        base.joint_combiner_chal,
        Obligation::JointCombiner,
    )?;
    let lambda = C::lambda();
    let ft = ft_comm(
        index,
        &proof.t_comm,
        plonk.perm,
        plonk.zeta_to_srs_length,
        plonk.zeta_to_domain_size,
    );
    let joint_combiner = plonk.joint_combiner.map(|c| c.to_field(lambda));
    let comms = commitments(index, proof, &base.public_comm, ft, joint_combiner);
    let combined = combined_commitment(&comms, values.xi.to_field(lambda));
    // The binding absorbs the value alone; the points and the point
    // combiner are the scalar half's, and `b0` is the deferred `b`.
    let statement = Statement {
        commitment: &combined,
        points: &[],
        evalscale: C::Scalar::ZERO,
        value: values.combined_inner_product,
    };
    let mut sponge = base.fq_sponge.clone();
    let (accumulator, expression) = kimchi_ipa(srs)
        .check_given(&mut sponge, &statement, &(&proof.proof).into(), |_| {
            values.b
        })
        .ok_or(VerifyError::OpeningProof)?;
    if !expression.is_identity() {
        return Err(DeferredError::Verify(VerifyError::OpeningProof));
    }
    expect(
        &values.bulletproof_challenges,
        &accumulator.chals,
        Obligation::BulletproofChallenges,
    )?;
    Ok(Deferred {
        values: values.clone(),
        digest: base.digest,
        prev_challenges: proof.prev_challenges.clone(),
        sg: accumulator.comm,
    })
}

/// The scalar half: every deferred value of `deferred` checked against its
/// definition on `evaluations` (see the module documentation), and the
/// accumulator `(u, sg)` it leaves.
///
/// # Errors
///
/// A [`DeferredError`]: the evaluations' shape, or a deferred value that is
/// not its definition.
///
/// # Panics
///
/// If the index's domain is not one of the field's.
pub fn scalar_half<C, T, G>(
    index: &VerifierIndex<C>,
    deferred: &Deferred<C>,
    opened: &Evaluations<C::Scalar>,
    gates: &G,
    scalar_sponge: &T,
) -> Result<RecursionChallenge<C>, DeferredError>
where
    C: Glv,
    C::Scalar: TwoAdicField,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
{
    let values = &deferred.values;
    let plonk = &values.plonk;
    let evals = &opened.evals;
    let lookup = evaluation_shape(index, evals)?;
    let prev: Vec<Vec<C::Scalar>> = deferred
        .prev_challenges
        .iter()
        .map(|p| p.chals.clone())
        .collect();
    let (xi, r) = scalar_transcript(
        scalar_sponge,
        deferred.digest,
        &prev,
        opened.ft_eval1,
        evals,
    );
    expect(values.xi, xi, Obligation::Xi)?;
    let lambda = C::lambda();
    let (xi, r) = (xi.to_field(lambda), r.to_field(lambda));
    let challenges = Challenges {
        alpha: plonk.alpha.to_field(lambda),
        beta: plonk.beta,
        gamma: plonk.gamma,
        zeta: plonk.zeta.to_field(lambda),
        joint_combiner: plonk.joint_combiner.map(|c| c.to_field(lambda)),
    };
    let lin = linearization(index, evals, lookup, &challenges, gates);
    expect(plonk.perm, lin.perm, Obligation::Perm)?;
    expect(
        plonk.zeta_to_domain_size,
        lin.zeta_to_domain_size,
        Obligation::ZetaToDomainSize,
    )?;
    expect(
        plonk.zeta_to_srs_length,
        lin.zeta_to_srs_length,
        Obligation::ZetaToSrsLength,
    )?;
    let zeta = challenges.zeta;
    let domain = match Domain::<C::Scalar>::containing(index.domain_size) {
        Some(d) => d,
        None => panic!("the index's domain is not one of the field's"),
    };
    let points = [zeta, zeta * domain.generator()];
    let all = evaluations(evals, &prev, [lin.ft_eval0, opened.ft_eval1], points);
    expect(
        values.combined_inner_product,
        combined_inner_product(&all, xi, r),
        Obligation::CombinedInnerProduct,
    )?;
    expect(
        values.b,
        folded_evaluation(&values.bulletproof_challenges, &points, r),
        Obligation::B,
    )?;
    Ok(RecursionChallenge {
        chals: values.bulletproof_challenges.clone(),
        comm: deferred.sg,
    })
}
