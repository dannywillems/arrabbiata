//! Kimchi's verifier (`kimchi/src/verifier.rs`: `oracles`, `to_batch`, and
//! `SRS::verify` of `poly-commitment`).
//!
//! [`oracles`] replays the transcript: the base sponge over the index
//! digest, the previous accumulators, the public commitment (computed from
//! the public inputs, `h` alone when there are none), the witness
//! commitments, with lookups the runtime table, the joint combiner, the
//! sorted columns and (after `gamma`) the lookup aggregation, the `z` and
//! `t` commitments, then the scalar sponge over the evaluations; it gives
//! the challenges, `ft(zeta)` and the combined inner product. `ft(zeta)`
//! (Kimchi's `ft_eval0`) is the permutation identity at `zeta`, minus the
//! public polynomial, plus the two boundary terms, minus every gate's
//! constraints at the evaluations (the constant term of Kimchi's
//! linearization: every column but the last sigma is evaluated, so it is
//! all of them), and minus the lookup constraints at the evaluations. The
//! joint table's commitment is the table columns' combined by powers of the
//! joint combiner, plus the ids' by `jc^max_joint_size`, plus the runtime
//! table's by `jc` (Kimchi's `combine_table`).
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
use core::{cell::RefCell, fmt};

use arrabbiata_curve::{Curve, Projective, glv::Glv, msm};
use arrabbiata_field::{Field, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_ipa::{RecursionChallenge, Statement, b_poly, b_poly_coefficients};
use arrabbiata_poly::Domain;
use arrabbiata_poseidon::kimchi::WIDTH;
use arrabbiata_transcript::{PermutationIn, Transcript};

use crate::{
    constraints::{COLUMNS, GateType},
    gates::Gates,
    ipa::kimchi_ipa,
    lookup::{LookupPattern, LookupPoint, constraint_count, constraints as lookup_constraints},
    oracle::{
        BaseReading, ChallengeModel, Duplexed, FqSponge, FrSponge, ScalarChallenge, SpongeIn, Value,
    },
    permutation::PERMUTS,
    prover::{
        CONSTRAINED, LOOKUP_ALPHA, LookupEvaluations, PERMUTATION_ALPHA, PointEvaluations,
        ProofEvaluations, ProverProof, combine, unnormalized_lagrange,
    },
    rng::Sampler,
    srs::{PolyComm, Srs, num_chunks},
    transcript::{self, Base, BaseElements, Message, Point, Scalar},
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
    /// The proof lacks the lookup commitments or evaluations the index
    /// needs, or has some it does not.
    LookupMismatch,
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
            Self::LookupMismatch => write!(f, "lookup parts missing or unexpected"),
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
const fn chunks<C: Curve>(index: &VerifierIndex<C>) -> usize {
    num_chunks(index.domain_size, index.max_poly_size)
}

/// The index's domain.
///
/// # Panics
///
/// If it is not one of the field's.
fn domain<C>(index: &VerifierIndex<C>) -> Domain<C::Scalar>
where
    C: Curve,
    C::Scalar: TwoAdicField,
{
    match Domain::<C::Scalar>::containing(index.domain_size) {
        Some(d) => d,
        None => panic!("the index's domain is not one of the field's"),
    }
}

/// `sum_i y^i c_i`, for commitments: [`arrabbiata_poly::combine`] in the group, the image of
/// the chunks' linearization under the commitment.
fn chunk_commitment<C: Curve>(c: &PolyComm<C>, y: C::Scalar) -> Projective<C> {
    arrabbiata_poly::combine(&c.chunks, y).unwrap_or_else(Projective::identity)
}

/// The two evaluations of `e`.
fn pair<F: Clone>(e: &PointEvaluations<F>) -> [Vec<F>; 2] {
    [e.zeta.clone(), e.zeta_omega.clone()]
}

/// The optional selectors of `index`, with their commitments, in the order
/// of the evaluations and of the opening.
fn optional_selectors<C: Curve>(index: &VerifierIndex<C>) -> [(GateType, &Option<PolyComm<C>>); 6] {
    [
        (GateType::RangeCheck0, &index.range_check0_comm),
        (GateType::RangeCheck1, &index.range_check1_comm),
        (GateType::ForeignFieldAdd, &index.foreign_field_add_comm),
        (GateType::ForeignFieldMul, &index.foreign_field_mul_comm),
        (GateType::Xor16, &index.xor_comm),
        (GateType::Rot64, &index.rot_comm),
    ]
}

/// The lookup selectors of `index`, in the order of the evaluations and of
/// the opening.
const LOOKUP_SELECTORS: [LookupPattern; 4] = [
    LookupPattern::Xor,
    LookupPattern::Lookup,
    LookupPattern::RangeCheck,
    LookupPattern::ForeignFieldMul,
];

/// The commitments' part of the shape: as many previous accumulators and
/// public inputs as the index says, a quotient of at most seven chunks per
/// chunk, and the lookup commitments exactly when the index has lookups,
/// the runtime table's exactly when it has runtime tables.
fn commitment_shape<C>(
    index: &VerifierIndex<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
) -> Result<(), VerifyError>
where
    C: Curve,
{
    match (&index.lookup_index, &proof.lookup) {
        (None, None) => {}
        (Some(li), Some(comms)) => {
            let runtime_ok = comms.runtime.is_some() == li.runtime_tables_selector.is_some();
            let sorted_ok = comms.sorted.len() == li.lookup_info.max_per_row + 1;
            if !(runtime_ok && sorted_ok) {
                return Err(VerifyError::LookupMismatch);
            }
        }
        _ => return Err(VerifyError::LookupMismatch),
    }
    if proof.prev_challenges.len() != index.prev_challenges {
        return Err(VerifyError::IncorrectPrevChallengesLength);
    }
    if public.len() != index.public {
        return Err(VerifyError::IncorrectPublicInputLength);
    }
    if proof.t_comm.chunks.len() > 7 * chunks(index) {
        return Err(VerifyError::IncorrectLength);
    }
    Ok(())
}

/// The evaluations' part of the shape, and the lookup evaluations if any:
/// one evaluation per chunk at each point; exactly the optional selectors
/// the index commits to, in its order; and with lookups, the sorted
/// columns, the runtime table and selector exactly when the index has
/// runtime tables, and exactly the index's lookup selectors, in its order.
///
/// The evaluations of a selector the index does not commit to are opened
/// against nothing, so a verifier that read them would let the prover add
/// to `ft(zeta)` a term it chooses after `zeta`; they are refused.
///
/// # Errors
///
/// [`VerifyError::LookupMismatch`] or [`VerifyError::IncorrectLength`].
pub fn evaluation_shape<'a, C>(
    index: &VerifierIndex<C>,
    evals: &'a ProofEvaluations<C::Scalar>,
) -> Result<Option<&'a LookupEvaluations<C::Scalar>>, VerifyError>
where
    C: Curve,
{
    let lookup = match (&index.lookup_index, &evals.lookup) {
        (None, None) => None,
        (Some(li), Some(l)) => {
            let runtime = li.runtime_tables_selector.is_some();
            let runtime_ok =
                l.runtime_table.is_some() == runtime && l.runtime_selector.is_some() == runtime;
            let sorted_ok = l.sorted.len() == li.lookup_info.max_per_row + 1;
            let expected = LOOKUP_SELECTORS
                .iter()
                .zip(&li.lookup_selectors)
                .filter(|(_, c)| c.is_some())
                .map(|(p, _)| *p);
            let selectors_ok = expected.eq(l.selectors.iter().map(|(p, _)| *p));
            if !(runtime_ok && sorted_ok && selectors_ok) {
                return Err(VerifyError::LookupMismatch);
            }
            Some(l)
        }
        _ => return Err(VerifyError::LookupMismatch),
    };
    let expected = optional_selectors(index)
        .into_iter()
        .filter(|(_, c)| c.is_some())
        .map(|(t, _)| t);
    if !expected.eq(evals.optional_selectors.iter().map(|(t, _)| *t)) {
        return Err(VerifyError::IncorrectLength);
    }
    let num_chunks = chunks(index);
    let e = evals;
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
        .chain(e.optional_selectors.iter().map(|(_, p)| p))
        .chain(e.lookup.iter().flat_map(|l| {
            [&l.aggregation, &l.table]
                .into_iter()
                .chain(l.sorted.iter())
                .chain(
                    [&l.runtime_table, &l.runtime_selector]
                        .into_iter()
                        .flatten(),
                )
                .chain(l.selectors.iter().map(|(_, p)| p))
        }));
    for p in all_evaluations {
        if p.zeta.len() != num_chunks || p.zeta_omega.len() != num_chunks {
            return Err(VerifyError::IncorrectLength);
        }
    }
    Ok(lookup)
}

/// The base-field half of the transcript: what Kimchi's verifier draws
/// from the commitments alone, before the evaluations. A verifier native in
/// the base field (Pickles' group side) computes exactly this.
#[derive(Clone, Debug)]
pub struct BaseTranscript<C, S>
where
    C: Curve,
{
    /// The first permutation challenge.
    pub beta: C::Scalar,
    /// The second permutation challenge.
    pub gamma: C::Scalar,
    /// With lookups, the 128-bit challenge the joint combiner comes from
    /// (zero when the index uses no joint lookup).
    pub joint_combiner_chal: Option<ScalarChallenge<C::Scalar>>,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha_chal: ScalarChallenge<C::Scalar>,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta_chal: ScalarChallenge<C::Scalar>,
    /// The base-field digest the scalar sponge starts from.
    pub digest: C::Scalar,
    /// The public commitment.
    pub public_comm: PolyComm<C>,
    /// The base sponge after `zeta`, where the opening continues.
    pub fq_sponge: FqSponge<C, S>,
}

/// The challenges of the linearization, as scalars.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Challenges<F> {
    /// The constraint combiner.
    pub alpha: F,
    /// The first permutation challenge.
    pub beta: F,
    /// The second permutation challenge.
    pub gamma: F,
    /// The evaluation point.
    pub zeta: F,
    /// With lookups, the joint combiner.
    pub joint_combiner: Option<F>,
}

impl<C, S> BaseTranscript<C, S>
where
    C: Glv,
{
    /// The challenges, the 128-bit ones mapped by the endomorphism.
    #[must_use]
    pub fn challenges(&self) -> Challenges<C::Scalar> {
        let lambda = C::lambda();
        Challenges {
            alpha: self.alpha_chal.to_field(lambda),
            beta: self.beta,
            gamma: self.gamma,
            zeta: self.zeta_chal.to_field(lambda),
            joint_combiner: self.joint_combiner_chal.map(|c| c.to_field(lambda)),
        }
    }
}

/// The base-field half of the transcript of `proof` against `index` and
/// `public`: the commitments' shape checked, the public commitment, then
/// the base sponge over the index digest, the previous accumulators, the
/// public commitment, the witness commitments, the lookup commitments, `z`
/// and `t`, as the type `transcript::Piop` says. It reads no evaluation.
///
/// # Errors
///
/// A [`VerifyError`] for commitments of the wrong shape.
///
/// # Panics
///
/// If the index's domain is not one of the field's.
pub fn base_transcript<C, S>(
    index: &VerifierIndex<C>,
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
    base_sponge: &S,
) -> Result<BaseTranscript<C, S>, VerifyError>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
{
    commitment_shape(index, proof, public)?;
    let num_chunks = chunks(index);
    let d1 = domain(index);
    let n = d1.size();

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
        srs.mask_fixed(srs.commit_non_hiding(coefficients.coefficients(), num_chunks))
    };

    let previous: Vec<Projective<C>> = proof.prev_challenges.iter().map(|p| p.comm).collect();
    let lookup = index.lookup_index.as_ref().zip(proof.lookup.as_ref());
    let inputs = PiopInputs {
        index_digest: index.digest(base_sponge.clone()),
        previous,
        public: public_comm.clone(),
        witness: proof.w_comm.clone(),
        lookup: lookup.map(|(li, comms)| LookupInputs {
            runtime: comms.runtime.clone(),
            joint: li.joint_lookup_used,
            sorted: comms.sorted.clone(),
            aggregation: comms.aggreg.clone(),
        }),
        z: proof.z_comm.clone(),
        t: proof.t_comm.clone(),
    };
    let (drawn, fq_sponge) = piop_transcript_on(FqSponge::new(base_sponge.clone()), &inputs);
    let joint_combiner_chal = inputs
        .lookup
        .as_ref()
        .map(|_| ScalarChallenge(drawn.joint_combiner.unwrap_or(C::Scalar::ZERO)));
    Ok(BaseTranscript {
        beta: drawn.beta,
        gamma: drawn.gamma,
        joint_combiner_chal,
        alpha_chal: ScalarChallenge(drawn.alpha),
        zeta_chal: ScalarChallenge(drawn.zeta),
        digest: drawn.digest,
        public_comm,
        fq_sponge,
    })
}

/// What the base-field half of the transcript absorbs, at the carriers `B`
/// of a base element, `P` of a point and `G` of a commitment: natively
/// `PolyComm<C>`s, in a circuit the chunks' coordinates.
#[derive(Clone, Debug)]
pub struct PiopInputs<B, P, G> {
    /// The verifier index's digest.
    pub index_digest: B,
    /// The previous accumulators' commitments.
    pub previous: Vec<P>,
    /// The public input's commitment.
    pub public: G,
    /// The witness columns' commitments.
    pub witness: [G; COLUMNS],
    /// With lookups, their commitments.
    pub lookup: Option<LookupInputs<G>>,
    /// The permutation's commitment.
    pub z: G,
    /// The quotient's commitment.
    pub t: G,
}

/// The lookups' part of [`PiopInputs`].
#[derive(Clone, Debug)]
pub struct LookupInputs<G> {
    /// With runtime tables, the runtime table's commitment.
    pub runtime: Option<G>,
    /// Whether a lookup is joint, so a joint combiner is drawn.
    pub joint: bool,
    /// The sorted columns' commitments.
    pub sorted: Vec<G>,
    /// The aggregation's commitment.
    pub aggregation: G,
}

/// The draws of the base-field half, as the sponge's model reads them
/// (natively scalars, in a circuit base-field values): the 128-bit
/// challenges come unmapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiopChallenges<R> {
    /// The 128-bit challenge of the joint combiner, when it is drawn.
    pub joint_combiner: Option<R>,
    /// `beta`.
    pub beta: R,
    /// `gamma`.
    pub gamma: R,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha: R,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta: R,
    /// The digest the scalar sponge starts from.
    pub digest: R,
}

/// The base-field half of the transcript, as `transcript::PiopOf` says, on
/// `sponge` in whichever model it computes: one definition for
/// [`base_transcript`] and [`base_transcript_in`]. Returns the draws and
/// the sponge after `zeta`, where the opening continues.
pub fn piop_transcript_on<C, S, H, P, G>(
    sponge: FqSponge<C, S, H>,
    inputs: &PiopInputs<Value<C::Base, H>, P, G>,
) -> (PiopChallenges<H::Read>, FqSponge<C, S, H>)
where
    C: Curve,
    H: SpongeIn<C::Base, S> + BaseReading<C>,
    S: Clone,
    Value<C::Base, H>: Clone,
    P: BaseElements<Value<C::Base, H>> + Clone,
    G: BaseElements<Value<C::Base, H>> + Clone,
{
    let tape = transcript::piop_on::<C, S, H, P, G>(sponge)
        .statement(&Message::new(Base(inputs.index_digest.clone())))
        .statement(&Message::new(inputs.previous.clone()))
        .statement(&Message::new(inputs.public.clone()))
        .receive(&Message::new(inputs.witness.clone()));
    let (joint_combiner, beta, gamma, tape) = match &inputs.lookup {
        None => {
            let (beta, tape) = tape.left().squeeze();
            let (gamma, tape) = tape.squeeze();
            (None, beta, gamma, tape)
        }
        Some(lookup) => {
            let tape = tape.right();
            let tape = match &lookup.runtime {
                Some(r) => tape.left().receive(&Message::new(r.clone())),
                None => tape.right(),
            };
            let (joint, tape) = if lookup.joint {
                let (j, tape) = tape.left().squeeze();
                (Some(j), tape)
            } else {
                (None, tape.right())
            };
            let tape = tape.receive(&Message::new(lookup.sorted.clone()));
            let (beta, tape) = tape.squeeze();
            let (gamma, tape) = tape.squeeze();
            let tape = tape.receive(&Message::new(lookup.aggregation.clone()));
            (joint, beta, gamma, tape)
        }
    };
    let tape = tape.receive(&Message::new(inputs.z.clone()));
    let (alpha, tape) = tape.squeeze();
    let tape = tape.receive(&Message::new(inputs.t.clone()));
    let (zeta, tape) = tape.squeeze();
    let (digest, tape) = tape.peek();
    let drawn = PiopChallenges {
        joint_combiner,
        beta,
        gamma,
        alpha,
        zeta,
        digest,
    };
    (drawn, tape.into_inner())
}

/// The base-field half of the transcript in a model of the base field, on
/// Kimchi's sponge over `permutation` in that model: [`piop_transcript_on`]
/// at the model's carrier, the commitments their chunks' coordinates. The
/// verifier of a circuit over the base field (Pickles' group half) draws
/// `beta`, `gamma`, `alpha`, `zeta` and the digest by the protocol the
/// native verifier follows; the 128-bit challenges come unmapped.
#[allow(clippy::type_complexity)]
pub fn base_transcript_in<'m, C, M, P>(
    model: &'m RefCell<M>,
    permutation: &'m P,
    inputs: &PiopInputs<M::Value, Point<M::Value>, Vec<Point<M::Value>>>,
) -> (
    PiopChallenges<M::Value>,
    FqSponge<C, Duplexed<'m, M::Value, P>, &'m RefCell<M>>,
)
where
    C: Curve,
    M: ChallengeModel<C::Base>,
    P: PermutationIn<M, C::Base, WIDTH>,
{
    piop_transcript_on(FqSponge::in_model(model, permutation), inputs)
}

/// The scalar-field half of the transcript, as `transcript::Evaluation`
/// says: the scalar sponge over the base digest, the previous accumulators'
/// challenges, `ft(zeta w)` and the evaluations; the 128-bit challenges `v`
/// (the polynomial combiner, Pickles' `xi`) and `u` (the point combiner,
/// Pickles' `r`).
#[must_use]
pub fn scalar_transcript<F, T>(
    scalar_sponge: &T,
    digest: F,
    prev_challenges: &[Vec<F>],
    ft_eval1: F,
    evals: &ProofEvaluations<F>,
) -> (ScalarChallenge<F>, ScalarChallenge<F>)
where
    F: PrimeField,
    T: Transcript<F> + Clone,
{
    let sponge = FrSponge::new(scalar_sponge.clone());
    scalar_transcript_on(&sponge, &digest, prev_challenges, &ft_eval1, evals)
}

/// [`scalar_transcript`] in a model of the field signature, on Kimchi's
/// sponge over `permutation` in that model ([`transcript::FrSpongeIn`]):
/// the same definition ([`scalar_transcript_on`]) at the model's carrier, so
/// a verifier in a circuit draws its `xi` and `r` by the protocol the native
/// verifier follows. The 128-bit challenges come unmapped.
pub fn scalar_transcript_in<F, M, P>(
    model: &mut M,
    permutation: &P,
    digest: &M::Value,
    prev_challenges: &[Vec<M::Value>],
    ft_eval1: &M::Value,
    evals: &ProofEvaluations<M::Value>,
) -> (ScalarChallenge<M::Value>, ScalarChallenge<M::Value>)
where
    F: Field,
    M: ChallengeModel<F>,
    P: PermutationIn<M, F, WIDTH>,
{
    let shared = RefCell::new(model);
    let sponge = FrSponge::in_model(&shared, permutation);
    scalar_transcript_on(&sponge, digest, prev_challenges, ft_eval1, evals)
}

/// The scalar-field half of the transcript on a fresh `sponge`, in
/// whichever model it computes: one definition for [`scalar_transcript`]
/// and [`scalar_transcript_in`]. `sponge` is copied twice, for the digest
/// of the previous challenges and for the evaluations.
#[allow(clippy::type_complexity)]
pub fn scalar_transcript_on<F, S, H>(
    sponge: &FrSponge<F, S, H>,
    digest: &Value<F, H>,
    prev_challenges: &[Vec<Value<F, H>>],
    ft_eval1: &Value<F, H>,
    evals: &ProofEvaluations<Value<F, H>>,
) -> (ScalarChallenge<Value<F, H>>, ScalarChallenge<Value<F, H>>)
where
    F: Field,
    H: SpongeIn<F, S>,
    H::Model: ChallengeModel<F>,
    S: Clone,
{
    let prev_digest =
        transcript::previous_challenges_digest_on(sponge.clone(), prev_challenges.to_vec());
    let tape = transcript::evaluation_on(sponge.clone())
        .statement(&Message::new(Scalar(digest.clone())))
        .statement(&Message::new(Scalar(prev_digest)))
        .receive(&Message::new(Scalar(ft_eval1.clone())))
        .receive(&Message::new(evals.clone()));
    let (v_chal, tape) = tape.squeeze();
    let (u_chal, tape) = tape.squeeze();
    let _ = tape.into_inner();
    (v_chal, u_chal)
}

/// The scalars of the linearization (Pickles' `derive_plonk` and
/// `ft_eval0`): what the verifier computes from the evaluations and the
/// challenges alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Linearization<F> {
    /// `ft(zeta)`, as the verifier computes it.
    pub ft_eval0: F,
    /// The scalar of the last sigma's commitment in `ft`'s: `-z(zeta w)
    /// beta alpha^21 zk(zeta) prod_i (gamma + beta s_i(zeta) + w_i(zeta))`.
    pub perm: F,
    /// `zeta^n`, the domain's size `n`.
    pub zeta_to_domain_size: F,
    /// `zeta^m`, the chunk size `m` (the SRS's length, for Pickles).
    pub zeta_to_srs_length: F,
}

/// The linearization of `evals` at `challenges` against `index`, the gates'
/// constraints given by `gates` (see the module documentation).
///
/// # Panics
///
/// If the index's domain is not one of the field's.
pub fn linearization<C, G>(
    index: &VerifierIndex<C>,
    evals: &ProofEvaluations<C::Scalar>,
    lookup: Option<&LookupEvaluations<C::Scalar>>,
    challenges: &Challenges<C::Scalar>,
    gates: &G,
) -> Linearization<C::Scalar>
where
    C: Curve,
    C::Scalar: TwoAdicField,
    G: Gates<C::Scalar>,
{
    let Challenges {
        alpha,
        beta,
        gamma,
        zeta,
        joint_combiner,
    } = *challenges;
    let d1 = domain(index);
    let n = d1.size();
    let m = index.max_poly_size;
    let e = evals;
    let zeta1 = zeta.pow(n as u64);
    let zetaw = zeta * d1.generator();
    let zeta_m = zeta.pow(m as u64);
    let zetaw_m = zetaw.pow(m as u64);

    // The evaluations, chunks combined.
    let at = |p: &PointEvaluations<C::Scalar>| {
        (combine(&p.zeta, zeta_m), combine(&p.zeta_omega, zetaw_m))
    };
    let lookup = index.lookup_index.as_ref().zip(lookup);
    let lookup_count = lookup.map_or(0, |(li, _)| constraint_count(&li.lookup_info));
    let mut alphas = vec![C::Scalar::ONE; LOOKUP_ALPHA + lookup_count];
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
    // The lookup constraints at the evaluations.
    if let (Some((li, l)), Some(jc)) = (lookup, joint_combiner) {
        let point = LookupPoint {
            witness: e.w.iter().map(|p| at(p).0).collect(),
            sorted: l.sorted.iter().map(at).collect(),
            aggregation: at(&l.aggregation),
            table: at(&l.table),
            selectors: l.selectors.iter().map(|(p, v)| (*p, at(v).0)).collect(),
            runtime: match (&l.runtime_table, &l.runtime_selector) {
                (Some(t), Some(sel)) => Some((at(t).0, at(sel).0)),
                _ => None,
            },
            vanishing: (n - zk_rows - 1..n)
                .fold(C::Scalar::ONE, |acc, i| acc * (zeta - d1.element(i))),
            first: unnormalized_lagrange(zeta, C::Scalar::ONE, n),
            last: unnormalized_lagrange(zeta, d1.element(n - zk_rows - 1), n),
        };
        let values = lookup_constraints(&li.lookup_info, &point, beta, gamma, jc);
        for (i, c) in values.iter().enumerate() {
            ft_eval0 = ft_eval0 - alphas[LOOKUP_ALPHA + i] * *c;
        }
    }

    // The scalar of the last sigma in ft's commitment.
    let mut perm = z.1 * beta * alpha0 * vanishing;
    for i in 0..PERMUTS - 1 {
        perm = perm * (gamma + beta * at(&e.s[i]).0 + at(&e.w[i]).0);
    }
    Linearization {
        ft_eval0,
        perm: -perm,
        zeta_to_domain_size: zeta1,
        zeta_to_srs_length: zeta_m,
    }
}

/// `ft`'s commitment from the scalars of the linearization: `perm
/// sigma_6(zeta^m) - (zeta^n - 1) t(zeta^m)`, each chunked commitment
/// combined by powers of `zeta^m` (Pickles' `ft_comm`).
#[must_use]
pub fn ft_comm<C>(
    index: &VerifierIndex<C>,
    t_comm: &PolyComm<C>,
    perm: C::Scalar,
    zeta_to_srs_length: C::Scalar,
    zeta_to_domain_size: C::Scalar,
) -> Projective<C>
where
    C: Curve,
{
    let sigma = chunk_commitment(&index.sigma_comm[PERMUTS - 1], zeta_to_srs_length);
    let t = chunk_commitment(t_comm, zeta_to_srs_length);
    sigma * perm - t * (zeta_to_domain_size - C::Scalar::ONE)
}

/// The commitments of the opening, in its order: the previous
/// accumulators' `sg`, the public commitment, `ft`'s, `z`, the six
/// selectors, the witness, the coefficients, the sigmas but the last, the
/// optional selectors, then with lookups the sorted columns, the
/// aggregation, the joint table (Kimchi's `combine_table`), the runtime
/// table and its selector, and the lookup selectors.
#[must_use]
pub fn commitments<C>(
    index: &VerifierIndex<C>,
    proof: &ProverProof<C>,
    public_comm: &PolyComm<C>,
    ft_comm: Projective<C>,
    joint_combiner: Option<C::Scalar>,
) -> Vec<PolyComm<C>>
where
    C: Curve,
{
    let one = |p: Projective<C>| PolyComm { chunks: vec![p] };
    let mut comms: Vec<PolyComm<C>> = proof.prev_challenges.iter().map(|p| one(p.comm)).collect();
    comms.push(public_comm.clone());
    comms.push(one(ft_comm));
    comms.push(proof.z_comm.clone());
    for c in [
        &index.generic_comm,
        &index.psm_comm,
        &index.complete_add_comm,
        &index.mul_comm,
        &index.emul_comm,
        &index.endomul_scalar_comm,
    ] {
        comms.push(c.clone());
    }
    comms.extend(proof.w_comm.iter().cloned());
    comms.extend(index.coefficients_comm.iter().cloned());
    comms.extend(index.sigma_comm[..PERMUTS - 1].iter().cloned());
    for (_, c) in optional_selectors(index) {
        if let Some(c) = c {
            comms.push(c.clone());
        }
    }
    if let (Some(li), Some(lc), Some(jc)) = (&index.lookup_index, &proof.lookup, joint_combiner) {
        comms.extend(lc.sorted.iter().cloned());
        comms.push(lc.aggreg.clone());
        // Kimchi's combine_table: columns by powers of jc, the ids by
        // jc^max_joint_size, the runtime table as the second column.
        let tid = jc.pow(li.lookup_info.max_joint_size as u64);
        let n = li
            .lookup_table
            .first()
            .map_or(chunks(index), |c| c.chunks.len());
        let table = PolyComm {
            chunks: (0..n)
                .map(|k| {
                    let mut acc = Projective::<C>::identity();
                    let mut power = C::Scalar::ONE;
                    for column in &li.lookup_table {
                        acc = acc + column.chunks[k] * power;
                        power = power * jc;
                    }
                    if let Some(ids) = &li.table_ids {
                        acc = acc + ids.chunks[k] * tid;
                    }
                    if let Some(r) = &lc.runtime {
                        acc = acc + r.chunks[k] * jc;
                    }
                    acc
                })
                .collect(),
        };
        comms.push(table);
        if let Some(c) = &lc.runtime {
            comms.push(c.clone());
        }
        if let Some(c) = &li.runtime_tables_selector {
            comms.push(c.clone());
        }
        comms.extend(li.lookup_selectors.iter().flatten().cloned());
    }
    comms
}

/// The evaluations of the opening, in the order of [`commitments`]: each
/// polynomial's at `zeta` and at `zeta w`, one per chunk; the previous
/// accumulators' are their challenge polynomials' at the two points, and
/// `ft`'s are `ft_evals`.
#[must_use]
pub fn evaluations<F>(
    evals: &ProofEvaluations<F>,
    prev_challenges: &[Vec<F>],
    ft_evals: [F; 2],
    points: [F; 2],
) -> Vec<[Vec<F>; 2]>
where
    F: Field,
{
    let e = evals;
    let mut out: Vec<[Vec<F>; 2]> = prev_challenges
        .iter()
        .map(|chals| {
            [
                vec![b_poly(chals, points[0])],
                vec![b_poly(chals, points[1])],
            ]
        })
        .collect();
    out.push(pair(&e.public));
    out.push([vec![ft_evals[0]], vec![ft_evals[1]]]);
    out.push(pair(&e.z));
    for p in [
        &e.generic_selector,
        &e.poseidon_selector,
        &e.complete_add_selector,
        &e.mul_selector,
        &e.emul_selector,
        &e.endomul_scalar_selector,
    ] {
        out.push(pair(p));
    }
    out.extend(e.w.iter().map(pair));
    out.extend(e.coefficients.iter().map(pair));
    out.extend(e.s.iter().map(pair));
    out.extend(e.optional_selectors.iter().map(|(_, p)| pair(p)));
    if let Some(l) = &e.lookup {
        out.extend(l.sorted.iter().map(pair));
        out.push(pair(&l.aggregation));
        out.push(pair(&l.table));
        for p in [&l.runtime_table, &l.runtime_selector]
            .into_iter()
            .flatten()
        {
            out.push(pair(p));
        }
        out.extend(l.selectors.iter().map(|(_, p)| pair(p)));
    }
    out
}

/// Kimchi's `combined_inner_product`: chunk `i` of each polynomial at the
/// next power of `xi`, its two points combined by `r`.
#[must_use]
pub fn combined_inner_product<F>(evaluations: &[[Vec<F>; 2]], xi: F, r: F) -> F
where
    F: Field,
{
    let mut combined = F::ZERO;
    let mut scale = F::ONE;
    for [at_zeta, at_zetaw] in evaluations {
        for (a, b) in at_zeta.iter().zip(at_zetaw) {
            combined = combined + scale * (*a + r * *b);
            scale = scale * xi;
        }
    }
    combined
}

/// `sum_i xi^i c_i` over every chunk of `commitments`: the commitment of
/// the polynomial the opening proves, in the order of its chunks.
#[must_use]
pub fn combined_commitment<C>(commitments: &[PolyComm<C>], xi: C::Scalar) -> Projective<C>
where
    C: Curve,
{
    let mut combined = Projective::<C>::identity();
    let mut scale = C::Scalar::ONE;
    for comm in commitments {
        for chunk in &comm.chunks {
            combined = combined + *chunk * scale;
            scale = scale * xi;
        }
    }
    combined
}

/// The transcript of `proof` against `index` and `public`, the gates'
/// constraints given by `gates` (see the module documentation): the
/// base-field half ([`base_transcript`]), then the scalar-field half
/// ([`scalar_transcript`]) and the linearization.
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
    let base = base_transcript(index, srs, proof, public, base_sponge)?;
    let lookup = evaluation_shape(index, &proof.evals)?;
    let prev: Vec<Vec<C::Scalar>> = proof
        .prev_challenges
        .iter()
        .map(|p| p.chals.clone())
        .collect();
    let (v_chal, u_chal) = scalar_transcript(
        scalar_sponge,
        base.digest,
        &prev,
        proof.ft_eval1,
        &proof.evals,
    );
    let lambda = C::lambda();
    let (v, u) = (v_chal.to_field(lambda), u_chal.to_field(lambda));
    let challenges = base.challenges();
    let lin = linearization(index, &proof.evals, lookup, &challenges, gates);
    let ft = ft_comm(
        index,
        &proof.t_comm,
        lin.perm,
        lin.zeta_to_srs_length,
        lin.zeta_to_domain_size,
    );
    let zeta = challenges.zeta;
    let zetaw = zeta * domain(index).generator();
    let comms = commitments(
        index,
        proof,
        &base.public_comm,
        ft,
        challenges.joint_combiner,
    );
    let evals = evaluations(
        &proof.evals,
        &prev,
        [lin.ft_eval0, proof.ft_eval1],
        [zeta, zetaw],
    );
    let combined = combined_inner_product(&evals, v, u);
    Ok(Oracles {
        beta: base.beta,
        gamma: base.gamma,
        alpha_chal: base.alpha_chal,
        alpha: challenges.alpha,
        zeta_chal: base.zeta_chal,
        zeta,
        zeta_omega: zetaw,
        v_chal,
        v,
        u_chal,
        u,
        ft_eval0: lin.ft_eval0,
        combined_inner_product: combined,
        digest: base.digest,
        public_comm: base.public_comm,
        fq_sponge: base.fq_sponge,
        evaluations: comms.into_iter().zip(evals).collect(),
    })
}

/// The opening, checked up to the commitment of `b`: the round challenges,
/// and the IPA's Schnorr-like expression, the identity exactly when that
/// part of the check holds.
fn opening_check<C, S>(
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    oracles: &Oracles<C, S>,
) -> Result<(Vec<C::Scalar>, Projective<C>), VerifyError>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    S: Transcript<C::Base> + Clone,
{
    let size = srs.g.len();
    if !size.is_power_of_two() || proof.proof.lr.len() != size.trailing_zeros() as usize {
        return Err(VerifyError::IncorrectRounds);
    }
    // C: every chunk at the next power of v.
    let comms: Vec<PolyComm<C>> = oracles.evaluations.iter().map(|(c, _)| c.clone()).collect();
    let combined = combined_commitment(&comms, oracles.v);
    let points = [oracles.zeta, oracles.zeta_omega];
    let statement = Statement {
        commitment: &combined,
        points: &points,
        evalscale: oracles.u,
        value: oracles.combined_inner_product,
    };
    // Kimchi's IPA (crate::ipa), continuing the base sponge after zeta.
    let mut sponge = oracles.fq_sponge.clone();
    let (accumulator, ipa) = kimchi_ipa(srs)
        .check(&mut sponge, &statement, &(&proof.proof).into())
        .ok_or(VerifyError::OpeningProof)?;
    let chals = accumulator.chals;
    Ok((chals, ipa))
}

/// Kimchi's verifier without its decider: everything but the MSM `<s, G>
/// = sg` of the size of the SRS. It replays the transcript and checks the
/// IPA's Schnorr-like equation, then outputs the claim it leaves, the
/// accumulator `(u, sg)`: "`sg` commits to `b_u`", which Halo decides once
/// for many proofs ([`crate::accumulation`]).
///
/// # Errors
///
/// A [`VerifyError`]; [`VerifyError::OpeningProof`] when the equation
/// fails.
pub fn succinct<C, S, T, G>(
    index: &VerifierIndex<C>,
    srs: &Srs<C>,
    proof: &ProverProof<C>,
    public: &[C::Scalar],
    gates: &G,
    sponges: (&S, &T),
) -> Result<RecursionChallenge<C>, VerifyError>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
{
    let oracles = oracles(index, srs, proof, public, gates, sponges)?;
    let (chals, ipa) = opening_check(srs, proof, &oracles)?;
    if ipa.is_identity() {
        Ok(RecursionChallenge {
            chals,
            comm: proof.proof.sg,
        })
    } else {
        Err(VerifyError::OpeningProof)
    }
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
    let (chals, ipa) = opening_check(srs, proof, &oracles)?;
    let s = b_poly_coefficients(&chals);
    let rand_base: C::Scalar = rng.sample();
    let sg_rand_base: C::Scalar = rng.sample();
    let sg_check = msm(&srs.g, &s) - proof.proof.sg;
    let total = ipa * rand_base + sg_check * sg_rand_base;
    if total.is_identity() {
        Ok(())
    } else {
        Err(VerifyError::OpeningProof)
    }
}
