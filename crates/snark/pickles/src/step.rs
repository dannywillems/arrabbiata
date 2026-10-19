//! Step: a proof over Vesta of an application rule, verifying a wrap proof
//! over Pallas (Pickles' `step_main`, for a rule with one predecessor whose
//! own step proof is a base case).
//!
//! The wrap proof's verifier splits across the cycle as the step proof's
//! did ([`crate::wrap`]), the other way round. Its group half is native in
//! the step circuit's field `Fp` (the wrap proof's commitments are points
//! of Pallas, coordinates in `Fp`), with its scalars of `Fq` held as Type2
//! shifted values; its scalar half is native in `Fq`, so it is deferred
//! again, to the next wrap circuit or to the verifier ([`finalize`]). What
//! the step circuit can finalize is the wrap statement itself: its
//! deferred values are those of the base step proof, scalars of `Fp`, so
//! the scalar half of that step proof runs here, natively.
//!
//! So the step circuit, on the wrap statement's words (private):
//!
//! - runs the wrap proof's group half, `x_hat` the action of those words,
//!   each lifted canonically under its sort's bound ([`wrap::bounds`]);
//! - finalizes the wrap statement: the base step proof's scalar half on
//!   its evaluations, each deferred value its definition;
//! - checks that the wrap statement's messages for the next step proof are
//!   the hash of the previous application state and the wrap key, applies
//!   the rule (the state plus one), and hashes the new state with the wrap
//!   proof's accumulator into the new messages;
//! - takes the wrap key as a witness: the wrap proof's transcript starts
//!   from the key's digest, computed in the circuit, and the key is hashed
//!   into both messages, so the step circuit does not depend on the wrap
//!   circuit and the verifier pins the key by checking the messages
//!   ([`finalize`]);
//! - leaves the wrap proof unfinalized in its own statement.
//!
//! The statement is Pickles' `Types.Step.Statement` for one predecessor,
//! in the order of its `spec`: the unfinalized proof (five Type2 fields,
//! the sponge digest, two challenges, three scalar challenges, the
//! prechallenges, `should_finalize`), the messages for the next step proof,
//! the predecessor's messages for the next wrap proof.
//!
//! Deviations from Mina: the wrap key's shape (its chunks, which of them
//! are the identity, the wrap domain and public input) is the circuit's,
//! from a reference index, while Mina's step circuits take keys of any
//! domain among the branches'; the step proof
//! accumulates the base step proof's accumulator (its `sg`, from the wrap
//! statement's messages, and its challenges, the wrap statement's deferred
//! ones) as Kimchi's previous challenge, as Mina's step prover does, while
//! the wrap proof's accumulator is decided by [`finalize`] when no wrap
//! follows; the wrap key's commitments are hashed in the order of
//! `index_to_field_elements` as read, not yet checked against vectors.
//!
//! The accumulators cross over between the levels (Mina's
//! `Per_proof_witness`): the step messages hold the `sg` of the wrap proof
//! verified and the challenges of the step proof it wrapped; the wrap
//! messages hold the `sg` of the step proof wrapped and the challenges of
//! the wrap proofs it verified. Each proof's previous challenges zip them
//! back into accumulators of its own curve.

use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use arrabbiata_curve::{Projective, glv::Glv, msm};
use arrabbiata_field::{Field, FieldModel, Native};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_ipa::{RecursionChallenge, b_poly_coefficients};
use arrabbiata_kimchi::{
    compiled,
    constraints::{COLUMNS, ConstraintSystem},
    deferred::expand_deferred,
    oracle::{BaseReading, ChallengeModel, FrSponge, ScalarChallenge, coordinates},
    prover::{ProofEvaluations, ProverIndex, ProverProof, prove},
    rng::Sampler,
    srs::Srs,
    verifier::{VerifyError, verify},
    verifier_index::VerifierIndex,
};
use arrabbiata_kimchi_gadgets::generic::PublicInput;
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, KimchiSponge, pallas_base, vesta_base};

use crate::{
    curve::{Shifted, held, point_at},
    group::{
        Constants, DeferredIn, KeyIn, ProofIn, PublicWord, group_half_in, group_half_with_key_in,
        key_digest_in,
    },
    model::{Affine, Circuit, load},
    scalar::{self, Inputs, scalar_half},
    statement::{Bool, Challenge, Digest, Sort, Type2, WINDOW_BITS, Word},
    wrap::{self, StepEvaluations, step_gates, wrap_gates},
};

/// The column variables of the compiled step circuit.
pub const VARIABLES: usize = 64;

/// The degree of the compiled step circuit.
pub const DEGREE: usize = 7;

/// A wrap proof left for the next wrap circuit to finalize: its deferred
/// values, scalars of `Fq` as Type2 words `[high, low]`, and its draws
/// (Pickles' `Per_proof.In_circuit`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unfinalized<V> {
    /// The combined inner product.
    pub combined_inner_product: [V; 2],
    /// `b`.
    pub b: [V; 2],
    /// `zeta^m`.
    pub zeta_to_srs_length: [V; 2],
    /// `zeta^n`.
    pub zeta_to_domain_size: [V; 2],
    /// `perm`.
    pub perm: [V; 2],
    /// The base sponge's digest before the evaluations.
    pub sponge_digest_before_evaluations: V,
    /// `beta`.
    pub beta: V,
    /// `gamma`.
    pub gamma: V,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha: V,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta: V,
    /// The 128-bit challenge `xi` comes from.
    pub xi: V,
    /// The IPA's round challenges, unmapped.
    pub bulletproof_challenges: Vec<V>,
    /// Whether the next wrap circuit finalizes it: one, for a real proof.
    pub should_finalize: V,
}

/// What the step proof proves, in Pickles' order, for one predecessor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement<V> {
    /// The wrap proof verified, unfinalized.
    pub unfinalized: Unfinalized<V>,
    /// The digest of the messages for the next step proof
    /// ([`MessagesForNextStepProof`]).
    pub messages_for_next_step_proof: V,
    /// The predecessor's digest of the messages for the next wrap proof,
    /// passed along.
    pub messages_for_next_wrap_proof: V,
}

/// The values the statement holds besides the prechallenges.
const FIXED: usize = 10 + 1 + 5 + 1 + 2;

impl<V: Clone> Statement<V> {
    /// The statement as the step proof's public input, in order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<V> {
        let u = &self.unfinalized;
        let mut out = Vec::new();
        for f in [
            &u.combined_inner_product,
            &u.b,
            &u.zeta_to_srs_length,
            &u.zeta_to_domain_size,
            &u.perm,
        ] {
            out.extend(f.iter().cloned());
        }
        out.extend([
            u.sponge_digest_before_evaluations.clone(),
            u.beta.clone(),
            u.gamma.clone(),
            u.alpha.clone(),
            u.zeta.clone(),
            u.xi.clone(),
        ]);
        out.extend(u.bulletproof_challenges.iter().cloned());
        out.extend([
            u.should_finalize.clone(),
            self.messages_for_next_step_proof.clone(),
            self.messages_for_next_wrap_proof.clone(),
        ]);
        out
    }

    /// The statement of `values`, [`Statement::to_vec`]'s inverse for a
    /// wrap proof of `rounds` rounds; `None` for another length.
    #[must_use]
    pub fn of(rounds: usize, values: &[V]) -> Option<Self> {
        if values.len() != FIXED + rounds {
            return None;
        }
        let v = |i: usize| values[i].clone();
        let pair = |i: usize| [v(i), v(i + 1)];
        Some(Self {
            unfinalized: Unfinalized {
                combined_inner_product: pair(0),
                b: pair(2),
                zeta_to_srs_length: pair(4),
                zeta_to_domain_size: pair(6),
                perm: pair(8),
                sponge_digest_before_evaluations: v(10),
                beta: v(11),
                gamma: v(12),
                alpha: v(13),
                zeta: v(14),
                xi: v(15),
                bulletproof_challenges: values[16..16 + rounds].to_vec(),
                should_finalize: v(16 + rounds),
            },
            messages_for_next_step_proof: v(17 + rounds),
            messages_for_next_wrap_proof: v(18 + rounds),
        })
    }
}

/// The bound of each word of a statement of `rounds` rounds, in order: the
/// sorts' ([`Sort::BOUNDS`]).
#[must_use]
pub fn bounds(rounds: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..5 {
        out.extend(Type2::<Pallas>::BOUNDS);
    }
    out.extend(Digest::BOUNDS);
    for _ in 0..5 + rounds {
        out.extend(Challenge::BOUNDS);
    }
    out.extend(Bool::BOUNDS);
    for _ in 0..2 {
        out.extend(Digest::BOUNDS);
    }
    out
}

/// The wrap key as the elements the messages hash: the coordinates of its
/// commitments, permutation, coefficients, then the selectors (Pickles'
/// `index_to_field_elements`, as read).
#[must_use]
pub fn index_elements(index: &VerifierIndex<Pallas>) -> Vec<Fp> {
    let mut comms: Vec<&Projective<Pallas>> = Vec::new();
    for c in index.sigma_comm.iter().chain(&index.coefficients_comm) {
        comms.extend(&c.chunks);
    }
    for c in [
        &index.generic_comm,
        &index.psm_comm,
        &index.complete_add_comm,
        &index.mul_comm,
        &index.emul_comm,
        &index.endomul_scalar_comm,
    ] {
        comms.extend(&c.chunks);
    }
    comms
        .into_iter()
        .flat_map(|p| {
            let (x, y) = coordinates(p);
            [x, y]
        })
        .collect()
}

/// What the next step proof needs of this one, hashed into the statement:
/// the application state, the wrap key, the `sg` of each wrap proof
/// verified, and the challenges each wrap statement deferred, those of the
/// step proof it wrapped, mapped into `Fp` (Pickles'
/// `Messages_for_next_step_proof`, prepared by `Ipa.Step.compute_challenges`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessagesForNextStepProof {
    /// The wrap key ([`index_elements`]).
    pub index: Vec<Fp>,
    /// The application state.
    pub app_state: Vec<Fp>,
    /// The wrap proofs' accumulator commitments, as coordinates.
    pub challenge_polynomial_commitments: Vec<[Fp; 2]>,
    /// The challenges the wrap statements deferred, mapped.
    pub old_bulletproof_challenges: Vec<Vec<Fp>>,
}

impl MessagesForNextStepProof {
    /// The elements hashed: the key, the state, then each commitment's
    /// coordinates followed by its challenges (Pickles' `to_field_elements`).
    #[must_use]
    pub fn field_elements(&self) -> Vec<Fp> {
        let mut out = self.index.clone();
        out.extend(&self.app_state);
        for (comm, chals) in self
            .challenge_polynomial_commitments
            .iter()
            .zip(&self.old_bulletproof_challenges)
        {
            out.extend(comm);
            out.extend(chals);
        }
        out
    }

    /// The digest: Kimchi's sponge over `Fp`, absorbing the elements, then
    /// one squeeze (Pickles' `Tick_field_sponge.digest`).
    #[must_use]
    pub fn digest(&self) -> Fp {
        let mut sponge = FrSponge::new(KimchiSponge::new(pallas_base()));
        sponge.absorb_multiple(&self.field_elements());
        sponge.digest()
    }
}

/// The application rule: the next state of a state.
#[must_use]
pub fn rule(state: Fp) -> Fp {
    state + Fp::ONE
}

/// The shape of the proofs a step circuit verifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The base step proof's IPA rounds (the wrap statement's prechallenges).
    pub step_rounds: usize,
    /// The chunks of a wrap proof's commitment.
    pub chunks: usize,
    /// The chunks of a wrap proof's quotient commitment.
    pub quotient_chunks: usize,
    /// The wrap proof's IPA rounds.
    pub wrap_rounds: usize,
    /// The base step proof's evaluations.
    pub evaluations: usize,
    /// The wrap key's chunks that are not the identity.
    pub key_points: usize,
}

impl Shape {
    /// The size of the statement.
    #[must_use]
    pub const fn statement(&self) -> usize {
        FIXED + self.wrap_rounds
    }

    /// The size of the private input ([`statement`]'s): the wrap key's
    /// points, the wrap statement, the wrap proof's points (as coordinates)
    /// and `z1`, `z2` as Type2, the base step proof's `ft(zeta w)` and
    /// evaluations, the previous state.
    #[must_use]
    pub const fn witness(&self) -> usize {
        let points =
            COLUMNS * self.chunks + self.chunks + self.quotient_chunks + 2 * self.wrap_rounds + 2;
        2 * self.key_points + (14 + self.step_rounds) + 2 * points + 4 + 1 + self.evaluations + 1
    }
}

/// Why a step proof cannot be made, or does not verify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The wrap proof does not verify: its transcript has the wrong shape.
    Wrap(VerifyError),
    /// The wrap proof's opening does not hold.
    Opening,
    /// An index has lookups, which the circuits do not cover.
    Lookups,
    /// The step circuit's sizes are not the shape's.
    Size,
    /// A value of a statement is past its sort's bound.
    Bound,
    /// The wrap statement's deferred values are not their definitions.
    Deferred(&'static str),
    /// The messages are not those a digest hashes.
    Messages,
    /// The step circuit does not compile to Kimchi rows.
    Circuit,
    /// The step proof could not be made.
    Prove,
    /// The step proof does not verify against the statement.
    Step(VerifyError),
    /// The accumulator's commitment is not the challenge polynomial's.
    Accumulator,
}

/// A wrap proof, with what verifying it needs.
#[derive(Clone, Copy, Debug)]
pub struct WrapProof<'a> {
    /// The wrap index.
    pub index: &'a VerifierIndex<Pallas>,
    /// The wrap SRS.
    pub srs: &'a Srs<Pallas>,
    /// The proof.
    pub proof: &'a ProverProof<Pallas>,
    /// Its statement.
    pub statement: &'a wrap::Statement<Fq>,
    /// The messages its statement's digest hashes: the accumulator of the
    /// step proof it wrapped.
    pub messages: &'a wrap::MessagesForNextWrapProof,
}

/// The challenges a wrap statement deferred, mapped into `Fp` by the
/// endomorphism of Vesta: those of the step proof it wrapped.
#[must_use]
pub fn deferred_challenges(statement: &wrap::Statement<Fq>) -> Vec<Fp> {
    let lambda = Vesta::lambda();
    statement
        .bulletproof_challenges
        .iter()
        .map(|c| ScalarChallenge(to_fp(*c)).to_field(lambda))
        .collect()
}

/// What finalizing a wrap proof needs beyond the statement: its
/// evaluations and `ft(zeta w)`.
#[derive(Clone, Debug)]
pub struct WrapEvaluations {
    /// The wrap proof's evaluations.
    pub evals: ProofEvaluations<Fq>,
    /// `ft(zeta w)`.
    pub ft_eval1: Fq,
}

impl WrapEvaluations {
    /// The evaluations `proof` carries.
    #[must_use]
    pub fn of(proof: &ProverProof<Pallas>) -> Self {
        Self {
            evals: proof.evals.clone(),
            ft_eval1: proof.ft_eval1,
        }
    }
}

/// A value of `Fq` below `2^254` as the element of `Fp` with its integer.
fn to_fp(x: Fq) -> Fp {
    Word::of(x).reduce()
}

/// A value of `Fp` below `2^254` as the element of `Fq` with its integer.
fn to_fq(x: Fp) -> Fq {
    Word::of(x).reduce()
}

/// The Type2 words of a scalar of `Fq`, in `Fp`.
fn type2(s: Fq) -> [Fp; 2] {
    let words = Type2::<Pallas>::encode(&s);
    [words[0].reduce(), words[1].reduce()]
}

/// `2^255 + 1` in `Fp`: Type1's shift.
fn type1_shift() -> Fp {
    (Fp::ONE + Fp::ONE).pow(255) + Fp::ONE
}

/// The constants of the wrap proof's group half.
fn wrap_constants(wrap: &WrapProof<'_>) -> Result<Constants<Pallas>, Error> {
    let base = KimchiSponge::new(pallas_base());
    Constants::new(wrap.index, wrap.srs, &base).ok_or(Error::Lookups)
}

/// The step statement of verifying `wrap`, whose step proof has the
/// evaluations `evaluations`, from the application state `state`; the
/// messages its digest hashes; the private input; the shape; and the
/// previous challenge the step proof carries, the wrapped step proof's
/// accumulator.
///
/// # Errors
///
/// [`Error::Wrap`] for a wrap proof of the wrong shape, [`Error::Opening`]
/// when its opening does not hold, [`Error::Lookups`] for an index with
/// lookups, [`Error::Messages`] when the wrap statement's messages are not
/// those of `state` or its digest not that of its messages,
/// [`Error::Bound`] for a value past its bound.
#[allow(clippy::type_complexity)]
pub fn statement(
    wrap: &WrapProof<'_>,
    evaluations: &StepEvaluations,
    state: Fp,
) -> Result<
    (
        Statement<Fp>,
        MessagesForNextStepProof,
        Vec<Fp>,
        Shape,
        RecursionChallenge<Vesta>,
    ),
    Error,
> {
    if wrap.messages.digest() != wrap.statement.messages_for_next_wrap_proof {
        return Err(Error::Messages);
    }
    let [x, y] = wrap.messages.challenge_polynomial_commitment;
    let previous_challenge = RecursionChallenge {
        chals: deferred_challenges(wrap.statement),
        comm: point_at::<Vesta>(x, y).ok_or(Error::Messages)?,
    };
    let wrap_public = wrap.statement.to_vec();
    let step_rounds = wrap.statement.bulletproof_challenges.len();
    let base_sponge = KimchiSponge::new(pallas_base());
    let scalar_sponge = KimchiSponge::new(vesta_base());
    let values = expand_deferred(
        wrap.index,
        wrap.srs,
        wrap.proof,
        &wrap_public,
        &wrap_gates(),
        (&base_sponge, &scalar_sponge),
    )
    .map_err(Error::Wrap)?;
    let constants = wrap_constants(wrap)?;
    let key_points = constants.key.chunks().flatten().count();
    let opening = &wrap.proof.proof;
    let proof = ProofIn {
        previous: wrap.proof.prev_challenges.iter().map(|p| p.comm).collect(),
        witness: core::array::from_fn(|i| wrap.proof.w_comm[i].chunks.clone()),
        z: wrap.proof.z_comm.chunks.clone(),
        t: wrap.proof.t_comm.chunks.clone(),
        rounds: opening.lr.clone(),
        delta: opening.delta,
        z1: opening.z1,
        z2: opening.z2,
        sg: opening.sg,
    };
    if !proof.previous.is_empty() {
        return Err(Error::Size);
    }
    let deferred = DeferredIn {
        public: wrap_public
            .iter()
            .zip(wrap::bounds(step_rounds))
            .map(|(w, bound)| PublicWord {
                value: to_fp(*w),
                bound,
            })
            .collect(),
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: to_fp(values.xi.0),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };
    let poseidon = pallas_base();
    let model = RefCell::new(Native);
    let out = group_half_in::<Pallas, _>(&model, &poseidon, &constants, &proof, &deferred);
    if out.left != out.right {
        return Err(Error::Opening);
    }

    let index = index_elements(wrap.index);
    let previous = MessagesForNextStepProof {
        index: index.clone(),
        app_state: vec![state],
        challenge_polynomial_commitments: vec![],
        old_bulletproof_challenges: vec![],
    };
    if to_fq(previous.digest()) != wrap.statement.messages_for_next_step_proof {
        return Err(Error::Messages);
    }
    let (sx, sy) = coordinates(&out.sg);
    let messages = MessagesForNextStepProof {
        index,
        app_state: vec![rule(state)],
        challenge_polynomial_commitments: vec![[sx, sy]],
        old_bulletproof_challenges: vec![previous_challenge.chals.clone()],
    };
    let statement = Statement {
        unfinalized: Unfinalized {
            combined_inner_product: type2(deferred.combined_inner_product),
            b: type2(deferred.b),
            zeta_to_srs_length: type2(deferred.zeta_to_srs_length),
            zeta_to_domain_size: type2(deferred.zeta_to_domain_size),
            perm: type2(deferred.perm),
            sponge_digest_before_evaluations: out.drawn.digest,
            beta: out.drawn.beta,
            gamma: out.drawn.gamma,
            alpha: out.drawn.alpha,
            zeta: out.drawn.zeta,
            xi: deferred.xi,
            bulletproof_challenges: out.prechallenges.clone(),
            should_finalize: Fp::ONE,
        },
        messages_for_next_step_proof: messages.digest(),
        messages_for_next_wrap_proof: to_fp(wrap.statement.messages_for_next_wrap_proof),
    };
    let shape = Shape {
        step_rounds,
        chunks: proof.z.len(),
        quotient_chunks: proof.t.len(),
        wrap_rounds: proof.rounds.len(),
        evaluations: {
            let mut n = 0;
            let _ = evaluations.evals.map(&mut |_| n += 1);
            n
        },
        key_points,
    };
    let in_bounds = statement
        .to_vec()
        .iter()
        .zip(bounds(shape.wrap_rounds))
        .all(|(v, n)| Word::of(*v).below(n));
    if !in_bounds {
        return Err(Error::Bound);
    }

    let mut private: Vec<Fp> = Vec::new();
    for p in constants.key.chunks().flatten() {
        let (x, y) = coordinates(p);
        private.extend([x, y]);
    }
    private.extend(deferred.public.iter().map(|w| w.value));
    let mut points: Vec<&Projective<Pallas>> = Vec::new();
    for w in &proof.witness {
        points.extend(w);
    }
    points.extend(&proof.z);
    points.extend(&proof.t);
    for (l, r) in &proof.rounds {
        points.push(l);
        points.push(r);
    }
    points.push(&proof.delta);
    points.push(&proof.sg);
    for p in points {
        let (x, y) = coordinates(p);
        private.extend([x, y]);
    }
    private.extend(held::<Pallas>(proof.z1));
    private.extend(held::<Pallas>(proof.z2));
    private.push(evaluations.ft_eval1);
    let _ = evaluations.evals.map(&mut |v| private.push(*v));
    private.push(state);
    Ok((statement, messages, private, shape, previous_challenge))
}

/// The step circuit for wrap proofs of `shape`, against a wrap key it takes
/// as a witness, whose step proofs are of one base index: its input the
/// statement (`S` values, the public input) and the private input (`W`
/// values).
#[derive(Clone, Debug)]
pub struct Step<'a, const S: usize, const W: usize> {
    wrap_constants: Constants<Pallas>,
    base_constants: scalar::Constants<Fp>,
    branch_data: Fp,
    poseidon: &'a KimchiPoseidon<Fp>,
    gates: arrabbiata_kimchi::gates::KimchiGates<Fp>,
    shape: Shape,
}

impl<'a, const S: usize, const W: usize> Step<'a, S, W> {
    /// The step circuit verifying wrap proofs on `wrap_srs` of step proofs
    /// on `base`, with Kimchi's `poseidon` over `Fp`. Of `wrap_index` only
    /// the shape is read (its domain, public input and which key chunks are
    /// the identity): the key itself is a witness.
    ///
    /// # Errors
    ///
    /// [`Error::Lookups`] for an index with lookups, [`Error::Size`] when
    /// `S` or `W` is not the shape's.
    pub fn new(
        wrap_index: &VerifierIndex<Pallas>,
        wrap_srs: &Srs<Pallas>,
        base: &VerifierIndex<Vesta>,
        poseidon: &'a KimchiPoseidon<Fp>,
        shape: Shape,
    ) -> Result<Self, Error> {
        if shape.statement() != S || shape.witness() != W {
            return Err(Error::Size);
        }
        let base_sponge = KimchiSponge::new(pallas_base());
        let wrap_constants =
            Constants::new(wrap_index, wrap_srs, &base_sponge).ok_or(Error::Lookups)?;
        let base_constants = scalar::Constants::of(base).map_err(|_| Error::Lookups)?;
        let log2 = u8::try_from(base.domain_size.trailing_zeros()).unwrap_or(u8::MAX);
        let branch =
            crate::statement::BranchData::encode(&(crate::statement::ProofsVerified::N0, log2));
        Ok(Self {
            wrap_constants,
            base_constants,
            branch_data: branch[0].reduce(),
            poseidon,
            gates: step_gates(),
            shape,
        })
    }
}

impl<const S: usize, const W: usize> TypedGadget<Fp> for Step<'_, S, W> {
    type Input = (Array<Scalar, S>, Array<Scalar, W>);
    type Output = Unit;
    const NAME: &'static str = "pickles-step";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        arrabbiata_gadget::measure(self, arrabbiata_kimchi_gadgets::LAYOUT).unwrap_or(0)
    }

    #[allow(clippy::too_many_lines)]
    fn synthesize<E>(&self, env: &mut E, (statement, private): Of<Self::Input, E::Var>)
    where
        E: Env<Fp>,
    {
        // The statement first: Kimchi's public input is the first rows.
        let public: Vec<E::Var> = statement
            .iter()
            .map(|v| env.call(&PublicInput, v.clone()))
            .collect();
        let cells = load(env, &private);
        let shape = self.shape;
        let Some(st) = Statement::of(shape.wrap_rounds, &public) else {
            unreachable!("the statement has the shape's size")
        };
        let mut next = cells.into_iter();
        let mut take = || match next.next() {
            Some(v) => v,
            None => unreachable!("the private input has the shape's size"),
        };
        let key = self.wrap_constants.key.map(&mut |_| [take(), take()]);
        let wrap_words: Vec<E::Var> = (0..14 + shape.step_rounds).map(|_| take()).collect();
        let Some(ws) = wrap::Statement::of(shape.step_rounds, &wrap_words) else {
            unreachable!("the wrap statement has the shape's size")
        };
        let mut point = || [take(), take()];
        let witness: [Vec<[E::Var; 2]>; COLUMNS] =
            core::array::from_fn(|_| (0..shape.chunks).map(|_| point()).collect());
        let z = (0..shape.chunks).map(|_| point()).collect();
        let t = (0..shape.quotient_chunks).map(|_| point()).collect();
        let rounds = (0..shape.wrap_rounds).map(|_| (point(), point())).collect();
        let delta = point();
        let sg = point();
        let value = |v: &E::Var| Affine::of(v.clone());
        let type2_of = |high: E::Var, low: E::Var| Shifted {
            high: value(&high),
            low: Some(value(&low)),
        };
        let z1 = type2_of(take(), take());
        let z2 = type2_of(take(), take());
        let ft_eval1 = value(&take());
        let evals = self.base_constants.shape.map(&mut |()| value(&take()));
        let state = value(&take());

        // The wrap proof's group half, x_hat on the wrap statement's words.
        let u = &st.unfinalized;
        let held = |f: &[E::Var; 2]| type2_of(f[0].clone(), f[1].clone());
        let proof = ProofIn {
            previous: vec![],
            witness,
            z,
            t,
            rounds,
            delta,
            z1,
            z2,
            sg: sg.clone(),
        };
        let deferred = DeferredIn {
            public: wrap_words
                .iter()
                .zip(wrap::bounds(shape.step_rounds))
                .map(|(w, bound)| PublicWord {
                    value: value(w),
                    bound,
                })
                .collect(),
            perm: held(&u.perm),
            zeta_to_srs_length: held(&u.zeta_to_srs_length),
            zeta_to_domain_size: held(&u.zeta_to_domain_size),
            xi: value(&u.xi),
            combined_inner_product: held(&u.combined_inner_product),
            b: held(&u.b),
        };
        let model = RefCell::new(Circuit::new(env));
        let key = KeyIn {
            digest: key_digest_in::<Pallas, _>(&model, self.poseidon, &key),
            key,
        };
        let out = group_half_with_key_in::<Pallas, _>(
            &model,
            self.poseidon,
            &self.wrap_constants,
            &key,
            &proof,
            &deferred,
        );

        // The messages: the previous state's, then the next state's with
        // the wrap proof's accumulator, both over the witnessed key (an
        // identity chunk as (0, 0), as `index_elements` has it).
        let digest_of = |tail: &[Affine<E::Var, Fp>]| {
            let mut sponge = FrSponge::in_model(&model, self.poseidon);
            for chunk in key.key.chunks() {
                let (x, y) = match chunk {
                    Some([x, y]) => (value(x), value(y)),
                    None => {
                        let zero = model.borrow_mut().constant(Fp::ZERO);
                        (zero.clone(), zero)
                    }
                };
                sponge.absorb_value(&x);
                sponge.absorb_value(&y);
            }
            for v in tail {
                sponge.absorb_value(v);
            }
            sponge.squeeze_value()
        };
        let previous = digest_of(core::slice::from_ref(&state));
        let next_state = {
            let mut m = model.borrow_mut();
            let one = m.constant(Fp::ONE);
            m.add(&state, &one)
        };
        // The challenges the wrap statement deferred, mapped: the wrapped
        // step proof's, which the messages carry and its scalar half reads.
        let lambda = self.base_constants.lambda;
        let bulletproof_challenges: Vec<Affine<E::Var, Fp>> = {
            let mut m = model.borrow_mut();
            ws.bulletproof_challenges
                .iter()
                .map(|c| m.endo(&value(c), lambda))
                .collect()
        };
        let mut tail = vec![next_state, value(&sg[0]), value(&sg[1])];
        tail.extend(bulletproof_challenges.iter().cloned());
        let messages = digest_of(&tail);

        // The wrap statement finalized: the base step proof's scalar half.
        let mut m = model.into_inner();
        let inputs = Inputs {
            alpha: value(&ws.alpha),
            beta: value(&ws.beta),
            gamma: value(&ws.gamma),
            zeta: value(&ws.zeta),
            digest: value(&ws.sponge_digest_before_evaluations),
            ft_eval1,
            prev_challenges: vec![],
            bulletproof_challenges,
            evals,
        };
        let fin = scalar_half(
            &mut m,
            &self.base_constants,
            self.poseidon,
            &inputs,
            &self.gates,
        );
        let shift = m.constant(type1_shift());
        let lin = &fin.linearization;
        for (computed, stated) in [
            (&fin.combined_inner_product, &ws.combined_inner_product),
            (&fin.b, &ws.b),
            (&lin.zeta_to_srs_length, &ws.zeta_to_srs_length),
            (&lin.zeta_to_domain_size, &ws.zeta_to_domain_size),
            (&lin.perm, &ws.perm),
        ] {
            let t = value(stated);
            let twice = m.add(&t, &t);
            let decoded = m.add(&twice, &shift);
            m.assert_equal(computed, &decoded);
        }
        m.assert_equal(&fin.xi, &value(&ws.xi));

        // The wrap proof's IPA, its draws the unfinalized proof's.
        m.env().connect(&out.left[0], &out.right[0]);
        m.env().connect(&out.left[1], &out.right[1]);
        let d = &out.drawn;
        for (drawn, stated) in [
            (&d.alpha, &u.alpha),
            (&d.beta, &u.beta),
            (&d.gamma, &u.gamma),
            (&d.zeta, &u.zeta),
            (&d.digest, &u.sponge_digest_before_evaluations),
        ] {
            m.assert_equal(drawn, &value(stated));
        }
        for (drawn, stated) in out.prechallenges.iter().zip(&u.bulletproof_challenges) {
            m.assert_equal(drawn, &value(stated));
        }

        // The messages, the passed digest, the flags.
        m.assert_equal(&previous, &value(&ws.messages_for_next_step_proof));
        m.assert_equal(&messages, &value(&st.messages_for_next_step_proof));
        m.assert_equal(
            &value(&st.messages_for_next_wrap_proof),
            &value(&ws.messages_for_next_wrap_proof),
        );
        m.assert_below(&value(&st.messages_for_next_step_proof), WINDOW_BITS);
        m.assert_below(&value(&u.sponge_digest_before_evaluations), WINDOW_BITS);
        m.assert_equal(&value(&u.should_finalize), &Affine::constant(Fp::ONE));
        m.assert_equal(&value(&ws.branch_data), &Affine::constant(self.branch_data));
    }
}

/// A prover index of the step circuit `step`, on a Vesta SRS of its
/// domain's size.
///
/// # Errors
///
/// [`Error::Circuit`] when the circuit is not Kimchi rows.
pub fn index<const S: usize, const W: usize>(
    step: &Step<'_, S, W>,
) -> Result<ProverIndex<Vesta>, Error> {
    let rows = step.rows();
    let gates = compiled::circuit_rows::<Fp, _, VARIABLES, DEGREE>(step, rows, 0)
        .map_err(|_| Error::Circuit)?;
    let cs = ConstraintSystem::create(gates)
        .public(S)
        .prev_challenges(1)
        .build()
        .map_err(|_| Error::Circuit)?;
    let depth = cs.domain.d1.size();
    Ok(ProverIndex::create(cs, Srs::create(depth)))
}

/// The step proof of `statement`, with the private input `private` and the
/// previous challenge `previous` ([`statement`]'s).
///
/// # Errors
///
/// [`Error::Circuit`] for a circuit that is not Kimchi rows,
/// [`Error::Prove`] when Kimchi refuses the witness.
pub fn prove_step<const S: usize, const W: usize, R>(
    step: &Step<'_, S, W>,
    index: &ProverIndex<Vesta>,
    statement: &Statement<Fp>,
    private: &[Fp],
    previous: &RecursionChallenge<Vesta>,
    rng: &mut R,
) -> Result<ProverProof<Vesta>, Error>
where
    R: Sampler<Fp>,
{
    let public: [Fp; S] = statement.to_vec().try_into().map_err(|_| Error::Size)?;
    let private: [Fp; W] = private.try_into().map_err(|_| Error::Size)?;
    let rows = step.rows();
    let (columns, ()) =
        compiled::witness_columns::<Fp, _, VARIABLES, DEGREE>(step, &(public, private), rows)
            .map_err(|_| Error::Circuit)?;
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    prove(
        index,
        columns,
        &[],
        core::slice::from_ref(previous),
        &step_gates(),
        (&base, &scalar),
        rng,
    )
    .map_err(|_| Error::Prove)
}

/// The unfinalized wrap proof's scalar half, natively in `Fq`, on the
/// statement: each deferred value its definition, the messages those the
/// statement hashes, and the wrap accumulator the challenge polynomial's on
/// the wrap SRS (what the next wrap circuit does in its field).
///
/// # Errors
///
/// [`Error::Deferred`] naming the first value that is not its definition,
/// [`Error::Bound`] for a word no Type2 value has, [`Error::Messages`] for
/// messages that are not the statement's or whose key is not
/// `wrap_index`'s (the step circuit takes the key as a witness, so this is
/// where it is pinned), [`Error::Accumulator`] for a false accumulator,
/// [`Error::Lookups`] for an index with lookups.
pub fn finalize(
    wrap_index: &VerifierIndex<Pallas>,
    wrap_srs: &Srs<Pallas>,
    statement: &Statement<Fp>,
    messages: &MessagesForNextStepProof,
    evaluations: &WrapEvaluations,
) -> Result<(), Error> {
    let u = &statement.unfinalized;
    let decode = |[high, low]: [Fp; 2]| {
        Type2::<Pallas>::decode(&[Word::of(high), Word::of(low)]).ok_or(Error::Bound)
    };
    let constants = scalar::Constants::of(wrap_index).map_err(|_| Error::Lookups)?;
    let lambda = Pallas::lambda();
    let prechallenges = &u.bulletproof_challenges;
    let challenges: Vec<Fq> = prechallenges
        .iter()
        .map(|c| ScalarChallenge(to_fq(*c)).to_field(lambda))
        .collect();
    let inputs = Inputs {
        alpha: to_fq(u.alpha),
        beta: to_fq(u.beta),
        gamma: to_fq(u.gamma),
        zeta: to_fq(u.zeta),
        digest: BaseReading::<Pallas>::read_digest(&Native, &u.sponge_digest_before_evaluations),
        ft_eval1: evaluations.ft_eval1,
        prev_challenges: vec![],
        bulletproof_challenges: challenges.clone(),
        evals: evaluations.evals.clone(),
    };
    let poseidon = vesta_base();
    let out = scalar_half(&mut Native, &constants, &poseidon, &inputs, &wrap_gates());
    let lin = &out.linearization;
    for (name, computed, stated) in [
        (
            "combined_inner_product",
            out.combined_inner_product,
            u.combined_inner_product,
        ),
        ("b", out.b, u.b),
        (
            "zeta_to_srs_length",
            lin.zeta_to_srs_length,
            u.zeta_to_srs_length,
        ),
        (
            "zeta_to_domain_size",
            lin.zeta_to_domain_size,
            u.zeta_to_domain_size,
        ),
        ("perm", lin.perm, u.perm),
    ] {
        if computed != decode(stated)? {
            return Err(Error::Deferred(name));
        }
    }
    if to_fp(out.xi) != u.xi {
        return Err(Error::Deferred("xi"));
    }
    if messages.index != index_elements(wrap_index)
        || messages.digest() != statement.messages_for_next_step_proof
    {
        return Err(Error::Messages);
    }
    let sg = msm(&wrap_srs.g, &b_poly_coefficients(&challenges));
    let (x, y) = coordinates(&sg);
    if messages.challenge_polynomial_commitments.last() != Some(&[x, y]) {
        return Err(Error::Accumulator);
    }
    Ok(())
}

/// Verify a step proof: its previous challenge the accumulator the
/// statement's messages hash (`previous`, the verified wrap proof's
/// messages, and the challenges of `messages`), the step proof against its
/// statement (Kimchi, on Vesta, which opens that accumulator at a fresh
/// point), then the unfinalized wrap proof's scalar half, messages and
/// accumulator ([`finalize`]).
///
/// # Errors
///
/// [`Error::Messages`] for a previous challenge that is not the messages',
/// [`Error::Step`] when the step proof does not verify, else
/// [`finalize`]'s.
#[allow(clippy::too_many_arguments)]
pub fn verify_step<R>(
    step_index: &VerifierIndex<Vesta>,
    step_srs: &Srs<Vesta>,
    step_proof: &ProverProof<Vesta>,
    statement: &Statement<Fp>,
    messages: &MessagesForNextStepProof,
    previous: &wrap::MessagesForNextWrapProof,
    wrap: (&VerifierIndex<Pallas>, &Srs<Pallas>, &WrapEvaluations),
    rng: &mut R,
) -> Result<(), Error>
where
    R: Sampler<Fp>,
{
    let [x, y] = previous.challenge_polynomial_commitment;
    let accumulated = step_proof
        .prev_challenges
        .iter()
        .map(|c| (coordinates(&c.comm), &c.chals))
        .collect::<Vec<_>>();
    if Word::of(previous.digest()) != Word::of(statement.messages_for_next_wrap_proof)
        || messages
            .old_bulletproof_challenges
            .first()
            .map(|c| vec![((x, y), c)])
            != Some(accumulated)
    {
        return Err(Error::Messages);
    }
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    verify(
        step_index,
        step_srs,
        step_proof,
        &statement.to_vec(),
        &step_gates(),
        (&base, &scalar),
        rng,
    )
    .map_err(Error::Step)?;
    let (wrap_index, wrap_srs, evaluations) = wrap;
    finalize(wrap_index, wrap_srs, statement, messages, evaluations)
}
