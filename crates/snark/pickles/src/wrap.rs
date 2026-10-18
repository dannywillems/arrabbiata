//! Wrap: a proof over Pallas that a step proof over Vesta verifies, its
//! statement the step proof's deferred values (Pickles' `wrap_main`, for a
//! step proof of a base-case rule).
//!
//! A step proof's verifier splits across the cycle
//! ([`crate::group`], [`crate::scalar`]). Its group half is native in the
//! wrap circuit's field `Fq` (the step proof's commitments are points of
//! Vesta, coordinates in `Fq`); its scalar half is native in `Fp`, the step
//! proof's own field. So the wrap circuit runs the group half and makes the
//! values the group half takes on trust its public input, the
//! [`Statement`]; whoever verifies the wrap proof runs the scalar half on
//! that statement natively ([`finalize`]), and decides the accumulator the
//! statement's messages carry.
//!
//! The statement is Pickles' `Types.Wrap.Statement.In_circuit`, in the
//! order of its `spec` and `to_data`, each value in its sort
//! ([`crate::statement`]):
//!
//! - five Type1 fields: the combined inner product, `b`, `zeta^m`,
//!   `zeta^n`, `perm`;
//! - two challenges, `beta` and `gamma`; three scalar challenges, `alpha`,
//!   `zeta`, `xi`;
//! - three digests: the base sponge's before the evaluations, the messages
//!   for the next wrap proof, the messages for the next step proof;
//! - the IPA's prechallenges; the branch data.
//!
//! The accumulator never crosses as a point. Its commitment `sg` and the
//! old bulletproof challenges are hashed in `Fq` into the messages for the
//! next wrap proof (Pickles' `hash_messages_for_next_wrap_proof`: the
//! challenges, then `sg`'s coordinates), and the circuit computes that hash.
//! A base-case step statement is one digest, the messages for the next step
//! proof, which is the step proof's whole public input.
//!
//! Deviations from Mina: one branch; no predecessor, so no unfinalized
//! proof to finalize in the circuit, and the old challenges are only the
//! padding, given as constants (Mina's are `Dummy.Ipa.Wrap.challenges_computed`,
//! two vectors of 15); no lookups and no optional gates, so the feature
//! flags are constants and add no word. Every word is canonical (below
//! `2^254`, see [`crate::statement`]), where Mina's digests and Type1
//! fields are 255-bit words.

use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use arrabbiata_curve::{Projective, glv::Glv, msm};
use arrabbiata_field::{Field, FieldModel, Native, PrimeField};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_ipa::{b_poly_coefficients, generic::shift_scalar};
use arrabbiata_kimchi::{
    compiled,
    constraints::{COLUMNS, ConstraintSystem},
    deferred::expand_deferred,
    gates::KimchiGates,
    oracle::{BaseReading, FrSponge, ScalarChallenge, coordinates},
    prover::{ProverIndex, ProverProof, prove},
    rng::Sampler,
    srs::Srs,
    verifier::{VerifyError, verify},
    verifier_index::VerifierIndex,
};
use arrabbiata_kimchi_gadgets::generic::PublicInput;
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, KimchiSponge, pallas_base, vesta_base};

use crate::{
    curve::Shifted,
    group::{Constants, DeferredIn, ProofIn, PublicWord, group_half_in},
    model::{Affine, Circuit, load},
    scalar::{self, Inputs, scalar_half},
    statement::{BranchData, Challenge, Digest, ProofsVerified, Sort, Type1, WINDOW_BITS, Word},
};

/// The column variables and the degree of the compiled wrap circuit.
pub const VARIABLES: usize = 64;

/// The degree of the compiled wrap circuit.
pub const DEGREE: usize = 7;

/// What the wrap proof proves about a step proof, in Pickles' order. The
/// fields are Type1 shifted values of scalars of `Fp`, the challenges and
/// the branch data small integers, the digests elements of `Fq` below
/// `2^254`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement<V> {
    /// The combined inner product, shifted.
    pub combined_inner_product: V,
    /// `b`, shifted.
    pub b: V,
    /// `zeta^m`, shifted.
    pub zeta_to_srs_length: V,
    /// `zeta^n`, shifted.
    pub zeta_to_domain_size: V,
    /// `perm`, shifted.
    pub perm: V,
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
    /// The base sponge's digest before the evaluations.
    pub sponge_digest_before_evaluations: V,
    /// The digest of the messages for the next wrap proof
    /// ([`MessagesForNextWrapProof`]).
    pub messages_for_next_wrap_proof: V,
    /// The digest of the messages for the next step proof: the step
    /// proof's statement.
    pub messages_for_next_step_proof: V,
    /// The IPA's round challenges, unmapped.
    pub bulletproof_challenges: Vec<V>,
    /// The branch data ([`BranchData`]).
    pub branch_data: V,
}

/// The values the statement holds besides the prechallenges.
const FIXED: usize = 14;

impl<V: Clone> Statement<V> {
    /// The statement as the wrap proof's public input, in order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<V> {
        let mut out = vec![
            self.combined_inner_product.clone(),
            self.b.clone(),
            self.zeta_to_srs_length.clone(),
            self.zeta_to_domain_size.clone(),
            self.perm.clone(),
            self.beta.clone(),
            self.gamma.clone(),
            self.alpha.clone(),
            self.zeta.clone(),
            self.xi.clone(),
            self.sponge_digest_before_evaluations.clone(),
            self.messages_for_next_wrap_proof.clone(),
            self.messages_for_next_step_proof.clone(),
        ];
        out.extend(self.bulletproof_challenges.iter().cloned());
        out.push(self.branch_data.clone());
        out
    }

    /// The statement of `values`, [`Statement::to_vec`]'s inverse for
    /// `rounds` rounds; `None` for another length.
    #[must_use]
    pub fn of(rounds: usize, values: &[V]) -> Option<Self> {
        if values.len() != FIXED + rounds {
            return None;
        }
        let v = |i: usize| values[i].clone();
        Some(Self {
            combined_inner_product: v(0),
            b: v(1),
            zeta_to_srs_length: v(2),
            zeta_to_domain_size: v(3),
            perm: v(4),
            beta: v(5),
            gamma: v(6),
            alpha: v(7),
            zeta: v(8),
            xi: v(9),
            sponge_digest_before_evaluations: v(10),
            messages_for_next_wrap_proof: v(11),
            messages_for_next_step_proof: v(12),
            bulletproof_challenges: values[13..13 + rounds].to_vec(),
            branch_data: v(13 + rounds),
        })
    }
}

/// The bound of each word of a statement of `rounds` rounds, in order: the
/// sorts' ([`Sort::BOUNDS`]), which a circuit reading the statement as a
/// public input checks.
#[must_use]
pub fn bounds(rounds: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..5 {
        out.extend(Type1::<Vesta>::BOUNDS);
    }
    for _ in 0..5 {
        out.extend(Challenge::BOUNDS);
    }
    for _ in 0..3 {
        out.extend(Digest::BOUNDS);
    }
    for _ in 0..rounds {
        out.extend(Challenge::BOUNDS);
    }
    out.extend(BranchData::BOUNDS);
    out
}

/// What the next wrap proof needs of this one, hashed into the statement:
/// the step proof's accumulator `sg` and the old bulletproof challenges
/// (Pickles' `Messages_for_next_wrap_proof`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessagesForNextWrapProof {
    /// The accumulator's commitment, as coordinates.
    pub challenge_polynomial_commitment: [Fq; 2],
    /// The old bulletproof challenges, padded (front first).
    pub old_bulletproof_challenges: Vec<Vec<Fq>>,
}

impl MessagesForNextWrapProof {
    /// The elements hashed: the challenges in order, then `sg`'s `x` and
    /// `y` (Pickles' `to_field_elements`).
    #[must_use]
    pub fn field_elements(&self) -> Vec<Fq> {
        let mut out: Vec<Fq> = self
            .old_bulletproof_challenges
            .iter()
            .flatten()
            .copied()
            .collect();
        out.extend(self.challenge_polynomial_commitment);
        out
    }

    /// The digest: Kimchi's sponge over `Fq`, absorbing the elements, then
    /// one squeeze (Pickles' `Tock_field_sponge.digest`).
    #[must_use]
    pub fn digest(&self) -> Fq {
        let mut sponge = FrSponge::new(KimchiSponge::new(vesta_base()));
        sponge.absorb_multiple(&self.field_elements());
        sponge.digest()
    }
}

/// The shape of the step proofs a wrap circuit wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The chunks of a commitment.
    pub chunks: usize,
    /// The chunks of the quotient's commitment.
    pub quotient_chunks: usize,
    /// The IPA's rounds.
    pub rounds: usize,
}

impl Shape {
    /// The size of the statement.
    #[must_use]
    pub const fn statement(&self) -> usize {
        FIXED + self.rounds
    }

    /// The size of the private input ([`witness`]): the commitments and the
    /// opening's points as coordinates, `sg` too, then `z1` and `z2`
    /// shifted.
    #[must_use]
    pub const fn witness(&self) -> usize {
        let points =
            COLUMNS * self.chunks + self.chunks + self.quotient_chunks + 2 * self.rounds + 2;
        2 * points + 2
    }
}

/// Why a step proof cannot be wrapped, or a wrap proof does not verify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The step proof does not verify: its transcript has the wrong shape.
    Step(VerifyError),
    /// The step proof's opening does not hold.
    Opening,
    /// The step index has lookups, which the wrap circuit does not cover.
    Lookups,
    /// The wrap circuit's sizes are not the shape's, or the step statement
    /// is not one digest.
    Size,
    /// A value of the statement is past its sort's bound.
    Bound,
    /// The wrap circuit does not compile to Kimchi rows.
    Circuit,
    /// The wrap proof could not be made.
    Prove,
    /// The wrap proof does not verify against the statement.
    Wrap(VerifyError),
    /// A deferred value of the statement is not its definition.
    Deferred(&'static str),
    /// The messages are not those the statement's digest hashes.
    Messages,
    /// The accumulator's commitment is not the challenge polynomial's.
    Accumulator,
}

/// A scalar of `Fp` as the wrap circuit holds it: its shifted value.
#[must_use]
pub fn shifted(s: Fp) -> Fq {
    let t = shift_scalar::<Vesta>(s);
    let limbs: Vec<u64> = (0..Fp::LIMBS).map(|i| t.limb(i)).collect();
    match Fq::from_limbs(&limbs) {
        Some(x) => x,
        None => unreachable!("Fp fits in Fq"),
    }
}

/// The scalar of `Fp` a shifted value `t` stands for: `2 t + 2^255 + 1`.
#[must_use]
pub fn unshifted(t: Fq) -> Fp {
    let limbs: Vec<u64> = (0..Fq::LIMBS).map(|i| t.limb(i)).collect();
    let t = Fp::from_limbs(&limbs).unwrap_or(Fp::ZERO);
    let two = Fp::ONE + Fp::ONE;
    t + t + two.pow(u64::from(Fp::MODULUS_BITS)) + Fp::ONE
}

/// A 128-bit value of `Fq` as the element of `Fp` it is.
fn to_scalar(x: Fq) -> Fp {
    match Fp::from_limbs(&[x.limb(0), x.limb(1)]) {
        Some(s) => s,
        None => unreachable!("128 bits fit in Fp"),
    }
}

/// A 128-bit value of `Fp` as the element of `Fq` it is.
fn to_base(x: Fp) -> Fq {
    match Fq::from_limbs(&[x.limb(0), x.limb(1)]) {
        Some(s) => s,
        None => unreachable!("128 bits fit in Fq"),
    }
}

/// The branch data of a base-case step proof on `index`'s domain, as the
/// element of `Fq` its word is.
fn branch_data_of(index: &VerifierIndex<Vesta>) -> Fq {
    let log2 = index.domain_size.trailing_zeros();
    let log2 = u8::try_from(log2).unwrap_or(u8::MAX);
    BranchData::encode(&(ProofsVerified::N0, log2))[0].reduce()
}

/// A step proof, with what verifying it needs.
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    /// The step index.
    pub index: &'a VerifierIndex<Vesta>,
    /// The step SRS.
    pub srs: &'a Srs<Vesta>,
    /// The proof.
    pub proof: &'a ProverProof<Vesta>,
    /// Its public input: the step statement, here the one digest of the
    /// messages for the next step proof.
    pub public: &'a [Fp],
}

/// Kimchi's gates over `Fp`, for step proofs.
#[must_use]
pub fn step_gates() -> KimchiGates<Fp> {
    match KimchiGates::new(&pallas_base(), Pallas::beta()) {
        Ok(g) => g,
        Err(_) => unreachable!("Kimchi's gates compile"),
    }
}

/// Kimchi's gates over `Fq`, for wrap proofs.
#[must_use]
pub fn wrap_gates() -> KimchiGates<Fq> {
    match KimchiGates::new(&vesta_base(), Vesta::beta()) {
        Ok(g) => g,
        Err(_) => unreachable!("Kimchi's gates compile"),
    }
}

/// The step proof's group half natively: the constants of its index, its
/// commitments and opening, and the deferred values its verifier expands.
#[allow(clippy::type_complexity)]
fn group_inputs(
    step: &Step<'_>,
) -> Result<
    (
        Constants<Vesta>,
        ProofIn<Projective<Vesta>, Fp>,
        DeferredIn<Fq, Fp>,
    ),
    Error,
> {
    let [digest] = step.public else {
        return Err(Error::Size);
    };
    let digest = Word::of(*digest);
    if !digest.below(WINDOW_BITS) {
        return Err(Error::Bound);
    }
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let values = expand_deferred(
        step.index,
        step.srs,
        step.proof,
        step.public,
        &step_gates(),
        (&base, &scalar),
    )
    .map_err(Error::Step)?;
    let constants = Constants::new(step.index, step.srs, &base).ok_or(Error::Lookups)?;
    let opening = &step.proof.proof;
    let proof = ProofIn {
        previous: step.proof.prev_challenges.iter().map(|p| p.comm).collect(),
        witness: core::array::from_fn(|i| step.proof.w_comm[i].chunks.clone()),
        z: step.proof.z_comm.chunks.clone(),
        t: step.proof.t_comm.chunks.clone(),
        rounds: opening.lr.clone(),
        delta: opening.delta,
        z1: opening.z1,
        z2: opening.z2,
        sg: opening.sg,
    };
    let deferred = DeferredIn {
        public: vec![PublicWord {
            value: digest.reduce(),
            bound: WINDOW_BITS,
        }],
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: to_base(values.xi.0),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };
    Ok((constants, proof, deferred))
}

/// The wrap statement of `step`, the messages its digest hashes, the wrap
/// circuit's private input, and the shape: the step proof's group half run
/// natively, its draws the statement, its accumulator hashed with the
/// `padding` challenges, its commitments and opening the witness.
///
/// # Errors
///
/// [`Error::Step`] for a step proof of the wrong shape, [`Error::Size`] for
/// a step statement that is not one digest or a step proof with
/// predecessors, [`Error::Opening`] when its opening does not hold,
/// [`Error::Lookups`] for an index with lookups, [`Error::Bound`] for a
/// value past its sort's bound (about `2^-128` of them).
#[allow(clippy::type_complexity)]
pub fn statement(
    step: &Step<'_>,
    padding: &[Vec<Fq>],
) -> Result<(Statement<Fq>, MessagesForNextWrapProof, Vec<Fq>, Shape), Error> {
    let (constants, proof, deferred) = group_inputs(step)?;
    if !proof.previous.is_empty() {
        return Err(Error::Size);
    }
    let poseidon = vesta_base();
    let model = RefCell::new(Native);
    let out = group_half_in::<Vesta, _>(&model, &poseidon, &constants, &proof, &deferred);
    if out.left != out.right {
        return Err(Error::Opening);
    }
    let (sx, sy) = coordinates(&out.sg);
    let messages = MessagesForNextWrapProof {
        challenge_polynomial_commitment: [sx, sy],
        old_bulletproof_challenges: padding.to_vec(),
    };
    let statement = Statement {
        combined_inner_product: shifted(deferred.combined_inner_product),
        b: shifted(deferred.b),
        zeta_to_srs_length: shifted(deferred.zeta_to_srs_length),
        zeta_to_domain_size: shifted(deferred.zeta_to_domain_size),
        perm: shifted(deferred.perm),
        beta: out.drawn.beta,
        gamma: out.drawn.gamma,
        alpha: out.drawn.alpha,
        zeta: out.drawn.zeta,
        xi: deferred.xi,
        sponge_digest_before_evaluations: out.drawn.digest,
        messages_for_next_wrap_proof: messages.digest(),
        messages_for_next_step_proof: deferred.public[0].value,
        bulletproof_challenges: out.prechallenges,
        branch_data: branch_data_of(step.index),
    };
    let shape = Shape {
        chunks: proof.z.len(),
        quotient_chunks: proof.t.len(),
        rounds: proof.rounds.len(),
    };
    let in_bounds = statement
        .to_vec()
        .iter()
        .zip(bounds(shape.rounds))
        .all(|(v, n)| Word::of(*v).below(n));
    if !in_bounds {
        return Err(Error::Bound);
    }
    Ok((statement, messages, witness(&proof), shape))
}

/// The wrap circuit's private input of a step proof's commitments and
/// opening: the witness columns', `z`'s and `t`'s chunks, the rounds' `L`
/// and `R`, `delta`, `sg` (points as `x`, `y`), then `z1` and `z2`
/// shifted. `sg` is private: the statement holds only its hash.
#[must_use]
pub fn witness(proof: &ProofIn<Projective<Vesta>, Fp>) -> Vec<Fq> {
    let mut points: Vec<&Projective<Vesta>> = Vec::new();
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
    let mut out: Vec<Fq> = points
        .into_iter()
        .flat_map(|p| {
            let (x, y) = coordinates(p);
            [x, y]
        })
        .collect();
    out.push(shifted(proof.z1));
    out.push(shifted(proof.z2));
    out
}

/// The wrap circuit for step proofs of `shape` against one step index: its
/// input the statement (`S` values, the public input) and the private
/// input (`W` values, [`witness`]).
#[derive(Clone, Debug)]
pub struct Wrap<'a, const S: usize, const W: usize> {
    constants: Constants<Vesta>,
    poseidon: &'a KimchiPoseidon<Fq>,
    shape: Shape,
    padding: Vec<Vec<Fq>>,
    branch_data: Fq,
}

impl<'a, const S: usize, const W: usize> Wrap<'a, S, W> {
    /// The wrap circuit for the step index `index` on `srs`, with Kimchi's
    /// `poseidon` over `Fq` for the sponges, and the `padding` challenges
    /// the messages for the next wrap proof start with.
    ///
    /// # Errors
    ///
    /// [`Error::Lookups`] for an index with lookups, [`Error::Size`] when
    /// `S` or `W` is not the shape's.
    pub fn new(
        index: &VerifierIndex<Vesta>,
        srs: &Srs<Vesta>,
        poseidon: &'a KimchiPoseidon<Fq>,
        shape: Shape,
        padding: Vec<Vec<Fq>>,
    ) -> Result<Self, Error> {
        if shape.statement() != S || shape.witness() != W {
            return Err(Error::Size);
        }
        let base = KimchiSponge::new(vesta_base());
        let constants = Constants::new(index, srs, &base).ok_or(Error::Lookups)?;
        Ok(Self {
            constants,
            poseidon,
            shape,
            padding,
            branch_data: branch_data_of(index),
        })
    }
}

impl<const S: usize, const W: usize> TypedGadget<Fq> for Wrap<'_, S, W> {
    type Input = (Array<Scalar, S>, Array<Scalar, W>);
    type Output = Unit;
    const NAME: &'static str = "pickles-wrap";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        arrabbiata_gadget::measure(self, arrabbiata_kimchi_gadgets::LAYOUT).unwrap_or(0)
    }

    fn synthesize<E>(&self, env: &mut E, (statement, private): Of<Self::Input, E::Var>)
    where
        E: Env<Fq>,
    {
        // The statement first: Kimchi's public input is the first rows.
        let public: Vec<E::Var> = statement
            .iter()
            .map(|v| env.call(&PublicInput, v.clone()))
            .collect();
        let cells = load(env, &private);
        let Some(st) = Statement::of(self.shape.rounds, &public) else {
            unreachable!("the statement has the shape's size")
        };
        let shape = self.shape;
        let mut next = cells.into_iter();
        let mut take = || match next.next() {
            Some(v) => v,
            None => unreachable!("the private input has the shape's size"),
        };
        let mut point = || [take(), take()];
        let witness: [Vec<[E::Var; 2]>; COLUMNS] =
            core::array::from_fn(|_| (0..shape.chunks).map(|_| point()).collect());
        let z = (0..shape.chunks).map(|_| point()).collect();
        let t = (0..shape.quotient_chunks).map(|_| point()).collect();
        let rounds = (0..shape.rounds).map(|_| (point(), point())).collect();
        let delta = point();
        let sg = point();
        let value = |v: &E::Var| Affine::of(v.clone());
        let type1 = |v: &E::Var| Shifted {
            high: value(v),
            low: None,
        };
        let z1 = type1(&take());
        let z2 = type1(&take());
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
            public: vec![PublicWord {
                value: value(&st.messages_for_next_step_proof),
                bound: WINDOW_BITS,
            }],
            perm: type1(&st.perm),
            zeta_to_srs_length: type1(&st.zeta_to_srs_length),
            zeta_to_domain_size: type1(&st.zeta_to_domain_size),
            xi: value(&st.xi),
            combined_inner_product: type1(&st.combined_inner_product),
            b: type1(&st.b),
        };
        let model = RefCell::new(Circuit::new(env));
        let out =
            group_half_in::<Vesta, _>(&model, self.poseidon, &self.constants, &proof, &deferred);
        // The messages for the next wrap proof: the padding, then sg.
        let digest = {
            let mut sponge = FrSponge::in_model(&model, self.poseidon);
            for c in self.padding.iter().flatten() {
                let c = model.borrow_mut().constant(*c);
                sponge.absorb_value(&c);
            }
            sponge.absorb_value(&value(&sg[0]));
            sponge.absorb_value(&value(&sg[1]));
            sponge.squeeze_value()
        };
        let mut m = model.into_inner();
        m.env().connect(&out.left[0], &out.right[0]);
        m.env().connect(&out.left[1], &out.right[1]);
        let d = &out.drawn;
        for (drawn, stated) in [
            (&d.alpha, &st.alpha),
            (&d.beta, &st.beta),
            (&d.gamma, &st.gamma),
            (&d.zeta, &st.zeta),
            (&d.digest, &st.sponge_digest_before_evaluations),
            (&digest, &st.messages_for_next_wrap_proof),
        ] {
            m.assert_equal(drawn, &value(stated));
        }
        for (drawn, stated) in out.prechallenges.iter().zip(&st.bulletproof_challenges) {
            m.assert_equal(drawn, &value(stated));
        }
        // The digests cross as words in the window, checked by their
        // producer; the branch data is this circuit's constant.
        m.assert_below(&value(&st.sponge_digest_before_evaluations), WINDOW_BITS);
        m.assert_below(&value(&st.messages_for_next_wrap_proof), WINDOW_BITS);
        m.assert_equal(&value(&st.branch_data), &Affine::constant(self.branch_data));
    }
}

/// A prover index of the wrap circuit `wrap`, on a Pallas SRS of its
/// domain's size.
///
/// # Errors
///
/// [`Error::Circuit`] when the circuit is not Kimchi rows.
pub fn index<const S: usize, const W: usize>(
    wrap: &Wrap<'_, S, W>,
) -> Result<ProverIndex<Pallas>, Error> {
    let rows = wrap.rows();
    let gates = compiled::circuit_rows::<Fq, _, VARIABLES, DEGREE>(wrap, rows, 0)
        .map_err(|_| Error::Circuit)?;
    let cs = ConstraintSystem::create(gates)
        .public(S)
        .build()
        .map_err(|_| Error::Circuit)?;
    let depth = cs.domain.d1.size();
    Ok(ProverIndex::create(cs, Srs::create(depth)))
}

/// The wrap proof of `statement`, with the private input `private`.
///
/// # Errors
///
/// [`Error::Circuit`] for a circuit that is not Kimchi rows,
/// [`Error::Prove`] when Kimchi refuses the witness.
pub fn prove_wrap<const S: usize, const W: usize, R>(
    wrap: &Wrap<'_, S, W>,
    index: &ProverIndex<Pallas>,
    statement: &Statement<Fq>,
    private: &[Fq],
    rng: &mut R,
) -> Result<ProverProof<Pallas>, Error>
where
    R: Sampler<Fq>,
{
    let public: [Fq; S] = statement.to_vec().try_into().map_err(|_| Error::Size)?;
    let private: [Fq; W] = private.try_into().map_err(|_| Error::Size)?;
    let rows = wrap.rows();
    let (columns, ()) =
        compiled::witness_columns::<Fq, _, VARIABLES, DEGREE>(wrap, &(public, private), rows)
            .map_err(|_| Error::Circuit)?;
    let base = KimchiSponge::new(pallas_base());
    let scalar = KimchiSponge::new(vesta_base());
    prove(
        index,
        columns,
        &[],
        &[],
        &wrap_gates(),
        (&base, &scalar),
        rng,
    )
    .map_err(|_| Error::Prove)
}

/// What finalizing a step proof needs beyond the statement: its
/// evaluations, `ft(zeta w)` and its previous accumulators' challenges,
/// which a wrap proof carries along (Pickles' `prev_evals`).
#[derive(Clone, Debug)]
pub struct StepEvaluations {
    /// The step proof's evaluations.
    pub evals: arrabbiata_kimchi::prover::ProofEvaluations<Fp>,
    /// `ft(zeta w)`.
    pub ft_eval1: Fp,
    /// The previous accumulators' challenges.
    pub prev_challenges: Vec<Vec<Fp>>,
}

impl StepEvaluations {
    /// The evaluations `proof` carries.
    #[must_use]
    pub fn of(proof: &ProverProof<Vesta>) -> Self {
        Self {
            evals: proof.evals.clone(),
            ft_eval1: proof.ft_eval1,
            prev_challenges: proof
                .prev_challenges
                .iter()
                .map(|p| p.chals.clone())
                .collect(),
        }
    }
}

/// The step proof's scalar half, natively, on the statement: each deferred
/// value its definition, the messages those the statement hashes, and the
/// accumulator's commitment the challenge polynomial's on the step SRS
/// (Pickles' `finalize_other_proof` and the accumulator check of `verify`).
///
/// # Errors
///
/// [`Error::Deferred`] naming the first value that is not its definition,
/// [`Error::Messages`] for messages that are not the statement's,
/// [`Error::Accumulator`] for a false accumulator, [`Error::Lookups`] for an
/// index with lookups.
pub fn finalize(
    step_index: &VerifierIndex<Vesta>,
    step_srs: &Srs<Vesta>,
    statement: &Statement<Fq>,
    messages: &MessagesForNextWrapProof,
    evaluations: &StepEvaluations,
) -> Result<(), Error> {
    let constants = scalar::Constants::of(step_index).map_err(|_| Error::Lookups)?;
    let lambda = Vesta::lambda();
    let challenges: Vec<Fp> = statement
        .bulletproof_challenges
        .iter()
        .map(|c| ScalarChallenge(to_scalar(*c)).to_field(lambda))
        .collect();
    let inputs = Inputs {
        alpha: to_scalar(statement.alpha),
        beta: to_scalar(statement.beta),
        gamma: to_scalar(statement.gamma),
        zeta: to_scalar(statement.zeta),
        digest: BaseReading::<Vesta>::read_digest(
            &Native,
            &statement.sponge_digest_before_evaluations,
        ),
        ft_eval1: evaluations.ft_eval1,
        prev_challenges: evaluations.prev_challenges.clone(),
        bulletproof_challenges: challenges.clone(),
        evals: evaluations.evals.clone(),
    };
    let poseidon = pallas_base();
    let out = scalar_half(&mut Native, &constants, &poseidon, &inputs, &step_gates());
    let lin = &out.linearization;
    for (name, computed, stated) in [
        (
            "combined_inner_product",
            out.combined_inner_product,
            statement.combined_inner_product,
        ),
        ("b", out.b, statement.b),
        (
            "zeta_to_srs_length",
            lin.zeta_to_srs_length,
            statement.zeta_to_srs_length,
        ),
        (
            "zeta_to_domain_size",
            lin.zeta_to_domain_size,
            statement.zeta_to_domain_size,
        ),
        ("perm", lin.perm, statement.perm),
    ] {
        if computed != unshifted(stated) {
            return Err(Error::Deferred(name));
        }
    }
    if to_base(out.xi) != statement.xi {
        return Err(Error::Deferred("xi"));
    }
    if messages.digest() != statement.messages_for_next_wrap_proof {
        return Err(Error::Messages);
    }
    let sg = msm(&step_srs.g, &b_poly_coefficients(&challenges));
    let [x, y] = messages.challenge_polynomial_commitment;
    if coordinates(&sg) != (x, y) {
        return Err(Error::Accumulator);
    }
    Ok(())
}

/// Verify a wrap proof: the wrap proof against its statement (Kimchi, on
/// Pallas), then the step proof's scalar half, messages and accumulator on
/// the statement ([`finalize`]).
///
/// # Errors
///
/// [`Error::Wrap`] when the wrap proof does not verify, else
/// [`finalize`]'s.
pub fn verify_wrap<R>(
    wrap_index: &VerifierIndex<Pallas>,
    wrap_srs: &Srs<Pallas>,
    wrap_proof: &ProverProof<Pallas>,
    statement: &Statement<Fq>,
    messages: &MessagesForNextWrapProof,
    step: (&VerifierIndex<Vesta>, &Srs<Vesta>, &StepEvaluations),
    rng: &mut R,
) -> Result<(), Error>
where
    R: Sampler<Fq>,
{
    let base = KimchiSponge::new(pallas_base());
    let scalar = KimchiSponge::new(vesta_base());
    verify(
        wrap_index,
        wrap_srs,
        wrap_proof,
        &statement.to_vec(),
        &wrap_gates(),
        (&base, &scalar),
        rng,
    )
    .map_err(Error::Wrap)?;
    let (step_index, step_srs, evaluations) = step;
    finalize(step_index, step_srs, statement, messages, evaluations)
}
