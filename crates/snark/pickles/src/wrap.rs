//! Wrap: a proof over Pallas that a step proof over Vesta verifies, its
//! statement the step proof's deferred values (Pickles' `wrap_main`, for a
//! step proof without predecessors).
//!
//! A step proof's verifier splits across the cycle
//! ([`crate::group`], [`crate::scalar`]). Its group half is native in the
//! wrap circuit's field `Fq` (the step proof's commitments are points of
//! Vesta, coordinates in `Fq`); its scalar half is native in `Fp`, the step
//! proof's own field. So the wrap circuit runs the group half and makes the
//! values the group half takes on trust its public input, the
//! [`Statement`]; whoever verifies the wrap proof runs the scalar half on
//! that statement natively ([`finalize`]), and decides the accumulator the
//! statement carries.
//!
//! - The statement's scalars of `Fp` are held in `Fq` as their shifted
//!   values (`Shifted_value.Type1`: `s = 2 t + 2^255 + 1`), as Pickles does.
//! - The circuit wires each draw of the group half (`alpha`, `beta`,
//!   `gamma`, `zeta`, the digest, the prechallenges) and the new accumulator
//!   `sg` to the statement, and the IPA's two sides to each other.
//!
//! Deviations from Mina: the statement is a flat list of its values, not
//! Pickles' packed `Types.Wrap.Statement` and its message hashes; one branch;
//! no predecessor, so no unfinalized proof to finalize in the circuit; no
//! lookups. The values are Mina's; their encoding is not.

use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use arrabbiata_curve::{Projective, glv::Glv, msm};
use arrabbiata_field::{Field, Native, PrimeField};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_ipa::{b_poly_coefficients, generic::shift_scalar};
use arrabbiata_kimchi::{
    compiled,
    constraints::{COLUMNS, ConstraintSystem},
    deferred::expand_deferred,
    gates::KimchiGates,
    oracle::{BaseReading, ScalarChallenge, coordinates},
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
    group::{Constants, DeferredIn, ProofIn, group_half_in},
    model::{Affine, Circuit, load},
    scalar::{self, Inputs, scalar_half},
};

/// The column variables and the degree of the compiled wrap circuit.
pub const VARIABLES: usize = 64;

/// The degree of the compiled wrap circuit.
pub const DEGREE: usize = 7;

/// What the wrap proof proves about a step proof: its public input. The
/// scalars of `Fp` are shifted values, the 128-bit challenges and the
/// digest values of `Fq` as they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement<V> {
    /// The step proof's public input, shifted.
    pub step_public: Vec<V>,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha: V,
    /// `beta`.
    pub beta: V,
    /// `gamma`.
    pub gamma: V,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta: V,
    /// The 128-bit challenge `xi` comes from.
    pub xi: V,
    /// The base sponge's digest before the evaluations.
    pub digest: V,
    /// `perm`, shifted.
    pub perm: V,
    /// `zeta^m`, shifted.
    pub zeta_to_srs_length: V,
    /// `zeta^n`, shifted.
    pub zeta_to_domain_size: V,
    /// The combined inner product, shifted.
    pub combined_inner_product: V,
    /// `b`, shifted.
    pub b: V,
    /// The new accumulator's commitment `sg`.
    pub sg: [V; 2],
    /// The IPA's round challenges, unmapped.
    pub prechallenges: Vec<V>,
}

/// The values the statement holds besides the public input and the
/// prechallenges.
const FIXED: usize = 14;

impl<V: Clone> Statement<V> {
    /// The statement as the wrap proof's public input, in order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<V> {
        let mut out = self.step_public.clone();
        out.extend([
            self.alpha.clone(),
            self.beta.clone(),
            self.gamma.clone(),
            self.zeta.clone(),
            self.xi.clone(),
            self.digest.clone(),
            self.perm.clone(),
            self.zeta_to_srs_length.clone(),
            self.zeta_to_domain_size.clone(),
            self.combined_inner_product.clone(),
            self.b.clone(),
            self.sg[0].clone(),
            self.sg[1].clone(),
        ]);
        out.extend(self.prechallenges.iter().cloned());
        out
    }

    /// The statement of `values`, [`Statement::to_vec`]'s inverse for a
    /// public input of `public` values and `rounds` rounds; `None` for
    /// another length.
    #[must_use]
    pub fn of(public: usize, rounds: usize, values: &[V]) -> Option<Self> {
        if values.len() != public + FIXED - 1 + rounds {
            return None;
        }
        let (step_public, rest) = values.split_at(public);
        let (fixed, prechallenges) = rest.split_at(FIXED - 1);
        let v = |i: usize| fixed[i].clone();
        Some(Self {
            step_public: step_public.to_vec(),
            alpha: v(0),
            beta: v(1),
            gamma: v(2),
            zeta: v(3),
            xi: v(4),
            digest: v(5),
            perm: v(6),
            zeta_to_srs_length: v(7),
            zeta_to_domain_size: v(8),
            combined_inner_product: v(9),
            b: v(10),
            sg: [v(11), v(12)],
            prechallenges: prechallenges.to_vec(),
        })
    }
}

/// The shape of the step proofs a wrap circuit wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The step proof's public input's length.
    pub public: usize,
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
        self.public + FIXED - 1 + self.rounds
    }

    /// The size of the private input ([`witness`]): the commitments and the
    /// opening's points as coordinates, then `z1` and `z2` shifted.
    #[must_use]
    pub const fn witness(&self) -> usize {
        let points =
            COLUMNS * self.chunks + self.chunks + self.quotient_chunks + 2 * self.rounds + 1;
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
    /// The wrap circuit's sizes are not the shape's.
    Size,
    /// The wrap circuit does not compile to Kimchi rows.
    Circuit,
    /// The wrap proof could not be made.
    Prove,
    /// The wrap proof does not verify against the statement.
    Wrap(VerifyError),
    /// A deferred value of the statement is not its definition.
    Deferred(&'static str),
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

/// A step proof, with what verifying it needs.
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    /// The step index.
    pub index: &'a VerifierIndex<Vesta>,
    /// The step SRS.
    pub srs: &'a Srs<Vesta>,
    /// The proof.
    pub proof: &'a ProverProof<Vesta>,
    /// Its public input.
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
        public: step.public.to_vec(),
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: to_base(values.xi.0),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };
    Ok((constants, proof, deferred))
}

/// The wrap statement of `step` and the wrap circuit's private input: the
/// step proof's group half run natively, its draws and accumulator the
/// statement, its commitments and opening the witness.
///
/// # Errors
///
/// [`Error::Step`] for a step proof of the wrong shape, [`Error::Opening`]
/// when its opening does not hold, [`Error::Lookups`] for an index with
/// lookups.
pub fn statement(step: &Step<'_>) -> Result<(Statement<Fq>, Vec<Fq>, Shape), Error> {
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
    let statement = Statement {
        step_public: deferred.public.iter().map(|p| shifted(*p)).collect(),
        alpha: out.drawn.alpha,
        beta: out.drawn.beta,
        gamma: out.drawn.gamma,
        zeta: out.drawn.zeta,
        xi: deferred.xi,
        digest: out.drawn.digest,
        perm: shifted(deferred.perm),
        zeta_to_srs_length: shifted(deferred.zeta_to_srs_length),
        zeta_to_domain_size: shifted(deferred.zeta_to_domain_size),
        combined_inner_product: shifted(deferred.combined_inner_product),
        b: shifted(deferred.b),
        sg: [sx, sy],
        prechallenges: out.prechallenges,
    };
    let shape = Shape {
        public: step.public.len(),
        chunks: proof.z.len(),
        quotient_chunks: proof.t.len(),
        rounds: proof.rounds.len(),
    };
    Ok((statement, witness(&proof), shape))
}

/// The wrap circuit's private input of a step proof's commitments and
/// opening: the witness columns', `z`'s and `t`'s chunks, the rounds' `L`
/// and `R`, `delta` (points as `x`, `y`), then `z1` and `z2` shifted. The
/// opening's `sg` is the statement's.
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
}

impl<'a, const S: usize, const W: usize> Wrap<'a, S, W> {
    /// The wrap circuit for the step index `index` on `srs`, with Kimchi's
    /// `poseidon` over `Fq` for the base sponge.
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
        let Some(st) = Statement::of(self.shape.public, self.shape.rounds, &public) else {
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
        let z1 = Affine::of(take());
        let z2 = Affine::of(take());
        let value = |v: &E::Var| Affine::of(v.clone());
        let proof = ProofIn {
            previous: vec![],
            witness,
            z,
            t,
            rounds,
            delta,
            z1,
            z2,
            sg: st.sg.clone(),
        };
        let deferred = DeferredIn {
            public: st.step_public.iter().map(value).collect(),
            perm: value(&st.perm),
            zeta_to_srs_length: value(&st.zeta_to_srs_length),
            zeta_to_domain_size: value(&st.zeta_to_domain_size),
            xi: value(&st.xi),
            combined_inner_product: value(&st.combined_inner_product),
            b: value(&st.b),
        };
        let model = RefCell::new(Circuit::new(env));
        let out =
            group_half_in::<Vesta, _>(&model, self.poseidon, &self.constants, &proof, &deferred);
        let mut m = model.into_inner();
        m.env().connect(&out.left[0], &out.right[0]);
        m.env().connect(&out.left[1], &out.right[1]);
        let d = &out.drawn;
        for (drawn, stated) in [
            (&d.alpha, &st.alpha),
            (&d.beta, &st.beta),
            (&d.gamma, &st.gamma),
            (&d.zeta, &st.zeta),
            (&d.digest, &st.digest),
        ] {
            m.assert_equal(drawn, &value(stated));
        }
        for (drawn, stated) in out.prechallenges.iter().zip(&st.prechallenges) {
            m.assert_equal(drawn, &value(stated));
        }
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
/// value its definition, and the accumulator's commitment the challenge
/// polynomial's on the step SRS (Pickles' `finalize_other_proof` and the
/// accumulator check of `verify`).
///
/// # Errors
///
/// [`Error::Deferred`] naming the first value that is not its definition,
/// [`Error::Accumulator`] for a false accumulator, [`Error::Lookups`] for an
/// index with lookups.
pub fn finalize(
    step_index: &VerifierIndex<Vesta>,
    step_srs: &Srs<Vesta>,
    statement: &Statement<Fq>,
    evaluations: &StepEvaluations,
) -> Result<(), Error> {
    let constants = scalar::Constants::of(step_index).map_err(|_| Error::Lookups)?;
    let lambda = Vesta::lambda();
    let challenges: Vec<Fp> = statement
        .prechallenges
        .iter()
        .map(|c| ScalarChallenge(to_scalar(*c)).to_field(lambda))
        .collect();
    let inputs = Inputs {
        alpha: to_scalar(statement.alpha),
        beta: to_scalar(statement.beta),
        gamma: to_scalar(statement.gamma),
        zeta: to_scalar(statement.zeta),
        digest: BaseReading::<Vesta>::read_digest(&Native, &statement.digest),
        ft_eval1: evaluations.ft_eval1,
        prev_challenges: evaluations.prev_challenges.clone(),
        bulletproof_challenges: challenges.clone(),
        evals: evaluations.evals.clone(),
    };
    let poseidon = pallas_base();
    let out = scalar_half(&mut Native, &constants, &poseidon, &inputs, &step_gates());
    let lin = &out.linearization;
    for (name, computed, stated) in [
        ("perm", lin.perm, statement.perm),
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
        (
            "combined_inner_product",
            out.combined_inner_product,
            statement.combined_inner_product,
        ),
        ("b", out.b, statement.b),
    ] {
        if computed != unshifted(stated) {
            return Err(Error::Deferred(name));
        }
    }
    if to_base(out.xi) != statement.xi {
        return Err(Error::Deferred("xi"));
    }
    let sg = msm(&step_srs.g, &b_poly_coefficients(&challenges));
    if coordinates(&sg) != (statement.sg[0], statement.sg[1]) {
        return Err(Error::Accumulator);
    }
    Ok(())
}

/// Verify a wrap proof: the wrap proof against its statement (Kimchi, on
/// Pallas), then the step proof's scalar half and accumulator on the
/// statement ([`finalize`]).
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
    finalize(step_index, step_srs, statement, evaluations)
}
