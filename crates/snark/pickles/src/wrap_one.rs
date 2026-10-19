//! Wrap of a step proof with one predecessor: a proof over Pallas that a
//! step proof over Vesta verifies, which finalizes in its circuit the wrap
//! proof the step proof left unfinalized (Pickles' `wrap_main` for a branch
//! of one proof verified).
//!
//! A step proof that verified a wrap proof carries that wrap proof's
//! deferred values in its statement ([`step::Unfinalized`]): scalars of
//! `Fq`, the wrap proof's own field, as Type2 words. The wrap circuit runs
//! over `Fq`, so their scalar half is native here, and this is where they
//! are finalized, each value checked against its definition on the wrap
//! proof's evaluations (a private input).
//!
//! So the circuit, over `Fq`:
//!
//! - rebuilds the step statement, the step proof's public input: the
//!   unfinalized proof (private, each word lifted canonically by `x_hat`
//!   under its bound), the messages for the next step proof (this
//!   statement's, passed along), and the predecessor's messages for the next
//!   wrap proof, hashed here from their accumulator and padding;
//! - runs the step proof's group half on it, as [`crate::wrap`] does;
//! - finalizes the unfinalized proof: the wrap proof's scalar half;
//! - hashes the new messages for the next wrap proof: the padding vector,
//!   the finalized proof's challenges (mapped by the endomorphism, as
//!   Pickles' `challenges_computed`), then the step proof's `sg`.
//!
//! The statement is [`wrap::Statement`], Mina's order, its branch data for
//! one proof verified.
//!
//! The accumulators: the step proof carries the base step proof's as its
//! previous challenge, whose commitment the group half reads among the
//! step proof's commitments, and whose cells are those hashed into the
//! predecessor's messages, so the accumulator the step proof opened is the
//! one the step statement passed. The wrap proof carries the predecessor's
//! accumulator in turn: the `sg` the step messages hold, with the finalized
//! challenges.
//!
//! Deviations from Mina: the unfinalized proof must be finalized
//! (`should_finalize` is asserted one, where Mina's circuit skips a dummy);
//! the padding is ours, not `Dummy.Ipa.Wrap.challenges_computed`.

use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use arrabbiata_curve::{glv::Glv, msm};
use arrabbiata_field::{Field, FieldModel, Native};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_ipa::{RecursionChallenge, b_poly_coefficients};
use arrabbiata_kimchi::{
    compiled,
    constraints::{COLUMNS, ConstraintSystem},
    deferred::expand_deferred,
    oracle::{ChallengeModel, FrSponge, ScalarChallenge, coordinates},
    prover::{ProverIndex, ProverProof, prove},
    rng::Sampler,
    srs::Srs,
    verifier::VerifyError,
    verifier_index::VerifierIndex,
};
use arrabbiata_kimchi_gadgets::generic::PublicInput;
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, KimchiSponge, pallas_base, vesta_base};

use crate::{
    curve::{Shifted, point_at},
    group::{Constants, DeferredIn, ProofIn, PublicWord, group_half_in},
    model::{Affine, Circuit, load},
    scalar::{self, Inputs, scalar_half},
    statement::{ProofsVerified, Sort, Type2, WINDOW_BITS, Word},
    step::{self, MessagesForNextStepProof, WrapEvaluations},
    wrap::{
        self, MessagesForNextWrapProof, Statement, StepEvaluations, shifted, step_gates, wrap_gates,
    },
};

/// The column variables of the compiled circuit.
pub const VARIABLES: usize = 64;

/// The degree of the compiled circuit.
pub const DEGREE: usize = 7;

/// A step proof with one predecessor, with what verifying it needs.
#[derive(Clone, Copy, Debug)]
pub struct StepProof<'a> {
    /// The step index.
    pub index: &'a VerifierIndex<Vesta>,
    /// The step SRS.
    pub srs: &'a Srs<Vesta>,
    /// The proof.
    pub proof: &'a ProverProof<Vesta>,
    /// Its statement.
    pub statement: &'a step::Statement<Fp>,
    /// The messages its statement's digest hashes: the predecessor's
    /// accumulator commitment, and the challenges the step proof carries.
    pub messages: &'a MessagesForNextStepProof,
}

/// The predecessor the step proof verified: the wrap proof's index and
/// evaluations, which finalizing it needs, and its messages for the next
/// wrap proof, whose digest the step statement passes along.
#[derive(Clone, Copy, Debug)]
pub struct Predecessor<'a> {
    /// The wrap proof's index.
    pub index: &'a VerifierIndex<Pallas>,
    /// Its evaluations.
    pub evaluations: &'a WrapEvaluations,
    /// Its messages for the next wrap proof.
    pub messages: &'a MessagesForNextWrapProof,
}

/// The shape of the proofs the circuit wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The step proof's.
    pub step: wrap::Shape,
    /// The predecessor's IPA rounds (the unfinalized proof's prechallenges).
    pub wrap_rounds: usize,
    /// The predecessor's evaluations.
    pub evaluations: usize,
}

impl Shape {
    /// The size of the statement.
    #[must_use]
    pub const fn statement(&self) -> usize {
        self.step.statement()
    }

    /// The size of the private input: the step proof's ([`wrap::witness`]),
    /// the unfinalized proof's words, the predecessor's `ft(zeta w)` and
    /// evaluations, and its accumulator's coordinates.
    #[must_use]
    pub const fn witness(&self) -> usize {
        self.step.witness() + (17 + self.wrap_rounds) + 1 + self.evaluations + 2
    }
}

/// Why a step proof cannot be wrapped, or the wrap proof does not verify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The step proof does not verify: its transcript has the wrong shape.
    Step(VerifyError),
    /// The step proof's opening does not hold.
    Opening,
    /// An index has lookups, which the circuits do not cover.
    Lookups,
    /// The circuit's sizes are not the shape's, or the padding is not one
    /// vector.
    Size,
    /// A value of a statement is past its sort's bound.
    Bound,
    /// The unfinalized proof is not finalized: a deferred value is not its
    /// definition.
    Deferred(&'static str),
    /// The predecessor's messages are not those the step statement passes.
    Messages,
    /// The circuit does not compile to Kimchi rows.
    Circuit,
    /// The wrap proof could not be made.
    Prove,
    /// The wrap proof does not verify, or the step proof it wraps does not
    /// finalize.
    Wrap(wrap::Error),
    /// The step messages, or the accumulator they hash, are not right.
    Accumulator,
}

/// The step proof's words, a statement of `Fp`, in `Fq`: each is below
/// `2^254`, so its integer is the same.
fn words(statement: &step::Statement<Fp>) -> Vec<Fq> {
    statement
        .to_vec()
        .iter()
        .map(|w| Word::of(*w).reduce())
        .collect()
}

/// The finalized proof's challenges mapped by the endomorphism of Pallas:
/// the scalars of `Fq` its accumulator's polynomial is made of.
fn computed(prechallenges: &[Fp]) -> Vec<Fq> {
    let lambda = Pallas::lambda();
    prechallenges
        .iter()
        .map(|c| ScalarChallenge(Word::of(*c).reduce::<Fq>()).to_field(lambda))
        .collect()
}

/// The unfinalized proof finalized natively: what the circuit checks, so
/// that a false one is refused before proving.
fn finalized(
    index: &VerifierIndex<Pallas>,
    unfinalized: &step::Unfinalized<Fp>,
    evaluations: &WrapEvaluations,
) -> Result<(), Error> {
    let decode = |[high, low]: [Fp; 2]| {
        Type2::<Pallas>::decode(&[Word::of(high), Word::of(low)]).ok_or(Error::Bound)
    };
    let constants = scalar::Constants::of(index).map_err(|_| Error::Lookups)?;
    let to_fq = |x: Fp| -> Fq { Word::of(x).reduce() };
    let inputs = Inputs {
        alpha: to_fq(unfinalized.alpha),
        beta: to_fq(unfinalized.beta),
        gamma: to_fq(unfinalized.gamma),
        zeta: to_fq(unfinalized.zeta),
        digest: to_fq(unfinalized.sponge_digest_before_evaluations),
        ft_eval1: evaluations.ft_eval1,
        prev_challenges: vec![],
        bulletproof_challenges: computed(&unfinalized.bulletproof_challenges),
        evals: evaluations.evals.clone(),
    };
    let out = scalar_half(
        &mut Native,
        &constants,
        &vesta_base(),
        &inputs,
        &wrap_gates(),
    );
    let lin = &out.linearization;
    for (name, value, stated) in [
        (
            "combined_inner_product",
            out.combined_inner_product,
            unfinalized.combined_inner_product,
        ),
        ("b", out.b, unfinalized.b),
        (
            "zeta_to_srs_length",
            lin.zeta_to_srs_length,
            unfinalized.zeta_to_srs_length,
        ),
        (
            "zeta_to_domain_size",
            lin.zeta_to_domain_size,
            unfinalized.zeta_to_domain_size,
        ),
        ("perm", lin.perm, unfinalized.perm),
    ] {
        if value != decode(stated)? {
            return Err(Error::Deferred(name));
        }
    }
    if Word::of(out.xi) != Word::of(unfinalized.xi) || unfinalized.should_finalize != Fp::ONE {
        return Err(Error::Deferred("xi"));
    }
    Ok(())
}

/// The wrap statement of `step`, the messages its digest hashes, the
/// private input, the shape, and the previous challenge the wrap proof
/// carries: the step proof's group half natively, its unfinalized proof
/// finalized on `predecessor`'s evaluations, the new messages hashed from
/// `padding` (one vector), the finalized challenges and the step proof's
/// accumulator; the previous challenge the predecessor's accumulator.
///
/// # Errors
///
/// [`Error::Step`] for a step proof of the wrong shape, [`Error::Opening`]
/// when its opening does not hold, [`Error::Messages`] when the
/// predecessor's messages are not the step statement's, [`Error::Deferred`]
/// for an unfinalized proof that does not finalize, [`Error::Bound`] for a
/// value past its bound, [`Error::Size`] for another padding.
#[allow(clippy::type_complexity)]
pub fn statement(
    step: &StepProof<'_>,
    predecessor: &Predecessor<'_>,
    padding: &[Fq],
) -> Result<
    (
        Statement<Fq>,
        MessagesForNextWrapProof,
        Vec<Fq>,
        Shape,
        RecursionChallenge<Pallas>,
    ),
    Error,
> {
    let unfinalized = &step.statement.unfinalized;
    let [px, py] = predecessor.messages.challenge_polynomial_commitment;
    let carried: Vec<(Fq, Fq)> = step
        .proof
        .prev_challenges
        .iter()
        .map(|c| coordinates(&c.comm))
        .collect();
    if Word::of(predecessor.messages.digest())
        != Word::of(step.statement.messages_for_next_wrap_proof)
        || step.messages.digest() != step.statement.messages_for_next_step_proof
        || carried != vec![(px, py)]
    {
        return Err(Error::Messages);
    }
    let Some([wx, wy]) = step
        .messages
        .challenge_polynomial_commitments
        .last()
        .copied()
    else {
        return Err(Error::Messages);
    };
    let previous_challenge = RecursionChallenge {
        chals: computed(&unfinalized.bulletproof_challenges),
        comm: point_at::<Pallas>(wx, wy).ok_or(Error::Messages)?,
    };
    finalized(predecessor.index, unfinalized, predecessor.evaluations)?;

    let public = step.statement.to_vec();
    let base = KimchiSponge::new(vesta_base());
    let scalar_sponge = KimchiSponge::new(pallas_base());
    let values = expand_deferred(
        step.index,
        step.srs,
        step.proof,
        &public,
        &step_gates(),
        (&base, &scalar_sponge),
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
    let wrap_rounds = unfinalized.bulletproof_challenges.len();
    let deferred = DeferredIn {
        public: words(step.statement)
            .into_iter()
            .zip(step::bounds(wrap_rounds))
            .map(|(value, bound)| PublicWord { value, bound })
            .collect(),
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: Word::of(values.xi.0).reduce(),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };
    let model = RefCell::new(Native);
    let out = group_half_in::<Vesta, _>(&model, &vesta_base(), &constants, &proof, &deferred);
    if out.left != out.right {
        return Err(Error::Opening);
    }
    let (sx, sy) = coordinates(&out.sg);
    let messages = MessagesForNextWrapProof {
        challenge_polynomial_commitment: [sx, sy],
        old_bulletproof_challenges: vec![
            padding.to_vec(),
            computed(&unfinalized.bulletproof_challenges),
        ],
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
        messages_for_next_step_proof: Word::of(step.statement.messages_for_next_step_proof)
            .reduce(),
        bulletproof_challenges: out.prechallenges,
        branch_data: wrap::branch_data_of(step.index, ProofsVerified::N1),
    };
    let step_shape = wrap::Shape {
        chunks: proof.z.len(),
        quotient_chunks: proof.t.len(),
        rounds: proof.rounds.len(),
    };
    let in_bounds = statement
        .to_vec()
        .iter()
        .zip(wrap::bounds(step_shape.rounds))
        .all(|(v, n)| Word::of(*v).below(n));
    if !in_bounds {
        return Err(Error::Bound);
    }
    let mut evaluations = 0;
    let _ = predecessor.evaluations.evals.map(&mut |_| evaluations += 1);
    let shape = Shape {
        step: step_shape,
        wrap_rounds,
        evaluations,
    };

    let mut private = wrap::witness(&proof);
    let step_words = words(step.statement);
    private.extend(&step_words[..17 + wrap_rounds]);
    private.push(predecessor.evaluations.ft_eval1);
    let _ = predecessor.evaluations.evals.map(&mut |v| private.push(*v));
    private.extend(predecessor.messages.challenge_polynomial_commitment);
    Ok((statement, messages, private, shape, previous_challenge))
}

/// The wrap circuit for step proofs of `shape` with one predecessor: its
/// input the statement (`S` values) and the private input (`W` values).
#[derive(Clone, Debug)]
pub struct WrapOne<'a, const S: usize, const W: usize> {
    constants: Constants<Vesta>,
    predecessor: scalar::Constants<Fq>,
    poseidon: &'a KimchiPoseidon<Fq>,
    gates: arrabbiata_kimchi::gates::KimchiGates<Fq>,
    shape: Shape,
    padding: Vec<Fq>,
    previous_padding: Vec<Vec<Fq>>,
    branch_data: Fq,
}

impl<'a, const S: usize, const W: usize> WrapOne<'a, S, W> {
    /// The circuit wrapping step proofs on `index` and `srs` whose
    /// predecessors are wrap proofs on `predecessor`, with Kimchi's
    /// `poseidon` over `Fq`; the new messages start with `padding`, the
    /// predecessor's with `previous_padding` (its [`wrap::Wrap`]'s).
    ///
    /// # Errors
    ///
    /// [`Error::Lookups`] for an index with lookups, [`Error::Size`] when
    /// `S` or `W` is not the shape's.
    pub fn new(
        index: &VerifierIndex<Vesta>,
        srs: &Srs<Vesta>,
        predecessor: &VerifierIndex<Pallas>,
        poseidon: &'a KimchiPoseidon<Fq>,
        shape: Shape,
        padding: Vec<Fq>,
        previous_padding: Vec<Vec<Fq>>,
    ) -> Result<Self, Error> {
        if shape.statement() != S || shape.witness() != W {
            return Err(Error::Size);
        }
        let base = KimchiSponge::new(vesta_base());
        let constants = Constants::new(index, srs, &base).ok_or(Error::Lookups)?;
        let predecessor = scalar::Constants::of(predecessor).map_err(|_| Error::Lookups)?;
        Ok(Self {
            constants,
            predecessor,
            poseidon,
            gates: wrap_gates(),
            shape,
            padding,
            previous_padding,
            branch_data: wrap::branch_data_of(index, ProofsVerified::N1),
        })
    }
}

impl<const S: usize, const W: usize> TypedGadget<Fq> for WrapOne<'_, S, W> {
    type Input = (Array<Scalar, S>, Array<Scalar, W>);
    type Output = Unit;
    const NAME: &'static str = "pickles-wrap-one";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        arrabbiata_gadget::measure(self, arrabbiata_kimchi_gadgets::LAYOUT).unwrap_or(0)
    }

    #[allow(clippy::too_many_lines)]
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
        let shape = self.shape;
        let Some(st) = Statement::of(shape.step.rounds, &public) else {
            unreachable!("the statement has the shape's size")
        };
        let mut next = cells.into_iter();
        let mut take = || match next.next() {
            Some(v) => v,
            None => unreachable!("the private input has the shape's size"),
        };
        let mut point = || [take(), take()];
        let witness: [Vec<[E::Var; 2]>; COLUMNS] =
            core::array::from_fn(|_| (0..shape.step.chunks).map(|_| point()).collect());
        let z = (0..shape.step.chunks).map(|_| point()).collect();
        let t = (0..shape.step.quotient_chunks).map(|_| point()).collect();
        let rounds = (0..shape.step.rounds).map(|_| (point(), point())).collect();
        let delta = point();
        let sg = point();
        let value = |v: &E::Var| Affine::of(v.clone());
        let type1 = |v: &E::Var| Shifted {
            high: value(v),
            low: None,
        };
        let z1 = type1(&take());
        let z2 = type1(&take());
        let unfinalized_words: Vec<E::Var> = (0..17 + shape.wrap_rounds).map(|_| take()).collect();
        let ft_eval1 = value(&take());
        let evals = self.predecessor.shape.map(&mut |()| value(&take()));
        let previous_sg = [take(), take()];

        let model = RefCell::new(Circuit::new(env));
        let digest_of = |constants: &[Fq], tail: &[Affine<E::Var, Fq>]| {
            let mut sponge = FrSponge::in_model(&model, self.poseidon);
            for c in constants {
                let c = model.borrow_mut().constant(*c);
                sponge.absorb_value(&c);
            }
            for v in tail {
                sponge.absorb_value(v);
            }
            sponge.squeeze_value()
        };

        // The step statement: the unfinalized proof, this statement's
        // messages for the next step proof, and the predecessor's messages
        // for the next wrap proof, hashed here.
        let previous_padding: Vec<Fq> = self.previous_padding.iter().flatten().copied().collect();
        let previous = digest_of(
            &previous_padding,
            &[value(&previous_sg[0]), value(&previous_sg[1])],
        );
        let previous_cell = model.borrow_mut().cell(&previous);
        let mut step_words = unfinalized_words.clone();
        step_words.push(st.messages_for_next_step_proof.clone());
        step_words.push(previous_cell);
        let Some(step_statement) = step::Statement::of(shape.wrap_rounds, &step_words) else {
            unreachable!("the step statement has the shape's size")
        };
        let u = &step_statement.unfinalized;

        // The step proof's group half, x_hat on the step statement, its
        // previous accumulator the one the predecessor's messages hash.
        let proof = ProofIn {
            previous: vec![previous_sg.clone()],
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
            public: step_words
                .iter()
                .zip(step::bounds(shape.wrap_rounds))
                .map(|(w, bound)| PublicWord {
                    value: value(w),
                    bound,
                })
                .collect(),
            perm: type1(&st.perm),
            zeta_to_srs_length: type1(&st.zeta_to_srs_length),
            zeta_to_domain_size: type1(&st.zeta_to_domain_size),
            xi: value(&st.xi),
            combined_inner_product: type1(&st.combined_inner_product),
            b: type1(&st.b),
        };
        let out =
            group_half_in::<Vesta, _>(&model, self.poseidon, &self.constants, &proof, &deferred);

        // The unfinalized proof finalized: the predecessor's scalar half.
        let lambda = self.predecessor.lambda;
        let challenges: Vec<Affine<E::Var, Fq>> = {
            let mut m = model.borrow_mut();
            u.bulletproof_challenges
                .iter()
                .map(|c| m.endo(&value(c), lambda))
                .collect()
        };
        let fin = {
            let mut m = model.borrow_mut();
            let inputs = Inputs {
                alpha: value(&u.alpha),
                beta: value(&u.beta),
                gamma: value(&u.gamma),
                zeta: value(&u.zeta),
                digest: value(&u.sponge_digest_before_evaluations),
                ft_eval1,
                prev_challenges: vec![],
                bulletproof_challenges: challenges.clone(),
                evals,
            };
            scalar_half(
                &mut *m,
                &self.predecessor,
                self.poseidon,
                &inputs,
                &self.gates,
            )
        };

        // The new messages for the next wrap proof: the padding, the
        // finalized challenges, the step proof's accumulator.
        let mut tail = challenges;
        tail.extend([value(&sg[0]), value(&sg[1])]);
        let messages = digest_of(&self.padding, &tail);

        let mut m = model.into_inner();
        let shift = m.constant((Fq::ONE + Fq::ONE).pow(255));
        let lin = &fin.linearization;
        for (computed, stated) in [
            (&fin.combined_inner_product, &u.combined_inner_product),
            (&fin.b, &u.b),
            (&lin.zeta_to_srs_length, &u.zeta_to_srs_length),
            (&lin.zeta_to_domain_size, &u.zeta_to_domain_size),
            (&lin.perm, &u.perm),
        ] {
            // Type2: s = 2 high + low + 2^255.
            let high = value(&stated[0]);
            let twice = m.add(&high, &high);
            let t = m.add(&twice, &value(&stated[1]));
            let decoded = m.add(&t, &shift);
            m.assert_equal(computed, &decoded);
        }
        m.assert_equal(&fin.xi, &value(&u.xi));
        m.assert_equal(&value(&u.should_finalize), &Affine::constant(Fq::ONE));

        // The step proof's IPA, its draws the statement's.
        m.env().connect(&out.left[0], &out.right[0]);
        m.env().connect(&out.left[1], &out.right[1]);
        let d = &out.drawn;
        for (drawn, stated) in [
            (&d.alpha, &st.alpha),
            (&d.beta, &st.beta),
            (&d.gamma, &st.gamma),
            (&d.zeta, &st.zeta),
            (&d.digest, &st.sponge_digest_before_evaluations),
            (&messages, &st.messages_for_next_wrap_proof),
        ] {
            m.assert_equal(drawn, &value(stated));
        }
        for (drawn, stated) in out.prechallenges.iter().zip(&st.bulletproof_challenges) {
            m.assert_equal(drawn, &value(stated));
        }
        m.assert_below(&value(&st.sponge_digest_before_evaluations), WINDOW_BITS);
        m.assert_below(&value(&st.messages_for_next_wrap_proof), WINDOW_BITS);
        m.assert_equal(&value(&st.branch_data), &Affine::constant(self.branch_data));
    }
}

/// A prover index of the circuit `wrap`, on a Pallas SRS of its domain's
/// size.
///
/// # Errors
///
/// [`Error::Circuit`] when the circuit is not Kimchi rows.
pub fn index<const S: usize, const W: usize>(
    wrap: &WrapOne<'_, S, W>,
) -> Result<ProverIndex<Pallas>, Error> {
    let rows = wrap.rows();
    let gates = compiled::circuit_rows::<Fq, _, VARIABLES, DEGREE>(wrap, rows, 0)
        .map_err(|_| Error::Circuit)?;
    let cs = ConstraintSystem::create(gates)
        .public(S)
        .prev_challenges(1)
        .build()
        .map_err(|_| Error::Circuit)?;
    let depth = cs.domain.d1.size();
    Ok(ProverIndex::create(cs, Srs::create(depth)))
}

/// The wrap proof of `statement`, with the private input `private` and the
/// previous challenge `previous` ([`statement`]'s).
///
/// # Errors
///
/// [`Error::Circuit`] for a circuit that is not Kimchi rows,
/// [`Error::Prove`] when Kimchi refuses the witness.
pub fn prove_wrap<const S: usize, const W: usize, R>(
    wrap: &WrapOne<'_, S, W>,
    index: &ProverIndex<Pallas>,
    statement: &Statement<Fq>,
    private: &[Fq],
    previous: &RecursionChallenge<Pallas>,
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
        core::slice::from_ref(previous),
        &wrap_gates(),
        (&base, &scalar),
        rng,
    )
    .map_err(|_| Error::Prove)
}

/// Verify a wrap proof of a step proof with one predecessor: its previous
/// challenge the accumulator the messages hash (the step messages' `sg`,
/// the new messages' challenges), the step proof's previous challenges
/// those its messages hold, the wrap proof and the step proof's scalar half,
/// messages and accumulator ([`wrap::verify_wrap`]), then the step messages
/// its statement passes: their digest, the trusted wrap key
/// `predecessor_key`, and the predecessor's accumulator on its SRS with
/// the challenges the new messages hold.
///
/// # Errors
///
/// [`Error::Wrap`] from [`wrap::verify_wrap`], [`Error::Accumulator`] for
/// step messages that are not the statement's or a false accumulator.
#[allow(clippy::too_many_arguments)]
pub fn verify<R>(
    wrap_index: &VerifierIndex<Pallas>,
    wrap_srs: &Srs<Pallas>,
    wrap_proof: &ProverProof<Pallas>,
    statement: &Statement<Fq>,
    messages: &MessagesForNextWrapProof,
    step: (&VerifierIndex<Vesta>, &Srs<Vesta>, &StepEvaluations),
    step_messages: &MessagesForNextStepProof,
    predecessor_key: (&VerifierIndex<Pallas>, &Srs<Pallas>),
    rng: &mut R,
) -> Result<(), Error>
where
    R: Sampler<Fq>,
{
    let carried: Vec<((Fp, Fp), &Vec<Fq>)> = wrap_proof
        .prev_challenges
        .iter()
        .map(|c| (coordinates(&c.comm), &c.chals))
        .collect();
    let expected = step_messages
        .challenge_polynomial_commitments
        .last()
        .zip(messages.old_bulletproof_challenges.last())
        .map(|(sg, chals)| vec![((sg[0], sg[1]), chals)]);
    let (_, _, evaluations) = step;
    if expected != Some(carried)
        || evaluations.prev_challenges != step_messages.old_bulletproof_challenges
    {
        return Err(Error::Accumulator);
    }
    wrap::verify_wrap(
        wrap_index, wrap_srs, wrap_proof, statement, messages, step, rng,
    )
    .map_err(Error::Wrap)?;
    let (key, srs) = predecessor_key;
    let digest: Fq = Word::of(step_messages.digest()).reduce();
    let (Some(old), Some(sg)) = (
        messages.old_bulletproof_challenges.last(),
        step_messages.challenge_polynomial_commitments.last(),
    ) else {
        return Err(Error::Accumulator);
    };
    let accumulator = coordinates(&msm(&srs.g, &b_poly_coefficients(old)));
    if digest != statement.messages_for_next_step_proof
        || step_messages.index != step::index_elements(key)
        || accumulator != (sg[0], sg[1])
    {
        return Err(Error::Accumulator);
    }
    Ok(())
}
