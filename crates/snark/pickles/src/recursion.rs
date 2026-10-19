//! Recursion: one step circuit and one wrap circuit that verify each other,
//! iterated without bound (Pickles' self-recursive rule, as in Mina's
//! `simple_chain`).
//!
//! [`RecStep`] proves the rule (the state plus one) and verifies a proof of
//! [`RecWrap`]; [`RecWrap`] wraps a proof of [`RecStep`] and finalizes the
//! wrap proof that step proof verified. Each verifies proofs of the other,
//! so each needs the other's constants before either index exists: the
//! two domains are fixed in advance ([`Params`]), and every constant either
//! circuit takes of the other, or of itself, comes from the shapes
//! (`scalar::Constants::standard`, `group::Constants::shape`). The step
//! circuit takes the wrap key as a witness, so it does not depend on the
//! wrap circuit at all; the wrap circuit takes the step key as a constant.
//!
//! The base case is the same step circuit, verifying a dummy predecessor
//! ([`dummy`]) with `must_verify` false: Mina's base case, a dummy proof
//! that need not verify. The dummy is consistent everywhere but its IPA:
//! its deferred values are the scalar halves of its evaluations, its
//! digests the hashes of its messages, its accumulators the commitments of
//! their challenge polynomials. So the only check the step circuit gates is
//! the IPA's final equation, by `must_verify`, which the rule sets to
//! `state != 0`; `should_finalize` is `must_verify`.
//!
//! The accumulators cross over as in [`crate::step`]: each step proof
//! carries the accumulator of the step proof its predecessor wrapped, each
//! wrap proof that of the wrap proof its step proof verified.
//!
//! Deviations from Mina: one rule of one predecessor; the domains are ours
//! and fixed by the caller; the padding and the dummy's values are ours.

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
    gates::KimchiGates,
    oracle::{ChallengeModel, FrSponge, ScalarChallenge, coordinates},
    prover::{ProofEvaluations, ProverIndex, ProverProof, prove},
    rng::Sampler,
    srs::Srs,
    verifier::VerifyError,
    verifier_index::VerifierIndex,
};
use arrabbiata_kimchi_gadgets::generic::PublicInput;
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, KimchiSponge, pallas_base, vesta_base};

use crate::{
    curve::{Shifted, held, point_at},
    group::{
        Constants, DeferredIn, Key, KeyIn, ProofIn, PublicWord, group_half_in,
        group_half_with_key_in, key_digest_in,
    },
    model::{Affine, Circuit, load},
    scalar::{self, Inputs, Outputs, scalar_half},
    statement::{BranchData, ProofsVerified, Sort, Type2, WINDOW_BITS, Word},
    step::{
        self, MessagesForNextStepProof, WrapEvaluations, index_elements, to_fp, to_fq, type1_shift,
        type2,
    },
    wrap::{self, MessagesForNextWrapProof, StepEvaluations, shifted, step_gates, wrap_gates},
    wrap_one::computed,
};

/// The column variables of the compiled circuits.
pub const VARIABLES: usize = 64;

/// The degree of the compiled circuits.
pub const DEGREE: usize = 7;

/// The zero-knowledge rows of an index of one chunk.
pub const ZK_ROWS: usize = 3;

/// The commitments of a key of the standard gates: seven sigmas, fifteen
/// coefficients, six selectors.
pub const KEY_COMMITMENTS: usize = 7 + COLUMNS + 6;

/// The evaluations of a proof of the standard gates of one chunk: the
/// public input, the witness, `z`, six sigmas, the coefficients and six
/// selectors, each at `zeta` and `zeta w`.
pub const EVALUATIONS: usize = 2 * (1 + COLUMNS + 1 + 6 + COLUMNS + 6);

/// The two domains, fixed in advance: each circuit is built on a domain of
/// `2^step_log2`, resp. `2^wrap_log2`, rows, on an SRS of that size, so a
/// proof of it has as many IPA rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// The step circuit's domain, its log2.
    pub step_log2: u32,
    /// The wrap circuit's domain, its log2.
    pub wrap_log2: u32,
}

impl Params {
    /// A step proof's IPA rounds.
    #[must_use]
    pub const fn step_rounds(&self) -> usize {
        self.step_log2 as usize
    }

    /// A wrap proof's IPA rounds.
    #[must_use]
    pub const fn wrap_rounds(&self) -> usize {
        self.wrap_log2 as usize
    }

    /// The step circuit's domain size.
    #[must_use]
    pub const fn step_domain(&self) -> usize {
        1 << self.step_log2
    }

    /// The wrap circuit's domain size.
    #[must_use]
    pub const fn wrap_domain(&self) -> usize {
        1 << self.wrap_log2
    }

    /// The size of a step statement ([`step::Statement`]).
    #[must_use]
    pub const fn step_statement(&self) -> usize {
        19 + self.wrap_rounds()
    }

    /// The size of a wrap statement ([`wrap::Statement`]).
    #[must_use]
    pub const fn wrap_statement(&self) -> usize {
        14 + self.step_rounds()
    }

    /// The points of a proof's commitments and opening (one chunk, a
    /// quotient of seven), `delta` and `sg` included, on `rounds` rounds.
    const fn points(rounds: usize) -> usize {
        COLUMNS + 1 + 7 + 2 * rounds + 2
    }

    /// The size of the step circuit's private input: the wrap key, the
    /// wrap statement, the wrap proof (points, `z1` and `z2` as Type2), its
    /// previous accumulator's commitment, the previous step proof's
    /// previous challenges, its `ft(zeta w)` and evaluations, the state.
    #[must_use]
    pub const fn step_private(&self) -> usize {
        2 * KEY_COMMITMENTS
            + self.wrap_statement()
            + 2 * Self::points(self.wrap_rounds())
            + 4
            + 2
            + self.step_rounds()
            + 1
            + EVALUATIONS
            + 1
    }

    /// The size of the wrap circuit's private input: the step proof
    /// ([`wrap::witness`]), the unfinalized proof, the previous wrap proof's
    /// `ft(zeta w)`, evaluations and previous challenges, and the previous
    /// messages' commitment.
    #[must_use]
    pub const fn wrap_private(&self) -> usize {
        2 * Self::points(self.step_rounds())
            + 2
            + 17
            + self.wrap_rounds()
            + 1
            + EVALUATIONS
            + self.wrap_rounds()
            + 2
    }
}

/// Why a recursion step cannot be made, or does not verify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A proof does not verify: its transcript has the wrong shape.
    Verify(VerifyError),
    /// The verified proof's opening does not hold.
    Opening,
    /// An index has lookups or optional gates.
    Lookups,
    /// A circuit's sizes are not the parameters'.
    Size,
    /// A circuit's domain is not the one fixed in advance: the rows it
    /// compiles to need another domain.
    Domain {
        /// The domain it needs.
        found: usize,
    },
    /// A value of a statement is past its sort's bound.
    Bound,
    /// A deferred value is not its definition.
    Deferred(&'static str),
    /// Messages are not those a digest hashes.
    Messages,
    /// A circuit does not compile to Kimchi rows.
    Circuit,
    /// A proof could not be made.
    Prove,
}

/// A wrap proof's own deferred values: the scalars of `Fq` its group half
/// takes on trust, which the next wrap circuit finalizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrapDeferred {
    /// `perm`.
    pub perm: Fq,
    /// `zeta^m`.
    pub zeta_to_srs_length: Fq,
    /// `zeta^n`.
    pub zeta_to_domain_size: Fq,
    /// The 128-bit challenge `xi` comes from.
    pub xi: Fq,
    /// The combined inner product.
    pub combined_inner_product: Fq,
    /// `b`.
    pub b: Fq,
}

/// A wrap proof as the step circuit reads it, with what the next circuits
/// need of it: a proof of [`RecWrap`] ([`WrapInput::of`]), or the dummy of
/// the base case ([`dummy`]).
#[derive(Clone, Debug)]
pub struct WrapInput {
    /// Its statement.
    pub statement: wrap::Statement<Fq>,
    /// The messages its statement's digest hashes.
    pub messages: MessagesForNextWrapProof,
    /// Its commitments and opening, its previous accumulator's commitment
    /// among them.
    pub proof: ProofIn<Projective<Pallas>, Fq>,
    /// Its deferred values.
    pub deferred: WrapDeferred,
    /// Its evaluations, for the wrap circuit that finalizes it.
    pub evaluations: WrapEvaluations,
    /// Its previous challenges (the accumulator it carries), mapped.
    pub previous_challenges: Vec<Fq>,
}

/// The proof part of a Kimchi proof, as the group half reads it.
fn proof_in<C: Glv>(proof: &ProverProof<C>) -> ProofIn<Projective<C>, C::Scalar> {
    let opening = &proof.proof;
    ProofIn {
        previous: proof.prev_challenges.iter().map(|p| p.comm).collect(),
        witness: core::array::from_fn(|i| proof.w_comm[i].chunks.clone()),
        z: proof.z_comm.chunks.clone(),
        t: proof.t_comm.chunks.clone(),
        rounds: opening.lr.clone(),
        delta: opening.delta,
        z1: opening.z1,
        z2: opening.z2,
        sg: opening.sg,
    }
}

impl WrapInput {
    /// A proof of the wrap circuit on `index` and `srs`, of `statement`,
    /// whose digest hashes `messages`.
    ///
    /// # Errors
    ///
    /// [`Error::Verify`] for a proof of the wrong shape, [`Error::Size`]
    /// for a proof that does not carry one accumulator.
    pub fn of(
        index: &VerifierIndex<Pallas>,
        srs: &Srs<Pallas>,
        proof: &ProverProof<Pallas>,
        statement: &wrap::Statement<Fq>,
        messages: &MessagesForNextWrapProof,
    ) -> Result<Self, Error> {
        let base = KimchiSponge::new(pallas_base());
        let scalar = KimchiSponge::new(vesta_base());
        let values = expand_deferred(
            index,
            srs,
            proof,
            &statement.to_vec(),
            &wrap_gates(),
            (&base, &scalar),
        )
        .map_err(Error::Verify)?;
        let [previous] = &proof.prev_challenges[..] else {
            return Err(Error::Size);
        };
        Ok(Self {
            statement: statement.clone(),
            messages: messages.clone(),
            proof: proof_in(proof),
            deferred: WrapDeferred {
                perm: values.plonk.perm,
                zeta_to_srs_length: values.plonk.zeta_to_srs_length,
                zeta_to_domain_size: values.plonk.zeta_to_domain_size,
                xi: values.xi.0,
                combined_inner_product: values.combined_inner_product,
                b: values.b,
            },
            evaluations: WrapEvaluations::of(proof),
            previous_challenges: previous.chals.clone(),
        })
    }
}

/// The branch data of a step proof of one proof verified on a domain of
/// `2^log2` rows, as the element of `F` its word is.
fn branch<F: arrabbiata_field::PrimeField>(log2: u32) -> F {
    let log2 = u8::try_from(log2).unwrap_or(u8::MAX);
    BranchData::encode(&(ProofsVerified::N1, log2))[0].reduce()
}

/// A key of the standard gates of one chunk, every chunk present, at the
/// point `p`: the shape a circuit reads of a key it takes as a witness.
fn full_key<C: Glv>(p: Projective<C>) -> Key<Projective<C>> {
    let one = || vec![Some(p)];
    Key {
        sigma: core::array::from_fn(|_| one()),
        coefficients: core::array::from_fn(|_| one()),
        selectors: core::array::from_fn(|_| one()),
    }
}

/// The step circuit: its input the step statement (`S` values) and the
/// private input (`W` values, [`Params::step_private`]).
#[derive(Clone, Debug)]
pub struct RecStep<'a, const S: usize, const W: usize> {
    params: Params,
    wrap_constants: Constants<Pallas>,
    own: scalar::Constants<Fp>,
    branch_data: Fp,
    poseidon: &'a KimchiPoseidon<Fp>,
    gates: KimchiGates<Fp>,
}

impl<'a, const S: usize, const W: usize> RecStep<'a, S, W> {
    /// The step circuit of `params`, verifying wrap proofs on `wrap_srs`,
    /// with Kimchi's `poseidon` over `Fp`. It reads nothing of the wrap
    /// circuit but its shape: the key is a witness.
    ///
    /// # Errors
    ///
    /// [`Error::Size`] when `S` or `W` is not the parameters', or an SRS
    /// is not of its domain's size.
    pub fn new(
        params: Params,
        wrap_srs: &Srs<Pallas>,
        poseidon: &'a KimchiPoseidon<Fp>,
    ) -> Result<Self, Error> {
        if params.step_statement() != S
            || params.step_private() != W
            || wrap_srs.g.len() != params.wrap_domain()
        {
            return Err(Error::Size);
        }
        let n = params.wrap_domain();
        let wrap_constants = Constants::shape(
            wrap_srs,
            n,
            params.wrap_statement(),
            n,
            full_key(Projective::<Pallas>::generator()),
            Fp::ZERO,
        );
        let m = params.step_domain();
        let own = scalar::Constants::standard::<Vesta>(m, m, ZK_ROWS).map_err(|_| Error::Size)?;
        Ok(Self {
            params,
            wrap_constants,
            own,
            branch_data: branch(params.step_log2),
            poseidon,
            gates: step_gates(),
        })
    }
}

impl<const S: usize, const W: usize> TypedGadget<Fp> for RecStep<'_, S, W> {
    type Input = (Array<Scalar, S>, Array<Scalar, W>);
    type Output = Unit;
    const NAME: &'static str = "pickles-recursive-step";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        arrabbiata_gadget::measure(self, arrabbiata_kimchi_gadgets::LAYOUT).unwrap_or(0)
    }

    #[allow(clippy::too_many_lines)]
    fn synthesize<E>(&self, env: &mut E, (statement, private): Of<Self::Input, E::Var>)
    where
        E: Env<Fp>,
    {
        let public: Vec<E::Var> = statement
            .iter()
            .map(|v| env.call(&PublicInput, v.clone()))
            .collect();
        let cells = load(env, &private);
        let (rs, rw) = (self.params.step_rounds(), self.params.wrap_rounds());
        let Some(st) = step::Statement::of(rw, &public) else {
            unreachable!("the statement has the parameters' size")
        };
        let mut next = cells.into_iter();
        let mut take = || match next.next() {
            Some(v) => v,
            None => unreachable!("the private input has the parameters' size"),
        };
        let key = self.wrap_constants.key.map(&mut |_| [take(), take()]);
        let wrap_words: Vec<E::Var> = (0..14 + rs).map(|_| take()).collect();
        let Some(ws) = wrap::Statement::of(rs, &wrap_words) else {
            unreachable!("the wrap statement has the parameters' size")
        };
        let mut point = || [take(), take()];
        let witness: [Vec<[E::Var; 2]>; COLUMNS] = core::array::from_fn(|_| vec![point()]);
        let z = vec![point()];
        let t = (0..7).map(|_| point()).collect();
        let rounds = (0..rw).map(|_| (point(), point())).collect();
        let delta = point();
        let sg = point();
        let value = |v: &E::Var| Affine::of(v.clone());
        let type2_of = |high: E::Var, low: E::Var| Shifted {
            high: value(&high),
            low: Some(value(&low)),
        };
        let z1 = type2_of(take(), take());
        let z2 = type2_of(take(), take());
        let previous_sg = [take(), take()];
        let previous_challenges: Vec<Affine<E::Var, Fp>> =
            (0..rs).map(|_| value(&take())).collect();
        let ft_eval1 = value(&take());
        let evals = self.own.shape.map(&mut |()| value(&take()));
        let state = value(&take());

        // The wrap proof's group half, x_hat on the wrap statement's words,
        // its previous accumulator the one the previous messages hash.
        let u = &st.unfinalized;
        let held = |f: &[E::Var; 2]| type2_of(f[0].clone(), f[1].clone());
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
            public: wrap_words
                .iter()
                .zip(wrap::bounds(rs))
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

        // The messages over the witnessed key: the previous step proof's
        // (its state, the accumulator commitment, its previous challenges)
        // and the next one's.
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
        let mut previous_tail = vec![
            state.clone(),
            value(&previous_sg[0]),
            value(&previous_sg[1]),
        ];
        previous_tail.extend(previous_challenges.iter().cloned());
        let previous = digest_of(&previous_tail);
        let next_state = {
            let mut m = model.borrow_mut();
            let one = m.constant(Fp::ONE);
            m.add(&state, &one)
        };
        let lambda = self.own.lambda;
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

        // The wrap statement finalized: the previous step proof's scalar
        // half, on this circuit's own constants.
        let mut m = model.into_inner();
        let inputs = Inputs {
            alpha: value(&ws.alpha),
            beta: value(&ws.beta),
            gamma: value(&ws.gamma),
            zeta: value(&ws.zeta),
            digest: value(&ws.sponge_digest_before_evaluations),
            ft_eval1,
            prev_challenges: vec![previous_challenges],
            bulletproof_challenges,
            evals,
        };
        let fin = scalar_half(&mut m, &self.own, self.poseidon, &inputs, &self.gates);
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

        // must_verify: state != 0, as b = state / state, and state (1 - b)
        // = 0. The IPA's final equation holds when it is one.
        let must_verify = {
            let inverse = m.witness([&state], |[v]| v.inverse().unwrap_or(Fp::ZERO));
            m.mul(&state, &inverse)
        };
        let one = m.constant(Fp::ONE);
        let not = m.sub(&one, &must_verify);
        let product = m.mul(&state, &not);
        m.assert_equal(&product, &Affine::constant(Fp::ZERO));
        for k in 0..2 {
            let difference = m.sub(&value(&out.left[k]), &value(&out.right[k]));
            let gated = m.mul(&must_verify, &difference);
            m.assert_equal(&gated, &Affine::constant(Fp::ZERO));
        }

        // The draws are the unfinalized proof's.
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
        m.assert_equal(&value(&u.should_finalize), &must_verify);
        m.assert_equal(&value(&ws.branch_data), &Affine::constant(self.branch_data));
    }
}

/// The wrap circuit: its input the wrap statement (`S` values) and the
/// private input (`W` values, [`Params::wrap_private`]).
#[derive(Clone, Debug)]
pub struct RecWrap<'a, const S: usize, const W: usize> {
    params: Params,
    step_constants: Constants<Vesta>,
    own: scalar::Constants<Fq>,
    branch_data: Fq,
    poseidon: &'a KimchiPoseidon<Fq>,
    gates: KimchiGates<Fq>,
    padding: Vec<Fq>,
}

impl<'a, const S: usize, const W: usize> RecWrap<'a, S, W> {
    /// The wrap circuit of `params`, wrapping proofs of the step circuit on
    /// `step` and `step_srs`, with Kimchi's `poseidon` over `Fq`; its
    /// messages start with `padding`.
    ///
    /// # Errors
    ///
    /// [`Error::Lookups`] for a step index with lookups, [`Error::Size`]
    /// when `S` or `W` is not the parameters'.
    pub fn new(
        params: Params,
        step: &VerifierIndex<Vesta>,
        step_srs: &Srs<Vesta>,
        poseidon: &'a KimchiPoseidon<Fq>,
        padding: Vec<Fq>,
    ) -> Result<Self, Error> {
        if params.wrap_statement() != S || params.wrap_private() != W {
            return Err(Error::Size);
        }
        let base = KimchiSponge::new(vesta_base());
        let step_constants = Constants::new(step, step_srs, &base).ok_or(Error::Lookups)?;
        let n = params.wrap_domain();
        let own = scalar::Constants::standard::<Pallas>(n, n, ZK_ROWS).map_err(|_| Error::Size)?;
        Ok(Self {
            params,
            step_constants,
            own,
            branch_data: branch(params.step_log2),
            poseidon,
            gates: wrap_gates(),
            padding,
        })
    }
}

impl<const S: usize, const W: usize> TypedGadget<Fq> for RecWrap<'_, S, W> {
    type Input = (Array<Scalar, S>, Array<Scalar, W>);
    type Output = Unit;
    const NAME: &'static str = "pickles-recursive-wrap";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        arrabbiata_gadget::measure(self, arrabbiata_kimchi_gadgets::LAYOUT).unwrap_or(0)
    }

    #[allow(clippy::too_many_lines)]
    fn synthesize<E>(&self, env: &mut E, (statement, private): Of<Self::Input, E::Var>)
    where
        E: Env<Fq>,
    {
        let public: Vec<E::Var> = statement
            .iter()
            .map(|v| env.call(&PublicInput, v.clone()))
            .collect();
        let cells = load(env, &private);
        let (rs, rw) = (self.params.step_rounds(), self.params.wrap_rounds());
        let Some(st) = wrap::Statement::of(rs, &public) else {
            unreachable!("the statement has the parameters' size")
        };
        let mut next = cells.into_iter();
        let mut take = || match next.next() {
            Some(v) => v,
            None => unreachable!("the private input has the parameters' size"),
        };
        let mut point = || [take(), take()];
        let witness: [Vec<[E::Var; 2]>; COLUMNS] = core::array::from_fn(|_| vec![point()]);
        let z = vec![point()];
        let t = (0..7).map(|_| point()).collect();
        let rounds = (0..rs).map(|_| (point(), point())).collect();
        let delta = point();
        let sg = point();
        let value = |v: &E::Var| Affine::of(v.clone());
        let type1 = |v: &E::Var| Shifted {
            high: value(v),
            low: None,
        };
        let z1 = type1(&take());
        let z2 = type1(&take());
        let unfinalized_words: Vec<E::Var> = (0..17 + rw).map(|_| take()).collect();
        let ft_eval1 = value(&take());
        let evals = self.own.shape.map(&mut |()| value(&take()));
        let previous_challenges: Vec<Affine<E::Var, Fq>> =
            (0..rw).map(|_| value(&take())).collect();
        let previous_sg = [take(), take()];

        let model = RefCell::new(Circuit::new(env));
        let digest_of = |tail: &[Affine<E::Var, Fq>]| {
            let mut sponge = FrSponge::in_model(&model, self.poseidon);
            for c in &self.padding {
                let c = model.borrow_mut().constant(*c);
                sponge.absorb_value(&c);
            }
            for v in tail {
                sponge.absorb_value(v);
            }
            sponge.squeeze_value()
        };

        // The step statement: the unfinalized proof, this statement's
        // messages for the next step proof, and the previous wrap proof's
        // messages, hashed here from its previous challenges and the
        // accumulator commitment the step proof carries.
        let mut previous_tail = previous_challenges.clone();
        previous_tail.extend([value(&previous_sg[0]), value(&previous_sg[1])]);
        let previous = digest_of(&previous_tail);
        let previous_cell = model.borrow_mut().cell(&previous);
        let mut step_words = unfinalized_words;
        step_words.push(st.messages_for_next_step_proof.clone());
        step_words.push(previous_cell);
        let Some(step_statement) = step::Statement::of(rw, &step_words) else {
            unreachable!("the step statement has the parameters' size")
        };
        let u = &step_statement.unfinalized;

        // The step proof's group half, its previous accumulator the one the
        // previous messages hash.
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
                .zip(step::bounds(rw))
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
        let out = group_half_in::<Vesta, _>(
            &model,
            self.poseidon,
            &self.step_constants,
            &proof,
            &deferred,
        );

        // The unfinalized proof finalized: the previous wrap proof's scalar
        // half, on this circuit's own constants.
        let lambda = self.own.lambda;
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
                prev_challenges: vec![previous_challenges],
                bulletproof_challenges: challenges.clone(),
                evals,
            };
            scalar_half(&mut *m, &self.own, self.poseidon, &inputs, &self.gates)
        };

        // The new messages: the padding, the finalized challenges, the step
        // proof's accumulator.
        let mut tail = challenges;
        tail.extend([value(&sg[0]), value(&sg[1])]);
        let messages = digest_of(&tail);

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
            let high = value(&stated[0]);
            let twice = m.add(&high, &high);
            let t = m.add(&twice, &value(&stated[1]));
            let decoded = m.add(&t, &shift);
            m.assert_equal(computed, &decoded);
        }
        m.assert_equal(&fin.xi, &value(&u.xi));
        let finalize = value(&u.should_finalize);
        let squared = m.mul(&finalize, &finalize);
        m.assert_equal(&squared, &finalize);

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

/// The scalar half of a wrap proof natively, on the wrap constants `own`.
fn wrap_scalar_half(
    own: &scalar::Constants<Fq>,
    unfinalized: &step::Unfinalized<Fp>,
    evaluations: &WrapEvaluations,
    previous: &[Fq],
) -> Outputs<Fq> {
    let inputs = Inputs {
        alpha: to_fq(unfinalized.alpha),
        beta: to_fq(unfinalized.beta),
        gamma: to_fq(unfinalized.gamma),
        zeta: to_fq(unfinalized.zeta),
        digest: to_fq(unfinalized.sponge_digest_before_evaluations),
        ft_eval1: evaluations.ft_eval1,
        prev_challenges: vec![previous.to_vec()],
        bulletproof_challenges: computed(&unfinalized.bulletproof_challenges),
        evals: evaluations.evals.clone(),
    };
    scalar_half(&mut Native, own, &vesta_base(), &inputs, &wrap_gates())
}

/// The scalar half of a step proof natively, on the step constants `own`.
fn step_scalar_half(
    own: &scalar::Constants<Fp>,
    statement: &wrap::Statement<Fq>,
    evaluations: &StepEvaluations,
) -> Outputs<Fp> {
    let small = |x: Fq| -> Fp { to_fp(x) };
    let inputs = Inputs {
        alpha: small(statement.alpha),
        beta: small(statement.beta),
        gamma: small(statement.gamma),
        zeta: small(statement.zeta),
        digest: small(statement.sponge_digest_before_evaluations),
        ft_eval1: evaluations.ft_eval1,
        prev_challenges: evaluations.prev_challenges.clone(),
        bulletproof_challenges: step::deferred_challenges(statement),
        evals: evaluations.evals.clone(),
    };
    scalar_half(&mut Native, own, &pallas_base(), &inputs, &step_gates())
}

/// The step statement of verifying `input` from the previous step messages
/// `previous` (its state, and the accumulator and challenges the previous
/// step proof hashed), the previous step proof's evaluations
/// `evaluations`; the new messages; the private input; the previous
/// challenge the step proof carries. `wrap` is the wrap circuit's index and
/// SRS, whose key the messages hash.
///
/// # Errors
///
/// [`Error::Messages`] when a digest is not its messages',
/// [`Error::Opening`] for a predecessor that must verify and does not,
/// [`Error::Bound`] for a value past its bound.
#[allow(clippy::type_complexity)]
pub fn step_statement(
    params: Params,
    wrap: (&VerifierIndex<Pallas>, &Srs<Pallas>),
    input: &WrapInput,
    previous: &MessagesForNextStepProof,
    evaluations: &StepEvaluations,
) -> Result<
    (
        step::Statement<Fp>,
        MessagesForNextStepProof,
        Vec<Fp>,
        RecursionChallenge<Vesta>,
    ),
    Error,
> {
    let ws = &input.statement;
    let state = previous.app_state.first().copied().unwrap_or(Fp::ZERO);
    let must_verify = state != Fp::ZERO;
    if to_fq(previous.digest()) != ws.messages_for_next_step_proof
        || input.messages.digest() != ws.messages_for_next_wrap_proof
    {
        return Err(Error::Messages);
    }
    let (wrap_index, wrap_srs) = wrap;
    let base = KimchiSponge::new(pallas_base());
    let constants = Constants::new(wrap_index, wrap_srs, &base).ok_or(Error::Lookups)?;
    let rs = params.step_rounds();
    let words: Vec<Fp> = ws.to_vec().iter().map(|w| to_fp(*w)).collect();
    let d = &input.deferred;
    let deferred = DeferredIn {
        public: words
            .iter()
            .zip(wrap::bounds(rs))
            .map(|(w, bound)| PublicWord { value: *w, bound })
            .collect(),
        perm: d.perm,
        zeta_to_srs_length: d.zeta_to_srs_length,
        zeta_to_domain_size: d.zeta_to_domain_size,
        xi: to_fp(d.xi),
        combined_inner_product: d.combined_inner_product,
        b: d.b,
    };
    let model = RefCell::new(Native);
    let out =
        group_half_in::<Pallas, _>(&model, &pallas_base(), &constants, &input.proof, &deferred);
    if must_verify && out.left != out.right {
        return Err(Error::Opening);
    }
    let chals = step::deferred_challenges(ws);
    let (sx, sy) = coordinates(&input.proof.sg);
    let messages = MessagesForNextStepProof {
        index: index_elements(wrap_index),
        app_state: vec![step::rule(state)],
        challenge_polynomial_commitments: vec![[sx, sy]],
        old_bulletproof_challenges: vec![chals.clone()],
    };
    let flag = if must_verify { Fp::ONE } else { Fp::ZERO };
    let statement = step::Statement {
        unfinalized: step::Unfinalized {
            combined_inner_product: type2(d.combined_inner_product),
            b: type2(d.b),
            zeta_to_srs_length: type2(d.zeta_to_srs_length),
            zeta_to_domain_size: type2(d.zeta_to_domain_size),
            perm: type2(d.perm),
            sponge_digest_before_evaluations: out.drawn.digest,
            beta: out.drawn.beta,
            gamma: out.drawn.gamma,
            alpha: out.drawn.alpha,
            zeta: out.drawn.zeta,
            xi: to_fp(d.xi),
            bulletproof_challenges: out.prechallenges.clone(),
            should_finalize: flag,
        },
        messages_for_next_step_proof: messages.digest(),
        messages_for_next_wrap_proof: to_fp(ws.messages_for_next_wrap_proof),
    };
    let in_bounds = statement
        .to_vec()
        .iter()
        .zip(step::bounds(params.wrap_rounds()))
        .all(|(v, n)| Word::of(*v).below(n));
    if !in_bounds {
        return Err(Error::Bound);
    }
    let [mx, my] = input.messages.challenge_polynomial_commitment;
    let previous_challenge = RecursionChallenge {
        chals,
        comm: point_at::<Vesta>(mx, my).ok_or(Error::Messages)?,
    };

    let mut private: Vec<Fp> = Vec::new();
    for p in constants.key.chunks().flatten() {
        let (x, y) = coordinates(p);
        private.extend([x, y]);
    }
    private.extend(&words);
    let proof = &input.proof;
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
    let [previous_sg] = &proof.previous[..] else {
        return Err(Error::Size);
    };
    let (px, py) = coordinates(previous_sg);
    private.extend([px, py]);
    let [previous_chals] = &evaluations.prev_challenges[..] else {
        return Err(Error::Size);
    };
    private.extend(previous_chals);
    private.push(evaluations.ft_eval1);
    let _ = evaluations.evals.map(&mut |v| private.push(*v));
    private.push(state);
    if private.len() != params.step_private() {
        return Err(Error::Size);
    }
    Ok((statement, messages, private, previous_challenge))
}

/// A step proof as the wrap circuit reads it.
#[derive(Clone, Copy, Debug)]
pub struct StepInput<'a> {
    /// The step index.
    pub index: &'a VerifierIndex<Vesta>,
    /// The step SRS.
    pub srs: &'a Srs<Vesta>,
    /// The proof.
    pub proof: &'a ProverProof<Vesta>,
    /// Its statement.
    pub statement: &'a step::Statement<Fp>,
    /// The messages its digest hashes.
    pub messages: &'a MessagesForNextStepProof,
}

/// The wrap statement of wrapping `step`, whose predecessor is
/// `predecessor` (the wrap proof the step proof verified, finalized here);
/// the new messages; the private input; the previous challenge the wrap
/// proof carries.
///
/// # Errors
///
/// [`Error::Verify`] for a step proof of the wrong shape, [`Error::Opening`]
/// when its opening does not hold, [`Error::Messages`] when a digest is not
/// its messages', [`Error::Deferred`] when the predecessor does not
/// finalize, [`Error::Bound`] for a value past its bound.
#[allow(clippy::type_complexity)]
pub fn wrap_statement(
    params: Params,
    step: &StepInput<'_>,
    predecessor: &WrapInput,
    padding: &[Fq],
) -> Result<
    (
        wrap::Statement<Fq>,
        MessagesForNextWrapProof,
        Vec<Fq>,
        RecursionChallenge<Pallas>,
    ),
    Error,
> {
    let u = &step.statement.unfinalized;
    let [px, py] = predecessor.messages.challenge_polynomial_commitment;
    let carried: Vec<(Fq, Fq)> = step
        .proof
        .prev_challenges
        .iter()
        .map(|c| coordinates(&c.comm))
        .collect();
    let expected_old = vec![padding.to_vec(), predecessor.previous_challenges.clone()];
    if Word::of(predecessor.messages.digest())
        != Word::of(step.statement.messages_for_next_wrap_proof)
        || step.messages.digest() != step.statement.messages_for_next_step_proof
        || carried != vec![(px, py)]
        || predecessor.messages.old_bulletproof_challenges != expected_old
    {
        return Err(Error::Messages);
    }

    // The unfinalized proof finalized natively, as the circuit does.
    let n = params.wrap_domain();
    let own = scalar::Constants::standard::<Pallas>(n, n, ZK_ROWS).map_err(|_| Error::Size)?;
    let fin = wrap_scalar_half(
        &own,
        u,
        &predecessor.evaluations,
        &predecessor.previous_challenges,
    );
    let decode = |[high, low]: [Fp; 2]| {
        Type2::<Pallas>::decode(&[Word::of(high), Word::of(low)]).ok_or(Error::Bound)
    };
    let lin = &fin.linearization;
    for (name, value, stated) in [
        (
            "combined_inner_product",
            fin.combined_inner_product,
            u.combined_inner_product,
        ),
        ("b", fin.b, u.b),
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
        if value != decode(stated)? {
            return Err(Error::Deferred(name));
        }
    }
    if Word::of(fin.xi) != Word::of(u.xi) {
        return Err(Error::Deferred("xi"));
    }

    // The step proof's group half natively.
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
    .map_err(Error::Verify)?;
    let constants = Constants::new(step.index, step.srs, &base).ok_or(Error::Lookups)?;
    let proof = proof_in(step.proof);
    let deferred = DeferredIn {
        public: public
            .iter()
            .zip(step::bounds(params.wrap_rounds()))
            .map(|(w, bound)| PublicWord {
                value: to_fq(*w),
                bound,
            })
            .collect(),
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: to_fq(values.xi.0),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };
    let model = RefCell::new(Native);
    let out = group_half_in::<Vesta, _>(&model, &vesta_base(), &constants, &proof, &deferred);
    if out.left != out.right {
        return Err(Error::Opening);
    }
    let finalized = computed(&u.bulletproof_challenges);
    let (sx, sy) = coordinates(&out.sg);
    let messages = MessagesForNextWrapProof {
        challenge_polynomial_commitment: [sx, sy],
        old_bulletproof_challenges: vec![padding.to_vec(), finalized.clone()],
    };
    let statement = wrap::Statement {
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
        messages_for_next_step_proof: to_fq(step.statement.messages_for_next_step_proof),
        bulletproof_challenges: out.prechallenges,
        branch_data: branch(params.step_log2),
    };
    let in_bounds = statement
        .to_vec()
        .iter()
        .zip(wrap::bounds(params.step_rounds()))
        .all(|(v, n)| Word::of(*v).below(n));
    if !in_bounds {
        return Err(Error::Bound);
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
        chals: finalized,
        comm: point_at::<Pallas>(wx, wy).ok_or(Error::Messages)?,
    };

    let mut private = wrap::witness(&proof);
    private.extend(
        public[..17 + params.wrap_rounds()]
            .iter()
            .map(|w| to_fq(*w)),
    );
    private.push(predecessor.evaluations.ft_eval1);
    let _ = predecessor.evaluations.evals.map(&mut |v| private.push(*v));
    private.extend(&predecessor.previous_challenges);
    private.extend([px, py]);
    if private.len() != params.wrap_private() {
        return Err(Error::Size);
    }
    Ok((statement, messages, private, previous_challenge))
}

/// The base case's predecessor: a wrap proof that need not verify, with
/// what the first step and wrap proofs read of it, consistent everywhere
/// but its IPA; the previous step messages (the state zero); and the
/// previous step proof's evaluations. `wrap` is the wrap circuit's index
/// and SRS (its key is hashed), `step_srs` the step SRS.
///
/// # Errors
///
/// [`Error::Bound`] when a dummy value falls past its bound (about `2^-128`
/// of them), [`Error::Lookups`] for a wrap index with lookups.
#[allow(clippy::too_many_lines)]
pub fn dummy(
    params: Params,
    wrap: (&VerifierIndex<Pallas>, &Srs<Pallas>),
    step_srs: &Srs<Vesta>,
    padding: &[Fq],
) -> Result<(WrapInput, MessagesForNextStepProof, StepEvaluations), Error> {
    let (wrap_index, wrap_srs) = wrap;
    let (rs, rw) = (params.step_rounds(), params.wrap_rounds());
    let small = |k: u64| -> u128 { u128::from(k) * 0x9e37_79b9_7f4a_7c15 + 1 };
    let fq_of = |x: u128| -> Fq { Word::of_u128(x).reduce() };
    let fp_of = |x: u128| -> Fp { Word::of_u128(x).reduce() };

    // The previous step proof: its evaluations and previous challenges,
    // and the previous step messages, the state zero.
    let mut k = 0u64;
    let mut fresh = || {
        k += 1;
        small(k)
    };
    let step_own =
        scalar::Constants::standard::<Vesta>(params.step_domain(), params.step_domain(), ZK_ROWS)
            .map_err(|_| Error::Size)?;
    let lambda_p = Vesta::lambda();
    let older: Vec<Fp> = (0..rs)
        .map(|_| ScalarChallenge(fp_of(fresh())).to_field(lambda_p))
        .collect();
    let step_evaluations = StepEvaluations {
        evals: step_own.shape.map(&mut |()| fp_of(fresh())),
        ft_eval1: fp_of(fresh()),
        prev_challenges: vec![older.clone()],
    };
    let older_wrap_sg = Projective::<Pallas>::generator() * Fq::from_u64(3);
    let (ox, oy) = coordinates(&older_wrap_sg);
    let previous = MessagesForNextStepProof {
        index: index_elements(wrap_index),
        app_state: vec![Fp::ZERO],
        challenge_polynomial_commitments: vec![[ox, oy]],
        old_bulletproof_challenges: vec![older],
    };

    // The dummy's statement: its deferred values the previous step proof's
    // scalar half, its digests its messages'.
    let lambda_q = Pallas::lambda();
    let older_wrap: Vec<Fq> = (0..rw)
        .map(|_| ScalarChallenge(fq_of(fresh())).to_field(lambda_q))
        .collect();
    let mut statement = wrap::Statement {
        combined_inner_product: Fq::ZERO,
        b: Fq::ZERO,
        zeta_to_srs_length: Fq::ZERO,
        zeta_to_domain_size: Fq::ZERO,
        perm: Fq::ZERO,
        beta: fq_of(fresh()),
        gamma: fq_of(fresh()),
        alpha: fq_of(fresh()),
        zeta: fq_of(fresh()),
        xi: Fq::ZERO,
        sponge_digest_before_evaluations: fq_of(fresh()),
        messages_for_next_wrap_proof: Fq::ZERO,
        messages_for_next_step_proof: to_fq(previous.digest()),
        bulletproof_challenges: (0..rs).map(|_| fq_of(fresh())).collect(),
        branch_data: branch(params.step_log2),
    };
    let fin = step_scalar_half(&step_own, &statement, &step_evaluations);
    let lin = &fin.linearization;
    statement.combined_inner_product = shifted(fin.combined_inner_product);
    statement.b = shifted(fin.b);
    statement.zeta_to_srs_length = shifted(lin.zeta_to_srs_length);
    statement.zeta_to_domain_size = shifted(lin.zeta_to_domain_size);
    statement.perm = shifted(lin.perm);
    statement.xi = Word::of(fin.xi).reduce();
    // The step accumulator the messages hold: honest, since the first
    // step proof opens it.
    let step_challenges = step::deferred_challenges(&statement);
    let step_sg = msm(&step_srs.g, &b_poly_coefficients(&step_challenges));
    let (gx, gy) = coordinates(&step_sg);
    let messages = MessagesForNextWrapProof {
        challenge_polynomial_commitment: [gx, gy],
        old_bulletproof_challenges: vec![padding.to_vec(), older_wrap.clone()],
    };
    statement.messages_for_next_wrap_proof = messages.digest();

    // The dummy's proof: points the group half reads, its deferred values
    // the scalar half of its evaluations (computed in two passes, since
    // the IPA's challenges follow the combined inner product), and its
    // accumulator honest, since the first wrap proof opens it.
    let base = KimchiSponge::new(pallas_base());
    let constants = Constants::new(wrap_index, wrap_srs, &base).ok_or(Error::Lookups)?;
    let g = Projective::<Pallas>::generator();
    let mut j = 5u64;
    let mut next_point = || {
        j += 1;
        g * Fq::from_u64(j)
    };
    let mut proof = ProofIn {
        previous: vec![older_wrap_sg],
        witness: core::array::from_fn(|_| vec![next_point()]),
        z: vec![next_point()],
        t: (0..7).map(|_| next_point()).collect(),
        rounds: (0..rw).map(|_| (next_point(), next_point())).collect(),
        delta: next_point(),
        z1: Fq::from_u64(11),
        z2: Fq::from_u64(13),
        sg: g,
    };
    let wrap_own =
        scalar::Constants::standard::<Pallas>(params.wrap_domain(), params.wrap_domain(), ZK_ROWS)
            .map_err(|_| Error::Size)?;
    let wrap_evaluations = WrapEvaluations {
        evals: wrap_own.shape.map(&mut |()| fq_of(fresh())),
        ft_eval1: fq_of(fresh()),
    };
    let words: Vec<Fp> = statement.to_vec().iter().map(|w| to_fp(*w)).collect();
    let mut deferred = DeferredIn {
        public: words
            .iter()
            .zip(wrap::bounds(rs))
            .map(|(w, bound)| PublicWord { value: *w, bound })
            .collect(),
        perm: Fq::ONE,
        zeta_to_srs_length: Fq::ONE,
        zeta_to_domain_size: Fq::ONE,
        xi: Fp::ONE,
        combined_inner_product: Fq::ONE,
        b: Fq::ONE,
    };
    let unfinalized_of = |out: &crate::group::Outputs<Fp, Projective<Pallas>>| step::Unfinalized {
        combined_inner_product: [Fp::ZERO; 2],
        b: [Fp::ZERO; 2],
        zeta_to_srs_length: [Fp::ZERO; 2],
        zeta_to_domain_size: [Fp::ZERO; 2],
        perm: [Fp::ZERO; 2],
        sponge_digest_before_evaluations: out.drawn.digest,
        beta: out.drawn.beta,
        gamma: out.drawn.gamma,
        alpha: out.drawn.alpha,
        zeta: out.drawn.zeta,
        xi: Fp::ZERO,
        bulletproof_challenges: out.prechallenges.clone(),
        should_finalize: Fp::ZERO,
    };
    let run = |deferred: &DeferredIn<Fp, Fq>, proof: &ProofIn<Projective<Pallas>, Fq>| {
        let model = RefCell::new(Native);
        group_half_in::<Pallas, _>(&model, &pallas_base(), &constants, proof, deferred)
    };
    let first = run(&deferred, &proof);
    let first_fin = wrap_scalar_half(
        &wrap_own,
        &unfinalized_of(&first),
        &wrap_evaluations,
        &older_wrap,
    );
    deferred.perm = first_fin.linearization.perm;
    deferred.zeta_to_srs_length = first_fin.linearization.zeta_to_srs_length;
    deferred.zeta_to_domain_size = first_fin.linearization.zeta_to_domain_size;
    deferred.combined_inner_product = first_fin.combined_inner_product;
    deferred.xi = to_fp(first_fin.xi);
    let second = run(&deferred, &proof);
    let fin = wrap_scalar_half(
        &wrap_own,
        &unfinalized_of(&second),
        &wrap_evaluations,
        &older_wrap,
    );
    deferred.b = fin.b;
    let wrap_challenges = computed(&second.prechallenges);
    proof.sg = msm(&wrap_srs.g, &b_poly_coefficients(&wrap_challenges));

    let words_ok = statement
        .to_vec()
        .iter()
        .zip(wrap::bounds(rs))
        .all(|(v, n)| Word::of(*v).below(n));
    if !words_ok {
        return Err(Error::Bound);
    }
    let input = WrapInput {
        statement,
        messages,
        proof,
        deferred: WrapDeferred {
            perm: deferred.perm,
            zeta_to_srs_length: deferred.zeta_to_srs_length,
            zeta_to_domain_size: deferred.zeta_to_domain_size,
            xi: first_fin.xi,
            combined_inner_product: deferred.combined_inner_product,
            b: deferred.b,
        },
        evaluations: wrap_evaluations,
        previous_challenges: older_wrap,
    };
    Ok((input, previous, step_evaluations))
}

/// A prover index of `gadget`, its `public` inputs first, carrying one
/// previous challenge, on `srs`, whose size must be the domain's.
fn index_on<C, G>(gadget: &G, public: usize, srs: &Srs<C>) -> Result<ProverIndex<C>, Error>
where
    C: Glv,
    C::Base: arrabbiata_field::PrimeField + arrabbiata_field::Sqrt,
    C::Scalar: arrabbiata_field::TwoAdicField,
    G: TypedGadget<C::Scalar>,
{
    let rows = gadget.rows();
    let gates = compiled::circuit_rows::<C::Scalar, _, VARIABLES, DEGREE>(gadget, rows, 0)
        .map_err(|_| Error::Circuit)?;
    let cs = ConstraintSystem::create(gates)
        .public(public)
        .prev_challenges(1)
        .build()
        .map_err(|_| Error::Circuit)?;
    let n = cs.domain.d1.size();
    if n != srs.g.len() || cs.zk_rows != ZK_ROWS {
        return Err(Error::Domain { found: n });
    }
    Ok(ProverIndex::create(cs, srs.clone()))
}

/// The step circuit's index, on the step SRS.
///
/// # Errors
///
/// [`Error::Domain`] when the circuit needs another domain than the
/// parameters', [`Error::Circuit`] when it is not Kimchi rows.
pub fn step_index<const S: usize, const W: usize>(
    step: &RecStep<'_, S, W>,
    srs: &Srs<Vesta>,
) -> Result<ProverIndex<Vesta>, Error> {
    index_on(step, S, srs)
}

/// The wrap circuit's index, on the wrap SRS.
///
/// # Errors
///
/// As [`step_index`].
pub fn wrap_index<const S: usize, const W: usize>(
    wrap: &RecWrap<'_, S, W>,
    srs: &Srs<Pallas>,
) -> Result<ProverIndex<Pallas>, Error> {
    index_on(wrap, S, srs)
}

/// A step proof of `statement`, with `private` and the previous challenge
/// `previous`.
///
/// # Errors
///
/// [`Error::Size`] for inputs of another size, [`Error::Circuit`] or
/// [`Error::Prove`] when Kimchi refuses.
pub fn prove_step<const S: usize, const W: usize, R>(
    step: &RecStep<'_, S, W>,
    index: &ProverIndex<Vesta>,
    statement: &step::Statement<Fp>,
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

/// A wrap proof of `statement`, with `private` and the previous challenge
/// `previous`.
///
/// # Errors
///
/// As [`prove_step`].
pub fn prove_wrap<const S: usize, const W: usize, R>(
    wrap: &RecWrap<'_, S, W>,
    index: &ProverIndex<Pallas>,
    statement: &wrap::Statement<Fq>,
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

/// The domains a step circuit and a wrap circuit of `params` need: the
/// smallest power of two above each's rows and zero-knowledge rows, to
/// find the parameters' fixed point.
#[must_use]
pub fn domains_needed<const SS: usize, const SW: usize, const WS: usize, const WW: usize>(
    step: &RecStep<'_, SS, SW>,
    wrap: &RecWrap<'_, WS, WW>,
) -> (usize, usize) {
    let need = |rows: usize| (rows + ZK_ROWS + 1).next_power_of_two();
    (need(step.rows()), need(wrap.rows()))
}

/// The scalar evaluations' count of a proof's evaluations, for a shape
/// check.
#[must_use]
pub fn evaluations<F: Clone>(evals: &ProofEvaluations<F>) -> usize {
    let mut n = 0;
    let _ = evals.map(&mut |_| n += 1);
    n
}
