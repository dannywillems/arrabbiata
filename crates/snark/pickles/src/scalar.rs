//! The scalar half of Kimchi's verifier, written once over [`FieldModel`]:
//! what Pickles' finalizer computes (`Plonk_checks.ft_eval0`,
//! `derive_plonk`, the combined inner product, `b`).
//!
//! [`linearization`], [`evaluations`], [`combined_inner_product`] and
//! [`b_poly`] are generic in the model ([`FieldModel`]). In
//! [`arrabbiata_field::Native`]
//! they are the native verifier's arithmetic (checked equal to
//! `arrabbiata-kimchi`'s, an independent implementation); in
//! [`crate::model::Circuit`] they are a circuit, which [`ScalarHalf`] wraps
//! as a typed gadget, so the circuit and its witness are the same code
//! again.
//!
//! The scalar sponge is in the model too, on Kimchi's typed pattern
//! (`arrabbiata_kimchi::verifier::scalar_transcript_in`), so the circuit
//! draws `xi` and `r` itself from the base digest, and maps every
//! 128-bit challenge through the endomorphism, as Pickles' finalizer does.
//! The opening's challenges come mapped.
//!
//! The circuit does not yet cover lookups: [`ScalarHalf::new`] refuses an
//! index with lookups.

use alloc::{vec, vec::Vec};
use core::fmt;

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::{Field, FieldModel, PrimeField, TwoAdicField};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, measure};
use arrabbiata_kimchi::{
    constraints::GateType,
    gates::KimchiGates,
    oracle::ChallengeModel,
    permutation::PERMUTS,
    prover::{CONSTRAINED, LOOKUP_ALPHA, PERMUTATION_ALPHA, PointEvaluations, ProofEvaluations},
    srs::num_chunks,
    verifier::{Challenges, Linearization, scalar_transcript_in},
    verifier_index::VerifierIndex,
};
use arrabbiata_kimchi_gadgets::LAYOUT;
use arrabbiata_poly::Domain;
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, WIDTH};
use arrabbiata_transcript::PermutationIn;

use crate::model::{Affine, Circuit, load};

/// What the scalar half needs of an index: its domain and chunks, its
/// permutation shifts, and the shape of its evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constants<F> {
    /// The domain's size `n`.
    pub domain_size: usize,
    /// The chunk size `m`.
    pub max_poly_size: usize,
    /// The domain's generator.
    pub omega: F,
    /// `w^(n - zk_rows)`, the last row the permutation constrains.
    pub w_end: F,
    /// `w^(n - zk_rows + 1)`.
    pub w_after: F,
    /// `w^(n - 1)`.
    pub w_last: F,
    /// The shifts of the copy argument.
    pub shift: [F; PERMUTS],
    /// The evaluations' shape: a proof's evaluations of this index, every
    /// value replaced by `()`.
    pub shape: ProofEvaluations<()>,
    /// The endomorphism's eigenvalue `lambda` the 128-bit challenges map
    /// through.
    pub lambda: F,
}

/// Why the scalar half cannot be built for an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarError {
    /// The index looks something up, which the circuit does not cover yet.
    Lookups,
    /// The index's domain is not one of the field's.
    Domain,
    /// The input is not of the size the index gives.
    InputSize {
        /// The size of the gadget's input.
        declared: usize,
        /// The size the index gives.
        expected: usize,
    },
}

impl fmt::Display for ScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lookups => write!(f, "the scalar half's circuit does not cover lookups"),
            Self::Domain => write!(f, "the index's domain is not one of the field's"),
            Self::InputSize { declared, expected } => {
                write!(
                    f,
                    "an input of {declared} values where the index gives {expected}"
                )
            }
        }
    }
}

impl<F> Constants<F>
where
    F: TwoAdicField,
{
    /// The constants of `index`.
    ///
    /// # Errors
    ///
    /// [`ScalarError::Lookups`] for an index with lookups,
    /// [`ScalarError::Domain`] for a domain not of the field.
    pub fn of<C>(index: &VerifierIndex<C>) -> Result<Self, ScalarError>
    where
        C: Glv<Scalar = F>,
    {
        if index.lookup_index.is_some() {
            return Err(ScalarError::Lookups);
        }
        let d = Domain::<F>::containing(index.domain_size).ok_or(ScalarError::Domain)?;
        let n = d.size();
        let zk = index.zk_rows;
        let chunks = num_chunks(index.domain_size, index.max_poly_size);
        let point = || PointEvaluations {
            zeta: vec![(); chunks],
            zeta_omega: vec![(); chunks],
        };
        let optional = [
            (GateType::RangeCheck0, index.range_check0_comm.is_some()),
            (GateType::RangeCheck1, index.range_check1_comm.is_some()),
            (
                GateType::ForeignFieldAdd,
                index.foreign_field_add_comm.is_some(),
            ),
            (
                GateType::ForeignFieldMul,
                index.foreign_field_mul_comm.is_some(),
            ),
            (GateType::Xor16, index.xor_comm.is_some()),
            (GateType::Rot64, index.rot_comm.is_some()),
        ];
        let shape = ProofEvaluations {
            public: point(),
            w: core::array::from_fn(|_| point()),
            z: point(),
            s: core::array::from_fn(|_| point()),
            coefficients: core::array::from_fn(|_| point()),
            generic_selector: point(),
            poseidon_selector: point(),
            complete_add_selector: point(),
            mul_selector: point(),
            emul_selector: point(),
            endomul_scalar_selector: point(),
            optional_selectors: optional
                .into_iter()
                .filter(|(_, present)| *present)
                .map(|(t, _)| (t, point()))
                .collect(),
            lookup: None,
        };
        Ok(Self {
            domain_size: n,
            max_poly_size: index.max_poly_size,
            omega: d.generator(),
            w_end: d.element(n - zk),
            w_after: d.element(n - zk + 1),
            w_last: d.element(n - 1),
            shift: index.shift,
            shape,
            lambda: C::lambda(),
        })
    }
}

/// `sum_i chunks[i] y^i`, by Horner: the value at the point of a chunked
/// evaluation (`y` the point to the chunk size).
pub fn combine<F, M>(m: &mut M, chunks: &[M::Value], y: &M::Value) -> M::Value
where
    F: Field,
    M: FieldModel<F>,
{
    let mut acc = m.constant(F::ZERO);
    for c in chunks.iter().rev() {
        acc = m.mul(&acc, y);
        acc = m.add(&acc, c);
    }
    acc
}

/// `b(x) = prod_i (1 + chals[i] x^(2^(k-1-i)))`, Kimchi's `b_poly`.
pub fn b_poly<F, M>(m: &mut M, chals: &[M::Value], x: &M::Value) -> M::Value
where
    F: Field,
    M: FieldModel<F>,
{
    let k = chals.len();
    let mut powers = Vec::with_capacity(k);
    let mut power = x.clone();
    for _ in 0..k {
        powers.push(power.clone());
        power = m.square(&power);
    }
    let mut product = m.constant(F::ONE);
    for (i, c) in chals.iter().enumerate() {
        let term = m.mul(c, &powers[k - 1 - i]);
        let factor = m.shift(&term, F::ONE);
        product = m.mul(&product, &factor);
    }
    product
}

/// The linearization of `evals` at `challenges` against `k`, Kimchi's
/// gates' constraints from `gates` interpreted in the model (see
/// `arrabbiata_kimchi::verifier::linearization`, the same computation in
/// the field).
pub fn linearization<F, M>(
    m: &mut M,
    k: &Constants<F>,
    evals: &ProofEvaluations<M::Value>,
    challenges: &Challenges<M::Value>,
    gates: &KimchiGates<F>,
) -> Linearization<M::Value>
where
    F: PrimeField,
    M: FieldModel<F>,
{
    let Challenges {
        alpha,
        beta,
        gamma,
        zeta,
        joint_combiner: _,
    } = challenges;
    let e = evals;
    let zeta1 = m.pow(zeta, k.domain_size as u64);
    let zetaw = m.scale(zeta, k.omega);
    let zeta_m = m.pow(zeta, k.max_poly_size as u64);
    let zetaw_m = m.pow(&zetaw, k.max_poly_size as u64);
    let at = |m: &mut M, p: &PointEvaluations<M::Value>| {
        (
            combine(m, &p.zeta, &zeta_m),
            combine(m, &p.zeta_omega, &zetaw_m),
        )
    };
    let w: Vec<(M::Value, M::Value)> = e.w.iter().map(|p| at(m, p)).collect();
    let s: Vec<M::Value> = e.s.iter().map(|p| at(m, p).0).collect();
    let coefficients: Vec<M::Value> = e.coefficients.iter().map(|p| at(m, p).0).collect();
    let z = at(m, &e.z);
    let mut alphas = Vec::with_capacity(LOOKUP_ALPHA);
    alphas.push(m.constant(F::ONE));
    for i in 1..LOOKUP_ALPHA {
        let next = m.mul(&alphas[i - 1], alpha);
        alphas.push(next);
    }
    let alpha0 = alphas[PERMUTATION_ALPHA].clone();
    let alpha1 = alphas[PERMUTATION_ALPHA + 1].clone();
    let alpha2 = alphas[PERMUTATION_ALPHA + 2].clone();
    let zeta_minus = |m: &mut M, c: F| m.shift(zeta, -c);
    let a = zeta_minus(m, k.w_end);
    let b = zeta_minus(m, k.w_after);
    let c = zeta_minus(m, k.w_last);
    let ab = m.mul(&a, &b);
    let vanishing = m.mul(&ab, &c);

    // The factors gamma + beta s_i(zeta) + w_i(zeta), i < 6, shared by
    // ft(zeta) and perm.
    let mut factors = Vec::with_capacity(PERMUTS - 1);
    for i in 0..PERMUTS - 1 {
        let bs = m.mul(beta, &s[i]);
        let sum = m.add(&bs, &w[i].0);
        factors.push(m.add(&sum, gamma));
    }
    let z1_alpha0 = m.mul(&z.1, &alpha0);
    let head = m.mul(&z1_alpha0, &vanishing);

    // (w_6(zeta) + gamma) z(zeta w) alpha0 zk(zeta) prod_i factors_i.
    let w6 = m.add(&w[PERMUTS - 1].0, gamma);
    let mut ft = m.mul(&w6, &head);
    for f in &factors {
        ft = m.mul(&ft, f);
    }
    let public = combine(m, &e.public.zeta, &zeta_m);
    ft = m.sub(&ft, &public);
    // alpha0 zk(zeta) z(zeta) prod_i (gamma + beta zeta shift_i + w_i(zeta)).
    let beta_zeta = m.mul(beta, zeta);
    let alpha0_vanishing = m.mul(&alpha0, &vanishing);
    let mut shifted = m.mul(&alpha0_vanishing, &z.0);
    for (shift, wi) in k.shift.iter().zip(&w) {
        let term = m.scale(&beta_zeta, *shift);
        let sum = m.add(&term, &wi.0);
        let factor = m.add(&sum, gamma);
        shifted = m.mul(&shifted, &factor);
    }
    ft = m.sub(&ft, &shifted);
    // The boundary terms.
    let zeta1m1 = m.shift(&zeta1, -F::ONE);
    let zeta_minus_one = m.shift(zeta, -F::ONE);
    let first = m.mul(&zeta1m1, &alpha1);
    let first = m.mul(&first, &a);
    let second = m.mul(&zeta1m1, &alpha2);
    let second = m.mul(&second, &zeta_minus_one);
    let sum = m.add(&first, &second);
    let minus_z = m.neg(&z.0);
    let one_minus_z = m.shift(&minus_z, F::ONE);
    let numerator = m.mul(&sum, &one_minus_z);
    let denominator = m.mul(&a, &zeta_minus_one);
    let inverse = m.inverse(&denominator);
    let boundary = m.mul(&numerator, &inverse);
    ft = m.add(&ft, &boundary);
    // The gates: each selector times its constraints weighted by alpha^i.
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
        let (Some(sel), Some(constraints)) = (selector(typ), gates.constraints(typ)) else {
            continue;
        };
        let witness = |column: usize, rotation: usize| {
            if rotation == 0 {
                w[column].0.clone()
            } else {
                w[column].1.clone()
            }
        };
        let coefficient = |column: usize, _: usize| coefficients[column].clone();
        let values = constraints.interpret(m, &witness, &coefficient);
        let mut combined = m.constant(F::ZERO);
        for (i, v) in values.iter().enumerate() {
            let weighted = m.mul(&alphas[i], v);
            combined = m.add(&combined, &weighted);
        }
        let sel = at(m, sel).0;
        let term = m.mul(&sel, &combined);
        ft = m.sub(&ft, &term);
    }
    // perm = -z(zeta w) beta alpha0 zk(zeta) prod_i factors_i.
    let mut perm = m.mul(&head, beta);
    for f in &factors {
        perm = m.mul(&perm, f);
    }
    let perm = m.neg(&perm);
    Linearization {
        ft_eval0: ft,
        perm,
        zeta_to_domain_size: zeta1,
        zeta_to_srs_length: zeta_m,
    }
}

/// The evaluations of the opening, in its order (see
/// `arrabbiata_kimchi::verifier::evaluations`): the previous accumulators'
/// challenge polynomials at the two points, the public polynomial, `ft`,
/// then the proof's evaluations, without lookups.
pub fn evaluations<F, M>(
    m: &mut M,
    evals: &ProofEvaluations<M::Value>,
    prev_challenges: &[Vec<M::Value>],
    ft_evals: [M::Value; 2],
    points: &[M::Value; 2],
) -> Vec<[Vec<M::Value>; 2]>
where
    F: Field,
    M: FieldModel<F>,
{
    let pair = |p: &PointEvaluations<M::Value>| [p.zeta.clone(), p.zeta_omega.clone()];
    let e = evals;
    let mut out = Vec::new();
    for chals in prev_challenges {
        let at_zeta = b_poly(m, chals, &points[0]);
        let at_zetaw = b_poly(m, chals, &points[1]);
        out.push([vec![at_zeta], vec![at_zetaw]]);
    }
    out.push(pair(&e.public));
    let [ft0, ft1] = ft_evals;
    out.push([vec![ft0], vec![ft1]]);
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
    out
}

/// Kimchi's `combined_inner_product`: chunk `i` of each polynomial at the
/// next power of `xi`, its two points combined by `r`.
pub fn combined_inner_product<F, M>(
    m: &mut M,
    evaluations: &[[Vec<M::Value>; 2]],
    xi: &M::Value,
    r: &M::Value,
) -> M::Value
where
    F: Field,
    M: FieldModel<F>,
{
    let mut combined = m.constant(F::ZERO);
    let mut scale = m.constant(F::ONE);
    for [at_zeta, at_zetaw] in evaluations {
        for (a, b) in at_zeta.iter().zip(at_zetaw) {
            let rb = m.mul(r, b);
            let sum = m.add(a, &rb);
            let term = m.mul(&scale, &sum);
            combined = m.add(&combined, &term);
            scale = m.mul(&scale, xi);
        }
    }
    combined
}

/// What the scalar half reads, as values of a model: Pickles' deferred
/// challenges as they cross (the 128-bit `alpha` and `zeta` before their
/// map), the base sponge's digest, and the proof's evaluations.
#[derive(Clone, Debug)]
pub struct Inputs<V> {
    /// The 128-bit challenge `alpha` comes from.
    pub alpha: V,
    /// `beta`.
    pub beta: V,
    /// `gamma`.
    pub gamma: V,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta: V,
    /// The base sponge's digest, where the scalar sponge starts.
    pub digest: V,
    /// `ft(zeta w)`.
    pub ft_eval1: V,
    /// The previous accumulators' challenges.
    pub prev_challenges: Vec<Vec<V>>,
    /// The opening's challenges.
    pub bulletproof_challenges: Vec<V>,
    /// The proof's evaluations.
    pub evals: ProofEvaluations<V>,
}

/// What the scalar half computes: the values a finalizer compares with the
/// deferred ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outputs<V> {
    /// The linearization's scalars.
    pub linearization: Linearization<V>,
    /// The combined inner product.
    pub combined_inner_product: V,
    /// `b(zeta) + r b(zeta w)`.
    pub b: V,
    /// The 128-bit challenge `xi` the scalar sponge draws.
    pub xi: V,
}

/// The scalar half on `inputs` against `k`, in the model `m`, the scalar
/// sponge over `poseidon`: the transcript's `xi` and `r`, the challenges
/// mapped through the endomorphism, then the linearization, the combined
/// inner product and `b`.
pub fn scalar_half<F, M>(
    m: &mut M,
    k: &Constants<F>,
    poseidon: &KimchiPoseidon<F>,
    inputs: &Inputs<M::Value>,
    gates: &KimchiGates<F>,
) -> Outputs<M::Value>
where
    F: PrimeField,
    M: ChallengeModel<F>,
    KimchiPoseidon<F>: PermutationIn<M, F, WIDTH>,
{
    let (xi_chal, r_chal) = scalar_transcript_in(
        m,
        poseidon,
        &inputs.digest,
        &inputs.prev_challenges,
        &inputs.ft_eval1,
        &inputs.evals,
    );
    let challenges = Challenges {
        alpha: m.endo(&inputs.alpha, k.lambda),
        beta: inputs.beta.clone(),
        gamma: inputs.gamma.clone(),
        zeta: m.endo(&inputs.zeta, k.lambda),
        joint_combiner: None,
    };
    let xi = m.endo(&xi_chal, k.lambda);
    let r = m.endo(&r_chal, k.lambda);
    let lin = linearization(m, k, &inputs.evals, &challenges, gates);
    let zeta = &challenges.zeta;
    let points = [zeta.clone(), m.scale(zeta, k.omega)];
    let all = evaluations(
        m,
        &inputs.evals,
        &inputs.prev_challenges,
        [lin.ft_eval0.clone(), inputs.ft_eval1.clone()],
        &points,
    );
    let cip = combined_inner_product(m, &all, &xi, &r);
    let at_zeta = b_poly(m, &inputs.bulletproof_challenges, &points[0]);
    let at_zetaw = b_poly(m, &inputs.bulletproof_challenges, &points[1]);
    let r_at = m.mul(&r, &at_zetaw);
    let b = m.add(&at_zeta, &r_at);
    Outputs {
        linearization: lin,
        combined_inner_product: cip,
        b,
        xi: xi_chal,
    }
}

/// The values of `inputs`, flat, in the order [`ScalarHalf`] reads them:
/// `alpha, beta, gamma, zeta`, the digest, `ft(zeta w)`, the previous
/// accumulators' challenges, the opening's challenges, then the evaluations
/// in the order of their fields.
#[must_use]
pub fn flatten<V: Clone>(inputs: &Inputs<V>) -> Vec<V> {
    let mut out = vec![
        inputs.alpha.clone(),
        inputs.beta.clone(),
        inputs.gamma.clone(),
        inputs.zeta.clone(),
        inputs.digest.clone(),
        inputs.ft_eval1.clone(),
    ];
    for chals in &inputs.prev_challenges {
        out.extend(chals.iter().cloned());
    }
    out.extend(inputs.bulletproof_challenges.iter().cloned());
    let _ = inputs.evals.map(&mut |v| out.push(v.clone()));
    out
}

/// The scalar half as a circuit: a typed gadget whose input is
/// [`flatten`] of [`Inputs`], `N` values, and whose output is `ft(zeta)`,
/// `perm`, `zeta^n`, `zeta^m`, the combined inner product, `b` and the
/// 128-bit `xi`, in cells. Its rows depend on the index
/// ([`TypedGadget::rows`]).
#[derive(Clone, Debug)]
pub struct ScalarHalf<'a, F, const N: usize> {
    constants: Constants<F>,
    poseidon: &'a KimchiPoseidon<F>,
    gates: &'a KimchiGates<F>,
    prev: Vec<usize>,
    rounds: usize,
}

/// The values [`ScalarHalf`] outputs.
pub const OUTPUTS: usize = 7;

impl<'a, F, const N: usize> ScalarHalf<'a, F, N>
where
    F: TwoAdicField,
{
    /// The circuit for `constants`, with Kimchi's `poseidon` for the scalar
    /// sponge and its `gates`, previous accumulators of `prev[i]`
    /// challenges, and an opening of `rounds` rounds.
    ///
    /// # Errors
    ///
    /// [`ScalarError::InputSize`] when `N` is not the input's size.
    pub fn new(
        constants: Constants<F>,
        poseidon: &'a KimchiPoseidon<F>,
        gates: &'a KimchiGates<F>,
        prev: Vec<usize>,
        rounds: usize,
    ) -> Result<Self, ScalarError> {
        let mut evaluations = 0;
        let _ = constants.shape.map(&mut |()| evaluations += 1);
        let expected = 6 + prev.iter().sum::<usize>() + rounds + evaluations;
        if expected != N {
            return Err(ScalarError::InputSize {
                declared: N,
                expected,
            });
        }
        Ok(Self {
            constants,
            poseidon,
            gates,
            prev,
            rounds,
        })
    }

    /// [`Inputs`] of the flat values `flat`, the shape this circuit's.
    fn unflatten<V: Clone>(&self, flat: &[V]) -> Inputs<V> {
        let mut next = flat.iter().cloned();
        let mut take = || match next.next() {
            Some(v) => v,
            None => unreachable!("the input has the size new checked"),
        };
        let alpha = take();
        let beta = take();
        let gamma = take();
        let zeta = take();
        let digest = take();
        let ft_eval1 = take();
        let prev_challenges = self
            .prev
            .iter()
            .map(|n| (0..*n).map(|_| take()).collect())
            .collect();
        let bulletproof_challenges = (0..self.rounds).map(|_| take()).collect();
        let evals = self.constants.shape.map(&mut |()| take());
        Inputs {
            alpha,
            beta,
            gamma,
            zeta,
            digest,
            ft_eval1,
            prev_challenges,
            bulletproof_challenges,
            evals,
        }
    }
}

impl<F, const N: usize> TypedGadget<F> for ScalarHalf<'_, F, N>
where
    F: PrimeField + TwoAdicField,
{
    type Input = Array<Scalar, N>;
    type Output = Array<Scalar, OUTPUTS>;
    const NAME: &'static str = "pickles-scalar-half";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let cells = load(env, &input);
        let values: Vec<Affine<E::Var, F>> = cells.into_iter().map(Affine::of).collect();
        let inputs = self.unflatten(&values);
        let mut model = Circuit::new(env);
        let out = scalar_half(
            &mut model,
            &self.constants,
            self.poseidon,
            &inputs,
            self.gates,
        );
        let lin = &out.linearization;
        let outputs = [
            &lin.ft_eval0,
            &lin.perm,
            &lin.zeta_to_domain_size,
            &lin.zeta_to_srs_length,
            &out.combined_inner_product,
            &out.b,
            &out.xi,
        ];
        let placed: Vec<E::Var> = outputs.iter().map(|v| model.cell(v)).collect();
        core::array::from_fn(|i| placed[i].clone())
    }
}
