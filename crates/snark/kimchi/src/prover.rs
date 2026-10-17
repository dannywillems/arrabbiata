//! Kimchi's prover (`kimchi/src/prover.rs`, `ProverProof::create_recursive`),
//! as a sequence of rounds, each a function from the state the previous
//! one left to a larger one, so that each can be checked against Kimchi on
//! its own.
//!
//! [`witness_round`], the first round:
//!
//! 1. pad each witness column with zeros to the domain, then overwrite its
//!    last `zk_rows` rows with random values (column 0 first, each column
//!    from its last row up);
//! 2. a base-field sponge absorbs the verifier-index digest (Kimchi then
//!    absorbs the previous accumulators' commitments; there are none yet);
//! 3. the public polynomial, the negated public inputs on the first rows,
//!    committed and masked by `h` with blinder 1, is absorbed;
//! 4. each column is committed, hidden by `h` with one random blinder per
//!    chunk (all fifteen columns' blinders drawn before any commitment),
//!    and the fifteen commitments are absorbed;
//! 5. `beta` and `gamma` are squeezed, 128-bit challenges.
//!
//! [`permutation_round`], the second:
//!
//! 1. the aggregation `z` on `d1`: `z_0 = 1` and
//!    `z_(j+1) = z_j prod_i (w_i + beta shift_i w^j + gamma) / prod_i (w_i +
//!    beta sigma_i + gamma)` at row `j`, over the 7 permuted columns, except that `z` is random at
//!    rows `n - zk_rows + 1` and `n - zk_rows + 2` (drawn in that order);
//!    the product closes, `z(w^(n - zk_rows)) = 1`, exactly when the
//!    witness respects the copies;
//! 2. `z` committed and hidden, one blinder per chunk, and absorbed;
//! 3. `alpha`, the endomorphism image of a 128-bit challenge.
//!
//! [`quotient_round`], the third: the quotient `t = f / Z_H + bnd`, with
//!
//! - `f` the sum of each gate's constraints weighted by `alpha^i`
//!   (constraint `i`; gates share the powers, their selectors being
//!   disjoint) and times its selector, of the permutation term
//!   `alpha^21 (z(X) prod_i (w_i + gamma + beta k_i X) - z(w X) prod_i (w_i +
//!   gamma + beta sigma_i)) M(X)`, `M` vanishing on rows `n - zk_rows`,
//!   `n - zk_rows + 1` and `n - 1`, and of the public polynomial;
//! - `bnd = alpha^22 (z - 1) / (X - 1) + alpha^23 (z - 1) / (X - w^(n -
//!   zk_rows))`, the two boundary conditions, already divided;
//!
//! then `t` committed in `7 num_chunks` chunks, hidden, absorbed, and
//! `zeta` the endomorphism image of a 128-bit challenge. `f` is evaluated
//! on `d8` and interpolated, exact since its degree is below `8 n` for
//! Kimchi's gates (Kimchi evaluates on `d4` the terms of degree below
//! `4 n`, and skips the points of `d1` in the permutation products, where
//! the term vanishes anyway; the polynomial is the same).
//!
//! [`evaluation_round`], the fourth:
//!
//! 1. every polynomial evaluated at `zeta` and `zeta w`, chunk by chunk
//!    (`num_chunks` chunks of `max_poly_size` coefficients each): the
//!    public polynomial, the witness, `z`, the first six sigmas, the
//!    coefficients and the selectors;
//! 2. Maller's `ft = lin(f) - (zeta^n - 1) lin(t)`, `lin` the sum of the
//!    chunks weighted by powers of `zeta^max_poly_size`, `f` the
//!    linearization of the constraints at `zeta`: as Kimchi evaluates
//!    every column but the last sigma, only the permutation's
//!    `scalar sigma_6(X)` remains, `scalar = -z(zeta w) beta alpha^21
//!    M(zeta) prod_(i < 6) (gamma + beta sigma_i(zeta) + w_i(zeta))`;
//! 3. the scalar sponge absorbs the base sponge's digest, the digest of
//!    the previous challenges (none, an empty sponge's), `ft(zeta w)`, the
//!    public evaluations, then `z`, the six selectors, the witness, the
//!    coefficients, the sigmas and the optional selectors, each at `zeta`
//!    then `zeta w`; `v` and `u` are squeezed and mapped by the
//!    endomorphism.
//!
//! The randomness is the [`Sampler`] passed in; with [`crate::rng::Arkworks`]
//! over the same stream as Kimchi's RNG, the round gives Kimchi's values.
//! Lookups are not supported yet: a circuit with Lookup, RangeCheck, Rot64,
//! Xor16 or ForeignFieldMul rows, or with tables, is refused.

use alloc::vec::Vec;
use core::fmt;

use arrabbiata_curve::{Curve, glv::Glv};
use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_poly::UniPoly;
use arrabbiata_transcript::Transcript;

use crate::{
    constraints::{COLUMNS, ConstraintSystem, GateType, zk_rows_strict_lower_bound},
    gates::Gates,
    oracle::{FqSponge, FrSponge, ScalarChallenge},
    permutation::PERMUTS,
    rng::Sampler,
    srs::{PolyComm, Srs},
    verifier_index::VerifierIndex,
};

/// Why a proof cannot be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The witness columns do not all have the same length.
    WitnessCsInconsistent,
    /// The witness leaves fewer than `zk_rows` rows of the domain free.
    NoRoomForZkInWitness,
    /// The constraint system has too few zero-knowledge rows for its chunks.
    NotZeroKnowledge {
        /// The rows needed.
        needed: usize,
        /// The rows there are.
        found: usize,
    },
    /// The circuit uses lookups, not supported yet.
    LookupsUnsupported,
    /// The permutation product does not close: the witness breaks a copy.
    PermutationFinalValue,
    /// `f` is not divisible by the vanishing polynomial: the witness breaks
    /// a constraint.
    QuotientRemainder,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WitnessCsInconsistent => write!(f, "witness columns of different lengths"),
            Self::NoRoomForZkInWitness => {
                write!(f, "the witness leaves no room for the zero-knowledge rows")
            }
            Self::NotZeroKnowledge { needed, found } => {
                write!(f, "{found} zero-knowledge rows, {needed} needed")
            }
            Self::LookupsUnsupported => write!(f, "lookups are not supported yet"),
            Self::PermutationFinalValue => {
                write!(
                    f,
                    "the permutation product does not close: a copy is broken"
                )
            }
            Self::QuotientRemainder => write!(f, "the constraints do not vanish on the domain"),
        }
    }
}

/// What the prover needs: the constraint system, the SRS, and the
/// verifier index with its digest.
#[derive(Clone, Debug)]
pub struct ProverIndex<C>
where
    C: Curve,
{
    /// The constraint system.
    pub cs: ConstraintSystem<C::Scalar>,
    /// The SRS.
    pub srs: Srs<C>,
    /// The verifier index.
    pub verifier_index: VerifierIndex<C>,
}

impl<C> ProverIndex<C>
where
    C: Curve,
    C::Scalar: TwoAdicField,
{
    /// The index of `cs` on `srs`.
    #[must_use]
    pub fn create(cs: ConstraintSystem<C::Scalar>, srs: Srs<C>) -> Self {
        let verifier_index = VerifierIndex::create(&cs, &srs);
        Self {
            cs,
            srs,
            verifier_index,
        }
    }

    /// The chunks of a polynomial of the domain: `n / max_poly_size`, at
    /// least one.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        let n = self.cs.domain.d1.size();
        let m = self.srs.max_poly_size();
        if n < m { 1 } else { n / m }
    }
}

/// The state after the witness round.
#[derive(Clone, Debug)]
pub struct WitnessRound<C, S>
where
    C: Curve,
{
    /// The witness columns on `d1`, padded, the zero-knowledge rows random.
    pub witness: [Vec<C::Scalar>; COLUMNS],
    /// The public inputs.
    pub public: Vec<C::Scalar>,
    /// The commitment to the public polynomial (masked).
    pub public_comm: PolyComm<C>,
    /// The blinders of the witness commitments, one per chunk.
    pub blinders: [Vec<C::Scalar>; COLUMNS],
    /// The witness commitments.
    pub w_comm: [PolyComm<C>; COLUMNS],
    /// The base-field sponge after `gamma`.
    pub fq_sponge: FqSponge<C, S>,
    /// The first permutation challenge.
    pub beta: C::Scalar,
    /// The second permutation challenge.
    pub gamma: C::Scalar,
}

/// The witness round (see the module documentation). `sponge` is a fresh
/// base-field sponge, cloned for the digest and for the proof's transcript.
///
/// # Errors
///
/// [`Error::WitnessCsInconsistent`] for columns of different lengths,
/// [`Error::NoRoomForZkInWitness`], [`Error::NotZeroKnowledge`], and
/// [`Error::LookupsUnsupported`].
pub fn witness_round<C, S, R>(
    index: &ProverIndex<C>,
    witness: [Vec<C::Scalar>; COLUMNS],
    sponge: &S,
    rng: &mut R,
) -> Result<WitnessRound<C, S>, Error>
where
    C: Curve,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    R: Sampler<C::Scalar>,
{
    let cs = &index.cs;
    if !cs.lookup_tables.is_empty()
        || cs.runtime_tables.is_some()
        || cs
            .gates
            .iter()
            .any(|g| g.typ.table().is_some() || g.typ == crate::constraints::GateType::Lookup)
    {
        return Err(Error::LookupsUnsupported);
    }
    let n = cs.domain.d1.size();
    let num_chunks = index.num_chunks();
    let limit = zk_rows_strict_lower_bound(num_chunks);
    if cs.zk_rows <= limit {
        return Err(Error::NotZeroKnowledge {
            needed: limit + 1,
            found: cs.zk_rows,
        });
    }
    let length = witness[0].len();
    if witness.iter().any(|w| w.len() != length) {
        return Err(Error::WitnessCsInconsistent);
    }
    let padding = n.checked_sub(length).ok_or(Error::NoRoomForZkInWitness)?;
    if padding < cs.zk_rows {
        return Err(Error::NoRoomForZkInWitness);
    }
    let mut witness = witness;
    for column in &mut witness {
        column.resize(n, C::Scalar::ZERO);
        for value in column.iter_mut().rev().take(cs.zk_rows) {
            *value = rng.sample();
        }
    }

    let mut fq_sponge = FqSponge::<C, S>::new(sponge.clone());
    let digest = index.verifier_index.digest(sponge.clone());
    fq_sponge.absorb_fq(&[digest]);

    let public: Vec<C::Scalar> = witness[0][..cs.public].to_vec();
    let mut negated = alloc::vec![C::Scalar::ZERO; n];
    for (slot, p) in negated.iter_mut().zip(&public) {
        *slot = -*p;
    }
    let public_comm = index.srs.mask_fixed(commit_column(index, &negated));
    fq_sponge.absorb_g(&public_comm.chunks);

    let blinders: [Vec<C::Scalar>; COLUMNS] =
        core::array::from_fn(|_| (0..num_chunks).map(|_| rng.sample()).collect());
    let w_comm: [PolyComm<C>; COLUMNS] =
        core::array::from_fn(|i| hide(index, commit_column(index, &witness[i]), &blinders[i]));
    for c in &w_comm {
        fq_sponge.absorb_g(&c.chunks);
    }
    let beta = fq_sponge.challenge();
    let gamma = fq_sponge.challenge();
    Ok(WitnessRound {
        witness,
        public,
        public_comm,
        blinders,
        w_comm,
        fq_sponge,
        beta,
        gamma,
    })
}

/// The commitment to the evaluations `column` on `d1`, chunked.
fn commit_column<C>(index: &ProverIndex<C>, column: &[C::Scalar]) -> PolyComm<C>
where
    C: Curve,
    C::Scalar: TwoAdicField,
{
    let d1 = index.cs.domain.d1;
    let polynomial = match d1.interpolate(column) {
        Some(p) => p,
        None => unreachable!("every column has one value per row"),
    };
    index
        .srs
        .commit_non_hiding(polynomial.coefficients(), d1.size())
}

/// `commitment` hidden by `h`, one blinder per chunk.
fn hide<C>(index: &ProverIndex<C>, commitment: PolyComm<C>, blinders: &[C::Scalar]) -> PolyComm<C>
where
    C: Curve,
{
    PolyComm {
        chunks: commitment
            .chunks
            .into_iter()
            .zip(blinders)
            .map(|(chunk, blinder)| chunk + index.srs.h * *blinder)
            .collect(),
    }
}

/// The state after the permutation round.
#[derive(Clone, Debug)]
pub struct PermutationRound<C, S>
where
    C: Curve,
{
    /// The state before it.
    pub previous: WitnessRound<C, S>,
    /// The aggregation `z` on `d1`.
    pub z: Vec<C::Scalar>,
    /// The blinders of its commitment, one per chunk.
    pub z_blinders: Vec<C::Scalar>,
    /// The commitment to `z`.
    pub z_comm: PolyComm<C>,
    /// The base-field sponge after `alpha`.
    pub fq_sponge: FqSponge<C, S>,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha_chal: ScalarChallenge<C::Scalar>,
    /// The constraint combiner.
    pub alpha: C::Scalar,
}

/// The permutation round (see the module documentation).
///
/// # Errors
///
/// [`Error::PermutationFinalValue`] when the product does not close.
pub fn permutation_round<C, S, R>(
    index: &ProverIndex<C>,
    previous: WitnessRound<C, S>,
    rng: &mut R,
) -> Result<PermutationRound<C, S>, Error>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    R: Sampler<C::Scalar>,
{
    let cs = &index.cs;
    let d1 = cs.domain.d1;
    let n = d1.size();
    let zk_rows = cs.zk_rows;
    let (beta, gamma) = (previous.beta, previous.gamma);
    let w = &previous.witness;
    let sigmas = cs.sigmas();
    let mut z = alloc::vec![C::Scalar::ONE; n];
    let mut row_label = C::Scalar::ONE;
    for j in 0..n - 1 {
        let random = j == n - zk_rows || j == n - zk_rows + 1;
        if random {
            z[j + 1] = rng.sample();
        } else {
            let mut numerator = C::Scalar::ONE;
            let mut denominator = C::Scalar::ONE;
            for i in 0..PERMUTS {
                numerator = numerator * (w[i][j] + beta * cs.shift[i] * row_label + gamma);
                denominator = denominator * (w[i][j] + beta * sigmas[i][j] + gamma);
            }
            // Kimchi inverts by batch inversion, which leaves a zero as zero.
            let inverse = denominator.inverse().unwrap_or(C::Scalar::ZERO);
            z[j + 1] = z[j] * numerator * inverse;
        }
        row_label = row_label * d1.generator();
    }
    if z[n - zk_rows] != C::Scalar::ONE {
        return Err(Error::PermutationFinalValue);
    }
    let num_chunks = index.num_chunks();
    let z_blinders: Vec<C::Scalar> = (0..num_chunks).map(|_| rng.sample()).collect();
    let z_comm = hide(index, commit_column(index, &z), &z_blinders);
    let mut fq_sponge = previous.fq_sponge.clone();
    fq_sponge.absorb_g(&z_comm.chunks);
    let alpha_chal = ScalarChallenge(fq_sponge.challenge());
    let alpha = alpha_chal.to_field(C::lambda());
    Ok(PermutationRound {
        previous,
        z,
        z_blinders,
        z_comm,
        fq_sponge,
        alpha_chal,
        alpha,
    })
}

/// The first power of `alpha` of the permutation: the gates use the first
/// 21, VarBaseMul's number of constraints.
pub const PERMUTATION_ALPHA: usize = 21;

/// The gate types with constraints, in Kimchi's order.
const CONSTRAINED: [GateType; 12] = [
    GateType::Generic,
    GateType::Poseidon,
    GateType::CompleteAdd,
    GateType::VarBaseMul,
    GateType::EndoMul,
    GateType::EndoMulScalar,
    GateType::RangeCheck0,
    GateType::RangeCheck1,
    GateType::ForeignFieldAdd,
    GateType::ForeignFieldMul,
    GateType::Xor16,
    GateType::Rot64,
];

/// `p / (X - a)`, or `None` when `a` is not a root of `p`.
fn divide_by_linear<F: Field>(p: &UniPoly<F>, a: F) -> Option<UniPoly<F>> {
    let coefficients = p.coefficients();
    if coefficients.is_empty() {
        return Some(UniPoly::new(Vec::new()));
    }
    let mut quotient = alloc::vec![F::ZERO; coefficients.len() - 1];
    let mut carry = F::ZERO;
    for k in (1..coefficients.len()).rev() {
        carry = coefficients[k] + carry * a;
        quotient[k - 1] = carry;
    }
    let remainder = coefficients[0] + carry * a;
    remainder.is_zero().then(|| UniPoly::new(quotient))
}

/// The state after the quotient round.
#[derive(Clone, Debug)]
pub struct QuotientRound<C, S>
where
    C: Curve,
{
    /// The state before it.
    pub previous: PermutationRound<C, S>,
    /// The quotient's coefficients, at most `7 n`.
    pub t: UniPoly<C::Scalar>,
    /// The blinders of its commitment, one per chunk.
    pub t_blinders: Vec<C::Scalar>,
    /// The commitment to `t`, in `7 num_chunks` chunks.
    pub t_comm: PolyComm<C>,
    /// The base-field sponge after `zeta`.
    pub fq_sponge: FqSponge<C, S>,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta_chal: ScalarChallenge<C::Scalar>,
    /// The evaluation point.
    pub zeta: C::Scalar,
}

/// The quotient round (see the module documentation), the gates' constraints
/// given by `gates`.
///
/// # Errors
///
/// [`Error::QuotientRemainder`] when the witness breaks a constraint.
///
/// # Panics
///
/// If the field has no domain of `8 n` points (the constraint system
/// builds `d8`, so it has one).
pub fn quotient_round<C, S, R, G>(
    index: &ProverIndex<C>,
    previous: PermutationRound<C, S>,
    gates: &G,
    rng: &mut R,
) -> Result<QuotientRound<C, S>, Error>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    R: Sampler<C::Scalar>,
    G: Gates<C::Scalar>,
{
    let cs = &index.cs;
    let (d1, d8) = (cs.domain.d1, cs.domain.d8);
    let (n, big) = (d1.size(), d8.size());
    let zk_rows = cs.zk_rows;
    let first = &previous.previous;
    let (alpha, beta, gamma) = (previous.alpha, first.beta, first.gamma);
    let extend = |column: &[C::Scalar]| match cs.extend(column, &d8) {
        Some(values) => values,
        None => unreachable!("every column has one value per row of d1"),
    };
    let w8: Vec<Vec<C::Scalar>> = first.witness.iter().map(|c| extend(c)).collect();
    let z8 = extend(&previous.z);
    let coefficients8: Vec<Vec<C::Scalar>> =
        (0..COLUMNS).map(|i| extend(&cs.coefficient(i))).collect();
    let sigmas = cs.sigmas();
    let sigmas8: Vec<Vec<C::Scalar>> = sigmas.iter().map(|c| extend(c)).collect();
    let used: Vec<(GateType, Vec<C::Scalar>)> = CONSTRAINED
        .iter()
        .filter(|typ| cs.uses(**typ))
        .map(|typ| (*typ, extend(&cs.selector(*typ))))
        .collect();
    let mut alphas = alloc::vec![C::Scalar::ONE; PERMUTATION_ALPHA + 3];
    for k in 1..alphas.len() {
        alphas[k] = alphas[k - 1] * alpha;
    }
    // M's roots: rows n - zk_rows, n - zk_rows + 1 and n - 1.
    let roots = [
        d1.element(n - zk_rows),
        d1.element(n - zk_rows + 1),
        d1.element(n - 1),
    ];

    let mut f8 = alloc::vec![C::Scalar::ZERO; big];
    let mut x = C::Scalar::ONE;
    for k in 0..big {
        let witness = |c: usize, r: usize| w8[c][(k + 8 * r) % big];
        let coefficient = |c: usize, r: usize| coefficients8[c][(k + 8 * r) % big];
        let mut value = C::Scalar::ZERO;
        for (typ, selector) in &used {
            let constraints = gates.evaluate(*typ, &witness, &coefficient);
            let mut combined = C::Scalar::ZERO;
            for (i, c) in constraints.iter().enumerate() {
                combined = combined + alphas[i] * *c;
            }
            value = value + selector[k] * combined;
        }
        let mut shifted = z8[k];
        let mut permuted = z8[(k + 8) % big];
        for i in 0..PERMUTS {
            shifted = shifted * (w8[i][k] + gamma + beta * cs.shift[i] * x);
            permuted = permuted * (w8[i][k] + gamma + beta * sigmas8[i][k]);
        }
        let vanishing = (x - roots[0]) * (x - roots[1]) * (x - roots[2]);
        value = value + alphas[PERMUTATION_ALPHA] * (shifted - permuted) * vanishing;
        f8[k] = value;
        x = x * d8.generator();
    }
    let f = match d8.interpolate(&f8) {
        Some(p) => p,
        None => unreachable!("one value per point of d8"),
    };
    let mut negated = alloc::vec![C::Scalar::ZERO; n];
    for (slot, p) in negated.iter_mut().zip(&first.public) {
        *slot = -*p;
    }
    let public = match d1.interpolate(&negated) {
        Some(p) => p,
        None => unreachable!("one value per row"),
    };
    let f = &f + &public;
    let quotient = f.divide_by_vanishing(n).ok_or(Error::QuotientRemainder)?;

    let z = match d1.interpolate(&previous.z) {
        Some(p) => p,
        None => unreachable!("one value per row"),
    };
    let z_minus_1 = &z - &UniPoly::constant(C::Scalar::ONE);
    let (Some(bnd1), Some(bnd2)) = (
        divide_by_linear(&z_minus_1, C::Scalar::ONE),
        divide_by_linear(&z_minus_1, d1.element(n - zk_rows)),
    ) else {
        return Err(Error::PermutationFinalValue);
    };
    let bnd =
        &bnd1.scale(alphas[PERMUTATION_ALPHA + 1]) + &bnd2.scale(alphas[PERMUTATION_ALPHA + 2]);
    let t = &quotient + &bnd;
    let mut coefficients = t.coefficients().to_vec();
    while coefficients.len() > 7 * n && coefficients.last().is_some_and(|c| c.is_zero()) {
        coefficients.pop();
    }
    if coefficients.len() > 7 * n {
        return Err(Error::QuotientRemainder);
    }
    let t = UniPoly::new(coefficients);

    let chunks = 7 * index.num_chunks();
    let t_blinders: Vec<C::Scalar> = (0..chunks).map(|_| rng.sample()).collect();
    let t_comm = hide(
        index,
        index.srs.commit_non_hiding(t.coefficients(), 7 * n),
        &t_blinders,
    );
    let mut fq_sponge = previous.fq_sponge.clone();
    fq_sponge.absorb_g(&t_comm.chunks);
    let zeta_chal = ScalarChallenge(fq_sponge.challenge());
    let zeta = zeta_chal.to_field(C::lambda());
    Ok(QuotientRound {
        previous,
        t,
        t_blinders,
        t_comm,
        fq_sponge,
        zeta_chal,
        zeta,
    })
}

/// The evaluations of a polynomial at `zeta` and `zeta w`, one per chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointEvaluations<F> {
    /// At `zeta`.
    pub zeta: Vec<F>,
    /// At `zeta w`.
    pub zeta_omega: Vec<F>,
}

/// The evaluations a proof carries (Kimchi's `ProofEvaluations`, without
/// lookups).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofEvaluations<F> {
    /// The public polynomial.
    pub public: PointEvaluations<F>,
    /// The witness columns.
    pub w: [PointEvaluations<F>; COLUMNS],
    /// The permutation aggregation.
    pub z: PointEvaluations<F>,
    /// The first six permutation columns.
    pub s: [PointEvaluations<F>; PERMUTS - 1],
    /// The coefficient columns.
    pub coefficients: [PointEvaluations<F>; COLUMNS],
    /// The Generic selector.
    pub generic_selector: PointEvaluations<F>,
    /// The Poseidon selector.
    pub poseidon_selector: PointEvaluations<F>,
    /// The CompleteAdd selector.
    pub complete_add_selector: PointEvaluations<F>,
    /// The VarBaseMul selector.
    pub mul_selector: PointEvaluations<F>,
    /// The EndoMul selector.
    pub emul_selector: PointEvaluations<F>,
    /// The EndoMulScalar selector.
    pub endomul_scalar_selector: PointEvaluations<F>,
    /// The optional selectors present, in Kimchi's order (RangeCheck0,
    /// RangeCheck1, ForeignFieldAdd, ForeignFieldMul, Xor16, Rot64).
    pub optional_selectors: Vec<(GateType, PointEvaluations<F>)>,
}

impl<F> ProofEvaluations<F>
where
    F: Copy,
{
    /// The evaluations in the order the scalar sponge absorbs them, after
    /// the public ones.
    #[must_use]
    pub fn absorbed(&self) -> Vec<&PointEvaluations<F>> {
        let mut points = alloc::vec![
            &self.z,
            &self.generic_selector,
            &self.poseidon_selector,
            &self.complete_add_selector,
            &self.mul_selector,
            &self.emul_selector,
            &self.endomul_scalar_selector,
        ];
        points.extend(self.w.iter());
        points.extend(self.coefficients.iter());
        points.extend(self.s.iter());
        points.extend(self.optional_selectors.iter().map(|(_, e)| e));
        points
    }
}

/// The optional selectors, in Kimchi's evaluation order.
const OPTIONAL: [GateType; 6] = [
    GateType::RangeCheck0,
    GateType::RangeCheck1,
    GateType::ForeignFieldAdd,
    GateType::ForeignFieldMul,
    GateType::Xor16,
    GateType::Rot64,
];

/// The chunks of `coefficients`, `num_chunks` of `size` (more if the
/// polynomial is longer), evaluated at `x`.
fn chunk_evaluations<F: Field>(coefficients: &[F], num_chunks: usize, size: usize, x: F) -> Vec<F> {
    let mut values: Vec<F> = coefficients
        .chunks(size)
        .map(|chunk| UniPoly::new(chunk.to_vec()).evaluate(x))
        .collect();
    values.resize(values.len().max(num_chunks), F::ZERO);
    values
}

/// `sum_i c_i y^i`: chunk evaluations combined into the value at the point
/// (`y` the point to the chunk size).
fn combine<F: Field>(chunks: &[F], y: F) -> F {
    chunks.iter().rev().fold(F::ZERO, |acc, c| acc * y + *c)
}

/// The chunks of `coefficients` summed with powers of `y`: a polynomial of
/// degree below `size` (Kimchi's `linearize`).
fn linearize<F: Field>(coefficients: &[F], size: usize, y: F) -> UniPoly<F> {
    let mut sum = alloc::vec![F::ZERO; size];
    let mut scale = F::ONE;
    for chunk in coefficients.chunks(size) {
        for (s, c) in sum.iter_mut().zip(chunk) {
            *s = *s + scale * *c;
        }
        scale = scale * y;
    }
    UniPoly::new(sum)
}

/// The state after the evaluation round.
#[derive(Clone, Debug)]
pub struct EvaluationRound<C, S, T>
where
    C: Curve,
{
    /// The state before it.
    pub previous: QuotientRound<C, S>,
    /// `zeta w`.
    pub zeta_omega: C::Scalar,
    /// The chunked evaluations.
    pub evals: ProofEvaluations<C::Scalar>,
    /// Maller's polynomial.
    pub ft: UniPoly<C::Scalar>,
    /// The blinder of `ft`'s commitment: `-(zeta^n - 1)` times `t`'s
    /// blinders combined by powers of `zeta^max_poly_size`.
    pub blinding_ft: C::Scalar,
    /// `ft(zeta w)`.
    pub ft_eval1: C::Scalar,
    /// The scalar sponge after `u`.
    pub fr_sponge: FrSponge<C::Scalar, T>,
    /// The 128-bit challenge `v` comes from.
    pub v_chal: ScalarChallenge<C::Scalar>,
    /// The polynomial combiner of the opening.
    pub v: C::Scalar,
    /// The 128-bit challenge `u` comes from.
    pub u_chal: ScalarChallenge<C::Scalar>,
    /// The evaluation-point combiner of the opening.
    pub u: C::Scalar,
}

/// The evaluation round (see the module documentation). `scalar_sponge` is
/// a fresh scalar-field sponge.
#[must_use]
pub fn evaluation_round<C, S, T>(
    index: &ProverIndex<C>,
    previous: QuotientRound<C, S>,
    scalar_sponge: &T,
) -> EvaluationRound<C, S, T>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
{
    let cs = &index.cs;
    let d1 = cs.domain.d1;
    let n = d1.size();
    let m = index.srs.max_poly_size();
    let num_chunks = index.num_chunks();
    let zeta = previous.zeta;
    let zeta_omega = zeta * d1.generator();
    let permutation = &previous.previous;
    let first = &permutation.previous;
    let (alpha, beta, gamma) = (permutation.alpha, first.beta, first.gamma);
    let interpolate = |column: &[C::Scalar]| match d1.interpolate(column) {
        Some(p) => p,
        None => unreachable!("one value per row"),
    };
    let at = |p: &UniPoly<C::Scalar>| PointEvaluations {
        zeta: chunk_evaluations(p.coefficients(), num_chunks, m, zeta),
        zeta_omega: chunk_evaluations(p.coefficients(), num_chunks, m, zeta_omega),
    };
    let column = |values: &[C::Scalar]| at(&interpolate(values));

    let mut negated = alloc::vec![C::Scalar::ZERO; n];
    for (slot, p) in negated.iter_mut().zip(&first.public) {
        *slot = -*p;
    }
    let sigmas = cs.sigmas();
    let evals = ProofEvaluations {
        public: column(&negated),
        w: core::array::from_fn(|i| column(&first.witness[i])),
        z: column(&permutation.z),
        s: core::array::from_fn(|i| column(&sigmas[i])),
        coefficients: core::array::from_fn(|i| column(&cs.coefficient(i))),
        generic_selector: column(&cs.selector(GateType::Generic)),
        poseidon_selector: column(&cs.selector(GateType::Poseidon)),
        complete_add_selector: column(&cs.selector(GateType::CompleteAdd)),
        mul_selector: column(&cs.selector(GateType::VarBaseMul)),
        emul_selector: column(&cs.selector(GateType::EndoMul)),
        endomul_scalar_selector: column(&cs.selector(GateType::EndoMulScalar)),
        optional_selectors: OPTIONAL
            .iter()
            .filter(|typ| cs.uses(**typ))
            .map(|typ| (*typ, column(&cs.selector(*typ))))
            .collect(),
    };

    let zeta_m = zeta.pow(m as u64);
    let zeta_omega_m = zeta_omega.pow(m as u64);
    let value = |e: &PointEvaluations<C::Scalar>| {
        (
            combine(&e.zeta, zeta_m),
            combine(&e.zeta_omega, zeta_omega_m),
        )
    };
    let mut alpha_21 = C::Scalar::ONE;
    for _ in 0..PERMUTATION_ALPHA {
        alpha_21 = alpha_21 * alpha;
    }
    let roots = [
        d1.element(n - cs.zk_rows),
        d1.element(n - cs.zk_rows + 1),
        d1.element(n - 1),
    ];
    let vanishing = (zeta - roots[0]) * (zeta - roots[1]) * (zeta - roots[2]);
    let mut scalar = value(&evals.z).1 * beta * alpha_21 * vanishing;
    for i in 0..PERMUTS - 1 {
        scalar = scalar * (gamma + beta * value(&evals.s[i]).0 + value(&evals.w[i]).0);
    }
    let scalar = -scalar;
    let f = interpolate(&sigmas[PERMUTS - 1]).scale(scalar);
    let zeta_n_minus_1 = zeta.pow(n as u64) - C::Scalar::ONE;
    let f_lin = linearize(f.coefficients(), m, zeta_m);
    let t_lin = linearize(previous.t.coefficients(), m, zeta_m);
    let ft = &f_lin - &t_lin.scale(zeta_n_minus_1);
    let blinding_t = combine(&previous.t_blinders, zeta_m);
    let blinding_ft = -(zeta_n_minus_1 * blinding_t);
    let ft_eval1 = ft.evaluate(zeta_omega);

    let mut fr_sponge = FrSponge::<C::Scalar, T>::new(scalar_sponge.clone());
    fr_sponge.absorb(previous.fq_sponge.clone().digest());
    let prev_challenge_digest = FrSponge::<C::Scalar, T>::new(scalar_sponge.clone()).digest();
    fr_sponge.absorb(prev_challenge_digest);
    fr_sponge.absorb(ft_eval1);
    fr_sponge.absorb_multiple(&evals.public.zeta);
    fr_sponge.absorb_multiple(&evals.public.zeta_omega);
    for e in evals.absorbed() {
        fr_sponge.absorb_multiple(&e.zeta);
        fr_sponge.absorb_multiple(&e.zeta_omega);
    }
    let v_chal = fr_sponge.challenge();
    let v = v_chal.to_field(C::lambda());
    let u_chal = fr_sponge.challenge();
    let u = u_chal.to_field(C::lambda());
    EvaluationRound {
        previous,
        zeta_omega,
        evals,
        ft,
        blinding_ft,
        ft_eval1,
        fr_sponge,
        v_chal,
        v,
        u_chal,
        u,
    }
}
