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
//! 2. a base-field sponge absorbs the verifier-index digest, then the
//!    commitment `sg` of each previous accumulator;
//! 3. the public polynomial, the negated public inputs on the first rows,
//!    committed and masked by `h` with blinder 1, is absorbed;
//! 4. each column is committed, hidden by `h` with one random blinder per
//!    chunk (all fifteen columns' blinders drawn before any commitment),
//!    and the fifteen commitments are absorbed;
//! 5. with lookups ([`LookupRound`]): with runtime tables, their column
//!    (random on the zero-knowledge rows, from the last up) committed,
//!    hidden and absorbed; the joint combiner `jc` (the endomorphism image
//!    of a challenge when some lookup reads several columns, of zero
//!    otherwise) and the table-id combiner (`jc^max_joint_size` when some
//!    table has a nonzero id); the joint table; the sorted columns
//!    ([`crate::lookup::sorted`]), each random on its zero-knowledge rows,
//!    committed, hidden and absorbed;
//! 6. `beta` and `gamma` are squeezed, 128-bit challenges.
//!
//! [`permutation_round`], the second:
//!
//! 0. with lookups, their aggregation ([`crate::lookup::aggregation`]),
//!    random on its zero-knowledge rows, committed, hidden, absorbed;
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
//!    the previous accumulators' challenges (a fresh sponge absorbing them
//!    all, then squeezed), `ft(zeta w)`, the
//!    public evaluations, then `z`, the six selectors, the witness, the
//!    coefficients, the sigmas and the optional selectors, each at `zeta`
//!    then `zeta w`; `v` and `u` are squeezed and mapped by the
//!    endomorphism.
//!
//! [`opening_round`], the last: Kimchi's batched, hidden IPA opening of
//! every polynomial at `zeta` and `zeta w`:
//!
//! 1. the polynomials, in Kimchi's order (each previous accumulator's
//!    `b`, unhidden, then public, `ft`, `z`, the six
//!    selectors, the witness, the coefficients, the first six sigmas, the
//!    optional selectors), each taking as many consecutive powers of `v`
//!    as it has chunks, chunk `j` of a polynomial starting at `v^s` scaled
//!    by `v^(s + j)`; their blinders combined the same way (1 for the
//!    public polynomial and the six selectors, 0 for the index columns);
//! 2. `b = sum_k u^k (1, x_k, x_k^2, ...)` over the points, the combined
//!    inner product `<p, b>`, absorbed into the base sponge (as it was after
//!    `zeta`) shifted by Kimchi's `shift_scalar`; the base `U` is the group
//!    map of a squeezed base-field element;
//! 3. each round `L = <G_lo, a_hi> + r_l H + <a_hi, b_lo> U`, `R = <G_hi,
//!    a_lo> + r_r H + <a_lo, b_hi> U` (`r_l`, `r_r` random), absorbed; the
//!    round challenge `u_j` is the endomorphism image of a 128-bit
//!    challenge; `a <- a_lo + u_j^(-1) a_hi`, `b <- b_lo + u_j b_hi`,
//!    `G <- G_lo + u_j G_hi`;
//! 4. with `a, b, G` single, random `d`, `r_delta`: `delta = d (G + b U) +
//!    r_delta H`, absorbed, `c` the endomorphism image of a challenge,
//!    `z1 = a c + d`, `z2 = r' c + r_delta` with `r'` the blinder plus the
//!    rounds' `r_l u_j^(-1) + r_r u_j`; `sg = G`.
//!
//! This is the generic IPA of `arrabbiata-ipa` with Kimchi's choice of each
//! parameter (hiding, the base of the value, endomorphism challenges, and
//! the scales drawn by the protocol): [`crate::ipa::KimchiIpa`].
//!
//! [`prove`] runs the five rounds and gives the proof.
//!
//! The randomness is the [`Sampler`] passed in; with [`crate::rng::Arkworks`]
//! over the same stream as Kimchi's RNG, the round gives Kimchi's values.
//!
//! With lookups, the quotient adds Kimchi's lookup constraints
//! ([`crate::lookup::constraints`]) from `alpha^24` ([`LOOKUP_ALPHA`]); the
//! evaluations add the aggregation, the joint table, the sorted columns,
//! the runtime table and selector and the pattern selectors; the opening
//! adds them in that order, the joint table blinded by the combination of
//! its parts' blinders (1 for each fixed column and for the ids, the
//! runtime table's own).

use alloc::vec::Vec;
use core::fmt;

use arrabbiata_curve::{Curve, Projective, glv::Glv, msm};
use arrabbiata_field::{Field, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_ipa::{
    RecursionChallenge, Statement, b_poly_coefficients, evaluation_vector, generic::Witness,
};
use arrabbiata_poly::UniPoly;
use arrabbiata_transcript::Transcript;

use crate::{
    constraints::{COLUMNS, ConstraintSystem, GateType, zk_rows_strict_lower_bound},
    gates::Gates,
    ipa::kimchi_ipa_with,
    lookup::{LookupError, LookupIndex, LookupPattern, LookupPoint, RuntimeTable},
    oracle::{FqSponge, FrSponge, ScalarChallenge},
    permutation::PERMUTS,
    rng::Sampler,
    srs::{GroupMap, PolyComm, Srs},
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
    /// A lookup fails: a value in no table, or runtime tables other than
    /// the configured ones.
    Lookup(LookupError),
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
            Self::Lookup(e) => write!(f, "lookup: {e}"),
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
    /// The previous accumulators the proof carries.
    pub prev_challenges: Vec<RecursionChallenge<C>>,
    /// The commitment to the public polynomial (masked).
    pub public_comm: PolyComm<C>,
    /// The blinders of the witness commitments, one per chunk.
    pub blinders: [Vec<C::Scalar>; COLUMNS],
    /// The witness commitments.
    pub w_comm: [PolyComm<C>; COLUMNS],
    /// The lookup part, when the circuit looks something up.
    pub lookup: Option<LookupRound<C>>,
    /// The base-field sponge after `gamma`.
    pub fq_sponge: FqSponge<C, S>,
    /// The first permutation challenge.
    pub beta: C::Scalar,
    /// The second permutation challenge.
    pub gamma: C::Scalar,
}

/// A column committed with hiding: its values on `d1`, its blinders (one
/// per chunk) and its commitment.
#[derive(Clone, Debug)]
pub struct Hidden<C>
where
    C: Curve,
{
    /// The values on `d1`.
    pub values: Vec<C::Scalar>,
    /// The blinders.
    pub blinders: Vec<C::Scalar>,
    /// The commitment.
    pub comm: PolyComm<C>,
}

/// The lookup part of the witness round: the runtime table, the combiners,
/// the joint table and the sorted columns.
#[derive(Clone, Debug)]
pub struct LookupRound<C>
where
    C: Curve,
{
    /// With runtime tables, their contribution to the second column, its
    /// zero-knowledge rows random.
    pub runtime: Option<Hidden<C>>,
    /// The 128-bit challenge the joint combiner comes from (zero when no
    /// lookup reads more than one column).
    pub joint_combiner_chal: ScalarChallenge<C::Scalar>,
    /// The joint combiner.
    pub joint_combiner: C::Scalar,
    /// The weight of the table ids: `jc^max_joint_size` when some table has
    /// a nonzero id, zero otherwise.
    pub table_id_combiner: C::Scalar,
    /// The joint table on `d1`.
    pub joint_table: Vec<C::Scalar>,
    /// The sorted columns, their zero-knowledge rows random.
    pub sorted: Vec<Hidden<C>>,
}

/// A column's zero-knowledge rows random (Kimchi's `zk_patch`): zeros up to
/// row `n - zk_rows`, then `zk_rows` random values in order.
fn zk_patch<F: Field, R: Sampler<F>>(
    mut values: Vec<F>,
    n: usize,
    zk_rows: usize,
    rng: &mut R,
) -> Vec<F> {
    values.resize(n - zk_rows, F::ZERO);
    values.extend((0..zk_rows).map(|_| rng.sample()));
    values
}

/// `values` committed and hidden, blinders drawn after.
fn hidden<C, R>(index: &ProverIndex<C>, values: Vec<C::Scalar>, rng: &mut R) -> Hidden<C>
where
    C: Curve,
    C::Scalar: TwoAdicField,
    R: Sampler<C::Scalar>,
{
    let blinders: Vec<C::Scalar> = (0..index.num_chunks()).map(|_| rng.sample()).collect();
    let comm = hide(index, commit_column(index, &values), &blinders);
    Hidden {
        values,
        blinders,
        comm,
    }
}

/// The lookup steps of the witness round, after the witness commitments:
/// the runtime table (its zero-knowledge rows random from the last row up,
/// committed, absorbed), the joint combiner (squeezed when some lookup
/// reads more than one column), the joint table, and the sorted columns
/// (their zero-knowledge rows random, committed, absorbed).
fn lookup_round<C, S, R>(
    index: &ProverIndex<C>,
    lookup: &LookupIndex<C::Scalar>,
    witness: &[Vec<C::Scalar>],
    runtime_tables: &[RuntimeTable<C::Scalar>],
    fq_sponge: &mut FqSponge<C, S>,
    rng: &mut R,
) -> Result<LookupRound<C>, Error>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    R: Sampler<C::Scalar>,
{
    let cs = &index.cs;
    let n = cs.domain.d1.size();
    let zk_rows = cs.zk_rows;
    let runtime = if lookup.runtime_selector.is_some() {
        let mut values = lookup
            .runtime_contribution(runtime_tables, n)
            .map_err(Error::Lookup)?;
        for value in values.iter_mut().rev().take(zk_rows) {
            *value = rng.sample();
        }
        let column = hidden(index, values, rng);
        fq_sponge.absorb_g(&column.comm.chunks);
        Some(column)
    } else {
        None
    };
    let raw = if lookup.info.joint_lookup_used() {
        fq_sponge.challenge()
    } else {
        C::Scalar::ZERO
    };
    let joint_combiner_chal = ScalarChallenge(raw);
    let joint_combiner = joint_combiner_chal.to_field(C::lambda());
    let table_id_combiner = if lookup.table_ids.is_some() {
        joint_combiner.pow(lookup.info.max_joint_size as u64)
    } else {
        C::Scalar::ZERO
    };
    let joint_table = lookup.joint_table(
        joint_combiner,
        table_id_combiner,
        runtime.as_ref().map(|r| r.values.as_slice()),
    );
    let columns = crate::lookup::sorted(
        &lookup.info,
        &cs.gates,
        witness,
        &joint_table,
        joint_combiner,
        table_id_combiner,
        C::Scalar::ZERO,
        zk_rows,
    )
    .map_err(Error::Lookup)?;
    let patched: Vec<Vec<C::Scalar>> = columns
        .into_iter()
        .map(|c| zk_patch(c, n, zk_rows, rng))
        .collect();
    let sorted: Vec<Hidden<C>> = patched.into_iter().map(|c| hidden(index, c, rng)).collect();
    for column in &sorted {
        fq_sponge.absorb_g(&column.comm.chunks);
    }
    Ok(LookupRound {
        runtime,
        joint_combiner_chal,
        joint_combiner,
        table_id_combiner,
        joint_table,
        sorted,
    })
}

/// The witness round (see the module documentation). `sponge` is a fresh
/// base-field sponge, cloned for the digest and for the proof's transcript.
///
/// # Errors
///
/// [`Error::WitnessCsInconsistent`] for columns of different lengths,
/// [`Error::NoRoomForZkInWitness`], [`Error::NotZeroKnowledge`], and
/// [`Error::Lookup`] for a lookup of a value in no table or runtime tables
/// other than the configured ones.
pub fn witness_round<C, S, R>(
    index: &ProverIndex<C>,
    witness: [Vec<C::Scalar>; COLUMNS],
    runtime_tables: &[RuntimeTable<C::Scalar>],
    prev_challenges: &[RecursionChallenge<C>],
    sponge: &S,
    rng: &mut R,
) -> Result<WitnessRound<C, S>, Error>
where
    C: Glv,
    C::Base: PrimeField,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    R: Sampler<C::Scalar>,
{
    let cs = &index.cs;
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
    for previous in prev_challenges {
        fq_sponge.absorb_g(&[previous.comm]);
    }

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
    let lookup = match &cs.lookup {
        None => None,
        Some(l) => Some(lookup_round(
            index,
            l,
            &witness,
            runtime_tables,
            &mut fq_sponge,
            rng,
        )?),
    };
    let beta = fq_sponge.challenge();
    let gamma = fq_sponge.challenge();
    Ok(WitnessRound {
        witness,
        public,
        prev_challenges: prev_challenges.to_vec(),
        public_comm,
        blinders,
        w_comm,
        lookup,
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
    /// With lookups, their aggregation, committed before `z`.
    pub aggregation: Option<Hidden<C>>,
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
    let mut fq_sponge = previous.fq_sponge.clone();
    let aggregation = match (&cs.lookup, &previous.lookup) {
        (Some(l), Some(round)) => {
            let sorted: Vec<Vec<C::Scalar>> =
                round.sorted.iter().map(|c| c.values.clone()).collect();
            let values = crate::lookup::aggregation(
                &l.info,
                &cs.gates,
                w,
                &round.joint_table,
                &sorted,
                (beta, gamma, round.joint_combiner, round.table_id_combiner),
                C::Scalar::ZERO,
                zk_rows,
            );
            let column = hidden(index, zk_patch(values, n, zk_rows, rng), rng);
            fq_sponge.absorb_g(&column.comm.chunks);
            Some(column)
        }
        _ => None,
    };
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
    fq_sponge.absorb_g(&z_comm.chunks);
    let alpha_chal = ScalarChallenge(fq_sponge.challenge());
    let alpha = alpha_chal.to_field(C::lambda());
    Ok(PermutationRound {
        previous,
        aggregation,
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

/// The first power of `alpha` of the lookup constraints, after the
/// permutation's three.
pub const LOOKUP_ALPHA: usize = PERMUTATION_ALPHA + 3;

/// `(x^n - 1) / (x - a)` for `a` an `n`-th root of unity (Kimchi's
/// unnormalized Lagrange basis), `n a^(n - 1)` at `x = a`.
pub(crate) fn unnormalized_lagrange<F: Field>(x: F, a: F, n: usize) -> F {
    match (x - a).inverse() {
        Some(inverse) => (x.pow(n as u64) - F::ONE) * inverse,
        None => F::from_u64(n as u64) * a.pow(n as u64 - 1),
    }
}

/// The gate types with constraints, in Kimchi's order.
pub(crate) const CONSTRAINED: [GateType; 12] = [
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
    let lookup_constraints = cs
        .lookup
        .as_ref()
        .map_or(0, |l| crate::lookup::constraint_count(&l.info));
    let mut alphas = alloc::vec![C::Scalar::ONE; LOOKUP_ALPHA + lookup_constraints];
    for k in 1..alphas.len() {
        alphas[k] = alphas[k - 1] * alpha;
    }
    // The lookup columns on d8.
    let lookup8 = match (&cs.lookup, &first.lookup, &previous.aggregation) {
        (Some(l), Some(round), Some(aggregation)) => Some((
            l,
            round,
            round
                .sorted
                .iter()
                .map(|c| extend(&c.values))
                .collect::<Vec<_>>(),
            extend(&aggregation.values),
            extend(&round.joint_table),
            l.selectors
                .iter()
                .map(|(p, sel)| (*p, extend(sel)))
                .collect::<Vec<_>>(),
            match (&round.runtime, &l.runtime_selector) {
                (Some(r), Some(sel)) => Some((extend(&r.values), extend(sel))),
                _ => None,
            },
        )),
        _ => None,
    };
    let lookup_roots: Vec<C::Scalar> = (n - zk_rows - 1..n).map(|i| d1.element(i)).collect();
    let last_lookup_row = d1.element(n - zk_rows - 1);
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
        if let Some((l, round, sorted8, aggregation8, joint8, selectors8, runtime8)) = &lookup8 {
            let next = (k + 8) % big;
            let point = LookupPoint {
                witness: w8.iter().map(|c| c[k]).collect(),
                sorted: sorted8.iter().map(|c| (c[k], c[next])).collect(),
                aggregation: (aggregation8[k], aggregation8[next]),
                table: (joint8[k], joint8[next]),
                selectors: selectors8.iter().map(|(p, c)| (*p, c[k])).collect(),
                runtime: runtime8.as_ref().map(|(t, sel)| (t[k], sel[k])),
                vanishing: lookup_roots
                    .iter()
                    .fold(C::Scalar::ONE, |acc, r| acc * (x - *r)),
                first: unnormalized_lagrange(x, C::Scalar::ONE, n),
                last: unnormalized_lagrange(x, last_lookup_row, n),
            };
            let values =
                crate::lookup::constraints(&l.info, &point, beta, gamma, round.joint_combiner);
            for (i, c) in values.iter().enumerate() {
                value = value + alphas[LOOKUP_ALPHA + i] * *c;
            }
        }
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

/// The lookup evaluations a proof carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupEvaluations<F> {
    /// The aggregation.
    pub aggregation: PointEvaluations<F>,
    /// The joint table.
    pub table: PointEvaluations<F>,
    /// The sorted columns.
    pub sorted: Vec<PointEvaluations<F>>,
    /// With runtime tables, the runtime table.
    pub runtime_table: Option<PointEvaluations<F>>,
    /// With runtime tables, the runtime selector.
    pub runtime_selector: Option<PointEvaluations<F>>,
    /// The selector of each pattern used, in Kimchi's order.
    pub selectors: Vec<(LookupPattern, PointEvaluations<F>)>,
}

/// The evaluations a proof carries (Kimchi's `ProofEvaluations`).
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
    /// The lookup evaluations, when the circuit looks something up.
    pub lookup: Option<LookupEvaluations<F>>,
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
        if let Some(l) = &self.lookup {
            points.push(&l.aggregation);
            points.push(&l.table);
            points.extend(l.sorted.iter());
            points.extend(
                [&l.runtime_table, &l.runtime_selector]
                    .into_iter()
                    .flatten(),
            );
            points.extend(l.selectors.iter().map(|(_, e)| e));
        }
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
pub(crate) fn chunk_evaluations<F: Field>(
    coefficients: &[F],
    num_chunks: usize,
    size: usize,
    x: F,
) -> Vec<F> {
    let mut values: Vec<F> = coefficients
        .chunks(size)
        .map(|chunk| UniPoly::new(chunk.to_vec()).evaluate(x))
        .collect();
    values.resize(values.len().max(num_chunks), F::ZERO);
    values
}

/// `sum_i c_i y^i`: chunk evaluations combined into the value at the point
/// (`y` the point to the chunk size).
pub(crate) fn combine<F: Field>(chunks: &[F], y: F) -> F {
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
        lookup: match (&cs.lookup, &first.lookup, &permutation.aggregation) {
            (Some(l), Some(round), Some(aggregation)) => Some(LookupEvaluations {
                aggregation: column(&aggregation.values),
                table: column(&round.joint_table),
                sorted: round.sorted.iter().map(|c| column(&c.values)).collect(),
                runtime_table: round.runtime.as_ref().map(|r| column(&r.values)),
                runtime_selector: l.runtime_selector.as_ref().map(|c| column(c)),
                selectors: l.selectors.iter().map(|(p, c)| (*p, column(c))).collect(),
            }),
            _ => None,
        },
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
    let mut challenges = FrSponge::<C::Scalar, T>::new(scalar_sponge.clone());
    for previous in &first.prev_challenges {
        challenges.absorb_multiple(&previous.chals);
    }
    let prev_challenge_digest = challenges.digest();
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

/// Kimchi's opening proof: the rounds' `(L, R)`, `delta`, `z1`, `z2` and
/// the folded generator `sg`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningProof<C>
where
    C: Curve,
{
    /// The `(L, R)` of each round.
    pub lr: Vec<(Projective<C>, Projective<C>)>,
    /// The commitment of the final Schnorr-like step.
    pub delta: Projective<C>,
    /// The response for the folded coefficient.
    pub z1: C::Scalar,
    /// The response for the blinder.
    pub z2: C::Scalar,
    /// The folded generator.
    pub sg: Projective<C>,
}

/// The lookup commitments of a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupCommitments<C>
where
    C: Curve,
{
    /// The sorted columns.
    pub sorted: Vec<PolyComm<C>>,
    /// The aggregation.
    pub aggreg: PolyComm<C>,
    /// With runtime tables, the runtime table.
    pub runtime: Option<PolyComm<C>>,
}

/// A Kimchi proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverProof<C>
where
    C: Curve,
{
    /// The witness commitments.
    pub w_comm: [PolyComm<C>; COLUMNS],
    /// The permutation aggregation's commitment.
    pub z_comm: PolyComm<C>,
    /// The quotient's commitment.
    pub t_comm: PolyComm<C>,
    /// The lookup commitments, when the circuit looks something up.
    pub lookup: Option<LookupCommitments<C>>,
    /// The evaluations.
    pub evals: ProofEvaluations<C::Scalar>,
    /// `ft(zeta w)`.
    pub ft_eval1: C::Scalar,
    /// The opening.
    pub proof: OpeningProof<C>,
    /// The previous accumulators the proof carries.
    pub prev_challenges: Vec<RecursionChallenge<C>>,
}

/// Kimchi's `shift_scalar`, now the generic IPA's (its [`BindValue`]
/// binding absorbs it).
///
/// [`BindValue`]: arrabbiata_ipa::generic::BindValue
pub use arrabbiata_ipa::generic::shift_scalar;

/// A polynomial to open, its coefficients and one blinder per chunk.
type Blinded<F> = (Vec<F>, Vec<F>);

/// The inner product.
pub(crate) fn inner<F: Field>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::ZERO, |acc, (x, y)| acc + *x * *y)
}

/// The opening round (see the module documentation), with Kimchi's group
/// map for `U`.
///
/// # Panics
///
/// If the SRS size is not a power of two.
pub fn opening_round<C, S, T, R>(
    index: &ProverIndex<C>,
    previous: &EvaluationRound<C, S, T>,
    group_map: &GroupMap<C>,
    rng: &mut R,
) -> ProverProof<C>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    R: Sampler<C::Scalar>,
{
    let cs = &index.cs;
    let d1 = cs.domain.d1;
    let n = d1.size();
    let m = index.srs.max_poly_size();
    assert!(m.is_power_of_two(), "the SRS size must be a power of two");
    let num_chunks = index.num_chunks();
    let quotient = &previous.previous;
    let permutation = &quotient.previous;
    let first = &permutation.previous;
    let (polyscale, evalscale) = (previous.v, previous.u);
    let interpolate = |column: &[C::Scalar]| match d1.interpolate(column) {
        Some(p) => p.coefficients().to_vec(),
        None => unreachable!("one value per row"),
    };
    let ones = alloc::vec![C::Scalar::ONE; num_chunks];
    let zeros = alloc::vec![C::Scalar::ZERO; num_chunks];
    let mut negated = alloc::vec![C::Scalar::ZERO; n];
    for (slot, p) in negated.iter_mut().zip(&first.public) {
        *slot = -*p;
    }
    let sigmas = cs.sigmas();
    let mut polynomials: Vec<Blinded<C::Scalar>> = first
        .prev_challenges
        .iter()
        .map(|previous| {
            let b = b_poly_coefficients(&previous.chals);
            (b, alloc::vec![C::Scalar::ZERO])
        })
        .collect();
    polynomials.extend([
        (interpolate(&negated), ones.clone()),
        (
            previous.ft.coefficients().to_vec(),
            alloc::vec![previous.blinding_ft],
        ),
        (interpolate(&permutation.z), permutation.z_blinders.clone()),
    ]);
    for typ in [
        GateType::Generic,
        GateType::Poseidon,
        GateType::CompleteAdd,
        GateType::VarBaseMul,
        GateType::EndoMul,
        GateType::EndoMulScalar,
    ] {
        polynomials.push((interpolate(&cs.selector(typ)), ones.clone()));
    }
    for (w, blinders) in first.witness.iter().zip(&first.blinders) {
        polynomials.push((interpolate(w), blinders.clone()));
    }
    for i in 0..COLUMNS {
        polynomials.push((interpolate(&cs.coefficient(i)), zeros.clone()));
    }
    for sigma in &sigmas[..PERMUTS - 1] {
        polynomials.push((interpolate(sigma), zeros.clone()));
    }
    for (typ, _) in &previous.evals.optional_selectors {
        polynomials.push((interpolate(&cs.selector(*typ)), zeros.clone()));
    }
    if let (Some(l), Some(round), Some(aggregation)) =
        (&cs.lookup, &first.lookup, &permutation.aggregation)
    {
        for column in &round.sorted {
            polynomials.push((interpolate(&column.values), column.blinders.clone()));
        }
        polynomials.push((
            interpolate(&aggregation.values),
            aggregation.blinders.clone(),
        ));
        // The joint table's blinder: each fixed column's 1 by its power of
        // jc, the ids' 1 by the id combiner, the runtime blinders by jc.
        let jc = round.joint_combiner;
        let width = l.table.len();
        let fixed = if width == 0 {
            C::Scalar::ZERO
        } else {
            (1..width).fold(C::Scalar::ONE, |acc, _| C::Scalar::ONE + jc * acc)
        };
        let base = fixed + round.table_id_combiner;
        let table_blinders = match &round.runtime {
            Some(r) => r.blinders.iter().map(|b| jc * *b + base).collect(),
            None => alloc::vec![base; num_chunks],
        };
        polynomials.push((interpolate(&round.joint_table), table_blinders));
        if let Some(r) = &round.runtime {
            polynomials.push((interpolate(&r.values), r.blinders.clone()));
        }
        if let Some(selector) = &l.runtime_selector {
            polynomials.push((interpolate(selector), zeros.clone()));
        }
        for (_, selector) in &l.selectors {
            polynomials.push((interpolate(selector), zeros.clone()));
        }
    }

    let mut a = alloc::vec![C::Scalar::ZERO; m];
    let mut blinding = C::Scalar::ZERO;
    let mut scale = C::Scalar::ONE;
    for (coefficients, blinders) in &polynomials {
        for (j, blinder) in blinders.iter().enumerate() {
            let start = (j * m).min(coefficients.len());
            let end = ((j + 1) * m).min(coefficients.len());
            for (slot, c) in a.iter_mut().zip(&coefficients[start..end]) {
                *slot = *slot + scale * *c;
            }
            blinding = blinding + scale * *blinder;
            scale = scale * polyscale;
        }
    }
    // Kimchi's IPA (crate::ipa) on the combined polynomial and blinder,
    // continuing the base sponge as it was after zeta.
    let ipa = kimchi_ipa_with(&index.srs, *group_map);
    let points = [quotient.zeta, previous.zeta_omega];
    let b = evaluation_vector(&points, evalscale, m);
    let combined_inner_product = inner(&a, &b);
    let commitment = msm(&index.srs.g, &a) + index.srs.h * blinding;
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale,
        value: combined_inner_product,
    };
    let witness = Witness {
        coefficients: &a,
        blinder: blinding,
    };
    let mut sponge = quotient.fq_sponge.clone();
    let (opening, _) = ipa.prove(&mut sponge, &statement, &witness, rng);
    ProverProof {
        w_comm: first.w_comm.clone(),
        z_comm: permutation.z_comm.clone(),
        t_comm: quotient.t_comm.clone(),
        lookup: match (&first.lookup, &permutation.aggregation) {
            (Some(round), Some(aggregation)) => Some(LookupCommitments {
                sorted: round.sorted.iter().map(|c| c.comm.clone()).collect(),
                aggreg: aggregation.comm.clone(),
                runtime: round.runtime.as_ref().map(|r| r.comm.clone()),
            }),
            _ => None,
        },
        evals: previous.evals.clone(),
        ft_eval1: previous.ft_eval1,
        proof: opening.into(),
        prev_challenges: first.prev_challenges.clone(),
    }
}

/// A whole proof: the five rounds in order, on fresh base- and scalar-field
/// sponges, with Kimchi's group map.
///
/// # Errors
///
/// As the rounds: an inconsistent or too long witness, a broken copy, or a
/// broken constraint.
pub fn prove<C, S, T, R, G>(
    index: &ProverIndex<C>,
    witness: [Vec<C::Scalar>; COLUMNS],
    runtime_tables: &[RuntimeTable<C::Scalar>],
    prev_challenges: &[RecursionChallenge<C>],
    gates: &G,
    sponges: (&S, &T),
    rng: &mut R,
) -> Result<ProverProof<C>, Error>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    R: Sampler<C::Scalar>,
    G: Gates<C::Scalar>,
{
    let (base_sponge, scalar_sponge) = sponges;
    let first = witness_round(
        index,
        witness,
        runtime_tables,
        prev_challenges,
        base_sponge,
        rng,
    )?;
    let second = permutation_round(index, first, rng)?;
    let third = quotient_round(index, second, gates, rng)?;
    let fourth = evaluation_round(index, third, scalar_sponge);
    let group_map = GroupMap::setup();
    Ok(opening_round(index, &fourth, &group_map, rng))
}
