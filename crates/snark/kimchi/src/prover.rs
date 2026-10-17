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
//! The randomness is the [`Sampler`] passed in; with [`crate::rng::Arkworks`]
//! over the same stream as Kimchi's RNG, the round gives Kimchi's values.
//! Lookups are not supported yet: a circuit with Lookup, RangeCheck, Rot64,
//! Xor16 or ForeignFieldMul rows, or with tables, is refused.

use alloc::vec::Vec;
use core::fmt;

use arrabbiata_curve::Curve;
use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_transcript::Transcript;

use crate::{
    constraints::{COLUMNS, ConstraintSystem, zk_rows_strict_lower_bound},
    oracle::FqSponge,
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
    let commit = |column: &[C::Scalar]| {
        let polynomial = match cs.domain.d1.interpolate(column) {
            Some(p) => p,
            None => unreachable!("every column has one value per row"),
        };
        index.srs.commit_non_hiding(polynomial.coefficients(), n)
    };
    let public_comm = index.srs.mask_fixed(commit(&negated));
    fq_sponge.absorb_g(&public_comm.chunks);

    let blinders: [Vec<C::Scalar>; COLUMNS] =
        core::array::from_fn(|_| (0..num_chunks).map(|_| rng.sample()).collect());
    let w_comm: [PolyComm<C>; COLUMNS] = core::array::from_fn(|i| {
        let c = commit(&witness[i]);
        PolyComm {
            chunks: c
                .chunks
                .into_iter()
                .zip(&blinders[i])
                .map(|(chunk, blinder)| chunk + index.srs.h * *blinder)
                .collect(),
        }
    });
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
