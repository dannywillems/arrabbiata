//! Kimchi's verifier as a reduction of knowledge to Halo's accumulation
//! (`arrabbiata-halo`), the shape Pickles uses it in.
//!
//! [`Satisfied`] is the relation a Kimchi proof is about: an instance is
//! the public inputs and the previous accumulators the proof carries, a
//! witness the witness columns and the runtime tables; the pair is in it
//! when every previous accumulator is (its decider accepts) and the circuit
//! is satisfied. The circuit is checked by running the prover's own checks
//! on a fixed stream: the copies close the permutation product, the lookups
//! are in their tables, and the constraints divide by the vanishing
//! polynomial (which is the definition of satisfaction, not a heuristic).
//!
//! [`KimchiSuccinct`] `: Satisfied -> Accumulated` is the prover and the
//! succinct verifier ([`crate::verifier::succinct`]): everything Kimchi's
//! verifier does but the MSM `<s, G> = sg` of the size of the key. Its
//! target is Halo's [`Accumulated`] for arrabbiata-ipa's [`Ipa`] on
//! Kimchi's generators ([`ipa_of`]), whose accumulator is `(u, sg)` and
//! whose decider is that MSM. The previous accumulators need no separate
//! step: Kimchi opens each one's `sg` at `zeta` inside the same IPA (Halo's
//! lift), so the new accumulator stands for all of them.
//!
//! Then the rest is composition, with Halo's morphisms:
//!
//! - [`Verify`] `= KimchiSuccinct ; Decide : Satisfied -> Top`, Kimchi's
//!   whole verifier (two checks where Kimchi combines them with random
//!   scalars);
//! - [`AccumulateProofs`] `= Each(KimchiSuccinct) ; Fold : Satisfied^* ->
//!   Accumulated`, many proofs to one accumulator, then one `Decide`.
//!
//! Previous accumulators can be folded two ways, and both are morphisms here:
//!
//! - merged ([`KimchiSuccinct`] itself, Kimchi's way): the proof absorbs
//!   them first and opens their `b` at its own points, in its own IPA;
//! - separate ([`Separate`] `= Parallel(KimchiSuccinct ; Single, Identity)
//!   ; Concat ; Fold : Product(Satisfied_0, Accumulated^*) ->
//!   Accumulated`): a proof of the circuit with no previous accumulators,
//!   and Halo's fold of its accumulator with the previous ones, one more
//!   IPA.
//!
//! Both decide the same statements: the circuit satisfied and every
//! previous accumulator valid.
//!
//! The reduction's transcript (over the scalar field) is the composition's:
//! Kimchi's Fiat-Shamir runs on its own two sponges, and the reduction
//! absorbs its output accumulator into the composition's transcript, so
//! what follows (a fold) is bound to it.

use core::{cell::RefCell, fmt};

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::{PrimeField, Sqrt, TwoAdicField};
use arrabbiata_halo::{Accumulated, Decide, Fold};
use arrabbiata_ipa::{Ipa, RecursionChallenge};
use arrabbiata_reduction::{
    Batch, Concat, Each, Identity, Output, OutputOf, Parallel, Reduction, Relation, Single, Then,
};
use arrabbiata_transcript::{Absorb, Transcript};

use crate::{
    constraints::COLUMNS,
    gates::Gates,
    lookup::RuntimeTable,
    prover::{self, ProverIndex, ProverProof, prove},
    rng::{Arkworks, Sampler, SplitMix64},
    srs::Srs,
    verifier::{VerifyError, succinct},
};

/// The IPA on Kimchi's SRS: its generators for the coefficients and its
/// blinding point for the value (the two have no known relation), with
/// challenges from clones of `transcript`. Its accumulators are Kimchi's
/// `(u, sg)`, and its decider Kimchi's final MSM.
#[must_use]
pub fn ipa_of<C, T>(srs: &Srs<C>, transcript: T) -> Ipa<C, T>
where
    C: Glv,
    C::Base: PrimeField,
    T: Transcript<C::Scalar> + Clone,
{
    Ipa::from_generators(srs.g.clone(), srs.h, transcript)
}

/// What a Kimchi proof is about: the public inputs and the previous
/// accumulators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement<C>
where
    C: Glv,
{
    /// The public inputs.
    pub public: alloc::vec::Vec<C::Scalar>,
    /// The previous accumulators the proof carries.
    pub prev_challenges: alloc::vec::Vec<RecursionChallenge<C>>,
}

/// What the prover knows in addition: the witness columns and the runtime
/// tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment<F> {
    /// The witness columns, before padding.
    pub columns: [alloc::vec::Vec<F>; COLUMNS],
    /// The runtime tables' data.
    pub runtime_tables: alloc::vec::Vec<RuntimeTable<F>>,
}

/// Why a pair is not in [`Satisfied`], or the reduction failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The public inputs are not the first rows of column 0.
    Public,
    /// The previous accumulator of this index is not decided true.
    PreviousAccumulator(usize),
    /// The circuit is not satisfied, or the prover cannot run.
    Prove(prover::Error),
    /// The proof is rejected.
    Verify(VerifyError),
    /// The proof carries other previous accumulators than the statement.
    PrevChallenges,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => write!(f, "the public inputs are not the witness's"),
            Self::PreviousAccumulator(i) => write!(f, "previous accumulator {i} is not valid"),
            Self::Prove(e) => write!(f, "prover: {e}"),
            Self::Verify(e) => write!(f, "verifier: {e}"),
            Self::PrevChallenges => write!(f, "the proof carries other previous accumulators"),
        }
    }
}

/// The pieces both the relation and the reduction need, by reference.
#[derive(Debug)]
pub struct Satisfied<'a, C, G, S, T>
where
    C: Glv,
{
    index: &'a ProverIndex<C>,
    gates: &'a G,
    base: &'a S,
    scalar: &'a T,
    ipa: &'a Ipa<C, T>,
}

impl<C, G, S, T> Clone for Satisfied<'_, C, G, S, T>
where
    C: Glv,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, G, S, T> Copy for Satisfied<'_, C, G, S, T> where C: Glv {}

impl<'a, C, G, S, T> Satisfied<'a, C, G, S, T>
where
    C: Glv,
{
    /// The relation of the circuit of `index`, with Kimchi's `gates`, the
    /// fresh base- and scalar-field sponges, and the IPA on the index's
    /// SRS ([`ipa_of`]) that decides accumulators.
    #[must_use]
    pub const fn new(
        index: &'a ProverIndex<C>,
        gates: &'a G,
        sponges: (&'a S, &'a T),
        ipa: &'a Ipa<C, T>,
    ) -> Self {
        Self {
            index,
            gates,
            base: sponges.0,
            scalar: sponges.1,
            ipa,
        }
    }
}

impl<C, G, S, T> Relation for Satisfied<'_, C, G, S, T>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
{
    type Instance = Statement<C>;
    type Witness = Assignment<C::Scalar>;
    type Error = Error;

    fn check(
        &self,
        statement: &Statement<C>,
        assignment: &Assignment<C::Scalar>,
    ) -> Result<(), Error> {
        let public = self.index.cs.public;
        if statement.public.len() != public
            || assignment.columns[0].get(..public) != Some(statement.public.as_slice())
        {
            return Err(Error::Public);
        }
        for (i, previous) in statement.prev_challenges.iter().enumerate() {
            if !self.ipa.decide(previous) {
                return Err(Error::PreviousAccumulator(i));
            }
        }
        // The prover's checks on any stream: copies, lookups, constraints.
        let mut rng = Arkworks(SplitMix64(0));
        prove(
            self.index,
            assignment.columns.clone(),
            &assignment.runtime_tables,
            &statement.prev_challenges,
            self.gates,
            (self.base, self.scalar),
            &mut rng,
        )
        .map(|_| ())
        .map_err(Error::Prove)
    }
}

/// `Satisfied -> Accumulated`: Kimchi's prover, and its succinct verifier
/// outputting the accumulator `(u, sg)`. The prover's randomness is the
/// sampler it holds.
#[derive(Debug)]
pub struct KimchiSuccinct<'a, C, G, S, T, R>
where
    C: Glv,
{
    relation: Satisfied<'a, C, G, S, T>,
    rng: RefCell<R>,
}

impl<'a, C, G, S, T, R> KimchiSuccinct<'a, C, G, S, T, R>
where
    C: Glv,
{
    /// The reduction for `relation`, the prover drawing from `rng`.
    #[must_use]
    pub const fn new(relation: Satisfied<'a, C, G, S, T>, rng: R) -> Self {
        Self {
            relation,
            rng: RefCell::new(rng),
        }
    }
}

impl<'a, C, G, S, T, R> Reduction<C::Scalar> for KimchiSuccinct<'a, C, G, S, T, R>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
    G: Gates<C::Scalar>,
    R: Sampler<C::Scalar>,
{
    type Source = Satisfied<'a, C, G, S, T>;
    type Target = Accumulated<'a, C::Scalar, Ipa<C, T>>;
    type Proof = ProverProof<C>;
    type Error = Error;

    fn source(&self) -> Self::Source {
        self.relation
    }

    fn target(&self) -> Self::Target {
        Accumulated::new(self.relation.ipa)
    }

    fn prove<X>(
        &self,
        mut transcript: X,
        statement: &Statement<C>,
        assignment: &Assignment<C::Scalar>,
    ) -> Result<OutputOf<C::Scalar, Self, X>, Error>
    where
        X: Transcript<C::Scalar>,
    {
        let r = &self.relation;
        let proof = prove(
            r.index,
            assignment.columns.clone(),
            &assignment.runtime_tables,
            &statement.prev_challenges,
            r.gates,
            (r.base, r.scalar),
            &mut *self.rng.borrow_mut(),
        )
        .map_err(Error::Prove)?;
        let accumulator = succinct(
            &r.index.verifier_index,
            &r.index.srs,
            &proof,
            &statement.public,
            r.gates,
            (r.base, r.scalar),
        )
        .map_err(Error::Verify)?;
        accumulator.absorb_into(&mut transcript);
        Ok(Output {
            instance: accumulator,
            witness: (),
            proof,
            transcript,
        })
    }

    fn verify<X>(
        &self,
        mut transcript: X,
        statement: &Statement<C>,
        proof: ProverProof<C>,
    ) -> Result<(RecursionChallenge<C>, X), Error>
    where
        X: Transcript<C::Scalar>,
    {
        if proof.prev_challenges != statement.prev_challenges {
            return Err(Error::PrevChallenges);
        }
        let r = &self.relation;
        let accumulator = succinct(
            &r.index.verifier_index,
            &r.index.srs,
            &proof,
            &statement.public,
            r.gates,
            (r.base, r.scalar),
        )
        .map_err(Error::Verify)?;
        accumulator.absorb_into(&mut transcript);
        Ok((accumulator, transcript))
    }
}

/// Kimchi's whole verifier as a composite: `KimchiSuccinct ; Decide`.
pub type Verify<'a, C, G, S, T, R> = Then<
    KimchiSuccinct<'a, C, G, S, T, R>,
    Decide<'a, <C as arrabbiata_curve::Curve>::Scalar, Ipa<C, T>>,
>;

/// Many proofs to one accumulator: `Each(KimchiSuccinct) ; Fold`.
pub type AccumulateProofs<'a, C, G, S, T, R> = Then<
    Each<KimchiSuccinct<'a, C, G, S, T, R>>,
    Fold<'a, <C as arrabbiata_curve::Curve>::Scalar, Ipa<C, T>>,
>;

/// The composite [`Verify`] over `reduction`'s IPA.
#[must_use]
pub const fn verify<'a, C, G, S, T, R>(
    reduction: KimchiSuccinct<'a, C, G, S, T, R>,
) -> Verify<'a, C, G, S, T, R>
where
    C: Glv,
{
    let ipa = reduction.relation.ipa;
    Then(reduction, Decide::new(ipa))
}

/// The composite [`AccumulateProofs`] over `reduction`'s IPA.
#[must_use]
pub const fn accumulate_proofs<'a, C, G, S, T, R>(
    reduction: KimchiSuccinct<'a, C, G, S, T, R>,
) -> AccumulateProofs<'a, C, G, S, T, R>
where
    C: Glv,
{
    let ipa = reduction.relation.ipa;
    Then(Each(reduction), Fold::new(ipa))
}

/// Halo's accumulated relation over `C`'s IPA with transcript `T`.
type Acc<'a, C, T> = Accumulated<'a, <C as arrabbiata_curve::Curve>::Scalar, Ipa<C, T>>;

/// Previous accumulators folded separately: `Parallel(KimchiSuccinct ;
/// Single, Identity) ; Concat ; Fold`, from a proof of a circuit with no
/// previous accumulator and the accumulators themselves.
pub type Separate<'a, C, G, S, T, R> = Then<
    Parallel<
        Then<KimchiSuccinct<'a, C, G, S, T, R>, Single<Acc<'a, C, T>>>,
        Identity<Batch<Acc<'a, C, T>>>,
    >,
    Then<Concat<Acc<'a, C, T>>, Fold<'a, <C as arrabbiata_curve::Curve>::Scalar, Ipa<C, T>>>,
>;

/// The composite [`Separate`] over `reduction`, whose index must carry no
/// previous accumulator.
#[must_use]
pub const fn separate<'a, C, G, S, T, R>(
    reduction: KimchiSuccinct<'a, C, G, S, T, R>,
) -> Separate<'a, C, G, S, T, R>
where
    C: Glv,
{
    let ipa = reduction.relation.ipa;
    let accumulated = Accumulated::new(ipa);
    Then(
        Parallel(
            Then(reduction, Single(accumulated)),
            Identity(Batch(accumulated)),
        ),
        Then(Concat(accumulated), Fold::new(ipa)),
    )
}
