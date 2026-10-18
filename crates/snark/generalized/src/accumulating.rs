//! Generalized PlonK compiled with an [`Accumulation`]: a reduction from
//! circuit satisfiability to the scheme's accumulator relation, a separate
//! implementation from [`crate::protocol`]'s batched openings.
//!
//! It runs the same [`Rounds`] of the polynomial IOP, then sends every
//! polynomial at every point of the protocol (Kimchi's layout: `zeta`, and
//! `zeta w^r` for each rotation `r` the protocol reads, `zeta w` among
//! them), and hands the whole batch to the scheme's succinct verifier,
//! [`arrabbiata_halo::Succinct`]. The verifier checks the identity at
//! `zeta` on the sparse view of the evaluations and outputs the
//! accumulator; it never runs the decider.
//!
//! In the category of reductions:
//!
//! ```text
//! Accumulating : Satisfiability -> Evaluations -> Accumulated
//!              = Piop ; Succinct
//! ```
//!
//! so `Then(Accumulating, Decide)` is the SNARK, and `Then(Each(Accumulating),
//! Fold)` accumulates many proofs into one accumulator with one decider:
//! the composites of `arrabbiata-halo`, with the IOP in front.

use alloc::{collections::BTreeSet, vec::Vec};
use core::marker::PhantomData;

use arrabbiata_commit::{PolynomialCommitment, PolynomialFamily, Univariate};
use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_halo::{Accumulated, Accumulation, Claims};
use arrabbiata_poly::{Domain, Rotation, UniPoly};
use arrabbiata_reduction::{Output, OutputOf, Reduction, Relation};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, absorb_slice,
    typed::{End, Label, Message, ProofOf, Protocol, Seq, Tail, Tape, prove_within, verify_within},
};

use crate::{
    Circuit, Error, Variable,
    lookup::Argument,
    protocol::{
        Committed, Opened, Polynomials, Preprocessed, ProverKey, Rounds, check_identity,
        prove_rounds, root_power, rotated, verify_rounds,
    },
};

/// The value of every opened polynomial at every point: one row per
/// polynomial (the witness columns, the fixed columns, the selectors, the
/// permutation polynomials, `z`, `t`, the lookup columns before `beta,
/// gamma`, then after), one column per point, in the order of
/// [`rotations`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Values<F>(pub Vec<Vec<F>>);

impl<F> Label for Values<F> {
    const LABEL: u64 = 0x6776_616c;
}

impl<F> Absorb<F> for Values<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_length(transcript, self.0.len());
        for row in &self.0 {
            absorb_slice(transcript, row);
        }
    }
}

/// The transcript of the PlonK part: the [`Rounds`], then the [`Values`].
/// The reduction's whole pattern is this, then the scheme's opening
/// ([`Accumulation::OpenPattern`]).
pub type AccumulatingTranscript<F, PC, const N: usize, const D: usize> =
    Rounds<F, PC, N, D, Message<Values<F>, End>>;

/// Generalized PlonK over `F`, compiled with the accumulation `PC`.
#[derive(Debug)]
pub struct AccumulatingPlonk<F, PC, const N: usize, const D: usize>(PhantomData<fn() -> (F, PC)>);

impl<F, PC, const N: usize, const D: usize> Protocol for AccumulatingPlonk<F, PC, N, D>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
{
    type Transcript = AccumulatingTranscript<F, PC, N, D>;
}

/// A proof: the messages of the pattern, and the scheme's opening of the
/// batch.
pub struct AccumulatingProof<F, PC, const N: usize, const D: usize>
where
    F: TwoAdicField,
    PC: Accumulation<F, Family = Univariate> + Label,
{
    /// The messages of the [`AccumulatingTranscript`].
    pub messages: ProofOf<AccumulatingPlonk<F, PC, N, D>>,
    /// The scheme's opening of the evaluations.
    pub opening: PC::Opening,
}

impl<F, PC, const N: usize, const D: usize> Clone for AccumulatingProof<F, PC, N, D>
where
    F: TwoAdicField,
    PC: Accumulation<F, Family = Univariate> + Label,
    ProofOf<AccumulatingPlonk<F, PC, N, D>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            opening: self.opening.clone(),
        }
    }
}

/// The rotations the protocol opens at: the current row first, then the
/// next row (the grand product and the lookups read it) and every other
/// rotation a gate reads, in increasing order.
#[must_use]
pub fn rotations(variables: &[Variable]) -> Vec<Rotation> {
    let mut set = BTreeSet::new();
    set.insert(Rotation::NEXT);
    for v in rotated(variables) {
        match v {
            Variable::Witness { rotation, .. } | Variable::Fixed { rotation, .. } => {
                set.insert(rotation);
            }
            Variable::Public(_) => {}
        }
    }
    set.remove(&Rotation::CURR);
    let mut all = Vec::with_capacity(set.len() + 1);
    all.push(Rotation::CURR);
    all.extend(set);
    all
}

/// Where each kind of polynomial starts in the list of [`polynomials`].
struct Layout {
    fixed: usize,
    selectors: usize,
    sigmas: usize,
    z: usize,
    t: usize,
    before: usize,
    after: usize,
    total: usize,
}

impl Layout {
    fn of<F, C, const N: usize, const D: usize>(key: &Preprocessed<F, C, N, D>) -> Self
    where
        F: Field,
    {
        let (before, after) = key.lookups().iter().fold((0, 0), |(b, a), l| {
            let (lb, la) = l.argument().columns();
            (b + lb, a + la)
        });
        let fixed = key.witness_columns();
        let selectors = fixed + key.fixed_commitments().len();
        let sigmas = selectors + key.selector_commitments().len();
        let z = sigmas + key.sigma_commitments().len();
        let t = z + 1;
        Self {
            fixed,
            selectors,
            sigmas,
            z,
            t,
            before: t + 1,
            after: t + 1 + before,
            total: t + 1 + before + after,
        }
    }
}

/// The opened polynomials, in order: the witness columns, the fixed
/// columns, the selectors, the permutation polynomials, `z`, `t`, the
/// lookup columns before `beta, gamma` (all lookups), then after.
fn polynomials<F>(
    pk_fixed: &[UniPoly<F>],
    pk_selectors: &[UniPoly<F>],
    pk_sigmas: &[UniPoly<F>],
    polys: &Polynomials<F>,
) -> Vec<Vec<F>>
where
    F: Field,
{
    let mut all: Vec<Vec<F>> = Vec::new();
    let mut push = |p: &UniPoly<F>| all.push(p.coefficients().to_vec());
    polys.wires.iter().for_each(&mut push);
    pk_fixed.iter().for_each(&mut push);
    pk_selectors.iter().for_each(&mut push);
    pk_sigmas.iter().for_each(&mut push);
    push(&polys.z);
    push(&polys.t);
    polys.before.iter().flatten().for_each(&mut push);
    polys.after.iter().flatten().for_each(&mut push);
    all
}

/// The commitments, in the order of [`polynomials`].
fn commitments<F, C, const N: usize, const D: usize>(
    key: &Preprocessed<F, C, N, D>,
    committed: &Committed<C>,
) -> Vec<C>
where
    F: Field,
    C: Clone,
{
    let mut all = committed.wires.clone();
    all.extend_from_slice(key.fixed_commitments());
    all.extend_from_slice(key.selector_commitments());
    all.extend_from_slice(key.sigma_commitments());
    all.push(committed.z.clone());
    all.push(committed.t.clone());
    all.extend_from_slice(&committed.before);
    all.extend_from_slice(&committed.after);
    all
}

/// The evaluations the identity reads, from the full table: the sparse
/// view [`Opened`] of [`crate::protocol`].
fn sparse<F, C, const N: usize, const D: usize>(
    key: &Preprocessed<F, C, N, D>,
    values: &[Vec<F>],
    rotations: &[Rotation],
) -> Opened<F>
where
    F: Field,
{
    let layout = Layout::of(key);
    // Every rotation read is among `rotations` (they are computed from the
    // same variables); CURR is first.
    let index = |r: Rotation| rotations.iter().position(|x| *x == r).unwrap_or(0);
    let next = index(Rotation::NEXT);
    let at = |row: usize, r: Rotation| values[row][index(r)];
    let range = |start: usize, end: usize| (start..end).map(|row| values[row][0]).collect();
    let rotated_vars = rotated(key.variables());
    let mut lookups = Vec::new();
    let (mut b, mut a) = (layout.before, layout.after);
    for l in key.lookups() {
        let (lb, la) = l.argument().columns();
        for row in (b..b + lb).chain(a..a + la) {
            lookups.push(values[row][0]);
            lookups.push(values[row][next]);
        }
        (b, a) = (b + lb, a + la);
        if l.argument() == Argument::Plookup {
            for k in l.table() {
                lookups.push(values[layout.fixed + *k][next]);
            }
        }
    }
    Opened {
        witness: range(0, layout.fixed),
        fixed: range(layout.fixed, layout.selectors),
        selectors: range(layout.selectors, layout.sigmas),
        sigmas: range(layout.sigmas, layout.z),
        z: values[layout.z][0],
        t: values[layout.t][0],
        z_next: values[layout.z][next],
        rotated: rotated_vars
            .iter()
            .map(|v| match v {
                Variable::Witness { column, rotation } => at(*column, *rotation),
                Variable::Fixed { column, rotation } => at(layout.fixed + *column, *rotation),
                Variable::Public(_) => unreachable!("rotated variables read columns"),
            })
            .collect(),
        lookups,
    }
}

/// The points `zeta w^r` of `rotations`.
fn points<F>(domain: &Domain<F>, zeta: F, rotations: &[Rotation]) -> Vec<F>
where
    F: TwoAdicField,
{
    rotations
        .iter()
        .map(|r| zeta * root_power(domain, *r))
        .collect()
}

/// `{(public, columns) : the columns satisfy the circuit for public}`.
pub struct Satisfiability<'a, F, const N: usize, const D: usize>(pub &'a Circuit<F, N, D>)
where
    F: Field;

impl<F, const N: usize, const D: usize> Clone for Satisfiability<'_, F, N, D>
where
    F: Field,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, const N: usize, const D: usize> Copy for Satisfiability<'_, F, N, D> where F: Field {}

impl<F, const N: usize, const D: usize> Relation for Satisfiability<'_, F, N, D>
where
    F: Field,
{
    type Instance = Vec<F>;
    type Witness = Vec<Vec<F>>;
    type Error = Error;

    fn check(&self, public: &Vec<F>, columns: &Vec<Vec<F>>) -> Result<(), Error> {
        if public.len() != self.0.public_inputs() {
            return Err(Error::PublicInputs);
        }
        if columns.len() != self.0.witness_columns() {
            return Err(Error::Shape);
        }
        if self.0.is_satisfied(columns, public) {
            Ok(())
        } else {
            Err(Error::Unsatisfied)
        }
    }
}

/// `Satisfiability -> Accumulated`: generalized PlonK, its openings left
/// as an accumulator. The prover key is needed by the prover; the verifier
/// reads only its verifier key.
pub struct Accumulating<'a, F, PC, const N: usize, const D: usize>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    pk: &'a ProverKey<F, PC, N, D>,
    pcs: &'a PC,
}

impl<'a, F, PC, const N: usize, const D: usize> Accumulating<'a, F, PC, N, D>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    /// The reduction for the circuit of `pk`, with the scheme `pcs`.
    #[must_use]
    pub const fn new(pk: &'a ProverKey<F, PC, N, D>, pcs: &'a PC) -> Self {
        Self { pk, pcs }
    }
}

impl<F, PC, const N: usize, const D: usize> Clone for Accumulating<'_, F, PC, N, D>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PC, const N: usize, const D: usize> Copy for Accumulating<'_, F, PC, N, D>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
}

impl<'a, F, PC, const N: usize, const D: usize> Reduction<F> for Accumulating<'a, F, PC, N, D>
where
    F: TwoAdicField + PrimeField,
    PC: Accumulation<F, Family = Univariate> + Label,
{
    type Source = Satisfiability<'a, F, N, D>;
    type Target = Accumulated<'a, F, PC>;
    type Proof = AccumulatingProof<F, PC, N, D>;
    type Error = Error;
    type Pattern = Seq<AccumulatingTranscript<F, PC, N, D>, PC::OpenPattern>;

    fn source(&self) -> Self::Source {
        Satisfiability(self.pk.circuit())
    }

    fn target(&self) -> Self::Target {
        Accumulated::new(self.pcs)
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        public: &Vec<F>,
        columns: &Vec<Vec<F>>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Error>
    where
        T: Transcript<F>,
    {
        let pk = self.pk;
        let key = &pk.verifier_key().value;
        let ((claims, polynomials), messages, tape) = prove_within(tape, |prover| {
            let (polys, committed, challenges, prover) =
                prove_rounds(pk, self.pcs, prover, public, columns)?;
            let rotations = rotations(key.variables());
            let points = points(pk.domain(), challenges.zeta, &rotations);
            let polynomials = polynomials(
                pk.fixed_polynomials(),
                pk.selector_polynomials(),
                pk.sigma_polynomials(),
                &polys,
            );
            let values: Vec<Vec<F>> = polynomials
                .iter()
                .map(|p| {
                    points
                        .iter()
                        .map(|x| <Univariate as PolynomialFamily<F>>::evaluate(p, x))
                        .collect()
                })
                .collect();
            let prover = prover.send(Values(values.clone()));
            let claims = Claims {
                commitments: commitments(key, &committed),
                points,
                values,
            };
            Ok(((claims, polynomials), prover))
        })?;
        let (opening, accumulator, tape) = self.pcs.open_claims(tape, &claims, &polynomials);
        Ok(Output {
            instance: accumulator,
            witness: (),
            proof: AccumulatingProof { messages, opening },
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        public: &Vec<F>,
        proof: AccumulatingProof<F, PC, N, D>,
    ) -> Result<(PC::Accumulator, Tape<F, T, R>), Error>
    where
        T: Transcript<F>,
    {
        let vk = self.pk.verifier_key();
        let key = &vk.value;
        if public.len() != key.public_inputs() {
            return Err(Error::PublicInputs);
        }
        let domain = Domain::<F>::new(key.log_size()).ok_or(Error::DomainTooLarge)?;
        let ((committed, challenges, values), tape) =
            verify_within(tape, proof.messages, |verifier| {
                let (committed, challenges, verifier) = verify_rounds(vk, verifier, public)?;
                let (values, verifier) = verifier.read();
                Ok(((committed, challenges, values), verifier))
            })?;
        let rotations = rotations(key.variables());
        let layout = Layout::of(key);
        if values.0.len() != layout.total || values.0.iter().any(|row| row.len() != rotations.len())
        {
            return Err(Error::Shape);
        }
        let opened = sparse(key, &values.0, &rotations);
        let rotated_vars = rotated(key.variables());
        check_identity(key, &domain, public, &opened, &challenges, &rotated_vars)?;
        let claims = Claims {
            commitments: commitments(key, &committed),
            points: points(&domain, challenges.zeta, &rotations),
            values: values.0,
        };
        self.pcs
            .check_claims(tape, &claims, proof.opening)
            .ok_or(Error::Accumulation)
    }
}
