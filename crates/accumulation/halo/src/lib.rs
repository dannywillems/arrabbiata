//! Halo's accumulation of polynomial openings, for any commitment scheme
//! with an [`Accumulation`], as a composite of reductions of knowledge.
//!
//! A verifier of evaluation proofs is two morphisms in sequence. The
//! succinct one does everything cheap and outputs a claim, the
//! accumulator; the decider checks the claim, and is the expensive part.
//! Halo (Bowe, Grigg, Hopwood, ePrint 2019/1021, section 3) never decides
//! until the end: accumulators fold into one, and only the last is
//! decided. Bunz, Chiesa, Mishra and Spooner (ePrint 2020/499, TCC 2020)
//! name this an accumulation scheme for the commitment, and give one for
//! `PC_DL` (the IPA: the accumulator is `(u, sg)`, the decider an MSM of
//! the size of the key) and one for `PC_AGM` (KZG: the accumulator is two
//! `G1` points, the decider one pairing check). Both implement the trait in
//! their own crates (`arrabbiata-ipa`, `arrabbiata-kzg`): this crate is the
//! abstraction only, and depends on no scheme.
//!
//! # What a scheme needs
//!
//! [`Accumulation`] asks for four things and nothing else: no linearity,
//! no particular family of polynomials, no particular oracle.
//!
//! 1. A **succinct verifier** of a batch of evaluation claims (`k`
//!    polynomials at `m` points) that outputs an accumulator, at a cost
//!    independent of the degree.
//! 2. A **decider** for the accumulator relation `R_acc`, run once.
//! 3. A **fold** `R_acc^* -> R_acc` with a succinct verifier.
//! 4. **Soundness** of the three: a false claim or a false accumulator
//!    leads to a false accumulator, except with negligible probability.
//!
//! How a scheme meets them is its own business, and the two here show two
//! different mechanisms, which is what to look for in another scheme (a
//! lattice one, say):
//!
//! - **Linear commitments** ([`LinearCommitment`]) batch `k` polynomials
//!   into one by powers of `polyscale` ([`Combine`]): the batch morphism
//!   factors through [`Opened`]. Lattice commitments are homomorphic only
//!   for small-norm combinations, and hash-based ones not at all, so this
//!   is a property of some schemes, not a requirement.
//! - **A linear accumulator relation** (KZG): `R_acc = {(P, W) : e(P, H) =
//!   e(W, [tau] H)}` is the kernel of a group homomorphism `G1^2 -> GT`, a
//!   subgroup, so accumulators fold by a random linear combination, with
//!   no proof: if one of `m` is outside the kernel, the combination is
//!   outside except for at most `m - 1` challenges.
//! - **A succinctly evaluable accumulator** (IPA): `R_acc = {(u, sg) : sg
//!   = commit(b_u)}` is not closed under combinations, but `b_u(z)` costs
//!   `O(k)`, so an accumulator lifts to an evaluation claim at a fresh
//!   point, and the fold is `Lift ; Succinct`, one more opening.
//!
//! # The category
//!
//! | relation | instance | witness | in it when |
//! | --- | --- | --- | --- |
//! | [`Evaluations`] | [`Claims`]: `C_i`, points `x_j`, values `v_ij` | the `p_i` | `C_i = commit(p_i)`, `v_ij = p_i(x_j)` |
//! | [`Opened`] | [`PointClaim`]: `C`, the `x_j`, the `v_j` | `p` | `C = commit(p)`, `v_j = p(x_j)` |
//! | [`Accumulated`] | the scheme's accumulator | none | the decider accepts |
//!
//! | morphism | from | to |
//! | --- | --- | --- |
//! | [`Succinct`] | `Evaluations` | `Accumulated` |
//! | [`Fold`] | `Accumulated^*` | `Accumulated` |
//! | [`Decide`] | `Accumulated` | `Top` |
//! | [`Combine`] (linear schemes) | `Evaluations` | `Opened` |
//!
//! and the rest is composition, the same for every scheme:
//!
//! - [`Open`] `= Succinct ; Decide : Evaluations -> Top`, a batched
//!   opening;
//! - [`Accumulate`] `= Each(Succinct) ; Fold : Evaluations^* ->
//!   Accumulated`;
//! - [`DecideAll`] `= Fold ; Decide : Accumulated^* -> Top`;
//! - `Then(Accumulate, Decide)`, an argument for many batches of openings
//!   with one decider.
//!
//! Every morphism runs on any [`Transcript`]: the oracle is a parameter,
//! never a concrete sponge. The laws are tested by generic functions in
//! `tests/halo.rs`, for both schemes and on two oracles.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::{borrow::Borrow, fmt, fmt::Debug, marker::PhantomData};

use arrabbiata_commit::{
    LinearCommitment, PolynomialCommitment, PolynomialFamily, combine_polynomials,
};
use arrabbiata_field::Field;
use arrabbiata_reduction::{Batch, Each, Output, OutputOf, Reduction, Relation, Then, Top};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_slice,
    hlist::HNil,
    typed::{Challenge, End, Label, Public, Sequence, Tail, Tape},
};

/// Why a pair is not in a relation, or a proof was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Lists of the wrong lengths (polynomials, points, values, or rounds).
    Shape,
    /// Commitment `i` is not the commitment to polynomial `i`.
    Commitment(usize),
    /// Value `(i, j)` is not the value of polynomial `i` at point `j`.
    Value(usize, usize),
    /// The succinct check of an opening failed.
    Succinct,
    /// The succinct check of a fold failed.
    Fold,
    /// The decider rejected an accumulator.
    Decide,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => write!(f, "lists of the wrong lengths"),
            Self::Commitment(i) => write!(f, "commitment {i} does not match its polynomial"),
            Self::Value(i, j) => write!(f, "value of polynomial {i} at point {j} is wrong"),
            Self::Succinct => write!(f, "the succinct check of an opening failed"),
            Self::Fold => write!(f, "the succinct check of a fold failed"),
            Self::Decide => write!(f, "the decider rejected an accumulator"),
        }
    }
}

impl core::error::Error for Error {}

/// The owned point type of `PC`'s family.
pub type PointOf<F, PC> =
    <<PC as PolynomialCommitment<F>>::Family as PolynomialFamily<F>>::OwnedPoint;

/// Evaluation claims: commitments to `k` polynomials, `m` points, and the
/// `k x m` values, `values[i][j]` of polynomial `i` at point `j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claims<F, C, P> {
    /// The commitments, one per polynomial.
    pub commitments: Vec<C>,
    /// The points.
    pub points: Vec<P>,
    /// The values, one row per polynomial, one column per point.
    pub values: Vec<Vec<F>>,
}

/// The claims of the scheme `PC`.
pub type ClaimsOf<F, PC> = Claims<F, <PC as PolynomialCommitment<F>>::Commitment, PointOf<F, PC>>;

impl<F, C, P> Claims<F, C, P> {
    /// Whether there is one row of values per commitment, each with one
    /// value per point.
    #[must_use]
    pub fn has_shape(&self) -> bool {
        self.values.len() == self.commitments.len()
            && self.values.iter().all(|row| row.len() == self.points.len())
    }
}

impl<F, C, P> Label for Claims<F, C, P> {
    const LABEL: u64 = 0x6576_616c;
}

impl<F, C, P> Absorb<F> for Claims<F, C, P>
where
    F: Field,
    C: Absorb<F>,
    P: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.commitments);
        absorb_slice(transcript, &self.points);
        for row in &self.values {
            absorb_slice(transcript, row);
        }
    }
}

/// One polynomial at several points: its commitment, the points, and the
/// claimed value at each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointClaim<F, C, P> {
    /// The commitment.
    pub commitment: C,
    /// The points.
    pub points: Vec<P>,
    /// The value at each point.
    pub values: Vec<F>,
}

/// The single-polynomial claim of the scheme `PC`.
pub type PointClaimOf<F, PC> =
    PointClaim<F, <PC as PolynomialCommitment<F>>::Commitment, PointOf<F, PC>>;

impl<F, C, P> Label for PointClaim<F, C, P> {
    const LABEL: u64 = 0x7074_636c;
}

impl<F, C, P> Absorb<F> for PointClaim<F, C, P>
where
    F: Field,
    C: Absorb<F>,
    P: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        self.commitment.absorb_into(transcript);
        absorb_slice(transcript, &self.points);
        absorb_slice(transcript, &self.values);
    }
}

/// A polynomial commitment whose verifier of batched evaluation claims is
/// a succinct reduction to an accumulator relation followed by a decider,
/// and whose accumulators fold.
///
/// Laws, tested for every implementation by the generic functions of
/// `tests/halo.rs`:
///
/// - completeness: `check_claims(open_claims(claims, ps))` is the prover's
///   accumulator, on the same transcript, and it is decided true when the
///   `ps` satisfy the claims;
/// - soundness: a false claim makes `check_claims` fail or its accumulator
///   false, except with negligible probability;
/// - fold: `check_fold(fold(accs))` is the prover's accumulator, decided
///   true when every input is, and false when one input is, except with
///   negligible probability;
/// - succinctness (a cost, not testable): `check_claims` and `check_fold`
///   cost a polynomial in `log n` and in the sizes of the batch, never `n`.
///
/// The methods run on a tape at their pattern followed by any `R` and
/// hand it back at `R`, as reductions do, so that an implementation can
/// itself be a composite of reductions and its Fiat-Shamir steps are in its
/// type ([`Accumulation::OpenPattern`], [`Accumulation::FoldPattern`]).
pub trait Accumulation<F>: PolynomialCommitment<F>
where
    F: Field,
{
    /// The instance of the accumulator relation.
    type Accumulator: Clone + Debug + PartialEq + Absorb<F> + Label;

    /// What the prover sends for one batch of claims.
    type Opening: Clone + Debug + PartialEq;

    /// What the prover sends for one fold.
    type FoldProof: Clone + Debug + PartialEq;

    /// The Fiat-Shamir steps of opening one batch of claims.
    type OpenPattern: Sequence;

    /// The Fiat-Shamir steps of one fold.
    type FoldPattern: Sequence;

    /// Open `polynomials` against `claims`: the message and the
    /// accumulator, which [`Accumulation::check_claims`] outputs too.
    fn open_claims<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::OpenPattern, R>>,
        claims: &ClaimsOf<F, Self>,
        polynomials: &[Vec<F>],
    ) -> (Self::Opening, Self::Accumulator, Tape<F, T, R>)
    where
        T: Transcript<F>;

    /// The succinct verifier of a batch: its accumulator, or `None`.
    fn check_claims<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::OpenPattern, R>>,
        claims: &ClaimsOf<F, Self>,
        opening: Self::Opening,
    ) -> Option<(Self::Accumulator, Tape<F, T, R>)>
    where
        T: Transcript<F>;

    /// Fold `accumulators` into one.
    fn fold<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::FoldPattern, R>>,
        accumulators: &[Self::Accumulator],
    ) -> (Self::FoldProof, Self::Accumulator, Tape<F, T, R>)
    where
        T: Transcript<F>;

    /// The succinct verifier of a fold: its accumulator, or `None`.
    fn check_fold<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::FoldPattern, R>>,
        accumulators: &[Self::Accumulator],
        proof: Self::FoldProof,
    ) -> Option<(Self::Accumulator, Tape<F, T, R>)>
    where
        T: Transcript<F>;

    /// The decider: whether the accumulator is in the relation.
    fn decide(&self, accumulator: &Self::Accumulator) -> bool;
}

/// The type parameters of a relation or reduction over a scheme, as a
/// marker.
type Over<F> = PhantomData<fn() -> F>;

/// A relation or a reduction over a scheme `PC`, by reference: what every
/// object and morphism of this crate is.
macro_rules! over_scheme {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        pub struct $name<'a, F, PC> {
            pcs: &'a PC,
            field: Over<F>,
        }

        impl<'a, F, PC> $name<'a, F, PC> {
            /// Over `pcs`.
            #[must_use]
            pub const fn new(pcs: &'a PC) -> Self {
                Self { pcs, field: PhantomData }
            }
        }

        impl<F, PC> Clone for $name<'_, F, PC> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<F, PC> Copy for $name<'_, F, PC> {}
    };
}

over_scheme!(
    /// `{(Claims, polynomials) : C_i = commit(p_i), v_ij = p_i(x_j)}`.
    Evaluations
);
over_scheme!(
    /// `{(PointClaim, p) : C = commit(p), v_j = p(x_j)}`.
    Opened
);
over_scheme!(
    /// `{(accumulator, ()) : decide(accumulator)}`, the scheme's
    /// accumulator relation.
    Accumulated
);
over_scheme!(
    /// `Evaluations -> Opened`, for a linear scheme: the polynomials
    /// combined by powers of `polyscale`, drawn after the claims.
    ///
    /// Knowledge soundness: a witness for the output is `sum_i polyscale^i
    /// p_i` (linearity and binding); if some `v_ij` is wrong, the combined
    /// value at `x_j` is not the combined polynomial's except for at most
    /// `k - 1` values of `polyscale`.
    Combine
);
over_scheme!(
    /// `Evaluations -> Accumulated`: the scheme's opening of a batch and
    /// its succinct check.
    Succinct
);
over_scheme!(
    /// `Accumulated^* -> Accumulated`: the scheme's fold.
    Fold
);
over_scheme!(
    /// `Accumulated -> Top`: the scheme's decider.
    Decide
);

/// The value of `polynomial` at `point`, in the scheme's family.
fn evaluate<F, PC>(polynomial: &[F], point: &PointOf<F, PC>) -> F
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    <PC::Family as PolynomialFamily<F>>::evaluate(polynomial, point.borrow())
}

impl<F, PC> Relation for Evaluations<'_, F, PC>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    type Instance = ClaimsOf<F, PC>;
    type Witness = Vec<Vec<F>>;
    type Error = Error;

    fn check(&self, claims: &Self::Instance, polynomials: &Vec<Vec<F>>) -> Result<(), Error> {
        if !claims.has_shape() || polynomials.len() != claims.commitments.len() {
            return Err(Error::Shape);
        }
        for (i, p) in polynomials.iter().enumerate() {
            if self.pcs.commit(p) != claims.commitments[i] {
                return Err(Error::Commitment(i));
            }
            for (j, x) in claims.points.iter().enumerate() {
                if evaluate::<F, PC>(p, x) != claims.values[i][j] {
                    return Err(Error::Value(i, j));
                }
            }
        }
        Ok(())
    }
}

impl<F, PC> Relation for Opened<'_, F, PC>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    type Instance = PointClaimOf<F, PC>;
    type Witness = Vec<F>;
    type Error = Error;

    fn check(&self, claim: &Self::Instance, polynomial: &Vec<F>) -> Result<(), Error> {
        if claim.points.len() != claim.values.len() {
            return Err(Error::Shape);
        }
        if self.pcs.commit(polynomial) != claim.commitment {
            return Err(Error::Commitment(0));
        }
        for (j, x) in claim.points.iter().enumerate() {
            if evaluate::<F, PC>(polynomial, x) != claim.values[j] {
                return Err(Error::Value(0, j));
            }
        }
        Ok(())
    }
}

impl<F, PC> Relation for Accumulated<'_, F, PC>
where
    F: Field,
    PC: Accumulation<F>,
{
    type Instance = PC::Accumulator;
    type Witness = ();
    type Error = Error;

    fn check(&self, accumulator: &PC::Accumulator, (): &()) -> Result<(), Error> {
        if self.pcs.decide(accumulator) {
            Ok(())
        } else {
            Err(Error::Decide)
        }
    }
}

/// [`Combine`]'s Fiat-Shamir steps: the claims are public, then
/// `polyscale` is drawn.
pub type CombinePattern<F, PC> = Public<ClaimsOf<F, PC>, Challenge<End>>;

impl<F, PC> Combine<'_, F, PC>
where
    F: Field,
    PC: LinearCommitment<F>,
{
    /// The claim on the combined polynomial. Shared by prover and verifier.
    fn claim(&self, claims: &ClaimsOf<F, PC>, polyscale: F) -> PointClaimOf<F, PC> {
        let mut terms = Vec::with_capacity(claims.commitments.len());
        let mut power = F::ONE;
        for c in &claims.commitments {
            terms.push((power, c));
            power = power * polyscale;
        }
        let mut values = Vec::with_capacity(claims.points.len());
        for j in 0..claims.points.len() {
            let mut value = F::ZERO;
            let mut power = F::ONE;
            for row in &claims.values {
                value = value + power * row[j];
                power = power * polyscale;
            }
            values.push(value);
        }
        PointClaim {
            commitment: self.pcs.combine(&terms),
            points: claims.points.clone(),
            values,
        }
    }
}

impl<'a, F, PC> Reduction<F> for Combine<'a, F, PC>
where
    F: Field,
    PC: LinearCommitment<F>,
{
    type Source = Evaluations<'a, F, PC>;
    type Target = Opened<'a, F, PC>;
    type Proof = HNil;
    type Error = Error;
    type Pattern = CombinePattern<F, PC>;

    fn source(&self) -> Self::Source {
        Evaluations::new(self.pcs)
    }

    fn target(&self) -> Self::Target {
        Opened::new(self.pcs)
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        claims: &ClaimsOf<F, PC>,
        polynomials: &Vec<Vec<F>>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Error>
    where
        T: Transcript<F>,
    {
        if !claims.has_shape() || polynomials.len() != claims.commitments.len() {
            return Err(Error::Shape);
        }
        let (polyscale, tape) = tape.public(claims).challenge();
        let slices: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
        Ok(Output {
            instance: self.claim(claims, polyscale),
            witness: combine_polynomials(&slices, polyscale),
            proof: HNil,
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        claims: &ClaimsOf<F, PC>,
        HNil: HNil,
    ) -> Result<(PointClaimOf<F, PC>, Tape<F, T, R>), Error>
    where
        T: Transcript<F>,
    {
        if !claims.has_shape() {
            return Err(Error::Shape);
        }
        let (polyscale, tape) = tape.public(claims).challenge();
        Ok((self.claim(claims, polyscale), tape))
    }
}

impl<'a, F, PC> Reduction<F> for Succinct<'a, F, PC>
where
    F: Field,
    PC: Accumulation<F>,
{
    type Source = Evaluations<'a, F, PC>;
    type Target = Accumulated<'a, F, PC>;
    type Proof = PC::Opening;
    type Error = Error;
    type Pattern = PC::OpenPattern;

    fn source(&self) -> Self::Source {
        Evaluations::new(self.pcs)
    }

    fn target(&self) -> Self::Target {
        Accumulated::new(self.pcs)
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        claims: &ClaimsOf<F, PC>,
        polynomials: &Vec<Vec<F>>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Error>
    where
        T: Transcript<F>,
    {
        if !claims.has_shape() || polynomials.len() != claims.commitments.len() {
            return Err(Error::Shape);
        }
        let (proof, accumulator, tape) = self.pcs.open_claims(tape, claims, polynomials);
        Ok(Output {
            instance: accumulator,
            witness: (),
            proof,
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        claims: &ClaimsOf<F, PC>,
        proof: PC::Opening,
    ) -> Result<(PC::Accumulator, Tape<F, T, R>), Error>
    where
        T: Transcript<F>,
    {
        if !claims.has_shape() {
            return Err(Error::Shape);
        }
        self.pcs
            .check_claims(tape, claims, proof)
            .ok_or(Error::Succinct)
    }
}

impl<'a, F, PC> Reduction<F> for Fold<'a, F, PC>
where
    F: Field,
    PC: Accumulation<F>,
{
    type Source = Batch<Accumulated<'a, F, PC>>;
    type Target = Accumulated<'a, F, PC>;
    type Proof = PC::FoldProof;
    type Error = Error;
    type Pattern = PC::FoldPattern;

    fn source(&self) -> Self::Source {
        Batch(Accumulated::new(self.pcs))
    }

    fn target(&self) -> Self::Target {
        Accumulated::new(self.pcs)
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        accumulators: &Vec<PC::Accumulator>,
        _witness: &Vec<()>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Error>
    where
        T: Transcript<F>,
    {
        let (proof, accumulator, tape) = self.pcs.fold(tape, accumulators);
        Ok(Output {
            instance: accumulator,
            witness: (),
            proof,
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        accumulators: &Vec<PC::Accumulator>,
        proof: PC::FoldProof,
    ) -> Result<(PC::Accumulator, Tape<F, T, R>), Error>
    where
        T: Transcript<F>,
    {
        self.pcs
            .check_fold(tape, accumulators, proof)
            .ok_or(Error::Fold)
    }
}

impl<'a, F, PC> Reduction<F> for Decide<'a, F, PC>
where
    F: Field,
    PC: Accumulation<F>,
{
    type Source = Accumulated<'a, F, PC>;
    type Target = Top;
    type Proof = ();
    type Error = Error;
    type Pattern = End;

    fn source(&self) -> Self::Source {
        Accumulated::new(self.pcs)
    }

    fn target(&self) -> Top {
        Top
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, R>,
        _accumulator: &PC::Accumulator,
        (): &(),
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Error>
    where
        T: Transcript<F>,
    {
        Ok(Output {
            instance: (),
            witness: (),
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, R>,
        accumulator: &PC::Accumulator,
        (): (),
    ) -> Result<((), Tape<F, T, R>), Error>
    where
        T: Transcript<F>,
    {
        if self.pcs.decide(accumulator) {
            Ok(((), tape))
        } else {
            Err(Error::Decide)
        }
    }
}

/// `Succinct ; Decide : Evaluations -> Top`: a batched opening.
pub type Open<'a, F, PC> = Then<Succinct<'a, F, PC>, Decide<'a, F, PC>>;

/// `Each(Succinct) ; Fold : Evaluations^* -> Accumulated`: many batches
/// into one accumulator.
pub type Accumulate<'a, F, PC> = Then<Each<Succinct<'a, F, PC>>, Fold<'a, F, PC>>;

/// `Fold ; Decide : Accumulated^* -> Top`: many accumulators, one decider.
pub type DecideAll<'a, F, PC> = Then<Fold<'a, F, PC>, Decide<'a, F, PC>>;

/// [`Open`] over `pcs`.
#[must_use]
pub const fn open<F, PC>(pcs: &PC) -> Open<'_, F, PC> {
    Then(Succinct::new(pcs), Decide::new(pcs))
}

/// [`Accumulate`] over `pcs`.
#[must_use]
pub const fn accumulate<F, PC>(pcs: &PC) -> Accumulate<'_, F, PC> {
    Then(Each(Succinct::new(pcs)), Fold::new(pcs))
}

/// [`DecideAll`] over `pcs`.
#[must_use]
pub const fn decide_all<F, PC>(pcs: &PC) -> DecideAll<'_, F, PC> {
    Then(Fold::new(pcs), Decide::new(pcs))
}
