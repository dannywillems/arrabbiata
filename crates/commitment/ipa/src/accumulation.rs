//! The IPA as an [`Accumulation`]: Halo's `PC_DL` scheme.
//!
//! - batch: [`Combine`] (`polyscale`), then one IPA at all the points,
//!   combined by `evalscale` drawn after the combined claim (Kimchi's
//!   `combined_inner_product`);
//! - accumulator: the [`RecursionChallenge`] `(u, sg)`;
//! - decider: `sg = <s, G>`, an MSM of the size of the key;
//! - fold: [`Lift`] to claims at a fresh point, then [`Succinct`]: one more
//!   IPA, no MSM. The fold is literally that composite, run as reductions.

use alloc::{vec, vec::Vec};

use arrabbiata_curve::{Curve, Projective};
use arrabbiata_field::PrimeField;
use arrabbiata_halo::{
    Accumulated, Accumulation, Claims, ClaimsOf, Combine, CombinePattern, Error, Evaluations,
    PointClaimOf, Succinct,
};
use arrabbiata_reduction::{Batch, Output, OutputOf, Reduction, Then};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_slice,
    hlist::HNil,
    typed::{Challenge, End, Label, Public, Seq, Tail, Tape},
};

use crate::{Ipa, IpaProof, OpeningSteps, RecursionChallenge, Statement};

/// The scalar field of the IPA over `C`.
type Scalar<C> = <C as Curve>::Scalar;

/// The accumulators, as one public value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulators<C>(pub Vec<RecursionChallenge<C>>)
where
    C: Curve;

impl<C> Label for Accumulators<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x6163_6373;
}

impl<C> Absorb<C::Scalar> for Accumulators<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<C::Scalar>,
    {
        absorb_slice(transcript, &self.0);
    }
}

/// The step after [`Combine`]: the combined claim is public, then
/// `evalscale`, which combines its points, is drawn.
pub type Evalscale<C, T> = Public<PointClaimOf<Scalar<C>, Ipa<C, T>>, Challenge<End>>;

/// The IPA's opening of a batch: [`Combine`], [`Evalscale`], then the
/// opening's steps at all the points.
pub type OpenPattern<C, T> =
    Seq<CombinePattern<Scalar<C>, Ipa<C, T>>, Seq<Evalscale<C, T>, OpeningSteps<C, End>>>;

/// [`Lift`]'s steps: the accumulators are public, then the point is drawn.
pub type LiftPattern<C> = Public<Accumulators<C>, Challenge<End>>;

/// `sum_j evalscale^j v_j`.
fn combined<F>(values: &[F], evalscale: F) -> F
where
    F: arrabbiata_field::Field,
{
    let mut value = F::ZERO;
    let mut j = values.len();
    while j > 0 {
        j -= 1;
        value = value * evalscale + values[j];
    }
    value
}

impl<C, S> Accumulation<Scalar<C>> for Ipa<C, S>
where
    C: Curve,
    C::Base: PrimeField,
    S: Transcript<Scalar<C>> + Clone,
{
    type Accumulator = RecursionChallenge<C>;
    type Opening = IpaProof<C>;
    type FoldProof = IpaProof<C>;
    type OpenPattern = OpenPattern<C, S>;
    type FoldPattern = Seq<LiftPattern<C>, OpenPattern<C, S>>;

    fn open_claims<T, R>(
        &self,
        tape: Tape<Scalar<C>, T, Tail<Self::OpenPattern, R>>,
        claims: &ClaimsOf<Scalar<C>, Self>,
        polynomials: &[Vec<Scalar<C>>],
    ) -> (IpaProof<C>, RecursionChallenge<C>, Tape<Scalar<C>, T, R>)
    where
        T: Transcript<Scalar<C>>,
    {
        let combine = Combine::new(self);
        let combined_claim = match combine.prove(tape, claims, &polynomials.to_vec()) {
            Ok(out) => out,
            Err(_) => unreachable!("Succinct checks the shape before opening"),
        };
        let claim = combined_claim.instance;
        let (evalscale, tape) = combined_claim.transcript.public(&claim).challenge();
        let statement = Statement {
            commitment: &claim.commitment,
            points: &claim.points,
            evalscale,
            value: combined(&claim.values, evalscale),
        };
        self.prove_tape(tape, &statement, &combined_claim.witness)
    }

    fn check_claims<T, R>(
        &self,
        tape: Tape<Scalar<C>, T, Tail<Self::OpenPattern, R>>,
        claims: &ClaimsOf<Scalar<C>, Self>,
        opening: IpaProof<C>,
    ) -> Option<(RecursionChallenge<C>, Tape<Scalar<C>, T, R>)>
    where
        T: Transcript<Scalar<C>>,
    {
        let (claim, tape) = Combine::new(self).verify(tape, claims, HNil).ok()?;
        let (evalscale, tape) = tape.public(&claim).challenge();
        let statement = Statement {
            commitment: &claim.commitment,
            points: &claim.points,
            evalscale,
            value: combined(&claim.values, evalscale),
        };
        self.succinct_tape(tape, &statement, &opening)
    }

    fn fold<T, R>(
        &self,
        tape: Tape<Scalar<C>, T, Tail<Self::FoldPattern, R>>,
        accumulators: &[RecursionChallenge<C>],
    ) -> (IpaProof<C>, RecursionChallenge<C>, Tape<Scalar<C>, T, R>)
    where
        T: Transcript<Scalar<C>>,
    {
        let fold = Then(Lift(self), Succinct::new(self));
        let witness = vec![(); accumulators.len()];
        match fold.prove(tape, &accumulators.to_vec(), &witness) {
            Ok(out) => (out.proof.1, out.instance, out.transcript),
            Err(_) => unreachable!("Lift produces well-shaped claims"),
        }
    }

    fn check_fold<T, R>(
        &self,
        tape: Tape<Scalar<C>, T, Tail<Self::FoldPattern, R>>,
        accumulators: &[RecursionChallenge<C>],
        proof: IpaProof<C>,
    ) -> Option<(RecursionChallenge<C>, Tape<Scalar<C>, T, R>)>
    where
        T: Transcript<Scalar<C>>,
    {
        let fold = Then(Lift(self), Succinct::new(self));
        fold.verify(tape, &accumulators.to_vec(), (HNil, proof))
            .ok()
    }

    fn decide(&self, accumulator: &RecursionChallenge<C>) -> bool {
        Ipa::decide(self, accumulator)
    }
}

/// `Accumulated^* -> Evaluations`, for the IPA: each accumulator as the
/// claim that `sg` opens to `b(z)` at a point `z` drawn after them. The
/// verifier computes each `b(z)` in `O(k)`; the witness is the
/// coefficients of each `b`. Sound because a polynomial opening `sg_i` to
/// `b_i(z)` is `b_i` except with probability `n / |F|`.
pub struct Lift<'a, C, T>(pub &'a Ipa<C, T>)
where
    C: Curve;

impl<C, T> Clone for Lift<'_, C, T>
where
    C: Curve,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, T> Copy for Lift<'_, C, T> where C: Curve {}

/// The claims of accumulators at `z`.
fn lifted<C>(
    accumulators: &[RecursionChallenge<C>],
    z: Scalar<C>,
) -> Claims<Scalar<C>, Projective<C>, Scalar<C>>
where
    C: Curve,
{
    Claims {
        commitments: accumulators.iter().map(|a| a.comm).collect(),
        points: vec![z],
        values: accumulators.iter().map(|a| vec![a.evaluate(z)]).collect(),
    }
}

impl<'a, C, T> Reduction<Scalar<C>> for Lift<'a, C, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<Scalar<C>> + Clone,
{
    type Source = Batch<Accumulated<'a, Scalar<C>, Ipa<C, T>>>;
    type Target = Evaluations<'a, Scalar<C>, Ipa<C, T>>;
    type Proof = HNil;
    type Error = Error;
    type Pattern = LiftPattern<C>;

    fn source(&self) -> Self::Source {
        Batch(Accumulated::new(self.0))
    }

    fn target(&self) -> Self::Target {
        Evaluations::new(self.0)
    }

    fn prove<Tr, R>(
        &self,
        tape: Tape<Scalar<C>, Tr, Tail<Self::Pattern, R>>,
        accumulators: &Vec<RecursionChallenge<C>>,
        _witness: &Vec<()>,
    ) -> Result<OutputOf<Scalar<C>, Self, Tape<Scalar<C>, Tr, R>>, Error>
    where
        Tr: Transcript<Scalar<C>>,
    {
        let (z, tape) = tape.public(&Accumulators(accumulators.clone())).challenge();
        Ok(Output {
            instance: lifted(accumulators, z),
            witness: accumulators
                .iter()
                .map(RecursionChallenge::coefficients)
                .collect(),
            proof: HNil,
            transcript: tape,
        })
    }

    fn verify<Tr, R>(
        &self,
        tape: Tape<Scalar<C>, Tr, Tail<Self::Pattern, R>>,
        accumulators: &Vec<RecursionChallenge<C>>,
        HNil: HNil,
    ) -> Result<
        (
            Claims<Scalar<C>, Projective<C>, Scalar<C>>,
            Tape<Scalar<C>, Tr, R>,
        ),
        Error,
    >
    where
        Tr: Transcript<Scalar<C>>,
    {
        let (z, tape) = tape.public(&Accumulators(accumulators.clone())).challenge();
        Ok((lifted(accumulators, z), tape))
    }
}
