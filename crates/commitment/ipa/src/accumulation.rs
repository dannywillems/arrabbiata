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
    Accumulated, Accumulation, Claims, ClaimsOf, Combine, Error, Evaluations, Succinct,
};
use arrabbiata_reduction::{Batch, Output, OutputOf, Reduction, Then};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_slice,
    hlist::HNil,
    typed::{Challenge, End, Label, ProofOf, Protocol, Public, prover_tape, verifier_tape},
};

use crate::{Ipa, IpaProof, RecursionChallenge, Statement};

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

/// Absorb the combined claim and draw `evalscale`, which combines its
/// points: what the IPA's prover and verifier both do first.
fn evalscale<F, C, P, T>(transcript: &mut T, claim: &arrabbiata_halo::PointClaim<F, C, P>) -> F
where
    F: arrabbiata_field::Field,
    C: Absorb<F>,
    P: Absorb<F>,
    T: Transcript<F>,
{
    claim.absorb_into(transcript);
    transcript.squeeze()
}

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

    fn open_claims<T>(
        &self,
        transcript: T,
        claims: &ClaimsOf<Scalar<C>, Self>,
        polynomials: &[Vec<Scalar<C>>],
    ) -> (IpaProof<C>, RecursionChallenge<C>, T)
    where
        T: Transcript<Scalar<C>>,
    {
        let combine = Combine::new(self);
        let combined_claim = match combine.prove(transcript, claims, &polynomials.to_vec()) {
            Ok(out) => out,
            Err(_) => unreachable!("Succinct checks the shape before opening"),
        };
        let mut transcript = combined_claim.transcript;
        let claim = combined_claim.instance;
        let evalscale = evalscale(&mut transcript, &claim);
        let statement = Statement {
            commitment: &claim.commitment,
            points: &claim.points,
            evalscale,
            value: combined(&claim.values, evalscale),
        };
        let (proof, accumulator) = self.prove(&mut transcript, &statement, &combined_claim.witness);
        (proof, accumulator, transcript)
    }

    fn check_claims<T>(
        &self,
        transcript: T,
        claims: &ClaimsOf<Scalar<C>, Self>,
        opening: IpaProof<C>,
    ) -> Option<(RecursionChallenge<C>, T)>
    where
        T: Transcript<Scalar<C>>,
    {
        let (claim, mut transcript) = Combine::new(self).verify(transcript, claims, HNil).ok()?;
        let evalscale = evalscale(&mut transcript, &claim);
        let statement = Statement {
            commitment: &claim.commitment,
            points: &claim.points,
            evalscale,
            value: combined(&claim.values, evalscale),
        };
        let accumulator = self.succinct(&mut transcript, &statement, &opening)?;
        Some((accumulator, transcript))
    }

    fn fold<T>(
        &self,
        transcript: T,
        accumulators: &[RecursionChallenge<C>],
    ) -> (IpaProof<C>, RecursionChallenge<C>, T)
    where
        T: Transcript<Scalar<C>>,
    {
        let fold = Then(Lift(self), Succinct::new(self));
        let witness = vec![(); accumulators.len()];
        match fold.prove(transcript, &accumulators.to_vec(), &witness) {
            Ok(out) => (out.proof.1, out.instance, out.transcript),
            Err(_) => unreachable!("Lift produces well-shaped claims"),
        }
    }

    fn check_fold<T>(
        &self,
        transcript: T,
        accumulators: &[RecursionChallenge<C>],
        proof: IpaProof<C>,
    ) -> Option<(RecursionChallenge<C>, T)>
    where
        T: Transcript<Scalar<C>>,
    {
        let fold = Then(Lift(self), Succinct::new(self));
        fold.verify(transcript, &accumulators.to_vec(), (HNil, proof))
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

impl<C, T> Protocol for Lift<'_, C, T>
where
    C: Curve,
{
    type Transcript = Public<Accumulators<C>, Challenge<End>>;
}

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
    type Proof = ProofOf<Self>;
    type Error = Error;

    fn source(&self) -> Self::Source {
        Batch(Accumulated::new(self.0))
    }

    fn target(&self) -> Self::Target {
        Evaluations::new(self.0)
    }

    fn prove<Tr>(
        &self,
        transcript: Tr,
        accumulators: &Vec<RecursionChallenge<C>>,
        _witness: &Vec<()>,
    ) -> Result<OutputOf<Scalar<C>, Self, Tr>, Error>
    where
        Tr: Transcript<Scalar<C>>,
    {
        let tape = prover_tape::<Scalar<C>, Tr, Self>(transcript)
            .public(&Accumulators(accumulators.clone()));
        let (z, tape) = tape.challenge();
        let (transcript, proof) = tape.finish();
        Ok(Output {
            instance: lifted(accumulators, z),
            witness: accumulators
                .iter()
                .map(RecursionChallenge::coefficients)
                .collect(),
            proof,
            transcript,
        })
    }

    fn verify<Tr>(
        &self,
        transcript: Tr,
        accumulators: &Vec<RecursionChallenge<C>>,
        proof: Self::Proof,
    ) -> Result<(Claims<Scalar<C>, Projective<C>, Scalar<C>>, Tr), Error>
    where
        Tr: Transcript<Scalar<C>>,
    {
        let tape = verifier_tape::<Scalar<C>, Tr, Self>(transcript, proof)
            .public(&Accumulators(accumulators.clone()));
        let (z, tape) = tape.challenge();
        let transcript = tape.finish();
        Ok((lifted(accumulators, z), transcript))
    }
}
