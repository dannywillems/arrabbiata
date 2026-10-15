//! KZG as an [`Accumulation`]: the `PC_AGM` scheme of Bunz, Chiesa,
//! Mishra and Spooner.
//!
//! - batch: [`Combine`] (`polyscale`), then one KZG opening proof `W_j` per
//!   point;
//! - accumulator: a [`PairingAccumulator`] `(P, W)`, claiming `e(P, H) =
//!   e(W, [tau] H)`; an opening of `C` at `x` to `v` with proof `W` is the
//!   pair `(C - [v] G + [x] W, W)`, and the `m` pairs of a batch are
//!   combined by powers of a challenge `r` drawn after the proofs;
//! - decider: one pairing check ([`Kzg::decide_pair`]);
//! - fold: the same random combination, with no proof. The pairs the
//!   decider accepts are the kernel of `(P, W) -> e(P, H) / e(W, [tau] H)`,
//!   a subgroup of `G1^2`; a combination of pairs one of which is outside
//!   it is outside it except for at most `m - 1` values of `r`.
//!
//! So the expensive check, the pairing, runs once, and the succinct
//! verifier only adds `G1` points.

use alloc::vec::Vec;

use arrabbiata_commit::PolynomialCommitment;
use arrabbiata_curve::{Curve, Pairing, Projective, absorb_point};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_halo::{Accumulation, ClaimsOf, Combine, PointClaimOf};
use arrabbiata_reduction::Reduction;
use arrabbiata_transcript::{Absorb, Transcript, absorb_slice, hlist::HNil, typed::Label};

use crate::Kzg;

/// The scalar field of the pairing groups.
type Scalar<E> = <<E as Pairing>::G1 as Curve>::Scalar;

/// A pair `(P, W)` of `G1` points, claiming `e(P, H) = e(W, [tau] H)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairingAccumulator<E>
where
    E: Pairing,
{
    /// The left point, paired with `H`.
    pub p: Projective<E::G1>,
    /// The right point, paired with `[tau] H`.
    pub w: Projective<E::G1>,
}

impl<E> PairingAccumulator<E>
where
    E: Pairing,
{
    /// `sum_i r^i (P_i, W_i)`.
    fn combination(pairs: &[Self], r: Scalar<E>) -> Self {
        let mut p = Projective::identity();
        let mut w = Projective::identity();
        let mut power = Scalar::<E>::ONE;
        for pair in pairs {
            p = p + pair.p * power;
            w = w + pair.w * power;
            power = power * r;
        }
        Self { p, w }
    }
}

impl<E> Label for PairingAccumulator<E>
where
    E: Pairing,
{
    const LABEL: u64 = 0x7061_6972;
}

impl<E> Absorb<Scalar<E>> for PairingAccumulator<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<Scalar<E>>,
    {
        absorb_point(&self.p, transcript);
        absorb_point(&self.w, transcript);
    }
}

/// The pairs of the openings of a combined claim, then their combination
/// by `r`, drawn after the proofs. Shared by prover and verifier.
fn accumulate<E, T>(
    transcript: &mut T,
    claim: &PointClaimOf<Scalar<E>, Kzg<E>>,
    proofs: &[Projective<E::G1>],
) -> PairingAccumulator<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
    T: Transcript<Scalar<E>>,
{
    absorb_slice(transcript, proofs);
    let r = transcript.squeeze();
    let g = Projective::<E::G1>::generator();
    let pairs: Vec<_> = claim
        .points
        .iter()
        .zip(&claim.values)
        .zip(proofs)
        .map(|((x, v), w)| PairingAccumulator {
            p: claim.commitment - g * *v + *w * *x,
            w: *w,
        })
        .collect();
    PairingAccumulator::combination(&pairs, r)
}

impl<E> Accumulation<Scalar<E>> for Kzg<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
{
    type Accumulator = PairingAccumulator<E>;
    type Opening = Vec<Projective<E::G1>>;
    type FoldProof = ();

    fn open_claims<T>(
        &self,
        transcript: T,
        claims: &ClaimsOf<Scalar<E>, Self>,
        polynomials: &[Vec<Scalar<E>>],
    ) -> (Vec<Projective<E::G1>>, PairingAccumulator<E>, T)
    where
        T: Transcript<Scalar<E>>,
    {
        let combined = match Combine::new(self).prove(transcript, claims, &polynomials.to_vec()) {
            Ok(out) => out,
            Err(_) => unreachable!("Succinct checks the shape before opening"),
        };
        let mut transcript = combined.transcript;
        let proofs: Vec<_> = combined
            .instance
            .points
            .iter()
            .map(|x| PolynomialCommitment::open(self, &combined.witness, x).1)
            .collect();
        let accumulator = accumulate(&mut transcript, &combined.instance, &proofs);
        (proofs, accumulator, transcript)
    }

    fn check_claims<T>(
        &self,
        transcript: T,
        claims: &ClaimsOf<Scalar<E>, Self>,
        opening: Vec<Projective<E::G1>>,
    ) -> Option<(PairingAccumulator<E>, T)>
    where
        T: Transcript<Scalar<E>>,
    {
        let (claim, mut transcript) = Combine::new(self).verify(transcript, claims, HNil).ok()?;
        if opening.len() != claim.points.len() {
            return None;
        }
        let accumulator = accumulate(&mut transcript, &claim, &opening);
        Some((accumulator, transcript))
    }

    fn fold<T>(
        &self,
        mut transcript: T,
        accumulators: &[PairingAccumulator<E>],
    ) -> ((), PairingAccumulator<E>, T)
    where
        T: Transcript<Scalar<E>>,
    {
        let accumulator = fold_pairs(&mut transcript, accumulators);
        ((), accumulator, transcript)
    }

    fn check_fold<T>(
        &self,
        mut transcript: T,
        accumulators: &[PairingAccumulator<E>],
        (): (),
    ) -> Option<(PairingAccumulator<E>, T)>
    where
        T: Transcript<Scalar<E>>,
    {
        let accumulator = fold_pairs(&mut transcript, accumulators);
        Some((accumulator, transcript))
    }

    fn decide(&self, accumulator: &PairingAccumulator<E>) -> bool {
        self.decide_pair(&accumulator.p, &accumulator.w)
    }
}

/// The combination of `accumulators` by powers of `r`, drawn after them.
fn fold_pairs<E, T>(
    transcript: &mut T,
    accumulators: &[PairingAccumulator<E>],
) -> PairingAccumulator<E>
where
    E: Pairing,
    <E::G1 as Curve>::Base: PrimeField,
    T: Transcript<Scalar<E>>,
{
    absorb_slice(transcript, accumulators);
    let r = transcript.squeeze();
    PairingAccumulator::combination(accumulators, r)
}
