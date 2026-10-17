//! Kimchi's opening as an instance of `arrabbiata-ipa`'s generic IPA:
//! Kimchi's choice for each parameter.
//!
//! - the oracle: Kimchi's base-field sponge ([`FqSponge`]), which absorbs
//!   scalars embedded in the base field and points by their coordinates,
//!   squeezes 128-bit challenges, and squeezes base-field elements;
//! - the binding: the shifted combined inner product only
//!   ([`BindValue`]), the protocol having absorbed the commitments and
//!   drawn the points and the scales itself;
//! - the challenges: endomorphism images of 128-bit challenges ([`Endo`]
//!   with the curve's `lambda`);
//! - the value base: Kimchi's group map of a squeezed base-field element
//!   ([`HashedToCurve`] over [`GroupMap`]);
//! - the hiding: blinded rounds and `(delta, z1, z2)` ([`Blinded`]).
//!
//! The prover's opening round and the verifier's check are this instance;
//! the tests against Kimchi's proofs are its tests.

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{PrimeField, Sqrt};
use arrabbiata_ipa::generic::{
    BaseOracle, BindValue, Blinded, Endo, GenericIpa, HashToCurve, HashedToCurve, IpaOracle,
    Opening, Schnorr,
};
use arrabbiata_transcript::Transcript;

use crate::{
    oracle::FqSponge,
    prover::OpeningProof,
    srs::{GroupMap, Srs},
};

impl<C, S> IpaOracle<C> for FqSponge<C, S>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn absorb_scalar(&mut self, x: C::Scalar) {
        self.absorb_fr(&[x]);
    }

    fn absorb_point(&mut self, p: &Projective<C>) {
        self.absorb_g(&[*p]);
    }

    fn squeeze_scalar(&mut self) -> C::Scalar {
        self.challenge()
    }
}

impl<C, S> BaseOracle<C> for FqSponge<C, S>
where
    C: Glv,
    C::Base: PrimeField,
    S: Transcript<C::Base>,
{
    fn squeeze_base(&mut self) -> C::Base {
        self.challenge_fq()
    }
}

impl<C> HashToCurve<C> for GroupMap<C>
where
    C: Glv,
    C::Base: Sqrt,
{
    fn to_group(&self, t: C::Base) -> Projective<C> {
        GroupMap::to_group(self, t).to_projective()
    }
}

/// Kimchi's IPA.
pub type KimchiIpa<C> = GenericIpa<
    C,
    BindValue,
    Endo<<C as arrabbiata_curve::Curve>::Scalar>,
    HashedToCurve<GroupMap<C>>,
    Blinded,
>;

/// Kimchi's IPA on `srs`: its generators, its blinding point `h`, and the
/// curve's group map and endomorphism.
#[must_use]
pub fn kimchi_ipa<C>(srs: &Srs<C>) -> KimchiIpa<C>
where
    C: Glv,
    C::Base: Sqrt,
{
    kimchi_ipa_with(srs, GroupMap::setup())
}

/// [`kimchi_ipa`] with a group map already set up.
#[must_use]
pub fn kimchi_ipa_with<C>(srs: &Srs<C>, group_map: GroupMap<C>) -> KimchiIpa<C>
where
    C: Glv,
    C::Base: Sqrt,
{
    GenericIpa {
        generators: srs.g.clone(),
        // Unused by HashedToCurve: the value base is hashed.
        u: srs.h,
        h: srs.h,
        binding: BindValue,
        challenges: Endo {
            lambda: C::lambda(),
        },
        value_base: HashedToCurve(group_map),
        hiding: Blinded,
    }
}

impl<C> From<Opening<C, Schnorr<C>>> for OpeningProof<C>
where
    C: Glv,
{
    fn from(o: Opening<C, Schnorr<C>>) -> Self {
        Self {
            lr: o.rounds,
            delta: o.last.delta,
            z1: o.last.z1,
            z2: o.last.z2,
            sg: o.sg,
        }
    }
}

impl<C> From<&OpeningProof<C>> for Opening<C, Schnorr<C>>
where
    C: Glv,
{
    fn from(o: &OpeningProof<C>) -> Self {
        Self {
            rounds: o.lr.clone(),
            last: Schnorr {
                delta: o.delta,
                z1: o.z1,
                z2: o.z2,
            },
            sg: o.sg,
        }
    }
}
