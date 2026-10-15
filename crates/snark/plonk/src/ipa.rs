//! PlonK with the inner-product argument: no trusted setup, logarithmic
//! proofs, a linear-time verifier.
//!
//! Two configurations:
//!
//! - over Pallas, polynomials in its scalar field `Fq` (the Halo 2
//!   configuration, the first half of the Pasta cycle);
//! - over BN254 `G1`, polynomials in `Fr`: the same field and the same
//!   commitment type as [`crate::kzg`], so the two schemes can be compared
//!   directly. Their proofs remain incompatible: the scheme is part of the
//!   proof type and of the transcript.
//!
//! Fiat-Shamir and the generators come from Poseidon sponges.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_pasta::Fq;
//! use arrabbiata_plonk::{ipa, prove, setup, vanilla, verify};
//!
//! let (c, w, ()) = build::<Fq, _, _>(|b| {
//!     let x = b.witness(Fq::from_u64(3));
//!     let y = b.mul(&x, &x);
//!     let out = b.public_input(Fq::from_u64(9));
//!     b.assert_equal(&y, &out);
//! });
//! let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
//! let pcs = ipa::pallas(&circuit);
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
//! verify(&vk, &pcs, ipa::pallas_transcript(), &[Fq::from_u64(9)], proof).unwrap();
//! ```
//!
//! A proof made with KZG is not accepted by the IPA verifier of the same
//! field and commitment type:
//!
//! ```compile_fail
//! use arrabbiata_bn254::Fr;
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_plonk::{ipa, kzg, prove, setup, vanilla, verify};
//!
//! let (c, w, ()) = build::<Fr, _, _>(|b| {
//!     b.witness(Fr::ONE);
//! });
//! let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
//! let kzg_pcs = kzg::insecure_srs(&circuit, Fr::from_u64(7));
//! let ipa_pcs = ipa::bn254(&circuit);
//! let (kzg_pk, _) = setup(&circuit, &kzg_pcs).unwrap();
//! let (_, ipa_vk) = setup(&circuit, &ipa_pcs).unwrap();
//! let proof = prove(&kzg_pk, &kzg_pcs, kzg::transcript(), &witness).unwrap();
//! let _ = verify(&ipa_vk, &ipa_pcs, kzg::transcript(), &[], proof);
//! ```

use arrabbiata_bn254::{Fr, G1};
use arrabbiata_ipa::Ipa;
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{
    PoseidonSponge, bn254_base_sponge, bn254_scalar_sponge, pallas_sponge, vesta_sponge,
};

use crate::{Circuit, GateSet, Plonk};

/// IPA on Pallas, polynomials over `Fq`.
pub type PallasIpa = Ipa<Pallas, PoseidonSponge<Fq>>;

/// PlonK over `Fq` for the gate set `G`, with IPA on Pallas, on a Poseidon
/// sponge over `Fq`.
pub type PlonkPallasIpa<G, const W: usize, const X: usize, const Q: usize> =
    Plonk<Fq, PallasIpa, PoseidonSponge<Fq>, G, W, X, Q>;

/// IPA on BN254 `G1`, polynomials over `Fr`.
pub type Bn254Ipa = Ipa<G1, PoseidonSponge<Fr>>;

/// PlonK over `Fr` for the gate set `G`, with IPA on BN254 `G1`, on a
/// Poseidon sponge over `Fr`.
pub type PlonkBn254Ipa<G, const W: usize, const X: usize, const Q: usize> =
    Plonk<Fr, Bn254Ipa, PoseidonSponge<Fr>, G, W, X, Q>;

const GENERATORS: u64 = 0x6970_615f_6765_6e73;
const OPENINGS: u64 = 0x6970_615f_6f70_656e;
const DOMAIN: u64 = 0x706c_6f6e_6b5f_6970;

/// The number of generators for `circuit`: a power of two at least the
/// number of coefficients of its largest polynomial (the quotient).
fn size<F, G, const W: usize, const X: usize, const Q: usize>(
    circuit: &Circuit<F, G, W, X, Q>,
) -> usize
where
    F: arrabbiata_field::Field,
    G: GateSet<F, W, X>,
{
    circuit.max_coefficients().next_power_of_two()
}

/// IPA on Pallas, sized for `circuit`.
///
/// ```
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_plonk::ipa;
/// use arrabbiata_plonk::vanilla;
/// let (c, _, ()) = build::<Fq, _, _>(|_| ());
/// let circuit = vanilla::circuit(&c).unwrap();
/// assert_eq!(ipa::pallas(&circuit).size(), 8);
/// ```
#[must_use]
pub fn pallas<G, const W: usize, const X: usize, const Q: usize>(
    circuit: &Circuit<Fq, G, W, X, Q>,
) -> PallasIpa
where
    G: GateSet<Fq, W, X>,
{
    Ipa::setup(
        &mut pallas_sponge(GENERATORS),
        size(circuit),
        vesta_sponge(OPENINGS),
    )
}

/// IPA on BN254 `G1`, sized for `circuit`.
///
/// ```
/// use arrabbiata_bn254::Fr;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_plonk::ipa;
/// use arrabbiata_plonk::vanilla;
/// let (c, _, ()) = build::<Fr, _, _>(|_| ());
/// let circuit = vanilla::circuit(&c).unwrap();
/// assert_eq!(ipa::bn254(&circuit).size(), 8);
/// ```
#[must_use]
pub fn bn254<G, const W: usize, const X: usize, const Q: usize>(
    circuit: &Circuit<Fr, G, W, X, Q>,
) -> Bn254Ipa
where
    G: GateSet<Fr, W, X>,
{
    Ipa::setup(
        &mut bn254_base_sponge(GENERATORS),
        size(circuit),
        bn254_scalar_sponge(OPENINGS),
    )
}

/// A fresh Poseidon transcript for PlonK over `Fq`.
///
/// ```
/// use arrabbiata_plonk::ipa;
/// use arrabbiata_transcript::Transcript;
/// assert_eq!(ipa::pallas_transcript().squeeze(), ipa::pallas_transcript().squeeze());
/// ```
#[must_use]
pub fn pallas_transcript() -> PoseidonSponge<Fq> {
    vesta_sponge(DOMAIN)
}

/// A fresh Poseidon transcript for PlonK over the BN254 scalar field.
///
/// ```
/// use arrabbiata_plonk::ipa;
/// use arrabbiata_transcript::Transcript;
/// assert_eq!(ipa::bn254_transcript().squeeze(), ipa::bn254_transcript().squeeze());
/// ```
#[must_use]
pub fn bn254_transcript() -> PoseidonSponge<Fr> {
    bn254_scalar_sponge(DOMAIN)
}
