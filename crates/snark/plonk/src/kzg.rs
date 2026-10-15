//! PlonK over BN254 with the KZG commitment: the configuration of most
//! Ethereum-verified PlonK proofs (constant-size proofs, a trusted setup).
//!
//! Fiat-Shamir runs on a Poseidon sponge over the BN254 scalar field.

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_poseidon::{PoseidonSponge, bn254_scalar_sponge};

use crate::{Circuit, GateSet, Plonk, QuotientSizing};

/// KZG over BN254.
pub type Kzg = arrabbiata_kzg::Kzg<Bn254>;

/// PlonK over BN254 for the gate set `G`, with KZG, on a Poseidon sponge
/// over `Fr`.
pub type PlonkKzg<G, const W: usize, const X: usize, const Q: usize> =
    Plonk<Fr, Kzg, PoseidonSponge<Fr>, G, W, X, Q>;

/// The domain separator of the PlonK transcript.
const DOMAIN: u64 = 0x706c_6f6e_6b5f_6b7a;

/// A reference string large enough for `circuit`: as many powers as the
/// quotient has coefficients.
///
/// INSECURE: generated from a known `tau`; see
/// [`arrabbiata_kzg::Kzg::insecure_setup`]. Load a ceremony's string with
/// [`arrabbiata_kzg::Kzg::from_powers`] for anything but tests.
#[must_use]
pub fn insecure_srs<G, const W: usize, const X: usize, const Q: usize>(
    circuit: &Circuit<Fr, G, W, X, Q>,
    tau: Fr,
) -> Kzg
where
    G: GateSet<Fr, W, X>,
{
    Kzg::insecure_setup(circuit.max_coefficients(), tau)
}

/// A fresh Poseidon transcript for PlonK over BN254.
#[must_use]
pub fn transcript() -> PoseidonSponge<Fr> {
    bn254_scalar_sponge(DOMAIN)
}
