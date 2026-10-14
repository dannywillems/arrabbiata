//! PlonK over BN254 with the KZG commitment: the configuration of most
//! Ethereum-verified PlonK proofs (constant-size proofs, a trusted setup).
//!
//! Fiat-Shamir runs on a Poseidon sponge over the BN254 scalar field.
//!
//! ```
//! use arrabbiata_bn254::Fr;
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::build;
//! use arrabbiata_plonk::{kzg, prove, setup, verify};
//!
//! let (circuit, witness, ()) = build::<Fr, _, _>(|b| {
//!     let x = b.witness(Fr::from_u64(3));
//!     let y = b.mul(&x, &x);
//!     let out = b.public_input(Fr::from_u64(9));
//!     b.assert_equal(&y, &out);
//! });
//! let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0xdead_beef));
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
//! verify(&vk, &pcs, kzg::transcript(), &[Fr::from_u64(9)], proof).unwrap();
//! ```

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_frontend::Circuit;
use arrabbiata_poseidon::{PoseidonSponge, bn254_scalar_sponge};

use crate::Plonk;

/// KZG over BN254.
pub type Kzg = arrabbiata_kzg::Kzg<Bn254>;

/// PlonK over BN254 with KZG, on a Poseidon sponge over `Fr`.
pub type PlonkKzg = Plonk<Fr, Kzg, PoseidonSponge<Fr>>;

/// The domain separator of the PlonK transcript.
const DOMAIN: u64 = 0x706c_6f6e_6b5f_6b7a;

/// A reference string large enough for `circuit` (four times its number of
/// rows: the quotient has degree below `3 n`).
///
/// INSECURE: generated from a known `tau`; see
/// [`arrabbiata_kzg::Kzg::insecure_setup`]. Load a ceremony's string with
/// [`arrabbiata_kzg::Kzg::from_powers`] for anything but tests.
///
/// ```
/// use arrabbiata_bn254::Fr;
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::build;
/// use arrabbiata_plonk::kzg;
/// let (circuit, _, ()) = build::<Fr, _, _>(|_| ());
/// assert_eq!(kzg::insecure_srs(&circuit, Fr::ONE + Fr::ONE).size(), 4 * circuit.num_rows());
/// ```
#[must_use]
pub fn insecure_srs(circuit: &Circuit<Fr>, tau: Fr) -> Kzg {
    Kzg::insecure_setup(4 * circuit.num_rows(), tau)
}

/// A fresh Poseidon transcript for PlonK over BN254.
///
/// ```
/// use arrabbiata_plonk::kzg;
/// use arrabbiata_transcript::Transcript;
/// assert_eq!(kzg::transcript().squeeze(), kzg::transcript().squeeze());
/// ```
#[must_use]
pub fn transcript() -> PoseidonSponge<Fr> {
    bn254_scalar_sponge(DOMAIN)
}
