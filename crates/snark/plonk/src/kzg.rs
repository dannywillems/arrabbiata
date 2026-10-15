//! PlonK over BN254 with the KZG commitment: the configuration of most
//! Ethereum-verified PlonK proofs (constant-size proofs, a trusted setup).
//!
//! Fiat-Shamir runs on a Poseidon sponge over the BN254 scalar field.
//!
//! ```
//! use arrabbiata_bn254::Fr;
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_plonk::{kzg, prove, setup, vanilla, verify};
//!
//! let (c, w, ()) = build::<Fr, _, _>(|b| {
//!     let x = b.witness(Fr::from_u64(3));
//!     let y = b.mul(&x, &x);
//!     let out = b.public_input(Fr::from_u64(9));
//!     b.assert_equal(&y, &out);
//! });
//! let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
//! let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0xdead_beef));
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
//! verify(&vk, &pcs, kzg::transcript(), &[Fr::from_u64(9)], proof).unwrap();
//! ```

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_poseidon::{PoseidonSponge, bn254_scalar_sponge};

use crate::{Circuit, GateSet, Plonk};

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
///
/// ```
/// use arrabbiata_bn254::Fr;
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_plonk::kzg;
/// use arrabbiata_plonk::vanilla;
/// let (c, _, ()) = build::<Fr, _, _>(|_| ());
/// let circuit = vanilla::circuit(&c).unwrap();
/// let srs = kzg::insecure_srs(&circuit, Fr::ONE + Fr::ONE);
/// assert_eq!(srs.size(), circuit.max_coefficients());
/// ```
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
