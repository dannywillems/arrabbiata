//! Vanilla PlonK: the protocol on the vanilla arithmetization of
//! [`arrabbiata_plonkish::vanilla`], re-exported here.
//!
//! The arithmetization of the PlonK paper is one gadget, [`Arithmetic`],
//! over three witness columns `a, b, c` and five fixed columns
//! `q_L, q_R, q_O, q_M, q_C`:
//!
//! ```text
//! q_L a + q_R b + q_O c + q_M a b + q_C = 0.
//! ```
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::{ipa, prove, setup, vanilla, verify};
//!
//! // x^3 + x + 5 = 35, with x = 3 private and 35 public.
//! let (c, w, ()) = build::<F, _, _>(|b| {
//!     let x = b.witness(F::from_u64(3));
//!     let x2 = b.mul(&x, &x);
//!     let x3 = b.mul(&x2, &x);
//!     let sum = b.add(&x3, &x);
//!     let lhs = b.add_constant(&sum, F::from_u64(5));
//!     let rhs = b.public_input(F::from_u64(35));
//!     b.assert_equal(&lhs, &rhs);
//! });
//! let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
//! let pcs = ipa::pallas(&circuit);
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
//! verify(&vk, &pcs, ipa::pallas_transcript(), &[F::from_u64(35)], proof).unwrap();
//! ```

pub use arrabbiata_plonkish::vanilla::{Arithmetic, Vanilla, VanillaCircuit, circuit, witness};

/// Vanilla PlonK over `F`, with the commitment scheme `PC` and the sponge
/// `S`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa::PallasIpa, vanilla::VanillaPlonk};
/// use arrabbiata_poseidon::PoseidonSponge;
/// use arrabbiata_transcript::typed::{Pattern, Protocol};
/// type P = <VanillaPlonk<F, PallasIpa, PoseidonSponge<F>> as Protocol>::Transcript;
/// assert_eq!(<P as Pattern>::MESSAGES, 5);
/// ```
pub type VanillaPlonk<F, PC, S> = crate::Plonk<F, PC, S, Vanilla, 3, 5, 1>;
