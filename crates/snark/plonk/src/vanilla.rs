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

pub use arrabbiata_plonkish::vanilla::{Arithmetic, Vanilla, VanillaCircuit, circuit, witness};

/// Vanilla PlonK over `F`, with the commitment scheme `PC` and the sponge
/// `S`.
pub type VanillaPlonk<F, PC, S> = crate::Plonk<F, PC, S, Vanilla, 3, 5, 1>;
