//! Frontends: typed APIs to write circuits for a given arithmetization.
//!
//! - [`vanilla`]: circuits for vanilla PlonK, the arithmetization of the
//!   PlonK paper (three wires and the gate
//!   `q_L a + q_R b + q_O c + q_M a b + q_C`). Its circuits compile to
//!   `arrabbiata_plonk::vanilla`, and nowhere else.
//!
//! Circuits over a custom gate set are written with the builder of
//! `arrabbiata_plonk::circuit` directly.

#![no_std]

extern crate alloc;

pub mod vanilla;
