//! Frontends: typed APIs to write circuits for a given arithmetization.
//!
//! - [`vanilla`]: circuits for vanilla PlonK, the arithmetization of the
//!   PlonK paper (three wires and the gate
//!   `q_L a + q_R b + q_O c + q_M a b + q_C`). Its circuits compile to
//!   `arrabbiata_plonk::vanilla`, and nowhere else.
//!
//! Circuits over a custom gate set are written with the builder of
//! `arrabbiata_plonk::circuit` directly.
//!
//! ```
//! use arrabbiata_field::{Field, Goldilocks as F};
//! use arrabbiata_frontend::vanilla::build;
//!
//! let (circuit, witness, ()) = build::<F, _, _>(|b| {
//!     let x = b.witness(F::new(3));
//!     let y = b.mul(&x, &x);
//!     let out = b.public_input(F::new(9));
//!     b.assert_equal(&y, &out);
//! });
//! assert!(circuit.is_satisfied(&witness));
//! ```

#![no_std]

extern crate alloc;

pub mod vanilla;
