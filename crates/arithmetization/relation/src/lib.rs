//! Homogeneous polynomial relations and their relaxed form.
//!
//! A [`ConstraintSystem`] is a list of polynomial constraints on the rows of
//! a trace. Folding needs the *relaxed* form of its relation, in which every
//! constraint is homogenized with a scalar `u` and a per-row error `E` is
//! allowed:
//!
//! ```text
//! sum_j c_j f_j^D(u, w_i) = E_i   for every row i.
//! ```
//!
//! This is the relation of the Arrabbiata note, which generalizes Nova's
//! relaxed R1CS (`D = 2`, one constraint) to any degree and any number of
//! constraints; how the constraints are combined is the type parameter
//! [`aggregation::Aggregation`].

#![no_std]

extern crate alloc;

pub mod aggregation;
pub mod error;
pub mod examples;
pub mod instance;
pub mod system;

pub use instance::{Instance, RelaxedInstance, RelaxedWitness};
pub use system::{ConstraintSystem, RowView, Trace};
