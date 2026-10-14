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
//!
//! ```
//! use arrabbiata_commit::IdentityCommitment;
//! use arrabbiata_relation::examples;
//! use arrabbiata_transcript::toy_sponge;
//!
//! // Arithmetize, commit, relax, check.
//! let cs = examples::fibonacci(8);
//! let trace = examples::fibonacci_trace(&cs, 1, 2);
//! cs.check_trace(&trace).unwrap();
//! let instance = cs.commit(&IdentityCommitment, &trace, &mut toy_sponge(0));
//! let (instance, witness) = cs.relax(&IdentityCommitment, instance, trace);
//! cs.check_relaxed(&IdentityCommitment, &instance, &witness).unwrap();
//! ```

#![no_std]

extern crate alloc;

pub mod aggregation;
pub mod error;
pub mod examples;
pub mod instance;
pub mod system;

pub use instance::{Instance, RelaxedInstance, RelaxedWitness};
pub use system::{ConstraintSystem, RowView, Trace};
