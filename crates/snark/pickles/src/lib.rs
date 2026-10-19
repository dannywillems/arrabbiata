//! Pickles (Mina's recursion over Kimchi), built from its structure on
//! arrabbiata's components.
//!
//! Kimchi's verifier for a proof over a curve factors into a group half,
//! native in the curve's base field, and a scalar half, native in its
//! scalar field (`arrabbiata_kimchi::deferred`). Over a cycle of curves,
//! each half runs in the circuit whose field it is native in, and the
//! values between them are Pickles' deferred values.
//!
//! - [`model`]: the signature of fields, its native model, and its circuit
//!   model, one Generic row per operation;
//! - [`transcript`]: what the circuit model needs to run Kimchi's typed
//!   scalar-sponge transcript (the permutation's rows, the challenges'
//!   splits and endomorphism rows);
//! - [`scalar`]: the scalar half written once over that signature, and the
//!   typed gadget that is its circuit;
//! - [`group`], [`curve`]: the group half over the points of a model, and
//!   its circuit reading, scalars of the other field shifted;
//! - [`statement`]: the sorts a statement crosses the cycle in, each word
//!   with a canonical lift;
//! - [`wrap`]: the wrap circuit, verifying a base-case step proof;
//! - [`step`]: the step circuit, verifying a wrap proof and applying a
//!   rule;
//! - [`wrap_one`]: the wrap circuit of a step proof with a predecessor,
//!   finalizing in its field the wrap proof the step proof left
//!   unfinalized.

#![no_std]

extern crate alloc;

pub mod curve;
pub mod group;
pub mod model;
pub mod scalar;
pub mod statement;
pub mod step;
pub mod transcript;
pub mod wrap;
pub mod wrap_one;
