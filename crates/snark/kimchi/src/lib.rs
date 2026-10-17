//! Kimchi's SNARK (o1-labs/proof-systems, crate `kimchi`), as a separate
//! implementation over arrabbiata's components: the gadgets of
//! `arrabbiata-kimchi-gadgets`, the IPA of `arrabbiata-ipa`, the
//! accumulation of `arrabbiata-halo`, and any sponge.
//!
//! Built milestone by milestone (plan 0006 of the project notes), each
//! checked against proof-systems' own output:
//!
//! - [`oracle`]: Kimchi's two-sponge Fiat-Shamir, the base-field sponge
//!   that absorbs commitments and squeezes 128-bit challenges, the
//!   scalar-field sponge that absorbs evaluations, and the map of a 128-bit
//!   challenge to a scalar through the curve's endomorphism;
//! - [`constraints`]: the constraint system, built as Kimchi builds it
//!   (lookup domain, zero-knowledge rows, domains, padding), with its
//!   permutation, selector and coefficient columns;
//! - [`permutation`]: the shifts of the copy argument, sampled as Kimchi
//!   samples them (Blake2b512 of a counter, the first non-residues not in
//!   the domain);
//! - [`prover`]: the prover, round by round (so far the witness round, up
//!   to `beta` and `gamma`), randomness a parameter ([`rng`]);
//! - [`srs`]: Kimchi's SRS, hashed to the curve by Kimchi's group map, and
//!   chunked commitments;
//! - [`verifier_index`]: the commitments to the fixed columns, masked as
//!   Kimchi masks them.

#![no_std]

extern crate alloc;

pub mod constraints;
pub mod oracle;
pub mod permutation;
pub mod prover;
pub mod rng;
pub mod srs;
pub mod verifier_index;
