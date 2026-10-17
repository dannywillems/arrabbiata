//! Kimchi's SNARK (o1-labs/proof-systems, crate `kimchi`), as a separate
//! implementation over arrabbiata's components: the gadgets of
//! `arrabbiata-kimchi-gadgets`, the IPA of `arrabbiata-ipa`, the
//! accumulation of `arrabbiata-halo`, and any sponge.
//!
//! Each module is checked against proof-systems' own output:
//!
//! - [`oracle`]: Kimchi's two-sponge Fiat-Shamir, the base-field sponge
//!   that absorbs commitments and squeezes 128-bit challenges, the
//!   scalar-field sponge that absorbs evaluations, and the map of a 128-bit
//!   challenge to a scalar through the curve's endomorphism;
//! - [`accumulation`]: the verifier as a reduction to Halo's accumulation,
//!   composed with Halo's decider and fold;
//! - [`constraints`]: the constraint system, built as Kimchi builds it
//!   (lookup domain, zero-knowledge rows, domains, padding), with its
//!   permutation, selector and coefficient columns;
//! - [`gates`]: the constraints of each gate type, a parameter of the
//!   prover, Kimchi's from the gadget crate;
//! - [`ipa`]: Kimchi's opening as an instance of the generic IPA of
//!   `arrabbiata-ipa`;
//! - [`lookup`]: the lookup patterns, the built-in tables and the lookup
//!   index;
//! - [`permutation`]: the shifts of the copy argument, sampled as Kimchi
//!   samples them (Blake2b512 of a counter, the first non-residues not in
//!   the domain);
//! - [`prover`]: the prover, round by round, lookups and previous
//!   accumulators included, randomness a parameter ([`rng`]);
//! - [`srs`]: Kimchi's SRS, hashed to the curve by Kimchi's group map, and
//!   chunked commitments;
//! - [`verifier`]: the verifier, replaying the transcript and checking the
//!   opening;
//! - [`verifier_index`]: the commitments to the fixed columns, masked as
//!   Kimchi masks them.

#![no_std]

extern crate alloc;

pub mod accumulation;
pub mod constraints;
pub mod gates;
pub mod ipa;
pub mod lookup;
pub mod oracle;
pub mod permutation;
pub mod prover;
pub mod rng;
pub mod srs;
pub mod verifier;
pub mod verifier_index;
