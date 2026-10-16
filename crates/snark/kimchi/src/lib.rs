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
//! - [`permutation`]: the shifts of the copy argument, sampled as Kimchi
//!   samples them (Blake2b512 of a counter, the first non-residues not in
//!   the domain).

#![no_std]

extern crate alloc;

pub mod oracle;
pub mod permutation;
