//! Grumpkin: `y^2 = x^3 - 17` over the BN254 scalar field `Fr`, with as
//! many points as the BN254 base field has elements.
//!
//! BN254 `G1` and Grumpkin form a cycle: each curve's scalar field is the
//! other's base field. A circuit over `Fr` (the field BN254 KZG commits
//! to) computes natively on Grumpkin points, as Vesta does in a circuit
//! over the Pasta field `Fq`.
//!
//! The parameters (`b = -17`, the generator `(1, sqrt(-16))` with the root
//! below) are those of the arkworks `ark-grumpkin` crate, from Aztec's
//! barretenberg. The order is checked in the tests: `r - 1` times the
//! generator, for `r` the BN254 base field modulus, is its negation.

#![no_std]

use arrabbiata_bn254::{Fq, Fr};
use arrabbiata_curve::Curve;
use arrabbiata_field::{Field, PrimeField};

/// The `y` coordinate of the generator, a square root of `-16`, as
/// little-endian limbs: 17631683881184975370165255887551781615748388533673675138860.
const GENERATOR_Y: [u64; 4] = [
    0x833f_c48d_823f_272c,
    0x2d27_0d45_f118_1294,
    0xcf13_5e75_06a4_5d63,
    0x0000_0000_0000_0002,
];

/// Grumpkin: `y^2 = x^3 - 17` over [`Fr`], of order the BN254 base field
/// modulus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grumpkin;

impl Curve for Grumpkin {
    type Base = Fr;
    type Scalar = Fq;

    fn b() -> Fr {
        -Fr::from_u64(17)
    }

    /// `(1, sqrt(-16))`, as in arkworks and barretenberg.
    fn generator() -> (Fr, Fr) {
        match Fr::from_limbs(&GENERATOR_Y) {
            Some(y) => (Fr::ONE, y),
            None => unreachable!("the generator ordinate is a canonical element of Fr"),
        }
    }
}
