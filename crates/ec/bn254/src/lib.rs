//! BN254 (also known as alt_bn128 or BN256): the Barreto-Naehrig curve of
//! Ethereum's precompiles (EIP-196, EIP-197), with its pairing.
//!
//! - `Fq`: the base field, `p` of 254 bits;
//! - `Fr`: the scalar field, `r` of 254 bits, two-adicity 28;
//! - `G1`: `y^2 = x^3 + 3` over `Fq`, of prime order `r`;
//! - `G2`: the subgroup of order `r` of the sextic twist
//!   `y^2 = x^3 + 3 / xi` over `Fq2 = Fq[u] / (u^2 + 1)`, `xi = 9 + u`;
//! - `Gt`: the subgroup of order `r` of `Fq12*`.
//!
//! The pairing is the reduced Tate pairing, computed for clarity rather
//! than speed (see [`pairing`]). The security of BN254 is estimated at
//! about 100 bits since the improved tower number field sieve (Kim and
//! Barbulescu, CRYPTO 2016), below the 128 its size once suggested.

#![no_std]

pub mod fq12;
pub mod pairing;

use arrabbiata_curve::{Curve, Pairing, PairingInput};
use arrabbiata_field::{
    Field, MontField, MontgomeryConfig,
    extension::{Quadratic, QuadraticConfig},
};

/// The modulus of the base field, `p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FqConfig;

impl MontgomeryConfig<4> for FqConfig {
    const MODULUS: [u64; 4] = [
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ];
    // 3 is a quadratic non-residue modulo p (tested).
    const GENERATOR: u64 = 3;
}

/// The modulus of the scalar field, `r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrConfig;

impl MontgomeryConfig<4> for FrConfig {
    const MODULUS: [u64; 4] = [
        0x43e1f593f0000001,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ];
    // 5 is a quadratic non-residue modulo r (tested).
    const GENERATOR: u64 = 5;
}

/// The base field.
pub type Fq = MontField<FqConfig, 4>;

/// The scalar field.
pub type Fr = MontField<FrConfig, 4>;

/// `Fq2 = Fq[u] / (u^2 + 1)`: `-1` is not a square since `p = 3 mod 4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fq2Config;

impl QuadraticConfig for Fq2Config {
    type Base = Fq;

    fn non_residue() -> Fq {
        -Fq::ONE
    }
}

/// The quadratic extension of the base field.
pub type Fq2 = Quadratic<Fq2Config>;

/// `xi = 9 + u`, the non-residue defining the twist and `Fq12`.
#[must_use]
pub const fn xi() -> Fq2 {
    Fq2::new(Fq::from_u64_const(9), Fq::from_u64_const(1))
}

/// `G1`: `y^2 = x^3 + 3` over `Fq`, generator `(1, 2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G1;

impl Curve for G1 {
    type Base = Fq;
    type Scalar = Fr;

    fn b() -> Fq {
        Fq::from_u64(3)
    }

    fn generator() -> (Fq, Fq) {
        (Fq::ONE, Fq::from_u64(2))
    }
}

/// `G2`: the twist `y^2 = x^3 + 3 / xi` over `Fq2`, with the generator of
/// EIP-197.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G2;

const fn fq(limbs: [u64; 4]) -> Fq {
    match Fq::from_canonical_limbs(limbs) {
        Some(x) => x,
        None => panic!("a G2 generator coordinate is not below p"),
    }
}

impl Curve for G2 {
    type Base = Fq2;
    type Scalar = Fr;

    fn b() -> Fq2 {
        match xi().inverse() {
            Some(inverse) => inverse.scale(Fq::from_u64(3)),
            None => panic!("xi is not zero"),
        }
    }

    fn generator() -> (Fq2, Fq2) {
        (
            Fq2::new(
                fq([
                    0x46debd5cd992f6ed,
                    0x674322d4f75edadd,
                    0x426a00665e5c4479,
                    0x1800deef121f1e76,
                ]),
                fq([
                    0x97e485b7aef312c2,
                    0xf1aa493335a9e712,
                    0x7260bfb731fb5d25,
                    0x198e9393920d483a,
                ]),
            ),
            Fq2::new(
                fq([
                    0x4ce6cc0166fa7daa,
                    0xe3d1e7690c43d37b,
                    0x4aab71808dcb408f,
                    0x12c85ea5db8c6deb,
                ]),
                fq([
                    0x55acdadcd122975b,
                    0xbc4b313370b38ef3,
                    0xec9e99ad690c3395,
                    0x090689d0585ff075,
                ]),
            ),
        )
    }
}

/// The BN254 pairing, as a [`Pairing`] instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bn254;

impl Pairing for Bn254 {
    type G1 = G1;
    type G2 = G2;

    fn pairing_product_is_one(pairs: &[PairingInput<Self>]) -> bool {
        pairing::multi_pairing(pairs).is_one()
    }
}
