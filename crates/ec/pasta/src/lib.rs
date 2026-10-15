//! The Pasta cycle of elliptic curves: Pallas and Vesta.
//!
//! - Pallas: `y^2 = x^3 + 5` over `Fp`, with `q` points;
//! - Vesta: `y^2 = x^3 + 5` over `Fq`, with `p` points;
//!
//! where `p = 2^254 + 45560315531419706090280762371685220353` and
//! `q = 2^254 + 45560315531506369815346746415080538113` are primes. Each
//! curve's scalar field is the other's base field, which is what a recursive
//! proof system on a cycle needs: the verifier of a proof over one curve is
//! a circuit over the other. Both fields have two-adicity 32.
//!
//! Reference: Hopwood, *The Pasta Curves for Halo 2 and Beyond*, 2020;
//! <https://github.com/zcash/pasta>.
//!
//! The moduli are checked in the tests by the group orders: `q` times the
//! Pallas generator, and `p` times the Vesta generator, is the identity.

#![no_std]

use arrabbiata_curve::Curve;
use arrabbiata_field::{Field, MontField, MontgomeryConfig};

/// The modulus of the Pallas base field, `p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FpConfig;

impl MontgomeryConfig<4> for FpConfig {
    const MODULUS: [u64; 4] = [
        0x992d30ed00000001,
        0x224698fc094cf91b,
        0,
        0x4000000000000000,
    ];
    // 5 is a quadratic non-residue modulo p (tested).
    const GENERATOR: u64 = 5;
}

/// The modulus of the Vesta base field, `q`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FqConfig;

impl MontgomeryConfig<4> for FqConfig {
    const MODULUS: [u64; 4] = [
        0x8c46eb2100000001,
        0x224698fc0994a8dd,
        0,
        0x4000000000000000,
    ];
    // 5 is a quadratic non-residue modulo q (tested).
    const GENERATOR: u64 = 5;
}

/// The Pallas base field, which is the Vesta scalar field.
pub type Fp = MontField<FpConfig, 4>;

/// The Vesta base field, which is the Pallas scalar field.
pub type Fq = MontField<FqConfig, 4>;

/// Pallas: `y^2 = x^3 + 5` over [`Fp`], of order `q`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pallas;

impl Curve for Pallas {
    type Base = Fp;
    type Scalar = Fq;

    fn b() -> Fp {
        Fp::from_u64(5)
    }

    /// `(-1, 2)`, as in the reference implementation.
    fn generator() -> (Fp, Fp) {
        (-Fp::ONE, Fp::from_u64(2))
    }
}

/// Vesta: `y^2 = x^3 + 5` over [`Fq`], of order `p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vesta;

impl Curve for Vesta {
    type Base = Fq;
    type Scalar = Fp;

    fn b() -> Fq {
        Fq::from_u64(5)
    }

    /// `(-1, 2)`, as in the reference implementation.
    fn generator() -> (Fq, Fq) {
        (-Fq::ONE, Fq::from_u64(2))
    }
}
