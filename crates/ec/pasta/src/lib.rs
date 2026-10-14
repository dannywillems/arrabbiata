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
//!
//! ```
//! use arrabbiata_curve::Projective;
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
//!
//! // The cycle: Pallas scalars are Vesta coordinates, and conversely.
//! let pallas = Projective::<Pallas>::generator() * Fq::from_u64(7);
//! let vesta = Projective::<Vesta>::generator() * Fp::from_u64(7);
//! assert!(pallas.is_on_curve() && vesta.is_on_curve());
//! ```

#![no_std]

use arrabbiata_curve::Curve;
use arrabbiata_field::{Field, MontField, MontgomeryConfig};

/// The modulus of the Pallas base field, `p`.
///
/// ```
/// use arrabbiata_field::MontgomeryConfig;
/// use arrabbiata_pasta::FpConfig;
/// assert_eq!(FpConfig::MODULUS[3], 0x4000000000000000);
/// ```
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
///
/// ```
/// use arrabbiata_field::MontgomeryConfig;
/// use arrabbiata_pasta::FqConfig;
/// assert_eq!(FqConfig::MODULUS[0], 0x8c46eb2100000001);
/// ```
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
///
/// ```
/// use arrabbiata_field::{Field, TwoAdicField};
/// use arrabbiata_pasta::Fp;
/// assert_eq!(Fp::TWO_ADICITY, 32);
/// assert_eq!(Fp::from_u64(2) * Fp::from_u64(3), Fp::from_u64(6));
/// ```
pub type Fp = MontField<FpConfig, 4>;

/// The Vesta base field, which is the Pallas scalar field.
///
/// ```
/// use arrabbiata_field::{Field, TwoAdicField};
/// use arrabbiata_pasta::Fq;
/// assert_eq!(Fq::TWO_ADICITY, 32);
/// ```
pub type Fq = MontField<FqConfig, 4>;

/// Pallas: `y^2 = x^3 + 5` over [`Fp`], of order `q`.
///
/// ```
/// use arrabbiata_curve::Projective;
/// use arrabbiata_pasta::Pallas;
/// assert!(Projective::<Pallas>::generator().is_on_curve());
/// ```
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
///
/// ```
/// use arrabbiata_curve::Projective;
/// use arrabbiata_pasta::Vesta;
/// assert!(Projective::<Vesta>::generator().is_on_curve());
/// ```
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
