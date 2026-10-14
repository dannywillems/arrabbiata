//! The degree-12 extension `Fq12 = Fq2[w] / (w^6 - xi)`, with
//! `xi = 9 + u`, where the pairing takes its values.
//!
//! The extension is built in one step over `Fq2` (rather than as the usual
//! tower `Fq2 -> Fq6 -> Fq12`) with schoolbook multiplication and the
//! reduction `w^6 = xi`: slower, and obviously correct. `w^6 - xi` is
//! irreducible over `Fq2` because `xi` is neither a square nor a cube
//! there, which the tests check through the non-degeneracy of the pairing.
//!
//! ```
//! use arrabbiata_bn254::fq12::Fq12;
//! let w = Fq12::w();
//! let w6 = w * w * w * w * w * w;
//! assert_eq!(w6, Fq12::from_fq2(arrabbiata_bn254::xi()));
//! ```

use core::ops::Mul;

use arrabbiata_field::Field;

use crate::{Fq2, xi};

/// An element `sum_i c_i w^i` of `Fq12`, with `c_i` in `Fq2`.
///
/// ```
/// use arrabbiata_bn254::fq12::Fq12;
/// assert_eq!(Fq12::one() * Fq12::w(), Fq12::w());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fq12 {
    c: [Fq2; 6],
}

/// `(p^12 - 1) / r`, the exponent of the final exponentiation, as
/// little-endian limbs. Computed offline and checked in the tests by
/// `(p^12 - 1) = r * FINAL_EXPONENT`.
pub const FINAL_EXPONENT: [u64; 44] = [
    0x86964b64ca86f120,
    0x40a4efb7e54523a4,
    0x837fa97896e84abb,
    0x361102b6b9b2b918,
    0xc0de81def35692da,
    0xbe04c7e8a6c3c760,
    0xd766f9c9d570bb7f,
    0xc230974d83561841,
    0x5bba1668c3be69a3,
    0x7f3811c410526294,
    0x29baee7ddadda71c,
    0xbf813b8d145da900,
    0x641bbadf423f9a2c,
    0xa80bb4ea44eacc5e,
    0xcd65664814fde37c,
    0x4a0364b9580291d2,
    0xee93dfb10826f0dd,
    0x6b42db8dc5514724,
    0xbb10cf430b0f3785,
    0x40494e406f804216,
    0x55cfe107acf3aafb,
    0x2088ec80e0ebae87,
    0x846a3ed011a337a0,
    0x48a45a4a1e3a5195,
    0xe5664568dfc50e16,
    0xab6a41294c0cc4eb,
    0x82d0d602d268c7da,
    0x6668449aed3cc48a,
    0x5062cd0fb2015dfc,
    0x7f2940a8b1ddb3d1,
    0x77f5b63a2a226448,
    0xfef0781361e443ae,
    0xf977870e88d5c6c8,
    0x790364a61f676baa,
    0x5887e72eceaddea3,
    0x1377e563a09a1b70,
    0x0c54efee1bd8c3b2,
    0x3ec3d15ad524d8f7,
    0xdaf15466b2383a5d,
    0xe1e30a73bb94fec0,
    0x6a1c71015f3f7be2,
    0x842d43bf6369b1ff,
    0x20fddadf107d20bc,
    0x0000002f4b6dc970,
];

impl Fq12 {
    /// The element with coefficients `c`.
    ///
    /// ```
    /// use arrabbiata_bn254::{Fq2, fq12::Fq12};
    /// use arrabbiata_field::Field;
    /// assert_eq!(Fq12::new([Fq2::ONE, Fq2::ZERO, Fq2::ZERO, Fq2::ZERO, Fq2::ZERO, Fq2::ZERO]), Fq12::one());
    /// ```
    #[must_use]
    pub const fn new(c: [Fq2; 6]) -> Self {
        Self { c }
    }

    /// The coefficients.
    ///
    /// ```
    /// use arrabbiata_bn254::fq12::Fq12;
    /// assert_eq!(Fq12::one().coefficients().len(), 6);
    /// ```
    #[must_use]
    pub const fn coefficients(&self) -> &[Fq2; 6] {
        &self.c
    }

    /// The multiplicative identity.
    ///
    /// ```
    /// use arrabbiata_bn254::fq12::Fq12;
    /// assert!(Fq12::one().is_one());
    /// ```
    #[must_use]
    pub const fn one() -> Self {
        Self::from_fq2(Fq2::ONE)
    }

    /// The generator `w` of the extension over `Fq2`.
    ///
    /// ```
    /// use arrabbiata_bn254::fq12::Fq12;
    /// assert!(!Fq12::w().is_one());
    /// ```
    #[must_use]
    pub const fn w() -> Self {
        let mut c = [Fq2::ZERO; 6];
        c[1] = Fq2::ONE;
        Self { c }
    }

    /// The embedding of `Fq2`.
    ///
    /// ```
    /// use arrabbiata_bn254::{Fq2, fq12::Fq12};
    /// use arrabbiata_field::Field;
    /// assert_eq!(Fq12::from_fq2(Fq2::ONE), Fq12::one());
    /// ```
    #[must_use]
    pub const fn from_fq2(a: Fq2) -> Self {
        let mut c = [Fq2::ZERO; 6];
        c[0] = a;
        Self { c }
    }

    /// Whether this is the identity.
    ///
    /// ```
    /// use arrabbiata_bn254::fq12::Fq12;
    /// assert!((Fq12::w() * Fq12::one()).is_one() == false);
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        *self == Self::one()
    }

    /// `self^e` for an exponent given as little-endian limbs.
    ///
    /// ```
    /// use arrabbiata_bn254::fq12::Fq12;
    /// let w = Fq12::w();
    /// assert_eq!(w.pow_limbs(&[3]), w * w * w);
    /// ```
    #[must_use]
    pub fn pow_limbs(&self, exponent: &[u64]) -> Self {
        let mut acc = Self::one();
        let mut i = exponent.len();
        while i > 0 {
            i -= 1;
            let mut bit: u32 = 64;
            while bit > 0 {
                bit -= 1;
                acc = acc * acc;
                if (exponent[i] >> bit) & 1 == 1 {
                    acc = acc * *self;
                }
            }
        }
        acc
    }
}

impl Mul for Fq12 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut wide = [Fq2::ZERO; 11];
        let mut i = 0;
        while i < 6 {
            let mut j = 0;
            while j < 6 {
                wide[i + j] = wide[i + j] + self.c[i] * rhs.c[j];
                j += 1;
            }
            i += 1;
        }
        // w^(6 + k) = xi w^k.
        let mut c = [Fq2::ZERO; 6];
        let mut k = 0;
        while k < 6 {
            c[k] = wide[k];
            if k + 6 < 11 {
                c[k] = c[k] + xi() * wide[k + 6];
            }
            k += 1;
        }
        Self { c }
    }
}
