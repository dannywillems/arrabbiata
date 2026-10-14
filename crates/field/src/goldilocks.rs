//! The Goldilocks field, `p = 2^64 - 2^32 + 1`.
//!
//! A 64-bit prime field, used here because it needs nothing beyond `u64` and
//! `u128` arithmetic. That keeps the reference implementation trivially
//! auditable and inside the subset of Rust that Aeneas extracts, so the
//! generic machinery can be tested and extracted end to end before the
//! 255-bit cycle fields exist.
//!
//! The implementation favours obvious correctness over speed: products are
//! reduced with a `u128` remainder, not with the special form of `p`.
//!
//! ```
//! use arrabbiata_field::{Field, Goldilocks, goldilocks::MODULUS};
//!
//! // Since p = 2^64 - 2^32 + 1, we have 2^64 = 2^32 - 1 in the field.
//! let two = Goldilocks::new(2);
//! assert_eq!(two.pow(64), Goldilocks::new((1u64 << 32) - 1));
//! assert_eq!((-Goldilocks::ONE).value(), MODULUS - 1);
//! ```

use core::ops::{Add, Mul, Neg, Sub};

use crate::{Field, PrimeField, TwoAdicField};

/// The modulus `2^64 - 2^32 + 1`.
///
/// ```
/// use arrabbiata_field::goldilocks::MODULUS;
/// assert_eq!(u128::from(MODULUS), (1u128 << 64) - (1u128 << 32) + 1);
/// ```
pub const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// An element of the Goldilocks field.
///
/// Invariant: the inner value is the canonical representative, in
/// `[0, MODULUS)`. Every constructor maintains it, so equality of elements
/// is equality of representatives.
///
/// ```
/// use arrabbiata_field::{Goldilocks, goldilocks::MODULUS};
/// assert_eq!(Goldilocks::new(MODULUS + 3), Goldilocks::new(3));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Goldilocks(u64);

impl Goldilocks {
    /// Reduce an arbitrary `u64` into the field.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, goldilocks::MODULUS};
    /// assert_eq!(Goldilocks::new(MODULUS), Goldilocks::ZERO);
    /// ```
    #[must_use]
    pub const fn new(value: u64) -> Self {
        // value < 2^64 < 2 * MODULUS, so one conditional subtraction reduces.
        if value >= MODULUS {
            Self(value - MODULUS)
        } else {
            Self(value)
        }
    }

    /// The canonical representative, in `[0, MODULUS)`.
    ///
    /// ```
    /// use arrabbiata_field::{Goldilocks, goldilocks::MODULUS};
    /// assert_eq!(Goldilocks::new(u64::MAX).value(), u64::MAX - MODULUS);
    /// ```
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }

    const fn reduce_u128(value: u128) -> Self {
        // The remainder is below MODULUS, so the cast does not truncate.
        #[allow(clippy::cast_possible_truncation)]
        Self((value % (MODULUS as u128)) as u64)
    }
}

impl Add for Goldilocks {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::reduce_u128(self.0 as u128 + rhs.0 as u128)
    }
}

impl Sub for Goldilocks {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            // self < rhs < MODULUS, so the sum is in (0, MODULUS).
            Self(self.0 + (MODULUS - rhs.0))
        }
    }
}

impl Mul for Goldilocks {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::reduce_u128(self.0 as u128 * rhs.0 as u128)
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    fn neg(self) -> Self {
        if self.0 == 0 {
            self
        } else {
            Self(MODULUS - self.0)
        }
    }
}

impl Field for Goldilocks {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);

    fn inverse(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            // Fermat: x^(p-2) = x^-1 for x != 0.
            Some(self.pow(MODULUS - 2))
        }
    }

    fn from_u64(n: u64) -> Self {
        Self::new(n)
    }
}

impl PrimeField for Goldilocks {
    const LIMBS: usize = 1;
    const MODULUS_BITS: u32 = 64;

    fn limb(self, i: usize) -> u64 {
        if i == 0 { self.0 } else { 0 }
    }
}

impl TwoAdicField for Goldilocks {
    const TWO_ADICITY: u32 = 32;

    // 7 generates the multiplicative group of the Goldilocks field.
    const MULTIPLICATIVE_GENERATOR: Self = Self(7);

    fn two_adic_generator(bits: u32) -> Option<Self> {
        if bits > Self::TWO_ADICITY {
            return None;
        }
        // g^((p - 1) / 2^bits) has order exactly 2^bits when g generates
        // the multiplicative group.
        Some(Self::MULTIPLICATIVE_GENERATOR.pow((MODULUS - 1) >> bits))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::laws;

    fn arb() -> impl Strategy<Value = Goldilocks> {
        prop_oneof![
            Just(Goldilocks::ZERO),
            Just(Goldilocks::ONE),
            Just(-Goldilocks::ONE),
            any::<u64>().prop_map(Goldilocks::new),
        ]
    }

    #[test]
    fn modulus_reduces_to_zero() {
        assert_eq!(Goldilocks::new(MODULUS), Goldilocks::ZERO);
        assert_eq!(Goldilocks::new(u64::MAX).value(), u64::MAX - MODULUS);
    }

    #[test]
    fn minus_one_squared_is_one() {
        assert_eq!((-Goldilocks::ONE).square(), Goldilocks::ONE);
    }

    #[test]
    fn two_adic_generator_has_exact_order() {
        use crate::TwoAdicField;
        let w = Goldilocks::two_adic_generator(32).unwrap();
        // Order exactly 2^32: w^(2^31) = -1.
        assert_eq!(w.pow(1 << 31), -Goldilocks::ONE);
        assert_eq!(w.pow(1 << 32), Goldilocks::ONE);
    }

    #[test]
    fn seven_generates_the_multiplicative_group() {
        // p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537. A generator g satisfies
        // g^((p-1)/q) != 1 for every prime q dividing p - 1.
        let g = Goldilocks::new(7);
        for q in [2u64, 3, 5, 17, 257, 65537] {
            assert_ne!(g.pow((MODULUS - 1) / q), Goldilocks::ONE, "q = {q}");
        }
        let mut cofactor = MODULUS - 1;
        for q in [2u64, 3, 5, 17, 257, 65537] {
            while cofactor.is_multiple_of(q) {
                cofactor /= q;
            }
        }
        assert_eq!(cofactor, 1, "the factorization of p - 1 is complete");
    }

    #[test]
    fn zero_has_no_inverse() {
        assert_eq!(Goldilocks::ZERO.inverse(), None);
    }

    #[test]
    fn axioms_on_constants() {
        assert!(laws::zero_ne_one::<Goldilocks>());
    }

    proptest! {
        #[test]
        fn additive_group(a in arb(), b in arb(), c in arb()) {
            prop_assert!(laws::additive_group(a, b, c));
        }

        #[test]
        fn multiplicative_monoid(a in arb(), b in arb(), c in arb()) {
            prop_assert!(laws::multiplicative_monoid(a, b, c));
        }

        #[test]
        fn distributive(a in arb(), b in arb(), c in arb()) {
            prop_assert!(laws::distributive(a, b, c));
        }

        #[test]
        fn inverse(a in arb()) {
            prop_assert!(laws::inverse(a));
        }

        #[test]
        fn pow_is_repeated_multiplication(a in arb(), e in 0u64..64) {
            let mut expected = Goldilocks::ONE;
            for _ in 0..e {
                expected = expected * a;
            }
            prop_assert_eq!(a.pow(e), expected);
        }

        #[test]
        fn canonical_representative(n in any::<u64>()) {
            prop_assert!(Goldilocks::new(n).value() < MODULUS);
        }
    }
}
