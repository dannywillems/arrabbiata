//! Finite fields for Arrabbiata.
//!
//! The rest of the workspace is generic over the [`Field`] trait defined here.
//! It is deliberately small: the operations of a field and nothing else. The
//! reason is formal verification. When the generic crates are extracted to
//! Lean with Charon/Aeneas, `Field` becomes a structure of operations, and the
//! Lean library instantiates it with Mathlib's `Field` (for instance `ZMod p`
//! with `p` prime). Every lemma proved over the abstract structure then holds
//! for every concrete field, including the ones in this crate.
//!
//! Concrete fields live in submodules and depend on nothing outside `core`.
//!
//! # Laws
//!
//! An implementation must satisfy the field axioms. They cannot be stated in
//! Rust's type system, so they are checked by property tests (see
//! [`laws`]) and assumed by the Lean side as the fields of the `Field`
//! structure:
//!
//! - `(F, +, 0)` is an abelian group, with inverse `neg`;
//! - `(F \ {0}, *, 1)` is an abelian group, with inverse `inverse`;
//! - `*` distributes over `+`;
//! - `0 != 1`.
//!
//! # Example
//!
//! ```
//! use arrabbiata_field::{Field, Goldilocks};
//!
//! let a = Goldilocks::new(3);
//! let b = Goldilocks::new(5);
//! assert_eq!(a * b + Goldilocks::ONE, Goldilocks::new(16));
//! assert_eq!(a * a.inverse().unwrap(), Goldilocks::ONE);
//! ```

#![no_std]

use core::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

pub mod bigint;
pub mod extension;
pub mod goldilocks;
pub mod laws;
pub mod montgomery;

pub use goldilocks::Goldilocks;
pub use montgomery::{MontField, MontgomeryConfig};

/// A finite field.
///
/// The arithmetic operators are inherited from `core::ops` so that circuit
/// code reads like mathematics (`a * b + c`). The trait adds the constants,
/// inversion, and the embedding of integers.
///
/// # Example
///
/// Code generic over the field is written once and runs on any instance:
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks};
///
/// fn horner<F>(coefficients: &[F], x: F) -> F
/// where
///     F: Field,
/// {
///     coefficients.iter().rev().fold(F::ZERO, |acc, c| acc * x + *c)
/// }
///
/// // 1 + 2x + 3x^2 at x = 2 is 17.
/// let p = [1, 2, 3].map(Goldilocks::new);
/// assert_eq!(horner(&p, Goldilocks::new(2)), Goldilocks::new(17));
/// ```
pub trait Field
where
    Self: Copy
        + Eq
        + Debug
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>,
{
    /// The additive identity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// let x = Goldilocks::new(7);
    /// assert_eq!(x + Goldilocks::ZERO, x);
    /// ```
    const ZERO: Self;

    /// The multiplicative identity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// let x = Goldilocks::new(7);
    /// assert_eq!(x * Goldilocks::ONE, x);
    /// ```
    const ONE: Self;

    /// The multiplicative inverse, or `None` for zero.
    ///
    /// Returning an `Option` rather than panicking keeps the operation total,
    /// which is what the Lean model needs: Mathlib's `x^-1` (`Inv.inv`) is total with
    /// `0^-1 = 0`, and the `Option` makes the junk value impossible to use by
    /// accident on the Rust side.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// let x = Goldilocks::new(7);
    /// assert_eq!(x * x.inverse().unwrap(), Goldilocks::ONE);
    /// assert_eq!(Goldilocks::ZERO.inverse(), None);
    /// ```
    fn inverse(self) -> Option<Self>;

    /// The image of `n` under the unique ring morphism `Z -> F`, restricted
    /// to the non-negative integers below `2^64`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// assert_eq!(Goldilocks::from_u64(2) + Goldilocks::from_u64(3), Goldilocks::from_u64(5));
    /// ```
    fn from_u64(n: u64) -> Self;

    /// Whether this is the additive identity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// assert!(Goldilocks::ZERO.is_zero());
    /// assert!(!Goldilocks::ONE.is_zero());
    /// ```
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    /// `self * self`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// assert_eq!(Goldilocks::new(9).square(), Goldilocks::new(81));
    /// ```
    #[must_use]
    fn square(self) -> Self {
        self * self
    }

    /// `self^exp`, by square-and-multiply from the most significant bit.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks};
    /// assert_eq!(Goldilocks::new(2).pow(10), Goldilocks::new(1024));
    /// assert_eq!(Goldilocks::new(5).pow(0), Goldilocks::ONE);
    /// ```
    #[must_use]
    fn pow(self, exp: u64) -> Self {
        let mut acc = Self::ONE;
        let mut bit: u32 = 64;
        while bit > 0 {
            bit -= 1;
            acc = acc.square();
            if (exp >> bit) & 1 == 1 {
                acc = acc * self;
            }
        }
        acc
    }
}

/// A field whose elements can be listed as a canonical sequence of 64-bit
/// limbs, least significant first.
///
/// This is what a transcript needs to absorb an element of one field into a
/// sponge over another, which is the situation on a cycle of curves.
///
/// ```
/// use arrabbiata_field::{Goldilocks, PrimeField};
///
/// let x = Goldilocks::new(42);
/// assert_eq!(x.limb(0), 42);
/// assert_eq!(x.limb(1), 0);
/// assert_eq!(Goldilocks::LIMBS, 1);
/// ```
pub trait PrimeField
where
    Self: Field,
{
    /// The number of 64-bit limbs of the canonical representative.
    ///
    /// ```
    /// use arrabbiata_field::{Goldilocks, PrimeField};
    /// assert_eq!(Goldilocks::LIMBS, 1);
    /// ```
    const LIMBS: usize;

    /// The number of bits of the modulus.
    ///
    /// ```
    /// use arrabbiata_field::{Goldilocks, PrimeField};
    /// assert_eq!(Goldilocks::MODULUS_BITS, 64);
    /// ```
    const MODULUS_BITS: u32;

    /// The `i`-th 64-bit limb of the canonical representative in
    /// `[0, p)`, least significant first. Zero when `i >= LIMBS`.
    ///
    /// ```
    /// use arrabbiata_field::{Goldilocks, PrimeField};
    /// assert_eq!(Goldilocks::new(u64::MAX).limb(0), u64::MAX - arrabbiata_field::goldilocks::MODULUS);
    /// ```
    fn limb(self, i: usize) -> u64;

    /// The element whose canonical representative has the given limbs
    /// (least significant first), or `None` if that integer is not below
    /// the modulus. Limbs past [`PrimeField::LIMBS`] must be zero.
    ///
    /// ```
    /// use arrabbiata_field::{Goldilocks, PrimeField, goldilocks::MODULUS};
    /// assert_eq!(Goldilocks::from_limbs(&[5]), Some(Goldilocks::new(5)));
    /// assert_eq!(Goldilocks::from_limbs(&[MODULUS]), None);
    /// assert_eq!(Goldilocks::from_limbs(&[1, 1]), None);
    /// ```
    fn from_limbs(limbs: &[u64]) -> Option<Self>;
}

/// A field with a multiplicative subgroup of order `2^TWO_ADICITY`, hence
/// roots of unity of every order `2^k` for `k <= TWO_ADICITY`: what a radix-2
/// FFT (NTT) over the field needs.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks, TwoAdicField};
///
/// let w = Goldilocks::two_adic_generator(3).unwrap(); // order 8
/// assert_eq!(w.pow(8), Goldilocks::ONE);
/// assert_eq!(w.pow(4), -Goldilocks::ONE);
/// ```
pub trait TwoAdicField
where
    Self: Field,
{
    /// The largest `k` such that `2^k` divides `p - 1`.
    ///
    /// ```
    /// use arrabbiata_field::{Goldilocks, TwoAdicField};
    /// assert_eq!(Goldilocks::TWO_ADICITY, 32);
    /// ```
    const TWO_ADICITY: u32;

    /// A generator of the multiplicative group, used to build cosets of the
    /// subgroups of roots of unity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, TwoAdicField};
    /// assert_ne!(Goldilocks::MULTIPLICATIVE_GENERATOR, Goldilocks::ONE);
    /// ```
    const MULTIPLICATIVE_GENERATOR: Self;

    /// A primitive `2^bits`-th root of unity, or `None` when
    /// `bits > TWO_ADICITY`.
    ///
    /// The roots are compatible: the square of the generator for `bits` is
    /// the generator for `bits - 1`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, TwoAdicField};
    /// let w2 = Goldilocks::two_adic_generator(2).unwrap();
    /// let w3 = Goldilocks::two_adic_generator(3).unwrap();
    /// assert_eq!(w3.square(), w2);
    /// assert!(Goldilocks::two_adic_generator(33).is_none());
    /// ```
    fn two_adic_generator(bits: u32) -> Option<Self>;
}

/// Square roots.
///
/// ```
/// use arrabbiata_field::{Field, Sqrt, montgomery::{MontField, MontgomeryConfig}};
/// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// struct F13;
/// impl MontgomeryConfig<1> for F13 {
///     const MODULUS: [u64; 1] = [13];
///     const GENERATOR: u64 = 2;
/// }
/// type F = MontField<F13, 1>;
/// let r = F::from_u64(10).sqrt().unwrap(); // 6^2 = 36 = 10 mod 13
/// assert_eq!(r * r, F::from_u64(10));
/// assert_eq!(F::from_u64(2).sqrt(), None);
/// ```
pub trait Sqrt
where
    Self: Field,
{
    /// A square root of `self`, or `None` if `self` is not a square. Which
    /// of the two roots is returned is unspecified.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Sqrt, montgomery::{MontField, MontgomeryConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct F13;
    /// impl MontgomeryConfig<1> for F13 {
    ///     const MODULUS: [u64; 1] = [13];
    ///     const GENERATOR: u64 = 2;
    /// }
    /// assert_eq!(MontField::<F13, 1>::ZERO.sqrt(), Some(MontField::ZERO));
    /// ```
    fn sqrt(self) -> Option<Self>;
}
