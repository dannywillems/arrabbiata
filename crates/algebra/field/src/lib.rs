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

#![no_std]

extern crate alloc;

use core::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

pub mod bigint;
pub mod extension;
pub mod goldilocks;
pub mod laws;
pub mod montgomery;
pub mod vector;

pub use goldilocks::Goldilocks;
pub use montgomery::{MontField, MontgomeryConfig};
pub use vector::{inner_product, powers};

/// A finite field.
///
/// The arithmetic operators are inherited from `core::ops` so that circuit
/// code reads like mathematics (`a * b + c`). The trait adds the constants,
/// inversion, and the embedding of integers.
///
///
/// Code generic over the field is written once and runs on any instance:
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
    const ZERO: Self;

    /// The multiplicative identity.
    const ONE: Self;

    /// The multiplicative inverse, or `None` for zero.
    ///
    /// Returning an `Option` rather than panicking keeps the operation total,
    /// which is what the Lean model needs: Mathlib's `x^-1` (`Inv.inv`) is total with
    /// `0^-1 = 0`, and the `Option` makes the junk value impossible to use by
    /// accident on the Rust side.
    fn inverse(self) -> Option<Self>;

    /// The image of `n` under the unique ring morphism `Z -> F`, restricted
    /// to the non-negative integers below `2^64`.
    fn from_u64(n: u64) -> Self;

    /// Whether this is the additive identity.
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    /// `self * self`.
    #[must_use]
    fn square(self) -> Self {
        self * self
    }

    /// `self^exp`, by square-and-multiply from the most significant bit.
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
pub trait PrimeField
where
    Self: Field,
{
    /// The number of 64-bit limbs of the canonical representative.
    const LIMBS: usize;

    /// The number of bits of the modulus.
    const MODULUS_BITS: u32;

    /// The `i`-th 64-bit limb of the canonical representative in
    /// `[0, p)`, least significant first. Zero when `i >= LIMBS`.
    fn limb(self, i: usize) -> u64;

    /// The element whose canonical representative has the given limbs
    /// (least significant first), or `None` if that integer is not below
    /// the modulus. Limbs past [`PrimeField::LIMBS`] must be zero.
    fn from_limbs(limbs: &[u64]) -> Option<Self>;
}

/// A field with a multiplicative subgroup of order `2^TWO_ADICITY`, hence
/// roots of unity of every order `2^k` for `k <= TWO_ADICITY`: what a radix-2
/// FFT (NTT) over the field needs.
pub trait TwoAdicField
where
    Self: Field,
{
    /// The largest `k` such that `2^k` divides `p - 1`.
    const TWO_ADICITY: u32;

    /// A generator of the multiplicative group, used to build cosets of the
    /// subgroups of roots of unity.
    const MULTIPLICATIVE_GENERATOR: Self;

    /// A primitive `2^bits`-th root of unity, or `None` when
    /// `bits > TWO_ADICITY`.
    ///
    /// The roots are compatible: the square of the generator for `bits` is
    /// the generator for `bits - 1`.
    fn two_adic_generator(bits: u32) -> Option<Self>;
}

/// Square roots.
pub trait Sqrt
where
    Self: Field,
{
    /// A square root of `self`, or `None` if `self` is not a square. Which
    /// of the two roots is returned is unspecified.
    fn sqrt(self) -> Option<Self>;
}
