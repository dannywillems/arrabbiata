//! Quadratic extensions `F[u] / (u^2 - beta)` of a field `F`, for a
//! non-residue `beta`.
//!
//! An element `c0 + c1 u` multiplies as
//! `(a0 + a1 u)(b0 + b1 u) = (a0 b0 + beta a1 b1) + (a0 b1 + a1 b0) u`, and
//! inverts through the norm `N(a) = a0^2 - beta a1^2`, which lies in `F`:
//! `a^-1 = (a0 - a1 u) / N(a)`. The norm vanishes only at zero because
//! `beta` is not a square, which is exactly the condition for the quotient
//! to be a field.

use core::{
    fmt,
    hash::Hash,
    ops::{Add, Mul, Neg, Sub},
};

use crate::Field;

/// A quadratic extension: its base field and the non-residue `beta`.
pub trait QuadraticConfig
where
    Self: 'static + Copy + fmt::Debug + Eq + Hash + Send + Sync,
{
    /// The base field.
    type Base: Field;

    /// The non-residue `beta`, with `u^2 = beta`.
    fn non_residue() -> Self::Base;
}

/// An element `c0 + c1 u` of the extension described by `C`.
pub struct Quadratic<C>
where
    C: QuadraticConfig,
{
    c0: C::Base,
    c1: C::Base,
}

impl<C> Clone for Quadratic<C>
where
    C: QuadraticConfig,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for Quadratic<C> where C: QuadraticConfig {}

impl<C> PartialEq for Quadratic<C>
where
    C: QuadraticConfig,
{
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl<C> Eq for Quadratic<C> where C: QuadraticConfig {}

impl<C> fmt::Debug for Quadratic<C>
where
    C: QuadraticConfig,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?} + {:?} u)", self.c0, self.c1)
    }
}

impl<C> Quadratic<C>
where
    C: QuadraticConfig,
{
    /// `c0 + c1 u`.
    pub const fn new(c0: C::Base, c1: C::Base) -> Self {
        Self { c0, c1 }
    }

    /// The coordinate `c0`.
    pub const fn c0(&self) -> C::Base {
        self.c0
    }

    /// The coordinate `c1`.
    pub const fn c1(&self) -> C::Base {
        self.c1
    }

    /// The conjugate `c0 - c1 u`, the image of the non-trivial automorphism
    /// over the base field.
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self::new(self.c0, -self.c1)
    }

    /// `self * s` for `s` in the base field.
    #[must_use]
    pub fn scale(&self, s: C::Base) -> Self {
        Self::new(self.c0 * s, self.c1 * s)
    }
}

impl<C> Add for Quadratic<C>
where
    C: QuadraticConfig,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}

impl<C> Sub for Quadratic<C>
where
    C: QuadraticConfig,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}

impl<C> Mul for Quadratic<C>
where
    C: QuadraticConfig,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.c0 * rhs.c0 + C::non_residue() * self.c1 * rhs.c1,
            self.c0 * rhs.c1 + self.c1 * rhs.c0,
        )
    }
}

impl<C> Neg for Quadratic<C>
where
    C: QuadraticConfig,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.c0, -self.c1)
    }
}

impl<C> Field for Quadratic<C>
where
    C: QuadraticConfig,
{
    const ZERO: Self = Self::new(C::Base::ZERO, C::Base::ZERO);
    const ONE: Self = Self::new(C::Base::ONE, C::Base::ZERO);

    fn inverse(self) -> Option<Self> {
        let norm = self.c0 * self.c0 - C::non_residue() * self.c1 * self.c1;
        match norm.inverse() {
            Some(n) => Some(Self::new(self.c0 * n, -self.c1 * n)),
            None => None,
        }
    }

    fn from_u64(n: u64) -> Self {
        Self::new(C::Base::from_u64(n), C::Base::ZERO)
    }
}
