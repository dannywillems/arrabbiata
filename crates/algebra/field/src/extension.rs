//! Quadratic extensions `F[u] / (u^2 - beta)` of a field `F`, for a
//! non-residue `beta`.
//!
//! An element `c0 + c1 u` multiplies as
//! `(a0 + a1 u)(b0 + b1 u) = (a0 b0 + beta a1 b1) + (a0 b1 + a1 b0) u`, and
//! inverts through the norm `N(a) = a0^2 - beta a1^2`, which lies in `F`:
//! `a^-1 = (a0 - a1 u) / N(a)`. The norm vanishes only at zero because
//! `beta` is not a square, which is exactly the condition for the quotient
//! to be a field.
//!
//! ```
//! use arrabbiata_field::{Field, extension::{Quadratic, QuadraticConfig}, montgomery::{MontField, MontgomeryConfig}};
//!
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
//! struct F7;
//! impl MontgomeryConfig<1> for F7 {
//!     const MODULUS: [u64; 1] = [7];
//!     const GENERATOR: u64 = 3;
//! }
//! /// F_49 = F_7[u] / (u^2 + 1), since -1 is not a square modulo 7.
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
//! struct F49;
//! impl QuadraticConfig for F49 {
//!     type Base = MontField<F7, 1>;
//!     fn non_residue() -> Self::Base {
//!         -MontField::ONE
//!     }
//! }
//! type E = Quadratic<F49>;
//! let u = E::new(MontField::ZERO, MontField::ONE);
//! assert_eq!(u * u, -E::ONE);
//! assert_eq!(u * u.inverse().unwrap(), E::ONE);
//! ```

use core::{
    fmt,
    hash::Hash,
    ops::{Add, Mul, Neg, Sub},
};

use crate::Field;

/// A quadratic extension: its base field and the non-residue `beta`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks, extension::QuadraticConfig};
/// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// struct G2;
/// impl QuadraticConfig for G2 {
///     type Base = Goldilocks;
///     // 7 is not a square modulo the Goldilocks prime.
///     fn non_residue() -> Goldilocks {
///         Goldilocks::new(7)
///     }
/// }
/// assert_eq!(G2::non_residue(), Goldilocks::new(7));
/// ```
pub trait QuadraticConfig
where
    Self: 'static + Copy + fmt::Debug + Eq + Hash + Send + Sync,
{
    /// The base field.
    type Base: Field;

    /// The non-residue `beta`, with `u^2 = beta`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, extension::QuadraticConfig};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct G2;
    /// impl QuadraticConfig for G2 {
    ///     type Base = Goldilocks;
    ///     fn non_residue() -> Goldilocks {
    ///         Goldilocks::new(7)
    ///     }
    /// }
    /// let _ = G2::non_residue();
    /// ```
    fn non_residue() -> Self::Base;
}

/// An element `c0 + c1 u` of the extension described by `C`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks, extension::{Quadratic, QuadraticConfig}};
/// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// struct G2;
/// impl QuadraticConfig for G2 {
///     type Base = Goldilocks;
///     fn non_residue() -> Goldilocks {
///         Goldilocks::new(7)
///     }
/// }
/// let a = Quadratic::<G2>::new(Goldilocks::new(1), Goldilocks::new(2));
/// assert_eq!(a.c1(), Goldilocks::new(2));
/// ```
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
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, extension::{Quadratic, QuadraticConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct G2;
    /// impl QuadraticConfig for G2 {
    ///     type Base = Goldilocks;
    ///     fn non_residue() -> Goldilocks {
    ///         Goldilocks::new(7)
    ///     }
    /// }
    /// let one = Quadratic::<G2>::new(Goldilocks::ONE, Goldilocks::ZERO);
    /// assert_eq!(one, Quadratic::ONE);
    /// ```
    pub const fn new(c0: C::Base, c1: C::Base) -> Self {
        Self { c0, c1 }
    }

    /// The coordinate `c0`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, extension::{Quadratic, QuadraticConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct G2;
    /// impl QuadraticConfig for G2 {
    ///     type Base = Goldilocks;
    ///     fn non_residue() -> Goldilocks {
    ///         Goldilocks::new(7)
    ///     }
    /// }
    /// assert_eq!(Quadratic::<G2>::ONE.c0(), Goldilocks::ONE);
    /// ```
    pub const fn c0(&self) -> C::Base {
        self.c0
    }

    /// The coordinate `c1`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, extension::{Quadratic, QuadraticConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct G2;
    /// impl QuadraticConfig for G2 {
    ///     type Base = Goldilocks;
    ///     fn non_residue() -> Goldilocks {
    ///         Goldilocks::new(7)
    ///     }
    /// }
    /// assert_eq!(Quadratic::<G2>::ONE.c1(), Goldilocks::ZERO);
    /// ```
    pub const fn c1(&self) -> C::Base {
        self.c1
    }

    /// The conjugate `c0 - c1 u`, the image of the non-trivial automorphism
    /// over the base field.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, extension::{Quadratic, QuadraticConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct G2;
    /// impl QuadraticConfig for G2 {
    ///     type Base = Goldilocks;
    ///     fn non_residue() -> Goldilocks {
    ///         Goldilocks::new(7)
    ///     }
    /// }
    /// let a = Quadratic::<G2>::new(Goldilocks::new(3), Goldilocks::new(5));
    /// // a times its conjugate is the norm, in the base field.
    /// assert_eq!((a * a.conjugate()).c1(), Goldilocks::ZERO);
    /// ```
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self::new(self.c0, -self.c1)
    }

    /// `self * s` for `s` in the base field.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks, extension::{Quadratic, QuadraticConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct G2;
    /// impl QuadraticConfig for G2 {
    ///     type Base = Goldilocks;
    ///     fn non_residue() -> Goldilocks {
    ///         Goldilocks::new(7)
    ///     }
    /// }
    /// let a = Quadratic::<G2>::new(Goldilocks::new(1), Goldilocks::new(2));
    /// assert_eq!(a.scale(Goldilocks::new(3)), a + a + a);
    /// ```
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
