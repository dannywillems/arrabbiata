//! Prime fields in Montgomery representation, generic over the modulus.
//!
//! A field is a [`MontgomeryConfig`]: a modulus `p` (odd, as `N` limbs) and
//! a small quadratic non-residue. Everything else (`R = 2^(64 N) mod p`,
//! `R^2`, `-p^-1 mod 2^64`, the two-adicity) is derived from the modulus by
//! `const fn`, so the modulus is the only number written by hand, and it is
//! checked independently (by the order of a curve, by primality tests).
//!
//! An element `a` is stored as `a R mod p`. Multiplication is Montgomery's
//! coarsely integrated operand scanning (CIOS): `(aR)(bR) R^-1 = (ab)R`.
//! Every operation is total and written with plain loops (no `unsafe`, no
//! closures), so it stays in the subset of Rust that Aeneas extracts.
//!
//! ```
//! use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
//!
//! /// The field of order 2^61 - 1 (a Mersenne prime), on one limb.
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
//! struct M61;
//! impl MontgomeryConfig<1> for M61 {
//!     const MODULUS: [u64; 1] = [(1 << 61) - 1];
//!     const GENERATOR: u64 = 37;
//! }
//! type F = MontField<M61, 1>;
//!
//! let a = F::from_u64(1 << 40);
//! assert_eq!(a * a, F::from_u64(1 << 19)); // 2^80 = 2^19 mod 2^61 - 1
//! assert_eq!(a * a.inverse().unwrap(), F::ONE);
//! ```

use core::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
};

use crate::{Field, PrimeField, Sqrt, TwoAdicField, bigint};

/// The parameters of a prime field on `N` 64-bit limbs.
///
/// ```
/// use arrabbiata_field::montgomery::MontgomeryConfig;
/// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// struct F7;
/// impl MontgomeryConfig<1> for F7 {
///     const MODULUS: [u64; 1] = [7];
///     const GENERATOR: u64 = 3;
/// }
/// assert_eq!(F7::MODULUS, [7]);
/// ```
pub trait MontgomeryConfig<const N: usize>
where
    Self: 'static + Copy + fmt::Debug + Eq + Hash + Send + Sync,
{
    /// The modulus `p`, an odd prime, least significant limb first.
    const MODULUS: [u64; N];

    /// A quadratic non-residue modulo `p`, used as the multiplicative
    /// generator of [`TwoAdicField`] (its `(p - 1) / 2^s`-th power is a
    /// primitive `2^s`-th root of unity) and for square roots.
    const GENERATOR: u64;
}

/// An element of the prime field described by `C`, in Montgomery form.
///
/// Invariant: the limbs are `a R mod p`, fully reduced (below `p`).
///
/// ```
/// use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
/// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// struct F13;
/// impl MontgomeryConfig<1> for F13 {
///     const MODULUS: [u64; 1] = [13];
///     const GENERATOR: u64 = 2;
/// }
/// let x = MontField::<F13, 1>::from_u64(20);
/// assert_eq!(x.to_canonical_limbs(), [7]);
/// ```
pub struct MontField<C, const N: usize> {
    limbs: [u64; N],
    config: PhantomData<C>,
}

impl<C, const N: usize> Clone for MontField<C, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, const N: usize> Copy for MontField<C, N> {}

impl<C, const N: usize> PartialEq for MontField<C, N> {
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}

impl<C, const N: usize> Eq for MontField<C, N> {}

impl<C, const N: usize> Hash for MontField<C, N> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.limbs.hash(state);
    }
}

impl<C, const N: usize> fmt::Debug for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.to_canonical_limbs();
        write!(f, "0x")?;
        let mut i = N;
        while i > 0 {
            i -= 1;
            write!(f, "{:016x}", limbs[i])?;
        }
        Ok(())
    }
}

impl<C, const N: usize> MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    /// `-p^-1 mod 2^64`.
    const INV: u64 = bigint::neg_inv_mod_2_64(C::MODULUS[0]);

    /// `R = 2^(64 N) mod p`, the Montgomery form of 1.
    const R: [u64; N] = bigint::pow2_mod(&C::MODULUS);

    /// `R^2 mod p`, to enter Montgomery form.
    const R2: [u64; N] = bigint::shift_mod(&Self::R, &C::MODULUS);

    /// Montgomery multiplication (CIOS): `a b R^-1 mod p`, for `a, b < p`.
    const fn mont_mul(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let p = &C::MODULUS;
        let mut t = [0u64; N];
        let mut t_n = 0u64;
        let mut i = 0;
        while i < N {
            // t += a * b[i]
            let mut carry = 0u64;
            let mut j = 0;
            while j < N {
                let s = t[j] as u128 + a[j] as u128 * b[i] as u128 + carry as u128;
                t[j] = s as u64;
                carry = (s >> 64) as u64;
                j += 1;
            }
            let s = t_n as u128 + carry as u128;
            t_n = s as u64;
            let t_n1 = (s >> 64) as u64;
            // t = (t + m p) / 2^64, with m chosen so the low limb vanishes.
            let m = t[0].wrapping_mul(Self::INV);
            let s = t[0] as u128 + m as u128 * p[0] as u128;
            let mut carry = (s >> 64) as u64;
            let mut j = 1;
            while j < N {
                let s = t[j] as u128 + m as u128 * p[j] as u128 + carry as u128;
                t[j - 1] = s as u64;
                carry = (s >> 64) as u64;
                j += 1;
            }
            let s = t_n as u128 + carry as u128;
            t[N - 1] = s as u64;
            t_n = t_n1 + (s >> 64) as u64;
            i += 1;
        }
        if t_n != 0 || bigint::geq(&t, p) {
            bigint::sub(&t, p).0
        } else {
            t
        }
    }

    /// The element with the given Montgomery limbs. The limbs must be below
    /// `p`; this is not checked.
    const fn from_montgomery(limbs: [u64; N]) -> Self {
        Self {
            limbs,
            config: PhantomData,
        }
    }

    /// The element whose canonical representative has the given limbs, or
    /// `None` if they are not below `p`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct F13;
    /// impl MontgomeryConfig<1> for F13 {
    ///     const MODULUS: [u64; 1] = [13];
    ///     const GENERATOR: u64 = 2;
    /// }
    /// type F = MontField<F13, 1>;
    /// assert_eq!(F::from_canonical_limbs([5]), Some(F::from_u64(5)));
    /// assert_eq!(F::from_canonical_limbs([13]), None);
    /// ```
    #[must_use]
    pub const fn from_canonical_limbs(limbs: [u64; N]) -> Option<Self> {
        if bigint::geq(&limbs, &C::MODULUS) {
            None
        } else {
            Some(Self::from_montgomery(Self::mont_mul(&limbs, &Self::R2)))
        }
    }

    /// The canonical representative in `[0, p)`, as limbs.
    ///
    /// ```
    /// use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct F13;
    /// impl MontgomeryConfig<1> for F13 {
    ///     const MODULUS: [u64; 1] = [13];
    ///     const GENERATOR: u64 = 2;
    /// }
    /// assert_eq!((-MontField::<F13, 1>::ONE).to_canonical_limbs(), [12]);
    /// ```
    #[must_use]
    pub const fn to_canonical_limbs(&self) -> [u64; N] {
        Self::mont_mul(&self.limbs, &bigint::one())
    }

    /// The image of `n`, as a `const fn`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct F13;
    /// impl MontgomeryConfig<1> for F13 {
    ///     const MODULUS: [u64; 1] = [13];
    ///     const GENERATOR: u64 = 2;
    /// }
    /// const FIVE: MontField<F13, 1> = MontField::from_u64_const(18);
    /// assert_eq!(FIVE, MontField::from_u64(5));
    /// ```
    #[must_use]
    pub const fn from_u64_const(n: u64) -> Self {
        let mut limbs = [0u64; N];
        limbs[0] = n;
        while bigint::geq(&limbs, &C::MODULUS) {
            limbs = bigint::sub(&limbs, &C::MODULUS).0;
        }
        Self::from_montgomery(Self::mont_mul(&limbs, &Self::R2))
    }

    /// `self^e` for an exponent given as little-endian limbs of any length.
    ///
    /// ```
    /// use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct F13;
    /// impl MontgomeryConfig<1> for F13 {
    ///     const MODULUS: [u64; 1] = [13];
    ///     const GENERATOR: u64 = 2;
    /// }
    /// let two = MontField::<F13, 1>::from_u64(2);
    /// // Fermat: 2^12 = 1 mod 13.
    /// assert_eq!(two.pow_limbs(&[12]), MontField::ONE);
    /// ```
    #[must_use]
    pub fn pow_limbs(self, exponent: &[u64]) -> Self {
        let mut acc = Self::ONE;
        let mut i = exponent.len();
        while i > 0 {
            i -= 1;
            let mut bit: u32 = 64;
            while bit > 0 {
                bit -= 1;
                acc = acc.square();
                if (exponent[i] >> bit) & 1 == 1 {
                    acc = acc * self;
                }
            }
        }
        acc
    }

    /// Euler's criterion: `self^((p - 1) / 2)`, which is 1 on non-zero
    /// squares, -1 on non-squares, and 0 on 0.
    ///
    /// ```
    /// use arrabbiata_field::{Field, montgomery::{MontField, MontgomeryConfig}};
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// struct F13;
    /// impl MontgomeryConfig<1> for F13 {
    ///     const MODULUS: [u64; 1] = [13];
    ///     const GENERATOR: u64 = 2;
    /// }
    /// type F = MontField<F13, 1>;
    /// assert_eq!(F::from_u64(4).legendre(), F::ONE);
    /// assert_eq!(F::from_u64(2).legendre(), -F::ONE);
    /// ```
    #[must_use]
    pub fn legendre(self) -> Self {
        let half = bigint::shr1(&bigint::sub(&C::MODULUS, &bigint::one()).0);
        self.pow_limbs(&half)
    }
}

impl<C, const N: usize> Add for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_montgomery(bigint::add_mod(&self.limbs, &rhs.limbs, &C::MODULUS))
    }
}

impl<C, const N: usize> Sub for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let (d, borrow) = bigint::sub(&self.limbs, &rhs.limbs);
        if borrow {
            Self::from_montgomery(bigint::add(&d, &C::MODULUS).0)
        } else {
            Self::from_montgomery(d)
        }
    }
}

impl<C, const N: usize> Mul for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_montgomery(Self::mont_mul(&self.limbs, &rhs.limbs))
    }
}

impl<C, const N: usize> Neg for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    type Output = Self;

    fn neg(self) -> Self {
        if bigint::is_zero(&self.limbs) {
            self
        } else {
            Self::from_montgomery(bigint::sub(&C::MODULUS, &self.limbs).0)
        }
    }
}

impl<C, const N: usize> Field for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    const ZERO: Self = Self::from_montgomery([0; N]);
    const ONE: Self = Self::from_montgomery(Self::R);

    fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // Fermat: a^(p - 2) = a^-1.
        let two = {
            let mut t = [0u64; N];
            t[0] = 2;
            t
        };
        Some(self.pow_limbs(&bigint::sub(&C::MODULUS, &two).0))
    }

    fn from_u64(n: u64) -> Self {
        Self::from_u64_const(n)
    }
}

impl<C, const N: usize> PrimeField for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    const LIMBS: usize = N;
    const MODULUS_BITS: u32 = bigint::num_bits(&C::MODULUS);

    fn limb(self, i: usize) -> u64 {
        if i < N {
            self.to_canonical_limbs()[i]
        } else {
            0
        }
    }

    fn from_limbs(limbs: &[u64]) -> Option<Self> {
        let mut array = [0u64; N];
        let mut i = 0;
        while i < limbs.len() {
            if i < N {
                array[i] = limbs[i];
            } else if limbs[i] != 0 {
                return None;
            }
            i += 1;
        }
        Self::from_canonical_limbs(array)
    }
}

impl<C, const N: usize> TwoAdicField for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    const TWO_ADICITY: u32 = bigint::two_adicity(&C::MODULUS);
    const MULTIPLICATIVE_GENERATOR: Self = Self::from_u64_const(C::GENERATOR);

    fn two_adic_generator(bits: u32) -> Option<Self> {
        if bits > Self::TWO_ADICITY {
            return None;
        }
        // g^((p - 1) / 2^bits) has order exactly 2^bits for a non-residue g,
        // since then g^((p - 1) / 2) = -1.
        let mut e = bigint::sub(&C::MODULUS, &bigint::one()).0;
        let mut i = 0;
        while i < bits {
            e = bigint::shr1(&e);
            i += 1;
        }
        Some(Self::MULTIPLICATIVE_GENERATOR.pow_limbs(&e))
    }
}

impl<C, const N: usize> Sqrt for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    /// Tonelli-Shanks, with the configured non-residue.
    fn sqrt(self) -> Option<Self> {
        if self.is_zero() {
            return Some(self);
        }
        if self.legendre() != Self::ONE {
            return None;
        }
        // p - 1 = q 2^s with q odd.
        let s = Self::TWO_ADICITY;
        let mut q = bigint::sub(&C::MODULUS, &bigint::one()).0;
        let mut i = 0;
        while i < s {
            q = bigint::shr1(&q);
            i += 1;
        }
        let q_plus_one_half = bigint::shr1(&bigint::add(&q, &bigint::one()).0);
        let mut m = s;
        let mut c = Self::MULTIPLICATIVE_GENERATOR.pow_limbs(&q);
        let mut t = self.pow_limbs(&q);
        let mut x = self.pow_limbs(&q_plus_one_half);
        while t != Self::ONE {
            // The least i with t^(2^i) = 1; it is below m.
            let mut i = 0;
            let mut t2 = t;
            while t2 != Self::ONE {
                t2 = t2.square();
                i += 1;
            }
            let mut b = c;
            let mut k = 0;
            while k + i + 1 < m {
                b = b.square();
                k += 1;
            }
            x = x * b;
            c = b.square();
            t = t * c;
            m = i;
        }
        Some(x)
    }
}
