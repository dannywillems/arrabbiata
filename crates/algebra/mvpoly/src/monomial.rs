//! Variables and monomials.
//!
//! A monomial `x_0^(e_0) ... x_(N-1)^(e_(N-1))` is its exponent vector
//! `e` in `N^N`. Monomials form the free commutative monoid on `N`
//! generators: multiplication adds exponent vectors, and `1` is the zero
//! vector. Every polynomial is a finite `F`-linear combination of them.

use core::cmp::Ordering;

/// A variable index below `N`.
///
/// ```
/// use arrabbiata_mvpoly::Var;
/// assert_eq!(Var::<3>::new::<2>().index(), 2);
/// assert!(Var::<3>::try_new(3).is_none());
/// ```
///
/// An index out of range is rejected at compile time:
///
/// ```compile_fail
/// use arrabbiata_mvpoly::Var;
/// let _ = Var::<3>::new::<3>();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Var<const N: usize>(usize);

impl<const N: usize> Var<N> {
    /// The variable `I`, checked against `N` at compile time.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Var;
    /// assert_eq!(Var::<2>::new::<0>().index(), 0);
    /// ```
    #[must_use]
    pub const fn new<const I: usize>() -> Self {
        const { assert!(I < N, "variable index out of range") };
        Self(I)
    }

    /// The variable `index`, or `None` if it is not below `N`.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Var;
    /// assert!(Var::<2>::try_new(1).is_some());
    /// ```
    #[must_use]
    pub const fn try_new(index: usize) -> Option<Self> {
        if index < N { Some(Self(index)) } else { None }
    }

    /// The index, below `N`.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Var;
    /// assert_eq!(Var::<4>::new::<3>().index(), 3);
    /// ```
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A monomial in `N` variables, as its exponent vector.
///
/// Monomials are ordered by the graded lexicographic order: by total degree
/// first, then lexicographically on the exponents. It is a monomial order
/// (a well-order compatible with multiplication), so a polynomial has a
/// canonical list of terms.
///
/// ```
/// use arrabbiata_mvpoly::Monomial;
/// let xy = Monomial::new([1, 1]);
/// let x2 = Monomial::new([2, 0]);
/// assert_eq!(xy.degree(), 2);
/// assert_eq!(xy.mul(&x2), Monomial::new([3, 1]));
/// // Same degree: lexicographic, so x^2 > x y.
/// assert!(x2 > xy);
/// assert!(Monomial::new([0, 3]) > x2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Monomial<const N: usize>([u32; N]);

impl<const N: usize> Monomial<N> {
    /// The monomial with exponents `exponents`.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert_eq!(Monomial::new([2, 0, 1]).exponents(), &[2, 0, 1]);
    /// ```
    #[must_use]
    pub const fn new(exponents: [u32; N]) -> Self {
        Self(exponents)
    }

    /// The monomial `1`.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert_eq!(Monomial::<3>::one().degree(), 0);
    /// ```
    #[must_use]
    pub const fn one() -> Self {
        Self([0; N])
    }

    /// The monomial `x_v`.
    ///
    /// ```
    /// use arrabbiata_mvpoly::{Monomial, Var};
    /// assert_eq!(Monomial::variable(Var::<3>::new::<1>()), Monomial::new([0, 1, 0]));
    /// ```
    #[must_use]
    pub const fn variable(v: Var<N>) -> Self {
        let mut e = [0; N];
        e[v.index()] = 1;
        Self(e)
    }

    /// The exponent vector.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert_eq!(Monomial::new([4]).exponents()[0], 4);
    /// ```
    #[must_use]
    pub const fn exponents(&self) -> &[u32; N] {
        &self.0
    }

    /// The total degree, the sum of the exponents.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert_eq!(Monomial::new([2, 0, 3]).degree(), 5);
    /// ```
    #[must_use]
    pub fn degree(&self) -> usize {
        self.0.iter().map(|e| *e as usize).sum()
    }

    /// The product, adding exponents.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert_eq!(Monomial::new([1, 2]).mul(&Monomial::new([3, 0])), Monomial::new([4, 2]));
    /// ```
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut e = self.0;
        for (a, b) in e.iter_mut().zip(other.0) {
            *a += b;
        }
        Self(e)
    }

    /// Whether every exponent is at most 1.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert!(Monomial::new([1, 0, 1]).is_multilinear());
    /// assert!(!Monomial::new([2, 0, 0]).is_multilinear());
    /// ```
    #[must_use]
    pub fn is_multilinear(&self) -> bool {
        self.0.iter().all(|e| *e <= 1)
    }

    /// Whether `self` divides `other`, exponent by exponent.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Monomial;
    /// assert!(Monomial::new([1, 0]).divides(&Monomial::new([2, 1])));
    /// assert!(!Monomial::new([0, 2]).divides(&Monomial::new([2, 1])));
    /// ```
    #[must_use]
    pub fn divides(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0).all(|(a, b)| *a <= b)
    }
}

impl<const N: usize> PartialOrd for Monomial<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for Monomial<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.degree()
            .cmp(&other.degree())
            .then_with(|| self.0.cmp(&other.0))
    }
}

/// Every monomial in `N` variables of degree at most `d`, in increasing
/// graded lexicographic order. There are `binomial(N + d, d)` of them.
pub(crate) fn all<const N: usize>(d: usize) -> alloc::vec::Vec<Monomial<N>> {
    let mut out = alloc::vec::Vec::new();
    let mut e = [0u32; N];
    fn go<const N: usize>(
        i: usize,
        left: usize,
        e: &mut [u32; N],
        out: &mut alloc::vec::Vec<Monomial<N>>,
    ) {
        if i == N {
            out.push(Monomial(*e));
            return;
        }
        for k in 0..=left {
            e[i] = k as u32;
            go(i + 1, left - k, e, out);
        }
        e[i] = 0;
    }
    go(0, d, &mut e, &mut out);
    out.sort_unstable();
    out
}
