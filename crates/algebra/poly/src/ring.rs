//! Commutative `F`-algebras that expressions can be evaluated in.
//!
//! Evaluation is not only done in `F`. To compute cross-terms, the folding
//! prover evaluates constraints in the polynomial ring `F[r]`, assigning to
//! each witness variable the line `w_acc + r * w_fresh`; the cross-terms are
//! then the coefficients of the result. The same code serves both, because
//! evaluation is defined once over an arbitrary commutative `F`-algebra.
//!
//! An algebra is passed as a *value* implementing [`Algebra`] (a record of
//! operations, in the style of an ML module or a Lean structure instance),
//! not as a bound on the element type. This sidesteps coherence conflicts
//! between `F` and `F[r]`, and maps directly onto a Lean `CommRing`
//! instance on the element type.

use alloc::{vec, vec::Vec};
use core::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use arrabbiata_field::Field;

/// The operations of a commutative `F`-algebra `A`, with structure map
/// [`Algebra::embed`] from `F`.
///
/// Laws: `(A, add, zero, neg)` is an abelian group, `(A, mul, one)` a
/// commutative monoid, `mul` distributes over `add`, and `embed` is a ring
/// morphism.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
///
/// // Code generic over the algebra runs in F and in F[r] alike.
/// fn square_plus_one<A>(ring: &A, x: &A::Elem) -> A::Elem
/// where
///     A: Algebra<F>,
/// {
///     ring.add(&ring.mul(x, x), &ring.one())
/// }
/// assert_eq!(square_plus_one(&Scalars::new(), &F::new(3)), F::new(10));
/// let p = square_plus_one(&Polynomials::new(), &UniPoly::line(F::ZERO, F::ONE));
/// assert_eq!(p.evaluate(F::new(3)), F::new(10));
/// ```
pub trait Algebra<F>
where
    F: Field,
{
    /// The elements of the algebra.
    type Elem: Clone;

    /// The additive identity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(Algebra::<F>::zero(&Scalars::new()), F::ZERO);
    /// ```
    fn zero(&self) -> Self::Elem;

    /// The multiplicative identity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(Algebra::<F>::one(&Scalars::new()), F::ONE);
    /// ```
    fn one(&self) -> Self::Elem;

    /// The structure map `F -> A`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(Polynomials::new().embed(F::new(4)).coefficient(0), F::new(4));
    /// ```
    fn embed(&self, c: F) -> Self::Elem;

    /// Sum.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(Scalars::new().add(&F::new(2), &F::new(3)), F::new(5));
    /// ```
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    /// Product.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(Scalars::new().mul(&F::new(2), &F::new(3)), F::new(6));
    /// ```
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    /// Additive inverse.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(Scalars::new().neg(&F::ONE), -F::ONE);
    /// ```
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
}

/// The field `F` as an algebra over itself.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
/// let ring = Scalars::<F>::new();
/// assert_eq!(ring.mul(&F::new(6), &F::new(7)), F::new(42));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalars<F>(PhantomData<F>);

impl<F> Scalars<F> {
    /// The algebra `F`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// let _ring = Scalars::<F>::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F> Algebra<F> for Scalars<F>
where
    F: Field,
{
    type Elem = F;

    fn zero(&self) -> F {
        F::ZERO
    }

    fn one(&self) -> F {
        F::ONE
    }

    fn embed(&self, c: F) -> F {
        c
    }

    fn add(&self, a: &F, b: &F) -> F {
        *a + *b
    }

    fn mul(&self, a: &F, b: &F) -> F {
        *a * *b
    }

    fn neg(&self, a: &F) -> F {
        -*a
    }
}

/// A univariate polynomial over `F`, by its coefficients, constant first.
///
/// Trailing zero coefficients are allowed; [`UniPoly::coefficient`] reads
/// past the end as zero, so representations differing only by trailing
/// zeros denote the same polynomial.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
/// let p = UniPoly::new(vec![F::ONE, F::ZERO, F::ZERO]);
/// assert_eq!(p.coefficient(5), F::ZERO);
/// assert_eq!(p.evaluate(F::new(9)), F::ONE);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniPoly<F> {
    coefficients: Vec<F>,
}

impl<F> UniPoly<F>
where
    F: Field,
{
    /// The polynomial with the given coefficients, constant first.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// // 1 + 2r
    /// let p = UniPoly::new(vec![F::new(1), F::new(2)]);
    /// assert_eq!(p.evaluate(F::new(3)), F::new(7));
    /// ```
    #[must_use]
    pub const fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    /// The affine polynomial `a + b * r`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// let l = UniPoly::line(F::new(1), F::new(2));
    /// assert_eq!(l.evaluate(F::new(10)), F::new(21));
    /// ```
    #[must_use]
    pub fn line(a: F, b: F) -> Self {
        Self::new(vec![a, b])
    }

    /// The coefficient of `r^k`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// let l = UniPoly::line(F::new(1), F::new(2));
    /// assert_eq!(l.coefficient(1), F::new(2));
    /// assert_eq!(l.coefficient(2), F::ZERO);
    /// ```
    #[must_use]
    pub fn coefficient(&self, k: usize) -> F {
        match self.coefficients.get(k) {
            Some(c) => *c,
            None => F::ZERO,
        }
    }

    /// The number of stored coefficients (an upper bound on degree + 1).
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert_eq!(UniPoly::line(F::ONE, F::ONE).len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Whether no coefficient is stored.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// assert!(UniPoly::<F>::new(vec![]).is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The value at `x`, by Horner's rule.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// // 1 + r + r^2 at r = 2
    /// let p = UniPoly::new(vec![F::ONE, F::ONE, F::ONE]);
    /// assert_eq!(p.evaluate(F::new(2)), F::new(7));
    /// ```
    #[must_use]
    pub fn evaluate(&self, x: F) -> F {
        let mut acc = F::ZERO;
        let mut i = self.coefficients.len();
        while i > 0 {
            i -= 1;
            acc = acc * x + self.coefficients[i];
        }
        acc
    }
}

impl<F> UniPoly<F>
where
    F: Field,
{
    /// The constant polynomial `c`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::UniPoly;
    /// assert_eq!(UniPoly::constant(F::new(4)).evaluate(F::new(9)), F::new(4));
    /// ```
    #[must_use]
    pub fn constant(c: F) -> Self {
        Self::new(vec![c])
    }

    /// The monomial `X`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::UniPoly;
    /// assert_eq!(UniPoly::<F>::x().evaluate(F::new(9)), F::new(9));
    /// ```
    #[must_use]
    pub fn x() -> Self {
        Self::new(vec![F::ZERO, F::ONE])
    }

    /// The stored coefficients, constant first.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::UniPoly;
    /// assert_eq!(UniPoly::line(F::new(1), F::new(2)).coefficients(), &[F::new(1), F::new(2)]);
    /// ```
    #[must_use]
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    /// The polynomial `p(w X)`: coefficient `k` scaled by `w^k`.
    ///
    /// On a multiplicative domain with generator `w`, dilation by `w` shifts
    /// the evaluations by one position, which is how a protocol refers to
    /// the next row.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::UniPoly;
    /// let p = UniPoly::new(vec![F::new(1), F::new(2), F::new(3)]);
    /// assert_eq!(p.dilate(F::new(5)).evaluate(F::new(7)), p.evaluate(F::new(35)));
    /// ```
    #[must_use]
    pub fn dilate(&self, w: F) -> Self {
        let mut out = Vec::with_capacity(self.coefficients.len());
        let mut power = F::ONE;
        let mut k = 0;
        while k < self.coefficients.len() {
            out.push(self.coefficients[k] * power);
            power = power * w;
            k += 1;
        }
        Self::new(out)
    }

    /// The polynomial `c * self`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::UniPoly;
    /// let p = UniPoly::line(F::new(1), F::new(2)).scale(F::new(3));
    /// assert_eq!(p.coefficient(1), F::new(6));
    /// ```
    #[must_use]
    pub fn scale(&self, c: F) -> Self {
        let mut out = self.coefficients.clone();
        let mut i = 0;
        while i < out.len() {
            out[i] = out[i] * c;
            i += 1;
        }
        Self::new(out)
    }

    /// The degree, or `None` for the zero polynomial.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::UniPoly;
    /// assert_eq!(UniPoly::new(vec![F::ONE, F::ZERO]).degree(), Some(0));
    /// assert_eq!(UniPoly::<F>::new(vec![F::ZERO]).degree(), None);
    /// ```
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        let mut i = self.coefficients.len();
        while i > 0 {
            i -= 1;
            if !self.coefficients[i].is_zero() {
                return Some(i);
            }
        }
        None
    }

    /// The quotient of `self` by `X^n - 1`, or `None` if the division is not
    /// exact (equivalently, if `self` does not vanish on every `n`-th root of
    /// unity, when `F` has them).
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::UniPoly;
    /// // (X^2 - 1)(X + 3) = X^3 + 3X^2 - X - 3.
    /// let p = UniPoly::new(vec![-F::new(3), -F::ONE, F::new(3), F::ONE]);
    /// let q = p.divide_by_vanishing(2).unwrap();
    /// assert_eq!((q.coefficient(0), q.coefficient(1)), (F::new(3), F::ONE));
    /// assert!(UniPoly::new(vec![F::ONE]).divide_by_vanishing(2).is_none());
    /// ```
    #[must_use]
    pub fn divide_by_vanishing(&self, n: usize) -> Option<Self> {
        if n == 0 {
            return None;
        }
        // X^n = 1 modulo X^n - 1: fold the coefficient of X^i (i >= n) onto
        // X^(i - n), recording it in the quotient.
        let mut remainder = self.coefficients.clone();
        let len = remainder.len();
        if len <= n {
            let mut i = 0;
            while i < len {
                if !remainder[i].is_zero() {
                    return None;
                }
                i += 1;
            }
            return Some(Self::new(Vec::new()));
        }
        let mut quotient = vec![F::ZERO; len - n];
        let mut i = len;
        while i > n {
            i -= 1;
            let c = remainder[i];
            quotient[i - n] = c;
            remainder[i - n] = remainder[i - n] + c;
            remainder[i] = F::ZERO;
        }
        let mut j = 0;
        while j < n {
            if !remainder[j].is_zero() {
                return None;
            }
            j += 1;
        }
        Some(Self::new(quotient))
    }
}

impl<F> Add for &UniPoly<F>
where
    F: Field,
{
    type Output = UniPoly<F>;

    fn add(self, rhs: Self) -> UniPoly<F> {
        Polynomials::new().add(self, rhs)
    }
}

impl<F> Sub for &UniPoly<F>
where
    F: Field,
{
    type Output = UniPoly<F>;

    fn sub(self, rhs: Self) -> UniPoly<F> {
        let ring = Polynomials::new();
        ring.add(self, &ring.neg(rhs))
    }
}

impl<F> Mul for &UniPoly<F>
where
    F: Field,
{
    type Output = UniPoly<F>;

    fn mul(self, rhs: Self) -> UniPoly<F> {
        Polynomials::new().mul(self, rhs)
    }
}

/// The polynomial ring `F[r]` as an `F`-algebra.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
/// let ring = Polynomials::new();
/// // (1 + r)^2 = 1 + 2r + r^2
/// let l = UniPoly::line(F::ONE, F::ONE);
/// let sq = ring.mul(&l, &l);
/// assert_eq!(sq.coefficient(1), F::new(2));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Polynomials<F>(PhantomData<F>);

impl<F> Polynomials<F> {
    /// The algebra `F[r]`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Algebra, Polynomials, Scalars, UniPoly};
    /// let _ring = Polynomials::<F>::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F> Algebra<F> for Polynomials<F>
where
    F: Field,
{
    type Elem = UniPoly<F>;

    fn zero(&self) -> UniPoly<F> {
        UniPoly::new(Vec::new())
    }

    fn one(&self) -> UniPoly<F> {
        UniPoly::new(vec![F::ONE])
    }

    fn embed(&self, c: F) -> UniPoly<F> {
        UniPoly::new(vec![c])
    }

    fn add(&self, a: &UniPoly<F>, b: &UniPoly<F>) -> UniPoly<F> {
        let len = if a.len() >= b.len() { a.len() } else { b.len() };
        let mut out = Vec::with_capacity(len);
        let mut k = 0;
        while k < len {
            out.push(a.coefficient(k) + b.coefficient(k));
            k += 1;
        }
        UniPoly::new(out)
    }

    fn mul(&self, a: &UniPoly<F>, b: &UniPoly<F>) -> UniPoly<F> {
        if a.is_empty() || b.is_empty() {
            return self.zero();
        }
        let len = a.len() + b.len() - 1;
        let mut out = vec![F::ZERO; len];
        let mut i = 0;
        while i < a.len() {
            let mut j = 0;
            while j < b.len() {
                out[i + j] = out[i + j] + a.coefficients[i] * b.coefficients[j];
                j += 1;
            }
            i += 1;
        }
        UniPoly::new(out)
    }

    fn neg(&self, a: &UniPoly<F>) -> UniPoly<F> {
        let mut out = Vec::with_capacity(a.len());
        let mut k = 0;
        while k < a.len() {
            out.push(-a.coefficients[k]);
            k += 1;
        }
        UniPoly::new(out)
    }
}
