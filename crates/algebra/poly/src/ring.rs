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

use arrabbiata_field::{Field, powers};

/// The operations of a commutative `F`-algebra `A`, with structure map
/// [`Algebra::embed`] from `F`.
///
/// Laws: `(A, add, zero, neg)` is an abelian group, `(A, mul, one)` a
/// commutative monoid, `mul` distributes over `add`, and `embed` is a ring
/// morphism.
pub trait Algebra<F>
where
    F: Field,
{
    /// The elements of the algebra.
    type Elem: Clone;

    /// The additive identity.
    fn zero(&self) -> Self::Elem;

    /// The multiplicative identity.
    fn one(&self) -> Self::Elem;

    /// The structure map `F -> A`.
    fn embed(&self, c: F) -> Self::Elem;

    /// Sum.
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    /// Product.
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;

    /// Additive inverse.
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
}

/// The field `F` as an algebra over itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalars<F>(PhantomData<F>);

impl<F> Scalars<F> {
    /// The algebra `F`.
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniPoly<F> {
    coefficients: Vec<F>,
}

impl<F> UniPoly<F>
where
    F: Field,
{
    /// The polynomial with the given coefficients, constant first.
    #[must_use]
    pub const fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    /// The affine polynomial `a + b * r`.
    #[must_use]
    pub fn line(a: F, b: F) -> Self {
        Self::new(vec![a, b])
    }

    /// The coefficient of `r^k`.
    #[must_use]
    pub fn coefficient(&self, k: usize) -> F {
        match self.coefficients.get(k) {
            Some(c) => *c,
            None => F::ZERO,
        }
    }

    /// The number of stored coefficients (an upper bound on degree + 1).
    #[must_use]
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Whether no coefficient is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The value at `x`, by Horner's rule.
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
    #[must_use]
    pub fn constant(c: F) -> Self {
        Self::new(vec![c])
    }

    /// The monomial `X`.
    #[must_use]
    pub fn x() -> Self {
        Self::new(vec![F::ZERO, F::ONE])
    }

    /// The stored coefficients, constant first.
    #[must_use]
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    /// The polynomial `p(w X)`: coefficient `k` scaled by `w^k`.
    ///
    /// On a multiplicative domain with generator `w`, dilation by `w` shifts
    /// the evaluations by one position, which is how a protocol refers to
    /// the next row.
    #[must_use]
    pub fn dilate(&self, w: F) -> Self {
        let w_powers = powers(w, self.coefficients.len());
        let mut out = Vec::with_capacity(self.coefficients.len());
        let mut k = 0;
        while k < self.coefficients.len() {
            out.push(self.coefficients[k] * w_powers[k]);
            k += 1;
        }
        Self::new(out)
    }

    /// The polynomial `c * self`.
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
#[derive(Clone, Copy, Debug, Default)]
pub struct Polynomials<F>(PhantomData<F>);

impl<F> Polynomials<F> {
    /// The algebra `F[r]`.
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
