//! The dense representation: every coefficient of degree at most `D`,
//! indexed through the primes.

use alloc::{vec, vec::Vec};
use core::ops::{Add, Neg, Sub};

use arrabbiata_field::Field;

use crate::{
    MVPoly,
    monomial::{self, Monomial},
    primes,
    sparse::Sparse,
};

/// A polynomial in `N` variables of degree at most `D`, as the vector of
/// the coefficients of all `binomial(N + D, D)` monomials of degree at most
/// `D`, the monomial `x^e` at the rank of its integer
/// `p_0^(e_0) ... p_(N-1)^(e_(N-1))` among those of all such monomials
/// (the representation of proof-systems' `mvpoly::prime`).
///
/// The integers are checked: a shape whose largest integer does not fit a
/// `u128` has no dense polynomial, and [`MVPoly::from_terms`] returns
/// `None` for it.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_mvpoly::{Dense, MVPoly, Monomial, Var};
/// use arrabbiata_pasta::Fq as F;
/// // 2 variables of degree at most 2: 1, x0, x1, x0^2, x0 x1, x1^2,
/// // indexed by 1, 2, 3, 4, 6, 9.
/// assert_eq!(Dense::<F, 2, 2>::codes(), Some(vec![1, 2, 3, 4, 6, 9]));
/// let x = Dense::<F, 2, 2>::variable(Var::new::<0>()).unwrap();
/// let y = Dense::<F, 2, 2>::variable(Var::new::<1>()).unwrap();
/// let xy = x.checked_mul(&y).unwrap();
/// assert_eq!(xy.coefficient(&Monomial::new([1, 1])), F::ONE);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dense<F, const N: usize, const D: usize> {
    coefficients: Vec<F>,
    codes: Vec<u128>,
}

impl<F, const N: usize, const D: usize> Dense<F, N, D>
where
    F: Field,
{
    /// The sorted integers of all monomials of degree at most `D`, or
    /// `None` if one overflows a `u128`.
    ///
    /// ```
    /// use arrabbiata_mvpoly::Dense;
    /// use arrabbiata_pasta::Fq as F;
    /// assert_eq!(Dense::<F, 1, 3>::codes(), Some(vec![1, 2, 4, 8]));
    /// assert_eq!(Dense::<F, 3, 3>::codes().unwrap().len(), 20);
    /// ```
    #[must_use]
    pub fn codes() -> Option<Vec<u128>> {
        let primes = primes::first(N);
        let mut codes = Vec::new();
        for m in monomial::all::<N>(D) {
            codes.push(primes::code(&m, &primes)?);
        }
        codes.sort_unstable();
        Some(codes)
    }

    /// The coefficient vector, in the order of [`Dense::codes`].
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_mvpoly::{Dense, MVPoly};
    /// use arrabbiata_pasta::Fq as F;
    /// let one = Dense::<F, 2, 1>::one();
    /// assert_eq!(one.coefficients(), &[F::ONE, F::ZERO, F::ZERO]);
    /// ```
    #[must_use]
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    fn pointwise(&self, other: &Self, sign: F) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&other.coefficients)
            .map(|(a, b)| *a + sign * *b)
            .collect();
        Self {
            coefficients,
            codes: self.codes.clone(),
        }
    }
}

impl<F, const N: usize, const D: usize> MVPoly<F, N, D> for Dense<F, N, D>
where
    F: Field,
{
    fn terms(&self) -> Vec<(Monomial<N>, F)> {
        let primes = primes::first(N);
        let mut terms: Vec<(Monomial<N>, F)> = self
            .codes
            .iter()
            .zip(&self.coefficients)
            .filter(|(_, c)| !c.is_zero())
            .filter_map(|(code, c)| primes::decode(*code, &primes).map(|m| (m, *c)))
            .collect();
        terms.sort_unstable_by_key(|t| t.0);
        terms
    }

    fn from_terms(terms: Vec<(Monomial<N>, F)>) -> Option<Self> {
        // Sum repeated monomials and drop zeros first, as `Sparse` does, so
        // that terms of degree above `D` which cancel are accepted.
        let normal = Sparse::<F, N, D>::from_terms(terms)?;
        let codes = Self::codes()?;
        let primes = primes::first(N);
        let mut coefficients = vec![F::ZERO; codes.len()];
        for (m, c) in normal.terms() {
            let i = codes.binary_search(&primes::code(&m, &primes)?).ok()?;
            coefficients[i] = c;
        }
        Some(Self {
            coefficients,
            codes,
        })
    }
}

impl<F, const N: usize, const D: usize> Add for Dense<F, N, D>
where
    F: Field,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.pointwise(&other, F::ONE)
    }
}

impl<F, const N: usize, const D: usize> Sub for Dense<F, N, D>
where
    F: Field,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.pointwise(&other, -F::ONE)
    }
}

impl<F, const N: usize, const D: usize> Neg for Dense<F, N, D>
where
    F: Field,
{
    type Output = Self;

    fn neg(self) -> Self {
        let coefficients = self.coefficients.iter().map(|c| -*c).collect();
        Self {
            coefficients,
            codes: self.codes,
        }
    }
}
