//! The sparse representation: the nonzero terms, keyed by monomial.

use alloc::{collections::BTreeMap, vec::Vec};
use core::ops::{Add, Neg, Sub};

use arrabbiata_field::Field;

use crate::{MVPoly, monomial::Monomial};

/// A polynomial in `N` variables of degree at most `D`, as the map from
/// each monomial of nonzero coefficient to that coefficient.
///
/// No coefficient stored is zero, so two equal polynomials have equal maps
/// and `==` is equality of polynomials.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
/// use arrabbiata_pasta::Fq as F;
/// let x = Sparse::<F, 2, 2>::variable(Var::new::<0>()).unwrap();
/// let y = Sparse::<F, 2, 2>::variable(Var::new::<1>()).unwrap();
/// // (x + y) (x - y) = x^2 - y^2.
/// let p = (x.clone() + y.clone()).checked_mul(&(x.clone() - y.clone())).unwrap();
/// let q = x.checked_mul(&x).unwrap() - y.checked_mul(&y).unwrap();
/// assert_eq!(p, q);
/// assert_eq!(p.terms().len(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sparse<F, const N: usize, const D: usize> {
    terms: BTreeMap<Monomial<N>, F>,
}

impl<F, const N: usize, const D: usize> Sparse<F, N, D>
where
    F: Field,
{
    /// The number of nonzero terms.
    ///
    /// ```
    /// use arrabbiata_mvpoly::{MVPoly, Sparse};
    /// use arrabbiata_pasta::Fq as F;
    /// assert_eq!(Sparse::<F, 3, 2>::zero().len(), 0);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether there is no term (the polynomial is zero).
    ///
    /// ```
    /// use arrabbiata_mvpoly::{MVPoly, Sparse};
    /// use arrabbiata_pasta::Fq as F;
    /// assert!(Sparse::<F, 3, 2>::zero().is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    fn combine(&self, other: &Self, sign: F) -> Self {
        let mut terms = self.terms.clone();
        for (m, c) in &other.terms {
            let sum = terms.get(m).copied().unwrap_or(F::ZERO) + sign * *c;
            if sum.is_zero() {
                terms.remove(m);
            } else {
                terms.insert(*m, sum);
            }
        }
        Self { terms }
    }
}

impl<F, const N: usize, const D: usize> MVPoly<F, N, D> for Sparse<F, N, D>
where
    F: Field,
{
    fn terms(&self) -> Vec<(Monomial<N>, F)> {
        self.terms.iter().map(|(m, c)| (*m, *c)).collect()
    }

    fn from_terms(terms: Vec<(Monomial<N>, F)>) -> Option<Self> {
        let mut map: BTreeMap<Monomial<N>, F> = BTreeMap::new();
        for (m, c) in terms {
            let sum = map.get(&m).copied().unwrap_or(F::ZERO) + c;
            map.insert(m, sum);
        }
        map.retain(|_, c| !c.is_zero());
        if map.keys().any(|m| m.degree() > D) {
            return None;
        }
        Some(Self { terms: map })
    }
}

impl<F, const N: usize, const D: usize> Add for Sparse<F, N, D>
where
    F: Field,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.combine(&other, F::ONE)
    }
}

impl<F, const N: usize, const D: usize> Add for &Sparse<F, N, D>
where
    F: Field,
{
    type Output = Sparse<F, N, D>;

    fn add(self, other: Self) -> Sparse<F, N, D> {
        self.combine(other, F::ONE)
    }
}

impl<F, const N: usize, const D: usize> Sub for Sparse<F, N, D>
where
    F: Field,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.combine(&other, -F::ONE)
    }
}

impl<F, const N: usize, const D: usize> Sub for &Sparse<F, N, D>
where
    F: Field,
{
    type Output = Sparse<F, N, D>;

    fn sub(self, other: Self) -> Sparse<F, N, D> {
        self.combine(other, -F::ONE)
    }
}

impl<F, const N: usize, const D: usize> Neg for Sparse<F, N, D>
where
    F: Field,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self::zero().combine(&self, -F::ONE)
    }
}
