//! Multivariate polynomials in normal form, and the cross-terms of folding.
//!
//! A polynomial of [`MVPoly`]`<F, N, D>` lives in `F[x_0, ..., x_(N-1)]`
//! with total degree at most `D`: the filtered piece `F[x]_(<= D)` of the
//! polynomial ring. Addition stays in it; multiplication may leave it, so
//! it is the checked [`MVPoly::checked_mul`], which returns `None` rather
//! than a polynomial of the wrong type.
//!
//! Two representations, with the same values:
//!
//! - [`Sparse`]: the nonzero terms, keyed by monomial;
//! - [`Dense`]: every coefficient of degree at most `D`, the monomial
//!   `x^e` indexed by the integer `p_0^(e_0) ... p_(N-1)^(e_(N-1))`, the
//!   product of the first `N` primes to the exponents. By unique
//!   factorization this is an isomorphism of the monoid of monomials onto
//!   the multiplicative monoid of the `p_(N-1)`-smooth integers built from
//!   those primes: multiplying monomials is multiplying their indices.
//!
//! # Cross-terms
//!
//! Folding two instances `(u_1, x_1)` and `(u_2, x_2)` of a relation `P`
//! with a challenge `r` evaluates the homogenization of `P` to degree `D`,
//! `P^D(u, x) = sum_e c_e u^(D - |e|) x^e`, at the folded point:
//!
//! ```text
//! P^D(u_1 + r u_2, x_1 + r x_2) = sum_(k = 0)^D r^k T_k,
//! ```
//!
//! with `T_0 = P^D(u_1, x_1)`, `T_D = P^D(u_2, x_2)`, and the cross-terms
//! `T_1, ..., T_(D-1)` in between ([`MVPoly::cross_terms`]). The
//! expansion holds in any characteristic: `T_k` is a Hasse derivative of
//! `P^D` in the direction `(u_2, x_2)` ([`MVPoly::hasse`]), which needs no
//! division by `k!`. This is the computation of the Arrabbiata note
//! "Behind Nova: cross-terms computation for high degree gates"
//! (<https://hackmd.io/qq_Awc1AR3ywzkruE4Wq9Q>), as in the `mvpoly` crate
//! of o1-labs/proof-systems.
//!
//! Constraints written as [`arrabbiata_poly::Expr`] convert with
//! [`from_expr`], under a [`Layout`] giving one variable per witness column
//! and rotation (the current and the next row, in the layout of
//! Arrabbiata's circuits).

#![no_std]

extern crate alloc;

pub mod dense;
pub mod expr;
pub mod monomial;
pub mod primes;
pub mod sparse;

use alloc::{vec, vec::Vec};
use core::{
    fmt::Debug,
    ops::{Add, Neg, Sub},
};

use arrabbiata_field::Field;

pub use crate::{
    dense::Dense,
    expr::{Layout, from_expr},
    monomial::{Monomial, Var},
    sparse::Sparse,
};

/// A polynomial in `N` variables over `F`, of total degree at most `D`.
///
/// A representation provides its terms ([`MVPoly::terms`]) and builds a
/// polynomial from terms ([`MVPoly::from_terms`]); everything else is
/// defined once, here, from these two. The laws (a commutative ring for the
/// partial product, evaluation a ring morphism, the cross-term expansion)
/// therefore hold for every representation at once, and the two in this
/// crate are tested to be isomorphic.
pub trait MVPoly<F, const N: usize, const D: usize>:
    Sized
    + Clone
    + PartialEq
    + Eq
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
where
    F: Field,
{
    /// The nonzero terms, in increasing monomial order.
    fn terms(&self) -> Vec<(Monomial<N>, F)>;

    /// The polynomial with the given terms (summed when a monomial
    /// repeats), or `None` if a term of nonzero coefficient has degree
    /// above `D`.
    fn from_terms(terms: Vec<(Monomial<N>, F)>) -> Option<Self>;

    /// The zero polynomial.
    #[must_use]
    fn zero() -> Self {
        Self::constant(F::ZERO)
    }

    /// The polynomial `1`.
    #[must_use]
    fn one() -> Self {
        Self::constant(F::ONE)
    }

    /// The constant polynomial `c`.
    #[must_use]
    fn constant(c: F) -> Self {
        match Self::from_terms(vec![(Monomial::one(), c)]) {
            Some(p) => p,
            None => unreachable!("a constant has degree 0"),
        }
    }

    /// The polynomial `x_v`, or `None` when `D = 0`.
    #[must_use]
    fn variable(v: Var<N>) -> Option<Self> {
        Self::from_terms(vec![(Monomial::variable(v), F::ONE)])
    }

    /// The coefficient of the monomial `m`.
    fn coefficient(&self, m: &Monomial<N>) -> F {
        self.terms()
            .into_iter()
            .find(|(n, _)| n == m)
            .map_or(F::ZERO, |(_, c)| c)
    }

    /// Whether `self` is zero.
    fn is_zero(&self) -> bool {
        self.terms().is_empty()
    }

    /// Whether `self` is a constant (zero included).
    fn is_constant(&self) -> bool {
        self.terms().iter().all(|(m, _)| m.degree() == 0)
    }

    /// The total degree, or `None` for the zero polynomial (whose degree
    /// is `-infinity`, not 0).
    fn degree(&self) -> Option<usize> {
        self.terms().iter().map(|(m, _)| m.degree()).max()
    }

    /// Whether every term has degree exactly `D` (the zero polynomial is
    /// homogeneous of every degree).
    fn is_homogeneous(&self) -> bool {
        self.terms().iter().all(|(m, _)| m.degree() == D)
    }

    /// Whether every variable appears with exponent at most 1.
    fn is_multilinear(&self) -> bool {
        self.terms().iter().all(|(m, _)| m.is_multilinear())
    }

    /// `c` times `self`.
    #[must_use]
    fn scale(&self, c: F) -> Self {
        let terms = self.terms().into_iter().map(|(m, a)| (m, a * c)).collect();
        match Self::from_terms(terms) {
            Some(p) => p,
            None => unreachable!("scaling does not raise the degree"),
        }
    }

    /// The product, or `None` if its degree would exceed `D`.
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        let (a, b) = (self.terms(), other.terms());
        let mut terms = Vec::with_capacity(a.len() * b.len());
        for (m, c) in &a {
            for (n, d) in &b {
                terms.push((m.mul(n), *c * *d));
            }
        }
        Self::from_terms(terms)
    }

    /// The value at `x`.
    fn eval(&self, x: &[F; N]) -> F {
        self.terms()
            .iter()
            .fold(F::ZERO, |acc, (m, c)| acc + *c * monomial_value(m, x))
    }

    /// The value at `(u, x)` of the homogenization of `self` to degree `D`,
    /// `sum_e c_e u^(D - |e|) x^e`.
    fn homogeneous_eval(&self, x: &[F; N], u: F) -> F {
        self.terms().iter().fold(F::ZERO, |acc, (m, c)| {
            acc + *c * monomial_value(m, x) * u.pow((D - m.degree()) as u64)
        })
    }

    /// The coefficients `T_0, ..., T_D` of `r^k` in
    /// `P^D(u_1 + r u_2, x_1 + r x_2)`, for the homogenization `P^D` of
    /// `self` to degree `D`. `T_0` and `T_D` are the homogeneous values at
    /// the two points; `T_1, ..., T_(D-1)` are the cross-terms of folding.
    fn cross_terms(&self, x1: &[F; N], x2: &[F; N], u1: F, u2: F) -> Vec<F> {
        let mut out = vec![F::ZERO; D + 1];
        for (m, c) in self.terms() {
            let mut factor = vec![c];
            factor = mul_line_power(&factor, u1, u2, D - m.degree());
            for (i, e) in m.exponents().iter().enumerate() {
                factor = mul_line_power(&factor, x1[i], x2[i], *e as usize);
            }
            for (k, v) in factor.into_iter().enumerate() {
                out[k] = out[k] + v;
            }
        }
        out
    }

    /// The coefficients of `r^k`, `k <= D + 1`, in
    /// `(s_1 + r s_2) P^D(u_1 + r u_2, x_1 + r x_2)`: the cross-terms of
    /// `s P` for a scalar `s` folded with the instances, as a constraint's
    /// coefficient in a random combination is. The degree is `D + 1`.
    fn cross_terms_scaled(&self, x1: &[F; N], x2: &[F; N], u1: F, u2: F, s1: F, s2: F) -> Vec<F> {
        mul_line_power(&self.cross_terms(x1, x2, u1, u2), s1, s2, 1)
    }

    /// The `k`-th Hasse derivative of `self` in the direction `y`:
    /// `sum_e c_e sum_(|j| = k) prod_i binomial(e_i, j_i) y^j x^(e - j)`,
    /// so that `P(x + r y) = sum_k r^k (H_y^(k) P)(x)` in any
    /// characteristic. Its degree is at most `D - k`, and
    /// `H^(i) H^(j) = binomial(i + j, i) H^(i + j)`: the directional form
    /// of Dvir, Kopparty, Saraf and Sudan, *Extensions to the method of
    /// multiplicities*, arXiv:0901.2529, Definition 2 and Proposition 4.
    #[must_use]
    fn hasse(&self, k: usize, y: &[F; N]) -> Self {
        let mut terms = Vec::new();
        for (m, c) in self.terms() {
            for (j, coefficient) in splittings(m.exponents(), k) {
                let mut rest = *m.exponents();
                let mut factor = c * coefficient;
                for i in 0..N {
                    rest[i] -= j[i];
                    factor = factor * y[i].pow(j[i] as u64);
                }
                terms.push((Monomial::new(rest), factor));
            }
        }
        match Self::from_terms(terms) {
            Some(p) => p,
            None => unreachable!("a derivative does not raise the degree"),
        }
    }
}

/// `x^m`.
fn monomial_value<F, const N: usize>(m: &Monomial<N>, x: &[F; N]) -> F
where
    F: Field,
{
    m.exponents()
        .iter()
        .zip(x)
        .fold(F::ONE, |acc, (e, v)| acc * v.pow(*e as u64))
}

/// `p(r) (a + r b)^k`, for `p` given by its coefficients in `r`.
fn mul_line_power<F>(p: &[F], a: F, b: F, k: usize) -> Vec<F>
where
    F: Field,
{
    let mut out = p.to_vec();
    for _ in 0..k {
        let mut next = vec![F::ZERO; out.len() + 1];
        for (i, c) in out.iter().enumerate() {
            next[i] = next[i] + *c * a;
            next[i + 1] = next[i + 1] + *c * b;
        }
        out = next;
    }
    out
}

/// `binomial(n, k)` in `F`, by Pascal's rule (no division, so in any
/// characteristic).
fn binomial<F>(n: usize, k: usize) -> F
where
    F: Field,
{
    if k > n {
        return F::ZERO;
    }
    let mut row = vec![F::ONE];
    for _ in 0..n {
        let mut next = vec![F::ONE; row.len() + 1];
        for i in 1..row.len() {
            next[i] = row[i - 1] + row[i];
        }
        row = next;
    }
    row[k]
}

/// Every `j <= e` (exponent by exponent) with `|j| = k`, with
/// `prod_i binomial(e_i, j_i)`.
fn splittings<F, const N: usize>(e: &[u32; N], k: usize) -> Vec<([u32; N], F)>
where
    F: Field,
{
    let mut out = Vec::new();
    let mut j = [0u32; N];
    fn go<F: Field, const N: usize>(
        i: usize,
        left: usize,
        e: &[u32; N],
        j: &mut [u32; N],
        weight: F,
        out: &mut Vec<([u32; N], F)>,
    ) {
        if i == N {
            if left == 0 {
                out.push((*j, weight));
            }
            return;
        }
        let top = (e[i] as usize).min(left);
        for take in 0..=top {
            j[i] = take as u32;
            go(
                i + 1,
                left - take,
                e,
                j,
                weight * binomial::<F>(e[i] as usize, take),
                out,
            );
        }
        j[i] = 0;
    }
    go(0, k, e, &mut j, F::ONE, &mut out);
    out
}

/// The cross-terms of `sum_i alpha^i P_i`, the constraints `P_i` combined
/// by the powers of a combiner `alpha`, each power folded linearly with the
/// instances (`alpha_1^i + r alpha_2^i`): the coefficients of `r^k`,
/// `k <= D + 1`. This is `compute_combined_cross_terms` of proof-systems.
pub fn combined_cross_terms<F, P, const N: usize, const D: usize>(
    polys: &[P],
    x1: &[F; N],
    x2: &[F; N],
    u1: F,
    u2: F,
    alpha1: F,
    alpha2: F,
) -> Vec<F>
where
    F: Field,
    P: MVPoly<F, N, D>,
{
    let (mut s1, mut s2) = (F::ONE, F::ONE);
    let mut c1 = Vec::with_capacity(polys.len());
    let mut c2 = Vec::with_capacity(polys.len());
    for _ in polys {
        c1.push(s1);
        c2.push(s2);
        s1 = s1 * alpha1;
        s2 = s2 * alpha2;
    }
    combined_cross_terms_with(polys, x1, x2, u1, u2, &c1, &c2)
}

/// The cross-terms of `sum_i c_i P_i` for coefficients `c_i` folded
/// linearly with the instances (`c_(1, i) + r c_(2, i)`): the coefficients
/// of `r^k`, `k <= D + 1`. The combiners of a folding scheme (powers of a
/// challenge, or folded powers) are instances of this.
pub fn combined_cross_terms_with<F, P, const N: usize, const D: usize>(
    polys: &[P],
    x1: &[F; N],
    x2: &[F; N],
    u1: F,
    u2: F,
    c1: &[F],
    c2: &[F],
) -> Vec<F>
where
    F: Field,
    P: MVPoly<F, N, D>,
{
    let mut out = vec![F::ZERO; D + 2];
    for ((p, a), b) in polys.iter().zip(c1).zip(c2) {
        for (k, v) in p
            .cross_terms_scaled(x1, x2, u1, u2, *a, *b)
            .into_iter()
            .enumerate()
        {
            out[k] = out[k] + v;
        }
    }
    out
}
