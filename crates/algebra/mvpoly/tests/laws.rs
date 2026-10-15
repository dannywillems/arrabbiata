//! The laws of `MVPoly`, on both representations.
//!
//! - `F[x]_(<= D)` is an abelian group, and the partial product is
//!   associative, commutative, unital and distributive wherever defined;
//! - `F[x]` has no zero divisors, so the product is defined exactly when
//!   the degrees add up to at most `D`;
//! - evaluation is a ring morphism, and homogenization is homogeneous;
//! - `Sparse` and `Dense` are isomorphic: conversion is a bijection that
//!   commutes with every operation;
//! - the prime coding is an injective monoid morphism;
//! - the cross-terms expand the folded homogeneous value, and the Hasse
//!   derivatives satisfy Taylor's formula and the Hasse-Schmidt rule.

use arrabbiata_field::Field;
use arrabbiata_mvpoly::{Dense, MVPoly, Monomial, Sparse, primes};
use arrabbiata_pasta::Fq as F;
use proptest::prelude::*;

const N: usize = 3;
const D: usize = 6;

type S = Sparse<F, N, D>;
type Dn = Dense<F, N, D>;

fn arb_f() -> impl Strategy<Value = F> {
    prop_oneof![
        1 => Just(F::ZERO),
        1 => Just(F::ONE),
        1 => Just(-F::ONE),
        6 => any::<u64>().prop_map(F::from_u64),
    ]
}

fn arb_point() -> impl Strategy<Value = [F; N]> {
    [arb_f(), arb_f(), arb_f()]
}

/// A polynomial of degree at most `max`.
fn arb_poly(max: u32) -> impl Strategy<Value = S> {
    prop::collection::vec(([0..=max, 0..=max, 0..=max], arb_f()), 0..6).prop_map(move |terms| {
        let terms = terms
            .into_iter()
            .filter(|(e, _)| e.iter().sum::<u32>() <= max)
            .map(|(e, c)| (Monomial::new(e), c))
            .collect();
        S::from_terms(terms).unwrap()
    })
}

fn binomial(n: usize, k: usize) -> F {
    let mut acc = F::ONE;
    for i in 0..k {
        acc = acc * F::from_u64((n - i) as u64);
    }
    let mut den = F::ONE;
    for i in 1..=k {
        den = den * F::from_u64(i as u64);
    }
    acc * den.inverse().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Zero is exactly the polynomial with no term, and the reference
    /// operations agree with the owned ones.
    #[test]
    fn zero_and_reference_operations(p in arb_poly(6), q in arb_poly(6)) {
        prop_assert_eq!(p.is_zero(), p.terms().is_empty());
        prop_assert_eq!(p.is_empty(), p.terms().is_empty());
        prop_assert_eq!(p.len(), p.terms().len());
        prop_assert_eq!(&p + &q, p.clone() + q.clone());
        prop_assert_eq!(&p - &q, p.clone() - q.clone());
        if !p.terms().is_empty() {
            prop_assert!(!p.is_zero());
            prop_assert!(!Dn::from_terms(p.terms()).unwrap().is_zero());
        }
    }

    #[test]
    fn additive_group(p in arb_poly(6), q in arb_poly(6), r in arb_poly(6)) {
        prop_assert_eq!((p.clone() + q.clone()) + r.clone(), p.clone() + (q.clone() + r));
        prop_assert_eq!(p.clone() + q.clone(), q.clone() + p.clone());
        prop_assert_eq!(p.clone() + S::zero(), p.clone());
        prop_assert!((p.clone() + (-p.clone())).is_zero());
        prop_assert_eq!(p.clone() - q.clone(), p + (-q));
    }

    /// Products of degree at most 2 each stay within degree 6.
    #[test]
    fn product_laws(p in arb_poly(2), q in arb_poly(2), r in arb_poly(2)) {
        let m = |a: &S, b: &S| a.checked_mul(b).unwrap();
        prop_assert_eq!(m(&m(&p, &q), &r), m(&p, &m(&q, &r)));
        prop_assert_eq!(m(&p, &q), m(&q, &p));
        prop_assert_eq!(m(&p, &S::one()), p.clone());
        prop_assert_eq!(m(&p, &(q.clone() + r.clone())), m(&p, &q) + m(&p, &r));
        prop_assert!(m(&p, &S::zero()).is_zero());
    }

    /// No zero divisors: the product exists iff the degrees fit, and then
    /// its degree is the sum.
    #[test]
    fn degree_of_a_product(p in arb_poly(4), q in arb_poly(4)) {
        let product = p.checked_mul(&q);
        match (p.degree(), q.degree()) {
            (Some(a), Some(b)) => {
                prop_assert_eq!(product.is_some(), a + b <= D);
                if let Some(pq) = product {
                    prop_assert_eq!(pq.degree(), Some(a + b));
                }
            }
            _ => prop_assert!(product.unwrap().is_zero()),
        }
    }

    /// Evaluation is a ring morphism.
    #[test]
    fn evaluation_is_a_morphism(p in arb_poly(3), q in arb_poly(3), x in arb_point(), c in arb_f()) {
        prop_assert_eq!((p.clone() + q.clone()).eval(&x), p.eval(&x) + q.eval(&x));
        prop_assert_eq!((p.clone() - q.clone()).eval(&x), p.eval(&x) - q.eval(&x));
        prop_assert_eq!(p.checked_mul(&q).unwrap().eval(&x), p.eval(&x) * q.eval(&x));
        prop_assert_eq!(p.scale(c).eval(&x), c * p.eval(&x));
        prop_assert_eq!(S::constant(c).eval(&x), c);
    }

    /// The homogenization is homogeneous of degree D, and is the
    /// polynomial at u = 1.
    #[test]
    fn homogenization(p in arb_poly(6), x in arb_point(), u in arb_f(), t in arb_f()) {
        prop_assert_eq!(p.homogeneous_eval(&x, F::ONE), p.eval(&x));
        let tx = x.map(|v| t * v);
        prop_assert_eq!(p.homogeneous_eval(&tx, t * u), t.pow(D as u64) * p.homogeneous_eval(&x, u));
    }

    /// Terms come sorted, distinct and nonzero, and rebuild the polynomial.
    #[test]
    fn terms_are_canonical(p in arb_poly(6)) {
        let terms = p.terms();
        prop_assert!(terms.windows(2).all(|w| w[0].0 < w[1].0));
        prop_assert!(terms.iter().all(|(m, c)| !c.is_zero() && m.degree() <= D));
        prop_assert_eq!(S::from_terms(terms).unwrap(), p);
    }

    /// Sparse and Dense are isomorphic: conversion round-trips and commutes
    /// with every operation.
    #[test]
    fn sparse_and_dense_are_isomorphic(p in arb_poly(3), q in arb_poly(3), x in arb_point(), c in arb_f()) {
        let to = |s: &S| Dn::from_terms(s.terms()).unwrap();
        let back = |d: &Dn| S::from_terms(d.terms()).unwrap();
        prop_assert_eq!(back(&to(&p)), p.clone());
        prop_assert_eq!(to(&p) + to(&q), to(&(p.clone() + q.clone())));
        prop_assert_eq!(to(&p) - to(&q), to(&(p.clone() - q.clone())));
        prop_assert_eq!(-to(&p), to(&(-p.clone())));
        prop_assert_eq!(to(&p).checked_mul(&to(&q)).unwrap(), to(&p.checked_mul(&q).unwrap()));
        prop_assert_eq!(to(&p).scale(c), to(&p.scale(c)));
        prop_assert_eq!(to(&p).eval(&x), p.eval(&x));
        prop_assert_eq!(to(&p).degree(), p.degree());
        prop_assert_eq!(to(&p).is_homogeneous(), p.is_homogeneous());
        prop_assert_eq!(to(&p).is_multilinear(), p.is_multilinear());
    }

    /// The prime coding is an injective monoid morphism into the integers.
    #[test]
    fn prime_coding_is_a_monoid_morphism(a in [0u32..4, 0..4, 0..4], b in [0u32..4, 0..4, 0..4]) {
        let p = primes::first(N);
        let (m, n) = (Monomial::new(a), Monomial::new(b));
        let code = |m: &Monomial<N>| primes::code(m, &p).unwrap();
        prop_assert_eq!(code(&m.mul(&n)), code(&m) * code(&n));
        prop_assert_eq!(code(&Monomial::one()), 1);
        prop_assert_eq!(primes::decode::<N>(code(&m), &p), Some(m));
        prop_assert_eq!(code(&m) == code(&n), m == n);
    }

    /// The cross-terms are the coefficients of the folded homogeneous
    /// value, a polynomial in r of degree D, checked at a random r.
    #[test]
    fn cross_terms_expand_the_fold(
        p in arb_poly(6),
        x1 in arb_point(), x2 in arb_point(),
        u1 in arb_f(), u2 in arb_f(), r in arb_f(),
    ) {
        let t = p.cross_terms(&x1, &x2, u1, u2);
        prop_assert_eq!(t.len(), D + 1);
        prop_assert_eq!(t[0], p.homogeneous_eval(&x1, u1));
        prop_assert_eq!(t[D], p.homogeneous_eval(&x2, u2));
        let folded: [F; N] = std::array::from_fn(|i| x1[i] + r * x2[i]);
        let value = t.iter().rev().fold(F::ZERO, |acc, c| acc * r + *c);
        prop_assert_eq!(value, p.homogeneous_eval(&folded, u1 + r * u2));
    }

    /// The scaled cross-terms are those of s P, with s folded as well.
    #[test]
    fn scaled_cross_terms(
        p in arb_poly(6),
        x1 in arb_point(), x2 in arb_point(),
        u1 in arb_f(), u2 in arb_f(), s1 in arb_f(), s2 in arb_f(), r in arb_f(),
    ) {
        let t = p.cross_terms_scaled(&x1, &x2, u1, u2, s1, s2);
        prop_assert_eq!(t.len(), D + 2);
        let folded: [F; N] = std::array::from_fn(|i| x1[i] + r * x2[i]);
        let value = t.iter().rev().fold(F::ZERO, |acc, c| acc * r + *c);
        prop_assert_eq!(value, (s1 + r * s2) * p.homogeneous_eval(&folded, u1 + r * u2));
    }

    /// Taylor's formula with Hasse derivatives, in any characteristic:
    /// P(x + r y) = sum_k r^k (H_y^(k) P)(x).
    #[test]
    fn taylor_formula(p in arb_poly(6), x in arb_point(), y in arb_point(), r in arb_f()) {
        let shifted: [F; N] = std::array::from_fn(|i| x[i] + r * y[i]);
        let mut sum = F::ZERO;
        for k in 0..=D {
            let h = p.hasse(k, &y);
            prop_assert!(h.degree().is_none_or(|d| d + k <= D));
            sum = sum + r.pow(k as u64) * h.eval(&x);
        }
        prop_assert_eq!(sum, p.eval(&shifted));
        prop_assert_eq!(p.hasse(0, &y), p.clone());
        prop_assert!(p.hasse(D + 1, &y).is_zero());
    }

    /// The Hasse-Schmidt rule: H^(i) H^(j) = binomial(i + j, i) H^(i + j).
    #[test]
    fn hasse_schmidt_rule(p in arb_poly(6), y in arb_point(), i in 0usize..4, j in 0usize..4) {
        let left = p.hasse(j, &y).hasse(i, &y);
        let right = p.hasse(i + j, &y).scale(binomial(i + j, i));
        prop_assert_eq!(left, right);
    }
}

/// `binomial(N + D, D)` monomials, with distinct sorted codes, each of
/// degree at most D.
#[test]
fn dense_shape() {
    let codes = Dense::<F, 3, 6>::codes().unwrap();
    // binomial(9, 6) = 84.
    assert_eq!(codes.len(), 84);
    assert!(codes.windows(2).all(|w| w[0] < w[1]));
    let p = primes::first(3);
    assert!(
        codes
            .iter()
            .all(|c| primes::decode::<3>(*c, &p).is_some_and(|m| m.degree() <= 6))
    );
    assert_eq!(Dense::<F, 5, 4>::codes().unwrap().len(), 126);
}

/// A shape whose largest code overflows a u128 has no dense polynomial:
/// in one variable, `x^200` has code `2^200 > 2^128`, while `x^127` fits.
#[test]
fn dense_overflow_is_reported() {
    assert!(Dense::<F, 1, 200>::codes().is_none());
    assert!(Dense::<F, 1, 200>::from_terms(vec![]).is_none());
    assert_eq!(
        Dense::<F, 1, 127>::codes().unwrap().last(),
        Some(&(1u128 << 127))
    );
}

#[test]
fn primes_are_prime() {
    let p = primes::first(200);
    assert_eq!(p[0], 2);
    assert_eq!(p[99], 541);
    assert!(p.windows(2).all(|w| w[0] < w[1]));
    for q in &p {
        assert!(
            (2..*q).take_while(|d| d * d <= *q).all(|d| q % d != 0),
            "{q}"
        );
    }
}

/// Terms of degree above D are accepted when they cancel, in both
/// representations alike.
#[test]
fn cancelling_terms_above_the_bound() {
    let high = Monomial::new([4, 4, 0]);
    let terms = vec![
        (high, F::ONE),
        (high, -F::ONE),
        (Monomial::new([1, 0, 0]), F::ONE),
    ];
    let s = S::from_terms(terms.clone()).unwrap();
    let d = Dn::from_terms(terms).unwrap();
    assert_eq!(s.terms(), d.terms());
    assert_eq!(s.degree(), Some(1));
    assert!(S::from_terms(vec![(high, F::ONE)]).is_none());
    assert!(Dn::from_terms(vec![(high, F::ONE)]).is_none());
}
