//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::Field;
use arrabbiata_mvpoly::{
    Dense, Layout, MVPoly, Monomial, Sparse, Var, combined_cross_terms, combined_cross_terms_with,
    from_expr, primes, primes::first,
};
use arrabbiata_pasta::Fq as F;
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};

#[test]
fn dense_dense() {
    // 2 variables of degree at most 2: 1, x0, x1, x0^2, x0 x1, x1^2,
    // indexed by 1, 2, 3, 4, 6, 9.
    assert_eq!(Dense::<F, 2, 2>::codes(), Some(vec![1, 2, 3, 4, 6, 9]));
    let x = Dense::<F, 2, 2>::variable(Var::new::<0>()).unwrap();
    let y = Dense::<F, 2, 2>::variable(Var::new::<1>()).unwrap();
    let xy = x.checked_mul(&y).unwrap();
    assert_eq!(xy.coefficient(&Monomial::new([1, 1])), F::ONE);
}

#[test]
fn dense_dense_codes() {
    assert_eq!(Dense::<F, 1, 3>::codes(), Some(vec![1, 2, 4, 8]));
    assert_eq!(Dense::<F, 3, 3>::codes().unwrap().len(), 20);
}

#[test]
fn dense_dense_coefficients() {
    let one = Dense::<F, 2, 1>::one();
    assert_eq!(one.coefficients(), &[F::ONE, F::ZERO, F::ZERO]);
}

#[test]
fn expr_layout() {
    // Arrabbiata's layout: the 3 columns of the row, then of the next row.
    let layout = Layout::<3, 6>::current_and_next().unwrap();
    let c = WitnessColumn::new::<2>();
    assert_eq!(layout.variable(c, Rotation::CURR).unwrap().index(), 2);
    assert_eq!(layout.variable(c, Rotation::NEXT).unwrap().index(), 5);
    assert!(layout.variable(c, Rotation::PREV).is_none());
}

#[test]
fn expr_layout_new() {
    assert!(Layout::<2, 6>::new(&[Rotation::PREV, Rotation::CURR, Rotation::NEXT]).is_some());
    assert!(Layout::<2, 5>::new(&[Rotation::PREV, Rotation::CURR, Rotation::NEXT]).is_none());
    assert!(Layout::<2, 6>::new(&[Rotation::CURR, Rotation::CURR]).is_none());
}

#[test]
fn expr_layout_current_and_next() {
    assert!(Layout::<4, 8>::current_and_next().is_some());
}

#[test]
fn expr_layout_rotations() {
    let l = Layout::<1, 2>::current_and_next().unwrap();
    assert_eq!(l.rotations(), &[Rotation::CURR, Rotation::NEXT]);
}

#[test]
fn expr_layout_variable() {
    let l = Layout::<2, 4>::current_and_next().unwrap();
    assert_eq!(
        l.variable(WitnessColumn::new::<1>(), Rotation::NEXT)
            .unwrap()
            .index(),
        3
    );
}

#[test]
fn expr_layout_point() {
    let l = Layout::<1, 2>::current_and_next().unwrap();
    let column = [vec![F::from_u64(5), F::from_u64(7)]];
    assert_eq!(l.point(&column, 1), [F::from_u64(7), F::from_u64(5)]);
}

#[test]
fn expr_from_expr() {
    // q * (next - curr^2), with q = 1 on this row.
    type E = Expr<F, 1, 1>;
    let curr = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let next = E::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
    let q = E::fixed(FixedColumn::new::<0>(), Rotation::CURR);
    let expr = q * (next - curr.clone() * curr);
    let layout = Layout::<1, 2>::current_and_next().unwrap();
    let p: Sparse<F, 2, 2> = from_expr(&expr, &layout, &|_, _| F::ONE).unwrap();
    // x_1 - x_0^2 at (3, 9) vanishes.
    assert!(p.eval(&[F::from_u64(3), F::from_u64(9)]).is_zero());
    // With the selector off, the polynomial is zero.
    let off: Sparse<F, 2, 2> = from_expr(&expr, &layout, &|_, _| F::ZERO).unwrap();
    assert!(off.is_zero());
}

#[test]
fn crate_overview() {
    // P = x0 * x1 - x2, homogenized to degree 2: x0 x1 - u x2.
    let x = |i| Sparse::<F, 3, 2>::variable(Var::try_new(i).unwrap()).unwrap();
    let p = x(0).checked_mul(&x(1)).unwrap() - x(2);
    let (x1, x2) = (
        [F::from_u64(2), F::from_u64(3), F::from_u64(6)],
        [F::ONE, F::from_u64(5), F::from_u64(9)],
    );
    let (u1, u2) = (F::ONE, F::from_u64(4));
    let t = p.cross_terms(&x1, &x2, u1, u2);
    // t_0 and t_2 are the two evaluations; check the folded one at r = 7.
    let r = F::from_u64(7);
    let folded: [F; 3] = std::array::from_fn(|i| x1[i] + r * x2[i]);
    assert_eq!(t[0], p.homogeneous_eval(&x1, u1));
    assert_eq!(t[2], p.homogeneous_eval(&x2, u2));
    assert_eq!(
        t[0] + r * t[1] + r * r * t[2],
        p.homogeneous_eval(&folded, u1 + r * u2)
    );
}

#[test]
fn mv_poly() {
    let x0 = Sparse::<F, 2, 3>::variable(Var::new::<0>()).unwrap();
    let p = x0.checked_mul(&x0).unwrap() + Sparse::constant(F::ONE);
    let d: Dense<F, 2, 3> = Dense::from_terms(p.terms()).unwrap();
    assert_eq!(d.terms(), p.terms());
    assert_eq!(p.degree(), Some(2));
}

#[test]
fn terms() {
    let p = Sparse::<F, 1, 2>::constant(F::from_u64(3));
    assert_eq!(p.terms(), vec![(Monomial::one(), F::from_u64(3))]);
}

#[test]
fn from_terms() {
    let x2 = (Monomial::new([2]), F::ONE);
    assert!(Sparse::<F, 1, 2>::from_terms(vec![x2]).is_some());
    assert!(Sparse::<F, 1, 1>::from_terms(vec![x2]).is_none());
}

#[test]
fn zero() {
    assert!(Sparse::<F, 2, 2>::zero().is_zero());
}

#[test]
fn one() {
    assert_eq!(Sparse::<F, 2, 2>::one().degree(), Some(0));
}

#[test]
fn constant() {
    assert!(Sparse::<F, 1, 1>::constant(F::from_u64(5)).is_constant());
}

#[test]
fn variable() {
    assert_eq!(
        Sparse::<F, 2, 1>::variable(Var::new::<1>())
            .unwrap()
            .degree(),
        Some(1)
    );
    assert!(Sparse::<F, 2, 0>::variable(Var::new::<1>()).is_none());
}

#[test]
fn coefficient() {
    let p = Sparse::<F, 2, 2>::from_terms(vec![(Monomial::new([1, 1]), F::from_u64(4))]).unwrap();
    assert_eq!(p.coefficient(&Monomial::new([1, 1])), F::from_u64(4));
    assert_eq!(p.coefficient(&Monomial::new([2, 0])), F::ZERO);
}

#[test]
fn is_zero() {
    let p = Sparse::<F, 1, 1>::one();
    assert!((p.clone() - p).is_zero());
}

#[test]
fn is_constant() {
    assert!(
        !Sparse::<F, 1, 1>::variable(Var::new::<0>())
            .unwrap()
            .is_constant()
    );
}

#[test]
fn degree() {
    assert_eq!(Sparse::<F, 1, 3>::zero().degree(), None);
}

#[test]
fn is_homogeneous() {
    let x = Sparse::<F, 2, 1>::variable(Var::new::<0>()).unwrap();
    assert!(x.is_homogeneous());
    assert!(!(x + Sparse::one()).is_homogeneous());
}

#[test]
fn is_multilinear() {
    let x = Sparse::<F, 2, 2>::variable(Var::new::<0>()).unwrap();
    let y = Sparse::<F, 2, 2>::variable(Var::new::<1>()).unwrap();
    assert!(x.checked_mul(&y).unwrap().is_multilinear());
    assert!(!x.checked_mul(&x).unwrap().is_multilinear());
}

#[test]
fn scale() {
    let p = Sparse::<F, 1, 1>::one().scale(F::from_u64(3));
    assert_eq!(p, Sparse::constant(F::from_u64(3)));
}

#[test]
fn checked_mul() {
    let x = Sparse::<F, 1, 2>::variable(Var::new::<0>()).unwrap();
    let x2 = x.checked_mul(&x).unwrap();
    assert!(x2.checked_mul(&x).is_none());
}

#[test]
fn eval() {
    let x = Sparse::<F, 2, 2>::variable(Var::new::<0>()).unwrap();
    let y = Sparse::<F, 2, 2>::variable(Var::new::<1>()).unwrap();
    let p = x.checked_mul(&y).unwrap() + Sparse::one();
    assert_eq!(p.eval(&[F::from_u64(2), F::from_u64(3)]), F::from_u64(7));
}

#[test]
fn homogeneous_eval() {
    // 1, homogenized to degree 3: u^3.
    let p = Sparse::<F, 1, 3>::one();
    assert_eq!(
        p.homogeneous_eval(&[F::ZERO], F::from_u64(2)),
        F::from_u64(8)
    );
}

#[test]
fn cross_terms() {
    // x^2, homogeneous: (a + r b)^2 = a^2 + 2 a b r + b^2 r^2.
    let x = Sparse::<F, 1, 2>::variable(Var::new::<0>()).unwrap();
    let p = x.checked_mul(&x).unwrap();
    let (a, b) = (F::from_u64(3), F::from_u64(5));
    let t = p.cross_terms(&[a], &[b], F::ONE, F::ONE);
    assert_eq!(t, vec![a * a, F::from_u64(2) * a * b, b * b]);
}

#[test]
fn cross_terms_scaled() {
    // s * 1, at degree 0: (s1 + r s2).
    let p = Sparse::<F, 1, 0>::one();
    let t = p.cross_terms_scaled(
        &[F::ZERO],
        &[F::ZERO],
        F::ONE,
        F::ONE,
        F::from_u64(2),
        F::from_u64(3),
    );
    assert_eq!(t, vec![F::from_u64(2), F::from_u64(3)]);
}

#[test]
fn hasse() {
    // x^3 in direction 1: 3 x^2, then 3 x, then 1.
    let x = Sparse::<F, 1, 3>::variable(Var::new::<0>()).unwrap();
    let x3 = x.checked_mul(&x).unwrap().checked_mul(&x).unwrap();
    let three = F::from_u64(3);
    assert_eq!(
        x3.hasse(1, &[F::ONE]),
        x.checked_mul(&x).unwrap().scale(three)
    );
    assert_eq!(x3.hasse(2, &[F::ONE]), x.scale(three));
    assert_eq!(x3.hasse(3, &[F::ONE]), Sparse::one());
    assert!(x3.hasse(4, &[F::ONE]).is_zero());
}

#[test]
fn combined_cross_terms_works() {
    let x = Sparse::<F, 1, 1>::variable(Var::new::<0>()).unwrap();
    let polys = [x.clone(), x.scale(F::from_u64(2))];
    let (a, b) = (F::from_u64(3), F::from_u64(4));
    let t = combined_cross_terms(
        &polys,
        &[a],
        &[b],
        F::ONE,
        F::ONE,
        F::from_u64(5),
        F::from_u64(6),
    );
    // At r: x(r) (1 + r) + 2 x(r) (5 + 6 r), with x(r) = 3 + 4 r.
    let r = F::from_u64(9);
    let at = |k: usize| t[k] * r.pow(k as u64);
    let x_r = a + r * b;
    assert_eq!(
        at(0) + at(1) + at(2),
        x_r * (F::ONE + r) + F::from_u64(2) * x_r * (F::from_u64(5) + F::from_u64(6) * r)
    );
}

#[test]
fn combined_cross_terms_with_works() {
    let one = Sparse::<F, 1, 0>::one();
    let t = combined_cross_terms_with(
        &[one.clone(), one],
        &[F::ZERO],
        &[F::ZERO],
        F::ONE,
        F::ONE,
        &[F::ONE, F::from_u64(2)],
        &[F::from_u64(3), F::from_u64(4)],
    );
    // (1 + 3 r) + (2 + 4 r).
    assert_eq!(t, vec![F::from_u64(3), F::from_u64(7)]);
}

#[test]
fn monomial_var() {
    assert_eq!(Var::<3>::new::<2>().index(), 2);
    assert!(Var::<3>::try_new(3).is_none());
}

#[test]
fn monomial_var_new() {
    assert_eq!(Var::<2>::new::<0>().index(), 0);
}

#[test]
fn monomial_var_try_new() {
    assert!(Var::<2>::try_new(1).is_some());
}

#[test]
fn monomial_var_index() {
    assert_eq!(Var::<4>::new::<3>().index(), 3);
}

#[test]
fn monomial_monomial() {
    let xy = Monomial::new([1, 1]);
    let x2 = Monomial::new([2, 0]);
    assert_eq!(xy.degree(), 2);
    assert_eq!(xy.mul(&x2), Monomial::new([3, 1]));
    // Same degree: lexicographic, so x^2 > x y.
    assert!(x2 > xy);
    assert!(Monomial::new([0, 3]) > x2);
}

#[test]
fn monomial_monomial_new() {
    assert_eq!(Monomial::new([2, 0, 1]).exponents(), &[2, 0, 1]);
}

#[test]
fn monomial_monomial_one() {
    assert_eq!(Monomial::<3>::one().degree(), 0);
}

#[test]
fn monomial_monomial_variable() {
    assert_eq!(
        Monomial::variable(Var::<3>::new::<1>()),
        Monomial::new([0, 1, 0])
    );
}

#[test]
fn monomial_monomial_exponents() {
    assert_eq!(Monomial::new([4]).exponents()[0], 4);
}

#[test]
fn monomial_monomial_degree() {
    assert_eq!(Monomial::new([2, 0, 3]).degree(), 5);
}

#[test]
fn monomial_monomial_mul() {
    assert_eq!(
        Monomial::new([1, 2]).mul(&Monomial::new([3, 0])),
        Monomial::new([4, 2])
    );
}

#[test]
fn monomial_monomial_is_multilinear() {
    assert!(Monomial::new([1, 0, 1]).is_multilinear());
    assert!(!Monomial::new([2, 0, 0]).is_multilinear());
}

#[test]
fn monomial_monomial_divides() {
    assert!(Monomial::new([1, 0]).divides(&Monomial::new([2, 1])));
    assert!(!Monomial::new([0, 2]).divides(&Monomial::new([2, 1])));
}

#[test]
fn primes_first() {
    assert_eq!(first(6), vec![2, 3, 5, 7, 11, 13]);
    assert!(first(0).is_empty());
}

#[test]
fn primes_code() {
    let p = primes::first(3);
    // x_0^2 x_2 = 2^2 * 5.
    assert_eq!(primes::code(&Monomial::new([2, 0, 1]), &p), Some(20));
    assert_eq!(primes::code(&Monomial::<3>::one(), &p), Some(1));
}

#[test]
fn primes_decode() {
    let p = primes::first(3);
    assert_eq!(primes::decode::<3>(20, &p), Some(Monomial::new([2, 0, 1])));
    // 7 is not among the first three primes.
    assert_eq!(primes::decode::<3>(14, &p), None);
}

#[test]
fn sparse_sparse() {
    let x = Sparse::<F, 2, 2>::variable(Var::new::<0>()).unwrap();
    let y = Sparse::<F, 2, 2>::variable(Var::new::<1>()).unwrap();
    // (x + y) (x - y) = x^2 - y^2.
    let p = (x.clone() + y.clone())
        .checked_mul(&(x.clone() - y.clone()))
        .unwrap();
    let q = x.checked_mul(&x).unwrap() - y.checked_mul(&y).unwrap();
    assert_eq!(p, q);
    assert_eq!(p.terms().len(), 2);
}

#[test]
fn sparse_sparse_len() {
    assert_eq!(Sparse::<F, 3, 2>::zero().len(), 0);
}

#[test]
fn sparse_sparse_is_empty() {
    assert!(Sparse::<F, 3, 2>::zero().is_empty());
}
