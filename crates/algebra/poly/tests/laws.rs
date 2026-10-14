//! Property tests for evaluation, graded evaluation and homogenization.
//!
//! Each property is a law of the mathematical object the code models; the
//! module docs of `arrabbiata_poly::eval` state them.

use arrabbiata_field::{Field, Goldilocks};
use arrabbiata_poly::{
    Algebra, Assignment, Expr, FixedColumn, Polynomials, Rotation, Scalars, UniPoly, WitnessColumn,
    evaluate, evaluate_graded, evaluate_homogeneous,
};
use proptest::prelude::*;

type F = Goldilocks;
const W: usize = 3;
const X: usize = 2;
type E = Expr<F, W, X>;

/// One row of values: current and next row of every column.
#[derive(Clone, Debug)]
struct Row {
    witness: [[F; 2]; W],
    fixed: [[F; 2]; X],
}

/// The slot of a rotation drawn by `arb_rotation` (offset 0 or 1).
const fn slot(rotation: Rotation) -> usize {
    rotation.offset() as usize
}

impl Assignment<F, Scalars<F>, W, X> for Row {
    fn witness(&self, _: &Scalars<F>, column: WitnessColumn<W>, rotation: Rotation) -> F {
        self.witness[column.index()][slot(rotation)]
    }

    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F {
        self.fixed[column.index()][slot(rotation)]
    }
}

/// The row scaled by `t` on witness variables only.
fn scale(row: &Row, t: F) -> Row {
    let mut out = row.clone();
    for column in &mut out.witness {
        for value in column.iter_mut() {
            *value = *value * t;
        }
    }
    out
}

/// Two rows on the line `acc + r * fresh`, as values in `F[r]`.
struct Line<'a> {
    acc: &'a Row,
    fresh: &'a Row,
}

impl Assignment<F, Polynomials<F>, W, X> for Line<'_> {
    fn witness(
        &self,
        _: &Polynomials<F>,
        column: WitnessColumn<W>,
        rotation: Rotation,
    ) -> UniPoly<F> {
        let i = column.index();
        let s = slot(rotation);
        UniPoly::line(self.acc.witness[i][s], self.fresh.witness[i][s])
    }

    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F {
        // Fixed columns are shared by both instances.
        self.acc.fixed[column.index()][slot(rotation)]
    }
}

fn arb_f() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(F::new)
}

fn arb_row() -> impl Strategy<Value = Row> {
    (
        prop::array::uniform3(prop::array::uniform2(arb_f())),
        prop::array::uniform2(prop::array::uniform2(arb_f())),
    )
        .prop_map(|(witness, fixed)| Row { witness, fixed })
}

fn arb_rotation() -> impl Strategy<Value = Rotation> {
    prop_oneof![Just(Rotation::CURR), Just(Rotation::NEXT)]
}

fn arb_expr() -> impl Strategy<Value = E> {
    let leaf = prop_oneof![
        arb_f().prop_map(E::constant),
        (0..W, arb_rotation())
            .prop_map(|(i, r)| { E::witness(WitnessColumn::try_new(i).expect("in range"), r) }),
        (0..X, arb_rotation())
            .prop_map(|(i, r)| { E::fixed(FixedColumn::try_new(i).expect("in range"), r) }),
    ];
    leaf.prop_recursive(5, 32, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|a| -a),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
            (inner.clone(), inner).prop_map(|(a, b)| a * b),
        ]
    })
}

proptest! {
    /// The graded components sum to the value: f = f_0 + ... + f_d.
    #[test]
    fn graded_components_sum_to_value(f in arb_expr(), row in arb_row()) {
        let ring = Scalars::new();
        let components = evaluate_graded(&ring, &f, &row);
        prop_assert_eq!(components.len(), f.degree() + 1);
        let sum = components.iter().fold(F::ZERO, |acc, c| acc + *c);
        prop_assert_eq!(sum, evaluate(&ring, &f, &row));
    }

    /// The k-th component is homogeneous of degree k: f_k(t a) = t^k f_k(a).
    #[test]
    fn components_are_homogeneous(f in arb_expr(), row in arb_row(), t in arb_f()) {
        let ring = Scalars::new();
        let plain = evaluate_graded(&ring, &f, &row);
        let scaled = evaluate_graded(&ring, &f, &scale(&row, t));
        for (k, (p, s)) in plain.iter().zip(scaled.iter()).enumerate() {
            prop_assert_eq!(*s, t.pow(k as u64) * *p);
        }
    }

    /// Dehomogenization: f^D(1, a) = f(a).
    #[test]
    fn homogenization_at_one_is_evaluation(
        f in arb_expr(), row in arb_row(), extra in 0usize..3,
    ) {
        let ring = Scalars::new();
        let degree = f.degree() + extra;
        let value = evaluate_homogeneous(&ring, &f, &row, &F::ONE, degree);
        prop_assert_eq!(value, Some(evaluate(&ring, &f, &row)));
    }

    /// Homogeneity: f^D(t u, t a) = t^D f^D(u, a).
    #[test]
    fn homogenization_is_homogeneous(
        f in arb_expr(), row in arb_row(), u in arb_f(), t in arb_f(),
        extra in 0usize..3,
    ) {
        let ring = Scalars::new();
        let degree = f.degree() + extra;
        let lhs = evaluate_homogeneous(&ring, &f, &scale(&row, t), &(t * u), degree);
        let rhs = evaluate_homogeneous(&ring, &f, &row, &u, degree)
            .map(|v| t.pow(degree as u64) * v);
        prop_assert_eq!(lhs, rhs);
    }

    /// Below the degree there is no homogenization.
    #[test]
    fn no_homogenization_below_degree(f in arb_expr(), row in arb_row()) {
        prop_assume!(f.degree() > 0);
        let ring = Scalars::new();
        let value = evaluate_homogeneous(&ring, &f, &row, &F::ONE, f.degree() - 1);
        prop_assert_eq!(value, None);
    }

    /// The folding identity, at the level of one expression: evaluating
    /// f^D on the line (u1 + r u2, a1 + r a2) in F[r] gives a polynomial
    /// whose value at any r is f^D at the folded point, whose constant
    /// coefficient is f^D(u1, a1), and whose coefficient of r^D is
    /// f^D(u2, a2).
    #[test]
    fn folding_identity(
        f in arb_expr(), acc in arb_row(), fresh in arb_row(),
        u1 in arb_f(), u2 in arb_f(), r in arb_f(), extra in 0usize..3,
    ) {
        // Fixed columns are shared by both instances.
        let mut fresh = fresh;
        fresh.fixed = acc.fixed;
        let degree = f.degree() + extra;
        let scalars = Scalars::new();
        let polys = Polynomials::new();
        let line = Line { acc: &acc, fresh: &fresh };
        let g = evaluate_homogeneous(&polys, &f, &line, &UniPoly::line(u1, u2), degree)
            .expect("degree is large enough");
        prop_assert!(g.len() <= degree + 1);

        let mut folded = acc.clone();
        for (column, fresh_column) in folded.witness.iter_mut().zip(fresh.witness.iter()) {
            for (value, fresh_value) in column.iter_mut().zip(fresh_column.iter()) {
                *value = *value + r * *fresh_value;
            }
        }
        let at_r = evaluate_homogeneous(&scalars, &f, &folded, &(u1 + r * u2), degree);
        prop_assert_eq!(Some(g.evaluate(r)), at_r);
        prop_assert_eq!(
            Some(g.coefficient(0)),
            evaluate_homogeneous(&scalars, &f, &acc, &u1, degree)
        );
        prop_assert_eq!(
            Some(g.coefficient(degree)),
            evaluate_homogeneous(&scalars, &f, &fresh, &u2, degree)
        );
    }

    /// F[r] is a commutative ring under the given operations (spot check of
    /// the laws the Algebra trait documents).
    #[test]
    fn polynomial_ring_laws(
        a in prop::collection::vec(arb_f(), 0..4),
        b in prop::collection::vec(arb_f(), 0..4),
        c in prop::collection::vec(arb_f(), 0..4),
        x in arb_f(),
    ) {
        let ring = Polynomials::new();
        let (a, b, c) = (UniPoly::new(a), UniPoly::new(b), UniPoly::new(c));
        // Compare through evaluation: polynomials equal up to trailing zeros.
        let ev = |p: &UniPoly<F>| p.evaluate(x);
        prop_assert_eq!(ev(&ring.mul(&a, &b)), ev(&a) * ev(&b));
        prop_assert_eq!(ev(&ring.add(&a, &b)), ev(&a) + ev(&b));
        prop_assert_eq!(ev(&ring.neg(&a)), -ev(&a));
        prop_assert_eq!(
            ev(&ring.mul(&a, &ring.add(&b, &c))),
            ev(&ring.add(&ring.mul(&a, &b), &ring.mul(&a, &c)))
        );
    }
}

#[test]
fn degree_ignores_fixed_columns() {
    let q = E::fixed(FixedColumn::new::<0>(), Rotation::CURR);
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::NEXT);
    assert_eq!((q.clone() * q * a.clone() * b).degree(), 2);
    assert_eq!(a.degree(), 1);
    assert_eq!(E::constant(F::ONE).degree(), 0);
}

#[test]
fn rotation_wraps() {
    assert_eq!(Rotation::NEXT.apply(3, 4), 0);
    assert_eq!(Rotation::NEXT.apply(1, 4), 2);
    assert_eq!(Rotation::CURR.apply(3, 4), 3);
}

mod domain {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_poly::{Domain, UniPoly};
    use proptest::prelude::*;

    proptest! {
        /// The NTT agrees with naive evaluation, and interpolation inverts it.
        #[test]
        fn ntt_is_evaluation(
            log in 0u32..6,
            coefficients in prop::collection::vec(any::<u64>().prop_map(F::new), 0..32),
        ) {
            let domain = Domain::<F>::new(log).unwrap();
            let mut coefficients = coefficients;
            coefficients.truncate(domain.size());
            let p = UniPoly::new(coefficients);
            let evaluations = domain.evaluate(&p).unwrap();
            for (i, e) in evaluations.iter().enumerate() {
                prop_assert_eq!(*e, p.evaluate(domain.element(i)));
            }
            let q = domain.interpolate(&evaluations).unwrap();
            for k in 0..domain.size() {
                prop_assert_eq!(q.coefficient(k), p.coefficient(k));
            }
        }

        /// p * Z_H divided by Z_H is p; anything not vanishing on H does not
        /// divide.
        #[test]
        fn division_by_vanishing(
            log in 0u32..4,
            coefficients in prop::collection::vec(any::<u64>().prop_map(F::new), 0..10),
            shift in 1u64..1000,
        ) {
            let n = 1usize << log;
            let mut z = vec![F::ZERO; n + 1];
            z[0] = -F::ONE;
            z[n] = F::ONE;
            let p = UniPoly::new(coefficients);
            let q = (&p * &UniPoly::new(z)).divide_by_vanishing(n).unwrap();
            let x = F::new(shift);
            prop_assert_eq!(q.evaluate(x), p.evaluate(x));
            let off = &(&p * &UniPoly::new({
                let mut z = vec![F::ZERO; n + 1];
                z[0] = -F::ONE;
                z[n] = F::ONE;
                z
            })) + &UniPoly::new(vec![F::ONE]);
            prop_assert!(off.divide_by_vanishing(n).is_none());
        }

        /// Dilation is precomposition with multiplication: p(w X) at x is
        /// p at w x; and dilation by a root of unity shifts evaluations on
        /// the domain by one position.
        #[test]
        fn dilation(
            log in 1u32..5,
            coefficients in prop::collection::vec(any::<u64>().prop_map(F::new), 0..16),
            x in any::<u64>().prop_map(F::new),
            w in any::<u64>().prop_map(F::new),
        ) {
            let p = UniPoly::new(coefficients);
            prop_assert_eq!(p.dilate(w).evaluate(x), p.evaluate(w * x));
            let domain = Domain::<F>::new(log).unwrap();
            let shifted = p.dilate(domain.generator());
            for i in 0..domain.size() {
                prop_assert_eq!(shifted.evaluate(domain.element(i)), p.evaluate(domain.element(i + 1)));
            }
            prop_assert_eq!(UniPoly::constant(x).evaluate(w), x);
            prop_assert_eq!(UniPoly::<F>::x().evaluate(w), w);
        }

        /// L_1 is 1 at 1 and 0 elsewhere on the domain.
        #[test]
        fn first_lagrange(log in 0u32..6) {
            let domain = Domain::<F>::new(log).unwrap();
            for i in 0..domain.size() {
                let expected = if i == 0 { F::ONE } else { F::ZERO };
                prop_assert_eq!(domain.first_lagrange_at(domain.element(i)), expected);
            }
        }
    }
}
