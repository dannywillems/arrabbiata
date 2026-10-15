//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::{Field, Goldilocks as F, Goldilocks};
use arrabbiata_poly::{
    Algebra, Assignment, Domain, Expr, FixedColumn, Polynomials, Rotation, Scalars, UniPoly,
    Variable, WitnessColumn, evaluate, evaluate_graded, evaluate_homogeneous,
};

#[test]
fn column_overview() {
    let a = WitnessColumn::<2>::new::<1>();
    let v: Variable<2, 0> = Variable::Witness(a, Rotation::NEXT);
    assert_eq!(a.index(), 1);
    let _ = v;
}

#[test]
fn column_witness_column() {
    assert_eq!(WitnessColumn::<3>::new::<2>().index(), 2);
}

#[test]
fn column_witness_column_new() {
    let c = WitnessColumn::<4>::new::<3>();
    assert_eq!(c.index(), 3);
}

#[test]
fn column_witness_column_try_new() {
    assert!(WitnessColumn::<2>::try_new(1).is_some());
    assert!(WitnessColumn::<2>::try_new(2).is_none());
}

#[test]
fn column_witness_column_index() {
    assert_eq!(WitnessColumn::<5>::new::<0>().index(), 0);
}

#[test]
fn column_fixed_column() {
    assert_eq!(FixedColumn::<1>::new::<0>().index(), 0);
}

#[test]
fn column_fixed_column_new() {
    assert_eq!(FixedColumn::<2>::new::<1>().index(), 1);
}

#[test]
fn column_fixed_column_try_new() {
    assert!(FixedColumn::<0>::try_new(0).is_none());
}

#[test]
fn column_fixed_column_index() {
    assert_eq!(FixedColumn::<3>::new::<2>().index(), 2);
}

#[test]
fn column_rotation() {
    assert_eq!(Rotation::NEXT.apply(7, 8), 0);
    assert_eq!(Rotation::PREV.apply(0, 8), 7);
    assert_eq!(Rotation::new(10).apply(3, 8), 5);
}

#[test]
fn column_rotation_curr() {
    assert_eq!(Rotation::CURR.offset(), 0);
}

#[test]
fn column_rotation_next() {
    assert_eq!(Rotation::NEXT.offset(), 1);
}

#[test]
fn column_rotation_prev() {
    assert_eq!(Rotation::PREV.offset(), -1);
}

#[test]
fn column_rotation_new() {
    assert_eq!(Rotation::new(1), Rotation::NEXT);
}

#[test]
fn column_rotation_offset() {
    assert_eq!(Rotation::new(-3).offset(), -3);
}

#[test]
fn column_rotation_compose() {
    assert_eq!(Rotation::NEXT.compose(Rotation::PREV), Some(Rotation::CURR));
    assert_eq!(Rotation::new(i32::MAX).compose(Rotation::NEXT), None);
}

#[test]
fn column_rotation_inverse() {
    assert_eq!(Rotation::NEXT.inverse(), Some(Rotation::PREV));
    assert_eq!(Rotation::new(i32::MIN).inverse(), None);
}

#[test]
fn column_rotation_code() {
    assert_eq!(Rotation::CURR.code(), 0);
    assert_eq!(Rotation::PREV.code(), 1);
    assert_eq!(Rotation::NEXT.code(), 2);
    assert_eq!(Rotation::new(i32::MIN).code(), u32::MAX as u64);
}

#[test]
fn column_rotation_apply() {
    assert_eq!(Rotation::CURR.apply(3, 8), 3);
    assert_eq!(Rotation::NEXT.apply(3, 8), 4);
    assert_eq!(Rotation::NEXT.apply(7, 8), 0);
    assert_eq!(Rotation::new(-9).apply(0, 8), 7);
}

#[test]
fn column_variable() {
    let selector: Variable<1, 1> = Variable::Fixed(FixedColumn::new::<0>(), Rotation::CURR);
    let _ = selector;
}

#[test]
fn domain_overview() {
    let domain = Domain::<F>::new(2).unwrap(); // n = 4
    let p = UniPoly::new(vec![F::new(1), F::new(2), F::new(3)]);
    let evaluations = domain.evaluate(&p).unwrap();
    assert_eq!(evaluations[1], p.evaluate(domain.element(1)));
    assert_eq!(
        domain.interpolate(&evaluations).unwrap().coefficient(2),
        F::new(3)
    );
}

#[test]
fn domain_domain() {
    let h = Domain::<F>::new(3).unwrap();
    assert_eq!(h.size(), 8);
    assert_eq!(h.element(8), F::ONE);
}

#[test]
fn domain_domain_new() {
    assert!(Domain::<F>::new(10).is_some());
    assert!(Domain::<F>::new(40).is_none());
}

#[test]
fn domain_domain_containing() {
    assert_eq!(Domain::<F>::containing(5).unwrap().size(), 8);
    assert_eq!(Domain::<F>::containing(1).unwrap().size(), 1);
}

#[test]
fn domain_domain_size() {
    assert_eq!(Domain::<F>::new(4).unwrap().size(), 16);
}

#[test]
fn domain_domain_log_size() {
    assert_eq!(Domain::<F>::new(4).unwrap().log_size(), 4);
}

#[test]
fn domain_domain_generator() {
    let h = Domain::<F>::new(2).unwrap();
    assert_eq!(h.generator().pow(2), -F::ONE);
}

#[test]
fn domain_domain_element() {
    let h = Domain::<F>::new(2).unwrap();
    assert_eq!(h.element(0), F::ONE);
    assert_eq!(h.element(5), h.element(1));
}

#[test]
fn domain_domain_vanishing_at() {
    let h = Domain::<F>::new(3).unwrap();
    assert_eq!(h.vanishing_at(h.element(3)), F::ZERO);
    assert_ne!(h.vanishing_at(F::new(2)), F::ZERO);
}

#[test]
fn domain_domain_first_lagrange_at() {
    let h = Domain::<F>::new(2).unwrap();
    assert_eq!(h.first_lagrange_at(F::ONE), F::ONE);
    assert_eq!(h.first_lagrange_at(h.element(1)), F::ZERO);
}

#[test]
fn domain_domain_lagrange_at() {
    let h = Domain::<F>::new(2).unwrap();
    assert_eq!(h.lagrange_at(2, h.element(2)), F::ONE);
    assert_eq!(h.lagrange_at(2, h.element(3)), F::ZERO);
    // The Lagrange polynomials sum to one everywhere.
    let x = F::new(12345);
    let sum = (0..4).fold(F::ZERO, |acc, j| acc + h.lagrange_at(j, x));
    assert_eq!(sum, F::ONE);
}

#[test]
fn domain_domain_evaluate() {
    let h = Domain::<F>::new(1).unwrap();
    // p = 3 + 2x on {1, -1}.
    let e = h
        .evaluate(&UniPoly::new(vec![F::new(3), F::new(2)]))
        .unwrap();
    assert_eq!(e, vec![F::new(5), F::new(1)]);
}

#[test]
fn domain_domain_interpolate() {
    let h = Domain::<F>::new(1).unwrap();
    let p = h.interpolate(&[F::new(5), F::new(1)]).unwrap();
    assert_eq!((p.coefficient(0), p.coefficient(1)), (F::new(3), F::new(2)));
}

#[test]
fn eval_overview() {
    struct Row(F);
    impl Assignment<F, Scalars<F>, 1, 0> for Row {
        fn witness(&self, _: &Scalars<F>, _: WitnessColumn<1>, _: Rotation) -> F {
            self.0
        }
        fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
            unreachable!("no fixed column: {c:?}")
        }
    }

    // f(x) = x^2 - 1, homogenized to degree 2: x^2 - u^2.
    let x = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let f = x.clone() * x - Expr::constant(F::ONE);
    let ring = Scalars::new();
    let (u, a, t) = (F::new(3), F::new(5), F::new(7));
    // Homogeneity: f^2(t u, t a) = t^2 f^2(u, a).
    let lhs = evaluate_homogeneous(&ring, &f, &Row(t * a), &(t * u), 2).unwrap();
    let rhs = evaluate_homogeneous(&ring, &f, &Row(a), &u, 2).unwrap();
    assert_eq!(lhs, t.square() * rhs);
}

#[test]
fn eval_assignment() {
    /// One row: witness columns a, b and no fixed column.
    struct Row([F; 2]);
    impl Assignment<F, Scalars<F>, 2, 0> for Row {
        fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
            self.0[c.index()]
        }
        fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
            unreachable!("no fixed column: {c:?}")
        }
    }

    type E = Expr<F, 2, 0>;
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
    // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
    let f = a.clone() * b + a + E::constant(F::ONE);
    let row = Row([F::new(2), F::new(3)]);
    let ring = Scalars::new();
    assert_eq!(evaluate(&ring, &f, &row), F::new(9));
}

#[test]
fn eval_witness() {
    /// Witness column 0 holds 4 on the current row and 5 on the next;
    /// the selector (fixed column 0) is 1 everywhere.
    struct Row;
    impl Assignment<F, Scalars<F>, 1, 1> for Row {
        fn witness(&self, _: &Scalars<F>, _: WitnessColumn<1>, r: Rotation) -> F {
            // Row offsets 0 and 1; this row reads no other.
            if r == Rotation::NEXT {
                F::new(5)
            } else {
                F::new(4)
            }
        }
        fn fixed(&self, _: FixedColumn<1>, _: Rotation) -> F {
            F::new(1)
        }
    }
    let ring = Scalars::new();
    assert_eq!(
        Row.witness(&ring, WitnessColumn::new::<0>(), Rotation::NEXT),
        F::new(5)
    );
}

#[test]
fn eval_fixed() {
    /// Witness column 0 holds 4 on the current row and 5 on the next;
    /// the selector (fixed column 0) is 1 everywhere.
    struct Row;
    impl Assignment<F, Scalars<F>, 1, 1> for Row {
        fn witness(&self, _: &Scalars<F>, _: WitnessColumn<1>, r: Rotation) -> F {
            // Row offsets 0 and 1; this row reads no other.
            if r == Rotation::NEXT {
                F::new(5)
            } else {
                F::new(4)
            }
        }
        fn fixed(&self, _: FixedColumn<1>, _: Rotation) -> F {
            F::new(1)
        }
    }
    assert_eq!(
        Row.fixed(FixedColumn::new::<0>(), Rotation::CURR),
        F::new(1)
    );
}

#[test]
fn eval_evaluate() {
    /// One row: witness columns a, b and no fixed column.
    struct Row([F; 2]);
    impl Assignment<F, Scalars<F>, 2, 0> for Row {
        fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
            self.0[c.index()]
        }
        fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
            unreachable!("no fixed column: {c:?}")
        }
    }

    type E = Expr<F, 2, 0>;
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
    // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
    let f = a.clone() * b + a + E::constant(F::ONE);
    let row = Row([F::new(2), F::new(3)]);
    let ring = Scalars::new();
    assert_eq!(evaluate(&ring, &f, &row), F::new(9));
}

#[test]
fn eval_evaluate_graded() {
    /// One row: witness columns a, b and no fixed column.
    struct Row([F; 2]);
    impl Assignment<F, Scalars<F>, 2, 0> for Row {
        fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
            self.0[c.index()]
        }
        fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
            unreachable!("no fixed column: {c:?}")
        }
    }

    type E = Expr<F, 2, 0>;
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
    // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
    let f = a.clone() * b + a + E::constant(F::ONE);
    let row = Row([F::new(2), F::new(3)]);
    let ring = Scalars::new();
    assert_eq!(
        evaluate_graded(&ring, &f, &row),
        vec![F::ONE, F::new(2), F::new(6)]
    );
}

#[test]
fn eval_evaluate_homogeneous() {
    /// One row: witness columns a, b and no fixed column.
    struct Row([F; 2]);
    impl Assignment<F, Scalars<F>, 2, 0> for Row {
        fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
            self.0[c.index()]
        }
        fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
            unreachable!("no fixed column: {c:?}")
        }
    }

    type E = Expr<F, 2, 0>;
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
    // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
    let f = a.clone() * b + a + E::constant(F::ONE);
    let row = Row([F::new(2), F::new(3)]);
    let ring = Scalars::new();
    // f^3(u, a, b) = u^3 + u^2 a + u a b; at u = 2: 8 + 8 + 12.
    assert_eq!(
        evaluate_homogeneous(&ring, &f, &row, &F::new(2), 3),
        Some(F::new(28))
    );
    assert_eq!(evaluate_homogeneous(&ring, &f, &row, &F::ONE, 1), None);
}

#[test]
fn expr_overview() {
    type E = Expr<Goldilocks, 3, 0>;
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
    let c = E::witness(WitnessColumn::new::<2>(), Rotation::CURR);
    let gate = a.clone() * b - c;
    assert_eq!(gate.degree(), 2);
}

#[test]
fn expr_expr() {
    // A selector-gated multiplication gate: q * (a * b - c).
    type E = Expr<F, 3, 1>;
    let q = E::fixed(FixedColumn::new::<0>(), Rotation::CURR);
    let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
    let c = E::witness(WitnessColumn::new::<2>(), Rotation::CURR);
    let gate = q * (a * b - c);
    // The selector is fixed, hence of degree zero.
    assert_eq!(gate.degree(), 2);
}

#[test]
fn expr_expr_constant() {
    assert_eq!(Expr::<F, 1, 0>::constant(F::ONE).degree(), 0);
}

#[test]
fn expr_expr_witness() {
    let next = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
    assert_eq!(next.degree(), 1);
}

#[test]
fn expr_expr_fixed() {
    let q = Expr::<F, 0, 1>::fixed(FixedColumn::new::<0>(), Rotation::CURR);
    assert_eq!(q.degree(), 0);
}

#[test]
fn expr_expr_degree() {
    let x = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    // x^5 - x^5 denotes zero but has syntactic degree 5.
    let x5 = x.clone() * x.clone() * x.clone() * x.clone() * x;
    assert_eq!((x5.clone() - x5).degree(), 5);
}

#[test]
fn crate_overview() {
    // The constraint of a squaring gadget: next = curr^2.
    type E = Expr<F, 1, 0>;
    let curr = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let next = E::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
    let gate = next - curr.clone() * curr;
    assert_eq!(gate.degree(), 2);
}

#[test]
fn ring_algebra() {
    // Code generic over the algebra runs in F and in F[r] alike.
    fn square_plus_one<A>(ring: &A, x: &A::Elem) -> A::Elem
    where
        A: Algebra<F>,
    {
        ring.add(&ring.mul(x, x), &ring.one())
    }
    assert_eq!(square_plus_one(&Scalars::new(), &F::new(3)), F::new(10));
    let p = square_plus_one(&Polynomials::new(), &UniPoly::line(F::ZERO, F::ONE));
    assert_eq!(p.evaluate(F::new(3)), F::new(10));
}

#[test]
fn ring_zero() {
    assert_eq!(Algebra::<F>::zero(&Scalars::new()), F::ZERO);
}

#[test]
fn ring_one() {
    assert_eq!(Algebra::<F>::one(&Scalars::new()), F::ONE);
}

#[test]
fn ring_embed() {
    assert_eq!(
        Polynomials::new().embed(F::new(4)).coefficient(0),
        F::new(4)
    );
}

#[test]
fn ring_add() {
    assert_eq!(Scalars::new().add(&F::new(2), &F::new(3)), F::new(5));
}

#[test]
fn ring_mul() {
    assert_eq!(Scalars::new().mul(&F::new(2), &F::new(3)), F::new(6));
}

#[test]
fn ring_neg() {
    assert_eq!(Scalars::new().neg(&F::ONE), -F::ONE);
}

#[test]
fn ring_scalars() {
    let ring = Scalars::<F>::new();
    assert_eq!(ring.mul(&F::new(6), &F::new(7)), F::new(42));
}

#[test]
fn ring_scalars_new() {
    let _ring = Scalars::<F>::new();
}

#[test]
fn ring_uni_poly() {
    let p = UniPoly::new(vec![F::ONE, F::ZERO, F::ZERO]);
    assert_eq!(p.coefficient(5), F::ZERO);
    assert_eq!(p.evaluate(F::new(9)), F::ONE);
}

#[test]
fn ring_uni_poly_new() {
    // 1 + 2r
    let p = UniPoly::new(vec![F::new(1), F::new(2)]);
    assert_eq!(p.evaluate(F::new(3)), F::new(7));
}

#[test]
fn ring_uni_poly_line() {
    let l = UniPoly::line(F::new(1), F::new(2));
    assert_eq!(l.evaluate(F::new(10)), F::new(21));
}

#[test]
fn ring_uni_poly_coefficient() {
    let l = UniPoly::line(F::new(1), F::new(2));
    assert_eq!(l.coefficient(1), F::new(2));
    assert_eq!(l.coefficient(2), F::ZERO);
}

#[test]
fn ring_uni_poly_len() {
    assert_eq!(UniPoly::line(F::ONE, F::ONE).len(), 2);
}

#[test]
fn ring_uni_poly_is_empty() {
    assert!(UniPoly::<F>::new(vec![]).is_empty());
}

#[test]
fn ring_uni_poly_evaluate() {
    // 1 + r + r^2 at r = 2
    let p = UniPoly::new(vec![F::ONE, F::ONE, F::ONE]);
    assert_eq!(p.evaluate(F::new(2)), F::new(7));
}

#[test]
fn ring_uni_poly_constant() {
    assert_eq!(UniPoly::constant(F::new(4)).evaluate(F::new(9)), F::new(4));
}

#[test]
fn ring_uni_poly_x() {
    assert_eq!(UniPoly::<F>::x().evaluate(F::new(9)), F::new(9));
}

#[test]
fn ring_uni_poly_coefficients() {
    assert_eq!(
        UniPoly::line(F::new(1), F::new(2)).coefficients(),
        &[F::new(1), F::new(2)]
    );
}

#[test]
fn ring_uni_poly_dilate() {
    let p = UniPoly::new(vec![F::new(1), F::new(2), F::new(3)]);
    assert_eq!(
        p.dilate(F::new(5)).evaluate(F::new(7)),
        p.evaluate(F::new(35))
    );
}

#[test]
fn ring_uni_poly_scale() {
    let p = UniPoly::line(F::new(1), F::new(2)).scale(F::new(3));
    assert_eq!(p.coefficient(1), F::new(6));
}

#[test]
fn ring_uni_poly_degree() {
    assert_eq!(UniPoly::new(vec![F::ONE, F::ZERO]).degree(), Some(0));
    assert_eq!(UniPoly::<F>::new(vec![F::ZERO]).degree(), None);
}

#[test]
fn ring_uni_poly_divide_by_vanishing() {
    // (X^2 - 1)(X + 3) = X^3 + 3X^2 - X - 3.
    let p = UniPoly::new(vec![-F::new(3), -F::ONE, F::new(3), F::ONE]);
    let q = p.divide_by_vanishing(2).unwrap();
    assert_eq!((q.coefficient(0), q.coefficient(1)), (F::new(3), F::ONE));
    assert!(UniPoly::new(vec![F::ONE]).divide_by_vanishing(2).is_none());
}

#[test]
fn ring_polynomials() {
    let ring = Polynomials::new();
    // (1 + r)^2 = 1 + 2r + r^2
    let l = UniPoly::line(F::ONE, F::ONE);
    let sq = ring.mul(&l, &l);
    assert_eq!(sq.coefficient(1), F::new(2));
}

#[test]
fn ring_polynomials_new() {
    let _ring = Polynomials::<F>::new();
}
