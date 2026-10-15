//! Agreement with the rest of the stack.
//!
//! - `from_expr` agrees with `arrabbiata_poly::evaluate`: the polynomial of
//!   an expression, at the point of a row, is the expression's value there;
//!   and its exact degree is at most the syntactic one.
//! - The cross-terms of folding computed here (normal forms and the
//!   binomial expansion) agree, coefficient by coefficient, with those the
//!   folding prover computes (the relation evaluated in `F[r]`), on the
//!   example relations: Nova's squaring (one constraint), Fibonacci (two
//!   constraints combined by folded powers), and a relation reading the
//!   previous row and the row after next.

use arrabbiata_field::Field;
use arrabbiata_mvpoly::{Layout, MVPoly, Sparse, combined_cross_terms_with, from_expr};
use arrabbiata_pasta::Fq as F;
use arrabbiata_poly::{
    Assignment, Expr, FixedColumn, Polynomials, Rotation, Scalars, UniPoly, WitnessColumn, evaluate,
};
use arrabbiata_relation::{ConstraintSystem, RowView, aggregation::Aggregation, examples};
use proptest::prelude::*;

const ROWS: usize = 6;

fn arb_f() -> impl Strategy<Value = F> {
    prop_oneof![
        1 => Just(F::ZERO),
        1 => Just(F::ONE),
        6 => any::<u64>().prop_map(F::from_u64),
    ]
}

fn arb_rotation() -> impl Strategy<Value = Rotation> {
    prop_oneof![
        Just(Rotation::PREV),
        Just(Rotation::CURR),
        Just(Rotation::NEXT)
    ]
}

type E = Expr<F, 2, 1>;

fn arb_expr() -> impl Strategy<Value = E> {
    let leaf = prop_oneof![
        arb_f().prop_map(E::constant),
        (0usize..2, arb_rotation())
            .prop_map(|(k, r)| E::witness(WitnessColumn::try_new(k).unwrap(), r)),
        arb_rotation().prop_map(|r| E::fixed(FixedColumn::new::<0>(), r)),
    ];
    leaf.prop_recursive(3, 16, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|a| -a),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
            (inner.clone(), inner).prop_map(|(a, b)| a * b),
        ]
    })
}

fn arb_columns<const W: usize>() -> impl Strategy<Value = [Vec<F>; W]> {
    prop::collection::vec(prop::collection::vec(arb_f(), ROWS), W)
        .prop_map(|v| <[Vec<F>; W]>::try_from(v).unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The polynomial of an expression, at a row's point, is the
    /// expression's value at the row.
    #[test]
    fn from_expr_agrees_with_evaluation(
        expr in arb_expr(),
        witness in arb_columns::<2>(),
        fixed in arb_columns::<1>(),
        row in 0..ROWS,
    ) {
        let layout = Layout::<2, 6>::new(&[Rotation::PREV, Rotation::CURR, Rotation::NEXT]).unwrap();
        let fixed_at = |c: FixedColumn<1>, r: Rotation| fixed[c.index()][r.apply(row, ROWS)];
        // Degree 8 holds every product of the depth-3 expressions drawn.
        let p: Sparse<F, 6, 8> = from_expr(&expr, &layout, &fixed_at).unwrap();
        let view = RowView::new(&witness, &fixed, row, ROWS);
        let expected = evaluate(&Scalars::new(), &expr, &view);
        prop_assert_eq!(p.eval(&layout.point(&witness, row)), expected);
        // The exact degree is at most the syntactic one.
        prop_assert!(p.degree().is_none_or(|d| d <= expr.degree()));
    }
}

/// The witness columns of two instances, read on a line `acc + r fresh`.
struct Line<'a, const W: usize, const X: usize> {
    acc: &'a [Vec<F>; W],
    fresh: &'a [Vec<F>; W],
    fixed: &'a [Vec<F>; X],
    row: usize,
}

impl<const W: usize, const X: usize> Assignment<F, Polynomials<F>, W, X> for Line<'_, W, X> {
    fn witness(&self, _: &Polynomials<F>, c: WitnessColumn<W>, r: Rotation) -> UniPoly<F> {
        let i = r.apply(self.row, ROWS);
        UniPoly::line(self.acc[c.index()][i], self.fresh[c.index()][i])
    }

    fn fixed(&self, c: FixedColumn<X>, r: Rotation) -> F {
        self.fixed[c.index()][r.apply(self.row, ROWS)]
    }
}

/// On every row, the folding polynomial in `r` computed by the relation (as
/// the folding prover does) equals the one computed from the constraints'
/// normal forms here.
fn agree<A, const W: usize, const X: usize, const M: usize, const D: usize, const N: usize>(
    cs: &ConstraintSystem<F, A, W, X, M, D>,
    layout: &Layout<W, N>,
    (acc, fresh): (&[Vec<F>; W], &[Vec<F>; W]),
    (u1, u2): (F, F),
    coefficients: Option<(&[F; M], &[F; M])>,
) -> Result<(), TestCaseError>
where
    A: Aggregation<F, M>,
{
    let ring = Polynomials::new();
    let u = UniPoly::line(u1, u2);
    let lines: Option<Vec<UniPoly<F>>> =
        coefficients.map(|(c1, c2)| (0..M).map(|j| UniPoly::line(c1[j], c2[j])).collect());
    for row in 0..ROWS {
        let view = Line {
            acc,
            fresh,
            fixed: cs.fixed_columns(),
            row,
        };
        let relation = cs.evaluate_row(&ring, &view, &u, lines.as_deref());
        let fixed_at =
            |c: FixedColumn<X>, r: Rotation| cs.fixed_columns()[c.index()][r.apply(row, ROWS)];
        let polys: Vec<Sparse<F, N, D>> = cs
            .constraints()
            .iter()
            .map(|e| from_expr(e, layout, &fixed_at).unwrap())
            .collect();
        let (x1, x2) = (layout.point(acc, row), layout.point(fresh, row));
        let ours = match coefficients {
            Some((c1, c2)) => combined_cross_terms_with(&polys, &x1, &x2, u1, u2, c1, c2),
            None => polys.iter().fold(vec![F::ZERO; D + 1], |mut t, p| {
                for (k, v) in p.cross_terms(&x1, &x2, u1, u2).into_iter().enumerate() {
                    t[k] = t[k] + v;
                }
                t
            }),
        };
        for (k, v) in ours.iter().enumerate() {
            prop_assert_eq!(relation.coefficient(k), *v, "row {}, r^{}", row, k);
        }
        prop_assert!((ours.len()..ours.len() + 4).all(|k| relation.coefficient(k).is_zero()));
    }
    Ok(())
}

/// `q (a(-1) a(+2) - a)`, `q` off on the first row.
fn rotated() -> ConstraintSystem<F, arrabbiata_relation::aggregation::Single, 1, 1, 1, 2> {
    type E = Expr<F, 1, 1>;
    let a = |k| E::witness(WitnessColumn::new::<0>(), Rotation::new(k));
    let q = E::fixed(FixedColumn::new::<0>(), Rotation::CURR);
    let mut selector = vec![F::ONE; ROWS];
    selector[0] = F::ZERO;
    ConstraintSystem::new(
        core::num::NonZeroUsize::new(ROWS).unwrap(),
        [selector],
        [q * (a(-1) * a(2) - a(0))],
    )
    .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Nova's squaring: one constraint, degree 2.
    #[test]
    fn squaring_cross_terms_agree(
        acc in arb_columns::<1>(), fresh in arb_columns::<1>(), u1 in arb_f(), u2 in arb_f(),
    ) {
        let cs = examples::squaring::<F>(ROWS);
        let layout = Layout::<1, 2>::current_and_next().unwrap();
        agree(&cs, &layout, (&acc, &fresh), (u1, u2), None)?;
    }

    /// Fibonacci: two constraints combined by coefficients folded with the
    /// instances, degree 2 + 1.
    #[test]
    fn fibonacci_cross_terms_agree(
        acc in arb_columns::<2>(), fresh in arb_columns::<2>(),
        u1 in arb_f(), u2 in arb_f(),
        c1 in [arb_f(), arb_f()], c2 in [arb_f(), arb_f()],
    ) {
        let cs = examples::fibonacci::<F>(ROWS);
        let layout = Layout::<2, 4>::current_and_next().unwrap();
        agree(&cs, &layout, (&acc, &fresh), (u1, u2), Some((&c1, &c2)))?;
    }

    /// A relation reading the previous row and the row after next.
    #[test]
    fn rotated_cross_terms_agree(
        acc in arb_columns::<1>(), fresh in arb_columns::<1>(), u1 in arb_f(), u2 in arb_f(),
    ) {
        let cs = rotated();
        let layout = Layout::<1, 3>::new(&[Rotation::PREV, Rotation::CURR, Rotation::new(2)]).unwrap();
        agree(&cs, &layout, (&acc, &fresh), (u1, u2), None)?;
    }
}
