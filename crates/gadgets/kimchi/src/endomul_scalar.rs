//! Kimchi's EndoMulScalar gate: eight two-bit crumbs per row of the scalar
//! that EndoMul multiplies by.
//!
//! A scalar challenge `x` of `2 m` bits stands for the scalar `a lambda + b`
//! of Kimchi's `ScalarChallenge::to_field` (Algorithm 2 of the Halo paper,
//! ePrint 2019/1021, page 29): from `a = b = 2`, each crumb `x_i` (two bits,
//! the most significant first) maps `(a, b)` to `(2 a + c(x_i), 2 b + d(x_i))`
//! with `c(0, 1, 2, 3) = (0, 0, -1, 1)` and `d(0, 1, 2, 3) = (-1, 1, 0, 0)`.
//! The gate interpolates `c` and `d` on `{0, 1, 2, 3}`:
//! `c(x) = 2/3 x^3 - 5/2 x^2 + 11/6 x`, `d(x) = c(x) - x^2 + 3 x - 1`.
//!
//! The row, as Kimchi lays it out (column 14 unused):
//!
//! | 0    | 1    | 2    | 3    | 4    | 5    | 6 .. 13      |
//! | ---- | ---- | ---- | ---- | ---- | ---- | ------------ |
//! | `n0` | `n8` | `a0` | `b0` | `a8` | `b8` | `x0 .. x7`   |
//!
//! The 11 constraints, in Kimchi's order: `n8 = 4^8 n0 + ...` (each crumb
//! `n -> 4 n + x`), `a8` and `b8` after the eight steps, and each crumb in
//! `{0, 1, 2, 3}` (`x (x - 1) (x - 2) (x - 3) = 0`). The witness is computed
//! as Kimchi's `endomul_scalar::gen_witness`.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, TypedGadget};

/// The crumbs of one row.
pub const CRUMBS_PER_ROW: usize = 8;

/// The columns of the row.
pub mod column {
    /// `n0`.
    pub const N0: usize = 0;
    /// `n8`.
    pub const N8: usize = 1;
    /// `a0`.
    pub const A0: usize = 2;
    /// `b0`.
    pub const B0: usize = 3;
    /// `a8`.
    pub const A8: usize = 4;
    /// `b8`.
    pub const B8: usize = 5;
    /// The first crumb; crumb `i` is in column `X + i`.
    pub const X: usize = 6;
}

use self::column::{A0, A8, B0, B8, N0, N8, X};

/// `c` and `d` of a crumb's value, as Kimchi's `c_func` and `d_func` (zero
/// outside `{0, 1, 2, 3}`, where the gate's range check fails).
fn c_d<F: Field>(x: F) -> (F, F) {
    let one = F::ONE;
    let values = [
        (F::ZERO, -one),
        (F::ZERO, one),
        (-one, F::ZERO),
        (one, F::ZERO),
    ];
    let mut found = (F::ZERO, F::ZERO);
    for (i, value) in values.iter().enumerate() {
        if x == F::from_u64(i as u64) {
            found = *value;
        }
    }
    found
}

/// The polynomial of `coefficients` (constant first) at `x`, by Horner, as
/// Kimchi's `polynomial`.
fn polynomial<V>(coefficients: &[V], x: &V, zero: V) -> V
where
    V: Clone + core::ops::Add<Output = V> + core::ops::Mul<Output = V>,
{
    coefficients
        .iter()
        .rev()
        .fold(zero, |acc, c| acc * x.clone() + c.clone())
}

/// One row: input `(n0, a0, b0, [x0 .. x7])`, output `[n8, a8, b8]`.
///
/// `n0`, `a0`, `b0` are wired by copies; the crumbs are written from their
/// values and constrained by the gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndoMulScalarGate;

impl<F> TypedGadget<F> for EndoMulScalarGate
where
    F: Field,
{
    type Input = (Array<3>, Array<CRUMBS_PER_ROW>);
    type Output = Array<3>;
    const NAME: &'static str = "kimchi-endomul-scalar";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (start, crumbs): Of<Self::Input, E::Var>,
    ) -> Of<Array<3>, E::Var>
    where
        E: Env<F>,
    {
        let n0 = env.copy(Cell::new(N0, 0), &start[0]);
        let a0 = env.copy(Cell::new(A0, 0), &start[1]);
        let b0 = env.copy(Cell::new(B0, 0), &start[2]);
        let x: [E::Var; CRUMBS_PER_ROW] =
            core::array::from_fn(|i| env.advice(Cell::new(X + i, 0), [&crumbs[i]], |[v]| v));
        let known = [
            &n0, &a0, &b0, &x[0], &x[1], &x[2], &x[3], &x[4], &x[5], &x[6], &x[7],
        ];
        let n8 = env.advice(Cell::new(N8, 0), known, |v| {
            v[3..].iter().fold(v[0], |n, x| n + n + n + n + *x)
        });
        let a8 = env.advice(Cell::new(A8, 0), known, |v| {
            v[3..].iter().fold(v[1], |a, x| a + a + c_d(*x).0)
        });
        let b8 = env.advice(Cell::new(B8, 0), known, |v| {
            v[3..].iter().fold(v[2], |b, x| b + b + c_d(*x).1)
        });

        let constant = |env: &mut E, numerator: u64, denominator: u64, negative: bool| {
            let fraction =
                F::from_u64(numerator) * F::from_u64(denominator).inverse().unwrap_or(F::ZERO);
            env.constant(if negative { -fraction } else { fraction })
        };
        let c_coefficients = [
            constant(env, 0, 1, false),
            constant(env, 11, 6, false),
            constant(env, 5, 2, true),
            constant(env, 2, 3, false),
        ];
        let crumb_over_x = [
            constant(env, 6, 1, true),
            constant(env, 11, 1, false),
            constant(env, 6, 1, true),
            constant(env, 1, 1, false),
        ];
        let d_minus_c = [
            constant(env, 1, 1, true),
            constant(env, 3, 1, false),
            constant(env, 1, 1, true),
        ];
        let zero = env.constant(F::ZERO);
        let c: [E::Var; CRUMBS_PER_ROW] =
            core::array::from_fn(|i| polynomial(&c_coefficients, &x[i], zero.clone()));
        let d: [E::Var; CRUMBS_PER_ROW] =
            core::array::from_fn(|i| c[i].clone() + polynomial(&d_minus_c, &x[i], zero.clone()));
        let n8_expected = x.iter().fold(n0, |n, xi| {
            let double = n.clone() + n;
            double.clone() + double + xi.clone()
        });
        let a8_expected = c.iter().fold(a0, |a, ci| a.clone() + a + ci.clone());
        let b8_expected = d.iter().fold(b0, |b, di| b.clone() + b + di.clone());
        env.assert_zero(n8_expected - n8.clone());
        env.assert_zero(a8_expected - a8.clone());
        env.assert_zero(b8_expected - b8.clone());
        for xi in &x {
            env.assert_zero(polynomial(&crumb_over_x, xi, zero.clone()) * xi.clone());
        }
        [n8, a8, b8]
    }
}

/// The scalar of `CRUMBS` crumbs, most significant first, on `CRUMBS / 8`
/// rows: input the crumbs, output `[n, a, b]`, with `n` the crumbs as an
/// integer (base 4) and `a lambda + b` the scalar Kimchi's
/// `ScalarChallenge::to_field` gives for `lambda` the GLV `lambda`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndoMulScalar<const CRUMBS: usize>;

impl<F, const C: usize> TypedGadget<F> for EndoMulScalar<C>
where
    F: Field,
{
    type Input = Array<C>;
    type Output = Array<3>;
    const NAME: &'static str = "kimchi-endomul-scalar-chain";
    const ROWS: usize = {
        assert!(
            C.is_multiple_of(CRUMBS_PER_ROW),
            "the crumbs come in rows of eight"
        );
        C / CRUMBS_PER_ROW
    };

    fn synthesize<E>(&self, env: &mut E, crumbs: Of<Array<C>, E::Var>) -> Of<Array<3>, E::Var>
    where
        E: Env<F>,
    {
        let zero = env.constant(F::ZERO);
        let two = env.constant(F::from_u64(2));
        let n = env.define(Cell::new(N0, 0), zero);
        let a = env.define(Cell::new(A0, 0), two.clone());
        let b = env.define(Cell::new(B0, 0), two);
        let mut state = [n, a, b];
        // Each row copies its start from the previous row's end, as Kimchi's
        // wiring does.
        for chunk in crumbs.chunks(CRUMBS_PER_ROW) {
            let eight: [E::Var; CRUMBS_PER_ROW] = core::array::from_fn(|k| chunk[k].clone());
            state = env.call(&EndoMulScalarGate, (state, eight));
        }
        state
    }
}
