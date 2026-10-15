//! Kimchi's CompleteAdd gate: the sum of two affine points of a curve
//! `y^2 = x^3 + b`, including doubling and the point at infinity.
//!
//! The row is
//!
//! | 0    | 1    | 2    | 3    | 4    | 5    | 6     | 7        | 8   | 9       | 10        |
//! | ---- | ---- | ---- | ---- | ---- | ---- | ----- | -------- | --- | ------- | --------- |
//! | `x1` | `y1` | `x2` | `y2` | `x3` | `y3` | `inf` | `same_x` | `s` | `inf_z` | `x21_inv` |
//!
//! where `(x3, y3)` is the sum, `inf` says that it is the point at infinity
//! (then `(x3, y3)` is not a point; Kimchi fills it with the doubling of
//! `(x1, y1)`), `same_x` says `x1 = x2`, and `s` is the slope. The seven
//! constraints, in Kimchi's order:
//!
//! 1. `x21_inv (x2 - x1) - (1 - same_x)` and 2. `same_x (x2 - x1)`: `same_x`
//!    is `x1 == x2`, and `x21_inv` the inverse of `x2 - x1` when it exists;
//! 3. `same_x (2 s y1 - 3 x1^2) + (1 - same_x) ((x2 - x1) s - (y2 - y1))`:
//!    the slope of the doubling or of the chord;
//! 4. `x1 + x2 + x3 - s^2` and 5. `s (x1 - x3) - y1 - y3`: the sum;
//! 6. `(y2 - y1) (same_x - inf)` and 7. `(y2 - y1) inf_z - inf`: `inf` is
//!    `same_x` and `y1 != y2`.
//!
//! The witness is computed as Kimchi's test (`tests/ec.rs`) writes it.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, TypedGadget};

/// The column of each cell of the row.
pub mod column {
    /// `x1`.
    pub const X1: usize = 0;
    /// `y1`.
    pub const Y1: usize = 1;
    /// `x2`.
    pub const X2: usize = 2;
    /// `y2`.
    pub const Y2: usize = 3;
    /// `x3`.
    pub const X3: usize = 4;
    /// `y3`.
    pub const Y3: usize = 5;
    /// `inf`.
    pub const INF: usize = 6;
    /// `same_x`.
    pub const SAME_X: usize = 7;
    /// The slope `s`.
    pub const S: usize = 8;
    /// `inf_z`.
    pub const INF_Z: usize = 9;
    /// `x21_inv`.
    pub const X21_INV: usize = 10;
}

fn bit<F: Field>(b: bool) -> F {
    if b { F::ONE } else { F::ZERO }
}

/// The inverse, or zero.
fn inverse_or_zero<F: Field>(x: F) -> F {
    x.inverse().unwrap_or(F::ZERO)
}

/// The slope Kimchi's witness uses: the tangent's when `x1 = x2`, the
/// chord's otherwise (zero when it does not exist).
fn slope<F: Field>([x1, y1, x2, y2]: [F; 4]) -> F {
    if x1 == x2 {
        let three_x1_squared = (x1 * x1) + (x1 * x1) + (x1 * x1);
        three_x1_squared * inverse_or_zero(y1 + y1)
    } else {
        (y2 - y1) * inverse_or_zero(x2 - x1)
    }
}

/// The sum of `(x1, y1)` and `(x2, y2)`: its input `[x1, y1, x2, y2]`, its
/// output `[x3, y3, inf]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompleteAdd;

impl<F> TypedGadget<F> for CompleteAdd
where
    F: Field,
{
    type Input = Array<4>;
    type Output = Array<3>;
    const NAME: &'static str = "kimchi-complete-add";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Array<4>, E::Var>) -> Of<Array<3>, E::Var>
    where
        E: Env<F>,
    {
        use column::{INF, INF_Z, S, SAME_X, X1, X2, X3, X21_INV, Y1, Y2, Y3};
        let x1 = env.copy(Cell::new(X1, 0), &input[0]);
        let y1 = env.copy(Cell::new(Y1, 0), &input[1]);
        let x2 = env.copy(Cell::new(X2, 0), &input[2]);
        let y2 = env.copy(Cell::new(Y2, 0), &input[3]);
        let points = [&x1, &y1, &x2, &y2];

        let same_x = env.advice(Cell::new(SAME_X, 0), [&x1, &x2], |[a, b]| bit(a == b));
        let x21_inv = env.advice(Cell::new(X21_INV, 0), [&x1, &x2], |[a, b]| {
            inverse_or_zero(b - a)
        });
        let s = env.advice(Cell::new(S, 0), points, slope);
        let x3 = env.advice(Cell::new(X3, 0), points, |p| {
            let s = slope(p);
            s * s - p[0] - p[2]
        });
        let y3 = env.advice(Cell::new(Y3, 0), points, |p| {
            let s = slope(p);
            let x3 = s * s - p[0] - p[2];
            s * (p[0] - x3) - p[1]
        });
        let inf = env.advice(Cell::new(INF, 0), points, |[a, b, c, d]| {
            bit(a == c && b != d)
        });
        let inf_z = env.advice(Cell::new(INF_Z, 0), points, |[a, b, c, d]| {
            if a == c && b != d {
                inverse_or_zero(d - b)
            } else {
                F::ZERO
            }
        });

        let one = env.constant(F::ONE);
        let x21 = x2.clone() - x1.clone();
        let y21 = y2 - y1.clone();
        // 1, 2: same_x and x21_inv.
        env.assert_zero(x21_inv * x21.clone() - (one.clone() - same_x.clone()));
        env.assert_zero(same_x.clone() * x21.clone());
        // 3: the slope.
        let x1_squared = x1.clone() * x1.clone();
        let doubling = s.clone() * y1.clone() + s.clone() * y1.clone()
            - x1_squared.clone()
            - x1_squared.clone()
            - x1_squared;
        let chord = x21 * s.clone() - y21.clone();
        env.assert_zero(same_x.clone() * doubling + (one - same_x.clone()) * chord);
        // 4, 5: the sum.
        env.assert_zero(x1.clone() + x2 + x3.clone() - s.clone() * s.clone());
        env.assert_zero(s * (x1 - x3.clone()) - y1 - y3.clone());
        // 6, 7: the point at infinity.
        env.assert_zero(y21.clone() * (same_x - inf.clone()));
        env.assert_zero(y21 * inf_z - inf.clone());
        [x3, y3, inf]
    }
}
