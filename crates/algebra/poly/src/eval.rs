//! Evaluation, graded evaluation and homogenization.
//!
//! # The three evaluations
//!
//! Let `f` be an expression, `A` an `F`-algebra, and `a` an assignment of
//! the variables to `A` (witness variables to `A`, fixed variables to `F`).
//!
//! - [`evaluate`] computes `f(a)`, the image of `f` under the unique algebra
//!   morphism extending `a` (universal property of the polynomial ring).
//! - [`evaluate_graded`] computes the components `f_0(a), ..., f_d(a)` of
//!   `f = f_0 + ... + f_d` split by degree in the witness variables, where
//!   `d = f.degree()`. Equivalently, `f_k(a)` is the coefficient of `t^k` in
//!   `f(t * a)`: graded evaluation is evaluation in `A[t]` under the
//!   assignment `w |-> t * a(w)` on witness variables.
//! - [`evaluate_homogeneous`] computes `f^D(u, a) = sum_k u^(D-k) f_k(a)`,
//!   the homogenization of `f` to degree `D >= d` with the extra variable
//!   `u`. It is homogeneous of degree `D` in `(u, witness variables)`:
//!   `f^D(t u, t a) = t^D f^D(u, a)`, and `f^D(1, a) = f(a)`.
//!
//! Homogeneity is what makes folding work: substituting `u1 + r u2` and
//! `a1 + r a2` into `f^D` gives a polynomial in `r` of degree at most `D`
//! whose constant coefficient is `f^D(u1, a1)` and whose leading coefficient
//! is `f^D(u2, a2)`. The coefficients in between are the cross-terms.
//!
//! ```
//! use arrabbiata_field::{Field, Goldilocks as F};
//! use arrabbiata_poly::{Assignment, Expr, FixedColumn, Rotation, Scalars, WitnessColumn, evaluate_homogeneous};
//!
//! struct Row(F);
//! impl Assignment<F, Scalars<F>, 1, 0> for Row {
//!     fn witness(&self, _: &Scalars<F>, _: WitnessColumn<1>, _: Rotation) -> F {
//!         self.0
//!     }
//!     fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
//!         unreachable!("no fixed column: {c:?}")
//!     }
//! }
//!
//! // f(x) = x^2 - 1, homogenized to degree 2: x^2 - u^2.
//! let x = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
//! let f = x.clone() * x - Expr::constant(F::ONE);
//! let ring = Scalars::new();
//! let (u, a, t) = (F::new(3), F::new(5), F::new(7));
//! // Homogeneity: f^2(t u, t a) = t^2 f^2(u, a).
//! let lhs = evaluate_homogeneous(&ring, &f, &Row(t * a), &(t * u), 2).unwrap();
//! let rhs = evaluate_homogeneous(&ring, &f, &Row(a), &u, 2).unwrap();
//! assert_eq!(lhs, t.square() * rhs);
//! ```

use alloc::{vec, vec::Vec};

use arrabbiata_field::Field;

use crate::{
    column::{FixedColumn, Rotation, Variable, WitnessColumn},
    expr::Expr,
    ring::Algebra,
};

/// An assignment of the variables of a circuit with `W` witness columns and
/// `X` fixed columns, at one row, into the algebra `R`.
///
/// Fixed columns always take values in `F`, since they are never folded.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{
///     Algebra, Assignment, Expr, FixedColumn, Rotation, Scalars, WitnessColumn,
///     evaluate, evaluate_graded, evaluate_homogeneous,
/// };
///
/// /// One row: witness columns a, b and no fixed column.
/// struct Row([F; 2]);
/// impl Assignment<F, Scalars<F>, 2, 0> for Row {
///     fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
///         self.0[c.index()]
///     }
///     fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
///         unreachable!("no fixed column: {c:?}")
///     }
/// }
///
/// type E = Expr<F, 2, 0>;
/// let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
/// // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
/// let f = a.clone() * b + a + E::constant(F::ONE);
/// let row = Row([F::new(2), F::new(3)]);
/// let ring = Scalars::new();
/// assert_eq!(evaluate(&ring, &f, &row), F::new(9));
/// ```
pub trait Assignment<F, R, const W: usize, const X: usize>
where
    F: Field,
    R: Algebra<F>,
{
    /// The value of witness column `column` at `rotation`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::{Assignment, FixedColumn, Rotation, Scalars, WitnessColumn};
    ///
    /// /// Witness column 0 holds 4 on the current row and 5 on the next;
    /// /// the selector (fixed column 0) is 1 everywhere.
    /// struct Row;
    /// impl Assignment<F, Scalars<F>, 1, 1> for Row {
    ///     fn witness(&self, _: &Scalars<F>, _: WitnessColumn<1>, r: Rotation) -> F {
    ///         // Row offsets 0 and 1; this row reads no other.
    ///         if r == Rotation::NEXT { F::new(5) } else { F::new(4) }
    ///     }
    ///     fn fixed(&self, _: FixedColumn<1>, _: Rotation) -> F {
    ///         F::new(1)
    ///     }
    /// }
    /// let ring = Scalars::new();
    /// assert_eq!(Row.witness(&ring, WitnessColumn::new::<0>(), Rotation::NEXT), F::new(5));
    /// ```
    fn witness(&self, ring: &R, column: WitnessColumn<W>, rotation: Rotation) -> R::Elem;

    /// The value of fixed column `column` at `rotation`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::{Assignment, FixedColumn, Rotation, Scalars, WitnessColumn};
    ///
    /// /// Witness column 0 holds 4 on the current row and 5 on the next;
    /// /// the selector (fixed column 0) is 1 everywhere.
    /// struct Row;
    /// impl Assignment<F, Scalars<F>, 1, 1> for Row {
    ///     fn witness(&self, _: &Scalars<F>, _: WitnessColumn<1>, r: Rotation) -> F {
    ///         // Row offsets 0 and 1; this row reads no other.
    ///         if r == Rotation::NEXT { F::new(5) } else { F::new(4) }
    ///     }
    ///     fn fixed(&self, _: FixedColumn<1>, _: Rotation) -> F {
    ///         F::new(1)
    ///     }
    /// }
    /// assert_eq!(Row.fixed(FixedColumn::new::<0>(), Rotation::CURR), F::new(1));
    /// ```
    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F;
}

/// The value of `expr` under `assignment`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{
///     Algebra, Assignment, Expr, FixedColumn, Rotation, Scalars, WitnessColumn,
///     evaluate, evaluate_graded, evaluate_homogeneous,
/// };
///
/// /// One row: witness columns a, b and no fixed column.
/// struct Row([F; 2]);
/// impl Assignment<F, Scalars<F>, 2, 0> for Row {
///     fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
///         self.0[c.index()]
///     }
///     fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
///         unreachable!("no fixed column: {c:?}")
///     }
/// }
///
/// type E = Expr<F, 2, 0>;
/// let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
/// // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
/// let f = a.clone() * b + a + E::constant(F::ONE);
/// let row = Row([F::new(2), F::new(3)]);
/// let ring = Scalars::new();
/// assert_eq!(evaluate(&ring, &f, &row), F::new(9));
/// ```
pub fn evaluate<F, R, A, const W: usize, const X: usize>(
    ring: &R,
    expr: &Expr<F, W, X>,
    assignment: &A,
) -> R::Elem
where
    F: Field,
    R: Algebra<F>,
    A: Assignment<F, R, W, X>,
{
    match expr {
        Expr::Constant(c) => ring.embed(*c),
        Expr::Variable(Variable::Witness(column, rotation)) => {
            assignment.witness(ring, *column, *rotation)
        }
        Expr::Variable(Variable::Fixed(column, rotation)) => {
            ring.embed(assignment.fixed(*column, *rotation))
        }
        Expr::Neg(a) => ring.neg(&evaluate(ring, a, assignment)),
        Expr::Add(a, b) => ring.add(
            &evaluate(ring, a, assignment),
            &evaluate(ring, b, assignment),
        ),
        Expr::Mul(a, b) => ring.mul(
            &evaluate(ring, a, assignment),
            &evaluate(ring, b, assignment),
        ),
    }
}

/// The homogeneous components `[f_0(a), ..., f_d(a)]` of `expr` under
/// `assignment`, where `d = expr.degree()`.
///
/// The returned vector has exactly `expr.degree() + 1` entries.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{
///     Algebra, Assignment, Expr, FixedColumn, Rotation, Scalars, WitnessColumn,
///     evaluate, evaluate_graded, evaluate_homogeneous,
/// };
///
/// /// One row: witness columns a, b and no fixed column.
/// struct Row([F; 2]);
/// impl Assignment<F, Scalars<F>, 2, 0> for Row {
///     fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
///         self.0[c.index()]
///     }
///     fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
///         unreachable!("no fixed column: {c:?}")
///     }
/// }
///
/// type E = Expr<F, 2, 0>;
/// let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
/// // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
/// let f = a.clone() * b + a + E::constant(F::ONE);
/// let row = Row([F::new(2), F::new(3)]);
/// let ring = Scalars::new();
/// assert_eq!(evaluate_graded(&ring, &f, &row), vec![F::ONE, F::new(2), F::new(6)]);
/// ```
pub fn evaluate_graded<F, R, A, const W: usize, const X: usize>(
    ring: &R,
    expr: &Expr<F, W, X>,
    assignment: &A,
) -> Vec<R::Elem>
where
    F: Field,
    R: Algebra<F>,
    A: Assignment<F, R, W, X>,
{
    match expr {
        Expr::Constant(c) => vec![ring.embed(*c)],
        Expr::Variable(Variable::Witness(column, rotation)) => {
            vec![ring.zero(), assignment.witness(ring, *column, *rotation)]
        }
        Expr::Variable(Variable::Fixed(column, rotation)) => {
            vec![ring.embed(assignment.fixed(*column, *rotation))]
        }
        Expr::Neg(a) => {
            let ca = evaluate_graded(ring, a, assignment);
            let mut out = Vec::with_capacity(ca.len());
            let mut k = 0;
            while k < ca.len() {
                out.push(ring.neg(&ca[k]));
                k += 1;
            }
            out
        }
        Expr::Add(a, b) => {
            let ca = evaluate_graded(ring, a, assignment);
            let cb = evaluate_graded(ring, b, assignment);
            let len = if ca.len() >= cb.len() {
                ca.len()
            } else {
                cb.len()
            };
            let mut out = Vec::with_capacity(len);
            let mut k = 0;
            while k < len {
                let value = match (ca.get(k), cb.get(k)) {
                    (Some(x), Some(y)) => ring.add(x, y),
                    (Some(x), None) | (None, Some(x)) => x.clone(),
                    (None, None) => ring.zero(),
                };
                out.push(value);
                k += 1;
            }
            out
        }
        Expr::Mul(a, b) => {
            // Product of graded elements: (fg)_k = sum_{i+j=k} f_i g_j.
            let ca = evaluate_graded(ring, a, assignment);
            let cb = evaluate_graded(ring, b, assignment);
            let len = ca.len() + cb.len() - 1;
            let mut out = vec![ring.zero(); len];
            let mut i = 0;
            while i < ca.len() {
                let mut j = 0;
                while j < cb.len() {
                    out[i + j] = ring.add(&out[i + j], &ring.mul(&ca[i], &cb[j]));
                    j += 1;
                }
                i += 1;
            }
            out
        }
    }
}

/// The homogenization of `expr` to degree `degree`, evaluated at
/// `(u, assignment)`: `sum_k u^(degree - k) f_k(assignment)`.
///
/// Returns `None` when `expr.degree() > degree`, since the expression then
/// has no homogenization of that degree.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::{
///     Algebra, Assignment, Expr, FixedColumn, Rotation, Scalars, WitnessColumn,
///     evaluate, evaluate_graded, evaluate_homogeneous,
/// };
///
/// /// One row: witness columns a, b and no fixed column.
/// struct Row([F; 2]);
/// impl Assignment<F, Scalars<F>, 2, 0> for Row {
///     fn witness(&self, _: &Scalars<F>, c: WitnessColumn<2>, _: Rotation) -> F {
///         self.0[c.index()]
///     }
///     fn fixed(&self, c: FixedColumn<0>, _: Rotation) -> F {
///         unreachable!("no fixed column: {c:?}")
///     }
/// }
///
/// type E = Expr<F, 2, 0>;
/// let a = E::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let b = E::witness(WitnessColumn::new::<1>(), Rotation::CURR);
/// // f = a * b + a + 1, with components f_0 = 1, f_1 = a, f_2 = a * b.
/// let f = a.clone() * b + a + E::constant(F::ONE);
/// let row = Row([F::new(2), F::new(3)]);
/// let ring = Scalars::new();
/// // f^3(u, a, b) = u^3 + u^2 a + u a b; at u = 2: 8 + 8 + 12.
/// assert_eq!(evaluate_homogeneous(&ring, &f, &row, &F::new(2), 3), Some(F::new(28)));
/// assert_eq!(evaluate_homogeneous(&ring, &f, &row, &F::ONE, 1), None);
/// ```
pub fn evaluate_homogeneous<F, R, A, const W: usize, const X: usize>(
    ring: &R,
    expr: &Expr<F, W, X>,
    assignment: &A,
    u: &R::Elem,
    degree: usize,
) -> Option<R::Elem>
where
    F: Field,
    R: Algebra<F>,
    A: Assignment<F, R, W, X>,
{
    let components = evaluate_graded(ring, expr, assignment);
    let top = components.len() - 1;
    if top > degree {
        return None;
    }
    // Horner in u from the constant component up:
    // h = sum_k u^(top - k) f_k, then multiply by u^(degree - top).
    let mut h = components[0].clone();
    let mut k = 1;
    while k <= top {
        h = ring.add(&ring.mul(&h, u), &components[k]);
        k += 1;
    }
    let mut padding = degree - top;
    while padding > 0 {
        h = ring.mul(&h, u);
        padding -= 1;
    }
    Some(h)
}
