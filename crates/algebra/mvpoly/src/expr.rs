//! Constraints as polynomials.
//!
//! An [`Expr`] reads witness columns at rotations and fixed columns. Its
//! polynomial on a row has one variable per pair (witness column, rotation)
//! of a [`Layout`], and the fixed columns substituted by their values on
//! that row: they are not folded, so they are coefficients, not variables.
//! The conversion is the unique ring morphism out of the free algebra of
//! expressions that sends each witness variable to its layout variable and
//! each fixed variable to its value; it is partial only because the
//! product may exceed the degree bound `D`.

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_poly::{Expr, FixedColumn, Rotation, Variable, WitnessColumn};

use crate::{MVPoly, monomial::Var};

/// The variables of a constraint polynomial: for each rotation of a list,
/// one variable per witness column. Variable `s W + k` is witness column
/// `k` at the `s`-th rotation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout<const W: usize, const N: usize> {
    rotations: Vec<Rotation>,
}

impl<const W: usize, const N: usize> Layout<W, N> {
    /// The layout of the given rotations, or `None` if one repeats or they
    /// need more than `N` variables (`W` per rotation).
    #[must_use]
    pub fn new(rotations: &[Rotation]) -> Option<Self> {
        let distinct = rotations
            .iter()
            .enumerate()
            .all(|(i, r)| !rotations[..i].contains(r));
        if !distinct || rotations.len() * W > N {
            return None;
        }
        Some(Self {
            rotations: rotations.to_vec(),
        })
    }

    /// The current row and the next one, in this order.
    #[must_use]
    pub fn current_and_next() -> Option<Self> {
        Self::new(&[Rotation::CURR, Rotation::NEXT])
    }

    /// The rotations, in the order of their variables.
    #[must_use]
    pub fn rotations(&self) -> &[Rotation] {
        &self.rotations
    }

    /// The variable of witness column `column` at `rotation`, or `None` if
    /// the layout has no slot for the rotation.
    #[must_use]
    pub fn variable(&self, column: WitnessColumn<W>, rotation: Rotation) -> Option<Var<N>> {
        let slot = self.rotations.iter().position(|r| *r == rotation)?;
        Var::try_new(slot * W + column.index())
    }

    /// The point of a row of a trace: variable `s W + k` is column `k` at
    /// the row `rotation_s(row)`, rows wrapping around. Unused variables are
    /// zero.
    #[must_use]
    pub fn point<F>(&self, columns: &[Vec<F>; W], row: usize) -> [F; N]
    where
        F: Field,
    {
        let mut x = [F::ZERO; N];
        for (s, rotation) in self.rotations.iter().enumerate() {
            for (k, column) in columns.iter().enumerate() {
                x[s * W + k] = column[rotation.apply(row, column.len())];
            }
        }
        x
    }
}

/// The polynomial of `expr` under `layout`, the fixed columns replaced by
/// `fixed(column, rotation)`; `None` if a witness rotation has no slot in
/// the layout or the degree exceeds `D`.
pub fn from_expr<F, P, Fx, const W: usize, const X: usize, const N: usize, const D: usize>(
    expr: &Expr<F, W, X>,
    layout: &Layout<W, N>,
    fixed: &Fx,
) -> Option<P>
where
    F: Field,
    P: MVPoly<F, N, D>,
    Fx: Fn(FixedColumn<X>, Rotation) -> F,
{
    match expr {
        Expr::Constant(c) => Some(P::constant(*c)),
        Expr::Variable(Variable::Witness(column, rotation)) => {
            P::variable(layout.variable(*column, *rotation)?)
        }
        Expr::Variable(Variable::Fixed(column, rotation)) => {
            Some(P::constant(fixed(*column, *rotation)))
        }
        Expr::Neg(a) => Some(-from_expr::<F, P, Fx, W, X, N, D>(a, layout, fixed)?),
        Expr::Add(a, b) => Some(
            from_expr::<F, P, Fx, W, X, N, D>(a, layout, fixed)?
                + from_expr::<F, P, Fx, W, X, N, D>(b, layout, fixed)?,
        ),
        Expr::Mul(a, b) => from_expr::<F, P, Fx, W, X, N, D>(a, layout, fixed)?
            .checked_mul(&from_expr::<F, P, Fx, W, X, N, D>(b, layout, fixed)?),
    }
}
