//! Constraint expressions.
//!
//! An [`Expr`] is a term of the free commutative `F`-algebra on the
//! variables of a circuit, that is, a multivariate polynomial written as a
//! syntax tree. Evaluating it under an assignment of the variables is the
//! unique algebra morphism extending that assignment (see
//! [`crate::eval`]); every property of evaluation the folding scheme relies
//! on is a consequence of that universal property.
//!
//! Expressions are built with the ordinary operators:

use alloc::boxed::Box;
use core::ops::{Add, Mul, Neg, Sub};

use arrabbiata_field::Field;

use crate::column::{FixedColumn, Rotation, Variable, WitnessColumn};

/// A polynomial over `F` in the variables of a circuit with `W` witness
/// columns and `X` fixed columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr<F, const W: usize, const X: usize> {
    /// A constant of the field.
    Constant(F),
    /// A column read at a rotation.
    Variable(Variable<W, X>),
    /// Additive inverse.
    Neg(Box<Self>),
    /// Sum.
    Add(Box<Self>, Box<Self>),
    /// Product.
    Mul(Box<Self>, Box<Self>),
}

impl<F, const W: usize, const X: usize> Expr<F, W, X>
where
    F: Field,
{
    /// The constant `c`.
    #[must_use]
    pub const fn constant(c: F) -> Self {
        Self::Constant(c)
    }

    /// The witness column `column` at `rotation`.
    #[must_use]
    pub const fn witness(column: WitnessColumn<W>, rotation: Rotation) -> Self {
        Self::Variable(Variable::Witness(column, rotation))
    }

    /// The fixed column `column` at `rotation`.
    #[must_use]
    pub const fn fixed(column: FixedColumn<X>, rotation: Rotation) -> Self {
        Self::Variable(Variable::Fixed(column, rotation))
    }

    /// The syntactic degree in the witness variables.
    ///
    /// Constants and fixed columns have degree zero, since they are not
    /// folded. This is an upper bound on the degree of the polynomial the
    /// expression denotes (cancellations are not detected), which is the
    /// bound the folding scheme needs: graded evaluation produces exactly
    /// `degree() + 1` components.
    #[must_use]
    pub fn degree(&self) -> usize {
        match self {
            Self::Constant(_) | Self::Variable(Variable::Fixed(_, _)) => 0,
            Self::Variable(Variable::Witness(_, _)) => 1,
            Self::Neg(a) => a.degree(),
            Self::Add(a, b) => {
                let da = a.degree();
                let db = b.degree();
                if da >= db { da } else { db }
            }
            Self::Mul(a, b) => a.degree() + b.degree(),
        }
    }
}

impl<F, const W: usize, const X: usize> From<Variable<W, X>> for Expr<F, W, X> {
    fn from(variable: Variable<W, X>) -> Self {
        Self::Variable(variable)
    }
}

impl<F, const W: usize, const X: usize> Add for Expr<F, W, X> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl<F, const W: usize, const X: usize> Sub for Expr<F, W, X> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(Self::Neg(Box::new(rhs))))
    }
}

impl<F, const W: usize, const X: usize> Mul for Expr<F, W, X> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl<F, const W: usize, const X: usize> Neg for Expr<F, W, X> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }
}
