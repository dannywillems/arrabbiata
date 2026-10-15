//! The builder's variables: an expression over cells, with its value when a
//! witness is being computed.
//!
//! A [`Wire`] is a pair: a term over the cells, read as a polynomial in
//! `F[cells]` (the constraint reading), and an optional field element (the
//! witness reading). Its operations act on both components at once, so the
//! two readings of a gadget cannot drift apart: the value is the polynomial
//! evaluated at the witness, because evaluation is a ring homomorphism
//! `F[cells] -> F`.
//!
//! Terms are simplified as Kimchi's `Expr` simplifies them while building:
//! constants fold, `0 + e = e`, `0 e = 0` and `1 e = e`. These are ring
//! identities, so the polynomial does not change; what changes is the
//! syntactic degree (Kimchi's `Expr::degree`), which then agrees with Kimchi's
//! on the same construction (for example a Horner evaluation starting from
//! `0`).

use alloc::{boxed::Box, collections::BTreeMap, vec::Vec};
use core::ops::{Add, Mul, Neg, Sub};

use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::Variable;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_poly::Rotation;

/// A polynomial expression over the cells of the table, rows absolute until
/// [`Expr::relative_to`] makes them rotations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Expr<F> {
    Constant(F),
    Witness {
        column: usize,
        row: usize,
    },
    Fixed {
        column: usize,
        row: usize,
    },
    /// An input of the circuit not yet placed in a cell.
    Input(usize),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Neg(Box<Self>),
}

/// A column variable of a gate: a column at a rotation of the gate's row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Atom {
    Witness { column: usize, rotation: usize },
    Fixed { column: usize, rotation: usize },
}

impl Atom {
    pub(crate) fn variable(self) -> Variable {
        let rotation = |r: usize| Rotation::new(i32::try_from(r).unwrap_or(i32::MAX));
        match self {
            Self::Witness {
                column,
                rotation: r,
            } => Variable::Witness {
                column,
                rotation: rotation(r),
            },
            Self::Fixed {
                column,
                rotation: r,
            } => Variable::Fixed {
                column,
                rotation: rotation(r),
            },
        }
    }
}

impl<F> Expr<F>
where
    F: Field,
{
    /// The smallest row of a cell the expression reads.
    pub(crate) fn min_row(&self) -> Option<usize> {
        match self {
            Self::Constant(_) | Self::Input(_) => None,
            Self::Witness { row, .. } | Self::Fixed { row, .. } => Some(*row),
            Self::Add(a, b) | Self::Mul(a, b) => match (a.min_row(), b.min_row()) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, y) => x.or(y),
            },
            Self::Neg(a) => a.min_row(),
        }
    }

    /// Whether the expression reads an input that was never placed.
    pub(crate) fn reads_input(&self) -> bool {
        match self {
            Self::Input(_) => true,
            Self::Constant(_) | Self::Witness { .. } | Self::Fixed { .. } => false,
            Self::Add(a, b) | Self::Mul(a, b) => a.reads_input() || b.reads_input(),
            Self::Neg(a) => a.reads_input(),
        }
    }

    /// The expression with every row taken relative to `anchor`.
    pub(crate) fn relative_to(&self, anchor: usize) -> Self {
        match self {
            Self::Constant(c) => Self::Constant(*c),
            Self::Input(i) => Self::Input(*i),
            Self::Witness { column, row } => Self::Witness {
                column: *column,
                row: row - anchor,
            },
            Self::Fixed { column, row } => Self::Fixed {
                column: *column,
                row: row - anchor,
            },
            Self::Add(a, b) => Self::Add(
                Box::new(a.relative_to(anchor)),
                Box::new(b.relative_to(anchor)),
            ),
            Self::Mul(a, b) => Self::Mul(
                Box::new(a.relative_to(anchor)),
                Box::new(b.relative_to(anchor)),
            ),
            Self::Neg(a) => Self::Neg(Box::new(a.relative_to(anchor))),
        }
    }

    /// The total degree in the cells.
    pub(crate) fn degree(&self) -> usize {
        match self {
            Self::Constant(_) => 0,
            Self::Input(_) | Self::Witness { .. } | Self::Fixed { .. } => 1,
            Self::Add(a, b) => a.degree().max(b.degree()),
            Self::Mul(a, b) => a.degree() + b.degree(),
            Self::Neg(a) => a.degree(),
        }
    }

    /// The column variables of a relative expression.
    pub(crate) fn atoms(&self, out: &mut Vec<Atom>) {
        match self {
            Self::Constant(_) | Self::Input(_) => {}
            Self::Witness { column, row } => out.push(Atom::Witness {
                column: *column,
                rotation: *row,
            }),
            Self::Fixed { column, row } => out.push(Atom::Fixed {
                column: *column,
                rotation: *row,
            }),
            Self::Add(a, b) | Self::Mul(a, b) => {
                a.atoms(out);
                b.atoms(out);
            }
            Self::Neg(a) => a.atoms(out),
        }
    }

    /// The relative expression as a polynomial in the variables `index`
    /// numbers; `None` if it has degree more than `D` or reads an atom
    /// `index` does not number.
    pub(crate) fn to_sparse<const N: usize, const D: usize>(
        &self,
        index: &BTreeMap<Atom, usize>,
    ) -> Option<Sparse<F, N, D>> {
        let atom = |a: Atom| {
            let i = *index.get(&a)?;
            Sparse::variable(Var::try_new(i)?)
        };
        match self {
            Self::Constant(c) => Some(Sparse::constant(*c)),
            Self::Input(_) => None,
            Self::Witness { column, row } => atom(Atom::Witness {
                column: *column,
                rotation: *row,
            }),
            Self::Fixed { column, row } => atom(Atom::Fixed {
                column: *column,
                rotation: *row,
            }),
            Self::Add(a, b) => Some(a.to_sparse(index)? + b.to_sparse(index)?),
            Self::Mul(a, b) => a
                .to_sparse::<N, D>(index)?
                .checked_mul(&b.to_sparse(index)?),
            Self::Neg(a) => Some(-a.to_sparse(index)?),
        }
    }
}

/// A variable of the builder: an expression and, when computing a witness,
/// its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wire<F> {
    pub(crate) expr: Expr<F>,
    pub(crate) value: Option<F>,
}

impl<F> Wire<F>
where
    F: Field,
{
    pub(crate) const fn new(expr: Expr<F>, value: Option<F>) -> Self {
        Self { expr, value }
    }

    /// The value, when a witness is being computed.
    #[must_use]
    pub const fn value(&self) -> Option<F> {
        self.value
    }

    /// The cell the wire is, if it is a single witness cell.
    pub(crate) const fn cell(&self) -> Option<(usize, usize)> {
        match self.expr {
            Expr::Witness { column, row } => Some((column, row)),
            _ => None,
        }
    }
}

fn zip<F: Copy>(a: Option<F>, b: Option<F>, f: impl FnOnce(F, F) -> F) -> Option<F> {
    Some(f(a?, b?))
}

impl<F> Add for Wire<F>
where
    F: Field,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let value = zip(self.value, other.value, |a, b| a + b);
        let expr = match (self.expr, other.expr) {
            (Expr::Constant(a), Expr::Constant(b)) => Expr::Constant(a + b),
            (Expr::Constant(z), e) | (e, Expr::Constant(z)) if z == F::ZERO => e,
            (a, b) => Expr::Add(Box::new(a), Box::new(b)),
        };
        Self { expr, value }
    }
}

impl<F> Mul for Wire<F>
where
    F: Field,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let value = zip(self.value, other.value, |a, b| a * b);
        let expr = match (self.expr, other.expr) {
            (Expr::Constant(a), Expr::Constant(b)) => Expr::Constant(a * b),
            (Expr::Constant(z), _) | (_, Expr::Constant(z)) if z == F::ZERO => {
                Expr::Constant(F::ZERO)
            }
            (Expr::Constant(o), e) | (e, Expr::Constant(o)) if o == F::ONE => e,
            (a, b) => Expr::Mul(Box::new(a), Box::new(b)),
        };
        Self { expr, value }
    }
}

impl<F> Neg for Wire<F>
where
    F: Field,
{
    type Output = Self;

    fn neg(self) -> Self {
        let value = self.value.map(|a| -a);
        let expr = match self.expr {
            Expr::Constant(a) => Expr::Constant(-a),
            a => Expr::Neg(Box::new(a)),
        };
        Self { expr, value }
    }
}

impl<F> Sub for Wire<F>
where
    F: Field,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}
