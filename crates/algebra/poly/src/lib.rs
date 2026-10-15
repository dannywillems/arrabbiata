//! Constraint expressions for Arrabbiata.
//!
//! A constraint is a polynomial in the columns of a circuit
//! ([`Expr`]). This crate provides the expressions, typed column handles
//! ([`WitnessColumn`], [`FixedColumn`]), and the evaluations the folding
//! scheme needs: plain, graded by degree, and homogenized (see [`eval`]).
//!
//! Evaluation is generic over the target [`Algebra`], so the same code
//! evaluates a constraint on a row of field elements and computes the
//! cross-terms of a fold in the polynomial ring `F[r]`.

#![no_std]

extern crate alloc;

pub mod column;
pub mod domain;
pub mod eval;
pub mod expr;
pub mod ring;

pub use column::{FixedColumn, Rotation, Variable, WitnessColumn};
pub use domain::Domain;
pub use eval::{Assignment, evaluate, evaluate_graded, evaluate_homogeneous};
pub use expr::Expr;
pub use ring::{Algebra, Polynomials, Scalars, UniPoly};
