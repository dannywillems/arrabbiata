//! Models of the signature of fields: where a computation over a field is
//! carried out.
//!
//! [`Field`] is the signature of fields on a type of values: its operators
//! compute. Some computations must be carried out elsewhere, with effects:
//! in a circuit, each operation lays out the gate constraining its result.
//! [`FieldModel`] is the same signature, its operations taking `&mut self`
//! so that a model may have effects; a computation written once against it
//! has one meaning in each model. [`Native`] is the field itself, its
//! operations [`Field`]'s operators: every field is a model of its own
//! signature, and that model is where the others are checked against.
//!
//! A term of the signature (a constraint, a verifier's arithmetic) has one
//! image in every model, and the image is a homomorphism of the term
//! algebra: that is the universal property of the term algebra, the free
//! algebra on the variables.

use crate::Field;

/// A model of the signature of fields with constants from `F`.
///
/// The laws are the field axioms, in [`Native`]; a model with effects
/// satisfies them on the values it computes (its witness), and its effects
/// constrain them (its circuit).
pub trait FieldModel<F>
where
    F: Field,
{
    /// An element.
    type Value: Clone;

    /// The constant `c`.
    fn constant(&mut self, c: F) -> Self::Value;

    /// `a + b`.
    fn add(&mut self, a: &Self::Value, b: &Self::Value) -> Self::Value;

    /// `a b`.
    fn mul(&mut self, a: &Self::Value, b: &Self::Value) -> Self::Value;

    /// `-a`.
    fn neg(&mut self, a: &Self::Value) -> Self::Value;

    /// `a^(-1)`. In [`Native`] the inverse of zero is zero (Mathlib's
    /// total inverse); a circuit has no witness for it.
    fn inverse(&mut self, a: &Self::Value) -> Self::Value;

    /// `a - b`.
    fn sub(&mut self, a: &Self::Value, b: &Self::Value) -> Self::Value {
        let minus = self.neg(b);
        self.add(a, &minus)
    }

    /// `c a`, for a constant `c`.
    fn scale(&mut self, a: &Self::Value, c: F) -> Self::Value {
        let k = self.constant(c);
        self.mul(a, &k)
    }

    /// `a + c`, for a constant `c`.
    fn shift(&mut self, a: &Self::Value, c: F) -> Self::Value {
        let k = self.constant(c);
        self.add(a, &k)
    }

    /// `a a`.
    fn square(&mut self, a: &Self::Value) -> Self::Value {
        self.mul(a, a)
    }

    /// `a^exp`, by square-and-multiply from the most significant bit, as
    /// [`Field::pow`].
    fn pow(&mut self, a: &Self::Value, exp: u64) -> Self::Value {
        let mut acc = self.constant(F::ONE);
        let mut bit = 64 - exp.leading_zeros();
        while bit > 0 {
            bit -= 1;
            acc = self.square(&acc);
            if (exp >> bit) & 1 == 1 {
                acc = self.mul(&acc, a);
            }
        }
        acc
    }
}

/// The field itself, the model in which a computation evaluates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Native;

impl<F> FieldModel<F> for Native
where
    F: Field,
{
    type Value = F;

    fn constant(&mut self, c: F) -> F {
        c
    }

    fn add(&mut self, a: &F, b: &F) -> F {
        *a + *b
    }

    fn mul(&mut self, a: &F, b: &F) -> F {
        *a * *b
    }

    fn neg(&mut self, a: &F) -> F {
        -*a
    }

    fn inverse(&mut self, a: &F) -> F {
        a.inverse().unwrap_or(F::ZERO)
    }

    fn square(&mut self, a: &F) -> F {
        a.square()
    }

    fn pow(&mut self, a: &F, exp: u64) -> F {
        a.pow(exp)
    }
}
