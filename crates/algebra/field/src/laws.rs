//! The field axioms as executable predicates.
//!
//! Each function returns whether one family of axioms holds on the given
//! elements. They are written once here and reused by the property tests of
//! every concrete field, so a new field is checked against exactly the
//! axioms the Lean side assumes.

use crate::Field;

/// `0 != 1`.
#[must_use]
pub fn zero_ne_one<F>() -> bool
where
    F: Field,
{
    F::ZERO != F::ONE
}

/// `(F, +, 0)` is an abelian group with inverse `neg`, and `-` is `+ neg`.
#[must_use]
// `a + b == b + a` is the commutativity law, not a mistake: clippy reads
// the two sides as equal expressions because `+` is commutative.
#[allow(clippy::eq_op)]
pub fn additive_group<F>(a: F, b: F, c: F) -> bool
where
    F: Field,
{
    (a + b) + c == a + (b + c)
        && a + b == b + a
        && a + F::ZERO == a
        && a + (-a) == F::ZERO
        && a - b == a + (-b)
}

/// `(F, *, 1)` is a commutative monoid.
#[must_use]
pub fn multiplicative_monoid<F>(a: F, b: F, c: F) -> bool
where
    F: Field,
{
    (a * b) * c == a * (b * c) && a * b == b * a && a * F::ONE == a
}

/// `*` distributes over `+`.
#[must_use]
pub fn distributive<F>(a: F, b: F, c: F) -> bool
where
    F: Field,
{
    a * (b + c) == a * b + a * c
}

/// Every non-zero element has a two-sided inverse, and zero has none.
#[must_use]
pub fn inverse<F>(a: F) -> bool
where
    F: Field,
{
    match a.inverse() {
        None => a.is_zero(),
        Some(inv) => !a.is_zero() && a * inv == F::ONE,
    }
}
