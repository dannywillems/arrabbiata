//! The laws of a prime-order elliptic-curve group, as executable
//! predicates.
//!
//! Each function returns whether one family of laws holds on the given
//! points and scalars. They are written once, generic over [`Curve`], and
//! checked by the property tests of every curve of the workspace (see
//! `crates/ec/curve/tests/laws.rs`, run per curve in CI), so a new curve is
//! held to exactly the same laws.

use arrabbiata_field::Field;

use crate::{Affine, Curve, Projective, msm};

/// `(E, +, O)` is an abelian group, and every result is on the curve.
#[must_use]
// `a + b == b + a` is the commutativity law, not a mistake: clippy reads
// the two sides as equal expressions because `+` is commutative.
#[allow(clippy::eq_op)]
pub fn abelian_group<C>(a: Projective<C>, b: Projective<C>, c: Projective<C>) -> bool
where
    C: Curve,
{
    let o = Projective::<C>::identity();
    (a + b) + c == a + (b + c)
        && a + b == b + a
        && a + o == a
        && (a + (-a)).is_identity()
        && a - b == a + (-b)
        && (a + b).is_on_curve()
        && (-a).is_on_curve()
}

/// The textbook affine sum, with its special cases (identity, inverse
/// points, doubling): the independent oracle for the complete formulas.
#[must_use]
pub fn affine_add<C>(p: Affine<C>, q: Affine<C>) -> Affine<C>
where
    C: Curve,
{
    let (x1, y1) = match (p.x(), p.y()) {
        (Some(x), Some(y)) => (x, y),
        _ => return q,
    };
    let (x2, y2) = match (q.x(), q.y()) {
        (Some(x), Some(y)) => (x, y),
        _ => return p,
    };
    let lambda = if x1 == x2 {
        if (y1 + y2).is_zero() {
            return Affine::identity();
        }
        let three = C::Base::from_u64(3);
        match (y1 + y1).inverse() {
            Some(inverse) => three * x1 * x1 * inverse,
            None => return Affine::identity(),
        }
    } else {
        match (x2 - x1).inverse() {
            Some(inverse) => (y2 - y1) * inverse,
            None => return Affine::identity(),
        }
    };
    let x3 = lambda * lambda - x1 - x2;
    let y3 = lambda * (x1 - x3) - y1;
    match Affine::new(x3, y3) {
        Some(r) => r,
        None => Affine::identity(),
    }
}

/// The complete projective formula agrees with the affine one, including on
/// doubling and on inverse points.
#[must_use]
pub fn complete_formula_is_affine_addition<C>(a: Projective<C>, b: Projective<C>) -> bool
where
    C: Curve,
{
    let (pa, pb) = (a.to_affine(), b.to_affine());
    (a + b).to_affine() == affine_add(pa, pb)
        && (a + a).to_affine() == affine_add(pa, pa)
        && (a + (-a)).to_affine() == affine_add(pa, (-a).to_affine())
}

/// Scalars act on points: `E` is a module over the scalar field.
#[must_use]
pub fn module_action<C>(a: Projective<C>, b: Projective<C>, s: C::Scalar, t: C::Scalar) -> bool
where
    C: Curve,
{
    (a + b) * s == a * s + b * s
        && a * (s + t) == a * s + a * t
        && a * (s * t) == (a * s) * t
        && a * C::Scalar::ONE == a
        && (a * C::Scalar::ZERO).is_identity()
        && a * (-C::Scalar::ONE) == -a
        && a * C::Scalar::from_u64(2) == a.double()
}

/// The generator has order the scalar modulus `r`: `(r - 1) G = -G`, and
/// `G` is not the identity. Since `r` is prime, every non-identity point
/// then has order `r`.
#[must_use]
pub fn generator_has_order_r<C>() -> bool
where
    C: Curve,
{
    let g = Projective::<C>::generator();
    g.is_on_curve() && !g.is_identity() && g * (-C::Scalar::ONE) == -g
}

/// Affine and projective coordinates describe the same point.
#[must_use]
pub fn affine_round_trip<C>(a: Projective<C>) -> bool
where
    C: Curve,
{
    let affine = a.to_affine();
    affine.to_projective() == a
        && affine.is_identity() == a.is_identity()
        && match (affine.x(), affine.y()) {
            (Some(x), Some(y)) => Affine::<C>::new(x, y) == Some(affine),
            _ => a.is_identity(),
        }
}

/// The multi-scalar multiplication is the sum of the products.
#[must_use]
pub fn msm_is_the_sum<C>(points: &[Projective<C>], scalars: &[C::Scalar]) -> bool
where
    C: Curve,
{
    let mut expected = Projective::<C>::identity();
    let mut i = 0;
    while i < points.len() && i < scalars.len() {
        expected = expected + points[i] * scalars[i];
        i += 1;
    }
    msm(points, scalars) == expected
}
