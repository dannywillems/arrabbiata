//! Grumpkin-specific checks. The group laws are tested by the generic curve
//! laws (`crates/ec/curve/tests/laws.rs`), its order among them.

use arrabbiata_bn254::{Fq, Fr, G1};
use arrabbiata_curve::{Curve, Projective, laws};
use arrabbiata_field::{Field, Sqrt};
use arrabbiata_grumpkin::Grumpkin;

/// The generator is `(1, y)` with `y^2 = -16`, and it has the order of the
/// BN254 base field.
#[test]
fn generator() {
    let (x, y) = Grumpkin::generator();
    assert_eq!(x, Fr::ONE);
    assert_eq!(y * y, -Fr::from_u64(16));
    assert!(laws::generator_has_order_r::<Grumpkin>());
}

/// The cycle with BN254: each curve's scalar field is the other's base
/// field.
#[test]
fn cycle_with_bn254() {
    fn base<C: Curve>() -> core::any::TypeId
    where
        C::Base: 'static,
    {
        core::any::TypeId::of::<C::Base>()
    }
    fn scalar<C: Curve>() -> core::any::TypeId
    where
        C::Scalar: 'static,
    {
        core::any::TypeId::of::<C::Scalar>()
    }
    assert_eq!(base::<Grumpkin>(), scalar::<G1>());
    assert_eq!(scalar::<Grumpkin>(), base::<G1>());
}

/// `-17` is not a square in `Fr`, so `y^2 = -17` has no solution: no point
/// has `x = 0`.
#[test]
fn no_point_with_zero_abscissa() {
    assert!((-Fr::from_u64(17)).sqrt().is_none());
}

/// Known multiples: `[1] G = G`, `[2] G = G + G`, `[r] G = 0`.
#[test]
fn known_multiples() {
    let g = Projective::<Grumpkin>::generator();
    assert_eq!(g * Fq::ONE, g);
    assert_eq!(g * Fq::from_u64(2), g + g);
    assert_eq!(g * Fq::from_u64(2), g.double());
    assert!((g * (-Fq::ONE) + g).is_identity());
}
