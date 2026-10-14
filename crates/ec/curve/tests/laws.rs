//! The laws of `arrabbiata_curve::laws`, property-tested on every curve of
//! the workspace by one generic runner. CI runs each curve as its own job:
//! `make test-curve CURVE=<name>`.

use arrabbiata_bn254::{G1, G2};
use arrabbiata_curve::{Curve, Projective, laws};
use arrabbiata_field::Field;
use arrabbiata_pasta::{Pallas, Vesta};
use proptest::{
    prelude::*,
    test_runner::{Config, TestRunner},
};

/// A scalar from four limbs (Horner in base 2^64), plus the edge cases.
fn arb_scalar<F>() -> impl Strategy<Value = F>
where
    F: Field,
{
    let two_64 = F::from_u64(1 << 32) * F::from_u64(1 << 32);
    prop_oneof![
        Just(F::ZERO),
        Just(F::ONE),
        Just(-F::ONE),
        Just(F::from_u64(2)),
        any::<[u64; 4]>().prop_map(move |l| l
            .iter()
            .fold(F::ZERO, |acc, x| acc * two_64 + F::from_u64(*x))),
    ]
}

/// A point as a multiple of the generator, including the identity and the
/// generator itself.
fn arb_point<C>() -> impl Strategy<Value = Projective<C>>
where
    C: Curve,
{
    arb_scalar::<C::Scalar>().prop_map(|k| Projective::<C>::generator() * k)
}

/// Every law, on `cases` random inputs.
fn check_laws<C>(cases: u32)
where
    C: Curve,
{
    assert!(
        laws::generator_has_order_r::<C>(),
        "the generator has order r"
    );
    let mut runner = TestRunner::new(Config::with_cases(cases));
    runner
        .run(
            &(arb_point::<C>(), arb_point::<C>(), arb_point::<C>()),
            |(a, b, c)| {
                prop_assert!(laws::abelian_group(a, b, c), "abelian group");
                prop_assert!(
                    laws::complete_formula_is_affine_addition(a, b),
                    "complete formula"
                );
                prop_assert!(laws::affine_round_trip(a), "affine round trip");
                Ok(())
            },
        )
        .unwrap();
    runner
        .run(
            &(
                arb_point::<C>(),
                arb_point::<C>(),
                arb_scalar::<C::Scalar>(),
                arb_scalar::<C::Scalar>(),
            ),
            |(a, b, s, t)| {
                prop_assert!(laws::module_action(a, b, s, t), "module action");
                Ok(())
            },
        )
        .unwrap();
    runner
        .run(
            &(
                prop::collection::vec(arb_point::<C>(), 0..6),
                prop::collection::vec(arb_scalar::<C::Scalar>(), 0..6),
            ),
            |(points, scalars)| {
                prop_assert!(laws::msm_is_the_sum(&points, &scalars), "msm");
                Ok(())
            },
        )
        .unwrap();
    // Larger inputs, for the wider Pippenger windows.
    let mut runner = TestRunner::new(Config::with_cases(4));
    runner
        .run(
            &(
                prop::collection::vec(arb_point::<C>(), 32..200),
                prop::collection::vec(arb_scalar::<C::Scalar>(), 32..200),
            ),
            |(points, scalars)| {
                prop_assert!(laws::msm_is_the_sum(&points, &scalars), "msm");
                Ok(())
            },
        )
        .unwrap();
}

const CASES: u32 = 64;

#[test]
fn pallas() {
    check_laws::<Pallas>(CASES);
}

#[test]
fn vesta() {
    check_laws::<Vesta>(CASES);
}

#[test]
fn bn254_g1() {
    check_laws::<G1>(CASES);
}

#[test]
fn bn254_g2() {
    check_laws::<G2>(CASES);
}
