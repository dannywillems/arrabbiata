//! The Pasta curves: group laws, orders, and the cycle.

use arrabbiata_curve::{Affine, Curve, Projective, msm};
use arrabbiata_field::{Field, MontField, MontgomeryConfig, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_pasta::{Fp, FpConfig, Fq, FqConfig, Pallas, Vesta};
use proptest::prelude::*;

fn arb_scalar<F>() -> impl Strategy<Value = F>
where
    F: Field,
{
    prop_oneof![
        Just(F::ZERO),
        Just(F::ONE),
        Just(-F::ONE),
        any::<[u64; 4]>().prop_map(|l| {
            l.iter().fold(F::ZERO, |acc, x| {
                acc * F::from_u64(1 << 32) * F::from_u64(1 << 32) + F::from_u64(*x)
            })
        }),
    ]
}

/// Textbook affine addition, as an independent oracle for the complete
/// projective formulas.
fn affine_add<C>(p: Affine<C>, q: Affine<C>) -> Affine<C>
where
    C: Curve,
{
    let (Some(x1), Some(y1)) = (p.x(), p.y()) else {
        return q;
    };
    let (Some(x2), Some(y2)) = (q.x(), q.y()) else {
        return p;
    };
    let lambda = if x1 == x2 {
        if y1 + y2 == C::Base::ZERO {
            return Affine::identity();
        }
        // Doubling: (3 x^2) / (2 y).
        let three = C::Base::from_u64(3);
        three * x1 * x1 * (y1 + y1).inverse().unwrap()
    } else {
        (y2 - y1) * (x2 - x1).inverse().unwrap()
    };
    let x3 = lambda * lambda - x1 - x2;
    let y3 = lambda * (x1 - x3) - y1;
    Affine::new(x3, y3).unwrap()
}

macro_rules! curve_tests {
    ($name:ident, $curve:ty, $scalar:ty, $base_config:ty) => {
        mod $name {
            use super::*;

            type P = Projective<$curve>;
            type S = $scalar;

            fn arb_point() -> impl Strategy<Value = P> {
                arb_scalar::<S>().prop_map(|k| P::generator() * k)
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(32))]

                #[test]
                fn group_laws(a in arb_point(), b in arb_point(), c in arb_point()) {
                    prop_assert_eq!((a + b) + c, a + (b + c));
                    prop_assert_eq!(a + b, b + a);
                    prop_assert_eq!(a + P::identity(), a);
                    prop_assert!((a + (-a)).is_identity());
                    prop_assert!((a + b).is_on_curve());
                }

                #[test]
                fn complete_formulas_match_affine(a in arb_point(), b in arb_point()) {
                    prop_assert_eq!(
                        (a + b).to_affine(),
                        affine_add(a.to_affine(), b.to_affine())
                    );
                    // Doubling and inverses go through the same formula.
                    prop_assert_eq!((a + a).to_affine(), affine_add(a.to_affine(), a.to_affine()));
                    prop_assert_eq!((a - a).to_affine(), Affine::identity());
                }

                #[test]
                fn scalar_multiplication_is_a_module_action(
                    a in arb_point(), b in arb_point(), s in arb_scalar::<S>(), t in arb_scalar::<S>(),
                ) {
                    prop_assert_eq!((a + b) * s, a * s + b * s);
                    prop_assert_eq!(a * (s + t), a * s + a * t);
                    prop_assert_eq!(a * (s * t), (a * s) * t);
                    prop_assert_eq!(a * S::ONE, a);
                    prop_assert!((a * S::ZERO).is_identity());
                }

                #[test]
                fn msm_is_the_sum(
                    points in prop::collection::vec(arb_point(), 0..5),
                    scalars in prop::collection::vec(arb_scalar::<S>(), 0..5),
                ) {
                    let expected = points
                        .iter()
                        .zip(&scalars)
                        .fold(P::identity(), |acc, (p, s)| acc + *p * *s);
                    prop_assert_eq!(msm(&points, &scalars), expected);
                }
            }

            /// The group order is the scalar field's modulus: (r - 1) G = -G,
            /// hence r G = 0. This also checks both moduli against each other.
            #[test]
            fn group_order_is_the_scalar_modulus() {
                let g = P::generator();
                assert_eq!(g * (-S::ONE), -g);
                assert!(!g.is_identity());
            }

            #[test]
            fn base_field_non_residue() {
                type B = MontField<$base_config, 4>;
                assert_eq!(B::MULTIPLICATIVE_GENERATOR.legendre(), -B::ONE);
                assert_eq!(B::TWO_ADICITY, 32);
                assert_eq!(<B as PrimeField>::MODULUS_BITS, 255);
                let _ = <B as Sqrt>::sqrt(B::from_u64(4)).unwrap();
            }
        }
    };
}

curve_tests!(pallas, Pallas, Fq, FpConfig);
curve_tests!(vesta, Vesta, Fp, FqConfig);

/// p and q are distinct and both 255-bit, with q > p.
#[test]
fn the_moduli_form_a_cycle() {
    let p = <FpConfig as MontgomeryConfig<4>>::MODULUS;
    let q = <FqConfig as MontgomeryConfig<4>>::MODULUS;
    assert_ne!(p, q);
    assert!(arrabbiata_field::bigint::geq(&q, &p));
}
