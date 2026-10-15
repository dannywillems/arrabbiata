//! Pasta-specific checks. The group laws of Pallas and Vesta are tested by
//! the generic curve laws (`crates/ec/curve/tests/laws.rs`).

use arrabbiata_field::{Field, MontField, MontgomeryConfig, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_pasta::{FpConfig, FqConfig};

fn base_field<C>()
where
    C: MontgomeryConfig<4>,
{
    type B<C> = MontField<C, 4>;
    assert_eq!(B::<C>::MULTIPLICATIVE_GENERATOR.legendre(), -B::<C>::ONE);
    assert_eq!(B::<C>::TWO_ADICITY, 32);
    assert_eq!(<B<C> as PrimeField>::MODULUS_BITS, 255);
    assert!(B::<C>::from_u64(4).sqrt().is_some());
}

#[test]
fn fields() {
    base_field::<FpConfig>();
    base_field::<FqConfig>();
}

/// p and q are distinct and both 255-bit, with q > p.
#[test]
fn the_moduli_form_a_cycle() {
    let p = <FpConfig as MontgomeryConfig<4>>::MODULUS;
    let q = <FqConfig as MontgomeryConfig<4>>::MODULUS;
    assert_ne!(p, q);
    assert!(arrabbiata_field::bigint::geq(&q, &p));
}

/// The GLV constants are Kimchi's `endos` (o1-labs/proof-systems
/// `poly_commitment::ipa::endos`, as Kimchi printed them at `9227444890`):
/// `(endo_q, endo_r)` is `(beta, lambda)`, for both curves.
#[test]
fn glv_constants_are_kimchis_endos() {
    use arrabbiata_curve::glv::Glv;
    use arrabbiata_field::PrimeField;
    use arrabbiata_pasta::{Pallas, Vesta};
    let limbs = |x: &dyn Fn(usize) -> u64| [x(0), x(1), x(2), x(3)];
    let pallas_beta = Pallas::beta();
    let pallas_lambda = Pallas::lambda();
    let vesta_beta = Vesta::beta();
    let vesta_lambda = Vesta::lambda();
    assert_eq!(
        limbs(&|i| pallas_beta.limb(i)),
        [
            0x7b7fd22f0201b547,
            0x05270d29d19fc7d2,
            0xd3552a23a8554e50,
            0x2d33357cb532458e
        ]
    );
    assert_eq!(
        limbs(&|i| pallas_lambda.limb(i)),
        [
            0x619d1840af55f1b1,
            0x1259527ec1d4752e,
            0xaee24b27e308f0a6,
            0x397e65a7d7c1ad71
        ]
    );
    assert_eq!(
        limbs(&|i| vesta_beta.limb(i)),
        [
            0x2aa9d2e050aa0e4f,
            0x0fed467d47c033af,
            0x511db4d81cf70f5a,
            0x06819a58283e528e
        ]
    );
    assert_eq!(
        limbs(&|i| vesta_lambda.limb(i)),
        [
            0x1dad5ebdfdfe4ab9,
            0x1d1f8bd237ad3149,
            0x2caad5dc57aab1b0,
            0x12ccca834acdba71
        ]
    );
}
