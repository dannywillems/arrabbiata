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
