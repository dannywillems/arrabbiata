//! BN254: fields, groups, and the pairing's defining properties.

use arrabbiata_bn254::{
    Fq, FqConfig, Fr, FrConfig, G1, G2,
    fq12::FINAL_EXPONENT,
    pairing::{multi_pairing, pairing},
};
use arrabbiata_curve::Projective;
use arrabbiata_field::{Field, MontgomeryConfig, Sqrt, TwoAdicField};
use proptest::prelude::*;

/// Schoolbook product of little-endian limb vectors.
fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len()];
    for i in 0..a.len() {
        let mut carry = 0u128;
        for j in 0..b.len() {
            let t = out[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
        }
        out[i + b.len()] = carry as u64;
    }
    while out.len() > 1 && *out.last().unwrap() == 0 {
        out.pop();
    }
    out
}

/// The final exponent is exactly (p^12 - 1) / r.
#[test]
fn final_exponent_times_r_is_p12_minus_1() {
    let p = <FqConfig as MontgomeryConfig<4>>::MODULUS;
    let r = <FrConfig as MontgomeryConfig<4>>::MODULUS;
    let mut p12 = vec![1u64];
    for _ in 0..12 {
        p12 = mul(&p12, &p);
    }
    // p^12 - 1 (p^12 is odd, so no borrow beyond the first limb).
    p12[0] -= 1;
    assert_eq!(mul(&FINAL_EXPONENT, &r), p12);
}

#[test]
fn fields() {
    assert_eq!(Fq::MULTIPLICATIVE_GENERATOR.legendre(), -Fq::ONE);
    assert_eq!(Fr::MULTIPLICATIVE_GENERATOR.legendre(), -Fr::ONE);
    assert_eq!(Fr::TWO_ADICITY, 28);
    assert!(Fq::from_u64(4).sqrt().is_some());
}

/// The groups have order r, which checks p and r against each other.
#[test]
fn group_orders() {
    let g1 = Projective::<G1>::generator();
    let g2 = Projective::<G2>::generator();
    assert_eq!(g1 * (-Fr::ONE), -g1);
    assert_eq!(g2 * (-Fr::ONE), -g2);
    assert!(g1.is_on_curve() && g2.is_on_curve());
}

#[test]
fn pairing_is_non_degenerate_and_of_order_r() {
    let e = pairing(
        &Projective::<G1>::generator(),
        &Projective::<G2>::generator(),
    );
    assert!(!e.is_one());
    let r = <FrConfig as MontgomeryConfig<4>>::MODULUS;
    assert!(e.pow_limbs(&r).is_one());
}

#[test]
fn pairing_of_the_identity_is_one() {
    let g1 = Projective::<G1>::generator();
    let g2 = Projective::<G2>::generator();
    assert!(pairing(&Projective::identity(), &g2).is_one());
    assert!(pairing(&g1, &Projective::identity()).is_one());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    /// e(aP, bQ) = e(P, Q)^(ab), through e(aP, bQ) = e(abP, Q) = e(P, abQ).
    #[test]
    fn bilinearity(a in 1u64..u64::MAX, b in 1u64..u64::MAX) {
        let (a, b) = (Fr::from_u64(a), Fr::from_u64(b));
        let g1 = Projective::<G1>::generator();
        let g2 = Projective::<G2>::generator();
        let lhs = pairing(&(g1 * a), &(g2 * b));
        prop_assert_eq!(lhs, pairing(&(g1 * (a * b)), &g2));
        prop_assert_eq!(lhs, pairing(&g1, &(g2 * (a * b))));
    }

    /// Additivity in each argument.
    #[test]
    fn additivity(a in 1u64..u64::MAX, b in 1u64..u64::MAX) {
        let (a, b) = (Fr::from_u64(a), Fr::from_u64(b));
        let g1 = Projective::<G1>::generator();
        let g2 = Projective::<G2>::generator();
        let (p1, p2) = (g1 * a, g1 * b);
        prop_assert_eq!(pairing(&(p1 + p2), &g2), pairing(&p1, &g2) * pairing(&p2, &g2));
        prop_assert_eq!(multi_pairing(&[(p1, g2), (p2, g2)]), pairing(&(p1 + p2), &g2));
    }
}
