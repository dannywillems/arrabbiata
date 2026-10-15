//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn fq12_overview() {
    use arrabbiata_bn254::fq12::Fq12;
    let w = Fq12::w();
    let w6 = w * w * w * w * w * w;
    assert_eq!(w6, Fq12::from_fq2(arrabbiata_bn254::xi()));
}

#[test]
fn fq12_fq12() {
    use arrabbiata_bn254::fq12::Fq12;
    assert_eq!(Fq12::one() * Fq12::w(), Fq12::w());
}

#[test]
fn fq12_fq12_new() {
    use arrabbiata_bn254::{Fq2, fq12::Fq12};
    use arrabbiata_field::Field;
    assert_eq!(
        Fq12::new([
            Fq2::ONE,
            Fq2::ZERO,
            Fq2::ZERO,
            Fq2::ZERO,
            Fq2::ZERO,
            Fq2::ZERO
        ]),
        Fq12::one()
    );
}

#[test]
fn fq12_fq12_coefficients() {
    use arrabbiata_bn254::fq12::Fq12;
    assert_eq!(Fq12::one().coefficients().len(), 6);
}

#[test]
fn fq12_fq12_one() {
    use arrabbiata_bn254::fq12::Fq12;
    assert!(Fq12::one().is_one());
}

#[test]
fn fq12_fq12_w() {
    use arrabbiata_bn254::fq12::Fq12;
    assert!(!Fq12::w().is_one());
}

#[test]
fn fq12_fq12_from_fq2() {
    use arrabbiata_bn254::{Fq2, fq12::Fq12};
    use arrabbiata_field::Field;
    assert_eq!(Fq12::from_fq2(Fq2::ONE), Fq12::one());
}

#[test]
fn fq12_fq12_is_one() {
    use arrabbiata_bn254::fq12::Fq12;
    assert!(!(Fq12::w() * Fq12::one()).is_one());
}

#[test]
fn fq12_fq12_pow_limbs() {
    use arrabbiata_bn254::fq12::Fq12;
    let w = Fq12::w();
    assert_eq!(w.pow_limbs(&[3]), w * w * w);
}

#[test]
fn crate_overview() {
    use arrabbiata_bn254::{Fr, G1, G2, pairing::pairing};
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;

    let (g1, g2) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    let (a, b) = (Fr::from_u64(3), Fr::from_u64(5));
    // Bilinearity: e(aP, bQ) = e(abP, Q).
    assert_eq!(pairing(&(g1 * a), &(g2 * b)), pairing(&(g1 * (a * b)), &g2));
}

#[test]
fn fq_config() {
    use arrabbiata_bn254::FqConfig;
    use arrabbiata_field::MontgomeryConfig;
    assert_eq!(FqConfig::MODULUS[3], 0x30644e72e131a029);
}

#[test]
fn fr_config() {
    use arrabbiata_bn254::FrConfig;
    use arrabbiata_field::MontgomeryConfig;
    assert_eq!(FrConfig::MODULUS[0], 0x43e1f593f0000001);
}

#[test]
fn fq() {
    use arrabbiata_bn254::Fq;
    use arrabbiata_field::TwoAdicField;
    assert_eq!(Fq::TWO_ADICITY, 1);
}

#[test]
fn fr() {
    use arrabbiata_bn254::Fr;
    use arrabbiata_field::TwoAdicField;
    assert_eq!(Fr::TWO_ADICITY, 28);
}

#[test]
fn fq2_config() {
    use arrabbiata_bn254::{Fq, Fq2Config};
    use arrabbiata_field::{Field, extension::QuadraticConfig};
    assert_eq!(Fq2Config::non_residue(), -Fq::ONE);
}

#[test]
fn fq2() {
    use arrabbiata_bn254::{Fq, Fq2};
    use arrabbiata_field::Field;
    let u = Fq2::new(Fq::ZERO, Fq::ONE);
    assert_eq!(u * u, -Fq2::ONE);
}

#[test]
fn xi() {
    use arrabbiata_bn254::{Fq, xi};
    use arrabbiata_field::Field;
    assert_eq!(xi().c0(), Fq::from_u64(9));
}

#[test]
fn g1() {
    use arrabbiata_bn254::G1;
    use arrabbiata_curve::Projective;
    assert!(Projective::<G1>::generator().is_on_curve());
}

#[test]
fn g2() {
    use arrabbiata_bn254::G2;
    use arrabbiata_curve::Projective;
    assert!(Projective::<G2>::generator().is_on_curve());
}

#[test]
fn bn254() {
    use arrabbiata_bn254::Bn254;
    use arrabbiata_curve::{Pairing, Projective};
    let (p, q) = (Projective::generator(), Projective::generator());
    assert!(Bn254::pairing_product_is_one(&[(p, q), (-p, q)]));
}

#[test]
fn pairing_overview() {
    use arrabbiata_bn254::{G1, G2, pairing::pairing};
    use arrabbiata_curve::Projective;
    let e = pairing(
        &Projective::<G1>::generator(),
        &Projective::<G2>::generator(),
    );
    assert!(!e.is_one());
}

#[test]
fn pairing_gt() {
    use arrabbiata_bn254::{
        G1, G2,
        pairing::{Gt, pairing},
    };
    use arrabbiata_curve::Projective;
    let e = pairing(
        &Projective::<G1>::generator(),
        &Projective::<G2>::generator(),
    );
    assert_eq!(e * Gt::one(), e);
}

#[test]
fn pairing_gt_one() {
    use arrabbiata_bn254::pairing::Gt;
    assert!(Gt::one().is_one());
}

#[test]
fn pairing_gt_is_one() {
    use arrabbiata_bn254::pairing::Gt;
    assert!((Gt::one() * Gt::one()).is_one());
}

#[test]
fn pairing_gt_pow_limbs() {
    use arrabbiata_bn254::{G1, G2, pairing::pairing};
    use arrabbiata_curve::Projective;
    let e = pairing(
        &Projective::<G1>::generator(),
        &Projective::<G2>::generator(),
    );
    assert_eq!(e.pow_limbs(&[2]), e * e);
}

#[test]
fn pairing_gt_as_fq12() {
    use arrabbiata_bn254::pairing::Gt;
    assert!(Gt::one().as_fq12().is_one());
}

#[test]
fn pairing_miller_loop() {
    use arrabbiata_bn254::{
        G1, G2,
        pairing::{final_exponentiation, miller_loop, pairing},
    };
    use arrabbiata_curve::Projective;
    let (p, q) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    assert_eq!(final_exponentiation(&miller_loop(&p, &q)), pairing(&p, &q));
}

#[test]
fn pairing_final_exponentiation() {
    use arrabbiata_bn254::{fq12::Fq12, pairing::final_exponentiation};
    assert!(final_exponentiation(&Fq12::one()).is_one());
}

#[test]
fn pairing_pairing() {
    use arrabbiata_bn254::{Fr, G1, G2, pairing::pairing};
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;
    let (p, q) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    let two = Fr::from_u64(2);
    assert_eq!(pairing(&(p * two), &q), pairing(&p, &(q * two)));
}

#[test]
fn pairing_multi_pairing() {
    use arrabbiata_bn254::{G1, G2, pairing::multi_pairing};
    use arrabbiata_curve::Projective;
    let (p, q) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    // e(P, Q) e(-P, Q) = 1.
    assert!(multi_pairing(&[(p, q), (-p, q)]).is_one());
}
