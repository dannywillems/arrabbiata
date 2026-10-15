//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn crate_overview() {
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};

    // The cycle: Pallas scalars are Vesta coordinates, and conversely.
    let pallas = Projective::<Pallas>::generator() * Fq::from_u64(7);
    let vesta = Projective::<Vesta>::generator() * Fp::from_u64(7);
    assert!(pallas.is_on_curve() && vesta.is_on_curve());
}

#[test]
fn fp_config() {
    use arrabbiata_field::MontgomeryConfig;
    use arrabbiata_pasta::FpConfig;
    assert_eq!(FpConfig::MODULUS[3], 0x4000000000000000);
}

#[test]
fn fq_config() {
    use arrabbiata_field::MontgomeryConfig;
    use arrabbiata_pasta::FqConfig;
    assert_eq!(FqConfig::MODULUS[0], 0x8c46eb2100000001);
}

#[test]
fn fp() {
    use arrabbiata_field::{Field, TwoAdicField};
    use arrabbiata_pasta::Fp;
    assert_eq!(Fp::TWO_ADICITY, 32);
    assert_eq!(Fp::from_u64(2) * Fp::from_u64(3), Fp::from_u64(6));
}

#[test]
fn fq() {
    use arrabbiata_field::TwoAdicField;
    use arrabbiata_pasta::Fq;
    assert_eq!(Fq::TWO_ADICITY, 32);
}

#[test]
fn pallas() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    assert!(Projective::<Pallas>::generator().is_on_curve());
}

#[test]
fn vesta() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Vesta;
    assert!(Projective::<Vesta>::generator().is_on_curve());
}
