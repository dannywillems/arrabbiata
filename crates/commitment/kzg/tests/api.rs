//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn crate_overview() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_commit::PolynomialCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;

    // A test setup whose secret is known: never use it in a deployment.
    let kzg = Kzg::<Bn254>::insecure_setup(8, Fr::from_u64(123_456_789));
    let p = [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
    let c = kzg.commit(&p);
    let (v, proof) = kzg.open(&p, &Fr::from_u64(2));
    assert_eq!(v, Fr::from_u64(17));
    assert!(kzg.verify(&c, &Fr::from_u64(2), v, &proof));
}

#[test]
fn kzg() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    assert_eq!(Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(5)).size(), 4);
}

#[test]
fn kzg_insecure_setup() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    let _ = Kzg::<Bn254>::insecure_setup(2, Fr::from_u64(42));
}

#[test]
fn kzg_from_powers() {
    use arrabbiata_bn254::{Bn254, Fr, G1, G2};
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    let tau = Fr::from_u64(9);
    let (g, h) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    let kzg = Kzg::<Bn254>::from_powers(vec![g, g * tau], h, h * tau);
    assert_eq!(kzg.size(), 2);
}

#[test]
fn kzg_size() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    assert_eq!(
        Kzg::<Bn254>::insecure_setup(16, Fr::ONE + Fr::ONE).size(),
        16
    );
}
