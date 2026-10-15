//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn homomorphic_commitment() {
    use arrabbiata_commit::HomomorphicCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let (a, b, s) = (
        [Fq::from_u64(1), Fq::from_u64(2)],
        [Fq::from_u64(3), Fq::from_u64(4)],
        Fq::from_u64(5),
    );
    let folded = [a[0] + s * b[0], a[1] + s * b[1]];
    // The verifier folds commitments as the prover folds vectors.
    assert_eq!(
        ck.commit(&folded),
        ck.add(&ck.commit(&a), &ck.scale(&ck.commit(&b), s))
    );
}

#[test]
fn commit() {
    use arrabbiata_commit::HomomorphicCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    assert_eq!(ck.commit(&[Fq::ONE]), ck.commit(&[Fq::ONE, Fq::ZERO]));
}

#[test]
fn add() {
    use arrabbiata_commit::HomomorphicCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let c = ck.add(&ck.commit(&[Fq::ONE]), &ck.commit(&[Fq::ONE]));
    assert_eq!(c, ck.commit(&[Fq::from_u64(2)]));
}

#[test]
fn scale() {
    use arrabbiata_commit::HomomorphicCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let c = ck.scale(&ck.commit(&[Fq::from_u64(3)]), Fq::from_u64(2));
    assert_eq!(c, ck.commit(&[Fq::from_u64(6)]));
}

#[test]
fn pcs_overview() {
    use arrabbiata_commit::{Multilinear, PolynomialFamily, Univariate};
    use arrabbiata_field::Goldilocks as F;

    // The reference semantics of the two families.
    let p = [F::new(1), F::new(2)]; // 1 + 2X
    assert_eq!(
        <Univariate as PolynomialFamily<F>>::evaluate(&p, &F::new(5)),
        F::new(11)
    );
    let table = [F::new(3), F::new(7)]; // f(0) = 3, f(1) = 7
    let v = <Multilinear as PolynomialFamily<F>>::evaluate(&table, &[F::new(2)][..]);
    assert_eq!(v, F::new(11)); // 3 + 2 * (7 - 3)
}

#[test]
fn pcs_polynomial_family() {
    use arrabbiata_commit::{PolynomialFamily, Univariate};
    use arrabbiata_field::Goldilocks as F;
    // 1 + X + X^2 at 2.
    let v = <Univariate as PolynomialFamily<F>>::evaluate(&[F::new(1); 3], &F::new(2));
    assert_eq!(v, F::new(7));
}

#[test]
fn pcs_evaluate() {
    use arrabbiata_commit::{Multilinear, PolynomialFamily};
    use arrabbiata_field::Goldilocks as F;
    // On the hypercube, the multilinear extension is the table.
    let table = [F::new(1), F::new(2), F::new(3), F::new(4)];
    let at = |b0: u64, b1: u64| {
        <Multilinear as PolynomialFamily<F>>::evaluate(&table, &[F::new(b0), F::new(b1)][..])
    };
    assert_eq!(at(1, 0), F::new(3));
}

#[test]
fn pcs_univariate() {
    use arrabbiata_commit::{PolynomialFamily, Univariate};
    use arrabbiata_field::Goldilocks as F;
    assert_eq!(
        <Univariate as PolynomialFamily<F>>::evaluate(&[F::new(4)], &F::new(9)),
        F::new(4)
    );
}

#[test]
fn pcs_multilinear() {
    use arrabbiata_commit::{Multilinear, PolynomialFamily};
    use arrabbiata_field::{Field, Goldilocks as F};
    // f(b0, b1) with table [f(0,0), f(0,1), f(1,0), f(1,1)] = [0, 1, 1, 0]
    // (XOR), at (1/2, 1/2) is the average, 1/2.
    let half = F::new(2).inverse().unwrap();
    let table = [F::ZERO, F::ONE, F::ONE, F::ZERO];
    assert_eq!(
        <Multilinear as PolynomialFamily<F>>::evaluate(&table, &[half, half][..]),
        half
    );
}

#[test]
fn pcs_polynomial_commitment() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_commit::{PolynomialCommitment, Univariate};
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;

    fn check<PC>(pc: &PC, p: &[Fr], x: Fr) -> bool
    where
        PC: PolynomialCommitment<Fr, Family = Univariate>,
    {
        let (v, proof) = pc.open(p, &x);
        pc.verify(&pc.commit(p), &x, v, &proof)
    }
    let kzg = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    assert!(check(&kzg, &[Fr::ONE, Fr::ONE], Fr::from_u64(3)));
}

#[test]
fn pcs_multilinear_commit() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_commit::PolynomialCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    // A test setup: its secret is known.
    let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    // A constant polynomial commits to a multiple of the generator.
    let c = pc.commit(&[Fr::from_u64(4)]);
    assert_eq!(c, pc.commit(&[Fr::from_u64(4), Fr::ZERO]));
}

#[test]
fn pcs_multilinear_open() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_commit::PolynomialCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    // A test setup: its secret is known.
    let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    assert_eq!(
        pc.open(&[Fr::ONE, Fr::ONE], &Fr::from_u64(9)).0,
        Fr::from_u64(10)
    );
}

#[test]
fn pcs_multilinear_verify() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_commit::PolynomialCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_kzg::Kzg;
    // A test setup: its secret is known.
    let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    let p = [Fr::from_u64(5)];
    let (v, proof) = pc.open(&p, &Fr::ZERO);
    assert!(pc.verify(&pc.commit(&p), &Fr::ZERO, v, &proof));
    assert!(!pc.verify(&pc.commit(&p), &Fr::ZERO, v + Fr::ONE, &proof));
}
