//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_commit::{
    HomomorphicCommitment, Multilinear, PolynomialCommitment, PolynomialFamily, Univariate,
};
use arrabbiata_field::{Field, Goldilocks as F};
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};

#[test]
fn homomorphic_commitment() {
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
        HomomorphicCommitment::commit(&ck, &folded),
        ck.add(
            &HomomorphicCommitment::commit(&ck, &a),
            &ck.scale(&HomomorphicCommitment::commit(&ck, &b), s)
        )
    );
}

#[test]
fn commit() {
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    assert_eq!(
        HomomorphicCommitment::commit(&ck, &[Fq::ONE]),
        HomomorphicCommitment::commit(&ck, &[Fq::ONE, Fq::ZERO])
    );
}

#[test]
fn add() {
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let c = ck.add(
        &HomomorphicCommitment::commit(&ck, &[Fq::ONE]),
        &HomomorphicCommitment::commit(&ck, &[Fq::ONE]),
    );
    assert_eq!(c, HomomorphicCommitment::commit(&ck, &[Fq::from_u64(2)]));
}

#[test]
fn scale() {
    // Pedersen vector commitments on Pallas (the commitment of IPA).
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let c = ck.scale(
        &HomomorphicCommitment::commit(&ck, &[Fq::from_u64(3)]),
        Fq::from_u64(2),
    );
    assert_eq!(c, HomomorphicCommitment::commit(&ck, &[Fq::from_u64(6)]));
}

#[test]
fn pcs_overview() {
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
    // 1 + X + X^2 at 2.
    let v = <Univariate as PolynomialFamily<F>>::evaluate(&[F::new(1); 3], &F::new(2));
    assert_eq!(v, F::new(7));
}

#[test]
fn pcs_evaluate() {
    // On the hypercube, the multilinear extension is the table.
    let table = [F::new(1), F::new(2), F::new(3), F::new(4)];
    let at = |b0: u64, b1: u64| {
        <Multilinear as PolynomialFamily<F>>::evaluate(&table, &[F::new(b0), F::new(b1)][..])
    };
    assert_eq!(at(1, 0), F::new(3));
}

#[test]
fn pcs_univariate() {
    assert_eq!(
        <Univariate as PolynomialFamily<F>>::evaluate(&[F::new(4)], &F::new(9)),
        F::new(4)
    );
}

#[test]
fn pcs_multilinear() {
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
    // A test setup: its secret is known.
    let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    // A constant polynomial commits to a multiple of the generator.
    let c = PolynomialCommitment::commit(&pc, &[Fr::from_u64(4)]);
    assert_eq!(
        c,
        PolynomialCommitment::commit(&pc, &[Fr::from_u64(4), Fr::ZERO])
    );
}

#[test]
fn pcs_multilinear_open() {
    // A test setup: its secret is known.
    let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    assert_eq!(
        pc.open(&[Fr::ONE, Fr::ONE], &Fr::from_u64(9)).0,
        Fr::from_u64(10)
    );
}

#[test]
fn pcs_multilinear_verify() {
    // A test setup: its secret is known.
    let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    let p = [Fr::from_u64(5)];
    let (v, proof) = pc.open(&p, &Fr::ZERO);
    assert!(pc.verify(&PolynomialCommitment::commit(&pc, &p), &Fr::ZERO, v, &proof));
    assert!(!pc.verify(
        &PolynomialCommitment::commit(&pc, &p),
        &Fr::ZERO,
        v + Fr::ONE,
        &proof
    ));
}
