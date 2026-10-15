//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_commit::PolynomialCommitment;
use arrabbiata_field::Field;
use arrabbiata_ipa::{Ipa, IpaProof};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};

#[test]
fn crate_overview() {
    // 8 generators on Pallas from Poseidon over Fp; challenges from Poseidon
    // over Fq, the field the polynomials live in.
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let p = [Fq::from_u64(1), Fq::from_u64(2), Fq::from_u64(3)]; // 1 + 2X + 3X^2
    let c = ipa.commit(&p);
    let (v, proof) = ipa.open(&p, &Fq::from_u64(2));
    assert_eq!(v, Fq::from_u64(17));
    assert!(ipa.verify(&c, &Fq::from_u64(2), v, &proof));
    assert!(!ipa.verify(&c, &Fq::from_u64(2), v + Fq::ONE, &proof));
}

#[test]
fn ipa() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    assert_eq!(ipa.size(), 4);
}

#[test]
fn ipa_proof() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let (_, proof) = ipa.open(&[Fq::ONE], &Fq::ZERO);
    assert_eq!(proof.rounds().len(), 3);
}

#[test]
fn ipa_proof_rounds() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    assert_eq!(ipa.open(&[Fq::ONE], &Fq::ONE).1.rounds().len(), 1);
}

#[test]
fn ipa_proof_from_parts() {
    let proof = IpaProof::<Pallas>::from_parts(vec![], Fq::ONE);
    assert!(proof.rounds().is_empty());
}

#[test]
fn ipa_proof_scalar() {
    assert_eq!(
        IpaProof::<Pallas>::from_parts(vec![], Fq::ONE).scalar(),
        Fq::ONE
    );
}

#[test]
fn ipa_setup() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(7), 16, vesta_sponge(1));
    assert_eq!(ipa.size(), 16);
}

#[test]
fn ipa_size() {
    assert_eq!(
        Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1)).size(),
        2
    );
}
