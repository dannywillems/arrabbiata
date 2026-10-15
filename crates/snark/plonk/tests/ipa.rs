//! PlonK with IPA, on Pallas (polynomials over Fq) and on BN254 G1.

mod common;

use arrabbiata_bn254::Fr;
use arrabbiata_field::Field;
use arrabbiata_ipa::IpaProof;
use arrabbiata_pasta::Fq;
use arrabbiata_plonk::{PlonkError, ipa, prove, setup, verify};
use common::{arb_program, circuit};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    /// Completeness on random circuits, on Pallas.
    #[test]
    fn completeness_pallas(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fq>(&program, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof).is_ok());
    }

    /// Completeness on random circuits, on BN254 G1.
    #[test]
    fn completeness_bn254(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fr>(&program, false);
        let pcs = ipa::bn254(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::bn254_transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, ipa::bn254_transcript(), witness.public_inputs(), proof).is_ok());
    }

    /// A false statement has no proof.
    #[test]
    fn false_statement(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fq>(&program, true);
        let pcs = ipa::pallas(&circuit);
        let (pk, _) = setup(&circuit, &pcs).unwrap();
        prop_assert_eq!(prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap_err(), PlonkError::Unsatisfied);
    }

    /// A changed evaluation fails its IPA opening.
    #[test]
    fn tampered_evaluation(program in arb_program(12), index in 0usize..15) {
        let (circuit, witness) = circuit::<Fq>(&program, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        let value = proof.tail.tail.tail.head.iter_mut().nth(index).unwrap();
        *value = *value + Fq::ONE;
        prop_assert_eq!(
            verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// A changed opening proof (its final scalar) is rejected.
    #[test]
    fn tampered_opening(program in arb_program(12), index in 0usize..15) {
        let (circuit, witness) = circuit::<Fq>(&program, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        let openings = &mut proof.tail.tail.tail.tail.head.value;
        let opening = openings.iter_mut().nth(index).unwrap();
        let original = opening.clone();
        *opening = IpaProof::from_parts(original.rounds().to_vec(), original.scalar() + Fq::ONE, original.sg());
        prop_assert_eq!(
            verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// A proof is bound to its statement.
    #[test]
    fn wrong_public_input(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fq>(&program, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        let mut wrong = witness.public_inputs().to_vec();
        let last = wrong.len() - 1;
        wrong[last] = wrong[last] + Fq::ONE;
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &wrong, proof).is_err());
    }
}

/// Every size up to 64 rows, on Pallas.
#[test]
fn sizes() {
    for gates in [1u64, 3, 7, 15, 31, 60] {
        let program: Vec<common::Op> = (0..gates).map(|i| common::Op::AddConstant(0, i)).collect();
        let (circuit, witness) = circuit::<Fq>(&program, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        assert!(
            verify(
                &vk,
                &pcs,
                ipa::pallas_transcript(),
                witness.public_inputs(),
                proof
            )
            .is_ok(),
            "{gates}"
        );
    }
}
