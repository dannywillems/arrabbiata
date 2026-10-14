//! PlonK over BN254 with KZG.

mod common;

use arrabbiata_bn254::{Fr, G1};
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_plonk::{PlonkError, kzg, prove, setup, verify};
use common::{arb_program, circuit};
use proptest::prelude::*;

const TAU: u64 = 0x5eed_5eed_5eed;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    /// Completeness on random circuits.
    #[test]
    fn completeness(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fr>(&program, false);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(TAU));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof).is_ok());
    }

    /// A false statement has no proof.
    #[test]
    fn false_statement(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fr>(&program, true);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(TAU));
        let (pk, _) = setup(&circuit, &pcs).unwrap();
        prop_assert_eq!(prove(&pk, &pcs, kzg::transcript(), &witness).unwrap_err(), PlonkError::Unsatisfied);
    }

    /// A changed evaluation fails its KZG opening.
    #[test]
    fn tampered_evaluation(program in arb_program(12), index in 0usize..15) {
        let (circuit, witness) = circuit::<Fr>(&program, false);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(TAU));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        let value = proof.tail.tail.tail.head.iter_mut().nth(index).unwrap();
        *value = *value + Fr::ONE;
        prop_assert_eq!(
            verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// A changed opening proof is rejected.
    #[test]
    fn tampered_opening(program in arb_program(12), index in 0usize..15) {
        let (circuit, witness) = circuit::<Fr>(&program, false);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(TAU));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        let openings = &mut proof.tail.tail.tail.tail.head.value;
        let opening = openings.iter_mut().nth(index).unwrap();
        *opening = *opening + Projective::<G1>::generator();
        prop_assert_eq!(
            verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// A proof is bound to its statement.
    #[test]
    fn wrong_public_input(program in arb_program(12)) {
        let (circuit, witness) = circuit::<Fr>(&program, false);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(TAU));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        let mut wrong = witness.public_inputs().to_vec();
        let last = wrong.len() - 1;
        wrong[last] = wrong[last] + Fr::ONE;
        prop_assert!(verify(&vk, &pcs, kzg::transcript(), &wrong, proof).is_err());
    }
}

/// Every size up to 64 rows.
#[test]
fn sizes() {
    for gates in [1u64, 3, 7, 15, 31, 60] {
        let program: Vec<common::Op> = (0..gates).map(|i| common::Op::AddConstant(0, i)).collect();
        let (circuit, witness) = circuit::<Fr>(&program, false);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(TAU));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        assert!(
            verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof).is_ok(),
            "{gates}"
        );
    }
}
