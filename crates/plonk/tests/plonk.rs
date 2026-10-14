//! End-to-end property tests of vanilla PlonK on random circuits.

use arrabbiata_commit::{RevealCommitment, Univariate};
use arrabbiata_field::{Field, Goldilocks as F};
use arrabbiata_frontend::{Builder, Wire, build};
use arrabbiata_plonk::{PlonkError, prove, setup, verify};
use arrabbiata_transcript::toy_sponge;
use proptest::prelude::*;

/// One step of a random straight-line program over the wires built so far.
#[derive(Clone, Debug)]
enum Op {
    Witness(u64),
    Constant(u64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    AddConstant(usize, u64),
    Scale(usize, u64),
}

fn arb_program() -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        any::<u64>().prop_map(Op::Witness),
        any::<u64>().prop_map(Op::Constant),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Add(i, j)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Sub(i, j)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Mul(i, j)),
        (any::<usize>(), any::<u64>()).prop_map(|(i, c)| Op::AddConstant(i, c)),
        (any::<usize>(), any::<u64>()).prop_map(|(i, c)| Op::Scale(i, c)),
    ];
    prop::collection::vec(op, 1..20)
}

/// Run the program and expose its last wire as a public output. When
/// `lie` is set, the public output claims a different value.
fn run<'id>(b: &mut Builder<'id, F>, program: &[Op], lie: bool) {
    let mut wires: Vec<Wire<'id, F>> = vec![b.witness(F::new(1))];
    for op in program {
        let pick = |i: usize| wires[i % wires.len()];
        let w = match op {
            Op::Witness(v) => b.witness(F::new(*v)),
            Op::Constant(c) => b.constant(F::new(*c)),
            Op::Add(i, j) => b.add(&pick(*i), &pick(*j)),
            Op::Sub(i, j) => b.sub(&pick(*i), &pick(*j)),
            Op::Mul(i, j) => b.mul(&pick(*i), &pick(*j)),
            Op::AddConstant(i, c) => b.add_constant(&pick(*i), F::new(*c)),
            Op::Scale(i, c) => b.scale(&pick(*i), F::new(*c)),
        };
        wires.push(w);
    }
    let last = *wires.last().unwrap();
    let claimed = if lie {
        last.value() + F::ONE
    } else {
        last.value()
    };
    let output = b.public_input(claimed);
    b.assert_equal(&last, &output);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Completeness: an honest proof of a satisfied circuit verifies.
    #[test]
    fn completeness(program in arb_program(), seed in any::<u64>()) {
        let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, &program, false));
        prop_assert!(circuit.is_satisfied(&witness));
        let pcs = RevealCommitment::<Univariate>::new();
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, toy_sponge(seed), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, toy_sponge(seed), witness.public_inputs(), proof).is_ok());
    }

    /// A false statement has no quotient: the honest prover fails.
    #[test]
    fn false_statement_has_no_proof(program in arb_program()) {
        let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, &program, true));
        prop_assert!(!circuit.is_satisfied(&witness));
        let pcs = RevealCommitment::<Univariate>::new();
        let (pk, _) = setup(&circuit, &pcs).unwrap();
        prop_assert_eq!(prove(&pk, &pcs, toy_sponge(0), &witness).unwrap_err(), PlonkError::Unsatisfied);
    }

    /// Soundness against a lying prover: changing any claimed evaluation
    /// makes verification fail, through its opening.
    #[test]
    fn tampered_evaluation_is_rejected(program in arb_program(), index in 0usize..13, delta in 1u64..1000) {
        let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, &program, false));
        let pcs = RevealCommitment::<Univariate>::new();
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, toy_sponge(0), &witness).unwrap();
        // The proof is the HList [wires, z, t, evaluations, openings].
        let evaluations = &mut proof.tail.tail.tail.head;
        evaluations.values[index] = evaluations.values[index] + F::new(delta);
        prop_assert!(verify(&vk, &pcs, toy_sponge(0), witness.public_inputs(), proof).is_err());
    }

    /// A proof is bound to its statement.
    #[test]
    fn wrong_public_input_is_rejected(program in arb_program(), delta in 1u64..1000) {
        let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, &program, false));
        let pcs = RevealCommitment::<Univariate>::new();
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, toy_sponge(0), &witness).unwrap();
        let wrong = vec![witness.public_inputs()[0] + F::new(delta)];
        prop_assert!(verify(&vk, &pcs, toy_sponge(0), &wrong, proof).is_err());
    }

    /// A proof is bound to its transcript: a verifier on a different sponge
    /// derives a different zeta, at which the claimed evaluations no longer
    /// match their openings (the first check to fail).
    #[test]
    fn transcript_mismatch_is_rejected(program in arb_program()) {
        let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, &program, false));
        let pcs = RevealCommitment::<Univariate>::new();
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, toy_sponge(0), &witness).unwrap();
        prop_assert!(matches!(
            verify(&vk, &pcs, toy_sponge(1), witness.public_inputs(), proof),
            Err(PlonkError::Opening(_) | PlonkError::IdentityFailed)
        ));
    }
}
