//! Wrap of a step proof with one predecessor, end to end: a base step
//! proof, its wrap proof, a step proof verifying that wrap proof, and the
//! wrap proof of that step proof, which finalizes in its circuit the wrap
//! proof the step proof left unfinalized. It verifies; a false deferred
//! value, evaluation, predecessor or key is caught.

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    prover::{ProverIndex, ProverProof, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_pasta::{Fp, Fq, Vesta};
use arrabbiata_pickles::{
    step::{self, MessagesForNextStepProof, Step, WrapEvaluations, WrapProof, index_elements},
    wrap::{self, StepEvaluations, Wrap, step_gates},
    wrap_one::{self, Error, Predecessor, StepProof, WrapOne},
};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// The base step index: row 0 binds w0 to the public input, row 1 is
/// w0 + w1 = w2.
fn base_index() -> ProverIndex<Vesta> {
    let gates: Vec<CircuitGate<Fp>> = [vec![Fp::ONE], vec![Fp::ONE, Fp::ONE, -Fp::ONE]]
        .into_iter()
        .enumerate()
        .map(|(row, coeffs)| CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs))
        .collect();
    let cs = ConstraintSystem::create(gates).public(1).build().unwrap();
    ProverIndex::create(cs, Srs::<Vesta>::create(8))
}

/// A base step proof of the statement `digest`.
fn base_proof(index: &ProverIndex<Vesta>, digest: Fp) -> ProverProof<Vesta> {
    let columns: [Vec<Fp>; 15] = core::array::from_fn(|c| match c {
        0 => vec![digest, digest],
        1 => vec![Fp::ZERO, Fp::ONE],
        2 => vec![Fp::ZERO, digest + Fp::ONE],
        _ => vec![Fp::ZERO; 2],
    });
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(7));
    prove(
        index,
        columns,
        &[],
        &[],
        &step_gates(),
        (&base, &scalar),
        &mut rng,
    )
    .unwrap()
}

/// The sizes of the circuits below: the first wrap circuit, the step
/// circuit (as tests/step.rs), and the wrap circuit of the step proof, for
/// a step proof of `STEP_ROUNDS` rounds.
const WS: usize = 14 + 3;
const WW: usize = 2 * (15 + 1 + 7 + 6 + 2) + 2;
const SS: usize = 19 + 13;
const SW: usize = 2 * 28 + 17 + 2 * (15 + 1 + 7 + 26 + 2) + 4 + 1 + 88 + 1;
const STEP_ROUNDS: usize = 14;
const S: usize = 14 + STEP_ROUNDS;
const W: usize = 2 * (15 + 1 + 7 + 2 * STEP_ROUNDS + 2) + 2 + (17 + 13) + 1 + 88 + 2;

#[test]
fn a_step_proof_with_a_predecessor_is_wrapped_and_finalized() {
    // The base step proof, wrapped.
    let base = base_index();
    let padding = vec![vec![Fq::ONE; 3], vec![Fq::from_u64(2); 3]];
    let first_shape = wrap::Shape {
        chunks: 1,
        quotient_chunks: 7,
        rounds: 3,
    };
    let fq_poseidon = vesta_base();
    let first = Wrap::<WS, WW>::new(
        &base.verifier_index,
        &base.srs,
        &fq_poseidon,
        first_shape,
        padding.clone(),
    )
    .unwrap();
    let first_index = wrap::index(&first).unwrap();
    let state = Fp::from_u64(41);
    let genesis = MessagesForNextStepProof {
        index: index_elements(&first_index.verifier_index),
        app_state: vec![state],
        challenge_polynomial_commitments: vec![],
        old_bulletproof_challenges: vec![],
    };
    let public = [genesis.digest()];
    let proof = base_proof(&base, public[0]);
    let base_step = wrap::Step {
        index: &base.verifier_index,
        srs: &base.srs,
        proof: &proof,
        public: &public,
    };
    let (first_statement, first_messages, first_private, _) =
        wrap::statement(&base_step, &padding).unwrap();
    let mut rng = Arkworks(SplitMix64(11));
    let first_proof = wrap::prove_wrap(
        &first,
        &first_index,
        &first_statement,
        &first_private,
        &mut rng,
    )
    .unwrap();

    // The step proof of the next state, verifying it.
    let verified = WrapProof {
        index: &first_index.verifier_index,
        srs: &first_index.srs,
        proof: &first_proof,
        statement: &first_statement,
    };
    let base_evaluations = StepEvaluations::of(&proof);
    let (step_statement, step_messages, step_private, step_shape) =
        step::statement(&verified, &base_evaluations, state).unwrap();
    let fp_poseidon = pallas_base();
    let step_circuit = Step::<SS, SW>::new(
        &first_index.verifier_index,
        &first_index.srs,
        &base.verifier_index,
        &fp_poseidon,
        step_shape,
    )
    .unwrap();
    let step_index = step::index(&step_circuit).unwrap();
    let mut rng = Arkworks(SplitMix64(13));
    let step_proof = step::prove_step(
        &step_circuit,
        &step_index,
        &step_statement,
        &step_private,
        &mut rng,
    )
    .unwrap();

    // The wrap proof of the step proof, finalizing the first wrap proof.
    let step = StepProof {
        index: &step_index.verifier_index,
        srs: &step_index.srs,
        proof: &step_proof,
        statement: &step_statement,
    };
    let first_evaluations = WrapEvaluations::of(&first_proof);
    let predecessor = Predecessor {
        index: &first_index.verifier_index,
        evaluations: &first_evaluations,
        messages: &first_messages,
    };
    let new_padding = vec![Fq::from_u64(7); 3];
    let (stmt, messages, private, shape) =
        wrap_one::statement(&step, &predecessor, &new_padding).unwrap();
    assert_eq!((shape.statement(), shape.witness()), (S, W));
    assert_eq!(messages.old_bulletproof_challenges.len(), 2);
    let circuit = WrapOne::<S, W>::new(
        &step_index.verifier_index,
        &step_index.srs,
        &first_index.verifier_index,
        &fq_poseidon,
        shape,
        new_padding.clone(),
        padding.clone(),
    )
    .unwrap();
    let index = wrap_one::index(&circuit).unwrap();
    let mut rng = Arkworks(SplitMix64(17));
    let wrap_proof = wrap_one::prove_wrap(&circuit, &index, &stmt, &private, &mut rng).unwrap();
    let step_evaluations = StepEvaluations::of(&step_proof);
    let step_side = (
        &step_index.verifier_index,
        &step_index.srs,
        &step_evaluations,
    );
    let key = (&first_index.verifier_index, &first_index.srs);
    let mut rng = Arkworks(SplitMix64(19));
    assert_eq!(
        wrap_one::verify(
            &index.verifier_index,
            &index.srs,
            &wrap_proof,
            &stmt,
            &messages,
            step_side,
            &step_messages,
            key,
            &mut rng,
        ),
        Ok(())
    );

    // Natively: an unfinalized proof that does not finalize, and another
    // predecessor, are refused before proving.
    let mut false_perm = step_statement.clone();
    false_perm.unfinalized.perm[1] = Fp::ONE - false_perm.unfinalized.perm[1];
    let false_step = StepProof {
        statement: &false_perm,
        ..step
    };
    assert_eq!(
        wrap_one::statement(&false_step, &predecessor, &new_padding).err(),
        Some(Error::Deferred("perm"))
    );
    let mut other_messages = first_messages.clone();
    other_messages.challenge_polynomial_commitment.swap(0, 1);
    let other = Predecessor {
        messages: &other_messages,
        ..predecessor
    };
    assert_eq!(
        wrap_one::statement(&step, &other, &new_padding).err(),
        Some(Error::Messages)
    );

    // In the circuit: no witness for a false evaluation of the first wrap
    // proof (its scalar half no longer gives the deferred values), a false
    // predecessor's accumulator (the step statement's digest no longer
    // matches the step proof), or a false new digest.
    let step_part = shape.step.witness();
    let mut false_evaluation = private.clone();
    false_evaluation[step_part + 17 + 13 + 1] = false_evaluation[step_part + 17 + 13 + 1] + Fq::ONE;
    let mut false_previous = private.clone();
    let last = false_previous.len() - 1;
    false_previous[last] = false_previous[last] + Fq::ONE;
    for false_private in [false_evaluation, false_previous] {
        assert!(wrap_one::prove_wrap(&circuit, &index, &stmt, &false_private, &mut rng).is_err());
    }
    let mut false_digest = stmt.clone();
    false_digest.messages_for_next_wrap_proof = stmt.messages_for_next_wrap_proof + Fq::ONE;
    assert!(wrap_one::prove_wrap(&circuit, &index, &false_digest, &private, &mut rng).is_err());

    // Verifying against another key for the first wrap proof fails: the
    // step messages hash the key the step circuit was given.
    let mut rng = Arkworks(SplitMix64(23));
    let wrong_key = (&index.verifier_index, &index.srs);
    assert_eq!(
        wrap_one::verify(
            &index.verifier_index,
            &index.srs,
            &wrap_proof,
            &stmt,
            &messages,
            step_side,
            &step_messages,
            wrong_key,
            &mut rng,
        ),
        Err(Error::Accumulator)
    );
}
