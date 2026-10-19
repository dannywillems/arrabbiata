//! Step over Wrap, end to end: a base-case step proof whose statement is
//! the digest of an application state, wrapped by a wrap proof over Pallas,
//! then a step proof over Vesta of the next state that verifies the wrap
//! proof; the step proof verifies, with its unfinalized wrap proof
//! finalized natively; a false state, message or accumulator is caught.

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    prover::{ProverIndex, ProverProof, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_pasta::{Fp, Fq, Vesta};
use arrabbiata_pickles::{
    step::{
        self, Error, MessagesForNextStepProof, Step, WrapEvaluations, WrapProof, index_elements,
        prove_step, verify_step,
    },
    wrap::{self, StepEvaluations, Wrap, step_gates},
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

/// The wrap statement's size for three rounds, and its private input's.
const WS: usize = 14 + 3;
const WW: usize = 2 * (15 + 1 + 7 + 6 + 2) + 2;

/// The wrap key's chunks that are not the identity, for the wrap circuit
/// these tests make.
const KEY: usize = 28;

/// The step statement's size and its private input's, for the wrap proof
/// these tests make: the key's points, then the wrap statement, the wrap
/// proof, the base step proof's evaluations and the state.
const S: usize = 19 + 13;
const W: usize = 2 * KEY + 17 + 2 * (15 + 1 + 7 + 26 + 2) + 4 + 1 + 88 + 1;

#[test]
fn a_step_proof_verifies_a_wrap_proof() {
    let base = base_index();
    let padding = vec![vec![Fq::ONE; 3], vec![Fq::from_u64(2); 3]];
    let wrap_shape = wrap::Shape {
        chunks: 1,
        quotient_chunks: 7,
        rounds: 3,
    };
    let fq_poseidon = vesta_base();
    let wrap_circuit = Wrap::<WS, WW>::new(
        &base.verifier_index,
        &base.srs,
        &fq_poseidon,
        wrap_shape,
        padding.clone(),
    )
    .unwrap();
    let wrap_index = wrap::index(&wrap_circuit).unwrap();

    // The base step proof: its statement the digest of the state 41 under
    // the wrap key.
    let state = Fp::from_u64(41);
    let genesis = MessagesForNextStepProof {
        index: index_elements(&wrap_index.verifier_index),
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
    let (wrap_statement, _, wrap_private, _) = wrap::statement(&base_step, &padding).unwrap();
    let mut rng = Arkworks(SplitMix64(11));
    let wrap_proof = wrap::prove_wrap(
        &wrap_circuit,
        &wrap_index,
        &wrap_statement,
        &wrap_private,
        &mut rng,
    )
    .unwrap();

    // The step proof of the next state, verifying the wrap proof.
    let verified = WrapProof {
        index: &wrap_index.verifier_index,
        srs: &wrap_index.srs,
        proof: &wrap_proof,
        statement: &wrap_statement,
    };
    let evaluations = StepEvaluations::of(&proof);
    let (stmt, messages, private, shape) = step::statement(&verified, &evaluations, state).unwrap();
    assert_eq!((shape.statement(), shape.witness()), (S, W));
    assert_eq!(messages.app_state, vec![Fp::from_u64(42)]);
    let fp_poseidon = pallas_base();
    let circuit = Step::<S, W>::new(
        &wrap_index.verifier_index,
        &wrap_index.srs,
        &base.verifier_index,
        &fp_poseidon,
        shape,
    )
    .unwrap();
    let step_index = step::index(&circuit).unwrap();
    let mut rng = Arkworks(SplitMix64(13));
    let step_proof = prove_step(&circuit, &step_index, &stmt, &private, &mut rng).unwrap();
    let wrap_evaluations = WrapEvaluations::of(&wrap_proof);
    let wrap_side = (
        &wrap_index.verifier_index,
        &wrap_index.srs,
        &wrap_evaluations,
    );
    let mut rng = Arkworks(SplitMix64(17));
    assert_eq!(
        verify_step(
            &step_index.verifier_index,
            &step_index.srs,
            &step_proof,
            &stmt,
            &messages,
            wrap_side,
            &mut rng,
        ),
        Ok(())
    );

    // From another state, the wrap statement's messages are not the
    // state's: no statement.
    assert_eq!(
        step::statement(&verified, &evaluations, Fp::from_u64(40)).err(),
        Some(Error::Messages)
    );

    // No witness of the step circuit for a false next digest, a false
    // passed digest, an unfinalized proof not to finalize, or a false
    // deferred value of the wrap proof.
    let mut false_messages = stmt.clone();
    false_messages.messages_for_next_step_proof =
        false_messages.messages_for_next_step_proof + Fp::ONE;
    let mut false_passed = stmt.clone();
    false_passed.messages_for_next_wrap_proof = false_passed.messages_for_next_wrap_proof + Fp::ONE;
    let mut not_finalized = stmt.clone();
    not_finalized.unfinalized.should_finalize = Fp::ZERO;
    let mut false_b = stmt.clone();
    false_b.unfinalized.b[0] = false_b.unfinalized.b[0] + Fp::ONE;
    for false_statement in [false_messages, false_passed, not_finalized, false_b] {
        assert!(prove_step(&circuit, &step_index, &false_statement, &private, &mut rng).is_err());
    }
    // Nor for a private wrap statement whose deferred value is not the base
    // step proof's, nor for another wrap key: the wrap proof's transcript
    // starts from the key's digest, and the messages hash the key.
    let mut false_private = private.clone();
    false_private[2 * KEY + 4] = false_private[2 * KEY + 4] + Fp::ONE;
    assert!(prove_step(&circuit, &step_index, &stmt, &false_private, &mut rng).is_err());
    let mut false_key = private.clone();
    false_key[0] = false_key[0] + Fp::ONE;
    assert!(prove_step(&circuit, &step_index, &stmt, &false_key, &mut rng).is_err());

    // Natively: false messages, a false accumulator under its own hash, and
    // a false deferred value of the unfinalized proof.
    let (wi, wsrs) = (&wrap_index.verifier_index, &wrap_index.srs);
    // Messages over another key, even hashed into the statement: the key is
    // a witness of the circuit, so the verifier pins it here.
    let mut other_key = messages.clone();
    other_key.index[0] = other_key.index[0] + Fp::ONE;
    let mut rekeyed = stmt.clone();
    rekeyed.messages_for_next_step_proof = other_key.digest();
    assert_eq!(
        step::finalize(wi, wsrs, &rekeyed, &other_key, &wrap_evaluations),
        Err(Error::Messages)
    );
    let mut other = messages.clone();
    other.app_state = vec![Fp::from_u64(43)];
    assert_eq!(
        step::finalize(wi, wsrs, &stmt, &other, &wrap_evaluations),
        Err(Error::Messages)
    );
    let [x, y] = messages.challenge_polynomial_commitments[0];
    let swapped = MessagesForNextStepProof {
        challenge_polynomial_commitments: vec![[y, x]],
        ..messages.clone()
    };
    let mut rehashed = stmt.clone();
    rehashed.messages_for_next_step_proof = swapped.digest();
    assert_eq!(
        step::finalize(wi, wsrs, &rehashed, &swapped, &wrap_evaluations),
        Err(Error::Accumulator)
    );
    let mut false_perm = stmt.clone();
    false_perm.unfinalized.perm[1] = Fp::ONE - false_perm.unfinalized.perm[1];
    assert_eq!(
        step::finalize(wi, wsrs, &false_perm, &messages, &wrap_evaluations),
        Err(Error::Deferred("perm"))
    );
}
