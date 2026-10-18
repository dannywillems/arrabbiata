//! Wrap, end to end: a base-case step proof over Vesta, wrapped by a Kimchi
//! proof over Pallas of the wrap circuit, verified by Kimchi on the wrap
//! statement then by the step proof's scalar half, messages and accumulator
//! on that statement; a false statement word, deferred value, message or
//! accumulator is caught.

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    prover::{ProverIndex, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_pasta::{Fp, Fq, Vesta};
use arrabbiata_pickles::{
    statement::Word,
    wrap::{
        Error, MessagesForNextWrapProof, Step, StepEvaluations, Wrap, bounds, finalize, index,
        prove_wrap, statement, step_gates, verify_wrap,
    },
};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// A step proof whose statement is one digest: row 0 binds w0 to the public
/// input, row 1 is w0 + w1 = w2.
fn step_proof() -> (
    ProverIndex<Vesta>,
    arrabbiata_kimchi::prover::ProverProof<Vesta>,
    [Fp; 1],
) {
    let gates: Vec<CircuitGate<Fp>> = [vec![Fp::ONE], vec![Fp::ONE, Fp::ONE, -Fp::ONE]]
        .into_iter()
        .enumerate()
        .map(|(row, coeffs)| CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs))
        .collect();
    let cs = ConstraintSystem::create(gates).public(1).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let public = [Fp::from_u64(5)];
    let columns: [Vec<Fp>; 15] = core::array::from_fn(|c| match c {
        0 => vec![public[0], Fp::from_u64(1)],
        1 => vec![Fp::ZERO, Fp::from_u64(2)],
        2 => vec![Fp::ZERO, Fp::from_u64(3)],
        _ => vec![Fp::ZERO; 2],
    });
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(7));
    let proof = prove(
        &index,
        columns,
        &[],
        &[],
        &step_gates(),
        (&base, &scalar),
        &mut rng,
    )
    .unwrap();
    (index, proof, public)
}

/// The padding the messages for the next wrap proof start with: two
/// vectors of three challenges.
fn padding() -> Vec<Vec<Fq>> {
    vec![
        (1..4).map(Fq::from_u64).collect(),
        (4..7).map(Fq::from_u64).collect(),
    ]
}

/// The statement's size for three rounds, and the private input's: 15 + 1
/// + 7 chunks, 3 rounds of two points, delta and sg, then z1 and z2.
const S: usize = 14 + 3;
const W: usize = 2 * (15 + 1 + 7 + 6 + 2) + 2;

#[test]
fn a_step_proof_is_wrapped_and_the_wrap_verifies() {
    let (step_index, step_proof, public) = step_proof();
    let step = Step {
        index: &step_index.verifier_index,
        srs: &step_index.srs,
        proof: &step_proof,
        public: &public,
    };
    let (stmt, messages, private, shape) = statement(&step, &padding()).unwrap();
    assert_eq!((shape.statement(), shape.witness()), (S, W));
    assert_eq!(
        Word::of(stmt.messages_for_next_step_proof),
        Word::of(public[0])
    );
    assert_eq!(stmt.messages_for_next_wrap_proof, messages.digest());
    // A base case on a domain of 2^3 rows: 4 * 3 + 0.
    assert_eq!(stmt.branch_data, Fq::from_u64(12));
    for (v, n) in stmt.to_vec().iter().zip(bounds(shape.rounds)) {
        assert!(Word::of(*v).below(n));
    }

    let poseidon = vesta_base();
    let wrap = Wrap::<S, W>::new(
        &step_index.verifier_index,
        &step_index.srs,
        &poseidon,
        shape,
        padding(),
    )
    .unwrap();
    let wrap_index = index(&wrap).unwrap();
    let mut rng = Arkworks(SplitMix64(11));
    let wrap_proof = prove_wrap(&wrap, &wrap_index, &stmt, &private, &mut rng).unwrap();
    let evaluations = StepEvaluations::of(&step_proof);
    let step_side = (&step_index.verifier_index, &step_index.srs, &evaluations);
    let mut rng = Arkworks(SplitMix64(13));
    assert_eq!(
        verify_wrap(
            &wrap_index.verifier_index,
            &wrap_index.srs,
            &wrap_proof,
            &stmt,
            &messages,
            step_side,
            &mut rng,
        ),
        Ok(())
    );

    // Another step statement: the wrap proof is not of that statement.
    let mut other = stmt.clone();
    other.messages_for_next_step_proof = Fq::from_u64(6);
    let refused = verify_wrap(
        &wrap_index.verifier_index,
        &wrap_index.srs,
        &wrap_proof,
        &other,
        &messages,
        step_side,
        &mut rng,
    );
    assert!(matches!(refused, Err(Error::Wrap(_))), "{refused:?}");

    // No witness of the wrap circuit, so no wrap proof, for a false b, a
    // digest that is not the messages', or another branch.
    let mut false_b = stmt.clone();
    false_b.b = false_b.b + Fq::ONE;
    let mut false_digest = stmt.clone();
    false_digest.messages_for_next_wrap_proof = stmt.messages_for_next_wrap_proof + Fq::ONE;
    let mut false_branch = stmt.clone();
    false_branch.branch_data = Fq::from_u64(14);
    for false_statement in [false_b, false_digest, false_branch] {
        assert!(prove_wrap(&wrap, &wrap_index, &false_statement, &private, &mut rng).is_err());
    }

    // The scalar half catches a false deferred value, the hash false
    // messages, and the accumulator check a false sg under its own hash.
    let mut false_perm = stmt.clone();
    false_perm.perm = false_perm.perm + Fq::ONE;
    let (vi, srs) = (&step_index.verifier_index, &step_index.srs);
    assert_eq!(
        finalize(vi, srs, &false_perm, &messages, &evaluations),
        Err(Error::Deferred("perm"))
    );
    let [x, y] = messages.challenge_polynomial_commitment;
    let swapped = MessagesForNextWrapProof {
        challenge_polynomial_commitment: [y, x],
        ..messages.clone()
    };
    assert_eq!(
        finalize(vi, srs, &stmt, &swapped, &evaluations),
        Err(Error::Messages)
    );
    let mut rehashed = stmt.clone();
    rehashed.messages_for_next_wrap_proof = swapped.digest();
    assert_eq!(
        finalize(vi, srs, &rehashed, &swapped, &evaluations),
        Err(Error::Accumulator)
    );
}

#[test]
fn a_step_statement_of_another_shape_is_not_wrapped() {
    // The base-case step statement is one digest below 2^254.
    let (step_index, step_proof, _) = step_proof();
    for public in [vec![], vec![Fp::ONE, Fp::ONE], vec![-Fp::ONE]] {
        let step = Step {
            index: &step_index.verifier_index,
            srs: &step_index.srs,
            proof: &step_proof,
            public: &public,
        };
        assert!(matches!(
            statement(&step, &padding()),
            Err(Error::Size | Error::Bound)
        ));
    }
}
