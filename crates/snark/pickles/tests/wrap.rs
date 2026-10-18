//! Wrap, end to end: a step proof over Vesta, wrapped by a Kimchi proof over
//! Pallas of the wrap circuit, verified by Kimchi on the wrap statement then
//! by the step proof's scalar half and accumulator on that statement; a
//! false public input, deferred value or accumulator is caught.

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    prover::{ProverIndex, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_pasta::{Fp, Fq, Vesta};
use arrabbiata_pickles::wrap::{
    Error, Step, StepEvaluations, Wrap, finalize, index, prove_wrap, shifted, statement,
    step_gates, unshifted, verify_wrap,
};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// A step proof: row 0 binds w0 to the public input, row 1 is
/// w0 + w1 = w2.
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

/// The statement's size for one public input and three rounds, and the
/// private input's: 15 + 1 + 7 chunks and 3 rounds of two points, delta.
const S: usize = 1 + 13 + 3;
const W: usize = 2 * (15 + 1 + 7 + 6 + 1) + 2;

#[test]
fn a_step_proof_is_wrapped_and_the_wrap_verifies() {
    let (step_index, step_proof, public) = step_proof();
    let step = Step {
        index: &step_index.verifier_index,
        srs: &step_index.srs,
        proof: &step_proof,
        public: &public,
    };
    let (stmt, private, shape) = statement(&step).unwrap();
    assert_eq!((shape.statement(), shape.witness()), (S, W));
    assert_eq!(unshifted(stmt.step_public[0]), public[0]);

    let poseidon = vesta_base();
    let wrap = Wrap::<S, W>::new(
        &step_index.verifier_index,
        &step_index.srs,
        &poseidon,
        shape,
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
            step_side,
            &mut rng,
        ),
        Ok(())
    );

    // Another public input: the wrap proof is not of that statement.
    let mut other = stmt.clone();
    other.step_public[0] = shifted(Fp::from_u64(6));
    let refused = verify_wrap(
        &wrap_index.verifier_index,
        &wrap_index.srs,
        &wrap_proof,
        &other,
        step_side,
        &mut rng,
    );
    assert!(matches!(refused, Err(Error::Wrap(_))), "{refused:?}");

    // A false b: no witness of the wrap circuit, so no wrap proof.
    let mut false_b = stmt.clone();
    false_b.b = false_b.b + Fq::ONE;
    assert!(prove_wrap(&wrap, &wrap_index, &false_b, &private, &mut rng).is_err());

    // The scalar half catches a false deferred value, and the accumulator
    // check a false sg, on the statement alone.
    let mut false_perm = stmt.clone();
    false_perm.perm = false_perm.perm + Fq::ONE;
    assert_eq!(
        finalize(
            &step_index.verifier_index,
            &step_index.srs,
            &false_perm,
            &evaluations
        ),
        Err(Error::Deferred("perm"))
    );
    let mut false_sg = stmt.clone();
    false_sg.sg = [stmt.sg[1], stmt.sg[0]];
    assert_eq!(
        finalize(
            &step_index.verifier_index,
            &step_index.srs,
            &false_sg,
            &evaluations
        ),
        Err(Error::Accumulator)
    );
}
