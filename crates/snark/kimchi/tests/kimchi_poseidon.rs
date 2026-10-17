//! Kimchi's `kimchi/src/tests/poseidon.rs`, test by test, with the same
//! inputs and outputs: the permutation of `[1, 2, 3]` on Kimchi's rows (ours
//! derived from our Poseidon gadget), its witness and the proof's verdict;
//! and the circuit of one block against the circuit of two, each proof
//! verified by its own verifier index and not by the other's
//! (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, Wire},
    polynomials::poseidon::generate_witness,
    prover::{ProverIndex, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
    verifier::verify,
};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};
use gate_tests::{case, debug, prove_and_verify, rows, vesta_gates};

/// The rows of one permutation and its output row.
const ROWS_PER: usize = 12;

/// Kimchi's `build_poseidon_instance`: one permutation per input, each on
/// its own 12 rows, the first and last rows wired to themselves.
fn instance(inputs: &[[Fp; 3]]) -> (Vec<CircuitGate<Fp>>, [Vec<Fp>; COLUMNS]) {
    let poseidon = pallas_base();
    let mut gates = Vec::new();
    for i in 0..inputs.len() {
        let row = i * ROWS_PER;
        let ends = [Wire::for_row(row), Wire::for_row(row + ROWS_PER - 1)];
        let (rows, _) = CircuitGate::create_poseidon_gadget(row, ends, &poseidon);
        gates.extend(rows);
    }
    let mut witness: [Vec<Fp>; COLUMNS] =
        core::array::from_fn(|_| vec![Fp::ZERO; inputs.len() * ROWS_PER]);
    for (i, input) in inputs.iter().enumerate() {
        generate_witness(i * ROWS_PER, &poseidon, &mut witness, *input).unwrap();
    }
    (gates, witness)
}

#[test]
fn test_poseidon() {
    let case = case("test_poseidon");
    let input: [Fp; 3] = case.elements("input").try_into().unwrap();
    assert_eq!(input, [1, 2, 3].map(Fp::from_u64));
    let (gates, witness) = instance(&[input]);
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

/// The prover index of `gates` with no public input, on `srs`.
fn index(gates: Vec<CircuitGate<Fp>>, srs: Srs<Vesta>) -> ProverIndex<Vesta> {
    let cs = ConstraintSystem::create(gates).build().unwrap();
    ProverIndex::create(cs, srs)
}

#[test]
fn test_poseidon_in_circuit_extra_zero_block() {
    let case = case("test_poseidon_in_circuit_extra_zero_block");
    let one_two_three = [1, 2, 3].map(Fp::from_u64);
    let (gates1, witness1) = instance(&[one_two_three]);
    let (gates2, witness2) = instance(&[one_two_three, [Fp::ZERO; 3]]);
    assert_eq!(gates1.len(), case.number("rows1"));
    assert_eq!(gates2.len(), case.number("rows2"));
    gate_tests::assert_kimchi_rows(&gates2, &case.gates[0]);
    assert_eq!(rows(&witness2), case.witnesses[0]);
    // Kimchi's test indexes use one SRS (its precomputed test SRS) for both
    // circuits; one large enough for the larger domain plays that part.
    let srs = Srs::<Vesta>::create(32);
    let (index1, index2) = (
        index(gates1.clone(), srs.clone()),
        index(gates2.clone(), srs),
    );
    let kimchi = vesta_gates();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let sponges = (&base, &scalar);
    let mut rng = Arkworks(SplitMix64(75));
    let proof1 = prove(&index1, witness1, &[], &[], &kimchi, sponges, &mut rng).unwrap();
    let verify1 = verify(
        &index1.verifier_index,
        &index1.srs,
        &proof1,
        &[],
        &kimchi,
        sponges,
        &mut rng,
    );
    assert_eq!(
        debug(&verify1.map_err(|e| e.to_string())),
        case.verdict("verify1")
    );
    let proof2 = prove(
        &index2,
        witness2.clone(),
        &[],
        &[],
        &kimchi,
        sponges,
        &mut rng,
    )
    .unwrap();
    let verify2 = verify(
        &index2.verifier_index,
        &index2.srs,
        &proof2,
        &[],
        &kimchi,
        sponges,
        &mut rng,
    );
    assert_eq!(
        debug(&verify2.map_err(|e| e.to_string())),
        case.verdict("verify2")
    );
    let bad = verify(
        &index1.verifier_index,
        &index1.srs,
        &proof2,
        &[],
        &kimchi,
        sponges,
        &mut rng,
    );
    assert_eq!(debug(&bad.is_err()), case.verdict("bad_is_err"));
    // Not Kimchi's: with an SRS of each domain's size, Kimchi's verifier
    // indexes past the smaller SRS and panics on the same check; ours
    // refuses the proof.
    let small = index(gates1, Srs::<Vesta>::create(16));
    let refused = verify(
        &small.verifier_index,
        &small.srs,
        &proof2,
        &[],
        &kimchi,
        sponges,
        &mut rng,
    );
    assert!(refused.is_err());
}
