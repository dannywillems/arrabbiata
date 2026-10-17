//! kimchi/src/tests/poseidon.rs
use ark_ff::Zero;
use groupmap::GroupMap;
use kimchi::{
    circuits::{
        gate::CircuitGate,
        polynomials::{self, poseidon::ROUNDS_PER_ROW},
        wires::{Wire, COLUMNS},
    },
    curve::KimchiCurve,
    proof::ProverProof,
    prover_index::ProverIndex,
    verifier::verify,
};
use mina_curves::pasta::{Fp, Vesta, VestaParameters};
use mina_poseidon::{
    constants::{PlonkSpongeConstantsKimchi, SpongeConstants},
    pasta::FULL_ROUNDS,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use poly_commitment::{
    commitment::CommitmentCurve,
    ipa::{OpeningProof, SRS},
};

use crate::common::*;

type SpongeParams = PlonkSpongeConstantsKimchi;
type BaseSponge = DefaultFqSponge<VestaParameters, SpongeParams, FULL_ROUNDS>;
type ScalarSponge = DefaultFrSponge<Fp, SpongeParams, FULL_ROUNDS>;
const POS_ROWS_PER_HASH: usize = SpongeParams::PERM_ROUNDS_FULL / ROUNDS_PER_ROW;

/// Kimchi's `build_poseidon_instance`.
fn instance(inputs: Vec<[Fp; 3]>) -> (Vec<CircuitGate<Fp>>, [Vec<Fp>; COLUMNS]) {
    let rounds = Vesta::sponge_params().round_constants.clone();
    let rows_per = POS_ROWS_PER_HASH + 1;
    let mut gates = Vec::with_capacity(inputs.len() * rows_per);
    let mut abs_row = 0;
    for _ in &inputs {
        let first_wire = Wire::for_row(abs_row);
        let last_wire = Wire::for_row(abs_row + POS_ROWS_PER_HASH);
        let (poseidon, _) = CircuitGate::<Fp>::create_poseidon_gadget(abs_row, [first_wire, last_wire], &rounds);
        gates.extend(poseidon);
        abs_row += rows_per;
    }
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); inputs.len() * rows_per]);
    for (i, input) in inputs.into_iter().enumerate() {
        polynomials::poseidon::generate_witness(i * rows_per, Vesta::sponge_params(), &mut witness, input);
    }
    (gates, witness)
}

/// Kimchi's `new_index_for_test` with no public input: the precomputed test
/// SRS (2^16 points) for every circuit, which is what makes the cross
/// verification below an error rather than a panic (with an SRS of the
/// domain's size, Kimchi's verifier indexes past it).
fn index(gates: Vec<CircuitGate<Fp>>) -> ProverIndex<FULL_ROUNDS, Vesta, SRS<Vesta>> {
    kimchi::prover_index::testing::new_index_for_test::<FULL_ROUNDS, Vesta>(gates, 0)
}

pub fn run() {
    {
        test("test_poseidon");
        let input = [Fp::from(1u32), Fp::from(2u32), Fp::from(3u32)];
        input_line(&input);
        let (gates, w) = instance(vec![input]);
        witness(&w);
        verdict("prove", &prove_and_verify(gates, w, vec![]));
    }
    {
        test("test_poseidon_in_circuit_extra_zero_block");
        let one_two_three = [Fp::from(1u32), Fp::from(2u32), Fp::from(3u32)];
        let (gates1, witness1) = instance(vec![one_two_three]);
        let (gates2, witness2) = instance(vec![one_two_three, [Fp::zero(); 3]]);
        number("rows1", gates1.len());
        number("rows2", gates2.len());
        gates_line(&gates2);
        witness(&witness2);
        let (index1, index2) = (index(gates1), index(gates2));
        let group_map = <Vesta as CommitmentCurve>::Map::setup();
        let mut rng = rng(75);
        type Proof = ProverProof<Vesta, OpeningProof<Vesta, FULL_ROUNDS>, FULL_ROUNDS>;
        let proof1 = Proof::create::<BaseSponge, ScalarSponge, _>(&group_map, witness1, &[], &index1, &mut rng).unwrap();
        let vi1 = index1.verifier_index();
        let ok1 = verify::<FULL_ROUNDS, Vesta, BaseSponge, ScalarSponge, OpeningProof<Vesta, FULL_ROUNDS>>(&group_map, &vi1, &proof1, &[]);
        verdict("verify1", &ok1.map_err(|e| e.to_string()));
        let proof2 = Proof::create::<BaseSponge, ScalarSponge, _>(&group_map, witness2, &[], &index2, &mut rng).unwrap();
        let ok2 = verify::<FULL_ROUNDS, Vesta, BaseSponge, ScalarSponge, OpeningProof<Vesta, FULL_ROUNDS>>(&group_map, &index2.verifier_index(), &proof2, &[]);
        verdict("verify2", &ok2.map_err(|e| e.to_string()));
        let bad = verify::<FULL_ROUNDS, Vesta, BaseSponge, ScalarSponge, OpeningProof<Vesta, FULL_ROUNDS>>(&group_map, &vi1, &proof2, &[]);
        verdict("bad_is_err", &bad.is_err());
    }
}

/// The input of the permutation.
fn input_line(x: &[Fp; 3]) {
    input("input", x);
}
