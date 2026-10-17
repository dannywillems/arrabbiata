//! Kimchi's `kimchi/src/tests/poseidon.rs`, test by test, with the same
//! inputs and outputs. Each circuit is one typed gadget, our Poseidon
//! permutations each on its own rows: the permutation of `[1, 2, 3]`, its
//! rows (the symbolic run), its witness (the numeric run), the proof's
//! verdict;
//! and the circuit of one block against the circuit of two, each proof
//! verified by its own verifier index and not by the other's
//! (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem},
    prover::{ProverIndex, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
    verifier::verify,
};
use arrabbiata_kimchi_gadgets::poseidon::{PoseidonPermutation, ROWS_PER_PERMUTATION};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, KimchiSponge, WIDTH, pallas_base, vesta_base};
use gate_tests::{case, debug, kimchi_rows, kimchi_witness, prove_and_verify, rows, vesta_gates};

/// One permutation of the input in columns 0-2 of its first row, its
/// output on the Zero row after its rounds.
#[derive(Clone, Copy, Debug)]
struct Permutation<'a, F> {
    poseidon: &'a KimchiPoseidon<F>,
}

impl<F: PrimeField> TypedGadget<F> for Permutation<'_, F> {
    type Input = Array<Scalar, WIDTH>;
    type Output = Array<Scalar, WIDTH>;
    const NAME: &'static str = "kimchi-poseidon-instance";
    const ROWS: usize = ROWS_PER_PERMUTATION + 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        input: Of<Array<Scalar, WIDTH>, E::Var>,
    ) -> Of<Array<Scalar, WIDTH>, E::Var> {
        let state = core::array::from_fn(|k| env.advice(Cell::new(k, 0), [&input[k]], |[v]| v));
        env.call(
            &PoseidonPermutation {
                poseidon: self.poseidon,
            },
            state,
        )
    }
}

/// Kimchi's `build_poseidon_instance`: `P` permutations, each on its own
/// rows. Input the `P` inputs.
#[derive(Clone, Copy, Debug)]
struct Instance<'a, F, const P: usize> {
    poseidon: &'a KimchiPoseidon<F>,
}

impl<F: PrimeField, const P: usize> TypedGadget<F> for Instance<'_, F, P> {
    type Input = Array<Array<Scalar, WIDTH>, P>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-poseidon-test";
    const ROWS: usize = P * (ROWS_PER_PERMUTATION + 1);

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        inputs: Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let permutation = Permutation {
            poseidon: self.poseidon,
        };
        for input in inputs {
            env.call(&permutation, input);
        }
    }
}

/// The rows and the witness of the permutations of `inputs`.
fn instance<const P: usize>(inputs: [[Fp; 3]; P]) -> (Vec<CircuitGate<Fp>>, [Vec<Fp>; COLUMNS]) {
    let poseidon = pallas_base();
    let gadget = Instance::<Fp, P> {
        poseidon: &poseidon,
    };
    let n = <Instance<Fp, P> as TypedGadget<Fp>>::ROWS;
    (kimchi_rows(&gadget, n), kimchi_witness(&gadget, &inputs, n))
}

#[test]
fn test_poseidon() {
    let case = case("test_poseidon");
    let input: [Fp; 3] = case.elements("input").try_into().unwrap();
    assert_eq!(input, [1, 2, 3].map(Fp::from_u64));
    let (gates, witness) = instance([input]);
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
    let (gates1, witness1) = instance([one_two_three]);
    let (gates2, witness2) = instance([one_two_three, [Fp::ZERO; 3]]);
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
