//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::Field;
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonk::{ipa, ipa::PallasIpa};
use arrabbiata_plonk_decider::{Decider, Error, Shared, relaxation};
use arrabbiata_plonkish::{
    Builder, RelaxedWitness,
    vanilla::{Arithmetic, Vanilla},
};
use arrabbiata_plonkish_folding::Params;
use arrabbiata_poseidon::vesta_sponge;

#[test]
fn shared() {
    fn shared<PC: Shared<F>>() {}
    shared::<PallasIpa>();
}

#[test]
fn error() {
    assert_eq!(
        Error::Commitment(0).to_string(),
        "the proof's commitment 0 is not the accumulator's"
    );
}

#[test]
fn relaxation_works() {
    // a b - c = 1 on one row: unsatisfied, relaxed with E = 1 at u = 1.
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let one = F::ONE;
    b.row::<Arithmetic, _>(
        [F::from_u64(2), F::from_u64(3), F::from_u64(5)],
        [F::ZERO, F::ZERO, -one, one, F::ZERO],
    );
    let (circuit, w) = b.build().unwrap();
    let g = relaxation::<F, _, 3, 5, 1, 11, 4>(&circuit, 2).unwrap();
    let mut columns = w.columns().to_vec();
    columns.push(vec![one, F::ZERO]);
    // Public inputs: none of the circuit's, then u = 1 and c_0 = 1.
    assert!(g.is_satisfied(&columns, &[one, one]));
    // The relaxation of the circuit agrees.
    let r = RelaxedWitness::new(w.columns().clone(), vec![one, F::ZERO]);
    assert!(circuit.is_relaxed_satisfied(&r, &[], one, &[one], 2));
}

#[test]
fn error_new() {
    // Sponge domain tags: each transcript gets its own, and prover and
    // verifier start from the same one.
    let digest_domain = 2;
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let pcs = ipa::pallas(&circuit);
    let params =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(digest_domain))
            .unwrap();
    // q_M c a b has degree 4: 3 is too low.
    assert_eq!(
        Decider::<_, _, _, 3, 5, 1, 3, 2, 11, 3>::new(&params).err(),
        Some(Error::Size)
    );
    assert!(Decider::<_, _, _, 3, 5, 1, 3, 2, 11, 4>::new(&params).is_ok());
}

#[test]
fn error_verifier_key() {
    // Sponge domain tags: each transcript gets its own, and prover and
    // verifier start from the same one.
    let digest_domain = 2;
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let pcs = ipa::pallas(&circuit);
    let params =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(digest_domain))
            .unwrap();
    let decider = Decider::<_, _, _, 3, 5, 1, 3, 2, 11, 4>::new(&params).unwrap();
    // No public input of the circuit, then u and one coefficient.
    assert_eq!(decider.verifier_key().value.public_inputs(), 2);
}

#[test]
fn error_circuit() {
    // Sponge domain tags: each transcript gets its own, and prover and
    // verifier start from the same one.
    let digest_domain = 2;
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let pcs = ipa::pallas(&circuit);
    let params =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(digest_domain))
            .unwrap();
    let decider = Decider::<_, _, _, 3, 5, 1, 3, 2, 11, 4>::new(&params).unwrap();
    // The witness columns and the error column; one gate per gadget, and idle.
    assert_eq!(decider.circuit().witness_columns(), 4);
    assert_eq!(decider.circuit().gates().len(), 2);
}
