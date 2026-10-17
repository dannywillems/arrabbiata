//! Kimchi's `kimchi/src/tests/generic.rs`, test by test, with the same
//! inputs and outputs: Kimchi's circuit from our constructors, the witness
//! with the outputs our Generic half-gates compute, each proof's verdict
//! against Kimchi's run (`gate_tests/`). `test_generic_gate_kzg` (Kimchi's
//! `bn254` feature, a KZG proof) has no counterpart: our Kimchi proves with
//! the IPA only.

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::COLUMNS,
    polynomials::generic::testing::{create_circuit, fill_in_witness},
};
use arrabbiata_pasta::Fp;
use gate_tests::{case, debug, prove_and_verify, rows};

/// One of the four: the circuit with `public` public rows, its witness,
/// TestFramework's verdict.
fn generic(name: &str, public: Vec<Fp>) {
    let case = case(name);
    assert_eq!(public, case.elements::<Fp>("public"));
    let gates = create_circuit(0, public.len());
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; gates.len()]);
    fill_in_witness(0, &mut witness, &public);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, public);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_generic_gate() {
    generic("test_generic_gate", vec![]);
}

#[test]
fn test_generic_gate_pub() {
    generic("test_generic_gate_pub", vec![Fp::from_u64(3); 5]);
}

#[test]
fn test_generic_gate_pub_all_zeros() {
    generic("test_generic_gate_pub_all_zeros", vec![Fp::ZERO; 5]);
}

#[test]
fn test_generic_gate_pub_empty() {
    generic("test_generic_gate_pub_empty", vec![]);
}
