//! kimchi/src/tests/generic.rs
use ark_ff::Zero;
use kimchi::circuits::{
    polynomials::generic::testing::{create_circuit, fill_in_witness},
    wires::COLUMNS,
};
use mina_curves::pasta::Fp;

use crate::common::*;

pub fn run() {
    for (name, public) in [
        ("test_generic_gate", vec![]),
        ("test_generic_gate_pub", vec![Fp::from(3u8); 5]),
        ("test_generic_gate_pub_all_zeros", vec![Fp::from(0u8); 5]),
        ("test_generic_gate_pub_empty", vec![]),
    ] {
        test(name);
        input("public", &public);
        let gates = create_circuit(0, public.len());
        let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); gates.len()]);
        fill_in_witness(0, &mut w, &public);
        witness(&w);
        verdict("prove", &prove_and_verify(gates, w, public));
    }
}
