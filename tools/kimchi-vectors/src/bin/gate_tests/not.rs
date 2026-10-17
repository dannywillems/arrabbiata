//! kimchi/src/tests/not.rs
use core::cmp::max;

use ark_ff::{Field, One, PrimeField, Zero};
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        gate::CircuitGate,
        polynomials::{generic::GenericGateSpec, not},
        wires::{Wire, COLUMNS},
    },
    curve::KimchiCurve,
};
use mina_curves::pasta::{Fp, Pallas, Vesta};
use o1_utils::{BigUintHelpers, FieldHelpers, RandomField};

use crate::common::*;

const R: usize = 55;

fn witness_unchecked<F: PrimeField>(inputs: &[F], bits: usize) -> [Vec<F>; COLUMNS] {
    let mut w: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::zero(); 1]);
    w[0][0] = F::from(2u8).pow([bits as u64]) - F::one();
    not::extend_not_witness_unchecked_length(&mut w, inputs, bits).unwrap();
    w
}

fn witness_checked<F: PrimeField>(input: F, bits: Option<usize>) -> [Vec<F>; COLUMNS] {
    let mut w: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::zero(); 1]);
    let real = max(input.to_biguint().bitlen(), bits.unwrap_or(0));
    w[0][0] = F::from(2u8).pow([real as u64]) - F::one();
    not::extend_not_witness_checked_length(&mut w, input, bits);
    w
}

fn cs_xor<F: PrimeField>(bits: usize) -> ConstraintSystem<F> {
    let mut gates = vec![CircuitGate::<F>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
    CircuitGate::<F>::extend_not_gadget_checked_length(&mut gates, 0, bits);
    ConstraintSystem::create(gates).public(1).build().unwrap()
}

fn cs_gnrc<F: PrimeField>(n: usize) -> ConstraintSystem<F> {
    let mut gates = vec![CircuitGate::<F>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
    CircuitGate::<F>::extend_not_gadget_unchecked_length(&mut gates, n, 0);
    ConstraintSystem::create(gates).public(1).build().unwrap()
}

/// Kimchi's `setup_not_xor` then `test_not_xor`, printing.
fn not_xor<const RR: usize, G: KimchiCurve<RR>>(k: u8, input: Option<G::ScalarField>, bits: Option<usize>, verify: bool)
    -> ([Vec<G::ScalarField>; COLUMNS], ConstraintSystem<G::ScalarField>)
where
    G::BaseField: PrimeField,
{
    let rng = &mut rng(k);
    let input = rng.gen(input, bits);
    input_line(input, bits);
    let real = max(input.to_biguint().bitlen(), bits.unwrap_or(0));
    let cs = cs_xor::<G::ScalarField>(real);
    let w = witness_checked(input, bits);
    witness(&w);
    if verify {
        rows::<RR, G>(&cs, &w);
    }
    (w, cs)
}

fn input_line<F: PrimeField>(input: F, bits: Option<usize>) {
    crate::common::input("in", &[input]);
    number("bits", bits.unwrap_or(0));
    number("given", usize::from(bits.is_some()));
}

/// Kimchi's `setup_not_gnrc` then `test_not_gnrc`, printing.
fn not_gnrc<const RR: usize, G: KimchiCurve<RR>>(k: u8, inputs: Option<Vec<G::ScalarField>>, bits: usize, len: Option<usize>, verify: bool)
    -> ([Vec<G::ScalarField>; COLUMNS], ConstraintSystem<G::ScalarField>)
where
    G::BaseField: PrimeField,
{
    let rng = &mut rng(k);
    let inputs = match inputs {
        Some(i) => i,
        None => (0..len.unwrap()).map(|_| rng.gen_field_with_bits(bits)).collect(),
    };
    input("inputs", &inputs);
    number("bits", bits);
    let cs = cs_gnrc::<G::ScalarField>(inputs.len());
    let w = witness_unchecked(&inputs, bits);
    witness(&w);
    if verify {
        rows::<RR, G>(&cs, &w);
    }
    (w, cs)
}

pub fn run() {
    {
        test("test_prove_and_verify_not_xor");
        let bits = 64;
        let rng = &mut rng(40);
        let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
        CircuitGate::<Fp>::extend_not_gadget_checked_length(&mut gates, 0, bits);
        let x: Fp = rng.gen_field_with_bits(bits);
        input("in", &[x]);
        let w = witness_checked(x, Some(bits));
        witness(&w);
        let public = vec![Fp::from(2u32).pow([bits as u64]) - Fp::one()];
        verdict("prove", &prove_and_verify(gates, w, public));
    }
    {
        test("test_prove_and_verify_five_not_gnrc");
        let bits = 64;
        let rng = &mut rng(41);
        let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
        CircuitGate::<Fp>::extend_not_gadget_unchecked_length(&mut gates, 5, 0);
        let inputs: Vec<Fp> = (0..5).map(|_| rng.gen_field_with_bits(bits)).collect();
        input("inputs", &inputs);
        let w = witness_unchecked(&inputs, bits);
        witness(&w);
        let public = vec![Fp::from(2u32).pow([bits as u64]) - Fp::one()];
        verdict("prove", &prove_and_verify(gates, w, public));
    }
    for i in 0..16u8 {
        let x = Fp::from(i);
        test(&format!("test_not_xor_all_crumb {i} none"));
        not_xor::<R, Vesta>(0, Some(x), None, true);
        for c in (4..=16).step_by(4) {
            test(&format!("test_not_xor_all_crumb {i} {c}"));
            not_xor::<R, Vesta>(0, Some(x), Some(c), true);
        }
    }
    for i in 2..=7u32 {
        let bits = 2u32.pow(i) as usize;
        let x: Fp = rng(42 + i as u8).gen_field_with_bits(bits);
        test(&format!("test_not_xor_crumbs_random {bits} some"));
        not_xor::<R, Vesta>(0, Some(x), Some(bits), true);
        test(&format!("test_not_xor_crumbs_random {bits} none"));
        not_xor::<R, Vesta>(0, Some(x), None, true);
    }
    {
        let rng = &mut rng(50);
        let x: Fp = rng.gen_field_with_bits(200);
        test("test_not_xor_big_random vesta");
        not_xor::<R, Vesta>(0, Some(x), None, true);
        let y: mina_curves::pasta::Fq = rng.gen_field_with_bits(200);
        test("test_not_xor_big_random pallas");
        not_xor::<R, Pallas>(0, Some(y), None, true);
    }
    for (name, len, k) in [("test_not_gnrc_double", 2, 51), ("test_not_gnrc_single", 1, 52)] {
        test(&format!("{name} vesta"));
        not_gnrc::<R, Vesta>(k, None, 64, Some(len), true);
        test(&format!("{name} pallas"));
        not_gnrc::<R, Pallas>(k + 100, None, 64, Some(len), true);
    }
    {
        let rng = &mut rng(53);
        let inputs: Vec<Fp> = (0..5).map(|i| rng.gen_field_with_bits(4 + i)).collect();
        test("test_not_gnrc_vector vesta");
        not_gnrc::<R, Vesta>(0, Some(inputs), 254, None, true);
        let inputs: Vec<mina_curves::pasta::Fq> = (0..5).map(|i| rng.gen_field_with_bits(4 + i)).collect();
        test("test_not_gnrc_vector pallas");
        not_gnrc::<R, Pallas>(0, Some(inputs), 254, None, true);
    }
    {
        test("test_bad_not_gnrc");
        let (mut w, cs) = not_gnrc::<R, Vesta>(54, None, 64, Some(1), false);
        gates_line(&cs.gates);
        w[0][0] += Fp::one();
        verdict("step0", &cs.gates[0].verify_witness::<R, Vesta>(0, &w, &cs, &w[0][0..cs.public]));
        w[0][1] += Fp::one();
        verdict("step1", &cs.gates[1].verify::<R, Vesta>(1, &w, &cs, &[]));
    }
    {
        test("test_bad_not_xor");
        let (mut w, cs) = not_xor::<R, Vesta>(55, None, Some(16), false);
        gates_line(&cs.gates);
        w[0][0] += Fp::one();
        verdict("step0", &cs.gates[0].verify_witness::<R, Vesta>(0, &w, &cs, &w[0][0..cs.public]));
        w[1][1] += Fp::one();
        verdict("step1", &cs.gates[1].verify_witness::<R, Vesta>(1, &w, &cs, &w[0][0..cs.public]));
        w[0][0] = Fp::zero();
        w[1][1] = Fp::zero();
        for c in 7..=10 {
            w[c][1] = Fp::zero();
        }
        verdict("prove", &prove_and_verify(cs.gates.to_vec(), w, vec![]));
    }
}
