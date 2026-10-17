//! kimchi/src/tests/xor.rs
use core::cmp::max;

use ark_ff::{Field, One, PrimeField, Zero};
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        gate::{CircuitGate, Connect},
        polynomials::{generic::GenericGateSpec, xor},
        wires::{Wire, COLUMNS},
    },
    curve::KimchiCurve,
};
use mina_curves::pasta::{Fp, Pallas, Vesta};
use o1_utils::{BigUintHelpers, FieldHelpers, RandomField};

use crate::common::*;

fn setup<const R: usize, G: KimchiCurve<R>>(
    k: u8,
    in1: Option<G::ScalarField>,
    in2: Option<G::ScalarField>,
    bits: Option<usize>,
) -> (ConstraintSystem<G::ScalarField>, [Vec<G::ScalarField>; COLUMNS])
where
    G::BaseField: PrimeField,
{
    let rng = &mut rng(k);
    let input1 = rng.gen(in1, bits);
    let input2 = rng.gen(in2, bits);
    let bits1 = input1.to_biguint().bitlen();
    let bits2 = input2.to_biguint().bitlen();
    let bits = max(bits.unwrap_or(0), max(bits1, bits2));
    input("in", &[input1, input2]);
    number("bits", bits);
    let mut gates = vec![];
    CircuitGate::<G::ScalarField>::extend_xor_gadget(&mut gates, bits);
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let w = xor::create_xor_witness(input1, input2, bits);
    witness(&w);
    (cs, w)
}

fn test_xor<const R: usize, G: KimchiCurve<R>>(
    k: u8,
    in1: Option<G::ScalarField>,
    in2: Option<G::ScalarField>,
    bits: Option<usize>,
) where
    G::BaseField: PrimeField,
{
    let (cs, w) = setup::<R, G>(k, in1, in2, bits);
    rows::<R, G>(&cs, &w);
}

pub fn run() {
    const R: usize = 55;
    {
        test("test_prove_and_verify_xor");
        let rng = &mut rng(1);
        let bits = 64;
        let mut gates = vec![];
        CircuitGate::<Fp>::extend_xor_gadget(&mut gates, bits);
        let input1: Fp = rng.gen_field_with_bits(bits);
        let input2: Fp = rng.gen_field_with_bits(bits);
        input("in", &[input1, input2]);
        number("bits", bits);
        let w = xor::create_xor_witness(input1, input2, bits);
        witness(&w);
        verdict("prove", &prove_and_verify(gates, w, vec![]));
    }
    test("test_xor64_alternating");
    test_xor::<R, Vesta>(0, Some(Fp::from(0x5A5A5A5A5A5A5A5Au64)), Some(Fp::from(0xA5A5A5A5A5A5A5A5u64)), Some(64));
    test("test_xor64_zeros");
    test_xor::<R, Vesta>(0, Some(Fp::zero()), Some(Fp::zero()), Some(64));
    test("test_xor64_zero_one");
    let ones = Fp::from(2u128).pow([64]) - Fp::one();
    test_xor::<R, Vesta>(0, Some(Fp::zero()), Some(ones), None);
    for (name, bits, k) in [
        ("test_xor8_random", 8, 2),
        ("test_xor16_random", 16, 3),
        ("test_xor32_random", 32, 4),
        ("test_xor64_random", 64, 5),
        ("test_xor128_random", 128, 6),
    ] {
        test(&format!("{name} vesta"));
        test_xor::<R, Vesta>(k, None, None, Some(bits));
        test(&format!("{name} pallas"));
        test_xor::<R, Pallas>(k + 100, None, None, Some(bits));
    }
    {
        test("test_bad_xor_decompsition");
        let (cs, mut w) = setup::<R, Vesta>(7, None, None, Some(16));
        gates_line(&cs.gates);
        for col in 0..COLUMNS {
            w[col][0] += Fp::one();
            let result = cs.gates[0].verify_witness::<R, Vesta>(0, &w, &cs, &w[0][0..cs.public]);
            verdict(&format!("col{col}"), &result);
            w[col][0] -= Fp::one();
        }
        let result = cs.gates[0].verify_witness::<R, Vesta>(0, &w, &cs, &w[0][0..cs.public]);
        verdict("undone", &result);
    }
    {
        test("test_extend_xor");
        let bits = 16;
        let rng = &mut rng(8);
        let input1: Fp = rng.gen(None, Some(bits));
        let input2: Fp = rng.gen(None, Some(bits));
        let bits = max(bits, max(input1.to_biguint().bitlen(), input2.to_biguint().bitlen()));
        input("in", &[input1, input2]);
        number("bits", bits);
        let mut gates = vec![];
        for row in 0..2 {
            gates.push(CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(row), GenericGateSpec::Pub, None));
        }
        CircuitGate::<Fp>::extend_xor_gadget(&mut gates, bits);
        gates.connect_cell_pair((0, 0), (2, 0));
        gates.connect_cell_pair((1, 0), (2, 1));
        let cs = ConstraintSystem::create(gates).public(2).build().unwrap();
        let mut w: [_; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 2]);
        w[0][0] = input1;
        w[0][1] = input2;
        xor::extend_xor_witness::<Fp>(&mut w, input1, input2, bits);
        witness(&w);
        rows::<R, Vesta>(&cs, &w);
    }
    {
        test("test_bad_xor");
        let bits = 16;
        let rng = &mut rng(9);
        let input1: Fp = rng.gen(None, Some(bits));
        let input2: Fp = rng.gen(None, Some(bits));
        let bits = max(bits, max(input1.to_biguint().bitlen(), input2.to_biguint().bitlen()));
        input("in", &[input1, input2]);
        number("bits", bits);
        let mut gates = vec![];
        CircuitGate::<Fp>::extend_xor_gadget(&mut gates, bits);
        let mut w = xor::create_xor_witness(input1, input2, bits);
        w[2][0] = Fp::zero();
        for i in 1..=4 {
            w[COLUMNS - i][0] = Fp::zero();
        }
        verdict("prove", &prove_and_verify(gates, w, vec![]));
    }
    {
        test("test_xor_finalization");
        let mut gates = vec![];
        for row in 0..2 {
            gates.push(CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(row), GenericGateSpec::Pub, None));
        }
        CircuitGate::<Fp>::extend_xor_gadget(&mut gates, 128);
        gates.connect_cell_pair((0, 0), (2, 0));
        gates.connect_cell_pair((1, 0), (2, 1));
        let mut w: [_; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 2]);
        let input1: Fp = 0xDC811727DAF22EC15927D6AA275F406Bu128.into();
        let input2: Fp = 0xA4F4417AF072DF9016A1EAB458DA80D1u128.into();
        w[0][0] = input1;
        w[0][1] = input2;
        xor::extend_xor_witness::<Fp>(&mut w, input1, input2, 128);
        witness(&w);
        let cs = ConstraintSystem::create(gates.clone()).public(2).build().unwrap();
        rows::<R, Vesta>(&cs, &w);
        verdict("prove", &prove_and_verify(gates, w.clone(), vec![w[0][0], w[0][1]]));
    }
}
