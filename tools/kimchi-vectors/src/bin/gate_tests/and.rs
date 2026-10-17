//! kimchi/src/tests/and.rs
use ark_ff::{One, PrimeField, Zero};
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        gate::CircuitGate,
        polynomials::and,
        wires::COLUMNS,
    },
    curve::KimchiCurve,
};
use mina_curves::pasta::{Fp, Fq, Pallas, Vesta};
use num_bigint::BigUint;
use o1_utils::{FieldHelpers, RandomField};

use crate::common::*;

const R: usize = 55;

fn gates<F: PrimeField>(bytes: usize) -> Vec<CircuitGate<F>> {
    let mut gates = vec![];
    CircuitGate::<F>::extend_and(&mut gates, bytes);
    gates
}

/// Kimchi's `setup_and` and, with `verify`, `test_and`.
fn and_test<const RR: usize, G: KimchiCurve<RR>>(
    label: &str,
    k: u8,
    in1: Option<G::ScalarField>,
    in2: Option<G::ScalarField>,
    bytes: usize,
    verify: bool,
) -> (ConstraintSystem<G::ScalarField>, [Vec<G::ScalarField>; COLUMNS])
where
    G::BaseField: PrimeField,
{
    test(label);
    let rng = &mut rng(k);
    let cs = ConstraintSystem::create(gates::<G::ScalarField>(bytes)).build().unwrap();
    let input1 = rng.gen(in1, Some(bytes * 8));
    let input2 = rng.gen(in2, Some(bytes * 8));
    input("in", &[input1, input2]);
    number("bytes", bytes);
    let w = and::create_and_witness(input1, input2, bytes);
    witness(&w);
    if verify {
        rows::<RR, G>(&cs, &w);
    }
    (cs, w)
}

fn modulus_minus_one<F: PrimeField>() -> F {
    F::from_biguint(&(F::modulus_biguint() - BigUint::one())).unwrap()
}

pub fn run() {
    {
        let bytes = 8;
        test("and test_prove_and_verify vesta");
        let rng = &mut rng(60);
        let input1: Fp = rng.gen(None, Some(bytes * 8));
        let input2: Fp = rng.gen(None, Some(bytes * 8));
        input("in", &[input1, input2]);
        let w = and::create_and_witness(input1, input2, bytes);
        witness(&w);
        verdict("prove", &prove_and_verify(gates(bytes), w, vec![]));
        test("and test_prove_and_verify pallas");
        let rng = &mut crate::common::rng(61);
        let input1: Fq = rng.gen(None, Some(bytes * 8));
        let input2: Fq = rng.gen(None, Some(bytes * 8));
        input("in", &[input1, input2]);
        let w = and::create_and_witness(input1, input2, bytes);
        witness(&w);
        verdict("prove", &prove_and_verify_pallas(gates(bytes), w, vec![]));
    }
    and_test::<R, Vesta>("test_and64_alternating vesta", 0, Some(Fp::from(0x5A5A5A5A5A5A5A5Au64)), Some(Fp::from(0xA5A5A5A5A5A5A5A5u64)), 8, true);
    and_test::<R, Pallas>("test_and64_alternating pallas", 0, Some(Fq::from(0x5A5A5A5A5A5A5A5Au64)), Some(Fq::from(0xA5A5A5A5A5A5A5A5u64)), 8, true);
    and_test::<R, Vesta>("test_and64_zeros vesta", 0, Some(Fp::zero()), Some(Fp::zero()), 8, true);
    and_test::<R, Pallas>("test_and64_zeros pallas", 0, Some(Fq::zero()), Some(Fq::zero()), 8, true);
    for (name, bytes, k) in [
        ("test_and8_random", 1, 62),
        ("test_and16_random", 2, 63),
        ("test_and32_random", 4, 64),
        ("test_and64_random", 8, 65),
        ("test_and128_random", 16, 66),
    ] {
        and_test::<R, Vesta>(&format!("{name} vesta"), k, None, None, bytes, true);
        and_test::<R, Pallas>(&format!("{name} pallas"), k + 100, None, None, bytes, true);
    }
    and_test::<R, Vesta>("test_and_overflow vesta", 0, Some(modulus_minus_one()), Some(modulus_minus_one()), 32, true);
    and_test::<R, Pallas>("test_and_overflow pallas", 0, Some(modulus_minus_one()), Some(modulus_minus_one()), 32, true);
    and_test::<R, Vesta>("test_and_overflow_one vesta", 0, Some(modulus_minus_one()), Some(Fp::one()), 32, true);
    and_test::<R, Pallas>("test_and_overflow_one pallas", 0, Some(modulus_minus_one()), Some(Fq::one()), 32, true);
    {
        let (cs, mut w) = and_test::<R, Vesta>("test_and_bad_decomposition", 67, None, None, 2, false);
        gates_line(&cs.gates);
        let mut step = 0;
        let mut check = |w: &[Vec<Fp>; COLUMNS]| {
            let result = cs.gates[0].verify_witness::<R, Vesta>(0, w, &cs, &w[0][0..cs.public]);
            verdict(&format!("step{step}"), &result);
            step += 1;
        };
        let (xor_row, and_row) = (0, 2);
        for col in 0..COLUMNS {
            w[col][xor_row] += Fp::one();
            if col < 2 {
                check(&w);
                w[col][and_row] += Fp::one();
            }
            if col == 2 {
                check(&w);
                w[4][and_row] += Fp::one();
            }
            check(&w);
            w[col][xor_row] -= Fp::one();
            if col < 2 {
                w[col][and_row] -= Fp::one();
            }
            if col == 2 {
                w[4][and_row] -= Fp::one();
            }
        }
        check(&w);
    }
    {
        let (cs, mut w) = and_test::<R, Vesta>("test_bad_and", 68, None, None, 2, false);
        gates_line(&cs.gates);
        w[2][0] = Fp::zero();
        for i in 1..=4 {
            w[COLUMNS - i][0] = Fp::zero();
        }
        w[4][2] = Fp::zero();
        let result = cs.verify_witness::<R, Vesta>(&w, &[]).map_err(|e| format!("{e:?}"));
        verdict("verify", &result);
    }
}

/// Kimchi's `test_serialization_regression`: RNG_SEED draws the inputs,
/// then the proof's randomness; the words the prover draws are recorded.
pub fn serialization() {
    const RNG_SEED: [u8; 32] = [
        255, 27, 111, 55, 22, 200, 10, 1, 0, 136, 56, 16, 2, 30, 31, 77, 18, 11, 40, 53, 5, 8, 189,
        92, 97, 25, 21, 12, 13, 44, 14, 12,
    ];
    use rand::SeedableRng;
    test("test_serialization_regression");
    let rng = &mut rand::rngs::StdRng::from_seed(RNG_SEED);
    let bytes = 8;
    let input1: Fp = rng.gen(None, Some(bytes * 8));
    let input2: Fp = rng.gen(None, Some(bytes * 8));
    input("in", &[input1, input2]);
    let w = and::create_and_witness(input1, input2, bytes);
    witness(&w);
    let (proof, words, other) = prove_recorded(gates(bytes), w, rng);
    number("other", other);
    let items: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    println!("input words {}", items.join(" "));
    print_proof("serialization", &proof);
}
