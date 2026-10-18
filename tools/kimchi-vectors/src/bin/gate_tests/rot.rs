//! kimchi/src/tests/rot.rs
use ark_ff::{One, PrimeField, Zero};
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        gate::{CircuitGate, Connect},
        polynomials::{
            generic::GenericGateSpec,
            keccak::OFF,
            rot::{self, RotMode},
        },
        wires::{Wire, COLUMNS},
    },
    curve::KimchiCurve,
};
use mina_curves::pasta::{Fp, Pallas, Vesta};
use o1_utils::Two;
use rand::Rng;

use crate::common::*;

fn side_name(side: RotMode) -> &'static str {
    if side == RotMode::Left { "left" } else { "right" }
}

fn gadget<const R: usize, G: KimchiCurve<R>>(rot: u32, side: RotMode) -> Vec<CircuitGate<G::ScalarField>>
where
    G::BaseField: PrimeField,
{
    let mut gates = vec![CircuitGate::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
    CircuitGate::extend_rot(&mut gates, rot, side, 0);
    gates
}

fn rot_witness<F: PrimeField>(word: u64, rot: u32, side: RotMode) -> [Vec<F>; COLUMNS] {
    let mut w: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::zero()]);
    rot::extend_rot(&mut w, word, rot, side);
    w
}

/// Kimchi's `test_rot`: rows 0 to 2.
fn test_rot<const R: usize, G: KimchiCurve<R>>(label: &str, word: u64, rot: u32, side: RotMode)
where
    G::BaseField: PrimeField,
{
    test(label);
    number("word", word as usize);
    number("rot", rot as usize);
    println!("input side {}", side_name(side));
    let cs = ConstraintSystem::create(gadget::<R, G>(rot, side)).build().unwrap();
    let w = rot_witness::<G::ScalarField>(word, rot, side);
    witness(&w);
    gates_line(&cs.gates);
    for row in 0..=2 {
        let result = cs.gates[row].verify_witness::<R, G>(row, &w, &cs, &w[0][0..cs.public]);
        verdict(&format!("row{row}"), &result);
    }
}

pub fn run() {
    const R: usize = 55;
    {
        let rng = &mut rng(20);
        let rot = rng.gen_range(1..64);
        let word = rng.gen_range(0..2u128.pow(64)) as u64;
        test("test_prove_and_verify vesta");
        number("word", word as usize);
        number("rot", rot as usize);
        let w = rot_witness::<Fp>(word, rot, RotMode::Left);
        witness(&w);
        verdict("prove", &prove_and_verify(gadget::<R, Vesta>(rot, RotMode::Left), w, vec![]));
        let rng = &mut crate::common::rng(21);
        let rot = rng.gen_range(1..64);
        let word = rng.gen_range(0..2u128.pow(64)) as u64;
        test("test_prove_and_verify pallas");
        number("word", word as usize);
        number("rot", rot as usize);
        let w = rot_witness(word, rot, RotMode::Left);
        witness(&w);
        verdict("prove", &prove_and_verify_pallas(gadget::<R, Pallas>(rot, RotMode::Left), w, vec![]));
    }
    {
        let rng = &mut rng(22);
        let rot = rng.gen_range(1..=63);
        let word = rng.gen_range(0..2u128.pow(64)) as u64;
        test_rot::<R, Vesta>("test_rot_random vesta left", word, rot, RotMode::Left);
        test_rot::<R, Vesta>("test_rot_random vesta right", word, rot, RotMode::Right);
        test_rot::<R, Pallas>("test_rot_random pallas left", word, rot, RotMode::Left);
        test_rot::<R, Pallas>("test_rot_random pallas right", word, rot, RotMode::Right);
    }
    {
        let word = rng(23).gen_range(0..2u128.pow(64)) as u64;
        test_rot::<R, Pallas>("test_zero_rot", word, 0, RotMode::Left);
        let word = rng(24).gen_range(0..2u128.pow(64)) as u64;
        test_rot::<R, Pallas>("test_large_rot", word, 64, RotMode::Left);
    }
    {
        let rng = &mut rng(25);
        let rot = rng.gen_range(1..=63);
        let word = rng.gen_range(0..2u128.pow(64)) as u64;
        test("test_bad_constraints");
        number("word", word as usize);
        number("rot", rot as usize);
        let cs = ConstraintSystem::create(gadget::<R, Vesta>(rot, RotMode::Left)).build().unwrap();
        let mut w = rot_witness::<Fp>(word, rot, RotMode::Left);
        witness(&w);
        gates_line(&cs.gates);
        let mut step = 0;
        let mut check = |w: &[Vec<Fp>; COLUMNS], row: usize| {
            let result = cs.gates[row].verify_witness::<R, Vesta>(row, w, &cs, &w[0][0..cs.public]);
            verdict(&format!("step{step}"), &result);
            step += 1;
        };
        let four = Fp::from(4u32);
        for i in 0..8 {
            w[i + 7][1] += four;
            check(&w, 1);
            w[i + 7][1] -= four;
        }
        w[0][1] += Fp::one();
        check(&w, 1);
        w[0][1] -= Fp::one();
        w[1][1] += Fp::one();
        check(&w, 1);
        w[1][1] -= Fp::one();
        for i in 0..4 {
            w[i + 3][1] += Fp::one();
            check(&w, 1);
            w[i + 3][1] -= Fp::one();
        }
        w[2][1] += Fp::one();
        w[0][3] += Fp::one();
        check(&w, 1);
        check(&w, 3);
        w[2][1] -= Fp::one();
        w[0][3] -= Fp::one();
        w[0][2] += Fp::one();
        check(&w, 1);
        check(&w, 2);
        w[0][2] -= Fp::one();
        w[0][2] += Fp::two_pow(64);
        check(&w, 2);
        w[2][2] += Fp::one();
        check(&w, 2);
        w[2][2] -= Fp::one();
        w[0][2] -= Fp::two_pow(64);
        w[0][3] += Fp::two_pow(64);
        w[2][1] += Fp::two_pow(64);
        check(&w, 3);
        w[2][3] += Fp::one();
        check(&w, 3);
    }
    {
        test("test_rot_finalization");
        let rot = 32;
        let mut gates = vec![];
        for row in 0..2 {
            gates.push(CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(row), GenericGateSpec::Pub, None));
        }
        CircuitGate::<Fp>::extend_rot(&mut gates, rot, RotMode::Left, 1);
        gates.connect_cell_pair((0, 0), (2, 0));
        let mut w: [_; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 2]);
        let input = 0xDC811727DAF22EC1u64;
        w[0][0] = input.into();
        rot::extend_rot::<Fp>(&mut w, input, rot, RotMode::Left);
        witness(&w);
        let cs = ConstraintSystem::create(gates.clone()).public(2).build().unwrap();
        rows::<R, Vesta>(&cs, &w);
        verdict("prove", &prove_and_verify(gates, w.clone(), vec![w[0][0], w[0][1]]));
    }
    {
        test("test_keccak_table");
        let offsets: Vec<String> = OFF.iter().flatten().map(|o| o.to_string()).collect();
        println!("input off {}", offsets.join(" "));
        let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
        let mut rot_row = 1;
        for col in OFF {
            for rot in col {
                if rot == 0 {
                    continue;
                }
                let mut rot64 = CircuitGate::create_rot64(rot_row, rot as u32);
                rot_row += rot64.len();
                gates.append(&mut rot64);
                gates.connect_64bit(0, rot_row - 1);
            }
        }
        let cs = ConstraintSystem::create(gates).build().unwrap();
        gates_line(&cs.gates);
        let rng = &mut rng(26);
        let state: [[u64; 5]; 5] =
            core::array::from_fn(|_| core::array::from_fn(|_| rng.gen_range(0..2u128.pow(64)) as u64));
        let words: Vec<String> = state.iter().flatten().map(|w| w.to_string()).collect();
        println!("input state {}", words.join(" "));
        let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero()]);
        for (y, col) in OFF.iter().enumerate() {
            for (x, &rot) in col.iter().enumerate() {
                if rot == 0 {
                    continue;
                }
                rot::extend_rot(&mut w, state[x][y], rot as u32, RotMode::Left);
            }
        }
        witness(&w);
        for row in 0..=48 {
            let result = cs.gates[row].verify_witness::<R, Vesta>(row, &w, &cs, &w[0][0..cs.public]);
            verdict(&format!("row{row}"), &result);
        }
    }
}
