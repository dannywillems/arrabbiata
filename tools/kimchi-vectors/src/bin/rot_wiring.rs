//! Kimchi's rotation wiring: which cells `connect_64bit` actually wires,
//! and whether a witness with a wrong rotation passes Kimchi's checks.
#[path = "gate_tests/common.rs"]
mod common;

use ark_ff::{Field, One, Zero};
use kimchi::circuits::{
    constraints::ConstraintSystem,
    gate::CircuitGate,
    polynomials::{
        generic::GenericGateSpec,
        range_check,
        rot::{self, RotMode},
    },
    wires::{Wire, COLUMNS},
};
use mina_curves::pasta::{Fp, Vesta};

fn main() {
    let rot = 32u32;
    let word: u64 = 0xDC81_1727_DAF2_2EC1;
    let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
    CircuitGate::extend_rot(&mut gates, rot, RotMode::Left, 0);
    for (r, g) in gates.iter().enumerate() {
        let wires: Vec<String> = g.wires[..3].iter().map(|w| format!("({},{})", w.row, w.col)).collect();
        println!("wires row {r} {:?} cols0-2 -> {}", g.typ, wires.join(" "));
    }
    let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero()]);
    rot::extend_rot(&mut w, word, rot, RotMode::Left);
    let honest = w.clone();
    // The forgery: shifted' = shifted + 2^76, excess' = excess - 2^12.
    let two = |k: u64| Fp::from(2u64).pow([k]);
    let shifted = w[0][2];
    let excess = w[2][1];
    let shifted2 = shifted + two(76);
    let excess2 = excess - two(12);
    let rotated2 = shifted2 + excess2;
    let bound2 = excess2 - two(rot as u64) + two(64);
    let rc = |v: Fp| range_check::witness::create(v);
    let (b, s, e) = (rc(bound2), rc(shifted2), rc(excess2));
    w[1][1] = rotated2;
    w[2][1] = excess2;
    for c in 3..COLUMNS {
        w[c][1] = b[c][0];
    }
    for c in 0..COLUMNS {
        w[c][2] = s[c][0];
        w[c][3] = e[c][0];
    }
    println!("honest rotated {}", common::hex(honest[1][1]));
    println!("forged rotated {}", common::hex(w[1][1]));
    println!("word.rotate_left({rot}) {:x}", word.rotate_left(rot));
    let cs = ConstraintSystem::create(gates.clone()).public(1).build().unwrap();
    for row in 0..w[0].len() {
        let result = cs.gates[row].verify_witness::<55, Vesta>(row, &w, &cs, &w[0][0..1]);
        println!("forged row{row} {result:?}");
    }
    println!(
        "forged prove {:?}",
        common::prove_and_verify(gates, w, vec![Fp::zero()])
    );
    let _ = Fp::one();
}
