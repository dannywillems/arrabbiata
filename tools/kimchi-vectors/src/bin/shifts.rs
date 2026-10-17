//! Kimchi's permutation shifts (`ConstraintSystem::shift`, sampled by
//! `Shifts::new` with Blake2b512) for circuits of a few sizes, over `Fp`.
use ark_ff::{BigInteger, PrimeField};
use kimchi::circuits::{
    constraints::ConstraintSystem,
    gate::{CircuitGate, GateType},
    wires::Wire,
};
use mina_curves::pasta::Fp;

fn hex<F: PrimeField>(x: F) -> String {
    x.into_bigint().to_bytes_be().iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    for gates in [10usize, 100, 1000] {
        let circuit: Vec<CircuitGate<Fp>> = (0..gates)
            .map(|row| CircuitGate::new(GateType::Zero, Wire::for_row(row), vec![]))
            .collect();
        let cs = ConstraintSystem::<Fp>::create(circuit).build().unwrap();
        let shifts: Vec<String> = cs.shift.iter().map(|s| hex(*s)).collect();
        println!("shifts {} {}", cs.domain.d1.size, shifts.join(" "));
    }
}
