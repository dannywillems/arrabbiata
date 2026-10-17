//! Kimchi's `kimchi/src/tests/xor.rs`, test by test, with the same inputs
//! and outputs: each test builds Kimchi's rows with our constructors, the
//! witness with our XOR gadget, and checks the witness and every verdict
//! against Kimchi's run of the same test (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, Connect, ConstraintSystem, Wire},
    gates::Gates,
    polynomials::{
        generic::GenericGateSpec,
        xor::{create_xor_witness, extend_xor_witness},
    },
};
use arrabbiata_kimchi_gadgets::generic::GenericSpec;
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{Case, case, debug, pallas_gates, prove_and_verify, rows, vesta_gates};

/// Kimchi's `setup_xor` on the inputs of `case`: the constraint system
/// and our witness, which is Kimchi's.
fn setup<F>(case: &Case) -> (ConstraintSystem<F>, [Vec<F>; COLUMNS])
where
    F: PrimeField + TwoAdicField,
{
    let [in1, in2]: [F; 2] = case.elements("in").try_into().unwrap();
    let bits = case.number("bits");
    let mut gates = Vec::new();
    CircuitGate::<F>::extend_xor_gadget(&mut gates, bits);
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let witness = create_xor_witness(in1, in2, bits).unwrap();
    assert_eq!(rows(&witness), case.witnesses[0]);
    (cs, witness)
}

/// Kimchi's `test_xor`: every row's verdict is Kimchi's.
fn test_xor<F, G>(name: &str, gates: &G) -> [Vec<F>; COLUMNS]
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let (cs, witness) = setup::<F>(&case);
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, &witness, gates);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "{name} row {row}"
        );
        assert_eq!(ours, Ok(()));
    }
    witness
}

fn two_to(bits: u64) -> Fp {
    Fp::from_u64(2).pow(bits)
}

#[test]
fn test_prove_and_verify_xor() {
    let case = case("test_prove_and_verify_xor");
    let [in1, in2]: [Fp; 2] = case.elements("in").try_into().unwrap();
    let bits = case.number("bits");
    let mut gates = Vec::new();
    CircuitGate::<Fp>::extend_xor_gadget(&mut gates, bits);
    let witness = create_xor_witness(in1, in2, bits).unwrap();
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_xor64_alternating() {
    let witness = test_xor::<Fp, _>("test_xor64_alternating", &vesta_gates());
    assert_eq!(witness[2][0], two_to(64) - Fp::ONE);
    assert_eq!(witness[2][1], two_to(48) - Fp::ONE);
    assert_eq!(witness[2][2], two_to(32) - Fp::ONE);
    assert_eq!(witness[2][3], two_to(16) - Fp::ONE);
    assert_eq!(witness[2][4], Fp::ZERO);
}

#[test]
fn test_xor64_zeros() {
    let witness = test_xor::<Fp, _>("test_xor64_zeros", &vesta_gates());
    assert_eq!(witness[2][0], Fp::ZERO);
}

#[test]
fn test_xor64_zero_one() {
    let witness = test_xor::<Fp, _>("test_xor64_zero_one", &vesta_gates());
    assert_eq!(witness[2][0], two_to(64) - Fp::ONE);
}

/// A random test, on Vesta then on Pallas.
fn random(name: &str) {
    test_xor::<Fp, _>(&format!("{name} vesta"), &vesta_gates());
    test_xor::<Fq, _>(&format!("{name} pallas"), &pallas_gates());
}

#[test]
fn test_xor8_random() {
    random("test_xor8_random");
}

#[test]
fn test_xor16_random() {
    random("test_xor16_random");
}

#[test]
fn test_xor32_random() {
    random("test_xor32_random");
}

#[test]
fn test_xor64_random() {
    random("test_xor64_random");
}

#[test]
fn test_xor128_random() {
    random("test_xor128_random");
}

#[test]
fn test_bad_xor_decompsition() {
    let case = case("test_bad_xor_decompsition");
    let (cs, mut witness) = setup::<Fp>(&case);
    let gates = vesta_gates();
    for col in 0..COLUMNS {
        witness[col][0] = witness[col][0] + Fp::ONE;
        let ours = cs.gates[0].verify_witness(0, &witness, &gates);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("col{col}")),
            "column {col}"
        );
        witness[col][0] = witness[col][0] - Fp::ONE;
    }
    let ours = cs.gates[0].verify_witness(0, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("undone"));
}

/// Two public rows wired to the XOR's inputs, then the XOR of `bits` bits.
fn public_xor(bits: usize) -> Vec<CircuitGate<Fp>> {
    let mut gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            CircuitGate::create_generic_gadget(Wire::for_row(row), GenericGateSpec::Pub, None)
        })
        .collect();
    CircuitGate::<Fp>::extend_xor_gadget(&mut gates, bits);
    gates.connect_cell_pair((0, 0), (2, 0));
    gates.connect_cell_pair((1, 0), (2, 1));
    gates
}

/// The two public rows holding the inputs, then the XOR's witness.
fn public_witness(in1: Fp, in2: Fp, bits: usize) -> [Vec<Fp>; COLUMNS] {
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    witness[0][0] = in1;
    witness[0][1] = in2;
    extend_xor_witness(&mut witness, in1, in2, bits).unwrap();
    witness
}

#[test]
fn test_extend_xor() {
    let case = case("test_extend_xor");
    let [in1, in2]: [Fp; 2] = case.elements("in").try_into().unwrap();
    let bits = case.number("bits");
    let cs = ConstraintSystem::create(public_xor(bits))
        .public(2)
        .build()
        .unwrap();
    let witness = public_witness(in1, in2, bits);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let gates = vesta_gates();
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, &witness, &gates);
        assert_eq!(debug(&ours), case.verdict(&format!("row{row}")));
    }
}

#[test]
fn test_bad_xor() {
    let case = case("test_bad_xor");
    let [in1, in2]: [Fp; 2] = case.elements("in").try_into().unwrap();
    let bits = case.number("bits");
    let mut gates = Vec::new();
    CircuitGate::<Fp>::extend_xor_gadget(&mut gates, bits);
    let mut witness = create_xor_witness(in1, in2, bits).unwrap();
    witness[2][0] = Fp::ZERO;
    for i in 1..=4 {
        witness[COLUMNS - i][0] = Fp::ZERO;
    }
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_xor_finalization() {
    let case = case("test_xor_finalization");
    let two_64 = Fp::from_u64(1 << 32) * Fp::from_u64(1 << 32);
    let word = |hi: u64, lo: u64| Fp::from_u64(hi) * two_64 + Fp::from_u64(lo);
    let in1 = word(0xDC81_1727_DAF2_2EC1, 0x5927_D6AA_275F_406B);
    let in2 = word(0xA4F4_417A_F072_DF90, 0x16A1_EAB4_58DA_80D1);
    let gates = public_xor(128);
    let witness = public_witness(in1, in2, 128);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let cs = ConstraintSystem::create(gates.clone())
        .public(2)
        .build()
        .unwrap();
    let kimchi = vesta_gates();
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, &witness, &kimchi);
        assert_eq!(debug(&ours), case.verdict(&format!("row{row}")));
    }
    let public = vec![witness[0][0], witness[0][1]];
    let result = prove_and_verify(gates, witness, public);
    assert_eq!(debug(&result), case.verdict("prove"));
}

/// The XOR's last row: the zero constant, its three first cells on one
/// copy cycle (in whatever order the derivation writes it).
#[test]
fn the_last_row_is_a_zero_constant() {
    let (_, gates) = CircuitGate::<Fp>::create_xor_gadget(0, 16);
    let expected = CircuitGate::create_generic_gadget(
        Wire::for_row(1),
        GenericGateSpec::Spec(GenericSpec::Const(Fp::ZERO)),
        None,
    );
    assert_eq!(gates[1].typ, expected.typ);
    let trimmed = |c: &[Fp]| {
        let mut c = c.to_vec();
        while c.last() == Some(&Fp::ZERO) {
            c.pop();
        }
        c
    };
    assert_eq!(trimmed(&gates[1].coeffs), trimmed(&expected.coeffs));
    let start = Wire { row: 1, col: 0 };
    let mut cycle = vec![start];
    let mut at = gates[1].wires[0];
    while at != start {
        cycle.push(at);
        at = gates[at.row].wires[at.col];
    }
    cycle.sort();
    let expected_cycle: Vec<Wire> = Wire::for_row(1)[..3].to_vec();
    assert_eq!(cycle, expected_cycle);
}
