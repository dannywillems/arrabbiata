//! Kimchi's `kimchi/src/tests/not.rs`, test by test, with the same inputs
//! and outputs: Kimchi's rows from our constructors, the witnesses from our
//! XOR and Generic half-gates, every verdict against Kimchi's run of the
//! same test (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, Wire},
    gates::Gates,
    polynomials::{
        bit_length,
        generic::GenericGateSpec,
        not::{extend_not_witness_checked_length, extend_not_witness_unchecked_length},
    },
};
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{Case, case, debug, pallas_gates, prove_and_verify, rows, vesta_gates};

/// `2^bits - 1`.
fn all_ones<F: Field>(bits: usize) -> F {
    F::from_u64(2).pow(bits as u64) - F::ONE
}

/// Kimchi's `create_not_witness_unchecked_length`: the all-ones row, then
/// the Generic NOTs.
fn witness_unchecked<F: PrimeField>(inputs: &[F], bits: usize) -> [Vec<F>; COLUMNS] {
    let mut witness: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::ZERO; 1]);
    witness[0][0] = all_ones(bits);
    extend_not_witness_unchecked_length(&mut witness, inputs, bits).unwrap();
    witness
}

/// Kimchi's `create_not_witness_checked_length`: the all-ones row, then
/// the XOR NOT.
fn witness_checked<F: PrimeField>(input: F, bits: Option<usize>) -> [Vec<F>; COLUMNS] {
    let mut witness: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::ZERO; 1]);
    let real = bit_length(input).max(bits.unwrap_or(0));
    witness[0][0] = all_ones(real);
    extend_not_witness_checked_length(&mut witness, input, bits).unwrap();
    witness
}

/// The public all-ones row, then the gadget.
fn public_row<F: PrimeField>() -> Vec<CircuitGate<F>> {
    vec![CircuitGate::create_generic_gadget(
        Wire::for_row(0),
        GenericGateSpec::Pub,
        None,
    )]
}

/// Kimchi's `create_test_constraint_system_not_xor`.
fn cs_xor<F: PrimeField + TwoAdicField>(bits: usize) -> ConstraintSystem<F> {
    let mut gates = public_row();
    CircuitGate::extend_not_gadget_checked_length(&mut gates, 0, bits);
    ConstraintSystem::create(gates).public(1).build().unwrap()
}

/// Kimchi's `create_test_constraint_system_not_gnrc`.
fn cs_gnrc<F: PrimeField + TwoAdicField>(n: usize) -> ConstraintSystem<F> {
    let mut gates = public_row();
    CircuitGate::extend_not_gadget_unchecked_length(&mut gates, n, 0);
    ConstraintSystem::create(gates).public(1).build().unwrap()
}

/// Every row's verdict is Kimchi's.
fn verify_rows<F, G>(case: &Case, cs: &ConstraintSystem<F>, witness: &[Vec<F>; COLUMNS], gates: &G)
where
    F: PrimeField,
    G: Gates<F>,
{
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, witness, gates);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "row {row}"
        );
    }
}

/// Kimchi's `setup_not_xor`, its inputs from Kimchi's run, and with
/// `verify` its `test_not_xor`.
fn not_xor<F, G>(name: &str, gates: &G, verify: bool) -> ([Vec<F>; COLUMNS], ConstraintSystem<F>)
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let input: F = case.elements("in")[0];
    let bits = (case.number("given") == 1).then(|| case.number("bits"));
    let real = bit_length(input).max(bits.unwrap_or(0));
    let cs = cs_xor::<F>(real);
    let witness = witness_checked(input, bits);
    assert_eq!(rows(&witness), case.witnesses[0], "{name}");
    // Kimchi's check_not_xor: the output is the NOT on `real` bits.
    assert_eq!(witness[2][1], all_ones::<F>(real) - input);
    if verify {
        verify_rows(&case, &cs, &witness, gates);
    }
    (witness, cs)
}

/// Kimchi's `setup_not_gnrc`, and with `verify` its `test_not_gnrc`.
fn not_gnrc<F, G>(name: &str, gates: &G, verify: bool) -> ([Vec<F>; COLUMNS], ConstraintSystem<F>)
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let inputs: Vec<F> = case.elements("inputs");
    let bits = case.number("bits");
    let cs = cs_gnrc::<F>(inputs.len());
    let witness = witness_unchecked(&inputs, bits);
    assert_eq!(rows(&witness), case.witnesses[0], "{name}");
    // Kimchi's check_not_gnrc.
    for (i, input) in inputs.iter().enumerate() {
        let (column, row) = if i % 2 == 0 {
            (2, 1 + i / 2)
        } else {
            (5, 1 + (i - 1) / 2)
        };
        assert_eq!(witness[column][row], all_ones::<F>(bits) - *input);
    }
    if verify {
        verify_rows(&case, &cs, &witness, gates);
    }
    (witness, cs)
}

#[test]
fn test_prove_and_verify_not_xor() {
    let case = case("test_prove_and_verify_not_xor");
    let bits = 64;
    let mut gates = public_row::<Fp>();
    CircuitGate::extend_not_gadget_checked_length(&mut gates, 0, bits);
    let witness = witness_checked(case.elements::<Fp>("in")[0], Some(bits));
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![all_ones(bits)]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_prove_and_verify_five_not_gnrc() {
    let case = case("test_prove_and_verify_five_not_gnrc");
    let bits = 64;
    let mut gates = public_row::<Fp>();
    CircuitGate::extend_not_gadget_unchecked_length(&mut gates, 5, 0);
    let witness = witness_unchecked(&case.elements::<Fp>("inputs"), bits);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![all_ones(bits)]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_not_xor_all_crumb() {
    let gates = vesta_gates();
    for i in 0..16 {
        not_xor::<Fp, _>(&format!("test_not_xor_all_crumb {i} none"), &gates, true);
        for c in (4..=16).step_by(4) {
            not_xor::<Fp, _>(&format!("test_not_xor_all_crumb {i} {c}"), &gates, true);
        }
    }
}

#[test]
fn test_not_xor_crumbs_random() {
    let gates = vesta_gates();
    for i in 2..=7 {
        let bits = 1 << i;
        not_xor::<Fp, _>(
            &format!("test_not_xor_crumbs_random {bits} some"),
            &gates,
            true,
        );
        not_xor::<Fp, _>(
            &format!("test_not_xor_crumbs_random {bits} none"),
            &gates,
            true,
        );
    }
}

#[test]
fn test_not_xor_big_random() {
    not_xor::<Fp, _>("test_not_xor_big_random vesta", &vesta_gates(), true);
    not_xor::<Fq, _>("test_not_xor_big_random pallas", &pallas_gates(), true);
}

#[test]
fn test_not_gnrc_double() {
    not_gnrc::<Fp, _>("test_not_gnrc_double vesta", &vesta_gates(), true);
    not_gnrc::<Fq, _>("test_not_gnrc_double pallas", &pallas_gates(), true);
}

#[test]
fn test_not_gnrc_single() {
    not_gnrc::<Fp, _>("test_not_gnrc_single vesta", &vesta_gates(), true);
    not_gnrc::<Fq, _>("test_not_gnrc_single pallas", &pallas_gates(), true);
}

#[test]
fn test_not_gnrc_vector() {
    not_gnrc::<Fp, _>("test_not_gnrc_vector vesta", &vesta_gates(), true);
    not_gnrc::<Fq, _>("test_not_gnrc_vector pallas", &pallas_gates(), true);
}

#[test]
fn test_bad_not_gnrc() {
    let gates = vesta_gates();
    let case = case("test_bad_not_gnrc");
    let (mut witness, cs) = not_gnrc::<Fp, _>("test_bad_not_gnrc", &gates, false);
    witness[0][0] = witness[0][0] + Fp::ONE;
    let ours = cs.gates[0].verify_witness(0, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("step0"));
    witness[0][1] = witness[0][1] + Fp::ONE;
    let ours = cs.gates[1].verify(1, &witness, &gates, &[]);
    assert_eq!(debug(&ours), case.verdict("step1"));
}

#[test]
fn test_bad_not_xor() {
    let gates = vesta_gates();
    let case = case("test_bad_not_xor");
    let (mut witness, cs) = not_xor::<Fp, _>("test_bad_not_xor", &gates, false);
    witness[0][0] = witness[0][0] + Fp::ONE;
    let ours = cs.gates[0].verify_witness(0, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("step0"));
    witness[1][1] = witness[1][1] + Fp::ONE;
    let ours = cs.gates[1].verify_witness(1, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("step1"));
    witness[0][0] = Fp::ZERO;
    witness[1][1] = Fp::ZERO;
    for column in &mut witness[7..=10] {
        column[1] = Fp::ZERO;
    }
    let result = prove_and_verify(cs.gates.clone(), witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
