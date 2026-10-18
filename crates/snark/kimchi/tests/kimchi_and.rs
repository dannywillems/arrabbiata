//! Kimchi's `kimchi/src/tests/and.rs`, test by test, with the same inputs
//! and outputs. The circuit is our AND gadget, its inputs placed: its rows
//! are the symbolic run, its witness the numeric run, both checked against
//! Kimchi's run of the same test (`gate_tests/`), with every verdict.

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_gadget::{Cell, Placed, TypedGadget};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem},
    gates::Gates,
    serialization::to_bytes,
};
use arrabbiata_kimchi_gadgets::{
    and::And,
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{
    Case, ForBits, case, debug, for_bits, kimchi_rows, kimchi_witness, pallas_gates,
    prove_and_verify, prove_and_verify_pallas, rows, vesta_gates,
};

/// The rows and the witness of the AND of `input` on words of `B` bits,
/// their cells in columns 0 and 1 of the first row (Kimchi's
/// `create_test_gates_and` and `create_and_witness`).
struct AndCase<F> {
    input: [F; 2],
}

impl<F: PrimeField> ForBits for AndCase<F> {
    type Output = (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]);

    fn call<const B: usize>(self) -> Self::Output {
        let gadget = Placed {
            gadget: And {
                xor: Xor::<F, B> { table: xor_table() },
            },
            cells: vec![Cell::new(0, 0), Cell::new(1, 0)],
        };
        let rows = And::<F, B>::ROWS;
        (
            kimchi_rows(&gadget, rows),
            kimchi_witness(&gadget, &self.input, rows),
        )
    }
}

/// The AND of `input` on `bytes` bytes.
fn and<F: PrimeField>(input: [F; 2], bytes: usize) -> (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]) {
    for_bits(8 * bytes, AndCase { input })
}

/// Kimchi's `setup_and` with the inputs of Kimchi's run, and with `verify`
/// its `test_and`: every row's verdict.
fn and_test<F, G>(
    name: &str,
    gates_of: &G,
    verify: bool,
) -> (Case, ConstraintSystem<F>, [Vec<F>; COLUMNS])
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let [in1, in2]: [F; 2] = case.elements("in").try_into().unwrap();
    let bytes = case.number("bytes");
    let (gates, witness) = and([in1, in2], bytes);
    let cs = ConstraintSystem::create(gates).build().unwrap();
    assert_eq!(rows(&witness), case.witnesses[0], "{name}");
    // Kimchi's check_and: the sum, the XOR and the AND on the AND row.
    let and_row = witness[0].len() - 1;
    assert_eq!(witness[3][and_row], in1 + in2);
    assert_eq!(witness[4][and_row], witness[2][0]);
    if verify {
        for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
            let ours = gate.verify_witness(row, &witness, gates_of);
            assert_eq!(
                debug(&ours),
                case.verdict(&format!("row{row}")),
                "{name} row {row}"
            );
        }
    }
    (case, cs, witness)
}

/// A test on Vesta then on Pallas.
fn both(name: &str) {
    and_test::<Fp, _>(&format!("{name} vesta"), &vesta_gates(), true);
    and_test::<Fq, _>(&format!("{name} pallas"), &pallas_gates(), true);
}

#[test]
fn test_prove_and_verify() {
    let bytes = 8;
    let case = gate_tests::case("and test_prove_and_verify vesta");
    let [in1, in2]: [Fp; 2] = case.elements("in").try_into().unwrap();
    let (gates, witness) = and([in1, in2], bytes);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
    let case = gate_tests::case("and test_prove_and_verify pallas");
    let [in1, in2]: [Fq; 2] = case.elements("in").try_into().unwrap();
    let (gates, witness) = and([in1, in2], bytes);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify_pallas(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_and64_alternating() {
    both("test_and64_alternating");
}

#[test]
fn test_and64_zeros() {
    both("test_and64_zeros");
}

#[test]
fn test_and8_random() {
    both("test_and8_random");
}

#[test]
fn test_and16_random() {
    both("test_and16_random");
}

#[test]
fn test_and32_random() {
    both("test_and32_random");
}

#[test]
fn test_and64_random() {
    both("test_and64_random");
}

#[test]
fn test_and128_random() {
    both("test_and128_random");
}

#[test]
fn test_and_overflow() {
    both("test_and_overflow");
}

#[test]
fn test_and_overflow_one() {
    both("test_and_overflow_one");
}

#[test]
fn test_and_bad_decomposition() {
    let gates_of = vesta_gates();
    let (case, cs, mut w) = and_test::<Fp, _>("test_and_bad_decomposition", &gates_of, false);
    let mut step = 0;
    let mut check = |w: &[Vec<Fp>; COLUMNS]| {
        let ours = cs.gates[0].verify_witness(0, w, &gates_of);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("step{step}")),
            "step {step}"
        );
        step += 1;
    };
    let add = |w: &mut [Vec<Fp>; COLUMNS], col: usize, row: usize, x: Fp| {
        w[col][row] = w[col][row] + x;
    };
    let (xor_row, and_row) = (0, 2);
    for col in 0..COLUMNS {
        add(&mut w, col, xor_row, Fp::ONE);
        if col < 2 {
            check(&w);
            add(&mut w, col, and_row, Fp::ONE);
        }
        if col == 2 {
            check(&w);
            add(&mut w, 4, and_row, Fp::ONE);
        }
        check(&w);
        add(&mut w, col, xor_row, -Fp::ONE);
        if col < 2 {
            add(&mut w, col, and_row, -Fp::ONE);
        }
        if col == 2 {
            add(&mut w, 4, and_row, -Fp::ONE);
        }
    }
    check(&w);
}

#[test]
fn test_bad_and() {
    let gates_of = vesta_gates();
    let (case, cs, mut w) = and_test::<Fp, _>("test_bad_and", &gates_of, false);
    w[2][0] = Fp::ZERO;
    for i in 1..=4 {
        w[COLUMNS - i][0] = Fp::ZERO;
    }
    w[4][2] = Fp::ZERO;
    let result = cs
        .verify_witness(&w, &[], &gates_of)
        .map_err(|e| format!("{e:?}"));
    assert_eq!(debug(&result), case.verdict("verify"));
}

#[test]
fn test_serialization_regression() {
    // RNG_SEED draws the inputs, then the proof's randomness: the words
    // Kimchi's prover drew, replayed, give Kimchi's proof, line for line
    // (Kimchi's test compares its serialization byte for byte).
    let case = case("test_serialization_regression");
    assert_eq!(case.number("other"), 0, "Kimchi drew only u64 words");
    let [in1, in2]: [Fp; 2] = case.elements("in").try_into().unwrap();
    let words: Vec<u64> = case
        .input("words")
        .iter()
        .map(|w| w.parse().unwrap())
        .collect();
    let (gates, witness) = and([in1, in2], 8);
    assert_eq!(rows(&witness), case.witnesses[0]);
    // Kimchi's TestFramework proves on its precomputed test key, of 2^16
    // generators: our key of that size, which is the same.
    let srs = case.number("srs");
    // Our rows are Kimchi's up to the order in each copy cycle; the proof,
    // which reads that order through the sigmas, is made on Kimchi's.
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let proof = gate_tests::prove_replayed(
        gate_tests::kimchi_gates(&case.gates[0]),
        witness,
        words,
        srs,
    );
    let mut expected = case.proof.clone();
    expected.sort();
    assert_eq!(gate_tests::proof_lines("serialization", &proof), expected);
    // And its bytes are the ones Kimchi's test hard-codes for it
    // (`buf_expected`, `rmp_serde::to_vec` of Kimchi's proof).
    let serialized = case.input("serialized")[0].as_str();
    let ours: String = to_bytes(&proof)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert!(ours == serialized, "our bytes are not Kimchi's");
}
