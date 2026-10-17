//! Kimchi's `kimchi/src/tests/endomul_scalar.rs`, test by test, with the
//! same inputs and outputs: the witness of 100 scalar challenges computed
//! by our EndoMulScalar gates (its Blake2b512 digest against Kimchi's), the
//! scalars they stand for, the proof's verdict; and the degenerate EndoMul
//! row Kimchi's `verify_endomul` refuses (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, GateType, Wire},
    gates::KimchiGates,
    oracle::ScalarChallenge,
    polynomials::endomul_scalar::gen_witness,
};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::kimchi::pallas_base;
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, prove_and_verify, rows};

/// The Blake2b512 digest of the witness as the tool prints it: each row's
/// cells in hex, joined by spaces, the rows by newlines.
fn digest(witness: &[Vec<Fp>; COLUMNS]) -> String {
    let lines: Vec<String> = rows(witness).iter().map(|row| row.join(" ")).collect();
    Blake2b512::digest(lines.join("\n").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn endomul_scalar_test() {
    let case = case("endomul_scalar_test");
    let (num_bits, rows_per_scalar) = (128, 8);
    let coeff: Fp = case.elements("coeff")[0];
    assert_eq!(coeff, Vesta::lambda(), "Kimchi's endo_r is Vesta's lambda");
    let xs: Vec<Fp> = case.elements("x");
    let results: Vec<Fp> = case.elements("result");
    let gates: Vec<CircuitGate<Fp>> = (0..xs.len() * rows_per_scalar)
        .map(|row| CircuitGate::new(GateType::EndoMulScalar, Wire::for_row(row), vec![]))
        .collect();
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);
    for (x, expected) in xs.iter().zip(&results) {
        let result = gen_witness(&mut witness, *x, coeff, num_bits).unwrap();
        assert_eq!(result, *expected);
        assert_eq!(ScalarChallenge(*x).to_field(coeff), result);
    }
    let kimchi_digest = case.input("digest");
    assert_eq!(kimchi_digest[0], witness[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&witness));
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_degenerate_case() {
    // Kimchi's witness: P = -R, which EndoMul's constraints refuse.
    let case = case("test_degenerate_case");
    let gate = CircuitGate::new(GateType::EndoMul, Wire::for_row(0), vec![]);
    let zero = CircuitGate::new(GateType::Zero, Wire::for_row(1), vec![]);
    let gates = vec![gate.clone(), zero];
    // Kimchi builds the constraint system and does not use it further.
    ConstraintSystem::<Fp>::create(gates).build().unwrap();
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        case.witnesses[0]
            .iter()
            .map(|row| gate_tests::field(&row[c]))
            .collect()
    });
    assert_eq!(witness[6][1], Fp::from_u64(7), "n consumed");
    assert_eq!(witness[7][0], witness[4][0], "x_r = x_p");
    assert_eq!(witness[8][0], -witness[5][0], "y_r = -y_p");
    // Kimchi's verify_endomul reads the endomorphism coefficient from
    // `cs.endo`, which `ConstraintSystem::build` sets to zero ("TODO: remove
    // endo as a field"; only `ProverIndex::create` sets it). The test's
    // witness is computed with that zero too (`xq2 = cs.endo xt`), so the
    // same verdict needs the gates with a zero coefficient: with Pallas's
    // beta, constraint 7 (`(xq2 - xr) s3 = yq2 - yr`) breaks first.
    let endo_zero = KimchiGates::new(&pallas_base(), Fp::ZERO).unwrap();
    let result = gate.verify_endomul(0, &witness, &endo_zero);
    assert_eq!(debug(&result.is_err()), case.verdict("is_err"));
    assert_eq!(debug(&result), case.verdict("verify"));
}
