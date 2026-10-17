//! Kimchi's `kimchi/src/tests/endomul_scalar.rs`, test by test, with the
//! same inputs and outputs. The circuit of 100 scalar challenges is one
//! typed gadget, our EndoMulScalar rows unwired as Kimchi lays them out:
//! its rows are the symbolic run, its witness (Blake2b512 digest against
//! Kimchi's) the numeric run, with the scalars it stands for and the
//! proof's verdict; and the degenerate EndoMul row Kimchi's
//! `verify_endomul` refuses (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, GateType, Wire},
    gates::KimchiGates,
    oracle::ScalarChallenge,
};
use arrabbiata_kimchi_gadgets::endomul_scalar::{CRUMBS_PER_ROW, EndoMulScalarGate, column};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::kimchi::pallas_base;
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, kimchi_rows, kimchi_witness, prove_and_verify, rows};

/// The scalar challenges of Kimchi's test.
const SCALARS: usize = 100;
/// The bits of each challenge.
const BITS: usize = 128;
/// The crumbs of each challenge.
const CRUMBS: usize = BITS / 2;
/// The rows of each challenge.
const ROWS_PER_SCALAR: usize = CRUMBS / CRUMBS_PER_ROW;

/// One EndoMulScalar row whose start `[n0, a0, b0]` is written, not wired:
/// Kimchi's test leaves its rows unconnected.
#[derive(Clone, Copy, Debug)]
struct Unwired;

impl<F: PrimeField> TypedGadget<F> for Unwired {
    type Input = (Array<Scalar, 3>, Array<Scalar, CRUMBS_PER_ROW>);
    type Output = Array<Scalar, 3>;
    const NAME: &'static str = "unwired-endomul-scalar";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (start, crumbs): Of<Self::Input, E::Var>,
    ) -> Of<Array<Scalar, 3>, E::Var> {
        let columns = [column::N0, column::A0, column::B0];
        let start: [E::Var; 3] =
            core::array::from_fn(|i| env.advice(Cell::new(columns[i], 0), [&start[i]], |[v]| v));
        env.call(&EndoMulScalarGate, (start, crumbs))
    }
}

/// Kimchi's circuit: each challenge's crumbs (most significant first) on
/// its own rows, from `n = 0, a = b = 2`.
#[derive(Clone, Copy, Debug)]
struct Challenges;

impl<F: PrimeField> TypedGadget<F> for Challenges {
    type Input = Array<Scalar, { SCALARS * CRUMBS }>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-endomul-scalar-test";
    const ROWS: usize = SCALARS * ROWS_PER_SCALAR;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        crumbs: Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        for scalar in crumbs.chunks(CRUMBS) {
            let mut state = [
                env.constant(F::ZERO),
                env.constant(F::from_u64(2)),
                env.constant(F::from_u64(2)),
            ];
            for row in scalar.chunks(CRUMBS_PER_ROW) {
                let eight: [E::Var; CRUMBS_PER_ROW] = core::array::from_fn(|k| row[k].clone());
                state = env.call(&Unwired, (state, eight));
            }
        }
    }
}

/// The crumbs of the low `BITS` bits of `x`, most significant first.
fn crumbs(x: Fp) -> impl Iterator<Item = Fp> {
    let bit = move |i: usize| (x.limb(i / 64) >> (i % 64)) & 1;
    (0..CRUMBS)
        .rev()
        .map(move |k| Fp::from_u64(2 * bit(2 * k + 1) + bit(2 * k)))
}

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
    gate_tests::with_stack(endomul_scalar_test_body);
}

fn endomul_scalar_test_body() {
    let case = case("endomul_scalar_test");
    let coeff: Fp = case.elements("coeff")[0];
    assert_eq!(coeff, Vesta::lambda(), "Kimchi's endo_r is Vesta's lambda");
    let xs: Vec<Fp> = case.elements("x");
    let results: Vec<Fp> = case.elements("result");
    assert_eq!(xs.len(), SCALARS);
    let n = <Challenges as TypedGadget<Fp>>::ROWS;
    let gates = kimchi_rows(&Challenges, n);
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let input: Vec<Fp> = xs.iter().flat_map(|x| crumbs(*x)).collect();
    let input: [Fp; SCALARS * CRUMBS] = input.try_into().unwrap();
    let witness = kimchi_witness(&Challenges, &input, n);
    for (k, (x, expected)) in xs.iter().zip(&results).enumerate() {
        let last = (k + 1) * ROWS_PER_SCALAR - 1;
        let (a, b) = (witness[column::A8][last], witness[column::B8][last]);
        let result = a * coeff + b;
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
