//! Kimchi's `kimchi/src/tests/foreign_field_mul.rs`, test by test, with the
//! same inputs and outputs. Each `run_test`'s circuit is one typed gadget:
//! our ForeignFieldMul gate and its Zero row, and, with the external checks,
//! Kimchi's layout of them (the compact range check of the remainder, the
//! high bounds as Generic rows, the multi-range checks of the quotient, of
//! the gate's bound cells, of the inputs and of the bounds). Its rows (the
//! symbolic run) equal Kimchi's up to the copy partition, its witness (the
//! numeric run) is Kimchi's, and every `verify_witness` and proof result is
//! Kimchi's, in the order Kimchi's run made them (`gate_tests/`). The test
//! text is Kimchi's, on our types, and the random draws are Kimchi's,
//! replayed.

mod foreign_field;
mod gate_tests;
mod kimchi_runs;

use std::cell::RefCell;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, GateType},
    gates::KimchiGates,
};
use arrabbiata_kimchi_gadgets::{
    foreign_field::{Limbs, field_of, two_pow},
    foreign_field_mul::ForeignFieldMulGate,
    generic::{GenericPair, GenericSpec},
    range_check::{CompactMultiRangeCheck, MultiRangeCheck, range_check_table},
    zero::Zero as ZeroRow,
};
use arrabbiata_pasta::{Fp, Fq};
use foreign_field::*;
use gate_tests::{Case, debug};

type PallasField = Fp;
type VestaField = Fq;

/// secp256k1's base field modulus, big-endian.
const FIELD_SIZE: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F,
];

// ---------------------------------------------------------------------
// Kimchi's run: the case of the current test and where each of its
// streams (inputs, rows, witnesses, verdicts, proofs, draws) is.

#[derive(Default)]
struct Run {
    case: Option<Case>,
    run: usize,
    step: usize,
    prove: usize,
    draw: usize,
    witness: usize,
}

thread_local! {
    static RUN: RefCell<Run> = RefCell::new(Run::default());
}

/// The test `name` starts: Kimchi's run of it.
fn test(name: &str) {
    RUN.with(|r| {
        *r.borrow_mut() = Run {
            case: Some(gate_tests::case(name)),
            ..Run::default()
        }
    });
}

/// `f` of the current case and its cursors.
fn run<R>(f: impl FnOnce(&mut Run) -> R) -> R {
    RUN.with(|r| f(&mut r.borrow_mut()))
}

/// The `k`-th input line of `key`.
fn nth(run: &Run, key: &str, k: usize) -> Vec<String> {
    let case = run.case.as_ref().unwrap();
    let all: Vec<&Vec<String>> = case
        .inputs
        .iter()
        .filter(|(l, _)| l == key)
        .map(|(_, v)| v)
        .collect();
    all.get(k)
        .unwrap_or_else(|| panic!("no {key} number {k}"))
        .to_vec()
}

/// The verdict `label` of the current case.
fn verdict(label: &str) -> String {
    run(|r| r.case.as_ref().unwrap().verdict(label).to_string())
}

/// The native fields: Kimchi's gates over each, and its test framework's
/// proof on the curve whose scalar field it is.
trait Native: PrimeField + arrabbiata_field::TwoAdicField {
    fn gates() -> KimchiGates<Self>;
    fn prove(gates: Vec<CircuitGate<Self>>, witness: [Vec<Self>; COLUMNS]) -> Result<(), String>;
}

impl Native for Fp {
    fn gates() -> KimchiGates<Fp> {
        gate_tests::vesta_gates()
    }
    fn prove(gates: Vec<CircuitGate<Fp>>, witness: [Vec<Fp>; COLUMNS]) -> Result<(), String> {
        gate_tests::prove_and_verify(gates, witness, vec![])
    }
}

impl Native for Fq {
    fn gates() -> KimchiGates<Fq> {
        gate_tests::pallas_gates()
    }
    fn prove(gates: Vec<CircuitGate<Fq>>, witness: [Vec<Fq>; COLUMNS]) -> Result<(), String> {
        gate_tests::prove_and_verify_pallas(gates, witness, vec![])
    }
}

/// Kimchi's `CircuitGateError` as these tests read it: our gates' errors,
/// and the two a failed proof is mapped to.
#[derive(Clone, Debug, PartialEq, Eq)]
enum CircuitGateError {
    Constraint(GateType, usize),
    InvalidLookupConstraint(GateType),
    InvalidConstraint(GateType),
    Other(arrabbiata_kimchi::constraints::CircuitGateError),
}

impl From<arrabbiata_kimchi::constraints::CircuitGateError> for CircuitGateError {
    fn from(e: arrabbiata_kimchi::constraints::CircuitGateError) -> Self {
        match e {
            arrabbiata_kimchi::constraints::CircuitGateError::Constraint(typ, n) => {
                Self::Constraint(typ, n)
            }
            other => Self::Other(other),
        }
    }
}

/// Our `verify_witness` on `row`, which must give Kimchi's next verdict (up
/// to the copy cycle).
fn verify_rec<F: Native>(
    gates: &[CircuitGate<F>],
    row: usize,
    witness: &[Vec<F>; COLUMNS],
) -> Result<(), CircuitGateError> {
    let ours = gates[row].verify_witness(row, witness, &F::gates());
    let k = run(|r| {
        r.step += 1;
        r.step - 1
    });
    let label = format!("step{k}");
    gate_tests::assert_verdict(&debug(&ours), &verdict(&label), gates, &label);
    ours.map_err(CircuitGateError::from)
}

/// Kimchi's `TestFramework::prove_and_verify`, whose result must be Kimchi's
/// next.
fn prove_rec<F: Native>(
    gates: &[CircuitGate<F>],
    witness: &[Vec<F>; COLUMNS],
) -> Result<(), String> {
    let result = F::prove(gates.to_vec(), witness.clone());
    let k = run(|r| {
        r.prove += 1;
        r.prove - 1
    });
    assert_eq!(debug(&result), verdict(&format!("prove{k}")), "proof {k}");
    result
}

/// Our witness is Kimchi's next one: by rows, or by digest when large.
fn kimchi_witness<F: PrimeField>(witness: &[Vec<F>; COLUMNS]) {
    let ours = gate_tests::rows(witness);
    if witness[0].len() <= 64 {
        let theirs = run(|r| {
            r.witness += 1;
            r.case.as_ref().unwrap().witnesses[r.witness - 1].clone()
        });
        assert_eq!(ours, theirs);
    } else {
        unreachable!("a multiplication has at most 28 rows")
    }
}

// ---------------------------------------------------------------------
// Kimchi's draws, replayed.

/// `make_test_rng(None)`: the draws come from Kimchi's run.
struct Draws;

fn next_rng() -> Draws {
    Draws
}

fn draw(_: &mut Draws) -> BigUint {
    run(|r| {
        r.draw += 1;
        BigUint::from_hex(&nth(r, "draw", r.draw - 1)[0])
    })
}

/// `gen_biguint_below`, Kimchi's draw.
fn draw_below(rng: &mut Draws, bound: &BigUint) -> BigUint {
    let x = draw(rng);
    assert!(x < *bound);
    x
}

/// `gen_biguint_range`, Kimchi's draw.
fn draw_range(rng: &mut Draws, low: &BigUint, high: &BigUint) -> BigUint {
    let x = draw(rng);
    assert!(*low <= x && x < *high);
    x
}

// ---------------------------------------------------------------------
// The circuit: one gadget.

/// The rows of a multiplication with the external checks.
const EXTERNAL_ROWS: usize = 28;

/// Kimchi's multiplication, as `run_test` lays it out: input `(left,
/// right)` in columns 0-5 of row 0, our ForeignFieldMul gate and its Zero
/// row; with `external`, Kimchi's example layout of the checks it leaves to
/// the circuit designer:
///
/// - rows 2-5, the compact range check of the remainder `(r01, r2)`;
/// - row 6, the high bound `r2 + 2^88 - m2 - 1` (a double Generic row,
///   its second half a dummy on zero);
/// - rows 7-10 and 11-14, the range checks of the quotient and of the
///   gate's `(q'2, p10, p110)`;
/// - row 15, the high bounds of the inputs' top limbs;
/// - rows 16-19 and 20-23, the range checks of the inputs;
/// - rows 24-27, the range check of the three high bounds.
///
/// Each check copies the cells it reads.
#[derive(Clone, Debug)]
struct Multiplication<F> {
    modulus: Limbs,
    table: Rom<F>,
    external: bool,
}

impl<F: PrimeField> Multiplication<F> {
    fn new(modulus: &BigUint, external: bool) -> Self {
        let [lo, mi, hi] = modulus.0.limbs();
        Self {
            modulus: [lo, mi, hi, 0],
            table: range_check_table(),
            external,
        }
    }

    /// The high bounds `x + 2^88 - m2 - 1` of `pair` on row `row`, as one
    /// double Generic row: Kimchi's `extend_high_bounds`, a dummy zero in
    /// the second half when there is one bound.
    fn high_bounds<E: Env<F>>(
        &self,
        env: &mut E,
        row: usize,
        pair: [Option<E::Var>; 2],
    ) -> [E::Var; 2] {
        let hi = two_pow::<F>(88) - field_of::<F>(self.modulus[2]) - F::ONE;
        let zero = env.constant(F::ZERO);
        let [x, y] = pair;
        let mut write = |column: usize| env.advice(Cell::new(column, row), [&zero], |[v]| v);
        let x = x.unwrap_or_else(|| write(0));
        let y = y.unwrap_or_else(|| write(3));
        let right = [write(1), write(4)];
        let plus = GenericPair {
            first: GenericSpec::Plus(hi),
            second: GenericSpec::Plus(hi),
        };
        env.call(&plus, ([x, right[0].clone()], [y, right[1].clone()]))
    }
}

impl<F: PrimeField> TypedGadget<F> for Multiplication<F> {
    type Input = (Array<Scalar, 3>, Array<Scalar, 3>);
    type Output = Unit;
    const NAME: &'static str = "kimchi-ffmul-test";
    // At most: the external checks are the gadget's parameter.
    const ROWS: usize = EXTERNAL_ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (left, right): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let left: [E::Var; 3] =
            core::array::from_fn(|k| env.advice(Cell::new(k, 0), [&left[k]], |[v]| v));
        let right: [E::Var; 3] =
            core::array::from_fn(|k| env.advice(Cell::new(3 + k, 0), [&right[k]], |[v]| v));
        let gate = ForeignFieldMulGate {
            modulus: self.modulus,
            table: self.table.clone(),
        };
        let ([r01, r2], quotient, bounds) = env.call(&gate, (left.clone(), right.clone()));
        env.call(&ZeroRow, ());
        if !self.external {
            return;
        }
        let compact = CompactMultiRangeCheck {
            table: self.table.clone(),
        };
        env.call(&compact, [r01, r2.clone()]);
        let [result_bound, _] = self.high_bounds(env, 6, [Some(r2), None]);
        let check = MultiRangeCheck {
            table: self.table.clone(),
        };
        env.call(&check, quotient);
        env.call(&check, bounds);
        let [left_bound, right_bound] =
            self.high_bounds(env, 15, [Some(left[2].clone()), Some(right[2].clone())]);
        env.call(&check, left);
        env.call(&check, right);
        env.call(&check, [result_bound, left_bound, right_bound]);
    }
}

// ---------------------------------------------------------------------
// Kimchi's tests, as Kimchi writes them.

// The secp256k1 base field modulus
fn secp256k1_modulus() -> BigUint {
    BigUint::from_bytes_be(&FIELD_SIZE)
}

// Maximum value in the secp256k1 base field
fn secp256k1_max() -> BigUint {
    secp256k1_modulus() - BigUint::from(1u32)
}

// Maximum value whose square fits in secp256k1 base field
fn secp256k1_sqrt() -> BigUint {
    secp256k1_max().sqrt()
}

// Maximum value in the pallas base field
fn pallas_max() -> BigUint {
    PallasField::modulus_biguint() - BigUint::from(1u32)
}

// Maximum value whose square fits in the pallas base field
fn pallas_sqrt() -> BigUint {
    pallas_max().sqrt()
}

// Boilerplate for tests: Kimchi's `run_test`, the circuit and witness from
// our gadget.
fn run_test<F: Native>(
    external_gates: bool,
    disable_gates_checks: bool,
    left_input: &BigUint,
    right_input: &BigUint,
    foreign_field_modulus: &BigUint,
    invalidations: Vec<((usize, usize), F)>,
    fixture_name: &'static str,
) -> (Result<(), CircuitGateError>, [Vec<F>; COLUMNS]) {
    let full = !fixture_name.is_empty();
    let (inputs, gates_kimchi) = run(|r| {
        r.run += 1;
        let k = r.run - 1;
        let keys = ["left", "right", "modulus", "external", "disable"];
        let inputs: Vec<Vec<String>> = keys.iter().map(|key| nth(r, key, k)).collect();
        (inputs, r.case.as_ref().unwrap().gates[k].clone())
    });
    assert_eq!(inputs[0], vec![left_input.hex()]);
    assert_eq!(inputs[1], vec![right_input.hex()]);
    assert_eq!(inputs[2], vec![foreign_field_modulus.hex()]);
    assert_eq!(inputs[3], vec![usize::from(external_gates).to_string()]);
    assert_eq!(
        inputs[4],
        vec![usize::from(disable_gates_checks).to_string()]
    );

    let gadget = Multiplication::<F>::new(foreign_field_modulus, external_gates);
    let n = if external_gates { EXTERNAL_ROWS } else { 2 };
    let gates = gate_tests::kimchi_rows(&gadget, n);
    gate_tests::assert_kimchi_rows(&gates, &gates_kimchi);
    let input = (left_input.to_field_limbs(), right_input.to_field_limbs());
    let mut witness = gate_tests::kimchi_witness(&gadget, &input, n);
    kimchi_witness(&witness);
    // Kimchi builds a constraint system; so do we, and use it the same way.
    ConstraintSystem::create(gates.clone()).build().unwrap();

    // Perform witness verification that everything is ok before invalidation (quick checks)
    for row in 0..witness[0].len() {
        let result = verify_rec(&gates, row, &witness);
        if result.is_err() {
            return (result, witness);
        }
    }

    if full {
        // Perform full test that everything is ok before invalidation
        assert_eq!(prove_rec(&gates, &witness), Ok(()));
    }

    if !invalidations.is_empty() {
        for ((row, col), mut value) in invalidations {
            // Invalidate witness
            if witness[col][row] == value {
                // If the invalidation would be the same value choose a different one
                // Don't let it wrap around
                assert_ne!(value, F::zero());
                value = value - F::one();
            }
            witness[col][row] = value;
        }

        if !disable_gates_checks {
            // Check witness verification fails
            for row in 0..witness[0].len() {
                let result = verify_rec(&gates, row, &witness);
                if result.is_err() {
                    return (result, witness);
                }
            }
        }

        // Run test on invalid witness
        if full {
            match prove_rec(&gates, &witness) {
                Err(err_msg) => {
                    if err_msg.starts_with("the lookup failed to find a match in the table") {
                        return (
                            Err(CircuitGateError::InvalidLookupConstraint(
                                GateType::ForeignFieldMul,
                            )),
                            witness,
                        );
                    } else {
                        return (
                            Err(CircuitGateError::InvalidConstraint(
                                GateType::ForeignFieldMul,
                            )),
                            witness,
                        );
                    }
                }
                Ok(()) => return (Ok(()), witness),
            }
        }
    }

    (Ok(()), witness)
}

// Test targeting each custom constraint (positive and negative tests for each)
fn test_custom_constraints<F: Native>(foreign_field_modulus: &BigUint) {
    let rng = &mut next_rng();

    for _ in 0..3 {
        let left_input = draw_range(rng, &BigUint::zero(), foreign_field_modulus);
        let right_input = draw_range(rng, &BigUint::zero(), foreign_field_modulus);

        // Test constraint (C1): invalidate product1_hi_1 is in [0, 2^2)
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((1, 7), F::from_kimchi(4u32))], // Invalidate product1_hi_1
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 1)),
        );

        // Test constraint (C2): invalidate carry0 in [0, 2^2)
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((1, 11), F::from_kimchi(4u32))], // Invalidate carry0
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 2)),
        );

        // Test constraint (C3): invalidate middle intermediate product p1 decomposition
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((0, 6), F::one())], // Invalidate product1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 3)),
        );

        // Test constraint (C4): invalidate carry0
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((1, 11), F::from_kimchi(3u32))], // Invalidate carry0
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 4)),
        );

        // Test constraint (C5): invalid native modulus check but binary modulus checks ok
        //     Triggering constraint C4 is challenging, so this is done with
        //     the test_native_modulus_constraint() test below

        // Test constraint (C6): invalidate carry1_crumb0
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((0, 11), F::from_kimchi(8u32))], // Invalidate carry1_crumb0
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 6)),
        );
        // Test constraint (C7): invalidate carry1_crumb1
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((0, 12), F::from_kimchi(8u32))], // Invalidate carry1_crumb0
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 7)),
        );
        // Test constraint (C8): invalidate carry1_crumb2
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((0, 13), F::from_kimchi(8u32))], // Invalidate carry1_crumb0
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 8)),
        );

        // Test constraint (C9): invalidate carry1_bit
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((0, 14), F::from_kimchi(3u32))], // Invalidate carry1_crumb0
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 9)),
        );

        // Test constraint (C10): invalidate zero check
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((0, 7), F::one())], // Invalidate carry1_0_11
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );

        // Test constraint (C11): invalidate quotient high bound
        let (result, witness) = run_test::<F>(
            false,
            false,
            &left_input,
            &right_input,
            foreign_field_modulus,
            vec![((1, 5), F::one())], // Invalidate quotient_bound
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 11)),
        );
    }
}

// Non-prover version of test_custom_constraints
// Test the multiplication of two zeros.
// This checks that small amounts get packed into limbs
#[test]
fn test_zero_mul() {
    test("test_zero_mul");
    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &BigUint::zero(),
        &BigUint::zero(),
        &secp256k1_modulus(),
        vec![],
        "test_zero_mul",
    );
    assert_eq!(result, Ok(()));

    // Check remainder is zero
    assert_eq!(witness[0][1], PallasField::zero()); // remainder01
    assert_eq!(witness[1][1], PallasField::zero()); // remainder2

    // Check quotient is zero
    assert_eq!(witness[2][1], PallasField::zero());
    assert_eq!(witness[3][1], PallasField::zero());
    assert_eq!(witness[4][1], PallasField::zero());
}
// Test the multiplication of largest foreign element and 1
#[test]
fn test_one_mul() {
    test("test_one_mul");
    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &secp256k1_max(),
        &One::one(),
        &secp256k1_modulus(),
        vec![],
        "test_one_mul",
    );
    assert_eq!(result, Ok(()));

    // Check remainder is secp256k1_max()
    let target = secp256k1_max().to_compact_field_limbs();
    assert_eq!(witness[0][1], target[0]);
    assert_eq!(witness[1][1], target[1]);

    // Check quotient is zero
    assert_eq!(witness[2][1], PallasField::zero());
    assert_eq!(witness[3][1], PallasField::zero());
    assert_eq!(witness[4][1], PallasField::zero());
}
// Test the maximum value m whose square fits in the native field
//    m^2 = q * f + r -> q should be 0 and r should be m^2 < n < f
#[test]
fn test_max_native_square() {
    test("test_max_native_square");
    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &pallas_sqrt(),
        &pallas_sqrt(),
        &secp256k1_modulus(),
        vec![],
        "test_max_native_square",
    );
    assert_eq!(result, Ok(()));

    // Check remainder is the square
    let multiplicand = pallas_sqrt();
    let square = multiplicand.pow(2u32);
    let product = square.to_compact_field_limbs();
    assert_eq!(witness[0][1], product[0]);
    assert_eq!(witness[1][1], product[1]);

    // Check quotient is zero
    assert_eq!(witness[2][1], PallasField::zero());
    assert_eq!(witness[3][1], PallasField::zero());
    assert_eq!(witness[4][1], PallasField::zero());
}
// Test the maximum value g whose square fits in the foreign field
//     g^2 = q * f + r -> q should be 0 and r should be g^2 < f
#[test]
fn test_max_foreign_square() {
    test("test_max_foreign_square");
    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &secp256k1_sqrt(),
        &secp256k1_sqrt(),
        &secp256k1_modulus(),
        vec![],
        "test_max_foreign_square",
    );
    assert_eq!(result, Ok(()));

    // Check remainder is the square
    let multiplicand = secp256k1_sqrt();
    let square = multiplicand.pow(2u32);
    let product = square.to_compact_field_limbs();
    assert_eq!(witness[0][1], product[0]);
    assert_eq!(witness[1][1], product[1]);

    // Check quotient is zero
    assert_eq!(witness[2][1], PallasField::zero());
    assert_eq!(witness[3][1], PallasField::zero());
    assert_eq!(witness[4][1], PallasField::zero());
}
// Test squaring the maximum native field elements
//     (n - 1) * (n - 1) = q * f + r
#[test]
fn test_max_native_multiplicands() {
    test("test_max_native_multiplicands");
    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &pallas_max(),
        &pallas_max(),
        &secp256k1_modulus(),
        vec![],
        "test_max_native_multiplicands",
    );
    assert_eq!(result, Ok(()));
    assert_eq!(
        pallas_max() * pallas_max() % secp256k1_modulus(),
        [witness[0][1], witness[1][1]].compose()
    );
}
// Test squaring the maximum foreign field elements
//     (f - 1) * (f - 1) = f^2 - 2f + 1 = f * (f - 2) + 1
#[test]
fn test_max_foreign_multiplicands() {
    test("test_max_foreign_multiplicands");
    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &secp256k1_max(),
        &secp256k1_max(),
        &secp256k1_modulus(),
        vec![],
        "test_max_foreign_multiplicands",
    );
    assert_eq!(result, Ok(()));
    assert_eq!(
        secp256k1_max() * secp256k1_max() % secp256k1_modulus(),
        [witness[0][1], witness[1][1]].compose()
    );
}
// Test with nonzero carry0 bits
#[test]
fn test_nonzero_carry0() {
    test("test_nonzero_carry0");
    let rng = &mut next_rng();

    for _ in 0..4 {
        let mut a = draw_below(rng, &secp256k1_modulus()).to_limbs();
        let mut b = draw_below(rng, &secp256k1_modulus()).to_limbs();

        // Adjust lowest limb to trigger carry bits into carry0
        a[0] = BigUint::two_to_limb() - BigUint::one();
        let a = a.compose();
        assert!(a < secp256k1_modulus());
        b[0] = BigUint::two_to_limb() - BigUint::one();
        let b = b.compose();
        assert!(b < secp256k1_modulus());

        // Valid witness test
        let (result, witness) =
            run_test::<Fp>(true, false, &a, &b, &secp256k1_modulus(), vec![], "");
        assert_eq!(result, Ok(()));
        assert_ne!(witness[11][1], PallasField::zero()); // carry0 is not zero
        assert_eq!(
            &a * &b % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );

        // Invalid carry0 witness test
        let (result, witness) = run_test::<Fp>(
            true,
            false,
            &a,
            &b,
            &secp256k1_modulus(),
            vec![((1, 11), PallasField::zero())], // Invalidate carry0
            "",
        );
        // The constraint (C4) should fail
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 4))
        );
        assert_eq!(
            a * b % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
    }
}
// Test with nonzero carry10 (this targets only the limbs and the first crumb of carry1)
#[test]
fn test_nonzero_carry10() {
    test("test_nonzero_carry10");
    // Max modulus
    // Actually this is larger than max_foreign_field_modulus, but it's fine because
    // we can still hold larger than 2^259-1. This is just for the test to produce nonzero carry10.
    let foreign_field_modulus = BigUint::two().pow(259u32);

    // Maximum quotient
    let q = &foreign_field_modulus - BigUint::one();

    // Compute operands
    let a = &foreign_field_modulus / BigUint::two().pow(5);
    let b = ((&q * &foreign_field_modulus) / &a) % &foreign_field_modulus;

    // Valid witness test
    let (result, witness) = run_test::<Fp>(true, false, &a, &b, &foreign_field_modulus, vec![], "");
    assert_eq!(result, Ok(()));
    let carry10 = witness[7][0]
        + witness[8][0] * PallasField::two_pow(12)
        + witness[9][0] * PallasField::two_pow(24)
        + witness[10][0] * PallasField::two_pow(36)
        + witness[8][1] * PallasField::two_pow(48)
        + witness[9][1] * PallasField::two_pow(60)
        + witness[10][1] * PallasField::two_pow(72)
        + witness[11][0] * PallasField::two_pow(84)
        + witness[12][0] * PallasField::two_pow(86);
    assert_ne!(carry10, PallasField::zero()); // carry10 is definitely not zero
    assert_eq!(
        &a * &b % &foreign_field_modulus,
        [witness[0][1], witness[1][1]].compose()
    );

    // Invalid carry0 witness test
    let (result, witness) = run_test::<Fp>(
        false, // Disable copy constraints so we can catch carry10 custom constraint failure
        false,
        &a,
        &b,
        &foreign_field_modulus,
        vec![((0, 10), PallasField::zero())], // Invalidate carry10
        "",
    );
    // The constraint (C10) should fail
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10))
    );
    assert_eq!(
        a * b % &foreign_field_modulus,
        [witness[0][1], witness[1][1]].compose()
    );
}
// Test with nonzero carry1_hi (this targets only carry1_crumb2 and carry1_bit)
#[test]
fn test_nonzero_carry1_hi() {
    test("test_nonzero_carry1_hi");
    // Big (rubbish) modulus
    let foreign_field_modulus = BigUint::two().pow(259u32) - BigUint::one();

    // Maximum operands
    let a = &foreign_field_modulus - BigUint::one();

    // Valid witness test
    let (result, witness) = run_test::<Fp>(true, false, &a, &a, &foreign_field_modulus, vec![], "");
    assert_eq!(result, Ok(()));
    let carry1_hi = witness[13][0] + witness[14][0] * PallasField::from_kimchi(4u32);
    assert_ne!(carry1_hi, PallasField::zero()); // carry1_hi is definitely not zero
    assert_eq!(
        &a * &a % &foreign_field_modulus,
        [witness[0][1], witness[1][1]].compose()
    );

    // Invalid carry1_hi witness test
    let (result, witness) = run_test::<Fp>(
        false, // Disable copy constraints so we can catch carry1_hi custom constraint failure
        false,
        &a,
        &a,
        &foreign_field_modulus,
        vec![((0, 7), PallasField::zero())], // Invalidate carry1_hi
        "",
    );
    // The constraint (C5) should fail
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10))
    );
    assert_eq!(
        &a * &a % &foreign_field_modulus,
        [witness[0][1], witness[1][1]].compose()
    );
}
// Test with nonzero second bit of carry1_hi
#[test]
fn test_nonzero_second_bit_carry1_hi() {
    test("test_nonzero_second_bit_carry1_hi");
    let rng = &mut next_rng();
    let a = draw_range(
        rng,
        &(secp256k1_modulus() - BigUint::two().pow(64)),
        &secp256k1_modulus(),
    );
    let b = secp256k1_max();

    // Valid witness test
    let (result, witness) = run_test::<Fp>(true, false, &a, &b, &secp256k1_modulus(), vec![], "");
    assert_eq!(result, Ok(()));
    let carry1_hi = witness[13][0] + witness[14][0] * PallasField::from_kimchi(4u32);
    assert_eq!(carry1_hi, PallasField::from_kimchi(2u32)); // carry1_hi is not zero
    assert_eq!(
        &a * &b % secp256k1_modulus(),
        [witness[0][1], witness[1][1]].compose()
    );

    // Invalid carry1_hi witness test
    let (result, witness) = run_test::<Fp>(
        false, // Disable copy constraints so we can catch carry1_hi custom constraint failure
        false,
        &a,
        &b,
        &secp256k1_modulus(),
        vec![((0, 13), PallasField::two())], // Invalidate carry1_hi
        "",
    );
    // The constraint (C10) should fail
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10))
    );
    assert_eq!(
        a * b % secp256k1_modulus(),
        [witness[0][1], witness[1][1]].compose()
    );
}
// Test invalid carry1_hi range bit
#[test]
fn test_invalid_carry1_bit() {
    test("test_invalid_carry1_bit");
    let a = BigUint::zero();
    let b = BigUint::zero();

    // Invalid carry1_hi witness test
    let (result, _) = run_test::<Fp>(
        false, // Disable external checks so we can catch carry1_hi plookup failure
        false,
        &a,
        &b,
        &secp256k1_modulus(),
        vec![
            ((0, 14), PallasField::from_kimchi(2u32)), // carry1_hi > 3 bits (invalid)
        ],
        "test_invalid_carry1_bit",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 9))
    );
}
// This is no longer a lookup test since the 3-bit check is now a crumb + a bit
#[test]
fn test_invalid_wraparound_carry1_hi() {
    test("test_invalid_wraparound_carry1_hi");
    let a = BigUint::zero();
    let b = BigUint::zero();

    // Sanity check wraparound values
    let two_to_9 = PallasField::from_kimchi(2u32).pow(9);
    // Wraparound (exploit) value x s.t. x >= 2^12 AND 2^9 * x < 2^12
    // (credit to querolita for computing the real instances of this value for these test cases!)
    let wraparound_0 = two_to_9.inverse().expect("failed to get inverse");
    for i in 0..8 {
        let wraparound_i = wraparound_0 + PallasField::from_kimchi(i);
        assert!(
            (wraparound_i).to_biguint() >= (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
        );
        assert!(
            (two_to_9 * wraparound_i).to_biguint()
                < (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
        );
        // Wraparound!!!
    }
    // edge case: x - 1 is not a wraparound value
    assert!(
        (wraparound_0 - PallasField::one()).to_biguint()
            >= (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
    );
    assert!(
        (two_to_9 * (wraparound_0 - PallasField::one())).to_biguint()
            >= (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
    );
    // edge case: x + 8 is not a wraparound value
    assert!(
        (wraparound_0 + PallasField::from_kimchi(8)).to_biguint()
            >= (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
    );
    assert!(
        (two_to_9 * (wraparound_0 + PallasField::from_kimchi(8))).to_biguint()
            >= (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
    );

    let value = PallasField::from_kimchi(512u32) // wraparound_0 + 1
        .inverse()
        .expect("failed to get inverse")
        .to_biguint();

    let carry1_bit = value.clone() / BigUint::from(4u32);

    let carry1_crumb2 = value % BigUint::from(4u32);

    // Invalid carry1_hi witness that causes wrap around to something less than 3-bits
    let (result, witness) = run_test::<Fp>(
        false, // Disable external checks so we can catch carry1_hi plookup failure
        false,
        &a,
        &b,
        &secp256k1_modulus(),
        vec![
            // Invalidate carry1_hi by wrapping
            // carry1_hi > 12 bits > 3 bits, but wraps around < 12-bits when scaled by 2^9
            ((0, 14), carry1_bit.into_field()),
            ((0, 13), carry1_crumb2.into_field()),
        ],
        "test_invalid_wraparound_carry1_hi",
    );
    // crumb is "1"
    // bit is "7222870800814035139336520183037050892714122003062567151295331946573649149952"
    // carry1_hi is >> 9 bits but
    // carry1_hi * 2^9 is < 12 bits, actually is equal to one
    let carry1_hi = witness[13][0] + witness[14][0] * PallasField::from_kimchi(4u32);
    assert!((carry1_hi).to_biguint() >= (PallasField::from_kimchi(2u32).pow(9)).to_biguint());
    assert!(
        (two_to_9 * carry1_hi).to_biguint() < (PallasField::from_kimchi(2u32).pow(12)).to_biguint()
    );
    // the bit is not a bit
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 9)),
    );
}
// Test witness with invalid quotient fails verification
#[test]
fn test_zero_mul_invalid_quotient() {
    test("test_zero_mul_invalid_quotient");
    let (result, _) = run_test::<Fp>(
        true,
        false,
        &BigUint::zero(),
        &BigUint::zero(),
        &secp256k1_modulus(),
        vec![((1, 2), PallasField::one())], // Invalidate q0
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 4)),
    );

    let (result, _) = run_test::<Fp>(
        true,
        false,
        &BigUint::zero(),
        &BigUint::zero(),
        &secp256k1_modulus(),
        vec![((1, 3), PallasField::one())], // Invalidate q1
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 3)),
    );

    let (result, _) = run_test::<Fp>(
        true,
        false,
        &BigUint::zero(),
        &BigUint::zero(),
        &secp256k1_modulus(),
        vec![((1, 4), PallasField::one())], // Invalidate q2
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 5))
    );

    let (result, _) = run_test::<Fp>(
        true,
        false,
        &secp256k1_sqrt(),
        &secp256k1_sqrt(),
        &secp256k1_modulus(),
        vec![((1, 2), PallasField::one())], // Invalidate q0
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 4))
    );

    let (result, _) = run_test::<Fp>(
        true,
        false,
        &secp256k1_sqrt(),
        &secp256k1_sqrt(),
        &secp256k1_modulus(),
        vec![((1, 3), PallasField::one())], // Invalidate q1
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 3))
    );

    let (result, _) = run_test::<Fp>(
        true,
        false,
        &secp256k1_sqrt(),
        &secp256k1_sqrt(),
        &secp256k1_modulus(),
        vec![((1, 4), PallasField::one())], // Invalidate q2
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 5))
    );
}
// Test witness with invalid remainder fails
#[test]
fn test_mul_invalid_remainder() {
    test("test_mul_invalid_remainder");
    let (result, _) = run_test::<Fp>(
        false,
        false,
        &secp256k1_sqrt(),
        &secp256k1_sqrt(),
        &secp256k1_modulus(),
        vec![((1, 0), PallasField::zero())], // Invalidate r01
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 4))
    );

    let (result, _) = run_test::<Fp>(
        false,
        false,
        &secp256k1_sqrt(),
        &secp256k1_sqrt(),
        &secp256k1_modulus(),
        vec![((1, 1), PallasField::one())], // Invalidate r2
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 5))
    );
}
// Test multiplying some random values and invalidating carry1_lo
#[test]
fn test_random_multiplicands_carry1_lo() {
    test("test_random_multiplicands_carry1_lo");
    let rng = &mut next_rng();

    for _ in 0..10 {
        let left_input = draw_range(rng, &BigUint::zero(), &secp256k1_max());
        let right_input = draw_range(rng, &BigUint::zero(), &secp256k1_max());

        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((0, 7), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((0, 8), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((0, 9), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((0, 10), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((1, 8), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((1, 9), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((1, 10), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((0, 11), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
        let (result, witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![((0, 12), PallasField::one())], // Invalidate carry1_lo
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(
            result,
            Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 10)),
        );
    }
}
// Test multiplying some random values with secp256k1 foreign modulus
#[test]
fn test_random_multiplicands_valid() {
    test("test_random_multiplicands_valid");
    let rng = &mut next_rng();

    for _ in 0..10 {
        let left_input = draw_range(rng, &BigUint::zero(), &secp256k1_max());
        let right_input = draw_range(rng, &BigUint::zero(), &secp256k1_max());

        let (result, witness) = run_test::<Fp>(
            true,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![],
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % secp256k1_modulus(),
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(result, Ok(()),);
    }
}
// Test multiplying some random values with foreign modulus smaller than native modulus
#[test]
fn test_smaller_foreign_field_modulus() {
    test("test_smaller_foreign_field_modulus");
    let foreign_field_modulus = BigUint::two().pow(252u32) - BigUint::one();

    let rng = &mut next_rng();

    for _ in 0..10 {
        let left_input = draw_range(rng, &BigUint::zero(), &foreign_field_modulus);
        let right_input = draw_range(rng, &BigUint::zero(), &foreign_field_modulus);

        let (result, witness) = run_test::<Fp>(
            true,
            false,
            &left_input,
            &right_input,
            &foreign_field_modulus,
            vec![],
            "",
        );
        assert_eq!(
            (&left_input * &right_input) % &foreign_field_modulus,
            [witness[0][1], witness[1][1]].compose()
        );
        assert_eq!(result, Ok(()),);
    }
}
// Tests targeting each custom constraint with secp256k1 (foreign field modulus)
// on Vesta (native field modulus)
#[test]
fn test_custom_constraints_secp256k1_on_vesta() {
    test("test_custom_constraints_secp256k1_on_vesta");
    test_custom_constraints::<Fp>(&secp256k1_modulus());
}
// Tests targeting each custom constraint with secp256k1 (foreign field modulus)
// on Pallas (native field modulus)
#[test]
fn test_custom_constraints_secp256k1_on_pallas() {
    test("test_custom_constraints_secp256k1_on_pallas");
    test_custom_constraints::<Fq>(&secp256k1_modulus());
}
// Tests targeting each custom constraint with Vesta (foreign field modulus) on Pallas (native field modulus)
#[test]
fn test_custom_constraints_vesta_on_pallas() {
    test("test_custom_constraints_vesta_on_pallas");
    test_custom_constraints::<Fq>(&VestaField::modulus_biguint());
}
// Tests targeting each custom constraint with Pallas (foreign field modulus) on Vesta (native field modulus)
#[test]
fn test_custom_constraints_pallas_on_vesta() {
    test("test_custom_constraints_pallas_on_vesta");
    test_custom_constraints::<Fp>(&PallasField::modulus_biguint());
}
// Tests targeting each custom constraint with Vesta (foreign field modulus) on Vesta (native field modulus)
#[test]
fn test_custom_constraints_vesta_on_vesta() {
    test("test_custom_constraints_vesta_on_vesta");
    test_custom_constraints::<Fp>(&VestaField::modulus_biguint());
}
// Tests targeting each custom constraint with Pallas (foreign field modulus) on Pallas (native field modulus)
#[test]
fn test_custom_constraints_pallas_on_pallas() {
    test("test_custom_constraints_pallas_on_pallas");
    test_custom_constraints::<Fq>(&PallasField::modulus_biguint());
}
// Tests targeting each custom constraint (foreign modulus smaller than native vesta)
#[test]
fn test_custom_constraints_small_foreign_field_modulus_on_vesta() {
    test("test_custom_constraints_small_foreign_field_modulus_on_vesta");
    test_custom_constraints::<Fp>(&(BigUint::two().pow(252u32) - BigUint::one()));
}
// Tests targeting each custom constraint (foreign modulus smaller than native pallas)
#[test]
fn test_custom_constraints_small_foreign_field_modulus_on_pallas() {
    test("test_custom_constraints_small_foreign_field_modulus_on_pallas");
    test_custom_constraints::<Fq>(&(BigUint::two().pow(252u32) - BigUint::one()));
}
#[test]
fn test_native_modulus_constraint() {
    test("test_native_modulus_constraint");
    let rng = &mut next_rng();
    let left_input = draw_range(
        rng,
        &(secp256k1_modulus() - BigUint::two().pow(96)),
        &secp256k1_modulus(),
    );
    let right_input = draw_range(
        rng,
        &(secp256k1_modulus() - BigUint::two().pow(96)),
        &secp256k1_modulus(),
    );

    let invalidations = vec![
        // Targeted attack on constraint 1. Make carry1_hi to be 8
        ((0, 13), PallasField::zero()),
        ((0, 14), PallasField::from_kimchi(2u32)),
        (
            (1, 1),
            PallasField::from_bytes(&[
                89, 18, 0, 0, 237, 48, 45, 153, 27, 249, 76, 9, 252, 152, 70, 34, 0, 0, 0, 0, 0, 0,
                249, 255, 255, 255, 255, 255, 255, 255, 255, 63,
            ])
            .unwrap(),
        ),
    ];

    let (result, _) = run_test::<Fp>(
        false,
        false,
        &left_input,
        &right_input,
        &secp256k1_modulus(),
        invalidations,
        "",
    );
    assert_eq!(
        result,
        Err(CircuitGateError::Constraint(GateType::ForeignFieldMul, 5))
    );
}
#[test]
fn test_gates_max_foreign_field_modulus() {
    test("test_gates_max_foreign_field_modulus");
    // Kimchi's `create_foreign_field_mul`: the gate and its Zero row.
    let gadget = Multiplication::<PallasField>::new(
        &BigUint::max_foreign_field_modulus::<PallasField>(),
        false,
    );
    gate_tests::kimchi_rows(&gadget, 2);
}
#[test]
fn test_witness_max_foreign_field_modulus() {
    test("test_witness_max_foreign_field_modulus");
    // Kimchi's `foreign_field_mul::witness::create`.
    let gadget = Multiplication::<PallasField>::new(
        &BigUint::max_foreign_field_modulus::<PallasField>(),
        false,
    );
    let zero = [PallasField::zero(); 3];
    gate_tests::kimchi_witness(&gadget, &(zero, zero), 2);
}
// Checks that the high bound check includes when q2 is exactly f2 and not just up to f2-1
#[test]
fn test_q2_exactly_f2() {
    test("test_q2_exactly_f2");
    let left_input = secp256k1_max() - BigUint::from(4u32);
    let right_input = secp256k1_max() - BigUint::from(1u32);

    let (result, witness) = run_test::<Fp>(
        true,
        false,
        &left_input,
        &right_input,
        &secp256k1_modulus(),
        vec![],
        "",
    );
    assert_eq!(witness[4][1], secp256k1_max().to_field_limbs()[2]); // q2 is f2
    assert_eq!(
        (&left_input * &right_input) % secp256k1_modulus(),
        [witness[0][1], witness[1][1]].compose()
    );
    assert_eq!(result, Ok(()),);
}
#[test]
fn test_carry_plookups() {
    test("test_carry_plookups");
    let left_input = secp256k1_max() - BigUint::from(4u32);
    let right_input = secp256k1_max() - BigUint::from(1u32);
    // Correct execution of this test has the following values in plookup cells
    // carry1_0 = 0xC26
    // carry1_12 = 0xFFF
    // carry1_24 = 0xEFF
    // carry1_36 = 0xFFF
    // product1_1_1 = 0x1
    // carry1_48 = 0xFFF
    // carry1_60 = 0xFFF
    // carry1_72 = 0xFF
    let (result, witness) = run_test::<Fp>(
        false,
        false,
        &left_input,
        &right_input,
        &secp256k1_modulus(),
        vec![],
        "",
    );
    assert_eq!(
        (&left_input * &right_input) % secp256k1_modulus(),
        [witness[0][1], witness[1][1]].compose()
    );
    assert_eq!(result, Ok(()),);

    // Adds 1 bit to carry1_36 (obtaining 0x1FFF) and removing 1 from carry1_48 (obtaining 0xFFE)
    // only works with a prover
    {
        let (result, _witness) = run_test::<Fp>(
            false,
            false,
            &left_input,
            &right_input,
            &secp256k1_modulus(),
            vec![
                ((0, 10), PallasField::from_kimchi(0x1FFFu32)),
                ((1, 8), PallasField::from_kimchi(0xFFEu32)),
            ],
            "test_carry_plookups",
        );
        assert_eq!(
            result,
            Err(CircuitGateError::InvalidLookupConstraint(
                GateType::ForeignFieldMul
            ))
        );
    }
}
