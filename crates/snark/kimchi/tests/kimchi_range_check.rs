//! Kimchi's `kimchi/src/tests/range_check.rs`, test by test, with the same
//! inputs and outputs: Kimchi's rows from our constructors, the witnesses
//! from our range-check gadgets, every verdict (the gate checks and the
//! proofs) against Kimchi's run of the same test (`gate_tests/`).

mod gate_tests;

use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, Wire},
    gates::KimchiGates,
    polynomials::{
        generic::GenericGateSpec,
        range_check::{create, create_multi, create_multi_compact_limbs},
    },
};
use arrabbiata_pasta::Fp;
use gate_tests::{Case, case, debug, from_hex_le, prove_and_verify, rows, vesta_gates};

type W = [Vec<Fp>; COLUMNS];

/// Kimchi's `create_test_cs`.
fn create_test_cs(compact: bool) -> ConstraintSystem<Fp> {
    let (_, gates) = if compact {
        CircuitGate::<Fp>::create_compact_multi_range_check(0)
    } else {
        CircuitGate::<Fp>::create_multi_range_check(0)
    };
    ConstraintSystem::create(gates).build().unwrap()
}

/// A test replayed against Kimchi's run: its witnesses and verdicts in
/// order.
struct Run {
    case: Case,
    cs: ConstraintSystem<Fp>,
    gates: KimchiGates<Fp>,
    step: usize,
    witness: usize,
}

impl Run {
    fn new(name: &str, cs: ConstraintSystem<Fp>) -> Self {
        Self {
            case: case(name),
            cs,
            gates: vesta_gates(),
            step: 0,
            witness: 0,
        }
    }

    /// `verify_witness` of row `row`, against Kimchi's next verdict.
    fn check(&mut self, w: &W, row: usize) {
        let ours = self.cs.gates[row].verify_witness(row, w, &self.gates);
        let label = format!("step{}", self.step);
        assert_eq!(debug(&ours), self.case.verdict(&label), "{label}");
        self.step += 1;
    }

    /// Our witness, against Kimchi's next one.
    fn witnessed(&mut self, w: W) -> W {
        assert_eq!(
            rows(&w),
            self.case.witnesses[self.witness],
            "witness {}",
            self.witness
        );
        self.witness += 1;
        w
    }

    /// Kimchi's `create_multi`, by our gadget.
    fn multi(&mut self, v0: Fp, v1: Fp, v2: Fp) -> W {
        self.witnessed(create_multi(v0, v1, v2).unwrap())
    }

    /// TestFramework's proof of `w` on the constraint system's rows.
    fn prove(&self, w: W, label: &str) {
        let result = prove_and_verify(self.cs.gates.clone(), w, vec![]);
        assert_eq!(debug(&result), self.case.verdict(label), "{label}");
    }
}

fn two_pow(k: u64) -> Fp {
    Fp::from_u64(2).pow(k)
}

fn add(w: &mut W, col: usize, row: usize, x: Fp) {
    w[col][row] = w[col][row] + x;
}

#[test]
fn verify_range_check0_zero_valid_witness() {
    let mut run = Run::new(
        "verify_range_check0_zero_valid_witness",
        create_test_cs(false),
    );
    let w: W = core::array::from_fn(|_| vec![Fp::ZERO; 4]);
    run.check(&w, 0);
    run.check(&w, 1);
}

#[test]
fn verify_range_check0_one_invalid_witness() {
    let mut run = Run::new(
        "verify_range_check0_one_invalid_witness",
        create_test_cs(false),
    );
    let w: W = core::array::from_fn(|_| vec![Fp::ONE; 4]);
    run.check(&w, 0);
    run.check(&w, 1);
}

#[test]
fn verify_range_check0_valid_witness() {
    let mut run = Run::new("verify_range_check0_valid_witness", create_test_cs(false));
    let w = run.multi(
        from_hex_le("115655443433221211ffef000000000000000000000000000000000000000000"),
        from_hex_le("eeddcdccbbabaa99898877000000000000000000000000000000000000000000"),
        from_hex_le("7766565544343322121100000000000000000000000000000000000000000000"),
    );
    run.check(&w, 0);
    run.check(&w, 1);
    let w = run.multi(
        from_hex_le("23d406ac800d1af73040dd000000000000000000000000000000000000000000"),
        from_hex_le("a8fe8555371eb021469863000000000000000000000000000000000000000000"),
        from_hex_le("3edff808d8f533be9af500000000000000000000000000000000000000000000"),
    );
    run.check(&w, 0);
    run.check(&w, 1);
}

#[test]
fn verify_range_check0_invalid_witness() {
    let mut run = Run::new("verify_range_check0_invalid_witness", create_test_cs(false));
    let mut w = run.multi(
        from_hex_le("22f6b4e7ecb4488433ade7000000000000000000000000000000000000000000"),
        from_hex_le("e20e9d80333f2fba463ffd000000000000000000000000000000000000000000"),
        from_hex_le("25d28bfd6cdff91ca9bc00000000000000000000000000000000000000000000"),
    );
    add(&mut w, 1, 0, Fp::ONE);
    run.check(&w, 0);
    add(&mut w, 2, 1, Fp::ONE);
    run.check(&w, 1);
    let mut w = run.multi(
        from_hex_le("22cab5e27101eeafd2cbe1000000000000000000000000000000000000000000"),
        from_hex_le("1ab61d31f4e27fe41a318c000000000000000000000000000000000000000000"),
        from_hex_le("449a45cd749f1e091a3000000000000000000000000000000000000000000000"),
    );
    w[8][0] = w[0][0] + Fp::ONE;
    run.check(&w, 0);
    w[9][1] = w[0][1] + Fp::ONE;
    run.check(&w, 1);
}

/// The tests that check one value at a time in one row: `v0` in row 0, `v1`
/// in row 1, `v2` in row 2, the others zero.
fn one_value(name: &str, row: usize, values: &[Fp]) {
    let mut run = Run::new(name, create_test_cs(false));
    let z = Fp::ZERO;
    for v in values {
        let w = match row {
            0 => run.multi(*v, z, z),
            1 => run.multi(z, *v, z),
            _ => run.multi(z, z, *v),
        };
        run.check(&w, row);
    }
}

#[test]
fn verify_range_check0_valid_v0_in_range() {
    let values = [
        two_pow(88) - Fp::ONE,
        two_pow(64),
        Fp::from_u64(42),
        Fp::ONE,
    ];
    one_value("verify_range_check0_valid_v0_in_range", 0, &values);
}

#[test]
fn verify_range_check0_valid_v1_in_range() {
    let values = [
        two_pow(88) - Fp::ONE,
        two_pow(63),
        Fp::from_u64(48),
        Fp::ONE + Fp::ONE,
    ];
    one_value("verify_range_check0_valid_v1_in_range", 1, &values);
}

#[test]
fn verify_range_check0_invalid_v0_not_in_range() {
    let values = [two_pow(88), two_pow(96)];
    one_value("verify_range_check0_invalid_v0_not_in_range", 0, &values);
}

#[test]
fn verify_range_check0_invalid_v1_not_in_range() {
    let values = [two_pow(88), two_pow(96)];
    one_value("verify_range_check0_invalid_v1_not_in_range", 1, &values);
}

#[test]
fn verify_range_check0_test_copy_constraints() {
    let mut run = Run::new(
        "verify_range_check0_test_copy_constraints",
        create_test_cs(false),
    );
    for row in 0..=1 {
        for col in 1..=2 {
            let mut w = run.multi(two_pow(88) - Fp::ONE, two_pow(88) - Fp::ONE, Fp::ZERO);
            run.check(&w, row);
            assert_ne!(w[col][row], Fp::ZERO);
            w[col][row] = Fp::ZERO;
            run.check(&w, row);
        }
    }
}

#[test]
fn verify_range_check0_v0_test_lookups() {
    let mut run = Run::new("verify_range_check0_v0_test_lookups", create_test_cs(false));
    let w = run.multi(two_pow(88) - Fp::ONE, Fp::ZERO, Fp::ZERO);
    run.check(&w, 0);
    for i in 3..=6 {
        let mut w = w.clone();
        add(&mut w, i, 0, Fp::from_u64(1 << 12));
        add(&mut w, i - 1, 0, -Fp::ONE);
        if i == 3 {
            add(&mut w, 4, 3, -Fp::ONE);
        }
        run.prove(w, &format!("prove{i}"));
    }
}

#[test]
fn verify_range_check0_v1_test_lookups() {
    let mut run = Run::new("verify_range_check0_v1_test_lookups", create_test_cs(false));
    let w = run.multi(Fp::ZERO, two_pow(88) - Fp::ONE, Fp::ZERO);
    run.check(&w, 1);
    for i in 3..=6 {
        let mut w = w.clone();
        add(&mut w, i, 1, Fp::from_u64(1 << 12));
        add(&mut w, i - 1, 1, -Fp::ONE);
        if i == 3 {
            add(&mut w, 6, 3, -Fp::ONE);
        }
        run.prove(w, &format!("prove{i}"));
    }
}

#[test]
fn verify_range_check1_zero_valid_witness() {
    let mut run = Run::new(
        "verify_range_check1_zero_valid_witness",
        create_test_cs(false),
    );
    let w: W = core::array::from_fn(|_| vec![Fp::ZERO; 4]);
    run.check(&w, 2);
}

#[test]
fn verify_range_check1_one_invalid_witness() {
    let mut run = Run::new(
        "verify_range_check1_one_invalid_witness",
        create_test_cs(false),
    );
    let w: W = core::array::from_fn(|_| vec![Fp::ONE; 4]);
    run.check(&w, 2);
}

#[test]
fn verify_range_check1_valid_witness() {
    let mut run = Run::new("verify_range_check1_valid_witness", create_test_cs(false));
    let w = run.multi(
        from_hex_le("22cab5e27101eeafd2cbe1000000000000000000000000000000000000000000"),
        from_hex_le("1ab61d31f4e27fe41a318c000000000000000000000000000000000000000000"),
        from_hex_le("449a45cd749f1e091a3000000000000000000000000000000000000000000000"),
    );
    run.check(&w, 2);
    let w = run.multi(
        from_hex_le("0d96f6fc210316c73bcc4d000000000000000000000000000000000000000000"),
        from_hex_le("59c8e7b0ffb3cab6ce8d48000000000000000000000000000000000000000000"),
        from_hex_le("686c10e73930b92f375800000000000000000000000000000000000000000000"),
    );
    run.check(&w, 2);
}

#[test]
fn verify_range_check1_invalid_witness() {
    let mut run = Run::new("verify_range_check1_invalid_witness", create_test_cs(false));
    let mut w = run.multi(
        from_hex_le("2ce2d3ac942f98d59e7e11000000000000000000000000000000000000000000"),
        from_hex_le("52dd43524b95399f5d458d000000000000000000000000000000000000000000"),
        from_hex_le("60ca087b427918fa0e2600000000000000000000000000000000000000000000"),
    );
    w[0][2] = w[7][2];
    run.check(&w, 2);
    let mut w = run.multi(
        from_hex_le("1bd50c94d2dc83d32f01c0000000000000000000000000000000000000000000"),
        from_hex_le("e983d7cd9e28e440930f86000000000000000000000000000000000000000000"),
        from_hex_le("ea226054772cd009d2af00000000000000000000000000000000000000000000"),
    );
    w[13][2] = w[3][2];
    run.check(&w, 2);
}

#[test]
fn verify_range_check1_valid_v2_in_range() {
    let values = [
        two_pow(88) - Fp::ONE,
        two_pow(64),
        Fp::from_u64(42),
        Fp::ONE,
    ];
    one_value("verify_range_check1_valid_v2_in_range", 2, &values);
}

#[test]
fn verify_range_check1_invalid_v2_not_in_range() {
    let values = [two_pow(88), two_pow(96)];
    one_value("verify_range_check1_invalid_v2_not_in_range", 2, &values);
}

#[test]
fn verify_range_check1_test_copy_constraints() {
    let mut run = Run::new(
        "verify_range_check1_test_copy_constraints",
        create_test_cs(false),
    );
    for row in 0..=1 {
        for col in 1..=2 {
            let mut w = run.multi(two_pow(88) - Fp::ONE, two_pow(88) - Fp::ONE, Fp::ZERO);
            run.check(&w, 2);
            assert_ne!(w[col][row], Fp::ZERO);
            w[col][row] = Fp::ZERO;
            run.check(&w, 2);
            run.check(&w, 3);
        }
    }
}

#[test]
fn verify_range_check1_test_curr_row_lookups() {
    let mut run = Run::new(
        "verify_range_check1_test_curr_row_lookups",
        create_test_cs(false),
    );
    let w = run.multi(Fp::ZERO, Fp::ZERO, two_pow(88) - Fp::ONE);
    run.check(&w, 2);
    for i in 3..=6 {
        let mut w = w.clone();
        add(&mut w, i, 2, Fp::from_u64(1 << 12));
        add(&mut w, i - 1, 2, -Fp::ONE);
        run.prove(w, &format!("prove{i}"));
    }
}

#[test]
fn verify_range_check1_test_next_row_lookups() {
    let mut run = Run::new(
        "verify_range_check1_test_next_row_lookups",
        create_test_cs(false),
    );
    let w = run.multi(two_pow(88) - Fp::ONE, two_pow(88) - Fp::ONE, Fp::ZERO);
    run.check(&w, 2);
    for row in 0..=1 {
        for col in 1..=2 {
            let mut w = w.clone();
            add(&mut w, col, row, Fp::from_u64(1 << 12));
            if col > 1 {
                add(&mut w, col - 1, row, -Fp::ONE);
                add(&mut w, col - 1 + 2 * row + 2, 3, -Fp::ONE);
            } else {
                add(&mut w, col - 1, row, two_pow(88));
            }
            add(&mut w, col - 1 + 2 * row + 3, 3, Fp::from_u64(1 << 12));
            run.prove(w, &format!("prove{row}{col}"));
        }
    }
}

#[test]
fn verify_64_bit_range_check() {
    let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(
        Wire::for_row(0),
        GenericGateSpec::Pub,
        None,
    )];
    gates.append(&mut CircuitGate::<Fp>::create_range_check(1).1);
    gates[1].wires[1] = Wire { row: 1, col: 2 };
    gates[1].wires[2] = Wire { row: 0, col: 0 };
    gates[0].wires[0] = Wire { row: 1, col: 1 };
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let mut run = Run::new("verify_64_bit_range_check", cs);
    for v in [two_pow(64) - Fp::ONE, two_pow(64)] {
        let mut w: W = core::array::from_fn(|_| vec![Fp::ZERO]);
        for (column, mut rows) in w.iter_mut().zip(create(v).unwrap()) {
            column.append(&mut rows);
        }
        let w = run.witnessed(w);
        run.check(&w, 1);
    }
}

/// Kimchi's `to_compact_limbs`: `[l0 + 2^88 l1, l2]`.
fn compact(limbs: &[Fp]) -> [Fp; 2] {
    [limbs[0] + two_pow(88) * limbs[1], limbs[2]]
}

#[test]
fn compact_multi_range_check() {
    let mut run = Run::new("compact_multi_range_check", create_test_cs(true));
    let all = run.case.input("limbs").len();
    assert_eq!(all, 3);
    let limbs: Vec<Vec<Fp>> = run
        .case
        .inputs
        .iter()
        .filter(|(k, _)| k == "limbs")
        .map(|(_, v)| v.iter().map(|h| gate_tests::field(h)).collect())
        .collect();
    let compacts: Vec<Vec<Fp>> = run
        .case
        .inputs
        .iter()
        .filter(|(k, _)| k == "compact")
        .map(|(_, v)| v.iter().map(|h| gate_tests::field(h)).collect())
        .collect();
    for (limbs, expected) in limbs.iter().zip(&compacts) {
        let limbs = compact(limbs);
        assert_eq!(limbs.to_vec(), *expected);
        let mut w = run.witnessed(create_multi_compact_limbs(&limbs).unwrap());
        run.check(&w, 1);
        w[1][2] = Fp::ONE;
        run.check(&w, 1);
    }
}

#[test]
fn verify_range_check_valid_proof1() {
    let (_, gates) = CircuitGate::<Fp>::create_multi_range_check(0);
    let cs = ConstraintSystem::create(gates.clone()).build().unwrap();
    let mut run = Run::new("verify_range_check_valid_proof1", cs);
    let w = run.multi(
        from_hex_le("2bc0afaa2f6f50b1d1424b000000000000000000000000000000000000000000"),
        from_hex_le("8b30889f3a39e297ac851a000000000000000000000000000000000000000000"),
        from_hex_le("c1c85ec47635e8edac5600000000000000000000000000000000000000000000"),
    );
    let result = prove_and_verify(gates, w, vec![]);
    assert_eq!(debug(&result), run.case.verdict("prove"));
}

#[test]
fn verify_compact_multi_range_check_proof() {
    let (_, gates) = CircuitGate::<Fp>::create_compact_multi_range_check(0);
    let cs = ConstraintSystem::create(gates.clone()).build().unwrap();
    let mut run = Run::new("verify_compact_multi_range_check_proof", cs);
    let limbs: Vec<Fp> = run.case.elements("limbs");
    let limbs = compact(&limbs);
    assert_eq!(limbs.to_vec(), run.case.elements::<Fp>("compact"));
    let w = run.witnessed(create_multi_compact_limbs(&limbs).unwrap());
    let result = prove_and_verify(gates, w, vec![]);
    assert_eq!(debug(&result), run.case.verdict("prove"));
}
