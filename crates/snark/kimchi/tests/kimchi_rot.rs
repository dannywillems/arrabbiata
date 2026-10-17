//! Kimchi's `kimchi/src/tests/rot.rs`, test by test, with the same inputs
//! and outputs: Kimchi's rows from our constructors, the witness from our
//! rotation gates, every verdict checked against Kimchi's run of the same
//! test (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, Connect, ConstraintSystem, Wire},
    gates::Gates,
    polynomials::{
        generic::GenericGateSpec,
        rot::{RotMode, extend_rot},
    },
};
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{
    case, debug, pallas_gates, prove_and_verify, prove_and_verify_pallas, rows, vesta_gates,
};

/// A public row (the zero the range checks' high limbs are wired to),
/// then the rotation: Kimchi's `create_rot_gadget`.
fn gadget<F: PrimeField>(rot: u32, side: RotMode) -> Vec<CircuitGate<F>> {
    let zero = CircuitGate::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None);
    let mut gates = vec![zero];
    CircuitGate::extend_rot(&mut gates, rot, side, 0);
    gates
}

/// The zero row, then the rotation's witness: Kimchi's
/// `create_rot_witness`.
fn rot_witness<F: PrimeField>(word: u64, rot: u32, side: RotMode) -> [Vec<F>; COLUMNS] {
    let mut witness: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::ZERO]);
    extend_rot(&mut witness, word, rot, side).unwrap();
    witness
}

/// `word` rotated by `rot` to `side`, the value Kimchi's `setup_rot`
/// checks is in cell `(1, 1)`.
fn rotated(word: u64, rot: u32, side: RotMode) -> u64 {
    match side {
        RotMode::Left => word.rotate_left(rot),
        RotMode::Right => word.rotate_right(rot),
    }
}

fn side_of(case: &gate_tests::Case) -> RotMode {
    match case.input("side")[0].as_str() {
        "left" => RotMode::Left,
        _ => RotMode::Right,
    }
}

/// Kimchi's `test_rot`: the witness is Kimchi's, rows 0 to 2 verdicts too.
fn test_rot<F, G>(name: &str, gates: &G)
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let word = case.number("word") as u64;
    let rot = case.number("rot") as u32;
    let side = side_of(&case);
    let cs = ConstraintSystem::create(gadget::<F>(rot, side))
        .build()
        .unwrap();
    let witness = rot_witness::<F>(word, rot, side);
    assert_eq!(rows(&witness), case.witnesses[0]);
    assert_eq!(witness[1][1], F::from_u64(rotated(word, rot, side)));
    for (row, gate) in cs.gates.iter().enumerate().take(3) {
        let ours = gate.verify_witness(row, &witness, gates);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "{name} row {row}"
        );
    }
}

#[test]
fn test_prove_and_verify() {
    let case = gate_tests::case("test_prove_and_verify vesta");
    let (word, rot) = (case.number("word") as u64, case.number("rot") as u32);
    let witness = rot_witness::<Fp>(word, rot, RotMode::Left);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gadget(rot, RotMode::Left), witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
    let case = gate_tests::case("test_prove_and_verify pallas");
    let (word, rot) = (case.number("word") as u64, case.number("rot") as u32);
    let witness = rot_witness::<Fq>(word, rot, RotMode::Left);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify_pallas(gadget(rot, RotMode::Left), witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_rot_random() {
    test_rot::<Fp, _>("test_rot_random vesta left", &vesta_gates());
    test_rot::<Fp, _>("test_rot_random vesta right", &vesta_gates());
    test_rot::<Fq, _>("test_rot_random pallas left", &pallas_gates());
    test_rot::<Fq, _>("test_rot_random pallas right", &pallas_gates());
}

#[test]
fn test_zero_rot() {
    test_rot::<Fq, _>("test_zero_rot", &pallas_gates());
}

#[test]
fn test_large_rot() {
    test_rot::<Fq, _>("test_large_rot", &pallas_gates());
}

#[test]
fn test_bad_constraints() {
    let case = case("test_bad_constraints");
    let (word, rot) = (case.number("word") as u64, case.number("rot") as u32);
    let cs = ConstraintSystem::create(gadget::<Fp>(rot, RotMode::Left))
        .build()
        .unwrap();
    let mut w = rot_witness::<Fp>(word, rot, RotMode::Left);
    assert_eq!(rows(&w), case.witnesses[0]);
    let gates = vesta_gates();
    let mut step = 0;
    let mut check = |w: &[Vec<Fp>; COLUMNS], row: usize| {
        let ours = cs.gates[row].verify_witness(row, w, &gates);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("step{step}")),
            "step {step}"
        );
        step += 1;
    };
    let (one, four) = (Fp::ONE, Fp::from_u64(4));
    let two_64 = Fp::from_u64(2).pow(64);
    let add = |w: &mut [Vec<Fp>; COLUMNS], col: usize, row: usize, x: Fp| {
        w[col][row] = w[col][row] + x;
    };
    // C1-C8, the crumbs.
    for i in 0..8 {
        add(&mut w, i + 7, 1, four);
        check(&w, 1);
        add(&mut w, i + 7, 1, -four);
    }
    // C9, C10, then C11 through the limbs.
    add(&mut w, 0, 1, one);
    check(&w, 1);
    add(&mut w, 0, 1, -one);
    add(&mut w, 1, 1, one);
    check(&w, 1);
    add(&mut w, 1, 1, -one);
    for i in 0..4 {
        add(&mut w, i + 3, 1, one);
        check(&w, 1);
        add(&mut w, i + 3, 1, -one);
    }
    // The excess, on the Rot64 row and on its range check.
    add(&mut w, 2, 1, one);
    add(&mut w, 0, 3, one);
    check(&w, 1);
    check(&w, 3);
    add(&mut w, 2, 1, -one);
    add(&mut w, 0, 3, -one);
    // The shifted word, on the Rot64 row and on its range check.
    add(&mut w, 0, 2, one);
    check(&w, 1);
    check(&w, 2);
    add(&mut w, 0, 2, -one);
    add(&mut w, 0, 2, two_64);
    check(&w, 2);
    add(&mut w, 2, 2, one);
    check(&w, 2);
    add(&mut w, 2, 2, -one);
    add(&mut w, 0, 2, -two_64);
    add(&mut w, 0, 3, two_64);
    add(&mut w, 2, 1, two_64);
    check(&w, 3);
    add(&mut w, 2, 3, one);
    check(&w, 3);
}

#[test]
fn test_rot_finalization() {
    let case = case("test_rot_finalization");
    let rot = 32;
    let mut gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            CircuitGate::create_generic_gadget(Wire::for_row(row), GenericGateSpec::Pub, None)
        })
        .collect();
    CircuitGate::extend_rot(&mut gates, rot, RotMode::Left, 1);
    gates.connect_cell_pair((0, 0), (2, 0));
    let input = 0xDC81_1727_DAF2_2EC1_u64;
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    witness[0][0] = Fp::from_u64(input);
    extend_rot(&mut witness, input, rot, RotMode::Left).unwrap();
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

#[test]
fn test_keccak_table() {
    let case = case("test_keccak_table");
    let off: Vec<u32> = case
        .input("off")
        .iter()
        .map(|o| o.parse().unwrap())
        .collect();
    let state: Vec<u64> = case
        .input("state")
        .iter()
        .map(|w| w.parse().unwrap())
        .collect();
    let (off, state) = (
        |y: usize, x: usize| off[5 * y + x],
        |x: usize, y: usize| state[5 * x + y],
    );
    let zero = CircuitGate::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None);
    let mut gates: Vec<CircuitGate<Fp>> = vec![zero];
    let mut rot_row = 1;
    for y in 0..5 {
        for x in 0..5 {
            let rot = off(y, x);
            if rot == 0 {
                continue;
            }
            let mut rot64 = CircuitGate::create_rot64(rot_row, rot);
            rot_row += rot64.len();
            gates.append(&mut rot64);
            gates.connect_64bit(0, rot_row - 1);
        }
    }
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO]);
    for y in 0..5 {
        for x in 0..5 {
            let rot = off(y, x);
            if rot != 0 {
                extend_rot(&mut witness, state(x, y), rot, RotMode::Left).unwrap();
            }
        }
    }
    assert_eq!(rows(&witness), case.witnesses[0]);
    let kimchi = vesta_gates();
    for (row, gate) in cs.gates.iter().enumerate().take(49) {
        let ours = gate.verify_witness(row, &witness, &kimchi);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "row {row}"
        );
    }
    let mut k = 0;
    for y in 0..5 {
        for x in 0..5 {
            let rot = off(y, x);
            if rot != 0 {
                let expected = Fp::from_u64(state(x, y).rotate_left(rot));
                assert_eq!(witness[1][1 + 3 * k], expected);
                k += 1;
            }
        }
    }
}

/// Not one of Kimchi's tests: Kimchi's `connect_64bit` makes three swaps of
/// successors, and the third joins two cells already on one cycle, which
/// splits it. Column 1 of each range check is left wired to itself, so the
/// top limb of `shifted` and of `excess` is free, and a rotation with a
/// wrong output (`shifted + 2^76`, `excess - 2^12`) satisfies every row and
/// proves. Our constructor is Kimchi's, so the wiring and the verdicts are
/// Kimchi's: this pins them.
#[test]
fn connect_64bit_forgery() {
    let case = case("connect_64bit_forgery");
    let (word, rot) = (case.number("word") as u64, case.number("rot") as u32);
    let gates = gadget::<Fp>(rot, RotMode::Left);
    for (r, gate) in gates.iter().enumerate() {
        let wires: Vec<String> = gate
            .wires
            .iter()
            .map(|w| format!("{}:{}", w.row, w.col))
            .collect();
        assert_eq!(wires, case.input(&format!("wires{r}")), "row {r}");
    }
    assert_eq!(gates[2].wires[1], Wire { row: 2, col: 1 });
    assert_eq!(gates[3].wires[1], Wire { row: 3, col: 1 });
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        case.witnesses[0]
            .iter()
            .map(|row| gate_tests::field(&row[c]))
            .collect()
    });
    let claimed = witness[1][1];
    assert_ne!(claimed, Fp::from_u64(rotated(word, rot, RotMode::Left)));
    let cs = ConstraintSystem::create(gates.clone())
        .public(1)
        .build()
        .unwrap();
    let kimchi = vesta_gates();
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, &witness, &kimchi);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "row {row}"
        );
    }
    let result = prove_and_verify(gates, witness, vec![Fp::ZERO]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
