//! Kimchi's `kimchi/src/tests/rot.rs`, test by test, with the same inputs
//! and outputs. The circuits are typed gadgets (a public zero, then our
//! rotation gates, the four high limbs of the range checks copied to the
//! zero): their rows are the symbolic run, their witness the numeric run,
//! both checked against Kimchi's run of the same test (`gate_tests/`), with
//! every verdict.

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, Scalar, TypedGadget};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, Wire},
    gates::Gates,
};
use arrabbiata_kimchi_gadgets::{
    generic::PublicInput,
    range_check::{RangeCheck0Gate, range_check_table},
    rot::{Rot64Gate, RotMode, WORD_BITS},
};
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{
    case, debug, kimchi_rows, kimchi_witness, pallas_gates, prove_and_verify,
    prove_and_verify_pallas, rows, vesta_gates,
};

/// The left offset of a rotation by `rot` to `side`.
const fn left(rot: u32, side: RotMode) -> u32 {
    match side {
        RotMode::Left => rot,
        RotMode::Right => WORD_BITS - rot,
    }
}

/// A rotation's three rows: Rot64 of the word, the RangeCheck0 rows of
/// `shifted` and of `excess`, their four high limbs copied to the zero.
/// Input `(zero, word)`, two cells.
#[derive(Clone, Debug)]
struct Rotation<F> {
    rot: u32,
    table: Rom<F>,
}

impl<F: PrimeField> TypedGadget<F> for Rotation<F> {
    type Input = (Scalar, Scalar);
    type Output = Scalar;
    const NAME: &'static str = "rotation";
    const ROWS: usize = 3;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (zero, word): Of<Self::Input, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let gate = Rot64Gate {
            table: self.table.clone(),
            rot: self.rot,
        };
        let [rotated, excess, shifted] = env.call(&gate, word);
        let check = RangeCheck0Gate {
            table: self.table.clone(),
            compact: false,
        };
        let [s0, s1] = env.call(&check, shifted);
        let [e0, e1] = env.call(&check, excess);
        for high in [s0, s1, e0, e1] {
            env.connect(&zero, &high);
        }
        rotated
    }
}

/// The zero as a public input on row 0, then the rotation of the word
/// (placed in column 0 of row 1): Kimchi's `create_rot_gadget` and
/// `create_rot_witness`. Input `(zero, word)`.
#[derive(Clone, Debug)]
struct ZeroThenRotation<F>(Rotation<F>);

impl<F: PrimeField> TypedGadget<F> for ZeroThenRotation<F> {
    type Input = (Scalar, Scalar);
    type Output = Scalar;
    const NAME: &'static str = "zero-then-rotation";
    const ROWS: usize = 4;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (zero, word): Of<Self::Input, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let zero = env.call(&PublicInput, zero);
        let word = env.advice(Cell::new(0, 1), [&word], |[v]| v);
        env.call(&self.0, (zero, word))
    }
}

/// The rotation by `rot` to `side`.
fn rotation<F: PrimeField>(rot: u32, side: RotMode) -> Rotation<F> {
    Rotation {
        rot: left(rot, side),
        table: range_check_table(),
    }
}

/// Kimchi's `create_rot_gadget`: the rows.
fn gadget<F: PrimeField>(rot: u32, side: RotMode) -> Vec<CircuitGate<F>> {
    kimchi_rows(&ZeroThenRotation(rotation::<F>(rot, side)), 4)
}

/// Kimchi's `create_rot_witness`: the witness.
fn rot_witness<F: PrimeField>(word: u64, rot: u32, side: RotMode) -> [Vec<F>; COLUMNS] {
    let gadget = ZeroThenRotation(rotation::<F>(rot, side));
    kimchi_witness(&gadget, &(F::ZERO, F::from_u64(word)), 4)
}

/// The word and the zero as public inputs on rows 0 and 1, then the
/// rotation of the word, copied from its public cell (Kimchi's
/// `test_rot_finalization`). Input `(word, zero)`.
#[derive(Clone, Debug)]
struct PublicThenRotation<F>(Rotation<F>);

impl<F: PrimeField> TypedGadget<F> for PublicThenRotation<F> {
    type Input = (Scalar, Scalar);
    type Output = Scalar;
    const NAME: &'static str = "public-then-rotation";
    const ROWS: usize = 5;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (word, zero): Of<Self::Input, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let word = env.call(&PublicInput, word);
        let zero = env.call(&PublicInput, zero);
        env.call(&self.0, (zero, word))
    }
}

/// The rotations of Keccak's state with a non-zero offset.
const ROTATIONS: usize = 24;

/// The zero as a public input on row 0, then every rotation of Keccak's
/// state, each word in column 0 of its first row (Kimchi's
/// `test_keccak_table`). Input `(zero, words)`.
#[derive(Clone, Debug)]
struct Keccak<F> {
    rotations: Vec<Rotation<F>>,
}

impl<F: PrimeField> TypedGadget<F> for Keccak<F> {
    type Input = (Scalar, Array<Scalar, ROTATIONS>);
    type Output = Array<Scalar, ROTATIONS>;
    const NAME: &'static str = "keccak-rotations";
    const ROWS: usize = 1 + 3 * ROTATIONS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (zero, words): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let zero = env.call(&PublicInput, zero);
        core::array::from_fn(|k| {
            let word = env.advice(Cell::new(0, 1 + 3 * k), [&words[k]], |[v]| v);
            env.call(&self.rotations[k], (zero.clone(), word))
        })
    }
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
        let label = format!("step{step}");
        gate_tests::assert_verdict(&debug(&ours), case.verdict(&label), &cs.gates, &label);
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
    let input = 0xDC81_1727_DAF2_2EC1_u64;
    let gadget = PublicThenRotation(rotation::<Fp>(rot, RotMode::Left));
    let gates = kimchi_rows(&gadget, 5);
    let witness = kimchi_witness(&gadget, &(Fp::from_u64(input), Fp::ZERO), 5);
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
    let mut offsets = Vec::new();
    let mut words = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            if off(y, x) != 0 {
                offsets.push(off(y, x));
                words.push(Fp::from_u64(state(x, y)));
            }
        }
    }
    let table = Keccak {
        rotations: offsets
            .iter()
            .map(|rot| rotation::<Fp>(*rot, RotMode::Left))
            .collect(),
    };
    let input: [Fp; ROTATIONS] = words.try_into().unwrap();
    let rows_used = 1 + 3 * ROTATIONS;
    let cs = ConstraintSystem::create(kimchi_rows(&table, rows_used))
        .build()
        .unwrap();
    let witness = kimchi_witness(&table, &(Fp::ZERO, input), rows_used);
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
/// proves (Kimchi's verdicts, from its run). Our rows are derived from the
/// rotation gadget, which puts the four high limbs on one cycle with the
/// zero cell: the same witness breaks our copy constraints and does not
/// prove.
#[test]
fn connect_64bit_forgery() {
    let case = case("connect_64bit_forgery");
    let (word, rot) = (case.number("word") as u64, case.number("rot") as u32);
    // Kimchi's wiring leaves (2, 1), a high limb of the first range check,
    // wired to itself.
    assert_eq!(case.input("wires2")[1], "2:1");
    // Ours joins it to the zero cell (0, 0).
    let gates = gadget::<Fp>(rot, RotMode::Left);
    let start = Wire { row: 2, col: 1 };
    let mut cycle = vec![start];
    let mut at = gates[2].wires[1];
    while at != start {
        cycle.push(at);
        at = gates[at.row].wires[at.col];
    }
    assert!(cycle.contains(&Wire { row: 0, col: 0 }), "{cycle:?}");
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        case.witnesses[0]
            .iter()
            .map(|row| gate_tests::field(&row[c]))
            .collect()
    });
    let claimed = witness[1][1];
    assert_ne!(claimed, Fp::from_u64(rotated(word, rot, RotMode::Left)));
    assert_ne!(
        witness[1][2],
        Fp::ZERO,
        "the forgery puts a value in (2, 1)"
    );
    let cs = ConstraintSystem::create(gates.clone())
        .public(1)
        .build()
        .unwrap();
    let kimchi = vesta_gates();
    // Kimchi accepts the forged witness row by row and proves it; we refuse
    // it at a copy constraint of the zero's cycle (the public zero and the
    // high limbs of both range checks, which the forgery moved), and refuse
    // to prove it. Which row of the cycle reports the break depends on the
    // order inside the cycle.
    let mut refused = 0;
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = debug(&gate.verify_witness(row, &witness, &kimchi));
        let theirs = case.verdict(&format!("row{row}"));
        if [0, 2, 3].contains(&row) {
            assert_eq!(theirs, "Ok(())");
            let broken = ours.starts_with("Err(CopyConstraint");
            assert!(broken || ours == "Ok(())", "row {row}: {ours}");
            refused += usize::from(broken);
        } else {
            gate_tests::assert_verdict(&ours, theirs, &cs.gates, &format!("row {row}"));
        }
    }
    assert!(refused > 0, "no row of the zero's cycle refuses");
    assert_eq!(case.verdict("prove"), "Ok(())");
    let result = prove_and_verify(gates, witness, vec![Fp::ZERO]);
    assert!(result.is_err(), "{result:?}");
}
