//! Kimchi's `kimchi/src/tests/foreign_field_add.rs`, test by test, with the
//! same inputs and outputs. Each test's circuit is one typed gadget, the
//! chain of our ForeignFieldAdd gates with its bound and, when Kimchi's test
//! has them, its multi-range checks: its rows (the symbolic run) are
//! compared with Kimchi's up to the copy partition, its witness (the numeric
//! run) with Kimchi's, and every `verify_witness` result with Kimchi's in
//! the order Kimchi's run made them (`gate_tests/`). The test text around
//! the circuits is Kimchi's, on our types (a `BigUint` over our `Nat`, a
//! `ForeignElement` of our field), and the random draws are Kimchi's,
//! replayed.

mod foreign_field;
mod gate_tests;
mod kimchi_runs;

use std::cell::RefCell;

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, CircuitGateError, ConstraintSystem, GateType},
    gates::KimchiGates,
};
pub use arrabbiata_kimchi_gadgets::foreign_field_add::FFOps;
use arrabbiata_kimchi_gadgets::{
    foreign_field::{Limbs, Nat, two_pow},
    foreign_field_add::ForeignFieldAddGate,
    generic::PublicInput,
    range_check::{MultiRangeCheck, range_check_table},
    zero::Zero as ZeroRow,
};
use arrabbiata_pasta::{Fp, Fq};
use blake2::{Blake2b512, Digest};
use foreign_field::*;
use gate_tests::{Case, debug};

type PallasField = Fp;
type VestaField = Fq;

const LIMB_BITS: usize = 88;
const LO: usize = 0;
const MI: usize = 1;
const HI: usize = 2;
const TWO_TO_LIMB: u128 = 1 << 88;

/// secp256k1's base field modulus, big-endian.
const FIELD_SIZE: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F,
];

// ---------------------------------------------------------------------
// Kimchi's run: the case of the current test and where each of its
// streams (verdicts, draws, inputs, witnesses, rows) is.

#[derive(Default)]
struct Run {
    case: Option<Case>,
    step: usize,
    draw: usize,
    op: usize,
    chain: usize,
    witness: usize,
    digest: usize,
    gates: usize,
    restored: usize,
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

/// Our `verify_witness`, the next of Kimchi's verdicts it must equal (up
/// to the copy cycle).
trait VerifyRec<F> {
    fn verify_rec(
        &self,
        row: usize,
        witness: &[Vec<F>; COLUMNS],
        cs: &ConstraintSystem<F>,
        public: &[F],
    ) -> Result<(), CircuitGateError>;
}

impl<F: Native> VerifyRec<F> for CircuitGate<F> {
    fn verify_rec(
        &self,
        row: usize,
        witness: &[Vec<F>; COLUMNS],
        cs: &ConstraintSystem<F>,
        _public: &[F],
    ) -> Result<(), CircuitGateError> {
        let ours = self.verify_witness(row, witness, &F::gates());
        let k = run(|r| {
            r.step += 1;
            r.step - 1
        });
        let label = format!("step{k}");
        gate_tests::assert_verdict(&debug(&ours), &verdict(&label), &cs.gates, &label);
        ours
    }
}

/// The native fields, with Kimchi's gates over each.
trait Native: PrimeField + arrabbiata_field::TwoAdicField {
    fn gates() -> KimchiGates<Self>;
}

impl Native for Fp {
    fn gates() -> KimchiGates<Fp> {
        gate_tests::vesta_gates()
    }
}

impl Native for Fq {
    fn gates() -> KimchiGates<Fq> {
        gate_tests::pallas_gates()
    }
}

/// Our rows are Kimchi's next block of rows (up to the copy partition),
/// except where Kimchi's swaps dropped a copy: in a chain of more than one
/// operation, Kimchi's test wires row `i`'s left input to the range check
/// it already wired as row `i - 1`'s output, and swapping two cells of one
/// 2-cycle leaves both alone. Every such cycle we keep is an intermediate
/// result (a ForeignFieldAdd row after the first, columns 0-2) and the cell
/// of its range check; Kimchi's cycles are all ours.
fn kimchi_rows<F: PrimeField>(gates: &[CircuitGate<F>]) {
    let rows = run(|r| {
        r.gates += 1;
        r.case.as_ref().unwrap().gates[r.gates - 1].clone()
    });
    let (missing, extra) = gate_tests::kimchi_rows_difference(gates, &rows);
    assert!(
        missing.is_empty(),
        "copies Kimchi has and we do not: {missing:?}"
    );
    for cycle in &extra {
        let [(row, col), (rc, rc_col)] = cycle[..] else {
            panic!("an extra cycle of more than two cells: {cycle:?}");
        };
        let chained = row >= 2 && gates[row - 1].typ == GateType::ForeignFieldAdd;
        assert!(chained && col < 3 && rc_col == 0, "{cycle:?}");
        // Column 0 of a multi-range check: RangeCheck0 for the low and
        // middle limbs, RangeCheck1 for the high one.
        let checked = matches!(gates[rc].typ, GateType::RangeCheck0 | GateType::RangeCheck1);
        assert!(checked, "{cycle:?}");
    }
    run(|r| r.restored += extra.len());
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
        let digest = run(|r| {
            r.digest += 1;
            nth(r, "digest", r.digest - 1)
        });
        let lines: Vec<String> = ours.iter().map(|row| row.join(" ")).collect();
        let hex: String = Blake2b512::digest(lines.join("\n").as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(digest, vec![witness[0].len().to_string(), hex]);
    }
}

/// The chain's inputs are Kimchi's next ones.
fn kimchi_inputs(modulus: &BigUint, inputs: &[BigUint], opcodes: &[FFOps]) {
    let (m, xs, os) = run(|r| {
        r.chain += 1;
        let k = r.chain - 1;
        (nth(r, "modulus", k), nth(r, "inputs", k), nth(r, "ops", k))
    });
    assert_eq!(m, vec![modulus.hex()]);
    assert_eq!(xs, inputs.iter().map(BigUint::hex).collect::<Vec<_>>());
    assert_eq!(
        os,
        opcodes
            .iter()
            .map(|o| op_name(*o).to_string())
            .collect::<Vec<_>>()
    );
}

fn op_name(op: FFOps) -> &'static str {
    match op {
        FFOps::Add => "add",
        FFOps::Sub => "sub",
    }
}

/// The proof's verdict is Kimchi's (its rows Kimchi's).
fn prove(gates: Vec<CircuitGate<Fp>>, witness: [Vec<Fp>; COLUMNS], public: Vec<Fp>) {
    kimchi_rows(&gates);
    let result = gate_tests::prove_and_verify(gates, witness, public);
    assert_eq!(debug(&result), verdict("prove"));
}

/// A test Kimchi expects to panic: ours panics as Kimchi's did.
fn panics(f: fn()) {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    assert_eq!(debug(&result.is_err()), verdict("panics"));
}

// ---------------------------------------------------------------------
// Kimchi's draws, replayed.

/// `make_test_rng(None)`: the draws come from Kimchi's run.
struct Draws;

fn next_rng() -> Draws {
    Draws
}

fn random_input(_: &mut Draws, _: BigUint, _: bool) -> Vec<u8> {
    run(|r| {
        r.draw += 1;
        BigUint::from_hex(&nth(r, "draw", r.draw - 1)[0]).to_bytes_be()
    })
}

fn random_operation(_: &mut Draws) -> FFOps {
    run(|r| {
        r.op += 1;
        match nth(r, "op", r.op - 1)[0].as_str() {
            "add" => FFOps::Add,
            _ => FFOps::Sub,
        }
    })
}

// ---------------------------------------------------------------------
// The circuits: one gadget each.

/// The multi-range checks a chain carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Checks {
    /// None (Kimchi's `short_circuit`).
    None,
    /// The bound's (`test_ffadd_no_rc`).
    Bound,
    /// The first left input's, each right input's and result's, and the
    /// bound's (Kimchi's `full_circuit`).
    All,
}

impl Checks {
    /// The multi-range checks of a chain of `n` operations.
    const fn count(self, n: usize) -> usize {
        match self {
            Self::None => 0,
            Self::Bound => 1,
            Self::All => 2 * n + 2,
        }
    }
}

/// The rows of one multi-range check.
const RANGE_CHECK_ROWS: usize = 4;

/// The rows of a chain of `n` operations with `checks`.
const fn chain_rows(n: usize, checks: Checks) -> usize {
    n + 3 + RANGE_CHECK_ROWS * checks.count(n)
}

/// Kimchi's chain of `N` operations modulo `modulus`, as one gadget: the
/// public 1 on row 0; one ForeignFieldAdd row per operation, its right
/// input in columns 3-5, its result on the next row, the next operation's
/// left input; the bound row `result + 2^264 - m` (right input
/// `(0, 0, 2^88)`, overflow copied from the public 1); the Zero row holding
/// the bound; then the multi-range checks of `checks`, each copying the
/// limbs it checks. Kimchi's chain has no copy inside it: each row reads the
/// cells the row before wrote. Input `(1, first, rights)`.
#[derive(Clone, Debug)]
struct Chain<F, const N: usize> {
    modulus: Limbs,
    ops: [FFOps; N],
    checks: Checks,
    table: Rom<F>,
}

impl<F: PrimeField, const N: usize> TypedGadget<F> for Chain<F, N> {
    type Input = (Scalar, Array<Scalar, 3>, Array<Array<Scalar, 3>, N>);
    type Output = Unit;
    const NAME: &'static str = "kimchi-ffadd-test";
    // At most: the checks are the gadget's parameter.
    const ROWS: usize = chain_rows(N, Checks::All);

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (one, first, rights): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let place = |env: &mut E, column: usize, row: usize, v: &E::Var| {
            env.advice(Cell::new(column, row), [v], |[x]| x)
        };
        let one = env.call(&PublicInput, one);
        let first: [E::Var; 3] = core::array::from_fn(|k| place(env, k, 1, &first[k]));
        let mut left = first.clone();
        let mut checked = vec![first];
        for (i, (op, right)) in self.ops.iter().zip(&rights).enumerate() {
            // Operation `i` is on row `1 + i`.
            let right: [E::Var; 3] = core::array::from_fn(|k| place(env, 3 + k, 1 + i, &right[k]));
            let gate = ForeignFieldAddGate {
                modulus: self.modulus,
                op: *op,
            };
            let [r0, r1, r2, _] = env.call(&gate, (left, right.clone()));
            left = [r0, r1, r2];
            checked.extend([right, left.clone()]);
        }
        let constants = [F::ZERO, F::ZERO, two_pow(88)].map(|c| env.constant(c));
        let binary: [E::Var; 3] = core::array::from_fn(|k| place(env, 3 + k, N + 1, &constants[k]));
        let bound = ForeignFieldAddGate {
            modulus: self.modulus,
            op: FFOps::Add,
        };
        let [b0, b1, b2, q] = env.call(&bound, (left, binary));
        env.connect(&one, &q);
        env.call(&ZeroRow, ());
        let check = MultiRangeCheck {
            table: self.table.clone(),
        };
        let bound = [b0, b1, b2];
        let limbs: Vec<[E::Var; 3]> = match self.checks {
            Checks::None => vec![],
            Checks::Bound => vec![bound],
            Checks::All => checked.into_iter().chain([bound]).collect(),
        };
        for v in limbs {
            env.call(&check, v);
        }
    }
}

/// The limbs of `x`, below `2^264`.
fn limbs(x: &Nat) -> Limbs {
    let [lo, mi, hi] = x.limbs();
    [lo, mi, hi, 0]
}

/// The rows of a chain, and its witness on `inputs` (each reduced modulo
/// the modulus first, as Kimchi's `create_chain` does).
struct ChainCase<'a> {
    modulus: &'a BigUint,
    ops: &'a [FFOps],
    checks: Checks,
    inputs: Option<&'a [BigUint]>,
}

/// A chain's rows and witness, the witness when inputs are given.
type Built<F> = (Vec<CircuitGate<F>>, Option<[Vec<F>; COLUMNS]>);

impl ChainCase<'_> {
    fn build<F: PrimeField, const N: usize>(self) -> Built<F> {
        // Kimchi's constructors and witness generators panic on a modulus
        // above the maximum; our gadget takes any modulus, so the test asks.
        assert!(
            *self.modulus <= BigUint::max_foreign_field_modulus::<F>(),
            "foreign_field_modulus exceeds maximum"
        );
        let gadget = Chain::<F, N> {
            modulus: limbs(self.modulus),
            ops: core::array::from_fn(|i| self.ops[i]),
            checks: self.checks,
            table: range_check_table(),
        };
        let n = chain_rows(N, self.checks);
        let gates = gate_tests::kimchi_rows(&gadget, n);
        let witness = self.inputs.map(|inputs| {
            assert_eq!(inputs.len(), N + 1, "one more input than operations");
            let reduced = |x: &BigUint| -> [F; 3] {
                let [lo, mi, hi] = x.divrem(self.modulus).1.limbs();
                [lo, mi, hi].map(|l| Nat::from_u128(l).to_field())
            };
            let input = (
                F::ONE,
                reduced(&inputs[0]),
                core::array::from_fn(|i| reduced(&inputs[1 + i])),
            );
            gate_tests::kimchi_witness(&gadget, &input, n)
        });
        (gates, witness)
    }

    fn run<F: PrimeField>(self) -> Built<F> {
        match self.ops.len() {
            1 => self.build::<F, 1>(),
            2 => self.build::<F, 2>(),
            3 => self.build::<F, 3>(),
            20 => self.build::<F, 20>(),
            50 => self.build::<F, 50>(),
            n => panic!("no chain of {n} operations in Kimchi's tests"),
        }
    }
}

/// The rows of the chain of `opcodes`.
fn rows_of<F: PrimeField>(
    opcodes: &[FFOps],
    modulus: &BigUint,
    checks: Checks,
) -> Vec<CircuitGate<F>> {
    let case = ChainCase {
        modulus,
        ops: opcodes,
        checks,
        inputs: None,
    };
    case.run().0
}

/// The witness of the chain of `opcodes` on `inputs`, Kimchi's next one.
fn witness_of<F: PrimeField>(
    inputs: &[BigUint],
    opcodes: &[FFOps],
    modulus: &BigUint,
    checks: Checks,
) -> [Vec<F>; COLUMNS] {
    kimchi_inputs(modulus, inputs, opcodes);
    let case = ChainCase {
        modulus,
        ops: opcodes,
        checks,
        inputs: Some(inputs),
    };
    match case.run().1 {
        Some(witness) => witness,
        None => unreachable!("inputs give a witness"),
    }
}

// ---------------------------------------------------------------------
// Kimchi's tests, as Kimchi writes them.

// The secp256k1 base field modulus
fn secp256k1_modulus() -> BigUint {
    BigUint::from_bytes_be(&FIELD_SIZE)
}

// Maximum value in the foreign field of secp256k1
fn secp256k1_max() -> BigUint {
    secp256k1_modulus() - BigUint::from(1u32)
}

// A value that produces a negative low carry when added to itself
static OVF_NEG_LO: &[u8] = &[
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// A value that produces a negative middle carry when added to itself
static OVF_NEG_MI: &[u8] = &[
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2E,
];

// A value that produces overflow but the high limb of the result is smaller than the high limb of the modulus
static OVF_LESS_HI_LEFT: &[u8] = &[
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2E,
];
static OVF_LESS_HI_RIGHT: &[u8] = &[
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03, 0xD1,
];

// A value that produces two negative carries when added together with [OVF_ZERO_MI_NEG_LO]
static OVF_NEG_BOTH: &[u8] = &[
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// A value that produces two negative carries when added to itself with a middle limb that is all zeros
static OVF_ZERO_MI_NEG_LO: &[u8] = &[
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// All 0x55 bytes meaning [0101 0101]
static TIC: &[u8] = &[
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
];

// Prefix 0xAA bytes but fits in foreign field (suffix is zeros)
static TOC: &[u8] = &[
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// Bytestring that produces carry in low limb when added to TIC
static TOC_LO: &[u8] = &[
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xA9, 0xBA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// Bytestring that produces carry in mid limb when added to TIC
static TOC_MI: &[u8] = &[
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xA9, 0xBA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// Bytestring that produces carry in low and mid limb when added to TIC
static TOC_TWO: &[u8] = &[
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xA9, 0xBA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xA9, 0xBA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// Bottom half of the secp256k1 base field modulus
fn secp256k1_modulus_bottom() -> BigUint {
    BigUint::from_bytes_be(&FIELD_SIZE[16..])
}

// Top half of the secp256k1 base field modulus
fn secp256k1_modulus_top() -> BigUint {
    let bytes = [&FIELD_SIZE[0..16], &[0u8; 16]].concat();
    BigUint::from_bytes_be(&bytes)
}

// returns the maximum value for a field of modulus size
fn field_max(modulus: BigUint) -> BigUint {
    modulus - 1u32
}

// Value that performs a + - 1 low carry when added to [MAX]
static NULL_CARRY_LO: &[u8] = &[0x01, 0x00, 0x00, 0x03, 0xD2];

// Value that performs a + - 1 middle carry when added to [MAX]
static NULL_CARRY_MI: &[u8] = &[
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
];

// Value that performs two + - 1 carries when added to [MAX]
static NULL_CARRY_BOTH: &[u8] = &[
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03, 0xD2,
];

/// Check if a given circuit gate is a given foreign field operation
fn check_ffadd_sign<F: PrimeField>(gate: &CircuitGate<F>, sign: FFOps) -> Result<(), String> {
    if gate.typ != GateType::ForeignFieldAdd {
        return Err("Gate is not a foreign field add gate".to_string());
    }
    match sign {
        FFOps::Add => {
            if gate.coeffs[3] != F::one() {
                return Err("Gate is not performing addition".to_string());
            }
        }
        FFOps::Sub => {
            if gate.coeffs[3] != -F::one() {
                return Err("Gate is not performing subtraction".to_string());
            }
        }
    }
    Ok(())
}

// Kimchi's `short_circuit`: the public 1 and the chain, no range checks.
fn short_circuit<F: PrimeField>(
    opcodes: &[FFOps],
    foreign_field_modulus: &BigUint,
) -> (usize, Vec<CircuitGate<F>>) {
    let gates = rows_of(opcodes, foreign_field_modulus, Checks::None);
    (gates.len(), gates)
}

// Kimchi's `full_circuit`: the chain, and the range checks of its inputs,
// its results and its bound.
fn full_circuit<F: PrimeField>(
    opcodes: &[FFOps],
    foreign_field_modulus: &BigUint,
) -> (usize, Vec<CircuitGate<F>>) {
    let gates = rows_of(opcodes, foreign_field_modulus, Checks::All);
    (gates.len(), gates)
}

// Kimchi's `short_witness`.
fn short_witness<F: PrimeField>(
    inputs: &[BigUint],
    opcodes: &[FFOps],
    modulus: BigUint,
) -> [Vec<F>; COLUMNS] {
    witness_of(inputs, opcodes, &modulus, Checks::None)
}

// Kimchi's `long_witness`.
fn long_witness<F: PrimeField>(
    inputs: &[BigUint],
    opcodes: &[FFOps],
    modulus: BigUint,
) -> [Vec<F>; COLUMNS] {
    witness_of(inputs, opcodes, &modulus, Checks::All)
}

// helper to reduce lines of code in repetitive test structure
fn test_ffadd(
    foreign_field_modulus: BigUint,
    inputs: Vec<BigUint>,
    opcodes: &[FFOps],
    full: bool,
) -> ([Vec<PallasField>; COLUMNS], ConstraintSystem<Fp>) {
    let cs = {
        let foreign_field_modulus = foreign_field_modulus.clone();
        let (_next_row, gates) = if full {
            full_circuit(opcodes, &foreign_field_modulus)
        } else {
            short_circuit(opcodes, &foreign_field_modulus)
        };

        ConstraintSystem::create(gates).public(1).build().unwrap()
    };

    let witness = if full {
        long_witness(&inputs, opcodes, foreign_field_modulus)
    } else {
        short_witness(&inputs, opcodes, foreign_field_modulus)
    };

    kimchi_rows(&cs.gates);
    kimchi_witness(&witness);
    let all_rows = witness[0].len();

    for row in 0..all_rows {
        cs.gates[row]
            .verify_rec(row, &witness, &cs, &witness[0][0..cs.public])
            .ok();
    }

    (witness, cs)
}

// checks that the result cells of the witness are computed as expected
fn check_result(
    witness: [Vec<PallasField>; COLUMNS],
    result: Vec<ForeignElement<PallasField, LIMB_BITS, 3>>,
) {
    for (i, res) in result.iter().enumerate() {
        assert_eq!(witness[0][i + 2], res[LO]);
        assert_eq!(witness[1][i + 2], res[MI]);
        assert_eq!(witness[2][i + 2], res[HI]);
    }
}

// checks the result of the overflow bit for one addition
fn check_ovf(witness: [Vec<PallasField>; COLUMNS], ovf: PallasField) {
    assert_eq!(witness[6][1], ovf);
}

// checks the result of the carry bits for one addition
fn check_carry(witness: [Vec<PallasField>; COLUMNS], carry: PallasField) {
    assert_eq!(witness[7][1], carry);
}

// computes the result of an addition
fn compute_sum(modulus: BigUint, left: &[u8], right: &[u8]) -> BigUint {
    let left_big = BigUint::from_bytes_be(left);
    let right_big = BigUint::from_bytes_be(right);
    (left_big + right_big) % modulus
}

// computes the result of a subtraction
fn compute_dif(modulus: BigUint, left: &[u8], right: &[u8]) -> BigUint {
    let left_big = BigUint::from_bytes_be(left);
    let right_big = BigUint::from_bytes_be(right);
    if left_big < right_big {
        left_big + modulus - right_big
    } else {
        left_big - right_big
    }
}

#[test]
// Add zero to zero. This checks that small amounts also get packed into limbs
fn test_zero_add() {
    test("test_zero_add");
    test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::zero(), BigUint::zero()],
        &[FFOps::Add],
        false,
    );
}

#[test]
// Adding terms that are zero modulo the foreign field
fn test_zero_sum_foreign() {
    test("test_zero_sum_foreign");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![secp256k1_modulus_bottom(), secp256k1_modulus_top()],
        &[FFOps::Add],
        false,
    );
    check_result(witness, vec![ForeignElement::zero()]);
}

#[test]
// Adding terms that are zero modulo the native field
fn test_zero_sum_native() {
    test("test_zero_sum_native");
    let native_modulus = PallasField::modulus_biguint();
    let one = BigUint::new(vec![1u32]);
    let mod_minus_one = native_modulus.clone() - one;
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![One::one(), mod_minus_one],
        &[FFOps::Add],
        false,
    );

    // Check result is the native modulus
    let native_limbs = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&native_modulus);
    check_result(witness, vec![native_limbs]);
}

#[test]
fn test_one_plus_one() {
    test("test_one_plus_one");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![One::one(), One::one()],
        &[FFOps::Add],
        false,
    );
    // check result is 2
    let two = ForeignElement::from_be(&[2]);
    check_result(witness, vec![two]);
}

#[test]
// Adds two terms that are the maximum value in the foreign field
fn test_max_number() {
    test("test_max_number");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![secp256k1_max(), secp256k1_max()],
        &[FFOps::Add],
        false,
    );

    // compute result in the foreign field after taking care of the exceeding bits
    let sum = secp256k1_max() + secp256k1_max();
    let sum_mod = sum - secp256k1_modulus();
    let sum_mod_limbs = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&sum_mod);
    check_ovf(witness.clone(), PallasField::one());
    check_result(witness, vec![sum_mod_limbs]);
}

#[test]
// test 0 - 1 where (-1) is in the foreign field
// this is tested first as 0 + neg(1)
// and then as 0 - 1
// and it is checked that in both cases the result is the same
fn test_zero_minus_one() {
    test("test_zero_minus_one");
    // FIRST AS NEG
    let right_be_neg = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&One::one())
        .neg(&secp256k1_modulus())
        .to_biguint();
    let right_for_neg: ForeignElement<PallasField, LIMB_BITS, 3> =
        ForeignElement::from_biguint(&right_be_neg);
    let (witness_neg, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::zero(), right_be_neg],
        &[FFOps::Add],
        false,
    );
    check_result(witness_neg, vec![right_for_neg.clone()]);

    // NEXT AS SUB
    let (witness_sub, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::zero(), One::one()],
        &[FFOps::Sub],
        false,
    );
    check_result(witness_sub, vec![right_for_neg]);
}

#[test]
// test 1 - 1 + 1 where (-1) is in the foreign field
// the first check is done with sub(1, 1) and then with add(neg(neg(1)))
fn test_one_minus_one_plus_one() {
    test("test_one_minus_one_plus_one");
    let neg_neg_one = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&One::one())
        .neg(&secp256k1_modulus())
        .neg(&secp256k1_modulus())
        .to_biguint();
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![One::one(), One::one(), neg_neg_one],
        &[FFOps::Sub, FFOps::Add],
        false,
    );
    // intermediate 1 - 1 should be zero
    // final 0 + 1 should be 1
    check_result(
        witness,
        vec![
            ForeignElement::zero(),
            ForeignElement::from_biguint(&One::one()),
        ],
    );
}

#[test]
// test -1-1 where (-1) is in the foreign field
// first tested as neg(1) + neg(1)
// then tested as 0 - 1 - 1 )
// TODO: tested as 0 - ( 1 + 1) -> put sign in front of left instead (perhaps in the future we want this)
fn test_minus_minus() {
    test("test_minus_minus");
    let neg_one_for = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&One::one())
        .neg(&secp256k1_modulus());
    let neg_one = neg_one_for.to_biguint();
    let two = BigUint::from(2u32);
    let neg_two =
        ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&two).neg(&secp256k1_modulus());
    let (witness_neg, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![neg_one.clone(), neg_one],
        &[FFOps::Add],
        false,
    );
    check_result(witness_neg, vec![neg_two.clone()]);

    let (witness_sub, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::zero(), One::one(), One::one()],
        &[FFOps::Sub, FFOps::Sub],
        false,
    );
    check_result(witness_sub, vec![neg_one_for, neg_two]);
}

#[test]
// test when the low carry is minus one
fn test_neg_carry_lo() {
    test("test_neg_carry_lo");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(OVF_NEG_LO),
            BigUint::from_bytes_be(OVF_NEG_LO),
        ],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::zero());
}

#[test]
// test when the middle carry is minus one
fn test_neg_carry_mi() {
    test("test_neg_carry_mi");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(OVF_NEG_MI),
            BigUint::from_bytes_be(OVF_NEG_MI),
        ],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, -PallasField::one());
}

#[test]
// test when there is negative low carry and 0 middle limb (carry bit propagates)
fn test_propagate_carry() {
    test("test_propagate_carry");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(OVF_ZERO_MI_NEG_LO),
            BigUint::from_bytes_be(OVF_ZERO_MI_NEG_LO),
        ],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, -PallasField::one());
}

#[test]
// test when the both carries are minus one
fn test_neg_carries() {
    test("test_neg_carries");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(OVF_NEG_BOTH),
            BigUint::from_bytes_be(OVF_ZERO_MI_NEG_LO),
        ],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, -PallasField::one());
}

#[test]
// test the upperbound of the result
fn test_upperbound() {
    test("test_upperbound");
    test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(OVF_LESS_HI_LEFT),
            BigUint::from_bytes_be(OVF_LESS_HI_RIGHT),
        ],
        &[FFOps::Add],
        false,
    );
}

#[test]
// test a carry that nullifies in the low limb
fn test_null_lo_carry() {
    test("test_null_lo_carry");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![secp256k1_max(), BigUint::from_bytes_be(NULL_CARRY_LO)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::zero());
}

#[test]
// test a carry that nullifies in the mid limb
fn test_null_mi_carry() {
    test("test_null_mi_carry");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![secp256k1_max(), BigUint::from_bytes_be(NULL_CARRY_MI)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::zero());
}

#[test]
// test a carry that nullifies in the mid limb
fn test_null_both_carry() {
    test("test_null_both_carry");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![secp256k1_max(), BigUint::from_bytes_be(NULL_CARRY_BOTH)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::zero());
}

#[test]
// test sums without carry bits in any limb
fn test_no_carry_limbs() {
    test("test_no_carry_limbs");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::from_bytes_be(TIC), BigUint::from_bytes_be(TOC)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness.clone(), PallasField::zero());
    // check middle limb is all ones
    let all_one_limb = PallasField::from_kimchi(2u128.pow(88) - 1);
    assert_eq!(witness[1][2], all_one_limb);
}

#[test]
// test sum with carry only in low part
fn test_pos_carry_limb_lo() {
    test("test_pos_carry_limb_lo");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::from_bytes_be(TIC), BigUint::from_bytes_be(TOC_LO)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::zero());
}

#[test]
fn test_pos_carry_limb_mid() {
    test("test_pos_carry_limb_mid");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::from_bytes_be(TIC), BigUint::from_bytes_be(TOC_MI)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::one());
}

#[test]
fn test_pos_carry_limb_lo_mid() {
    test("test_pos_carry_limb_lo_mid");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::from_bytes_be(TIC), BigUint::from_bytes_be(TOC_TWO)],
        &[FFOps::Add],
        false,
    );
    check_carry(witness, PallasField::one());
}

#[test]
// Check it fails if given a wrong result (sum)
fn test_wrong_sum() {
    test("test_wrong_sum");
    let (mut witness, cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::from_bytes_be(TIC), BigUint::from_bytes_be(TOC)],
        &[FFOps::Add],
        true,
    );
    // wrong result
    let all_ones_limb = PallasField::from_kimchi(2u128.pow(88) - 1);
    witness[0][2] = all_ones_limb;
    witness[0][12] = all_ones_limb;

    cs.gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
}

#[test]
// Check it fails if given a wrong result (difference)
fn test_wrong_dif() {
    test("test_wrong_dif");
    let (mut witness, cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::from_bytes_be(TIC), BigUint::from_bytes_be(TOC)],
        &[FFOps::Sub],
        true,
    );
    // wrong result
    witness[0][2] = PallasField::zero();
    witness[0][12] = PallasField::zero();

    cs.gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
}

#[test]
// Test subtraction of the foreign field
fn test_zero_sub_fmod() {
    test("test_zero_sub_fmod");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::zero(), secp256k1_modulus()],
        &[FFOps::Sub],
        false,
    );
    // -f should be 0 mod f
    check_result(witness, vec![ForeignElement::zero()]);
}

#[test]
// Test subtraction of the foreign field maximum value
fn test_zero_sub_fmax() {
    test("test_zero_sub_fmax");
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![BigUint::zero(), secp256k1_max()],
        &[FFOps::Sub],
        false,
    );
    let negated = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&secp256k1_max())
        .neg(&secp256k1_modulus());
    check_result(witness, vec![negated]);
}

// The order of the Pallas curve is 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001.
// The order of the Vesta curve is  0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001.

#[test]
// Test with Pasta curves where foreign field is smaller than the native field
fn test_pasta_add_max_vesta() {
    test("test_pasta_add_max_vesta");
    let vesta_modulus = VestaField::modulus_biguint();
    let right_input = field_max(vesta_modulus.clone());
    let (witness, _cs) = test_ffadd(
        vesta_modulus.clone(),
        vec![BigUint::zero(), right_input.clone()],
        &[FFOps::Add],
        false,
    );
    let right = right_input % vesta_modulus;
    let right_foreign = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&right);
    check_result(witness, vec![right_foreign]);
}

#[test]
// Test with Pasta curves where foreign field is smaller than the native field
fn test_pasta_sub_max_vesta() {
    test("test_pasta_sub_max_vesta");
    let vesta_modulus = VestaField::modulus_biguint();
    let right_input = field_max(vesta_modulus.clone());
    let (witness, _cs) = test_ffadd(
        vesta_modulus.clone(),
        vec![BigUint::zero(), right_input.clone()],
        &[FFOps::Sub],
        false,
    );
    let neg_max_vesta =
        ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&right_input).neg(&vesta_modulus);
    check_result(witness, vec![neg_max_vesta]);
}

#[test]
// Test with Pasta curves where foreign field is smaller than the native field
fn test_pasta_add_max_pallas() {
    test("test_pasta_add_max_pallas");
    let vesta_modulus = VestaField::modulus_biguint();
    let right_input = field_max(PallasField::modulus_biguint());
    let (witness, _cs) = test_ffadd(
        vesta_modulus.clone(),
        vec![BigUint::zero(), right_input.clone()],
        &[FFOps::Add],
        false,
    );
    let right = right_input % vesta_modulus;
    let foreign_right = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&right);
    check_result(witness, vec![foreign_right]);
}

#[test]
// Test with Pasta curves where foreign field is smaller than the native field
fn test_pasta_sub_max_pallas() {
    test("test_pasta_sub_max_pallas");
    let vesta_modulus = VestaField::modulus_biguint();
    let right_input = field_max(PallasField::modulus_biguint());
    let (witness, _cs) = test_ffadd(
        vesta_modulus.clone(),
        vec![BigUint::zero(), right_input.clone()],
        &[FFOps::Sub],
        false,
    );
    let neg_max_pallas =
        ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&right_input).neg(&vesta_modulus);
    check_result(witness, vec![neg_max_pallas]);
}

#[test]
// Test with a random addition
fn test_random_add() {
    test("test_random_add");
    let rng = &mut next_rng();
    let foreign_mod = secp256k1_modulus();
    let left_input = random_input(rng, foreign_mod.clone(), true);
    let right_input = random_input(rng, foreign_mod.clone(), true);
    let left_big = BigUint::from_bytes_be(&left_input);
    let right_big = BigUint::from_bytes_be(&right_input);
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![left_big.clone(), right_big.clone()],
        &[FFOps::Add],
        false,
    );
    let result = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(
        &((left_big + right_big) % foreign_mod),
    );
    check_result(witness, vec![result]);
}

#[test]
// Test with a random subtraction
fn test_random_sub() {
    test("test_random_sub");
    let foreign_mod = secp256k1_modulus();
    let rng = &mut next_rng();
    let left_input = random_input(rng, foreign_mod.clone(), true);
    let right_input = random_input(rng, foreign_mod.clone(), true);
    let left_big = BigUint::from_bytes_be(&left_input);
    let right_big = BigUint::from_bytes_be(&right_input);
    let (witness, _cs) = test_ffadd(
        secp256k1_modulus(),
        vec![left_big.clone(), right_big.clone()],
        &[FFOps::Sub],
        false,
    );
    let result = if left_big < right_big {
        ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(
            &(left_big + foreign_mod - right_big),
        )
    } else {
        ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&(left_big - right_big))
    };
    check_result(witness, vec![result]);
}

#[test]
// Random test with foreign field being the native field add
fn test_foreign_is_native_add() {
    test("test_foreign_is_native_add");
    let rng = &mut next_rng();
    let pallas = PallasField::modulus_biguint();
    let left_input = random_input(rng, pallas.clone(), true);
    let right_input = random_input(rng, pallas.clone(), true);
    let (witness, _cs) = test_ffadd(
        pallas.clone(),
        vec![
            BigUint::from_bytes_be(&left_input),
            BigUint::from_bytes_be(&right_input),
        ],
        &[FFOps::Add],
        false,
    );
    // check result was computed correctly
    let sum_big = compute_sum(pallas, &left_input, &right_input);
    let result = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&sum_big);
    check_result(witness, vec![result.clone()]);
    // check result is in the native field
    let two_to_limb = PallasField::from_kimchi(TWO_TO_LIMB);
    let left = ForeignElement::<PallasField, LIMB_BITS, 3>::from_be(&left_input);
    let right = ForeignElement::<PallasField, LIMB_BITS, 3>::from_be(&right_input);
    let left = (left[HI] * two_to_limb + left[MI]) * two_to_limb + left[LO];
    let right = (right[HI] * two_to_limb + right[MI]) * two_to_limb + right[LO];
    let sum = left + right;
    let result = (result[HI] * two_to_limb + result[MI]) * two_to_limb + result[LO];
    let sum_from = PallasField::from_kimchi(sum_big);
    assert_eq!(result, sum);
    assert_eq!(result, sum_from);
}

#[test]
// Random test with foreign field being the native field add
fn test_foreign_is_native_sub() {
    test("test_foreign_is_native_sub");
    let rng = &mut next_rng();
    let pallas = PallasField::modulus_biguint();
    let left_input = random_input(rng, pallas.clone(), true);
    let right_input = random_input(rng, pallas.clone(), true);
    let (witness, _cs) = test_ffadd(
        pallas.clone(),
        vec![
            BigUint::from_bytes_be(&left_input),
            BigUint::from_bytes_be(&right_input),
        ],
        &[FFOps::Sub],
        false,
    );
    // check result was computed correctly
    let dif_big = compute_dif(pallas, &left_input, &right_input);
    let result = ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&dif_big);
    check_result(witness, vec![result.clone()]);
    // check result is in the native field
    let two_to_limb = PallasField::from_kimchi(TWO_TO_LIMB);
    let left = ForeignElement::<PallasField, LIMB_BITS, 3>::from_be(&left_input);
    let right = ForeignElement::<PallasField, LIMB_BITS, 3>::from_be(&right_input);
    let left = (left[HI] * two_to_limb + left[MI]) * two_to_limb + left[LO];
    let right = (right[HI] * two_to_limb + right[MI]) * two_to_limb + right[LO];
    let dif = left - right;
    let result = (result[HI] * two_to_limb + result[MI]) * two_to_limb + result[LO];
    let dif_from = PallasField::from_kimchi(dif_big);
    assert_eq!(result, dif);
    assert_eq!(result, dif_from);
}

#[test]
// Test with a random addition
fn test_random_small_add() {
    test("test_random_small_add");
    let rng = &mut next_rng();
    // 2^200 - 75 is prime with 200 bits (3 limbs but smaller than Pallas)
    let two = BigUint::from(2u32);
    let prime = two.pow(100).pow(2) - BigUint::from(75u32);
    let foreign_mod = prime.clone();
    let left_input = random_input(rng, foreign_mod.clone(), false);
    let right_input = random_input(rng, foreign_mod.clone(), false);
    let (witness, _cs) = test_ffadd(
        prime,
        vec![
            BigUint::from_bytes_be(&left_input),
            BigUint::from_bytes_be(&right_input),
        ],
        &[FFOps::Add],
        false,
    );
    let result = compute_sum(foreign_mod, &left_input, &right_input);
    check_result(
        witness,
        vec![ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(
            &result,
        )],
    );
}

#[test]
// Test with a random subtraction
fn test_random_small_sub() {
    test("test_random_small_sub");
    let rng = &mut next_rng();

    // 2^200 - 75 is prime with 200 bits (3 limbs but smaller than Pallas)
    let two = BigUint::from(2u32);
    let prime = two.clone().pow(100).pow(2) - BigUint::from(75u32);
    let foreign_mod = prime.clone();
    let left_input = random_input(rng, foreign_mod.clone(), false);
    let right_input = random_input(rng, foreign_mod.clone(), false);
    let (witness, _cs) = test_ffadd(
        prime,
        vec![
            BigUint::from_bytes_be(&left_input),
            BigUint::from_bytes_be(&right_input),
        ],
        &[FFOps::Sub],
        false,
    );
    let result = compute_dif(foreign_mod, &left_input, &right_input);
    check_result(
        witness,
        vec![ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(
            &result,
        )],
    );
}

#[test]
// Test with bad parameters in bound check
fn test_bad_bound() {
    test("test_bad_bound");
    let rng = &mut next_rng();

    let foreign_mod = secp256k1_modulus();
    let left_input = BigUint::from_bytes_be(&random_input(rng, foreign_mod.clone(), false));
    let right_input = BigUint::from_bytes_be(&random_input(rng, foreign_mod.clone(), false));
    let (mut witness, mut cs) = test_ffadd(
        foreign_mod,
        vec![left_input, right_input],
        &[FFOps::Add],
        true,
    );
    let mut gates = cs.gates.clone(); // to allow mutability during tests

    // Modify sign of bound
    // It should be constrained that sign needs to be 1
    gates[2].coeffs[3] = -PallasField::two();
    assert_eq!(
        check_ffadd_sign(&gates[2], FFOps::Add),
        Err("Gate is not performing addition".to_string()),
    );
    gates[2].coeffs[3] = PallasField::one();
    // Modify overflow to check first the copy constraint and then the ovf constraint
    witness[6][2] = -PallasField::one();
    cs.gates = gates;
    cs.gates[2]
        .verify_rec(2, &witness, &cs, &witness[0][0..cs.public])
        .ok();
    witness[0][0] = -PallasField::one();
    cs.gates[2]
        .verify_rec(2, &witness, &cs, &witness[0][0..cs.public])
        .ok();
    witness[6][2] = PallasField::one();
    witness[0][0] = PallasField::one();
    cs.gates[2]
        .verify_rec(2, &witness, &cs, &witness[0][0..cs.public])
        .ok();
}

#[test]
// Test with bad left input
fn test_random_bad_input() {
    test("test_random_bad_input");
    let rng = &mut next_rng();
    let foreign_mod = secp256k1_modulus();
    let left_input = random_input(rng, foreign_mod.clone(), false);
    let right_input = random_input(rng, foreign_mod, false);
    let (mut witness, cs) = test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(&left_input),
            BigUint::from_bytes_be(&right_input),
        ],
        &[FFOps::Sub],
        true,
    );
    // First modify left input only to cause an invalid copy constraint
    witness[0][1] = witness[0][1] + PallasField::one();
    cs.gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
    // then modify the value in the range check to cause an invalid FFAdd constraint
    witness[0][4] = witness[0][4] + PallasField::one();
    cs.gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
}

#[test]
// Test with bad parameters
fn test_random_bad_parameters() {
    test("test_random_bad_parameters");
    let rng = &mut next_rng();
    let foreign_mod = secp256k1_modulus();
    let left_input = random_input(rng, foreign_mod.clone(), false);
    let right_input = random_input(rng, foreign_mod, false);
    let (mut witness, mut cs) = test_ffadd(
        secp256k1_modulus(),
        vec![
            BigUint::from_bytes_be(&left_input),
            BigUint::from_bytes_be(&right_input),
        ],
        &[FFOps::Add],
        false,
    );

    let mut gates = cs.gates.clone();

    // Modify bot carry
    witness[7][1] = witness[7][1] + PallasField::one();
    gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
    witness[7][1] = witness[7][1] - PallasField::one();
    // Modify overflow
    witness[6][1] = witness[6][1] + PallasField::one();
    gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
    witness[6][1] = witness[6][1] - PallasField::one();
    // Modify sign
    gates[1].coeffs[3] = PallasField::zero() - gates[1].coeffs[3];
    cs.gates = gates.clone();
    gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();

    gates[1].coeffs[3] = PallasField::zero() - gates[1].coeffs[3];
    cs.gates = gates.clone();
    // Check back to normal
    gates[1]
        .verify_rec(1, &witness, &cs, &witness[0][0..cs.public])
        .ok();
}

#[test]
// Test with chain of random operations
fn test_random_chain() {
    test("test_random_chain");
    let rng = &mut next_rng();

    let nops = 20;
    let foreign_mod = secp256k1_modulus();
    let inputs = (0..nops + 1)
        .map(|_| random_input(rng, foreign_mod.clone(), true))
        .collect::<Vec<_>>();
    let big_inputs = inputs
        .clone()
        .into_iter()
        .map(|v| BigUint::from_bytes_be(&v))
        .collect::<Vec<_>>();
    let operations = (0..nops).map(|_| random_operation(rng)).collect::<Vec<_>>();
    let (witness, _cs) = test_ffadd(secp256k1_modulus(), big_inputs, &operations, true);
    // Not Kimchi's: the copies its wiring drops, which ours keeps.
    assert_eq!(run(|r| r.restored), 3 * (nops - 1));
    let mut left = vec![inputs[0].clone()];
    let results: Vec<ForeignElement<PallasField, LIMB_BITS, 3>> = operations
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let result = match op {
                FFOps::Add => compute_sum(foreign_mod.clone(), &left[i], &inputs[i + 1]),
                FFOps::Sub => compute_dif(foreign_mod.clone(), &left[i], &inputs[i + 1]),
            };
            left.push(result.to_bytes_be());
            ForeignElement::<PallasField, LIMB_BITS, 3>::from_biguint(&result)
        })
        .collect();
    check_result(witness, results);
}
// Prove and verify used for end-to-end tests
fn prove_and_verify_ops(operation_count: usize, _fixture_name: &'static str) {
    let rng = &mut next_rng();

    // Create random operations
    let operations = (0..operation_count)
        .map(|_| random_operation(rng))
        .collect::<Vec<_>>();

    // Create foreign modulus
    let foreign_field_modulus = secp256k1_modulus();

    // Create circuit
    // Initialize public input
    let (_next_row, gates) = short_circuit(&operations, &foreign_field_modulus);

    // Create random inputs
    let inputs = (0..operation_count + 1)
        .map(|_| BigUint::from_bytes_be(&random_input(rng, foreign_field_modulus.clone(), true)))
        .collect::<Vec<BigUint>>();

    // Create witness
    let witness = short_witness(&inputs, &operations, foreign_field_modulus);

    prove(gates, witness, vec![PallasField::one()]);
}

// Prove and verify a randomly generated operation (only ffadd)
#[test]
fn prove_and_verify_1() {
    test("prove_and_verify_1");
    {
        prove_and_verify_ops(1, "prove_and_verify_1");
    }
}

// Prove and verify a chain of 50 randomly generated operations (only ffadd)
#[test]
fn prove_and_verify_50() {
    test("prove_and_verify_50");
    {
        prove_and_verify_ops(50, "prove_and_verify_50");
    }
}

// Test with FFAdd gates without range checks
#[test]
fn test_ffadd_no_rc() {
    test("test_ffadd_no_rc");
    let operation_count = 3;
    let rng = &mut next_rng();

    // Create foreign modulus
    let foreign_mod = secp256k1_modulus();

    // Create random operations
    let operations = (0..operation_count)
        .map(|_| random_operation(rng))
        .collect::<Vec<_>>();

    // Create circuit, with the bound's range check (Kimchi's
    // `extend_gate_bound_rc`)
    let gates = rows_of(&operations, &foreign_mod, Checks::Bound);

    let cs = ConstraintSystem::create(gates).public(1).build().unwrap();

    // Create inputs
    let inputs = (0..operation_count + 1)
        .map(|_| BigUint::from_bytes_be(&random_input(rng, foreign_mod.clone(), false)))
        .collect::<Vec<BigUint>>();

    // Create witness, with the bound's range check (Kimchi's
    // `extend_witness_bound_rc`)
    let witness: [Vec<PallasField>; COLUMNS] =
        witness_of(&inputs, &operations, &foreign_mod, Checks::Bound);

    for row in 0..witness[0].len() {
        cs.gates[row]
            .verify_rec(row, &witness, &cs, &witness[0][0..cs.public])
            .ok();
    }
}

// TESTS CHANGING NATIVE FIELD

#[test]
// Tests targeting each custom constraint with Vesta (foreign field modulus) on Pallas (native field modulus)
fn test_vesta_on_pallas() {
    test("test_vesta_on_pallas");
    let test = run_test::<Fq>(&VestaField::modulus_biguint());
    assert_eq!(test.0, Ok(()));
}

#[test]
// Tests targeting each custom constraint with Pallas (foreign field modulus) on Vesta (native field modulus)
fn test_pallas_on_vesta() {
    test("test_pallas_on_vesta");
    let test = run_test::<Fp>(&PallasField::modulus_biguint());
    assert_eq!(test.0, Ok(()));
}

#[test]
// Tests targeting each custom constraint with Vesta (foreign field modulus) on Vesta (native field modulus)
fn test_vesta_on_vesta() {
    test("test_vesta_on_vesta");
    let test = run_test::<Fp>(&VestaField::modulus_biguint());
    assert_eq!(test.0, Ok(()));
}

#[test]
// Tests targeting each custom constraint with Pallas (foreign field modulus) on Pallas (native field modulus)
fn test_pallas_on_pallas() {
    test("test_pallas_on_pallas");
    let test = run_test::<Fq>(&PallasField::modulus_biguint());
    assert_eq!(test.0, Ok(()));
}

// Boilerplate for tests
fn run_test<F: Native>(
    foreign_field_modulus: &BigUint,
) -> (Result<(), CircuitGateError>, [Vec<F>; COLUMNS]) {
    let rng = &mut next_rng();

    // Create foreign field addition gates
    let (_next_row, gates) = short_circuit(&[FFOps::Add], foreign_field_modulus);

    let left_input =
        BigUint::from_bytes_be(&random_input(rng, foreign_field_modulus.clone(), true));
    let right_input =
        BigUint::from_bytes_be(&random_input(rng, foreign_field_modulus.clone(), true));

    // Compute addition witness
    let witness = short_witness(
        &[left_input, right_input],
        &[FFOps::Add],
        foreign_field_modulus.clone(),
    );

    let cs = ConstraintSystem::create(gates.clone())
        .public(1)
        .build()
        .unwrap();

    // Perform witness verification that everything is ok before invalidation (quick checks)
    for (row, gate) in gates.iter().enumerate().take(witness[0].len()) {
        let result = gate.verify_rec(row, &witness, &cs, &witness[0][0..cs.public]);
        if result.is_err() {
            return (result, witness);
        }
    }

    (Ok(()), witness)
}

#[test]
// Finalization test
fn test_ffadd_finalization() {
    test("test_ffadd_finalization");
    {
        // Includes a row to store value 1
        let num_public_inputs = 1;
        let operation = &[FFOps::Add];
        let modulus = BigUint::from_bytes_be(&[
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
            0xFF, 0xFF, 0xFC, 0x2F,
        ]);

        // circuit
        // [0]       -> Public input row to store the value 1
        // [1]       -> 1 ForeignFieldAdd row
        // [2]       -> 1 ForeignFieldAdd row for final bound
        // [3]       -> 1 Zero row for bound result
        // [4..=7]   -> 1 Multi RangeCheck for left input
        // [8..=11]  -> 1 Multi RangeCheck for right input
        // [12..=15] -> 1 Multi RangeCheck for result
        // [16..=19] -> 1 Multi RangeCheck for bound check
        assert_eq!(num_public_inputs, 1);
        let gates = rows_of(operation, &modulus, Checks::All);

        // witness: the addition of `modulus - 1` to itself
        let left = modulus.clone() - BigUint::one();
        let right = modulus.clone() - BigUint::one();
        // Kimchi's test calls `create_chain` itself, so its run records no
        // chain inputs to compare.
        let inputs = [left, right];
        let case = ChainCase {
            modulus: &modulus,
            ops: operation,
            checks: Checks::All,
            inputs: Some(&inputs),
        };
        let Some(witness) = case.run::<Fp>().1 else {
            unreachable!("inputs give a witness")
        };

        let index = ProverIndex {
            cs: ConstraintSystem::create(gates.clone())
                .public(num_public_inputs)
                .build()
                .unwrap(),
        };

        for row in 0..witness[0].len() {
            index.cs.gates[row]
                .verify_rec(row, &witness, &index.cs, &witness[0][0..index.cs.public])
                .ok();
        }

        prove(gates, witness.clone(), vec![witness[0][0]]);
    }
}

#[test]
fn test_gate_max_foreign_field_modulus() {
    test("test_gate_max_foreign_field_modulus");
    let _ = create_single_ffadd(
        FFOps::Add,
        &BigUint::max_foreign_field_modulus::<PallasField>(),
    );
}

#[test]
fn test_gate_invalid_foreign_field_modulus() {
    test("test_gate_invalid_foreign_field_modulus");
    panics(test_gate_invalid_foreign_field_modulus_body);
}

fn test_gate_invalid_foreign_field_modulus_body() {
    let _ = create_single_ffadd(
        FFOps::Add,
        &(BigUint::max_foreign_field_modulus::<PallasField>() + BigUint::one()),
    );
}

/// One ForeignFieldAdd row on inputs in columns 0-5, and the Zero row of
/// its result: Kimchi's `create_single_ffadd`.
#[derive(Clone, Copy, Debug)]
struct Single {
    gate: ForeignFieldAddGate,
}

impl TypedGadget<PallasField> for Single {
    type Input = (Array<Scalar, 3>, Array<Scalar, 3>);
    type Output = Unit;
    const NAME: &'static str = "kimchi-single-ffadd";
    const ROWS: usize = 2;

    fn synthesize<E: Env<PallasField>>(
        &self,
        env: &mut E,
        (left, right): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let left = core::array::from_fn(|k| env.advice(Cell::new(k, 0), [&left[k]], |[x]| x));
        let right = core::array::from_fn(|k| env.advice(Cell::new(3 + k, 0), [&right[k]], |[x]| x));
        env.call(&self.gate, (left, right));
        env.call(&ZeroRow, ());
    }
}

/// The rows of [`Single`], refusing a modulus above the maximum as
/// Kimchi's constructor does (our gadget takes any).
fn create_single_ffadd(op: FFOps, modulus: &BigUint) -> Vec<CircuitGate<PallasField>> {
    assert!(
        *modulus <= BigUint::max_foreign_field_modulus::<PallasField>(),
        "foreign_field_modulus exceeds maximum"
    );
    let single = Single {
        gate: ForeignFieldAddGate {
            modulus: limbs(modulus),
            op,
        },
    };
    let gates = gate_tests::kimchi_rows(&single, 2);
    assert_eq!(gates[0].typ, GateType::ForeignFieldAdd);
    gates
}

#[test]
fn test_witness_max_foreign_field_modulus() {
    test("test_witness_max_foreign_field_modulus");
    short_witness::<PallasField>(
        &[BigUint::zero(), BigUint::zero()],
        &[FFOps::Add],
        BigUint::max_foreign_field_modulus::<PallasField>(),
    );
}

#[test]
fn test_witness_invalid_foreign_field_modulus() {
    test("test_witness_invalid_foreign_field_modulus");
    panics(test_witness_invalid_foreign_field_modulus_body);
}

fn test_witness_invalid_foreign_field_modulus_body() {
    short_witness::<PallasField>(
        &[BigUint::zero(), BigUint::zero()],
        &[FFOps::Add],
        BigUint::max_foreign_field_modulus::<PallasField>() + BigUint::one(),
    );
}

/// The prover index of the finalization test: only its constraint system
/// is read.
struct ProverIndex {
    cs: ConstraintSystem<Fp>,
}
