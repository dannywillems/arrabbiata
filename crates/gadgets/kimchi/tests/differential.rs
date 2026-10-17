//! A differential test against Kimchi (o1-labs/proof-systems
//! `9227444890`), on Kimchi's own output (`vectors/kimchi_differential.txt`,
//! produced by the `differential` tool of `tools/kimchi-vectors`):
//!
//! - the constraint polynomials: for every gate, at 4 pseudo-random points
//!   (all cells of the gate's row and the next, and all coefficients,
//!   random field elements, so far from a satisfying witness), each
//!   constraint of our gate kind takes the value Kimchi's
//!   `constraint_checks` gives, in Kimchi's order. Two polynomials of
//!   degree at most 7 agreeing at 4 random points of a 255-bit field are
//!   equal except with probability about `4 * 7 / 2^255` each (our
//!   framing of Schwartz-Zippel), so this pins the constraints, not only
//!   their counts and degrees as `tests/figures.rs` does;
//! - the witnesses: on fixed inputs, our gadgets fill the rows Kimchi's
//!   witness generators fill (`range_check::witness::create_multi`,
//!   `xor::create_xor_witness`, `rot::extend_rot` left and right,
//!   `foreign_field_add::witness::create_chain` for an addition and a
//!   subtraction, `foreign_field_mul::witness::create`), cell for cell.

mod kimchi_runs;

use std::sync::LazyLock;

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit, gates, witness};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    foreign_field::Limbs,
    foreign_field_add::{FFOps, ForeignFieldAdd},
    foreign_field_mul::ForeignFieldMul,
    harness::{self, NAMES},
    range_check::{MultiRangeCheck, range_check_table},
    rot::{Rot64, RotMode},
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::{Fp, Pallas};
use arrabbiata_poseidon::kimchi::pallas_base;

/// Kimchi's output.
static VECTORS: LazyLock<String> = LazyLock::new(|| kimchi_runs::read("kimchi_differential.txt"));

/// The column variables: every cell of a row, those of the next, and the
/// coefficients.
const VARIABLES: usize = 40;
/// The largest constraint degree of the gadgets compiled here.
const DEGREE: usize = 4;

/// A field element from 64 hex digits, big-endian.
fn element(hex: &str) -> Fp {
    let limbs: Vec<u64> = (0..4)
        .rev()
        .map(|i| u64::from_str_radix(&hex[16 * i..16 * (i + 1)], 16).unwrap())
        .collect();
    Fp::from_limbs(&limbs).unwrap()
}

/// The elements after the tag of `line`.
fn elements(line: &str, tag: &str) -> Vec<Fp> {
    let mut words = line.split_whitespace();
    assert_eq!(words.next(), Some(tag), "{line}");
    words.map(element).collect()
}

/// One evaluation point of a gate.
struct Point {
    gate: String,
    curr: Vec<Fp>,
    next: Vec<Fp>,
    coeffs: Vec<Fp>,
    values: Vec<Fp>,
}

/// One witness: its name and rows.
struct Rows {
    name: String,
    rows: Vec<Vec<Fp>>,
}

fn parse() -> (Vec<Point>, Vec<Rows>) {
    let mut lines = VECTORS.lines().peekable();
    let (mut points, mut witnesses) = (Vec::new(), Vec::new());
    while let Some(line) = lines.next() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[0] {
            "constraints" => points.push(Point {
                gate: words[1].to_string(),
                curr: elements(lines.next().unwrap(), "curr"),
                next: elements(lines.next().unwrap(), "next"),
                coeffs: elements(lines.next().unwrap(), "coeffs"),
                values: elements(lines.next().unwrap(), "values"),
            }),
            "witness" => {
                let n: usize = words[2].parse().unwrap();
                let rows = (0..n)
                    .map(|_| elements(lines.next().unwrap(), "row"))
                    .collect();
                witnesses.push(Rows {
                    name: words[1].to_string(),
                    rows,
                });
            }
            other => panic!("unexpected line {other}"),
        }
    }
    (points, witnesses)
}

#[test]
fn every_constraint_is_kimchis() {
    let (points, _) = parse();
    assert_eq!(points.len(), 4 * NAMES.len());
    let poseidon = pallas_base();
    for point in &points {
        let (_, ours) = NAMES.iter().find(|(k, _)| *k == point.gate).unwrap();
        let harness = harness::of(ours, &poseidon, Pallas::beta());
        let kinds = gates::<Fp, _>(&harness, LAYOUT).unwrap();
        let kind = kinds
            .iter()
            .find(|k| k.gadget == *ours && k.offset == 0)
            .unwrap_or_else(|| panic!("no gate kind {ours}"));
        let witness = |c: usize, r: usize| if r == 0 { point.curr[c] } else { point.next[c] };
        let fixed = |c: usize, _: usize| point.coeffs[c];
        let found = kind.evaluate(&witness, &fixed);
        assert_eq!(
            found.len(),
            point.values.len(),
            "{}: constraint count",
            point.gate
        );
        for (i, (f, k)) in found.iter().zip(&point.values).enumerate() {
            assert_eq!(
                f, k,
                "{}: constraint {} differs from Kimchi's",
                point.gate, i
            );
        }
        // The comparison is not vacuous: at a random point the constraints
        // are not zero, and moving the cells moves the values.
        assert!(
            point.values.iter().any(|v| *v != Fp::ZERO),
            "{}",
            point.gate
        );
        let moved = |c: usize, r: usize| witness(c, r) + Fp::ONE;
        assert_ne!(
            kind.evaluate(&moved, &fixed),
            point.values,
            "{}",
            point.gate
        );
    }
}

/// The values of column 0 of rows 0-2, then the multi-range check.
#[derive(Clone, Debug)]
struct RangeCheck;

impl TypedGadget<Fp> for RangeCheck {
    type Input = Array<Scalar, 3>;
    type Output = Unit;
    const NAME: &'static str = "range-check";
    const ROWS: usize = 4;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        v: Of<Array<Scalar, 3>, E::Var>,
    ) -> Of<Unit, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(0, i), [&v[i]], |[x]| x));
        let gadget = MultiRangeCheck {
            table: range_check_table(),
        };
        env.call(&gadget, cells);
    }
}

/// Two words in columns 0 and 1 of row 0, then their 64-bit XOR.
#[derive(Clone, Debug)]
struct XorWords;

impl TypedGadget<Fp> for XorWords {
    type Input = Array<Scalar, 2>;
    type Output = arrabbiata_gadget::Scalar;
    const NAME: &'static str = "xor-words";
    const ROWS: usize = 5;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        w: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<arrabbiata_gadget::Scalar, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&w[i]], |[x]| x));
        env.call(&Xor::<Fp, 64> { table: xor_table() }, cells)
    }
}

/// A word in column 0 of row 0, then its rotation.
#[derive(Clone, Debug)]
struct Rotate(u32, RotMode);

impl TypedGadget<Fp> for Rotate {
    type Input = arrabbiata_gadget::Scalar;
    type Output = arrabbiata_gadget::Scalar;
    const NAME: &'static str = "rotate";
    const ROWS: usize = 4;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, w: E::Var) -> E::Var {
        let cell = env.advice(Cell::new(0, 0), [&w], |[x]| x);
        let gadget = Rot64 {
            table: range_check_table(),
            rot: self.0,
            side: self.1,
        };
        env.call(&gadget, cell)
    }
}

/// secp256k1's base field, as limbs.
fn secp256k1() -> Limbs {
    let words: [u64; 4] = [
        0xffff_fffe_ffff_fc2f,
        0xffff_ffff_ffff_ffff,
        0xffff_ffff_ffff_ffff,
        0xffff_ffff_ffff_ffff,
    ];
    let bit = |i: usize| (words[i / 64] >> (i % 64)) & 1;
    let mut limbs = [0u128; 4];
    for i in 0..256 {
        limbs[i / 88] |= u128::from(bit(i)) << (i % 88);
    }
    limbs
}

/// `a` and `b` in the row after the gadget, then the operation `op`
/// (`None` for a multiplication).
#[derive(Clone, Debug)]
struct Foreign(Option<FFOps>);

/// The rows before the inputs: the larger of the two gadgets.
const FOREIGN_ROWS: usize = 19;

impl TypedGadget<Fp> for Foreign {
    type Input = Array<Scalar, 6>;
    type Output = Array<Scalar, 3>;
    const NAME: &'static str = "foreign";
    const ROWS: usize = FOREIGN_ROWS + 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        x: Of<Array<Scalar, 6>, E::Var>,
    ) -> Of<Array<Scalar, 3>, E::Var> {
        let cells: [E::Var; 6] =
            core::array::from_fn(|c| env.advice(Cell::new(c, FOREIGN_ROWS), [&x[c]], |[v]| v));
        let [a0, a1, a2, b0, b1, b2] = cells;
        let (modulus, table) = (secp256k1(), range_check_table());
        match self.0 {
            Some(op) => env.call(
                &ForeignFieldAdd { modulus, op, table },
                ([a0, a1, a2], [b0, b1, b2]),
            ),
            None => env.call(
                &ForeignFieldMul { modulus, table },
                ([a0, a1, a2], [b0, b1, b2]),
            ),
        }
    }
}

/// Our columns, rows `from ..`, against Kimchi's rows.
fn same_rows(name: &str, columns: &[Vec<Fp>], from: usize, kimchi: &[Vec<Fp>]) {
    for (k, row) in kimchi.iter().enumerate() {
        let found: Vec<Fp> = (0..15).map(|c| columns[c][from + k]).collect();
        assert_eq!(&found, row, "{name}: row {k}");
    }
}

#[test]
fn every_witness_is_kimchis() {
    let (_, witnesses) = parse();
    let get = |name: &str| &witnesses.iter().find(|w| w.name == name).unwrap().rows;

    let rows = get("RangeCheck");
    let input = [rows[0][0], rows[1][0], rows[2][0]];
    let (c, ()) = witness::<Fp, _, VARIABLES, DEGREE>(&RangeCheck, LAYOUT, &input).unwrap();
    same_rows("RangeCheck", &c.columns, 0, rows);

    let rows = get("Xor");
    let input = [rows[0][0], rows[0][1]];
    let (c, _) = witness::<Fp, _, VARIABLES, DEGREE>(&XorWords, LAYOUT, &input).unwrap();
    same_rows("Xor", &c.columns, 0, rows);

    for (name, gadget) in [
        ("RotLeft3", Rotate(3, RotMode::Left)),
        ("RotRight17", Rotate(17, RotMode::Right)),
    ] {
        let rows = get(name);
        let (c, _) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &rows[0][0]).unwrap();
        same_rows(name, &c.columns, 0, rows);
    }

    for (name, op, from) in [
        ("ForeignFieldAddSecp", Some(FFOps::Add), 2),
        ("ForeignFieldSubSecp", Some(FFOps::Sub), 2),
        ("ForeignFieldMulSecp", None, 0),
    ] {
        let rows = get(name);
        let input: [Fp; 6] = core::array::from_fn(|c| rows[0][c]);
        let (c, _) = witness::<Fp, _, VARIABLES, DEGREE>(&Foreign(op), LAYOUT, &input).unwrap();
        same_rows(name, &c.columns, from, rows);
    }
}
