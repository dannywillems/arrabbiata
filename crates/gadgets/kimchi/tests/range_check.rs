//! Kimchi's multi-range check (RangeCheck0, RangeCheck0, RangeCheck1, Zero)
//! against Kimchi.
//!
//! - for random values below `2^88`, the four rows are Kimchi's
//!   `range_check::witness::create_multi`, cell for cell (each cell the bit
//!   range of its value Kimchi's layout gives), and the witness satisfies the
//!   circuit;
//! - a value of `2^88` or more is rejected;
//! - a 12-bit limb pushed out of range while keeping the recombination
//!   exact is caught by the lookup alone.
//!
//! No proof here: Kimchi's 12-bit table makes the circuit 8192 rows, and
//! generalized PlonK opens each of its polynomials with its own IPA proof,
//! which at that size takes hours. Batched openings (plan 0001, step 5) are
//! what makes it practical; proofs with lookups are tested at a small size
//! in `arrabbiata-gadget`'s `tests/lookups.rs`.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, TypedGadget, Unit, witness};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    range_check::{MultiRangeCheck, range_check_table},
};
use arrabbiata_pasta::Fp;
use proptest::prelude::*;

/// The column variables: every cell of a row, those of the next, and
/// coefficient 0.
const VARIABLES: usize = 32;
/// A crumb's check.
const DEGREE: usize = 4;

/// Below `2^88`.
const LIMIT: u128 = 1 << 88;

/// The three values on column 0 of rows 0-2, then the check.
#[derive(Clone, Debug)]
struct Check;

impl TypedGadget<Fp> for Check {
    type Input = Array<3>;
    type Output = Unit;
    const NAME: &'static str = "check";
    const ROWS: usize = 4;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        values: Of<Array<3>, E::Var>,
    ) -> Of<Unit, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(0, i), [&values[i]], |[v]| v));
        let gadget = MultiRangeCheck {
            table: range_check_table(),
        };
        env.call(&gadget, cells);
    }
}

fn fp(v: u128) -> Fp {
    let two_64 = Fp::from_u64(1 << 32) * Fp::from_u64(1 << 32);
    Fp::from_u64(v as u64) + Fp::from_u64((v >> 64) as u64) * two_64
}

/// The bits `start .. end` of `v`.
fn range(v: u128, start: usize, end: usize) -> u64 {
    ((v >> start) & ((1u128 << (end - start)) - 1)) as u64
}

/// Kimchi's `create_multi(v0, v1, v2)`, from its layout: the bit range of
/// each cell, and the copies of row 3.
fn kimchi_rows(v: [u128; 3]) -> [[u64; 15]; 4] {
    let zero_row = |x: u128| -> [u64; 15] {
        let mut row = [0u64; 15];
        row[0] = x as u64;
        let ranges = [
            (76, 88),
            (64, 76),
            (52, 64),
            (40, 52),
            (28, 40),
            (16, 28),
            (14, 16),
            (12, 14),
            (10, 12),
            (8, 10),
            (6, 8),
            (4, 6),
            (2, 4),
            (0, 2),
        ];
        for (i, (s, e)) in ranges.iter().enumerate() {
            row[1 + i] = range(x, *s, *e);
        }
        row
    };
    let r0 = zero_row(v[0]);
    let r1 = zero_row(v[1]);
    let mut r2 = [0u64; 15];
    let curr = [
        (2, 86, 88),
        (3, 74, 86),
        (4, 62, 74),
        (5, 50, 62),
        (6, 38, 50),
        (7, 36, 38),
        (8, 34, 36),
        (9, 32, 34),
        (10, 30, 32),
        (11, 28, 30),
        (12, 26, 28),
        (13, 24, 26),
        (14, 22, 24),
    ];
    for (c, s, e) in curr {
        r2[c] = range(v[2], s, e);
    }
    let mut r3 = [0u64; 15];
    let next = [
        (0, 20, 22),
        (1, 18, 20),
        (2, 16, 18),
        (7, 14, 16),
        (8, 12, 14),
        (9, 10, 12),
        (10, 8, 10),
        (11, 6, 8),
        (12, 4, 6),
        (13, 2, 4),
        (14, 0, 2),
    ];
    for (c, s, e) in next {
        r3[c] = range(v[2], s, e);
    }
    r3[3] = r0[1];
    r3[4] = r0[2];
    r3[5] = r1[1];
    r3[6] = r1[2];
    [r0, r1, r2, r3]
}

fn arb_value() -> impl Strategy<Value = u128> {
    0..LIMIT
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn kimchis_layout(v in prop::array::uniform3(arb_value())) {
        let input = v.map(fp);
        let (compiled, ()) = witness::<Fp, _, VARIABLES, DEGREE>(&Check, LAYOUT, &input).unwrap();
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        let rows = kimchi_rows(v);
        for (r, expected) in rows.iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][r]);
            // Column 0 of rows 0-2 is the value itself, which may not fit a
            // u64.
            let expected: [Fp; 15] = core::array::from_fn(|c| if c == 0 && r < 3 { input[r] } else { Fp::from_u64(expected[c]) });
            prop_assert_eq!(found, expected, "row {}", r);
        }
    }

    #[test]
    fn values_of_88_bits_or_more_are_rejected(position in 0usize..3, excess in 0u128..1000) {
        let mut input = [Fp::ONE; 3];
        input[position] = fp(LIMIT + excess);
        let found = witness::<Fp, _, VARIABLES, DEGREE>(&Check, LAYOUT, &input);
        prop_assert!(matches!(found, Err(Error::Unsatisfied { .. })), "{:?}", found.err());
    }
}

/// `v0p2` (column 3) raised by `4096` and `v0p1` (column 2) lowered by 1:
/// `v0` is unchanged, every crumb is a crumb, and only the lookup of column
/// 3 fails.
#[test]
fn only_the_lookup_catches_a_large_limb() {
    let v0: u128 = (5 << 64) | (7 << 52) | 12_345;
    let input = [fp(v0), Fp::ONE, Fp::ONE];
    let (compiled, ()) = witness::<Fp, _, VARIABLES, DEGREE>(&Check, LAYOUT, &input).unwrap();
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
    let mut columns = compiled.columns.clone();
    columns[3][0] = columns[3][0] + Fp::from_u64(4096);
    columns[2][0] = columns[2][0] - Fp::ONE;
    // Row 3 carries a copy of column 2 of row 0.
    columns[4][3] = columns[2][0];
    assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
}
