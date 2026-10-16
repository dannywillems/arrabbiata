//! Kimchi's RangeCheck0 and RangeCheck1 gates: three values of up to 88
//! bits on four rows (`MultiRangeCheck`), decomposed into 12-bit limbs,
//! looked up in the table of `0 .. 2^12`, and 2-bit crumbs, constrained by
//! `x (x - 1) (x - 2) (x - 3)`.
//!
//! The rows, as Kimchi lays them out (`vipj` the 12-bit limb `j` of `vi`,
//! `vicj` its crumb `j`, most significant first):
//!
//! | row | gate        | 0      | 1      | 2      | 3-6                 | 7-14          |
//! | --- | ----------- | ------ | ------ | ------ | ------------------- | ------------- |
//! | 0   | RangeCheck0 | `v0`   | `v0p0` | `v0p1` | `v0p2 .. v0p5`      | `v0c0 .. c7`  |
//! | 1   | RangeCheck0 | `v1`   | `v1p0` | `v1p1` | `v1p2 .. v1p5`      | `v1c0 .. c7`  |
//! | 2   | RangeCheck1 | `v2`   | `v12`  | `v2c0` | `v2p0 .. v2p3`      | `v2c1 .. c8`  |
//! | 3   | (Zero)      | `v2c9` | `v2c10`| `v2c11`| `v0p0 v0p1 v1p0 v1p1` | `v2c12 .. c19` |
//!
//! Columns 3-6 are looked up on rows 0-3; `v0p0, v0p1, v1p0, v1p1` are copied
//! to row 3 to be looked up there (four lookups per row). The witness is
//! Kimchi's `range_check::witness::create_multi`: each cell holds a bit range
//! of its value.
//!
//! RangeCheck0 has 10 constraints: the eight crumbs, the recombination of
//! `v`, and, under coefficient 0 (the compact mode, here always 0), the
//! check of the next row's `v12`. RangeCheck1 has 21: its 20 crumbs and the
//! recombination of `v2`.

use alloc::vec::Vec;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, Scalar, TypedGadget, Unit};

use crate::TABLE_COLUMNS;

/// The bits of a limb of the 88-bit values.
pub const LIMB_BITS: usize = 88;

/// The table's name.
pub const RANGE_CHECK_TABLE: &str = "kimchi-range-check-12";

/// The numbers `0 .. 2^12`, in the first table column (Kimchi's
/// `range_check_table`).
#[must_use]
pub fn range_check_table<F: Field>() -> Rom<F> {
    let rows = (0..1u64 << 12)
        .map(|v| alloc::vec![F::from_u64(v)])
        .collect();
    let column = alloc::vec![TABLE_COLUMNS[0]];
    Rom::new(RANGE_CHECK_TABLE, column, rows)
}

/// The bits `start .. end` of `x`, for a range of at most 128 bits.
pub(crate) fn bits<F: PrimeField>(x: F, start: usize, end: usize) -> F {
    debug_assert!(end - start <= 128, "at most 128 bits");
    let mut value = 0u128;
    for i in (start..end).rev() {
        let bit = (x.limb(i / 64) >> (i % 64)) & 1;
        value = (value << 1) | u128::from(bit);
    }
    crate::foreign_field::field_of(value)
}

/// `x (x - 1) (x - 2) (x - 3)`, as Kimchi's `crumb`.
pub(crate) fn crumb<F, E>(env: &mut E, x: &E::Var) -> E::Var
where
    F: Field,
    E: Env<F>,
{
    let [one, two, three] = [1, 2, 3].map(|k| env.constant(F::from_u64(k)));
    x.clone() * (x.clone() - one) * (x.clone() - two) * (x.clone() - three)
}

/// `sum_k 2^(w_k) cells_k`, accumulated as Kimchi does: from `0`, each cell
/// times the running power, the power then multiplied by `4` for a crumb or
/// `4096` for a limb (`widths`, in the order the cells are added).
pub(crate) fn recombine<F, E>(env: &mut E, cells: &[(E::Var, u64)]) -> E::Var
where
    F: Field,
    E: Env<F>,
{
    let mut power = F::ONE;
    let mut sum = env.constant(F::ZERO);
    for (cell, factor) in cells {
        let p = env.constant(power);
        sum = sum + p * cell.clone();
        power = power * F::from_u64(*factor);
    }
    sum
}

/// The bit ranges of the cells of a RangeCheck0 row, columns 1-14.
const RANGE_CHECK_0: [(usize, usize); 14] = [
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

/// The bit ranges of the cells of a RangeCheck1 row, columns 2-14.
const RANGE_CHECK_1_CURR: [(usize, usize); 13] = [
    (86, 88),
    (74, 86),
    (62, 74),
    (50, 62),
    (38, 50),
    (36, 38),
    (34, 36),
    (32, 34),
    (30, 32),
    (28, 30),
    (26, 28),
    (24, 26),
    (22, 24),
];

/// The bit ranges of columns 0-2 and 7-14 of the row after a RangeCheck1.
const RANGE_CHECK_1_NEXT: [(usize, usize, usize); 11] = [
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

/// The 12-bit limbs Kimchi looks up on a row.
const LOOKED_UP: [usize; 4] = [3, 4, 5, 6];

/// The cell `at` of `v`'s bits `start .. end`.
fn limb<F, E>(env: &mut E, at: Cell, v: &E::Var, (start, end): (usize, usize)) -> E::Var
where
    F: PrimeField,
    E: Env<F>,
{
    env.advice(at, [v], |[x]| bits(x, start, end))
}

/// RangeCheck0: `v`, input, wired by copy to column 0, decomposed on its
/// row; the output is its two highest limbs `[vp0, vp1]`, for
/// [`RangeCheck1Gate`] to look up.
///
/// `compact` is Kimchi's coefficient 0: when set, the gate also checks
/// `v12 = v + 2^88 v'` with `v12, v'` columns 1 and 0 of the next row, which
/// is how a 176-bit compact limb is range-checked
/// ([`CompactMultiRangeCheck`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeCheck0Gate<F> {
    /// The 12-bit table.
    pub table: Rom<F>,
    /// Kimchi's compact mode (coefficient 0 = 1).
    pub compact: bool,
}

impl<F> TypedGadget<F> for RangeCheck0Gate<F>
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Array<2>;
    const NAME: &'static str = "kimchi-range-check-0";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, v: Of<Scalar, E::Var>) -> Of<Array<2>, E::Var>
    where
        E: Env<F>,
    {
        let v = env.copy(Cell::new(0, 0), &v);
        let cells: Vec<E::Var> = RANGE_CHECK_0
            .iter()
            .enumerate()
            .map(|(i, range)| limb(env, Cell::new(1 + i, 0), &v, *range))
            .collect();
        for c in &cells[6..] {
            let check = crumb(env, c);
            env.assert_zero(check);
        }
        // From the least significant: the crumbs (columns 14 down to 7), then
        // the limbs (6 down to 1).
        let ordered: Vec<(E::Var, u64)> = (7..=14)
            .rev()
            .map(|c| (cells[c - 1].clone(), 4))
            .chain((1..=6).rev().map(|c| (cells[c - 1].clone(), 4096)))
            .collect();
        let sum = recombine(env, &ordered);
        env.assert_zero(sum - v.clone());
        // The compact mode: coefficient 0 turns on `v12 = v + 2^88 v'`, the
        // next row's columns 1 and 0.
        let mode = if self.compact { F::ONE } else { F::ZERO };
        let compact = env.fixed(Cell::new(0, 0), mode);
        let next0 = env.cell(Cell::new(0, 1));
        let next1 = env.cell(Cell::new(1, 1));
        let two_to_limb = env.constant(F::from_u64(1 << 44) * F::from_u64(1 << 44));
        env.assert_zero(compact * (next1 - (v + two_to_limb * next0)));
        for c in LOOKED_UP {
            env.read(&self.table, &[cells[c - 1].clone()]);
        }
        [cells[0].clone(), cells[1].clone()]
    }
}

/// RangeCheck1, on its row and the next: input `(v2, [v0p0, v0p1, v1p0,
/// v1p1])`, `v2` wired to column 0, the four limbs copied to columns 3-6 of
/// the next row, where they are looked up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeCheck1Gate<F> {
    /// The 12-bit table.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for RangeCheck1Gate<F>
where
    F: PrimeField,
{
    type Input = (Scalar, Array<4>);
    type Output = Unit;
    const NAME: &'static str = "kimchi-range-check-1";
    const ROWS: usize = 2;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (v2, deferred): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var>
    where
        E: Env<F>,
    {
        let v2 = env.copy(Cell::new(0, 0), &v2);
        let curr: Vec<E::Var> = RANGE_CHECK_1_CURR
            .iter()
            .enumerate()
            .map(|(i, range)| limb(env, Cell::new(2 + i, 0), &v2, *range))
            .collect();
        let next: Vec<E::Var> = RANGE_CHECK_1_NEXT
            .iter()
            .map(|(column, start, end)| limb(env, Cell::new(*column, 1), &v2, (*start, *end)))
            .collect();
        let copies: Vec<E::Var> = deferred
            .iter()
            .zip(LOOKED_UP)
            .map(|(d, column)| env.copy(Cell::new(column, 1), d))
            .collect();
        // `curr[k]` is column `2 + k`; `next` is columns 0-2, then 7-14.
        let curr_column = |c: usize| curr[c - 2].clone();
        let next_column = |c: usize| {
            if c <= 2 {
                next[c].clone()
            } else {
                next[c - 4].clone()
            }
        };
        let mut crumbs = alloc::vec![curr_column(2)];
        crumbs.extend((7..=14).map(curr_column));
        crumbs.extend((0..=2).map(next_column));
        crumbs.extend((7..=14).map(next_column));
        for c in &crumbs {
            let check = crumb(env, c);
            env.assert_zero(check);
        }
        let ordered: Vec<(E::Var, u64)> = (7..=14)
            .rev()
            .map(|c| (next_column(c), 4))
            .chain((0..=2).rev().map(|c| (next_column(c), 4)))
            .chain((7..=14).rev().map(|c| (curr_column(c), 4)))
            .chain((3..=6).rev().map(|c| (curr_column(c), 4096)))
            .chain(core::iter::once((curr_column(2), 1)))
            .collect();
        let sum = recombine(env, &ordered);
        env.assert_zero(sum - v2);
        for c in LOOKED_UP {
            env.read(&self.table, &[curr_column(c)]);
        }
        for c in &copies {
            env.read(&self.table, core::slice::from_ref(c));
        }
    }
}

/// Three values of up to 88 bits, on four rows: RangeCheck0 for `v0` and
/// `v1`, RangeCheck1 for `v2`. Each value is wired by copy to column 0 of
/// its row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiRangeCheck<F> {
    /// The 12-bit table.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for MultiRangeCheck<F>
where
    F: PrimeField,
{
    type Input = Array<3>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-multi-range-check";
    const ROWS: usize = 4;

    fn synthesize<E>(&self, env: &mut E, [v0, v1, v2]: Of<Array<3>, E::Var>) -> Of<Unit, E::Var>
    where
        E: Env<F>,
    {
        let zero = RangeCheck0Gate {
            table: self.table.clone(),
            compact: false,
        };
        let one = RangeCheck1Gate {
            table: self.table.clone(),
        };
        let [v0p0, v0p1] = env.call(&zero, v0);
        let [v1p0, v1p1] = env.call(&zero, v1);
        env.call(&one, (v2, [v0p0, v0p1, v1p0, v1p1]));
    }
}

/// A compact multi-range check: a 176-bit limb `v01` and an 88-bit limb
/// `v2`, as Kimchi's `create_compact_multi_range_check` lays them out:
/// RangeCheck0 of `v2`, RangeCheck0 in compact mode of `v0`, RangeCheck1 of
/// `v1` with `v01` in its column 1, and the Zero row, where `v01 = v0 + 2^88
/// v1`. Input `[v01, v2]`, cells of copied columns; output `[v0, v1]`, the
/// low and middle limbs, in column 0 of rows 1 and 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactMultiRangeCheck<F> {
    /// The 12-bit table.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for CompactMultiRangeCheck<F>
where
    F: PrimeField,
{
    type Input = Array<2>;
    type Output = Array<2>;
    const NAME: &'static str = "kimchi-compact-multi-range-check";
    const ROWS: usize = 4;

    fn synthesize<E>(&self, env: &mut E, [v01, v2]: Of<Array<2>, E::Var>) -> Of<Array<2>, E::Var>
    where
        E: Env<F>,
    {
        let v0 = env.advice(Cell::new(0, 1), [&v01], |[x]| bits(x, 0, LIMB_BITS));
        let v1 = env.advice(Cell::new(0, 2), [&v01], |[x]| {
            bits(x, LIMB_BITS, 2 * LIMB_BITS)
        });
        env.copy(Cell::new(1, 2), &v01);
        let zero = RangeCheck0Gate {
            table: self.table.clone(),
            compact: false,
        };
        let compact = RangeCheck0Gate {
            table: self.table.clone(),
            compact: true,
        };
        let one = RangeCheck1Gate {
            table: self.table.clone(),
        };
        let [v2p0, v2p1] = env.call(&zero, v2);
        let [v0p0, v0p1] = env.call(&compact, v0.clone());
        env.call(&one, (v1.clone(), [v2p0, v2p1, v0p0, v0p1]));
        [v0, v1]
    }
}
