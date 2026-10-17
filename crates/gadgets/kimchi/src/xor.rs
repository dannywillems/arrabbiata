//! Kimchi's Xor16 gate: `in1 XOR in2 = out`, 16 bits per row, by lookups of
//! 4-bit nybbles in the XOR table.
//!
//! The row, as Kimchi lays it out (`in'` is `in >> 16`, on the next row,
//! where the next gate reads it):
//!
//! | 0     | 1     | 2     | 3-6             | 7-10            | 11-14           |
//! | ----- | ----- | ----- | --------------- | --------------- | --------------- |
//! | `in1` | `in2` | `out` | `in1` nybbles   | `in2` nybbles   | `out` nybbles   |
//!
//! nybble 0 being the least significant. The three constraints recombine
//! each word: `sum_k 16^k nybble_k + 2^16 next - curr`. The four lookups are
//! the triples of nybbles `(in1_k, in2_k, out_k)` in the table of all
//! `(i, j, i XOR j)` for `i, j < 16`.
//!
//! [`Xor`] chains `BITS / 16` gates and closes with Kimchi's zero row: a
//! Generic row `w0 = 0` whose three first cells are wired together, so that
//! the words have no bits left. The witness is Kimchi's `xor::init_xor`.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, Scalar, TypedGadget};

use crate::{
    TABLE_COLUMNS,
    generic::{GenericRow, GenericSpec},
};

/// The bits of one row.
pub const BITS_PER_ROW: usize = 16;

/// The table's name.
pub const XOR_TABLE: &str = "kimchi-xor-4";

/// All `(i, j, i XOR j)` for `i, j < 16`, in Kimchi's order (`xor_table`:
/// reversed, so that `(0, 0, 0)` is the last row), in table columns 1-3.
#[must_use]
pub fn xor_table<F: Field>() -> Rom<F> {
    let mut rows = alloc::vec::Vec::new();
    for i in 0..16u64 {
        for j in 0..16u64 {
            rows.push(alloc::vec![
                F::from_u64(i),
                F::from_u64(j),
                F::from_u64(i ^ j)
            ]);
        }
    }
    rows.reverse();
    let columns = alloc::vec![TABLE_COLUMNS[1], TABLE_COLUMNS[2], TABLE_COLUMNS[3]];
    Rom::new(XOR_TABLE, columns, rows)
}

/// The bits `start .. start + count` of `x`.
fn bits<F: PrimeField>(x: F, start: usize, count: usize) -> F {
    let mut value = 0u64;
    for i in (start..start + count).rev() {
        let bit = (x.limb(i / 64) >> (i % 64)) & 1;
        value = (value << 1) | bit;
    }
    F::from_u64(value)
}

/// `x >> 16`, as an integer, over all the field's limbs.
fn shifted<F: PrimeField>(x: F) -> F {
    let two_64 = F::from_u64(1 << 32) * F::from_u64(1 << 32);
    let mut value = F::ZERO;
    for limb in (0..F::LIMBS).rev() {
        let high = if limb + 1 < F::LIMBS {
            x.limb(limb + 1) << (64 - BITS_PER_ROW)
        } else {
            0
        };
        let chunk = (x.limb(limb) >> BITS_PER_ROW) | high;
        value = value * two_64 + F::from_u64(chunk);
    }
    value
}

/// One row: input `[in1, in2, out]`, wired by copies to columns 0-2;
/// output the three words shifted by 16, on the next row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xor16Gate<F> {
    /// The XOR table.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for Xor16Gate<F>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 3>;
    type Output = Array<Scalar, 3>;
    const NAME: &'static str = "kimchi-xor16";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        words: Of<Array<Scalar, 3>, E::Var>,
    ) -> Of<Array<Scalar, 3>, E::Var>
    where
        E: Env<F>,
    {
        let curr: [E::Var; 3] = core::array::from_fn(|i| env.copy(Cell::new(i, 0), &words[i]));
        let nybbles: [[E::Var; 4]; 3] = core::array::from_fn(|w| {
            core::array::from_fn(|k| {
                env.advice(Cell::new(3 + 4 * w + k, 0), [&curr[w]], |[x]| {
                    bits(x, 4 * k, 4)
                })
            })
        });
        let next: [E::Var; 3] =
            core::array::from_fn(|w| env.advice(Cell::new(w, 1), [&curr[w]], |[x]| shifted(x)));
        for w in 0..3 {
            let [n0, n1, n2, n3] = nybbles[w].clone();
            let [p4, p8, p12, p16] = [4u64, 8, 12, 16].map(|e| env.constant(F::from_u64(1 << e)));
            env.assert_zero(
                n0 + n1 * p4 + n2 * p8 + n3 * p12 + p16 * next[w].clone() - curr[w].clone(),
            );
        }
        let [left, right, output] = &nybbles;
        for ((l, r), o) in left.iter().zip(right).zip(output) {
            env.read(&self.table, &[l.clone(), r.clone(), o.clone()]);
        }
        next
    }
}

/// `in1 XOR in2` of `BITS` bits (a multiple of 16), on `BITS / 16 + 1`
/// rows: input `[in1, in2]` (cells of columns 0 and 1), output `out`, in
/// column 2 of the first row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xor<F, const BITS: usize> {
    /// The XOR table.
    pub table: Rom<F>,
}

impl<F, const B: usize> TypedGadget<F> for Xor<F, B>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-xor";
    const ROWS: usize = {
        assert!(
            B.is_multiple_of(BITS_PER_ROW),
            "the words come in rows of 16 bits"
        );
        B / BITS_PER_ROW + 1
    };

    fn synthesize<E>(
        &self,
        env: &mut E,
        [in1, in2]: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let out = env.advice(Cell::new(2, 0), [&in1, &in2], |[a, b]| {
            let mut value = F::ZERO;
            let two_64 = F::from_u64(1 << 32) * F::from_u64(1 << 32);
            for limb in (0..F::LIMBS).rev() {
                value = value * two_64 + F::from_u64(a.limb(limb) ^ b.limb(limb));
            }
            value
        });
        let gate = Xor16Gate {
            table: self.table.clone(),
        };
        let mut words = [in1, in2, out.clone()];
        for _ in 0..B / BITS_PER_ROW {
            words = env.call(&gate, words);
        }
        // The zero row: w0 = 0, and its first three cells wired together.
        let [rest1, rest2, rest_out] = words;
        let zero: [E::Var; 3] = core::array::from_fn(|i| {
            env.advice(Cell::new(3 + i, B / BITS_PER_ROW), [], |[]| F::ZERO)
        });
        let [a, b, c] = zero;
        let row = GenericRow::new(GenericSpec::Const(F::ZERO), None);
        env.call(
            &row,
            [rest1.clone(), rest2.clone(), rest_out.clone(), a, b, c],
        );
        env.connect(&rest1, &rest2);
        env.connect(&rest1, &rest_out);
        out
    }
}
