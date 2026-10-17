//! Kimchi's VarBaseMul witness (`polynomials/varbasemul.rs`): a
//! variable-base scalar multiplication, five bits per two rows, computed by
//! our VarBaseMul gates.

use alloc::vec::Vec;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget, assign};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    varbasemul::{ACCUMULATORS, BITS_PER_GATE, ROWS_PER_GATE, VarBaseMul},
};

use crate::constraints::COLUMNS;

/// The result of a multiplication (Kimchi's `VarbaseMulResult`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarbaseMulResult<F> {
    /// The accumulator after the last step, `(x, y)`.
    pub acc: (F, F),
    /// The bits as an integer, most significant first.
    pub n: F,
}

/// `T` in columns 0-1 and the accumulator in columns 2-3 of the first row,
/// then our VarBaseMul chain over `B` bits.
#[derive(Clone, Copy, Debug)]
struct Placed<const B: usize>;

impl<F, const B: usize> TypedGadget<F> for Placed<B>
where
    F: Field,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Array<Scalar, B>);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-varbasemul-placed";
    const ROWS: usize = <VarBaseMul<B> as TypedGadget<F>>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (t, a, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let (_, x0) = ACCUMULATORS[0];
        let t = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&t[i]], |[v]| v));
        let a = core::array::from_fn(|i| env.advice(Cell::new(x0 + i, 0), [&a[i]], |[v]| v));
        env.call(&VarBaseMul::<B>, (t, a, bits))
    }
}

/// The column variables of the rows.
const VARIABLES: usize = 26;
/// The degree of the VarBaseMul constraints.
const DEGREE: usize = 6;

/// The rows, the result, or the gadget's error.
type Computed<F> = Result<([Vec<F>; COLUMNS], VarbaseMulResult<F>), Error>;

/// The rows of a multiplication over `B` bits.
fn rows<F: PrimeField, const B: usize>(base: (F, F), bits: &[bool], acc0: (F, F)) -> Computed<F> {
    let bits: [F; B] = core::array::from_fn(|i| F::from_u64(u64::from(bits[i])));
    let input = ([base.0, base.1], [acc0.0, acc0.1], bits);
    let (compiled, ([x, y], n)) = assign::<F, _, VARIABLES, DEGREE>(&Placed::<B>, LAYOUT, &input)?;
    let rows = ROWS_PER_GATE * (B / BITS_PER_GATE);
    let columns =
        core::array::from_fn(|c| compiled.columns[c].iter().take(rows).copied().collect());
    Ok((columns, VarbaseMulResult { acc: (x, y), n }))
}

/// `w` written from row `row0` with the multiplication of `base` by `bits`,
/// most significant first, from the accumulator `acc0`: two rows per five
/// bits (Kimchi's `varbasemul::witness`; the cells Kimchi leaves alone, six
/// on each pair of rows, are written as zero). The result.
///
/// # Errors
///
/// [`Error::Rows`] for a bit length other than a multiple of 5 up to 60, or
/// 255 (Pasta's), or when `w` has too few rows from `row0`.
pub fn witness<F: PrimeField>(
    w: &mut [Vec<F>; COLUMNS],
    row0: usize,
    base: (F, F),
    bits: &[bool],
    acc0: (F, F),
) -> Result<VarbaseMulResult<F>, Error> {
    let unsupported = Error::Rows {
        gadget: "kimchi-varbasemul",
    };
    let height = ROWS_PER_GATE * (bits.len() / BITS_PER_GATE);
    if w.iter().any(|c| c.len() < row0 + height) {
        return Err(unsupported);
    }
    let (columns, result) = match bits.len() {
        5 => rows::<F, 5>(base, bits, acc0),
        10 => rows::<F, 10>(base, bits, acc0),
        15 => rows::<F, 15>(base, bits, acc0),
        20 => rows::<F, 20>(base, bits, acc0),
        25 => rows::<F, 25>(base, bits, acc0),
        30 => rows::<F, 30>(base, bits, acc0),
        35 => rows::<F, 35>(base, bits, acc0),
        40 => rows::<F, 40>(base, bits, acc0),
        45 => rows::<F, 45>(base, bits, acc0),
        50 => rows::<F, 50>(base, bits, acc0),
        55 => rows::<F, 55>(base, bits, acc0),
        60 => rows::<F, 60>(base, bits, acc0),
        255 => rows::<F, 255>(base, bits, acc0),
        _ => Err(unsupported),
    }?;
    for (column, new) in w.iter_mut().zip(columns) {
        column[row0..row0 + height].copy_from_slice(&new);
    }
    Ok(result)
}
