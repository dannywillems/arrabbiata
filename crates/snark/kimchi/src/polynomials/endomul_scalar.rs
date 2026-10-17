//! Kimchi's EndoMulScalar witness (`polynomials/endomul_scalar.rs`): the
//! rows of a scalar challenge of `num_bits` bits, eight crumbs per row, and
//! the scalar `a lambda + b` it stands for.

use alloc::vec::Vec;

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Error, assign};
use arrabbiata_kimchi_gadgets::{LAYOUT, endomul_scalar::EndoMulScalar};

use crate::constraints::COLUMNS;

/// The column variables of the rows.
const VARIABLES: usize = 32;
/// The crumb checks' degree.
const DEGREE: usize = 4;

/// The rows and the output `[n, a, b]` of `C` crumbs.
fn rows<F: PrimeField, const C: usize>(crumbs: &[F]) -> Result<([Vec<F>; COLUMNS], [F; 3]), Error> {
    let input: [F; C] = core::array::from_fn(|i| crumbs[i]);
    let (compiled, out) = assign::<F, _, VARIABLES, DEGREE>(&EndoMulScalar::<C>, LAYOUT, &input)?;
    let n = C / 8;
    let columns = core::array::from_fn(|c| compiled.columns[c].iter().take(n).copied().collect());
    Ok((columns, out))
}

/// `witness` extended by the rows of the scalar challenge `scalar`, its low
/// `num_bits` bits read two at a time from the most significant, and the
/// scalar it stands for, `a endo_scalar + b` (Kimchi's
/// `endomul_scalar::gen_witness`), computed by our EndoMulScalar gates.
///
/// # Errors
///
/// [`Error::Rows`] unless `num_bits` is a multiple of 16 up to 256 (Kimchi
/// asserts the multiple).
pub fn gen_witness<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    scalar: F,
    endo_scalar: F,
    num_bits: usize,
) -> Result<F, Error> {
    let bit = |i: usize| (scalar.limb(i / 64) >> (i % 64)) & 1;
    let crumbs: Vec<F> = (0..num_bits / 2)
        .rev()
        .map(|k| F::from_u64(2 * bit(2 * k + 1) + bit(2 * k)))
        .collect();
    let (columns, [_, a, b]) = match num_bits {
        16 => rows::<F, 8>(&crumbs),
        32 => rows::<F, 16>(&crumbs),
        48 => rows::<F, 24>(&crumbs),
        64 => rows::<F, 32>(&crumbs),
        80 => rows::<F, 40>(&crumbs),
        96 => rows::<F, 48>(&crumbs),
        112 => rows::<F, 56>(&crumbs),
        128 => rows::<F, 64>(&crumbs),
        144 => rows::<F, 72>(&crumbs),
        160 => rows::<F, 80>(&crumbs),
        176 => rows::<F, 88>(&crumbs),
        192 => rows::<F, 96>(&crumbs),
        208 => rows::<F, 104>(&crumbs),
        224 => rows::<F, 112>(&crumbs),
        240 => rows::<F, 120>(&crumbs),
        256 => rows::<F, 128>(&crumbs),
        _ => Err(Error::Rows {
            gadget: "kimchi-endomul-scalar",
        }),
    }?;
    for (column, new) in witness.iter_mut().zip(columns) {
        column.extend(new);
    }
    Ok(a * endo_scalar + b)
}
