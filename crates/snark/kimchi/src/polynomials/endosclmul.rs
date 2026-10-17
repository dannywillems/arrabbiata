//! Kimchi's EndoMul witness (`polynomials/endosclmul.rs`): an
//! endomorphism-optimized scalar multiplication, four bits per row, the
//! result on the row after, computed by our EndoMul gates.

use alloc::vec::Vec;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget, assign};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    endomul::{EndoMul, column},
};

use crate::{
    constraints::COLUMNS,
    polynomials::xor::{ForBits, for_bits},
};

/// The result of a multiplication (Kimchi's `EndoMulResult`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoMulResult<F> {
    /// The accumulator after the last row, `(x, y)`.
    pub acc: (F, F),
    /// The bits as an integer, most significant first.
    pub n: F,
}

/// `T` in columns 0-1 and `P` in columns 4-5 of the first row, then our
/// EndoMul chain over `B` bits.
#[derive(Clone, Copy, Debug)]
struct Placed<F, const B: usize>(EndoMul<F, B>);

impl<F, const B: usize> TypedGadget<F> for Placed<F, B>
where
    F: Field,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Array<Scalar, B>);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-endomul-placed";
    const ROWS: usize = EndoMul::<F, B>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (t, p, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let t =
            core::array::from_fn(|i| env.advice(Cell::new(column::XT + i, 0), [&t[i]], |[v]| v));
        let p =
            core::array::from_fn(|i| env.advice(Cell::new(column::XP + i, 0), [&p[i]], |[v]| v));
        env.call(&self.0, (t, p, bits))
    }
}

/// The column variables of the rows.
const VARIABLES: usize = 17;
/// The degree of the EndoMul constraints.
const DEGREE: usize = 4;

/// The columns Kimchi's `gen_witness` writes on a row of the chain (all but
/// column 3), and on the row after it (`xS`, `yS`, `n`).
const WRITTEN: [usize; 14] = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
const AFTER: [usize; 3] = [4, 5, 6];

/// The rows of a multiplication over `B` bits.
struct Rows<'a, F> {
    endo: F,
    base: (F, F),
    bits: &'a [bool],
    acc0: (F, F),
}

/// The rows, the result, or the gadget's error.
type Computed<F> = Result<([Vec<F>; COLUMNS], EndoMulResult<F>), Error>;

impl<F: PrimeField> ForBits for Rows<'_, F> {
    type Output = Computed<F>;

    fn call<const B: usize>(self) -> Self::Output {
        let bits: [F; B] = core::array::from_fn(|i| F::from_u64(u64::from(self.bits[i])));
        let input = ([self.base.0, self.base.1], [self.acc0.0, self.acc0.1], bits);
        let gadget = Placed(EndoMul::<F, B> { endo: self.endo });
        let (compiled, ([x, y], n)) = assign::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input)?;
        let rows = B / 4 + 1;
        let columns =
            core::array::from_fn(|c| compiled.columns[c].iter().take(rows).copied().collect());
        Ok((columns, EndoMulResult { acc: (x, y), n }))
    }
}

/// `w` written from row `row0` with the multiplication of `base` by `bits`,
/// most significant first, from the accumulator `acc0`: `bits / 4` EndoMul
/// rows and `xS`, `yS`, `n` on the row after, the cells Kimchi's
/// `endosclmul::gen_witness` writes. The result.
///
/// # Errors
///
/// [`Error::Rows`] unless the bits are a multiple of 16 up to 256 (Kimchi
/// asks for a multiple of 4 and panics otherwise), or when `w` has fewer
/// than `row0 + bits / 4 + 1` rows.
pub fn gen_witness<F: PrimeField>(
    w: &mut [Vec<F>; COLUMNS],
    row0: usize,
    endo: F,
    base: (F, F),
    bits: &[bool],
    acc0: (F, F),
) -> Result<EndoMulResult<F>, Error> {
    let unsupported = Error::Rows {
        gadget: "kimchi-endomul",
    };
    let chain = bits.len() / 4;
    if !bits.len().is_multiple_of(16) || w.iter().any(|c| c.len() <= row0 + chain) {
        return Err(unsupported);
    }
    let rows = Rows {
        endo,
        base,
        bits,
        acc0,
    };
    let (columns, result) = for_bits(bits.len(), rows).ok_or(unsupported)??;
    for r in 0..chain {
        for c in WRITTEN {
            w[c][row0 + r] = columns[c][r];
        }
    }
    for c in AFTER {
        w[c][row0 + chain] = columns[c][chain];
    }
    Ok(result)
}
