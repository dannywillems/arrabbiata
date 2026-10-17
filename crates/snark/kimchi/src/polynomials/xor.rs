//! Kimchi's XOR gadget (`polynomials/xor.rs`): `ceil(bits / 16)` Xor16 rows
//! and a Generic row pinning what is left of the words to zero.

use alloc::vec::Vec;

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::xor::{Xor, xor_table};

use crate::{
    constraints::{COLUMNS, CircuitGate},
    polynomials::{DeriveError, Rows, bit_length, derive},
};

/// The Xor16 rows of a XOR of `bits` bits.
#[must_use]
pub const fn num_xors(bits: usize) -> usize {
    bits.div_ceil(16)
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// The rows of a XOR of `bits` bits from row `new_row`, derived from our
    /// XOR gadget: the Xor16 rows and the zero row, its three first cells on
    /// one cycle. The row after them and the rows (Kimchi's
    /// `create_xor_gadget`, which leaves the zero row's cells to
    /// `extend_xor_gadget`).
    ///
    /// # Panics
    ///
    /// Beyond 256 bits, which our XOR does not cover.
    #[must_use]
    pub fn create_xor_gadget(new_row: usize, bits: usize) -> (usize, Vec<Self>) {
        match xor_rows([F::ZERO; 2], bits, new_row) {
            Ok(rows) => (new_row + rows.gates.len(), rows.gates),
            Err(e) => panic!("no XOR of {bits} bits: {e:?}"),
        }
    }

    /// `gates` extended by a XOR of `bits` bits: the number of rows after it
    /// (Kimchi's `extend_xor_gadget`).
    ///
    /// # Panics
    ///
    /// As [`CircuitGate::create_xor_gadget`].
    pub fn extend_xor_gadget(gates: &mut Vec<Self>, bits: usize) -> usize {
        let (_, mut xor) = Self::create_xor_gadget(gates.len(), bits);
        gates.append(&mut xor);
        gates.len()
    }
}

/// The inputs in columns 0 and 1 of row 0, then the XOR gadget.
#[derive(Clone, Debug)]
struct Words<F, const BITS: usize>(Xor<F, BITS>);

impl<F, const B: usize> TypedGadget<F> for Words<F, B>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-xor-words";
    const ROWS: usize = Xor::<F, B>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        words: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&words[i]], |[v]| v));
        env.call(&self.0, cells)
    }
}

/// The column variables of the XOR's rows.
const VARIABLES: usize = 32;
/// The Generic row's `c3 w0 w1`.
const DEGREE: usize = 3;

/// The rows and witness of [`Words`] of `B` bits, from row `offset`.
fn words<F: PrimeField, const B: usize>(
    input: [F; 2],
    offset: usize,
) -> Result<Rows<F>, DeriveError> {
    let gadget = Words(Xor::<F, B> { table: xor_table() });
    derive::<F, _, VARIABLES, DEGREE>(&gadget, &input, Xor::<F, B>::ROWS, offset)
}

/// The rows and witness of a XOR of `bits` bits of `input`, from row
/// `offset`.
fn xor_rows<F: PrimeField>(
    input: [F; 2],
    bits: usize,
    offset: usize,
) -> Result<Rows<F>, DeriveError> {
    match num_xors(bits) {
        0 => words::<F, 0>(input, offset),
        1 => words::<F, 16>(input, offset),
        2 => words::<F, 32>(input, offset),
        3 => words::<F, 48>(input, offset),
        4 => words::<F, 64>(input, offset),
        5 => words::<F, 80>(input, offset),
        6 => words::<F, 96>(input, offset),
        7 => words::<F, 112>(input, offset),
        8 => words::<F, 128>(input, offset),
        9 => words::<F, 144>(input, offset),
        10 => words::<F, 160>(input, offset),
        11 => words::<F, 176>(input, offset),
        12 => words::<F, 192>(input, offset),
        13 => words::<F, 208>(input, offset),
        14 => words::<F, 224>(input, offset),
        15 => words::<F, 240>(input, offset),
        16 => words::<F, 256>(input, offset),
        _ => Err(DeriveError::Gadget(Error::Rows {
            gadget: "kimchi-xor",
        })),
    }
}

/// The witness of a XOR of `bits` bits of `input1` and `input2`, from
/// row 0 (Kimchi's `create_xor_witness`), computed by the gadget of
/// `arrabbiata-kimchi-gadgets`.
///
/// # Errors
///
/// [`Error::Rows`] when an input is wider than `bits` (Kimchi panics), or
/// when `bits` is above 256.
pub fn create_xor_witness<F: PrimeField>(
    input1: F,
    input2: F,
    bits: usize,
) -> Result<[Vec<F>; COLUMNS], Error> {
    let wide = Error::Rows {
        gadget: "kimchi-xor",
    };
    if bits < bit_length(input1) || bits < bit_length(input2) {
        return Err(wide);
    }
    xor_rows([input1, input2], bits, 0)
        .map(|rows| rows.witness)
        .map_err(|_| wide)
}

/// `witness` extended by [`create_xor_witness`] (Kimchi's
/// `extend_xor_witness`).
///
/// # Errors
///
/// As [`create_xor_witness`].
pub fn extend_xor_witness<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    input1: F,
    input2: F,
    bits: usize,
) -> Result<(), Error> {
    let xor = create_xor_witness(input1, input2, bits)?;
    for (column, rows) in witness.iter_mut().zip(xor) {
        column.extend(rows);
    }
    Ok(())
}
