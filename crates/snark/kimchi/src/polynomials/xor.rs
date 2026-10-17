//! Kimchi's XOR gadget (`polynomials/xor.rs`): `ceil(bits / 16)` Xor16 rows
//! and a Generic row pinning what is left of the words to zero.

use alloc::{vec, vec::Vec};

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::{
    generic::GenericSpec,
    xor::{Xor, xor_table},
};

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect, GateType, Wire},
    polynomials::{columns, generic::GenericGateSpec},
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
    /// The rows of a XOR of `bits` bits from row `new_row`: the row after
    /// them and the rows (Kimchi's `create_xor_gadget`).
    #[must_use]
    pub fn create_xor_gadget(new_row: usize, bits: usize) -> (usize, Vec<Self>) {
        let num_xors = num_xors(bits);
        let mut gates: Vec<Self> = (0..num_xors)
            .map(|i| Self::new(GateType::Xor16, Wire::for_row(new_row + i), vec![]))
            .collect();
        let zero_row = new_row + num_xors;
        gates.push(Self::create_generic_gadget(
            Wire::for_row(zero_row),
            GenericGateSpec::Spec(GenericSpec::Const(F::ZERO)),
            None,
        ));
        (new_row + gates.len(), gates)
    }

    /// `gates` extended by a XOR of `bits` bits whose last row's three
    /// words are wired together: the number of rows after it (Kimchi's
    /// `extend_xor_gadget`).
    pub fn extend_xor_gadget(gates: &mut Vec<Self>, bits: usize) -> usize {
        let new_row = gates.len();
        let (_, mut xor) = Self::create_xor_gadget(new_row, bits);
        gates.append(&mut xor);
        let zero_row = gates.len() - 1;
        gates.connect_cell_pair((zero_row, 0), (zero_row, 1));
        gates.connect_cell_pair((zero_row, 0), (zero_row, 2));
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
    type Input = Array<2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-xor-words";
    const ROWS: usize = Xor::<F, B>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        words: Of<Array<2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&words[i]], |[v]| v));
        env.call(&self.0, cells)
    }
}

/// The column variables of the XOR's rows.
const VARIABLES: usize = 32;
/// The Generic row's `c3 w0 w1`.
const DEGREE: usize = 3;

/// The witness of [`Words`] of `B` bits.
fn words<F: PrimeField, const B: usize>(input: [F; 2]) -> Result<[Vec<F>; COLUMNS], Error> {
    let gadget = Words(Xor::<F, B> { table: xor_table() });
    columns::<F, _, VARIABLES, DEGREE>(&gadget, &input, Xor::<F, B>::ROWS)
}

/// The witness of a XOR of `bits` bits of `input1` and `input2`, from
/// row 0 (Kimchi's `create_xor_witness`), computed by the gadget of
/// `arrabbiata-kimchi-gadgets`.
///
/// # Errors
///
/// The gadget's [`Error`] when an input is wider than `bits` (Kimchi
/// panics), or when `bits` is above 128.
pub fn create_xor_witness<F: PrimeField>(
    input1: F,
    input2: F,
    bits: usize,
) -> Result<[Vec<F>; COLUMNS], Error> {
    let input = [input1, input2];
    match num_xors(bits) {
        0 | 1 => words::<F, 16>(input),
        2 => words::<F, 32>(input),
        3 => words::<F, 48>(input),
        4 => words::<F, 64>(input),
        5 => words::<F, 80>(input),
        6 => words::<F, 96>(input),
        7 => words::<F, 112>(input),
        8 => words::<F, 128>(input),
        _ => Err(Error::Rows {
            gadget: "kimchi-xor",
        }),
    }
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
