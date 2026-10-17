//! Kimchi's AND gadget (`polynomials/and.rs`): `a AND b` from the identity
//! `a + b = (a XOR b) + 2 (a AND b)`, as the XOR of `8 bytes` bits and one
//! double Generic row (`sum = a + b`, then `sum - xor - 2 and = 0`).
//!
//! The rows and the witness are derived from `AndWords`, the XOR then the
//! Generic row, its inputs wired to the XOR's.

use alloc::vec::Vec;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::{
    generic::{GenericRow, GenericSpec},
    xor::{Xor, xor_table},
};

use crate::{
    constraints::{COLUMNS, CircuitGate},
    polynomials::{
        DeriveError, Rows, bit_length, derive,
        xor::{DEGREE, ForBits, VARIABLES, for_bits},
    },
};

/// The first half-gate: `sum = a + b`.
fn sum<F: Field>() -> GenericSpec<F> {
    GenericSpec::Add {
        left: None,
        right: None,
        output: None,
    }
}

/// The second: `sum - xor - 2 and = 0`.
fn and<F: Field>() -> GenericSpec<F> {
    GenericSpec::Add {
        left: None,
        right: Some(-F::ONE),
        output: Some(-(F::ONE + F::ONE)),
    }
}

/// The AND of two words of `B` bits: the words in columns 0 and 1 of row
/// 0, their XOR, then the Generic row `[a, b, sum, sum, xor, and]`, `a` and
/// `b` wired to the XOR's inputs, `sum` to itself across the halves and
/// `xor` to the XOR's output.
struct AndWords<F, const B: usize>(Xor<F, B>);

impl<F, const B: usize> TypedGadget<F> for AndWords<F, B>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-and-words";
    const ROWS: usize = Xor::<F, B>::ROWS + 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        words: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let [a, b] = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&words[i]], |[v]| v));
        let inputs = [a.clone(), b.clone()];
        let xor = env.call(&self.0, inputs);
        let row = Xor::<F, B>::ROWS;
        let at = |column| Cell::new(column, row);
        let output =
            |spec: GenericSpec<F>| move |[l, r]: [F; 2]| spec.output(l, r).unwrap_or(F::ZERO);
        let left = env.copy(at(0), &a);
        let right = env.copy(at(1), &b);
        let total = env.advice(at(2), [&left, &right], output(sum()));
        let again = env.copy(at(3), &total);
        let xored = env.copy(at(4), &xor);
        let conjunction = env.advice(at(5), [&again, &xored], output(and()));
        let generic = GenericRow::new(sum(), Some(and()));
        let cells = [left, right, total, again, xored, conjunction.clone()];
        env.call(&generic, cells);
        conjunction
    }
}

/// The rows and witness of [`AndWords`] on `input`, from row `offset`.
struct AndRows<F> {
    input: [F; 2],
    offset: usize,
}

impl<F: PrimeField> ForBits for AndRows<F> {
    type Output = Result<Rows<F>, DeriveError>;

    fn call<const B: usize>(self) -> Self::Output {
        let gadget = AndWords(Xor::<F, B> { table: xor_table() });
        let rows = AndWords::<F, B>::ROWS;
        derive::<F, _, VARIABLES, DEGREE>(&gadget, &self.input, rows, self.offset)
    }
}

/// The rows and witness of the AND of `input` on `bytes` bytes, from row
/// `offset`.
fn and_rows<F: PrimeField>(
    input: [F; 2],
    bytes: usize,
    offset: usize,
) -> Result<Rows<F>, DeriveError> {
    let rows = AndRows { input, offset };
    for_bits(bytes * 8, rows).unwrap_or(Err(DeriveError::Gadget(Error::Rows {
        gadget: "kimchi-and",
    })))
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// `gates` extended by the AND of words of `bytes` bytes, derived from
    /// `AndWords`: the XOR, then the Generic row whose inputs are wired to
    /// the XOR's (Kimchi's `extend_and`). The number of rows after it.
    ///
    /// # Panics
    ///
    /// If `bytes` is zero, as Kimchi does, or above 32, which our XOR does
    /// not cover.
    pub fn extend_and(gates: &mut Vec<Self>, bytes: usize) -> usize {
        assert!(bytes > 0, "Bytes must be a positive number");
        match and_rows([F::ZERO; 2], bytes, gates.len()) {
            Ok(mut rows) => gates.append(&mut rows.gates),
            Err(e) => panic!("no AND of {bytes} bytes: {e:?}"),
        }
        gates.len()
    }
}

/// The witness of the AND of `input1` and `input2` on `bytes` bytes, from
/// row 0 (Kimchi's `create_and_witness`), computed by `AndWords`: the
/// XOR's rows, then the Generic row `[a, b, sum, sum, xor, and]`.
///
/// # Errors
///
/// [`Error::Rows`] when an input is wider than `8 bytes` bits (Kimchi
/// panics), or when `bytes` is above 32.
pub fn create_and_witness<F: PrimeField>(
    input1: F,
    input2: F,
    bytes: usize,
) -> Result<[Vec<F>; COLUMNS], Error> {
    let wide = Error::Rows {
        gadget: "kimchi-and",
    };
    let bits = bytes * 8;
    if bits < bit_length(input1) || bits < bit_length(input2) {
        return Err(wide);
    }
    and_rows([input1, input2], bytes, 0)
        .map(|rows| rows.witness)
        .map_err(|_| wide)
}

/// `witness` extended by [`create_and_witness`] (Kimchi's
/// `extend_and_witness`).
///
/// # Errors
///
/// As [`create_and_witness`].
pub fn extend_and_witness<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    input1: F,
    input2: F,
    bytes: usize,
) -> Result<(), Error> {
    let rows = create_and_witness(input1, input2, bytes)?;
    for (column, new) in witness.iter_mut().zip(rows) {
        column.extend(new);
    }
    Ok(())
}
