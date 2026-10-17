//! Kimchi's NOT gadgets (`polynomials/not.rs`): `not x = 2^bits - 1 - x`,
//! either as the XOR of `x` with all ones (its length checked by the XOR's
//! decomposition) or as Generic half-gates `ones - x` (its length unchecked,
//! the inputs assumed to fit).

use alloc::{format, string::String, vec, vec::Vec};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::generic::{Generic, GenericPair, GenericSpec};

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect},
    polynomials::{
        DeriveError, Rows, bit_length, derive,
        xor::{DEGREE, VARIABLES, create_xor_witness},
    },
};

/// The half-gate of the Generic NOT: `l - r = o`.
fn negation<F: Field>() -> GenericSpec<F> {
    GenericSpec::Add {
        left: None,
        right: Some(-F::ONE),
        output: None,
    }
}

/// A row of Generic NOTs, one or two: the inputs `[ones, x]` (and the
/// second pair) placed where the half-gates read them, then our Generic
/// gadgets on them.
#[derive(Clone, Copy, Debug)]
struct NotRow {
    /// Two NOTs, on columns 0-2 and 3-5.
    double_generic: bool,
}

impl<F> TypedGadget<F> for NotRow
where
    F: PrimeField,
{
    type Input = Array<Array<Scalar, 2>, 2>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "kimchi-not-row";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        pairs: Of<Array<Array<Scalar, 2>, 2>, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let mut place = |half: usize| -> [E::Var; 2] {
            core::array::from_fn(|i| {
                env.advice(Cell::new(3 * half + i, 0), [&pairs[half][i]], |[v]| v)
            })
        };
        let first = place(0);
        if self.double_generic {
            let second = place(1);
            let pair = GenericPair {
                first: negation(),
                second: negation(),
            };
            env.call(&pair, (first, second))
        } else {
            let single = Generic { spec: negation() };
            let not = env.call(&single, first);
            [not.clone(), not]
        }
    }
}

/// The Kimchi row of [`NotRow`], at row `offset`.
fn not_row<F: PrimeField>(double_generic: bool, offset: usize) -> Result<Rows<F>, DeriveError> {
    let zeros = [[F::ZERO; 2]; 2];
    let gadget = NotRow { double_generic };
    derive::<F, _, VARIABLES, DEGREE>(&gadget, &zeros, 1, offset)
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// `gates` extended by NOTs of `n` words as Generic half-gates, two per
    /// row, derived from our Generic gadgets, each left input wired to the
    /// all-ones cell `(pub_row, 0)`: the row after them (Kimchi's
    /// `extend_not_gadget_unchecked_length`).
    ///
    /// # Panics
    ///
    /// If `pub_row` is not a row of `gates`.
    pub fn extend_not_gadget_unchecked_length(
        gates: &mut Vec<Self>,
        n: usize,
        pub_row: usize,
    ) -> usize {
        for i in 0..n.div_ceil(2) {
            let double_generic = 2 * i + 1 < n;
            let new_row = gates.len();
            match not_row(double_generic, new_row) {
                Ok(mut rows) => gates.append(&mut rows.gates),
                Err(e) => panic!("no Generic NOT row: {e:?}"),
            }
            gates.join((pub_row, 0), (new_row, 0));
            if double_generic {
                gates.join((pub_row, 0), (new_row, 3));
            }
        }
        gates.len()
    }

    /// `gates` extended by the NOT of a word of `bits` bits as the derived
    /// XOR ([`CircuitGate::extend_xor_gadget`]) with the all-ones cell
    /// `(all_ones_row, 0)`: the number of rows after it (Kimchi's
    /// `extend_not_gadget_checked_length`).
    ///
    /// # Panics
    ///
    /// As [`CircuitGate::extend_xor_gadget`].
    pub fn extend_not_gadget_checked_length(
        gates: &mut Vec<Self>,
        all_ones_row: usize,
        bits: usize,
    ) -> usize {
        let new_row = gates.len();
        let after = Self::extend_xor_gadget(gates, bits);
        gates.join((all_ones_row, 0), (new_row, 1));
        after
    }
}

/// `2^bits - 1`.
fn all_ones<F: Field>(bits: usize) -> F {
    F::from_u64(2).pow(bits as u64) - F::ONE
}

/// `witness` extended by the XOR NOT of `input` on `max(bitlen(input),
/// bits)` bits (Kimchi's `extend_not_witness_checked_length`), computed by
/// our XOR.
///
/// # Errors
///
/// The XOR's [`Error`] beyond 256 bits.
pub fn extend_not_witness_checked_length<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    input: F,
    bits: Option<usize>,
) -> Result<(), Error> {
    let bits = bit_length(input).max(bits.unwrap_or(0));
    let rows = create_xor_witness(input, all_ones(bits), bits)?;
    for (column, new) in witness.iter_mut().zip(rows) {
        column.extend(new);
    }
    Ok(())
}

/// `witness` extended by the Generic NOTs of `inputs` on `bits` bits, two
/// per row (Kimchi's `extend_not_witness_unchecked_length`), each output
/// computed by our Generic half-gate.
///
/// # Errors
///
/// Kimchi's message when an input is wider than `bits`.
pub fn extend_not_witness_unchecked_length<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    inputs: &[F],
    bits: usize,
) -> Result<(), String> {
    for input in inputs {
        let length = bit_length(*input);
        if bits < length {
            return Err(format!(
                "Bits must be greater or equal than the inputs length: {bits} < {length}"
            ));
        }
    }
    let ones = all_ones::<F>(bits);
    let not = |x: F| match negation::<F>().output(ones, x) {
        Some(o) => o,
        None => unreachable!("the NOT half-gate has an output"),
    };
    let rows = inputs.len().div_ceil(2);
    let mut new: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::ZERO; rows]);
    for (row, pair) in inputs.chunks(2).enumerate() {
        new[0][row] = ones;
        new[1][row] = pair[0];
        new[2][row] = not(pair[0]);
        if let Some(second) = pair.get(1) {
            new[3][row] = ones;
            new[4][row] = *second;
            new[5][row] = not(*second);
        }
    }
    for (column, rows) in witness.iter_mut().zip(new) {
        column.extend(rows);
    }
    Ok(())
}
