//! Kimchi's NOT gadgets (`polynomials/not.rs`): `not x = 2^bits - 1 - x`,
//! either as the XOR of `x` with all ones (its length checked by the XOR's
//! decomposition) or as Generic half-gates `ones - x` (its length unchecked,
//! the inputs assumed to fit).

use alloc::{format, string::String, vec, vec::Vec};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::Error;
use arrabbiata_kimchi_gadgets::generic::GenericSpec;

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect, GateType, Wire},
    polynomials::{
        bit_length,
        generic::GenericGateSpec,
        xor::{create_xor_witness, num_xors},
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

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// `gates` extended by NOTs of `n` words as Generic half-gates, two per
    /// row, each left input wired to the all-ones cell `(pub_row, 0)`: the
    /// row after them (Kimchi's `extend_not_gadget_unchecked_length`).
    pub fn extend_not_gadget_unchecked_length(
        gates: &mut Vec<Self>,
        n: usize,
        pub_row: usize,
    ) -> usize {
        let mut new_row = gates.len();
        for _ in 0..n / 2 {
            new_row = Self::extend_not_gnrc(gates, pub_row, new_row, true);
        }
        if n % 2 == 1 {
            new_row = Self::extend_not_gnrc(gates, pub_row, new_row, false);
        }
        new_row
    }

    /// One Generic NOT row, one or two half-gates.
    fn extend_not_gnrc(
        gates: &mut Vec<Self>,
        pub_row: usize,
        new_row: usize,
        double_generic: bool,
    ) -> usize {
        let first = GenericGateSpec::Spec(negation());
        let second = double_generic.then(|| GenericGateSpec::Spec(negation()));
        gates.push(Self::create_generic_gadget(
            Wire::for_row(new_row),
            first,
            second,
        ));
        gates.connect_cell_pair((pub_row, 0), (new_row, 0));
        if double_generic {
            gates.connect_cell_pair((pub_row, 0), (new_row, 3));
        }
        new_row + 1
    }

    /// `gates` extended by the NOT of a word of `bits` bits as a XOR with
    /// the all-ones cell `(all_ones_row, 0)`: the number of rows after it
    /// (Kimchi's `extend_not_gadget_checked_length`).
    pub fn extend_not_gadget_checked_length(
        gates: &mut Vec<Self>,
        all_ones_row: usize,
        bits: usize,
    ) -> usize {
        let n = num_xors(bits);
        let new_row = gates.len();
        let mut rows: Vec<Self> = (0..n)
            .map(|i| Self::new(GateType::Xor16, Wire::for_row(new_row + i), vec![]))
            .collect();
        let zero_row = new_row + n;
        rows.push(Self::create_generic_gadget(
            Wire::for_row(zero_row),
            GenericGateSpec::Spec(GenericSpec::Const(F::ZERO)),
            None,
        ));
        gates.extend(rows);
        gates.connect_cell_pair((zero_row, 0), (zero_row, 1));
        gates.connect_cell_pair((zero_row, 0), (zero_row, 2));
        gates.connect_cell_pair((all_ones_row, 0), (new_row, 1));
        gates.len()
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
