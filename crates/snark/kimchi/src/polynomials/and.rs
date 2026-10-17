//! Kimchi's AND gadget (`polynomials/and.rs`): `a AND b` from the identity
//! `a + b = (a XOR b) + 2 (a AND b)`, as the XOR of `8 bytes` bits and one
//! double Generic row (`sum = a + b`, then `sum - xor - 2 and = 0`).

use alloc::vec::Vec;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::Error;
use arrabbiata_kimchi_gadgets::generic::GenericSpec;

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect, Wire},
    polynomials::{bit_length, generic::GenericGateSpec, xor::create_xor_witness},
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

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// `gates` extended by the AND of words of `bytes` bytes: the XOR, then
    /// the Generic row whose inputs are wired to the XOR's (Kimchi's
    /// `extend_and`). The number of rows after it.
    ///
    /// # Panics
    ///
    /// If `bytes` is zero, as Kimchi does.
    pub fn extend_and(gates: &mut Vec<Self>, bytes: usize) -> usize {
        assert!(bytes > 0, "Bytes must be a positive number");
        let xor_row = gates.len();
        let and_row = Self::extend_xor_gadget(gates, bytes * 8);
        gates.push(Self::create_generic_gadget(
            Wire::for_row(and_row),
            GenericGateSpec::Spec(sum()),
            Some(GenericGateSpec::Spec(and())),
        ));
        gates.connect_cell_pair((xor_row, 0), (and_row, 0));
        gates.connect_cell_pair((xor_row, 1), (and_row, 1));
        gates.connect_cell_pair((and_row, 2), (and_row, 3));
        gates.connect_cell_pair((xor_row, 2), (and_row, 4));
        gates.len()
    }
}

/// The witness of the AND of `input1` and `input2` on `bytes` bytes, from
/// row 0 (Kimchi's `create_and_witness`): our XOR's rows, then the Generic
/// row `[a, b, sum, sum, xor, and]`, `sum` and `and` computed by our
/// half-gates.
///
/// # Errors
///
/// [`Error::Rows`] when an input is wider than `8 bytes` bits (Kimchi
/// panics).
pub fn create_and_witness<F: PrimeField>(
    input1: F,
    input2: F,
    bytes: usize,
) -> Result<[Vec<F>; COLUMNS], Error> {
    let bits = bytes * 8;
    if bits < bit_length(input1) || bits < bit_length(input2) {
        return Err(Error::Rows {
            gadget: "kimchi-and",
        });
    }
    let mut witness = create_xor_witness(input1, input2, bits)?;
    let xor = witness[2][0];
    let output = |spec: GenericSpec<F>, l: F, r: F| match spec.output(l, r) {
        Some(o) => o,
        None => unreachable!("the AND's half-gates have an output"),
    };
    let total = output(sum(), input1, input2);
    let conjunction = output(and(), total, xor);
    let row = [input1, input2, total, total, xor, conjunction];
    for (c, column) in witness.iter_mut().enumerate() {
        column.push(row.get(c).copied().unwrap_or(F::ZERO));
    }
    Ok(witness)
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
