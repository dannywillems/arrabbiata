//! Kimchi's rows and columns, read off a compiled typed gadget.
//!
//! A typed gadget is one program over the environment of
//! `arrabbiata-gadget`, run twice: symbolically, it compiles to a circuit
//! (constraints over the cells, the copy partition, the lookups, the fixed
//! columns), which [`arrabbiata_gadget::compile`] returns with the gadget
//! kind of each selector; numerically, on an input, it computes the
//! witness, which [`arrabbiata_gadget::assign`] returns. The two agree by
//! construction (evaluation at the witness is a homomorphism the program
//! commutes with).
//!
//! Kimchi writes the same circuit as rows of gates. That is a presentation
//! of the compiled circuit, and this module is the map: each row's gate
//! type is the kind of the selector on it (none: Zero), its coefficients
//! are the fixed columns, and its wires are the copy permutation, each
//! permuted cell wired to the next cell of its cycle; a public input, a
//! binding of the compiled circuit, is Kimchi's Generic row with
//! coefficient 1. Two presentations of
//! one circuit agree up to the order inside each cycle, which the
//! permutation argument does not see.

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_gadget::{Compiled, Error, Of, TypedGadget, assign, compile};
use arrabbiata_kimchi_gadgets::{COEFFICIENTS, LAYOUT};

use crate::{
    constraints::{COLUMNS, CircuitGate, GateType, Wire},
    gates::gate_type_of,
};

/// Why a compiled circuit is not Kimchi rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowsError {
    /// The gadget does not compile.
    Gadget(Error),
    /// A row carries gates of a gadget that is not one of Kimchi's gates.
    UnknownGate(&'static str),
    /// A row carries the gates of two gadgets.
    TwoGates(usize),
    /// A copy cycle leaves the rows asked for.
    CycleOutside(usize),
    /// A public input on a row that carries a gate.
    PublicRowTaken(usize),
}

impl From<Error> for RowsError {
    fn from(e: Error) -> Self {
        Self::Gadget(e)
    }
}

/// The first `rows` rows of `compiled` as Kimchi's rows, numbered from
/// `offset`.
///
/// # Errors
///
/// A [`RowsError`].
pub fn rows<F, const N: usize, const D: usize>(
    compiled: &Compiled<F, N, D>,
    rows: usize,
    offset: usize,
) -> Result<Vec<CircuitGate<F>>, RowsError>
where
    F: Field,
{
    let circuit = &compiled.circuit;
    let n = circuit.num_rows();
    let mut gates = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut typ = None;
        for (k, selector) in circuit.selectors().iter().enumerate() {
            if selector[row].is_zero() {
                continue;
            }
            let (name, at) = compiled.kinds[k];
            let found = gate_type_of(name)
                .filter(|_| at == 0)
                .ok_or(RowsError::UnknownGate(name))?;
            if typ.replace(found).is_some_and(|t| t != found) {
                return Err(RowsError::TwoGates(row));
            }
        }
        let mut coeffs: Vec<F> = circuit.fixed()[..COEFFICIENTS]
            .iter()
            .map(|column| column[row])
            .collect();
        // A public input is a binding of the compiled circuit; Kimchi
        // presents it as a Generic row `w0 - p = 0`, coefficient 1.
        if circuit.bindings().iter().any(|(_, bound)| *bound == row) {
            if typ.is_some() {
                return Err(RowsError::PublicRowTaken(row));
            }
            typ = Some(GateType::Generic);
            coeffs[0] = F::ONE;
        }
        while coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }
        let mut wires = Wire::for_row(row + offset);
        for (col, wire) in wires.iter_mut().enumerate() {
            let target = circuit.sigma()[col * n + row];
            if target % n >= rows {
                return Err(RowsError::CycleOutside(row));
            }
            *wire = Wire {
                row: target % n + offset,
                col: target / n,
            };
        }
        gates.push(CircuitGate::new(
            typ.unwrap_or(GateType::Zero),
            wires,
            coeffs,
        ));
    }
    Ok(gates)
}

/// The first `rows` rows of the witness of `compiled`, in Kimchi's 15
/// columns.
#[must_use]
pub fn columns<F, const N: usize, const D: usize>(
    compiled: &Compiled<F, N, D>,
    rows: usize,
) -> [Vec<F>; COLUMNS]
where
    F: Field,
{
    core::array::from_fn(|c| compiled.columns[c].iter().take(rows).copied().collect())
}

/// The first `rows` rows `gadget` compiles to, numbered from `offset`: the
/// symbolic run alone, with `N` column variables and degree `D`.
///
/// # Errors
///
/// A [`RowsError`].
pub fn circuit_rows<F, G, const N: usize, const D: usize>(
    gadget: &G,
    rows: usize,
    offset: usize,
) -> Result<Vec<CircuitGate<F>>, RowsError>
where
    F: Field,
    G: TypedGadget<F>,
{
    let compiled = compile::<F, G, N, D>(gadget, LAYOUT)?;
    self::rows(&compiled, rows, offset)
}

/// The first `rows` rows of the witness `gadget` computes on `input`, and
/// its output: the numeric run, whether or not the witness satisfies the
/// circuit (as Kimchi's witness generators, which compute and do not
/// check).
///
/// # Errors
///
/// The gadget's structural [`Error`].
#[allow(clippy::type_complexity)]
pub fn witness_columns<F, G, const N: usize, const D: usize>(
    gadget: &G,
    input: &Of<G::Input, F>,
    rows: usize,
) -> Result<([Vec<F>; COLUMNS], Of<G::Output, F>), Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let (compiled, output) = assign::<F, G, N, D>(gadget, LAYOUT, input)?;
    Ok((columns(&compiled, rows), output))
}
