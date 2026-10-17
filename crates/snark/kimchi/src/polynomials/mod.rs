//! Kimchi's gadgets as Kimchi's constructors build them
//! (`kimchi/src/circuits/polynomials`): the rows of gates of each gadget,
//! and its witness, the witness computed by the gadget of
//! `arrabbiata-kimchi-gadgets` on Kimchi's rows.

pub mod and;
pub mod endomul_scalar;
pub mod endosclmul;
pub mod generic;
pub mod not;
pub mod range_check;
pub mod rot;
pub mod xor;

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_gadget::{Error, Of, TypedGadget, assign};
use arrabbiata_kimchi_gadgets::{COEFFICIENTS, LAYOUT};

use crate::{
    constraints::{COLUMNS, CircuitGate, GateType, Wire},
    gates::gate_type_of,
};

/// Why a gadget's rows are not Kimchi rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeriveError {
    /// The gadget does not build.
    Gadget(Error),
    /// A row carries gates of a gadget that is not one of Kimchi's gates.
    UnknownGate(&'static str),
    /// A row carries the gates of two gadgets.
    TwoGates(usize),
    /// A copy cycle leaves the rows asked for.
    CycleOutside(usize),
}

impl From<Error> for DeriveError {
    fn from(e: Error) -> Self {
        Self::Gadget(e)
    }
}

/// A gadget's rows, as Kimchi's constructors would build them, and its
/// witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rows<F> {
    /// The rows.
    pub gates: Vec<CircuitGate<F>>,
    /// The witness, one value per row and column.
    pub witness: [Vec<F>; COLUMNS],
}

/// The first `rows` rows of the circuit `gadget` compiles to, as Kimchi's
/// rows starting at row `offset`, and the witness it computes on `input`:
/// each row's gate type is the gadget whose gates are on it (none: Zero),
/// its coefficients are the fixed columns, its wires the copy permutation,
/// each permuted cell wired to the next cell of its cycle. This is what
/// Kimchi's constructors write by hand.
///
/// # Errors
///
/// A [`DeriveError`].
pub fn derive<F, G, const N: usize, const D: usize>(
    gadget: &G,
    input: &Of<G::Input, F>,
    rows: usize,
    offset: usize,
) -> Result<Rows<F>, DeriveError>
where
    F: Field,
    G: TypedGadget<F>,
{
    let (compiled, _) = assign::<F, G, N, D>(gadget, LAYOUT, input)?;
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
                .ok_or(DeriveError::UnknownGate(name))?;
            if typ.replace(found).is_some_and(|t| t != found) {
                return Err(DeriveError::TwoGates(row));
            }
        }
        let mut coeffs: Vec<F> = circuit.fixed()[..COEFFICIENTS]
            .iter()
            .map(|column| column[row])
            .collect();
        while coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }
        let mut wires = Wire::for_row(row + offset);
        for (col, wire) in wires.iter_mut().enumerate() {
            let target = circuit.sigma()[col * n + row];
            if target % n >= rows {
                return Err(DeriveError::CycleOutside(row));
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
    let witness =
        core::array::from_fn(|c| compiled.columns[c].iter().take(rows).copied().collect());
    Ok(Rows { gates, witness })
}

/// The first `rows` rows of the witness `gadget` computes on `input`, in
/// Kimchi's 15 columns, with `N` column variables and degree `D`, whether
/// or not it satisfies the gadget (as Kimchi's witness functions, which
/// compute and do not check).
///
/// # Errors
///
/// The gadget's structural [`Error`] (a cell written twice, a column
/// outside the layout).
pub fn columns<F, G, const N: usize, const D: usize>(
    gadget: &G,
    input: &Of<G::Input, F>,
    rows: usize,
) -> Result<[Vec<F>; COLUMNS], Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let (compiled, _) = assign::<F, G, N, D>(gadget, LAYOUT, input)?;
    Ok(core::array::from_fn(|c| {
        compiled.columns[c].iter().take(rows).copied().collect()
    }))
}

/// The number of bits of `x`, as an integer, with o1-utils' convention that
/// zero has one bit (`BigUintHelpers::bitlen`): Kimchi sizes its XOR and NOT
/// by it.
#[must_use]
pub fn bit_length<F: arrabbiata_field::PrimeField>(x: F) -> usize {
    (0..F::LIMBS)
        .rev()
        .find(|&i| x.limb(i) != 0)
        .map_or(1, |i| 64 * i + 64 - x.limb(i).leading_zeros() as usize)
}
