//! Kimchi's gadgets as Kimchi's constructors build them
//! (`kimchi/src/circuits/polynomials`): the rows of gates of each gadget,
//! and its witness, the witness computed by the gadget of
//! `arrabbiata-kimchi-gadgets` on Kimchi's rows.

pub mod generic;
pub mod rot;
pub mod xor;

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_gadget::{Error, Of, TypedGadget, witness};
use arrabbiata_kimchi_gadgets::LAYOUT;

use crate::constraints::COLUMNS;

/// The first `rows` rows of the witness `gadget` computes on `input`, in
/// Kimchi's 15 columns, with `N` column variables and degree `D`.
///
/// # Errors
///
/// The gadget's [`Error`], when its witness does not satisfy it.
pub fn columns<F, G, const N: usize, const D: usize>(
    gadget: &G,
    input: &Of<G::Input, F>,
    rows: usize,
) -> Result<[Vec<F>; COLUMNS], Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let (compiled, _) = witness::<F, G, N, D>(gadget, LAYOUT, input)?;
    Ok(core::array::from_fn(|c| {
        compiled.columns[c].iter().take(rows).copied().collect()
    }))
}
