//! Kimchi's gadgets as Kimchi's constructors build them
//! (`kimchi/src/circuits/polynomials`): the rows of gates of each gadget,
//! and its witness, the witness computed by the gadget of
//! `arrabbiata-kimchi-gadgets` on Kimchi's rows.

pub mod and;
pub mod generic;
pub mod not;
pub mod range_check;
pub mod rot;
pub mod xor;

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_gadget::{Error, Of, TypedGadget, assign};
use arrabbiata_kimchi_gadgets::LAYOUT;

use crate::constraints::COLUMNS;

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
