//! Kimchi's gates (o1-labs/proof-systems, crate `kimchi`) as typed gadgets.
//!
//! Kimchi's table has 15 witness columns, of which the first 7 can be
//! copied between, and 15 coefficient columns ([`LAYOUT`]). Each gate here
//! writes the cells Kimchi's gate of the same name constrains, in the same
//! columns, with the same constraints; its coefficients are the fixed
//! columns.
//!
//! - [`complete_add`]: the CompleteAdd gate, the complete sum of two affine
//!   points;
//! - [`generic`]: the Generic gate, two PlonK half-gates per row;
//! - [`poseidon`]: the Poseidon gate, five rounds per row.

#![no_std]

pub mod complete_add;
pub mod generic;
pub mod poseidon;

use arrabbiata_gadget::Layout;

/// The number of witness columns.
pub const COLUMNS: usize = 15;

/// The number of witness columns the permutation covers.
pub const PERMUTS: usize = 7;

/// The number of coefficient (fixed) columns.
pub const COEFFICIENTS: usize = 15;

/// Kimchi's table.
pub const LAYOUT: Layout = Layout {
    witness: COLUMNS,
    fixed: COEFFICIENTS,
    permuted: PERMUTS,
};
