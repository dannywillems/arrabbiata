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
//! - [`poseidon`]: the Poseidon gate, five rounds per row;
//! - [`varbasemul`]: the VarBaseMul gate, five steps of a variable-base
//!   scalar multiplication on two rows.
//!
//! Their figures, which `tests/figures.rs` checks against the ones Kimchi
//! itself computes (each gate is one selector; degrees are without it, a
//! cell counting 1):
//!
//! | gate        | constraints | max degree | rows | witness cells read        | coefficients |
//! | ----------- | ----------- | ---------- | ---- | ------------------------- | ------------ |
//! | Generic     | 2           | 3          | 1    | 0-5                       | 0-9          |
//! | Poseidon    | 15          | 7          | 1    | 0-14, next row 0-2        | 0-14         |
//! | CompleteAdd | 7           | 3          | 1    | 0-10                      | none         |
//! | VarBaseMul  | 21          | 6          | 2    | 0-5, 7-14, next row 0-11  | none         |

#![no_std]

pub mod complete_add;
pub mod generic;
pub mod poseidon;
pub mod varbasemul;

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
