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
//! - [`endomul`]: the EndoMul gate, four bits per row of an
//!   endomorphism-optimized scalar multiplication;
//! - [`endomul_scalar`]: the EndoMulScalar gate, the scalar EndoMul
//!   multiplies by, eight crumbs per row;
//! - [`foreign_field_add`]: the ForeignFieldAdd gate, `left +/- right`
//!   modulo a foreign modulus on 88-bit limbs, with its bound check
//!   ([`foreign_field`] holds the limb arithmetic of the witness);
//! - [`foreign_field_mul`]: the ForeignFieldMul gate, `left right` modulo
//!   a foreign modulus, checked modulo the native field and modulo `2^264`,
//!   with the external range checks;
//! - [`generic`]: the Generic gate, two PlonK half-gates per row;
//! - [`poseidon`]: the Poseidon gate, five rounds per row;
//! - [`range_check`]: the RangeCheck0 and RangeCheck1 gates, three 88-bit
//!   values on four rows, with 12-bit lookups;
//! - [`rot`]: the Rot64 gate, a 64-bit word rotated by a known offset, with
//!   12-bit lookups and two RangeCheck0 rows;
//! - [`varbasemul`]: the VarBaseMul gate, five steps of a variable-base
//!   scalar multiplication on two rows;
//! - [`xor`]: the Xor16 gate, 16 bits of a XOR per row, with 4-bit lookups;
//! - [`zero`]: the Zero gate, a row with no constraints.
//!
//! Their figures, which `tests/figures.rs` checks against the ones Kimchi
//! itself computes (each gate is one selector; degrees are without it, a
//! cell counting 1). RangeCheck0 also looks up columns 3-6 of its row, and
//! RangeCheck1 columns 3-6 of its row and the next, in the 12-bit table;
//! Xor16 looks up four triples of nybbles in the 4-bit XOR table:
//!
//! | gate        | constraints | max degree | rows | witness cells read        | coefficients |
//! | ----------- | ----------- | ---------- | ---- | ------------------------- | ------------ |
//! | Generic     | 2           | 3          | 1    | 0-5                       | 0-9          |
//! | Poseidon    | 15          | 7          | 1    | 0-14, next row 0-2        | 0-14         |
//! | CompleteAdd | 7           | 3          | 1    | 0-10                      | none         |
//! | EndoMul     | 12          | 4          | 1    | 0-2, 4-14, next row 4-6   | none         |
//! | EndoMulScalar | 11        | 4          | 1    | 0-13                      | none         |
//! | RangeCheck0 | 10          | 4          | 1    | 0-14, next row 0-1        | 0            |
//! | RangeCheck1 | 21          | 4          | 2    | 0, 2-14, next row 0-2, 7-14 | none       |
//! | Xor16       | 3           | 1          | 1    | 0-14, next row 0-2        | none         |
//! | Rot64       | 11          | 4          | 1    | 0-14, next row 0          | 0            |
//! | ForeignFieldAdd | 4       | 3          | 1    | 0-7, next row 0-2         | 0-3          |
//! | ForeignFieldMul | 11      | 4          | 2    | 0-14, next row 0-11       | 0-3          |
//! | VarBaseMul  | 21          | 6          | 2    | 0-5, 7-14, next row 0-11  | none         |

#![no_std]

extern crate alloc;

pub mod complete_add;
pub mod endomul;
pub mod endomul_scalar;
pub mod foreign_field;
pub mod foreign_field_add;
pub mod foreign_field_mul;
pub mod generic;
pub mod poseidon;
pub mod range_check;
pub mod rot;
pub mod varbasemul;
pub mod xor;
pub mod zero;

use arrabbiata_gadget::Layout;

/// The number of witness columns.
pub const COLUMNS: usize = 15;

/// The number of witness columns the permutation covers.
pub const PERMUTS: usize = 7;

/// The number of coefficient (fixed) columns.
pub const COEFFICIENTS: usize = 15;

/// The fixed columns holding the lookup tables, after the coefficients: the
/// 12-bit range table in the first, the 4-bit XOR table in the next three.
/// (Kimchi stores all its tables in the same columns, told apart by a table
/// id; here each table has columns of its own.)
pub const TABLE_COLUMNS: [usize; 4] = [
    COEFFICIENTS,
    COEFFICIENTS + 1,
    COEFFICIENTS + 2,
    COEFFICIENTS + 3,
];

/// Kimchi's table: its witness and coefficient columns, and the columns of
/// its lookup tables.
pub const LAYOUT: Layout = Layout {
    witness: COLUMNS,
    fixed: COEFFICIENTS + TABLE_COLUMNS.len(),
    permuted: PERMUTS,
};
