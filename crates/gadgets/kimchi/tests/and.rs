//! Kimchi's AND gadget: the XOR of the words and one double Generic row.
//!
//! - for random words of 16 and 64 bits, the witness satisfies the
//!   circuit and the output is `a AND b`;
//! - an AND output off by one breaks the Generic row;
//! - placed by [`Placed`], the words are written into columns 0 and 1 of
//!   the first row before the gadget runs.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Cell, Placed, witness};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    and::And,
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::Fp;
use proptest::prelude::*;

/// The column variables: every cell of a row, the first three of the next,
/// and the Generic row's coefficients.
const VARIABLES: usize = 32;
/// The Generic row's `c3 w0 w1`.
const DEGREE: usize = 3;

/// The AND of two words of `B` bits in columns 0 and 1 of the first row.
fn and<const B: usize>() -> Placed<And<Fp, B>> {
    Placed {
        gadget: And {
            xor: Xor { table: xor_table() },
        },
        cells: vec![Cell::new(0, 0), Cell::new(1, 0)],
    }
}

/// `x` in `Fp`.
fn fp(x: u64) -> Fp {
    Fp::from_u64(x)
}

proptest! {
    #[test]
    fn the_output_is_the_and(a in any::<u16>(), b in any::<u16>(), c in any::<u64>(), d in any::<u64>()) {
        let input = [fp(u64::from(a)), fp(u64::from(b))];
        let (_, output) = witness::<Fp, _, VARIABLES, DEGREE>(&and::<16>(), LAYOUT, &input).unwrap();
        prop_assert_eq!(output, fp(u64::from(a & b)));
        let (_, output) = witness::<Fp, _, VARIABLES, DEGREE>(&and::<64>(), LAYOUT, &[fp(c), fp(d)]).unwrap();
        prop_assert_eq!(output, fp(c & d));
    }
}

#[test]
fn a_wrong_output_breaks_the_generic_row() {
    let input = [fp(0b1100), fp(0b1010)];
    let gadget = and::<16>();
    let (compiled, output) =
        arrabbiata_gadget::assign::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input).unwrap();
    assert_eq!(output, fp(0b1000));
    let last = <And<Fp, 16> as arrabbiata_gadget::TypedGadget<Fp>>::ROWS - 1;
    assert_eq!(compiled.check(&compiled.columns), Ok(()));
    let mut columns = compiled.columns.clone();
    columns[5][last] = columns[5][last] + Fp::ONE;
    assert!(compiled.check(&columns).is_err());
}
