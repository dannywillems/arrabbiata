//! Kimchi's Poseidon gadget (`polynomials/poseidon.rs`): the 55 rounds of
//! the permutation on 11 Poseidon rows, five per row, the output on a Zero
//! row after them. The rows and the witness are derived from our Poseidon
//! permutation gadget.

use alloc::vec::Vec;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget};
use arrabbiata_kimchi_gadgets::poseidon::{PoseidonPermutation, ROWS_PER_PERMUTATION};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, WIDTH};

use crate::{
    constraints::{COLUMNS, CircuitGate, Wire},
    permutation::PERMUTS,
    polynomials::{DeriveError, Rows, derive},
};

/// The column variables of the rows.
const VARIABLES: usize = 33;
/// The degree of the rounds, `x^7`.
const DEGREE: usize = 7;

/// The rows of one permutation and its output row.
const ROWS: usize = ROWS_PER_PERMUTATION + 1;

/// The input in columns 0-2 of the first row, then our permutation.
#[derive(Clone, Copy, Debug)]
struct Placed<'a, F>(PoseidonPermutation<'a, F>);

impl<F> TypedGadget<F> for Placed<'_, F>
where
    F: Field,
{
    type Input = Array<Scalar, WIDTH>;
    type Output = Array<Scalar, WIDTH>;
    const NAME: &'static str = "kimchi-poseidon-placed";
    const ROWS: usize = ROWS_PER_PERMUTATION;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        input: Of<Array<Scalar, WIDTH>, E::Var>,
    ) -> Of<Array<Scalar, WIDTH>, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&input[i]], |[v]| v));
        env.call(&self.0, cells)
    }
}

/// The rows and witness of the permutation of `input` by `poseidon`, from
/// row `offset`.
fn rows<F: PrimeField>(
    poseidon: &KimchiPoseidon<F>,
    input: [F; WIDTH],
    offset: usize,
) -> Result<Rows<F>, DeriveError> {
    let gadget = Placed(PoseidonPermutation { poseidon });
    derive::<F, _, VARIABLES, DEGREE>(&gadget, &input, ROWS, offset)
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// The rows of a permutation from row `row`, derived from our Poseidon
    /// gadget: 11 Poseidon rows, the coefficients of each its five rounds'
    /// constants, then the Zero row holding the output; the first row wired
    /// as `first_and_last_row[0]`, the Zero row as `first_and_last_row[1]`.
    /// The rows and the row of the output (Kimchi's
    /// `create_poseidon_gadget`, which takes the round constants; ours takes
    /// the permutation, whose MDS matrix the gate also reads).
    ///
    /// # Panics
    ///
    /// Never: the gadget derives on any row.
    #[must_use]
    pub fn create_poseidon_gadget(
        row: usize,
        first_and_last_row: [[Wire; PERMUTS]; 2],
        poseidon: &KimchiPoseidon<F>,
    ) -> (Vec<Self>, usize) {
        match rows(poseidon, [F::ZERO; WIDTH], row) {
            Ok(mut rows) => {
                let last = rows.gates.len() - 1;
                rows.gates[0].wires = first_and_last_row[0];
                rows.gates[last].wires = first_and_last_row[1];
                (rows.gates, row + last)
            }
            Err(e) => panic!("no Poseidon rows: {e:?}"),
        }
    }
}

/// `witness` written from row `row` with the permutation of `input`: the
/// input and the state after each round, the output on the 12th row
/// (Kimchi's `poseidon::generate_witness`), computed by our Poseidon gates.
///
/// # Errors
///
/// [`Error::Rows`] when `witness` has fewer than `row + 12` rows.
pub fn generate_witness<F: PrimeField>(
    row: usize,
    poseidon: &KimchiPoseidon<F>,
    witness: &mut [Vec<F>; COLUMNS],
    input: [F; WIDTH],
) -> Result<(), Error> {
    let short = Error::Rows {
        gadget: "kimchi-poseidon",
    };
    if witness.iter().any(|column| column.len() < row + ROWS) {
        return Err(short);
    }
    let rows = rows(poseidon, input, 0).map_err(|_| short)?;
    for (column, new) in witness.iter_mut().zip(rows.witness) {
        column[row..row + ROWS].copy_from_slice(&new);
    }
    Ok(())
}
