//! Kimchi's range checks (`polynomials/range_check`): three 88-bit values
//! on four rows (RangeCheck0, RangeCheck0, RangeCheck1, Zero), the compact
//! variant for a 176-bit limb and an 88-bit one, and the single RangeCheck0
//! row.

use alloc::{vec, vec::Vec};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Rom, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi_gadgets::range_check::{
    CompactMultiRangeCheck, MultiRangeCheck, RangeCheck0Gate, range_check_table,
};

use crate::{
    constraints::{COLUMNS, CircuitGate, Connect, GateType, Wire},
    polynomials::columns,
};

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// A multi-range check from `start_row`: the row after it and its rows
    /// (Kimchi's `create_multi_range_check`).
    #[must_use]
    pub fn create_multi_range_check(start_row: usize) -> (usize, Vec<Self>) {
        Self::create_multi_range_check_gadget(start_row, false)
    }

    /// The compact multi-range check (Kimchi's
    /// `create_compact_multi_range_check`): the second row's coefficient is
    /// 1.
    #[must_use]
    pub fn create_compact_multi_range_check(start_row: usize) -> (usize, Vec<Self>) {
        Self::create_multi_range_check_gadget(start_row, true)
    }

    /// `gates` extended by a multi-range check at `curr_row`, which moves
    /// past it (Kimchi's `extend_multi_range_check`).
    pub fn extend_multi_range_check(gates: &mut Vec<Self>, curr_row: &mut usize) {
        let (next_row, rows) = Self::create_multi_range_check(*curr_row);
        *curr_row = next_row;
        gates.extend_from_slice(&rows);
    }

    /// `gates` extended by a compact multi-range check at `curr_row`
    /// (Kimchi's `extend_compact_multi_range_check`).
    pub fn extend_compact_multi_range_check(gates: &mut Vec<Self>, curr_row: &mut usize) {
        let (next_row, rows) = Self::create_compact_multi_range_check(*curr_row);
        *curr_row = next_row;
        gates.extend_from_slice(&rows);
    }

    /// One RangeCheck0 row at `start_row` (Kimchi's `create_range_check`).
    #[must_use]
    pub fn create_range_check(start_row: usize) -> (usize, Vec<Self>) {
        let gate = Self::new(
            GateType::RangeCheck0,
            Wire::for_row(start_row),
            vec![F::ZERO],
        );
        (start_row + 1, vec![gate])
    }

    /// `gates` extended by one RangeCheck0 row at `curr_row` (Kimchi's
    /// `extend_range_check`).
    pub fn extend_range_check(gates: &mut Vec<Self>, curr_row: &mut usize) {
        let (next_row, rows) = Self::create_range_check(*curr_row);
        *curr_row = next_row;
        gates.extend_from_slice(&rows);
    }

    /// The four rows, the two high limbs of each RangeCheck0 row wired to
    /// columns 3-6 of the Zero row. Kimchi wires them by the rows' positions
    /// in the gadget, as here.
    fn create_multi_range_check_gadget(start_row: usize, compact: bool) -> (usize, Vec<Self>) {
        let coeff = if compact { F::ONE } else { F::ZERO };
        let mut rows = vec![
            Self::new(
                GateType::RangeCheck0,
                Wire::for_row(start_row),
                vec![F::ZERO],
            ),
            Self::new(
                GateType::RangeCheck0,
                Wire::for_row(start_row + 1),
                vec![coeff],
            ),
            Self::new(GateType::RangeCheck1, Wire::for_row(start_row + 2), vec![]),
            Self::new(GateType::Zero, Wire::for_row(start_row + 3), vec![]),
        ];
        rows.connect_cell_pair((0, 1), (3, 3));
        rows.connect_cell_pair((0, 2), (3, 4));
        rows.connect_cell_pair((1, 1), (3, 5));
        rows.connect_cell_pair((1, 2), (3, 6));
        (start_row + rows.len(), rows)
    }
}

/// The three values in column 0 of rows 0-2, then the multi-range check.
#[derive(Clone, Debug)]
struct Multi<F>(Rom<F>);

impl<F> TypedGadget<F> for Multi<F>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 3>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-multi-range-check-values";
    const ROWS: usize = 4;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        values: Of<Array<Scalar, 3>, E::Var>,
    ) -> Of<Unit, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(0, i), [&values[i]], |[v]| v));
        env.call(
            &MultiRangeCheck {
                table: self.0.clone(),
            },
            cells,
        );
    }
}

/// `v2` in column 0 of row 0 and `v01` in column 1 of row 2, then the
/// compact multi-range check.
#[derive(Clone, Debug)]
struct Compact<F>(Rom<F>);

impl<F> TypedGadget<F> for Compact<F>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 2>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "kimchi-compact-multi-range-check-values";
    const ROWS: usize = 4;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        [v01, v2]: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let v2 = env.advice(Cell::new(0, 0), [&v2], |[v]| v);
        let v01 = env.advice(Cell::new(1, 2), [&v01], |[v]| v);
        env.call(
            &CompactMultiRangeCheck {
                table: self.0.clone(),
            },
            [v01, v2],
        )
    }
}

/// `v` in column 0 of row 0, then one RangeCheck0 row.
#[derive(Clone, Debug)]
struct Single<F>(Rom<F>);

impl<F> TypedGadget<F> for Single<F>
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "kimchi-range-check-value";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        v: Of<Scalar, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let v = env.advice(Cell::new(0, 0), [&v], |[x]| x);
        let gate = RangeCheck0Gate {
            table: self.0.clone(),
            compact: false,
        };
        env.call(&gate, v)
    }
}

/// The column variables of the range checks' rows.
const VARIABLES: usize = 40;
/// The degree of the crumb checks.
const DEGREE: usize = 4;

/// The witness of a multi-range check of `v0`, `v1`, `v2`, from row 0
/// (Kimchi's `range_check::witness::create_multi`), computed by our
/// multi-range check.
///
/// # Errors
///
/// The gadget's structural [`Error`], which it does not raise.
pub fn create_multi<F: PrimeField>(v0: F, v1: F, v2: F) -> Result<[Vec<F>; COLUMNS], Error> {
    columns::<F, _, VARIABLES, DEGREE>(&Multi(range_check_table()), &[v0, v1, v2], 4)
}

/// The witness of a compact multi-range check of the 176-bit limb `v01`
/// and the 88-bit `v2` (Kimchi's `create_multi_compact`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn create_multi_compact<F: PrimeField>(v01: F, v2: F) -> Result<[Vec<F>; COLUMNS], Error> {
    columns::<F, _, VARIABLES, DEGREE>(&Compact(range_check_table()), &[v01, v2], 4)
}

/// [`create_multi`] of three limbs (Kimchi's `create_multi_limbs`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn create_multi_limbs<F: PrimeField>(limbs: &[F; 3]) -> Result<[Vec<F>; COLUMNS], Error> {
    create_multi(limbs[0], limbs[1], limbs[2])
}

/// [`create_multi_compact`] of two limbs (Kimchi's
/// `create_multi_compact_limbs`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn create_multi_compact_limbs<F: PrimeField>(
    limbs: &[F; 2],
) -> Result<[Vec<F>; COLUMNS], Error> {
    create_multi_compact(limbs[0], limbs[1])
}

/// The witness of one RangeCheck0 row of `v0` (Kimchi's `create`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn create<F: PrimeField>(v0: F) -> Result<[Vec<F>; COLUMNS], Error> {
    columns::<F, _, VARIABLES, DEGREE>(&Single(range_check_table()), &v0, 1)
}

/// `witness` extended by `rows`.
fn extend_with<F: Field>(witness: &mut [Vec<F>; COLUMNS], rows: [Vec<F>; COLUMNS]) {
    for (column, new) in witness.iter_mut().zip(rows) {
        column.extend(new);
    }
}

/// `witness` extended by [`create_multi`] (Kimchi's `extend_multi`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn extend_multi<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    v0: F,
    v1: F,
    v2: F,
) -> Result<(), Error> {
    extend_with(witness, create_multi(v0, v1, v2)?);
    Ok(())
}

/// `witness` extended by [`create_multi_compact`] (Kimchi's
/// `extend_multi_compact`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn extend_multi_compact<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    v01: F,
    v2: F,
) -> Result<(), Error> {
    extend_with(witness, create_multi_compact(v01, v2)?);
    Ok(())
}

/// `witness` extended by [`create_multi_limbs`].
///
/// # Errors
///
/// As [`create_multi`].
pub fn extend_multi_limbs<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    limbs: &[F; 3],
) -> Result<(), Error> {
    extend_with(witness, create_multi_limbs(limbs)?);
    Ok(())
}

/// `witness` extended by [`create_multi_compact_limbs`].
///
/// # Errors
///
/// As [`create_multi`].
pub fn extend_multi_compact_limbs<F: PrimeField>(
    witness: &mut [Vec<F>; COLUMNS],
    limbs: &[F; 2],
) -> Result<(), Error> {
    extend_with(witness, create_multi_compact_limbs(limbs)?);
    Ok(())
}

/// `witness` extended by [`create`] (Kimchi's `extend` and
/// `extend_single`).
///
/// # Errors
///
/// As [`create_multi`].
pub fn extend<F: PrimeField>(witness: &mut [Vec<F>; COLUMNS], v0: F) -> Result<(), Error> {
    extend_with(witness, create(v0)?);
    Ok(())
}
