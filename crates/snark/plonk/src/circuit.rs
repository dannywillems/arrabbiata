//! Circuits and witnesses of the generic PlonK arithmetization.
//!
//! A circuit over `W` witness columns, `X` fixed columns and `Q` selectors
//! fixes, on each of its `n` rows (a power of two):
//!
//! - the values of the fixed columns (constants and coefficients);
//! - which gadget of the gate set `G` is on, if any (one-hot selectors);
//! - the copy constraints, as a permutation `sigma` of the `W n` witness
//!   cells (cell `(k, i)` has index `k n + i`);
//! - the rows whose witness column 0 is a public input.
//!
//! A witness gives the `W` witness columns and the public inputs. It
//! satisfies the circuit when every constraint of every active gadget
//! vanishes, copied cells are equal, and column 0 holds the public inputs
//! on the public rows.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::{
//!     circuit::{Builder, Cell},
//!     gate::Gadget,
//! };
//! use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
//!
//! /// `b = a^2`.
//! struct Square;
//! impl Gadget<F, 2, 0> for Square {
//!     const NAME: &'static str = "square";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::Curr);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::Curr);
//!         vec![b - a.clone() * a]
//!     }
//! }
//!
//! // 3 -> 9 -> 81, the output of each row copied to the input of the next,
//! // and 81 published (on a row of its own, in column 0).
//! let mut b = Builder::<F, (Square, ()), 2, 0, 1>::new();
//! let r0 = b.row::<Square, _>([F::from_u64(3), F::from_u64(9)], []);
//! let r1 = b.row::<Square, _>([F::from_u64(9), F::from_u64(81)], []);
//! let out = b.plain_row([F::from_u64(81), F::ZERO], []);
//! let (a, c) = (WitnessColumn::new::<0>(), WitnessColumn::new::<1>());
//! b.copy(Cell::new(c, r0), Cell::new(a, r1));
//! b.copy(Cell::new(c, r1), Cell::new(a, out));
//! b.public(out);
//! let (circuit, witness) = b.build().unwrap();
//! assert_eq!(circuit.num_rows(), 4);
//! assert!(circuit.is_satisfied(&witness));
//! ```

use alloc::{vec, vec::Vec};
use core::{array, fmt, marker::PhantomData};

use arrabbiata_field::Field;
use arrabbiata_poly::{Assignment, FixedColumn, Rotation, Scalars, WitnessColumn, evaluate};

use crate::{
    PlonkError,
    gate::{GateSet, Member},
};

/// A witness cell: a column and a row.
///
/// ```
/// use arrabbiata_plonk::circuit::Cell;
/// use arrabbiata_poly::WitnessColumn;
/// let c = Cell::<3>::new(WitnessColumn::new::<2>(), 5);
/// assert_eq!(c.index(8), 2 * 8 + 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell<const W: usize> {
    /// The witness column.
    pub column: WitnessColumn<W>,
    /// The row.
    pub row: usize,
}

impl<const W: usize> Cell<W> {
    /// The cell of `column` at `row`.
    ///
    /// ```
    /// use arrabbiata_plonk::circuit::Cell;
    /// use arrabbiata_poly::WitnessColumn;
    /// assert_eq!(Cell::<2>::new(WitnessColumn::new::<1>(), 0).row, 0);
    /// ```
    #[must_use]
    pub const fn new(column: WitnessColumn<W>, row: usize) -> Self {
        Self { column, row }
    }

    /// The index of the cell among the `W n` witness cells: `k n + i`.
    ///
    /// ```
    /// use arrabbiata_plonk::circuit::Cell;
    /// use arrabbiata_poly::WitnessColumn;
    /// assert_eq!(Cell::<2>::new(WitnessColumn::new::<1>(), 3).index(4), 7);
    /// ```
    #[must_use]
    pub const fn index(self, num_rows: usize) -> usize {
        self.column.index() * num_rows + self.row
    }
}

/// A witness: `W` columns of `n` values, and the public inputs.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::circuit::Witness;
/// let w = Witness::new([vec![F::ONE, F::ZERO]], vec![F::ONE]);
/// assert_eq!(w.public_inputs(), &[F::ONE]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness<F, const W: usize> {
    columns: [Vec<F>; W],
    public: Vec<F>,
}

impl<F, const W: usize> Witness<F, W> {
    /// The witness with the given columns and public inputs. Shapes are
    /// checked against a circuit by [`Circuit::is_satisfied`] and by the
    /// prover.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::circuit::Witness;
    /// let w = Witness::new([vec![F::ONE; 2], vec![F::ZERO; 2]], vec![]);
    /// assert_eq!(w.columns()[1], vec![F::ZERO; 2]);
    /// ```
    #[must_use]
    pub const fn new(columns: [Vec<F>; W], public: Vec<F>) -> Self {
        Self { columns, public }
    }

    /// The witness columns.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::circuit::Witness;
    /// assert_eq!(Witness::new([vec![F::ONE]], vec![]).columns()[0], vec![F::ONE]);
    /// ```
    #[must_use]
    pub const fn columns(&self) -> &[Vec<F>; W] {
        &self.columns
    }

    /// The public inputs, in the order of the circuit's public rows.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::circuit::Witness;
    /// assert!(Witness::<F, 1>::new([vec![]], vec![]).public_inputs().is_empty());
    /// ```
    #[must_use]
    pub fn public_inputs(&self) -> &[F] {
        &self.public
    }
}

/// A circuit over the gate set `G`, with `W` witness columns, `X` fixed
/// columns and `Q` selectors. `Q` must be the number of gadgets of `G`,
/// which is checked at compile time:
///
/// ```compile_fail
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
/// // `Vanilla` has one gadget, hence one selector, not two.
/// let _ = Builder::<F, Vanilla, 3, 5, 2>::new();
/// ```
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
/// let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
/// assert_eq!(circuit.num_rows(), 2);
/// ```
pub struct Circuit<F, G, const W: usize, const X: usize, const Q: usize> {
    num_rows: usize,
    fixed: [Vec<F>; X],
    selectors: [Vec<F>; Q],
    sigma: Vec<usize>,
    public_rows: Vec<usize>,
    gates: PhantomData<fn() -> G>,
}

// Written out rather than derived: a derive would require `G: Clone` (and
// so on) of the gate set, a list of marker types that need not implement
// anything.
impl<F, G, const W: usize, const X: usize, const Q: usize> Clone for Circuit<F, G, W, X, Q>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            num_rows: self.num_rows,
            fixed: self.fixed.clone(),
            selectors: self.selectors.clone(),
            sigma: self.sigma.clone(),
            public_rows: self.public_rows.clone(),
            gates: PhantomData,
        }
    }
}

impl<F, G, const W: usize, const X: usize, const Q: usize> fmt::Debug for Circuit<F, G, W, X, Q>
where
    F: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circuit")
            .field("num_rows", &self.num_rows)
            .field("fixed", &self.fixed)
            .field("selectors", &self.selectors)
            .field("sigma", &self.sigma)
            .field("public_rows", &self.public_rows)
            .finish()
    }
}

impl<F, G, const W: usize, const X: usize, const Q: usize> PartialEq for Circuit<F, G, W, X, Q>
where
    F: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.num_rows == other.num_rows
            && self.fixed == other.fixed
            && self.selectors == other.selectors
            && self.sigma == other.sigma
            && self.public_rows == other.public_rows
    }
}

impl<F, G, const W: usize, const X: usize, const Q: usize> Eq for Circuit<F, G, W, X, Q> where F: Eq {}

impl<F, G, const W: usize, const X: usize, const Q: usize> Circuit<F, G, W, X, Q>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    /// `Q` is the number of gadgets of `G`.
    const SELECTORS: () = assert!(
        G::SELECTORS == Q,
        "Q must be the number of gadgets of the gate set"
    );

    /// The circuit with the given columns, copy permutation and public
    /// rows.
    ///
    /// # Errors
    ///
    /// [`PlonkError::Shape`] if `num_rows` is not a power of two at least
    /// 2, a column does not have `num_rows` values, `sigma` is not a
    /// permutation of the `W num_rows` cells, or a public row is out of
    /// range or repeated; [`PlonkError::Selectors`] if a selector value is
    /// not 0 or 1, or two selectors are on in the same row.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{PlonkError, circuit::Circuit, vanilla::Vanilla};
    /// let fixed: [Vec<F>; 5] = std::array::from_fn(|_| vec![F::ZERO; 2]);
    /// let identity: Vec<usize> = (0..6).collect();
    /// let ok = Circuit::<F, Vanilla, 3, 5, 1>::new(2, fixed.clone(), [vec![F::ONE; 2]], identity.clone(), vec![]);
    /// assert!(ok.is_ok());
    /// let two = F::ONE + F::ONE;
    /// let bad = Circuit::<F, Vanilla, 3, 5, 1>::new(2, fixed, [vec![two; 2]], identity, vec![]);
    /// assert_eq!(bad.unwrap_err(), PlonkError::Selectors);
    /// ```
    pub fn new(
        num_rows: usize,
        fixed: [Vec<F>; X],
        selectors: [Vec<F>; Q],
        sigma: Vec<usize>,
        public_rows: Vec<usize>,
    ) -> Result<Self, PlonkError> {
        let () = Self::SELECTORS;
        if num_rows < 2 || !num_rows.is_power_of_two() {
            return Err(PlonkError::Shape);
        }
        if fixed.iter().any(|c| c.len() != num_rows)
            || selectors.iter().any(|c| c.len() != num_rows)
        {
            return Err(PlonkError::Shape);
        }
        let mut row = 0;
        while row < num_rows {
            let mut on = 0;
            for column in &selectors {
                if column[row] == F::ONE {
                    on += 1;
                } else if !column[row].is_zero() {
                    return Err(PlonkError::Selectors);
                }
            }
            if on > 1 {
                return Err(PlonkError::Selectors);
            }
            row += 1;
        }
        if sigma.len() != W * num_rows {
            return Err(PlonkError::Shape);
        }
        let mut seen = vec![false; sigma.len()];
        for target in &sigma {
            if *target >= seen.len() || seen[*target] {
                return Err(PlonkError::Shape);
            }
            seen[*target] = true;
        }
        let mut used = vec![false; num_rows];
        for row in &public_rows {
            if *row >= num_rows || used[*row] {
                return Err(PlonkError::Shape);
            }
            used[*row] = true;
        }
        Ok(Self {
            num_rows,
            fixed,
            selectors,
            sigma,
            public_rows,
            gates: PhantomData,
        })
    }

    /// The number of rows, a power of two.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// assert!(c.num_rows().is_power_of_two());
    /// ```
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The fixed columns.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// assert_eq!(c.fixed().len(), 5);
    /// ```
    #[must_use]
    pub const fn fixed(&self) -> &[Vec<F>; X] {
        &self.fixed
    }

    /// The selector columns, one per gadget of `G`, in the order of `G`.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// assert_eq!(c.selectors().len(), 1);
    /// ```
    #[must_use]
    pub const fn selectors(&self) -> &[Vec<F>; Q] {
        &self.selectors
    }

    /// The copy permutation of the witness cells.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// // No copy: the identity.
    /// assert!(c.sigma().iter().enumerate().all(|(i, s)| i == *s));
    /// ```
    #[must_use]
    pub fn sigma(&self) -> &[usize] {
        &self.sigma
    }

    /// The rows whose witness column 0 is a public input, in the order of
    /// the public inputs.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    /// b.public(r);
    /// let (c, w) = b.build().unwrap();
    /// assert_eq!(c.public_rows(), &[r]);
    /// assert_eq!(w.public_inputs(), &[F::ONE]);
    /// ```
    #[must_use]
    pub fn public_rows(&self) -> &[usize] {
        &self.public_rows
    }

    /// The largest total degree of a gated constraint (selector included),
    /// and of the copy and public-input identities: the quotient has fewer
    /// than `(quotient_factor() - 1) n` coefficients.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// // q * q_M a b has degree 4, and so has z (a + ..) (b + ..) (c + ..).
    /// assert_eq!(c.quotient_factor(), 4);
    /// ```
    #[must_use]
    pub fn quotient_factor(&self) -> usize {
        quotient_factor::<F, G, W, X>()
    }

    /// The number of coefficients a polynomial commitment for this circuit
    /// must accept: the size of the quotient, or of a column if larger.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// assert_eq!(c.max_coefficients(), 3 * c.num_rows());
    /// ```
    #[must_use]
    pub fn max_coefficients(&self) -> usize {
        let quotient = (self.quotient_factor() - 1) * self.num_rows;
        if quotient > self.num_rows {
            quotient
        } else {
            self.num_rows
        }
    }

    /// Whether `witness` satisfies the circuit: the gates of the active
    /// gadgets vanish, copied cells are equal, and column 0 holds the
    /// public inputs on the public rows.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::{Builder, Witness}, vanilla::Vanilla};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    /// b.public(r);
    /// let (c, w) = b.build().unwrap();
    /// assert!(c.is_satisfied(&w));
    /// // Another public input for the same witness.
    /// let other = Witness::new(w.columns().clone(), vec![F::ZERO]);
    /// assert!(!c.is_satisfied(&other));
    /// ```
    #[must_use]
    pub fn is_satisfied(&self, witness: &Witness<F, W>) -> bool {
        let n = self.num_rows;
        if witness.columns.iter().any(|c| c.len() != n)
            || witness.public.len() != self.public_rows.len()
        {
            return false;
        }
        for (column, gadget) in self.selectors.iter().zip(0..Q) {
            let constraints = G::constraints(gadget);
            let mut row = 0;
            while row < n {
                if column[row] == F::ONE {
                    let view = RowView {
                        witness: &witness.columns,
                        fixed: &self.fixed,
                        row,
                        num_rows: n,
                    };
                    let ring = Scalars::new();
                    if constraints
                        .iter()
                        .any(|c| !evaluate(&ring, c, &view).is_zero())
                    {
                        return false;
                    }
                }
                row += 1;
            }
        }
        if !witness.columns.is_empty() {
            for (row, value) in self.public_rows.iter().zip(&witness.public) {
                if witness.columns[0][*row] != *value {
                    return false;
                }
            }
        }
        for (cell, target) in self.sigma.iter().enumerate() {
            if witness.columns[cell / n][cell % n] != witness.columns[target / n][target % n] {
                return false;
            }
        }
        true
    }
}

/// The quotient factor of a gate set: see [`Circuit::quotient_factor`].
pub(crate) fn quotient_factor<F, G, const W: usize, const X: usize>() -> usize
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut factor = if W + 1 > 2 { W + 1 } else { 2 };
    let mut gadget = 0;
    while gadget < G::SELECTORS {
        for c in G::constraints(gadget) {
            let gated = crate::gate::total_degree(&c) + 1;
            if gated > factor {
                factor = gated;
            }
        }
        gadget += 1;
    }
    factor
}

/// The values of one row (and of the next) of a witness and the fixed
/// columns, as an assignment of the variables of a constraint.
pub(crate) struct RowView<'a, F, const W: usize, const X: usize> {
    pub(crate) witness: &'a [Vec<F>; W],
    pub(crate) fixed: &'a [Vec<F>; X],
    pub(crate) row: usize,
    pub(crate) num_rows: usize,
}

impl<F, const W: usize, const X: usize> Assignment<F, Scalars<F>, W, X> for RowView<'_, F, W, X>
where
    F: Field,
{
    fn witness(&self, _: &Scalars<F>, column: WitnessColumn<W>, rotation: Rotation) -> F {
        self.witness[column.index()][rotation.apply(self.row, self.num_rows)]
    }

    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F {
        self.fixed[column.index()][rotation.apply(self.row, self.num_rows)]
    }
}

/// A builder of a circuit and its witness, row by row.
///
/// Each row either turns on one gadget of `G` ([`Builder::row`]) or none
/// ([`Builder::plain_row`]). Copy constraints and public inputs refer to
/// rows by the index these calls return. [`Builder::build`] pads to a
/// power of two with rows of zeros and no gadget.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{circuit::Builder, vanilla::{Arithmetic, Vanilla}};
/// // 2 * 3 = 6: q_M a b - c = 0, with q_M = 1 and q_O = -1.
/// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
/// let (two, three, six) = (F::from_u64(2), F::from_u64(3), F::from_u64(6));
/// b.row::<Arithmetic, _>([two, three, six], [F::ZERO, F::ZERO, -F::ONE, F::ONE, F::ZERO]);
/// let (circuit, witness) = b.build().unwrap();
/// assert!(circuit.is_satisfied(&witness));
/// ```
pub struct Builder<F, G, const W: usize, const X: usize, const Q: usize> {
    witness: [Vec<F>; W],
    fixed: [Vec<F>; X],
    active: Vec<Option<usize>>,
    copies: Vec<(Cell<W>, Cell<W>)>,
    public: Vec<usize>,
    gates: PhantomData<fn() -> G>,
}

impl<F, G, const W: usize, const X: usize, const Q: usize> Default for Builder<F, G, W, X, Q>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, G, const W: usize, const X: usize, const Q: usize> Builder<F, G, W, X, Q>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    /// An empty builder.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// assert_eq!(Builder::<F, Vanilla, 3, 5, 1>::new().num_rows(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let () = Circuit::<F, G, W, X, Q>::SELECTORS;
        Self {
            witness: array::from_fn(|_| Vec::new()),
            fixed: array::from_fn(|_| Vec::new()),
            active: Vec::new(),
            copies: Vec::new(),
            public: Vec::new(),
            gates: PhantomData,
        }
    }

    /// The number of rows added so far.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// b.plain_row([F::ZERO; 3], [F::ZERO; 5]);
    /// assert_eq!(b.num_rows(), 1);
    /// ```
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.active.len()
    }

    /// Add a row turning on the gadget `H`, which `G` must contain; returns
    /// its index.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::{Arithmetic, Vanilla}};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// // a = 1: q_L a + q_C = 0 with q_L = 1, q_C = -1.
    /// let r = b.row::<Arithmetic, _>([F::ONE, F::ZERO, F::ZERO], [F::ONE, F::ZERO, F::ZERO, F::ZERO, -F::ONE]);
    /// assert_eq!(r, 0);
    /// ```
    pub fn row<H, I>(&mut self, witness: [F; W], fixed: [F; X]) -> usize
    where
        G: Member<H, I>,
    {
        self.push(witness, fixed, Some(<G as Member<H, I>>::INDEX))
    }

    /// Add a row with no gadget on (its cells are only constrained by
    /// copies and public inputs); returns its index.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// assert_eq!(b.plain_row([F::ONE; 3], [F::ZERO; 5]), 0);
    /// assert_eq!(b.plain_row([F::ONE; 3], [F::ZERO; 5]), 1);
    /// ```
    pub fn plain_row(&mut self, witness: [F; W], fixed: [F; X]) -> usize {
        self.push(witness, fixed, None)
    }

    fn push(&mut self, witness: [F; W], fixed: [F; X], gadget: Option<usize>) -> usize {
        for (column, value) in self.witness.iter_mut().zip(witness) {
            column.push(value);
        }
        for (column, value) in self.fixed.iter_mut().zip(fixed) {
            column.push(value);
        }
        self.active.push(gadget);
        self.active.len() - 1
    }

    /// Require two cells to hold the same value.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::{Builder, Cell}, vanilla::Vanilla};
    /// use arrabbiata_poly::WitnessColumn;
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    /// b.copy(Cell::new(WitnessColumn::new::<0>(), r), Cell::new(WitnessColumn::new::<1>(), r));
    /// let (c, w) = b.build().unwrap();
    /// assert!(!c.is_satisfied(&w));
    /// ```
    pub fn copy(&mut self, x: Cell<W>, y: Cell<W>) {
        self.copies.push((x, y));
    }

    /// Make witness column 0 of `row` a public input. Public inputs are
    /// ordered by the calls to this method, and take the value of the cell.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{circuit::Builder, vanilla::Vanilla};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// let r = b.plain_row([F::from_u64(7), F::ZERO, F::ZERO], [F::ZERO; 5]);
    /// b.public(r);
    /// let (_, w) = b.build().unwrap();
    /// assert_eq!(w.public_inputs(), &[F::from_u64(7)]);
    /// ```
    pub fn public(&mut self, row: usize) {
        self.public.push(row);
    }

    /// The circuit and its witness, padded to a power of two (at least 2)
    /// rows of zeros with no gadget on.
    ///
    /// # Errors
    ///
    /// [`PlonkError::Shape`] if a copy or a public input refers to a row
    /// that was not added, or a row is public twice.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{PlonkError, circuit::Builder, vanilla::Vanilla};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// b.public(0);
    /// assert_eq!(b.build().unwrap_err(), PlonkError::Shape);
    /// ```
    pub fn build(self) -> Result<(Circuit<F, G, W, X, Q>, Witness<F, W>), PlonkError> {
        let rows = self.active.len();
        let mut n = 2;
        while n < rows {
            n *= 2;
        }
        let mut witness = self.witness;
        for column in &mut witness {
            column.resize(n, F::ZERO);
        }
        let mut fixed = self.fixed;
        for column in &mut fixed {
            column.resize(n, F::ZERO);
        }
        let mut selectors: [Vec<F>; Q] = array::from_fn(|_| vec![F::ZERO; n]);
        for (row, gadget) in self.active.iter().enumerate() {
            if let Some(g) = gadget {
                selectors[*g][row] = F::ONE;
            }
        }
        for (x, y) in &self.copies {
            if x.row >= rows || y.row >= rows {
                return Err(PlonkError::Shape);
            }
        }
        let sigma = permutation(&self.copies, W, n);
        let mut public = Vec::with_capacity(self.public.len());
        for row in &self.public {
            if *row >= rows || W == 0 {
                return Err(PlonkError::Shape);
            }
            public.push(witness[0][*row]);
        }
        let circuit = Circuit::new(n, fixed, selectors, sigma, self.public)?;
        Ok((circuit, Witness::new(witness, public)))
    }
}

/// The permutation whose cycles are the classes of cells equated by
/// `copies`, each cycle in increasing order of cell index.
pub(crate) fn permutation<const W: usize>(
    copies: &[(Cell<W>, Cell<W>)],
    columns: usize,
    num_rows: usize,
) -> Vec<usize> {
    let size = columns * num_rows;
    let mut parent: Vec<usize> = (0..size).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (x, y) in copies {
        let rx = find(&mut parent, x.index(num_rows));
        let ry = find(&mut parent, y.index(num_rows));
        if rx != ry {
            parent[rx] = ry;
        }
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); size];
    for cell in 0..size {
        let root = find(&mut parent, cell);
        members[root].push(cell);
    }
    let mut sigma: Vec<usize> = (0..size).collect();
    for class in &members {
        for (i, cell) in class.iter().enumerate() {
            sigma[*cell] = class[(i + 1) % class.len()];
        }
    }
    sigma
}
