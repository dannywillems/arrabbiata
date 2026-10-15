//! Circuits of generalized PlonK.
//!
//! A circuit has `n` rows (a power of two), `W` witness columns, `X` fixed
//! columns, `Q` selector columns and `P` public inputs, all chosen at
//! runtime. Its gates are multivariate polynomials of degree at most `D`
//! in `N` variables ([`Sparse`]), each variable naming a witness or fixed
//! column read at a [`Rotation`] of the constrained row, or a public input
//! ([`Variable`]). A gate is turned on by a selector, or by none (then it
//! holds on every row). A witness satisfies the circuit when:
//!
//! - every gate vanishes on every row where its selector is nonzero;
//! - the copy permutation `sigma` of the `W n` witness cells maps every
//!   cell to one holding the same value;
//! - every binding `(j, i)` holds: witness column 0 at row `i` is public
//!   input `j`.
//!
//! Selectors are columns with any values; one-hot selectors, sums of
//! selectors equal to one, are circuits, not requirements.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_generalized_plonk::{Circuit, Gate, Variable};
//! use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_poly::Rotation;
//!
//! // One gate, w0 * w0 - x0 = 0 on the rows of selector 0, with x0 public.
//! let var = |i| Sparse::<F, 2, 2>::variable(Var::try_new(i).unwrap()).unwrap();
//! let gate = Gate::new(Some(0), var(0).checked_mul(&var(0)).unwrap() - var(1));
//! let variables = vec![
//!     Variable::Witness { column: 0, rotation: Rotation::CURR },
//!     Variable::Public(0),
//! ];
//! let circuit = Circuit::new(2, 1, vec![], vec![vec![F::ONE, F::ZERO]], variables, vec![gate], (0..2).collect(), 1, vec![]).unwrap();
//! let columns = vec![vec![F::from_u64(3), F::from_u64(5)]];
//! assert!(circuit.is_satisfied(&columns, &[F::from_u64(9)]));
//! assert!(!circuit.is_satisfied(&columns, &[F::from_u64(10)]));
//! ```

use alloc::{vec, vec::Vec};

use arrabbiata_field::Field;
use arrabbiata_mvpoly::{MVPoly, Sparse};
use arrabbiata_poly::Rotation;
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, typed::Label};

use crate::Error;

/// What a variable of a gate reads.
///
/// ```
/// use arrabbiata_generalized_plonk::Variable;
/// use arrabbiata_poly::Rotation;
/// let v = Variable::Fixed { column: 2, rotation: Rotation::NEXT };
/// assert!(v.is_column());
/// assert!(!Variable::Public(0).is_column());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Variable {
    /// A witness column at a rotation of the constrained row.
    Witness {
        /// The column.
        column: usize,
        /// The rotation.
        rotation: Rotation,
    },
    /// A fixed column at a rotation of the constrained row.
    Fixed {
        /// The column.
        column: usize,
        /// The rotation.
        rotation: Rotation,
    },
    /// A public input, the same on every row.
    Public(usize),
}

impl Variable {
    /// Whether the variable reads a column (and so has degree one in the
    /// polynomial interpolating it), rather than a public input (a scalar).
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Variable;
    /// use arrabbiata_poly::Rotation;
    /// assert!(Variable::Witness { column: 0, rotation: Rotation::CURR }.is_column());
    /// ```
    #[must_use]
    pub const fn is_column(&self) -> bool {
        !matches!(self, Self::Public(_))
    }

    /// The code absorbed into a transcript: kind, index and rotation.
    fn code(&self) -> [usize; 3] {
        match self {
            Self::Witness { column, rotation } => [1, *column, rotation.code() as usize],
            Self::Fixed { column, rotation } => [2, *column, rotation.code() as usize],
            Self::Public(i) => [3, *i, 0],
        }
    }
}

/// A gate: a polynomial of degree at most `D` in `N` variables, turned on
/// by a selector (`Some(q)`) or on every row (`None`).
///
/// ```
/// use arrabbiata_generalized_plonk::Gate;
/// use arrabbiata_mvpoly::{MVPoly, Sparse};
/// use arrabbiata_pasta::Fq as F;
/// let g = Gate::new(None, Sparse::<F, 1, 1>::zero());
/// assert_eq!(g.selector(), None);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate<F, const N: usize, const D: usize>
where
    F: Field,
{
    selector: Option<usize>,
    polynomial: Sparse<F, N, D>,
}

impl<F, const N: usize, const D: usize> Gate<F, N, D>
where
    F: Field,
{
    /// The gate `polynomial`, turned on by `selector`.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::Gate;
    /// use arrabbiata_mvpoly::{MVPoly, Sparse};
    /// use arrabbiata_pasta::Fq as F;
    /// let g = Gate::new(Some(1), Sparse::<F, 1, 1>::one());
    /// assert_eq!(g.selector(), Some(1));
    /// ```
    #[must_use]
    pub const fn new(selector: Option<usize>, polynomial: Sparse<F, N, D>) -> Self {
        Self {
            selector,
            polynomial,
        }
    }

    /// The selector turning the gate on, if any.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Gate;
    /// use arrabbiata_mvpoly::{MVPoly, Sparse};
    /// use arrabbiata_pasta::Fq as F;
    /// assert_eq!(Gate::new(Some(0), Sparse::<F, 1, 1>::zero()).selector(), Some(0));
    /// ```
    #[must_use]
    pub const fn selector(&self) -> Option<usize> {
        self.selector
    }

    /// The polynomial.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Gate;
    /// use arrabbiata_mvpoly::{MVPoly, Sparse};
    /// use arrabbiata_pasta::Fq as F;
    /// assert!(Gate::new(None, Sparse::<F, 1, 1>::zero()).polynomial().is_zero());
    /// ```
    #[must_use]
    pub const fn polynomial(&self) -> &Sparse<F, N, D> {
        &self.polynomial
    }
}

/// A circuit of generalized PlonK: see the [module documentation](self).
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_generalized_plonk::{Circuit, Error};
/// use arrabbiata_pasta::Fq as F;
/// // Three rows is not a power of two.
/// let c = Circuit::<F, 1, 1>::new(3, 1, vec![], vec![], vec![], vec![], (0..3).collect(), 0, vec![]);
/// assert_eq!(c.unwrap_err(), Error::Shape);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuit<F, const N: usize, const D: usize>
where
    F: Field,
{
    num_rows: usize,
    witness_columns: usize,
    fixed: Vec<Vec<F>>,
    selectors: Vec<Vec<F>>,
    variables: Vec<Variable>,
    gates: Vec<Gate<F, N, D>>,
    sigma: Vec<usize>,
    public_inputs: usize,
    bindings: Vec<(usize, usize)>,
}

impl<F, const N: usize, const D: usize> Circuit<F, N, D>
where
    F: Field,
{
    /// The circuit with `num_rows` rows, `witness_columns` witness columns,
    /// the given fixed and selector columns, variables, gates, copy
    /// permutation of the `witness_columns * num_rows` cells (cell
    /// `(k, i)` has index `k n + i`), number of public inputs, and bindings
    /// `(input, row)` of public inputs to witness column 0.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if the number of rows is not a power of two at
    /// least 2, a column does not have one value per row, there are more
    /// than `N` variables or a variable names a column or an input that
    /// does not exist, a gate names a selector that does not exist or reads
    /// a variable that is not declared, `sigma` is not a permutation of the
    /// witness cells, or a binding is out of range, binds a row twice, or
    /// has no witness column to bind to.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::{Circuit, Error, Gate, Variable};
    /// use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
    /// use arrabbiata_pasta::Fq as F;
    /// // The gate reads variable 0, but no variable is declared.
    /// let gate = Gate::new(None, Sparse::<F, 1, 1>::variable(Var::new::<0>()).unwrap());
    /// let c = Circuit::new(2, 1, vec![], vec![], vec![], vec![gate], vec![0, 1], 0, vec![]);
    /// assert_eq!(c.unwrap_err(), Error::Shape);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_rows: usize,
        witness_columns: usize,
        fixed: Vec<Vec<F>>,
        selectors: Vec<Vec<F>>,
        variables: Vec<Variable>,
        gates: Vec<Gate<F, N, D>>,
        sigma: Vec<usize>,
        public_inputs: usize,
        bindings: Vec<(usize, usize)>,
    ) -> Result<Self, Error> {
        let n = num_rows;
        let shapes = n >= 2
            && n.is_power_of_two()
            && fixed.iter().chain(&selectors).all(|c| c.len() == n)
            && variables.len() <= N;
        if !shapes {
            return Err(Error::Shape);
        }
        let variable_ok = |v: &Variable| match v {
            Variable::Witness { column, .. } => *column < witness_columns,
            Variable::Fixed { column, .. } => *column < fixed.len(),
            Variable::Public(i) => *i < public_inputs,
        };
        if !variables.iter().all(variable_ok) {
            return Err(Error::Shape);
        }
        for gate in &gates {
            if gate.selector.is_some_and(|q| q >= selectors.len()) {
                return Err(Error::Shape);
            }
            let reads_undeclared = gate.polynomial.terms().iter().any(|(m, _)| {
                m.exponents()
                    .iter()
                    .enumerate()
                    .any(|(i, e)| *e > 0 && i >= variables.len())
            });
            if reads_undeclared {
                return Err(Error::Shape);
            }
        }
        let cells = witness_columns * n;
        let mut seen = vec![false; cells];
        if sigma.len() != cells {
            return Err(Error::Shape);
        }
        for target in &sigma {
            if *target >= cells || seen[*target] {
                return Err(Error::Shape);
            }
            seen[*target] = true;
        }
        let bindings_ok = bindings.iter().enumerate().all(|(k, (j, i))| {
            *j < public_inputs
                && *i < n
                && witness_columns > 0
                && bindings[..k].iter().all(|(_, other)| other != i)
        });
        if !bindings_ok {
            return Err(Error::Shape);
        }
        Ok(Self {
            num_rows,
            witness_columns,
            fixed,
            selectors,
            variables,
            gates,
            sigma,
            public_inputs,
            bindings,
        })
    }

    /// The number of rows, a power of two.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(4, 0, vec![], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// assert_eq!(c.num_rows(), 4);
    /// ```
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of witness columns.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 3, vec![], vec![], vec![], vec![], (0..6).collect(), 0, vec![]).unwrap();
    /// assert_eq!(c.witness_columns(), 3);
    /// ```
    #[must_use]
    pub const fn witness_columns(&self) -> usize {
        self.witness_columns
    }

    /// The fixed columns.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![vec![F::ONE; 2]], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// assert_eq!(c.fixed(), &[vec![F::ONE; 2]]);
    /// ```
    #[must_use]
    pub fn fixed(&self) -> &[Vec<F>] {
        &self.fixed
    }

    /// The selector columns.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![vec![F::ONE, F::ZERO]], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// assert_eq!(c.selectors().len(), 1);
    /// ```
    #[must_use]
    pub fn selectors(&self) -> &[Vec<F>] {
        &self.selectors
    }

    /// The variables of the gates, in the order of their indices.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::{Circuit, Variable};
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![Variable::Public(0)], vec![], vec![], 1, vec![]).unwrap();
    /// assert_eq!(c.variables(), &[Variable::Public(0)]);
    /// ```
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// The gates.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// assert!(c.gates().is_empty());
    /// ```
    #[must_use]
    pub fn gates(&self) -> &[Gate<F, N, D>] {
        &self.gates
    }

    /// The copy permutation of the witness cells.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![1, 0], 0, vec![]).unwrap();
    /// assert_eq!(c.sigma(), &[1, 0]);
    /// ```
    #[must_use]
    pub fn sigma(&self) -> &[usize] {
        &self.sigma
    }

    /// The number of public inputs.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![], vec![], vec![], 5, vec![]).unwrap();
    /// assert_eq!(c.public_inputs(), 5);
    /// ```
    #[must_use]
    pub const fn public_inputs(&self) -> usize {
        self.public_inputs
    }

    /// The bindings `(input, row)`: witness column 0 at `row` holds public
    /// input `input`.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 1], 1, vec![(0, 1)]).unwrap();
    /// assert_eq!(c.bindings(), &[(0, 1)]);
    /// ```
    #[must_use]
    pub fn bindings(&self) -> &[(usize, usize)] {
        &self.bindings
    }

    /// The degree of a gate in the polynomials the protocol interpolates:
    /// the largest number of column variables in a monomial (public inputs
    /// are scalars), plus one for the selector.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::{Circuit, Gate, Variable};
    /// use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_poly::Rotation;
    /// let var = |i| Sparse::<F, 2, 2>::variable(Var::try_new(i).unwrap()).unwrap();
    /// // w0 * x0: one column, one scalar; gated.
    /// let gate = Gate::new(Some(0), var(0).checked_mul(&var(1)).unwrap());
    /// let vars = vec![Variable::Witness { column: 0, rotation: Rotation::CURR }, Variable::Public(0)];
    /// let c = Circuit::new(2, 1, vec![], vec![vec![F::ONE; 2]], vars, vec![gate], vec![0, 1], 1, vec![]).unwrap();
    /// assert_eq!(c.column_degree(&c.gates()[0]), 2);
    /// ```
    #[must_use]
    pub fn column_degree(&self, gate: &Gate<F, N, D>) -> usize {
        let mut degree = 0;
        for (m, _) in gate.polynomial.terms() {
            let mut d = 0;
            for (i, e) in m.exponents().iter().enumerate() {
                if self.variables.get(i).is_some_and(Variable::is_column) {
                    d += *e as usize;
                }
            }
            degree = degree.max(d);
        }
        degree + usize::from(gate.selector.is_some())
    }

    /// The point of the gates' variables given the value of each variable;
    /// variables beyond the declared ones are zero.
    pub(crate) fn point(&self, mut value: impl FnMut(&Variable) -> F) -> [F; N] {
        let mut x = [F::ZERO; N];
        for (slot, v) in x.iter_mut().zip(&self.variables) {
            *slot = value(v);
        }
        x
    }

    /// Whether the witness columns `columns` and the public inputs `public`
    /// satisfy the circuit: see the [module documentation](self).
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// // Two cells copied to each other, and one bound to x0.
    /// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![1, 0], 1, vec![(0, 0)]).unwrap();
    /// assert!(c.is_satisfied(&[vec![F::ONE, F::ONE]], &[F::ONE]));
    /// assert!(!c.is_satisfied(&[vec![F::ONE, F::ZERO]], &[F::ONE]));
    /// ```
    #[must_use]
    pub fn is_satisfied(&self, columns: &[Vec<F>], public: &[F]) -> bool {
        let n = self.num_rows;
        if columns.len() != self.witness_columns
            || columns.iter().any(|c| c.len() != n)
            || public.len() != self.public_inputs
        {
            return false;
        }
        for row in 0..n {
            let x = self.point(|v| match v {
                Variable::Witness { column, rotation } => columns[*column][rotation.apply(row, n)],
                Variable::Fixed { column, rotation } => self.fixed[*column][rotation.apply(row, n)],
                Variable::Public(i) => public[*i],
            });
            for gate in &self.gates {
                let on = match gate.selector {
                    Some(q) => !self.selectors[q][row].is_zero(),
                    None => true,
                };
                if on && !gate.polynomial.eval(&x).is_zero() {
                    return false;
                }
            }
        }
        for (j, i) in &self.bindings {
            if columns[0][*i] != public[*j] {
                return false;
            }
        }
        for (cell, target) in self.sigma.iter().enumerate() {
            if columns[cell / n][cell % n] != columns[target / n][target % n] {
                return false;
            }
        }
        true
    }
}

impl<F, const N: usize, const D: usize> Label for Circuit<F, N, D>
where
    F: Field,
{
    const LABEL: u64 = 0x6770_6369;
}

/// The shape of a circuit and its gates, without the column values: what
/// a verifier key absorbs besides the column commitments. The counts are
/// `(rows, witness columns, fixed columns, selectors, public inputs)`.
pub(crate) fn absorb_parts<F, T, const N: usize, const D: usize>(
    transcript: &mut T,
    (rows, witness, fixed, selectors, public): (usize, usize, usize, usize, usize),
    variables: &[Variable],
    gates: &[Gate<F, N, D>],
    bindings: &[(usize, usize)],
) where
    F: Field,
    T: Transcript<F>,
{
    for count in [
        rows,
        witness,
        fixed,
        selectors,
        public,
        N,
        D,
        variables.len(),
    ] {
        absorb_length(transcript, count);
    }
    for v in variables {
        for c in v.code() {
            absorb_length(transcript, c);
        }
    }
    absorb_length(transcript, gates.len());
    for gate in gates {
        match gate.selector {
            None => absorb_length(transcript, 0),
            Some(q) => absorb_length(transcript, q + 1),
        }
        let terms = gate.polynomial.terms();
        absorb_length(transcript, terms.len());
        for (m, c) in terms {
            for e in m.exponents() {
                absorb_length(transcript, *e as usize);
            }
            transcript.absorb(c);
        }
    }
    absorb_length(transcript, bindings.len());
    for (j, i) in bindings {
        absorb_length(transcript, *j);
        absorb_length(transcript, *i);
    }
}

/// The whole circuit: its shape, then every fixed and selector value and
/// the copy permutation.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_generalized_plonk::Circuit;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_poseidon::vesta_sponge;
/// use arrabbiata_transcript::{Absorb, Transcript};
/// let a = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 1], 0, vec![]).unwrap();
/// let b = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![1, 0], 0, vec![]).unwrap();
/// let (mut s, mut t) = (vesta_sponge(0), vesta_sponge(0));
/// a.absorb_into(&mut s);
/// b.absorb_into(&mut t);
/// assert_ne!(s.squeeze(), t.squeeze());
/// ```
impl<F, const N: usize, const D: usize> Absorb<F> for Circuit<F, N, D>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_parts::<F, T, N, D>(
            transcript,
            (
                self.num_rows,
                self.witness_columns,
                self.fixed.len(),
                self.selectors.len(),
                self.public_inputs,
            ),
            &self.variables,
            &self.gates,
            &self.bindings,
        );
        for column in self.fixed.iter().chain(&self.selectors) {
            for v in column {
                transcript.absorb(*v);
            }
        }
        for target in &self.sigma {
            absorb_length(transcript, *target);
        }
    }
}
