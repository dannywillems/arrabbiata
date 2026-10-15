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
//! let selector = Some(0);
//! let polynomial = var(0).checked_mul(&var(0)).unwrap() - var(1);
//! let gate = Gate::new(selector, polynomial);
//! let variables = vec![
//!     Variable::Witness { column: 0, rotation: Rotation::CURR },
//!     Variable::Public(0),
//! ];
//! let num_rows = 2;
//! let witness_columns = 1;
//! let fixed = vec![];
//! let selectors = vec![vec![F::ONE, F::ZERO]];
//! let gates = vec![gate];
//! let sigma = (0..2).collect();
//! let public_inputs = 1;
//! let bindings = vec![];
//! let circuit = Circuit::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
//! let columns = vec![vec![F::from_u64(3), F::from_u64(5)]];
//! assert!(circuit.is_satisfied(&columns, &[F::from_u64(9)]));
//! assert!(!circuit.is_satisfied(&columns, &[F::from_u64(10)]));
//! ```

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_mvpoly::{MVPoly, Sparse};
use arrabbiata_permutation::Permutation;
use arrabbiata_poly::Rotation;
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, typed::Label};

use crate::{
    Error,
    lookup::{Argument, Lookup},
};

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
/// let selector = None;
/// let polynomial = Sparse::<F, 1, 1>::zero();
/// let g = Gate::new(selector, polynomial);
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
    /// let selector = Some(1);
    /// let polynomial = Sparse::<F, 1, 1>::one();
    /// let g = Gate::new(selector, polynomial);
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
    /// let selector = Some(0);
    /// let polynomial = Sparse::<F, 1, 1>::zero();
    /// assert_eq!(Gate::new(selector, polynomial).selector(), Some(0));
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
    /// let selector = None;
    /// let polynomial = Sparse::<F, 1, 1>::zero();
    /// assert!(Gate::new(selector, polynomial).polynomial().is_zero());
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
/// let num_rows = 3;
/// let witness_columns = 1;
/// let fixed = vec![];
/// let selectors = vec![];
/// let variables = vec![];
/// let gates = vec![];
/// let sigma = (0..3).collect();
/// let public_inputs = 0;
/// let bindings = vec![];
/// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings);
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
    lookups: Vec<Lookup<F, N, D>>,
    permuted: Vec<usize>,
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
    /// let selector = None;
    /// let polynomial = Sparse::<F, 1, 1>::variable(Var::new::<0>()).unwrap();
    /// let gate = Gate::new(selector, polynomial);
    /// let num_rows = 2;
    /// let witness_columns = 1;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![gate];
    /// let sigma = vec![0, 1];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings);
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
        Permutation::new(witness_columns, n, sigma.clone()).map_err(|_| Error::Shape)?;
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
            lookups: Vec::new(),
            permuted: (0..witness_columns).collect(),
        })
    }

    /// The circuit with copy constraints on the witness columns `columns`
    /// only, `sigma` permuting their cells (cell `(c, i)` of the `c`-th
    /// listed column has index `c n + i`). A circuit from [`Circuit::new`]
    /// permutes every witness column; restricting the permutation to the
    /// columns that carry copies shrinks the argument to their number
    /// (Kimchi permutes 7 of its 15 columns).
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if a column repeats or does not exist, or `sigma`
    /// is not a permutation of the cells of `columns`.
    ///
    /// See `tests/permutation.rs` for circuits copying between some of
    /// their columns only.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::{Circuit, Error};
    /// use arrabbiata_pasta::Fq as F;
    /// let num_rows = 2;
    /// let witness_columns = 3;
    /// let sigma = (0..6).collect();
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
    /// let repeated = vec![2, 2];
    /// assert_eq!(c.clone().with_permutation(repeated, vec![0, 1, 2, 3]).unwrap_err(), Error::Shape);
    /// let only_column_2 = vec![2];
    /// let swap_its_rows = vec![1, 0];
    /// assert_eq!(c.with_permutation(only_column_2, swap_its_rows).unwrap().permuted(), &[2]);
    /// ```
    pub fn with_permutation(
        mut self,
        columns: Vec<usize>,
        sigma: Vec<usize>,
    ) -> Result<Self, Error> {
        let distinct = columns
            .iter()
            .enumerate()
            .all(|(i, c)| *c < self.witness_columns && !columns[..i].contains(c));
        if !distinct {
            return Err(Error::Shape);
        }
        Permutation::new(columns.len(), self.num_rows, sigma.clone()).map_err(|_| Error::Shape)?;
        self.permuted = columns;
        self.sigma = sigma;
        Ok(self)
    }

    /// The witness columns the copy permutation acts on, in the order of
    /// its cells.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let num_rows = 2;
    /// let witness_columns = 2;
    /// let sigma = (0..4).collect();
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
    /// assert_eq!(c.permuted(), &[0, 1]);
    /// ```
    #[must_use]
    pub fn permuted(&self) -> &[usize] {
        &self.permuted
    }

    /// The circuit with the given lookups attached.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if a lookup has no input, not one input per table
    /// column, names a fixed column or a selector that does not exist, has
    /// a selector that is not boolean, reads an undeclared variable, or is
    /// a plookup that is ungated or turned on on the last row (which
    /// plookup reserves).
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::{Argument, Circuit, Error, Lookup};
    /// use arrabbiata_mvpoly::{MVPoly, Sparse};
    /// use arrabbiata_pasta::Fq as F;
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![vec![F::ZERO; 2]];
    /// let selectors = vec![vec![F::ONE; 2]];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
    /// // A plookup on the last row.
    /// let argument = Argument::Plookup;
    /// let selector = Some(0);
    /// let inputs = vec![Sparse::zero()];
    /// let table = vec![0];
    /// let l = Lookup::new(argument, selector, inputs, table);
    /// assert_eq!(c.clone().with_lookups(vec![l]).unwrap_err(), Error::Shape);
    /// // LogUp has no reserved row.
    /// let argument = Argument::LogUp;
    /// let selector = Some(0);
    /// let inputs = vec![Sparse::zero()];
    /// let table = vec![0];
    /// let l = Lookup::new(argument, selector, inputs, table);
    /// assert!(c.with_lookups(vec![l]).is_ok());
    /// ```
    pub fn with_lookups(mut self, lookups: Vec<Lookup<F, N, D>>) -> Result<Self, Error> {
        let n = self.num_rows;
        let mut attached = Vec::with_capacity(lookups.len());
        for mut lookup in lookups {
            let shape = !lookup.inputs().is_empty()
                && lookup.inputs().len() == lookup.table().len()
                && lookup.table().iter().all(|k| *k < self.fixed.len())
                && lookup.inputs().iter().all(|p| self.declares(p));
            if !shape {
                return Err(Error::Shape);
            }
            match lookup.selector() {
                Some(q) => {
                    let column = self.selectors.get(q).ok_or(Error::Shape)?;
                    if column.iter().any(|v| !v.is_zero() && *v != F::ONE) {
                        return Err(Error::Shape);
                    }
                    if lookup.argument() == Argument::Plookup && !column[n - 1].is_zero() {
                        return Err(Error::Shape);
                    }
                }
                None => {
                    if lookup.argument() == Argument::Plookup {
                        return Err(Error::Shape);
                    }
                }
            }
            let default = lookup.table().iter().map(|k| self.fixed[*k][0]).collect();
            lookup.set_default(default);
            attached.push(lookup);
        }
        self.lookups = attached;
        Ok(self)
    }

    /// Whether `p` reads only declared variables.
    fn declares(&self, p: &Sparse<F, N, D>) -> bool {
        p.terms().iter().all(|(m, _)| {
            m.exponents()
                .iter()
                .enumerate()
                .all(|(i, e)| *e == 0 || i < self.variables.len())
        })
    }

    /// The lookups.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
    /// assert!(c.lookups().is_empty());
    /// ```
    #[must_use]
    pub fn lookups(&self) -> &[Lookup<F, N, D>] {
        &self.lookups
    }

    /// The number of rows, a power of two.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Circuit;
    /// use arrabbiata_pasta::Fq as F;
    /// let num_rows = 4;
    /// let witness_columns = 0;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 3;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = (0..6).collect();
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![vec![F::ONE; 2]];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![];
    /// let selectors = vec![vec![F::ONE, F::ZERO]];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![Variable::Public(0)];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 1;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 1;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![1, 0];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 0;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![];
    /// let public_inputs = 5;
    /// let bindings = vec![];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let num_rows = 2;
    /// let witness_columns = 1;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![0, 1];
    /// let public_inputs = 1;
    /// let bindings = vec![(0, 1)];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
    /// let selector = Some(0);
    /// let polynomial = var(0).checked_mul(&var(1)).unwrap();
    /// let gate = Gate::new(selector, polynomial);
    /// let vars = vec![Variable::Witness { column: 0, rotation: Rotation::CURR }, Variable::Public(0)];
    /// let num_rows = 2;
    /// let witness_columns = 1;
    /// let fixed = vec![];
    /// let selectors = vec![vec![F::ONE; 2]];
    /// let variables = vars;
    /// let gates = vec![gate];
    /// let sigma = vec![0, 1];
    /// let public_inputs = 1;
    /// let bindings = vec![];
    /// let c = Circuit::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
    /// assert_eq!(c.column_degree(&c.gates()[0]), 2);
    /// ```
    #[must_use]
    pub fn column_degree(&self, gate: &Gate<F, N, D>) -> usize {
        self.polynomial_degree(&gate.polynomial) + usize::from(gate.selector.is_some())
    }

    /// The degree of a lookup's query in the interpolated polynomials: the
    /// largest column degree of an input, plus one for the selector.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_generalized_plonk::{Argument, Circuit, Lookup, Variable};
    /// use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_poly::Rotation;
    /// let w = Sparse::<F, 1, 2>::variable(Var::new::<0>()).unwrap();
    /// let vars = vec![Variable::Witness { column: 0, rotation: Rotation::CURR }];
    /// let argument = Argument::LogUp;
    /// let selector = Some(0);
    /// let inputs = vec![w.checked_mul(&w).unwrap()];
    /// let table = vec![0];
    /// let l = Lookup::new(argument, selector, inputs, table);
    /// let num_rows = 2;
    /// let witness_columns = 1;
    /// let fixed = vec![vec![F::ZERO; 2]];
    /// let selectors = vec![vec![F::ONE; 2]];
    /// let variables = vars;
    /// let gates = vec![];
    /// let sigma = vec![0, 1];
    /// let public_inputs = 0;
    /// let bindings = vec![];
    /// let c = Circuit::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap().with_lookups(vec![l]).unwrap();
    /// assert_eq!(c.query_degree(&c.lookups()[0]), 3);
    /// ```
    #[must_use]
    pub fn query_degree(&self, lookup: &Lookup<F, N, D>) -> usize {
        let inputs = lookup
            .inputs()
            .iter()
            .map(|p| self.polynomial_degree(p))
            .max()
            .unwrap_or(0);
        inputs + usize::from(lookup.selector().is_some())
    }

    /// The largest number of column variables in a monomial of `p`.
    fn polynomial_degree(&self, p: &Sparse<F, N, D>) -> usize {
        let mut degree = 0;
        for (m, _) in p.terms() {
            let mut d = 0;
            for (i, e) in m.exponents().iter().enumerate() {
                if self.variables.get(i).is_some_and(Variable::is_column) {
                    d += *e as usize;
                }
            }
            degree = degree.max(d);
        }
        degree
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
    /// let num_rows = 2;
    /// let witness_columns = 1;
    /// let fixed = vec![];
    /// let selectors = vec![];
    /// let variables = vec![];
    /// let gates = vec![];
    /// let sigma = vec![1, 0];
    /// let public_inputs = 1;
    /// let bindings = vec![(0, 0)];
    /// let c = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
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
        for lookup in &self.lookups {
            if !self.lookup_holds(lookup, columns, public) {
                return false;
            }
        }
        let permuted: Vec<&[F]> = self
            .permuted
            .iter()
            .map(|c| columns[*c].as_slice())
            .collect();
        match Permutation::new(self.permuted.len(), n, self.sigma.clone()) {
            Ok(p) => p.is_respected_by(&permuted),
            Err(_) => false,
        }
    }
}

impl<F, const N: usize, const D: usize> Circuit<F, N, D>
where
    F: Field,
{
    /// Whether on every row `lookup` turns on, its inputs are a row of its
    /// table. Quadratic in the number of rows: a specification, not a
    /// prover.
    fn lookup_holds(&self, lookup: &Lookup<F, N, D>, columns: &[Vec<F>], public: &[F]) -> bool {
        let n = self.num_rows;
        for row in 0..n {
            let on = match lookup.selector() {
                Some(q) => !self.selectors[q][row].is_zero(),
                None => true,
            };
            if !on {
                continue;
            }
            let x = self.point(|v| match v {
                Variable::Witness { column, rotation } => columns[*column][rotation.apply(row, n)],
                Variable::Fixed { column, rotation } => self.fixed[*column][rotation.apply(row, n)],
                Variable::Public(i) => public[*i],
            });
            let tuple: Vec<F> = lookup.inputs().iter().map(|p| p.eval(&x)).collect();
            let found = (0..n).any(|j| {
                lookup
                    .table()
                    .iter()
                    .zip(&tuple)
                    .all(|(k, v)| self.fixed[*k][j] == *v)
            });
            if !found {
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
    lookups: &[Lookup<F, N, D>],
    permuted: &[usize],
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
        absorb_polynomial(transcript, &gate.polynomial);
    }
    absorb_length(transcript, bindings.len());
    for (j, i) in bindings {
        absorb_length(transcript, *j);
        absorb_length(transcript, *i);
    }
    absorb_length(transcript, permuted.len());
    for c in permuted {
        absorb_length(transcript, *c);
    }
    absorb_length(transcript, lookups.len());
    for lookup in lookups {
        absorb_length(transcript, lookup.argument() as usize);
        match lookup.selector() {
            None => absorb_length(transcript, 0),
            Some(q) => absorb_length(transcript, q + 1),
        }
        absorb_length(transcript, lookup.inputs().len());
        for (p, k) in lookup.inputs().iter().zip(lookup.table()) {
            absorb_length(transcript, *k);
            absorb_polynomial(transcript, p);
        }
        for v in lookup.default() {
            transcript.absorb(*v);
        }
    }
}

/// A polynomial's terms, with their number.
fn absorb_polynomial<F, T, const N: usize, const D: usize>(transcript: &mut T, p: &Sparse<F, N, D>)
where
    F: Field,
    T: Transcript<F>,
{
    let terms = p.terms();
    absorb_length(transcript, terms.len());
    for (m, c) in terms {
        for e in m.exponents() {
            absorb_length(transcript, *e as usize);
        }
        transcript.absorb(c);
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
/// // Sponge domain tags: each transcript gets its own, and prover and
/// // verifier start from the same one.
/// let transcript_domain = 0;
/// let num_rows = 2;
/// let witness_columns = 1;
/// let fixed = vec![];
/// let selectors = vec![];
/// let variables = vec![];
/// let gates = vec![];
/// let sigma = vec![0, 1];
/// let public_inputs = 0;
/// let bindings = vec![];
/// let a = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
/// let num_rows = 2;
/// let witness_columns = 1;
/// let fixed = vec![];
/// let selectors = vec![];
/// let variables = vec![];
/// let gates = vec![];
/// let sigma = vec![1, 0];
/// let public_inputs = 0;
/// let bindings = vec![];
/// let b = Circuit::<F, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
/// let (mut s, mut t) = (vesta_sponge(transcript_domain), vesta_sponge(transcript_domain));
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
            &self.lookups,
            &self.permuted,
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
