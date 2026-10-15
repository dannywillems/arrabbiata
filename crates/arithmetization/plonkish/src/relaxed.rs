//! The relaxation of a circuit: the relation an accumulation scheme folds.
//!
//! Fix a degree `D` at least the [degree](crate::gate::degree) of the gate
//! set in the witness columns. For a gadget `g` with constraints
//! `f_(g,0), ..., f_(g,m-1)`, homogenize each to degree `D` with a slack
//! scalar `u`, `f^D(u, w) = sum_j u^(D-j) f_j(w)` with `f_j` the component
//! of degree `j` in the witness ([`evaluate_homogeneous`]); fixed columns
//! are coefficients, of degree 0, since both instances of a fold share
//! them. A *relaxed witness* `(w, E)` satisfies the circuit for the slack
//! `u` and the coefficients `c_0, c_1, ...` when:
//!
//! - on every row `i` turning on gadget `g`,
//!   `sum_k c_k f^D_(g,k)(u, w at i) = E_i`; on every other row, `E_i = 0`;
//! - copied cells are equal;
//! - column 0 holds the public inputs on the public rows.
//!
//! Gadgets share the coefficients: the selectors are one-hot, so a row
//! reads one gadget, and `c` needs as many entries as the largest gadget
//! has constraints ([`crate::gate::combiner_length`]), not their sum.
//!
//! # The chart `u = 1, E = 0`
//!
//! At `u = 1` the homogenization is the constraint itself, so a witness
//! of the circuit, with a zero error column, satisfies the relaxation for
//! every `c`. Conversely, for `c = (1, a, a^2, ...)` with `a` drawn after
//! the witness is fixed, a relaxed witness with `u = 1` and `E = 0` is a
//! witness of the circuit except with probability at most `(m - 1) / |F|`
//! per row (a nonzero polynomial of degree `m - 1` in `a` has at most
//! `m - 1` roots). This is the embedding of the circuit relation into its
//! relaxation that an accumulation scheme starts from.
//!
//! # What folds
//!
//! The gate condition is homogeneous of degree `D + 1` in `(u, c, w, E)`
//! (the coefficients add one), so on the line through two relaxed
//! witnesses it is a polynomial of degree `D + 1` in the folding challenge
//! whose middle coefficients are the cross-terms. The copy and public-input
//! conditions are linear, so they hold on the whole line as soon as they
//! hold at both ends: they need no error term, and the decider checks
//! them once, at the end.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonkish::{RelaxedWitness, vanilla};
//!
//! let (c, w, ()) = build::<F, _, _>(|b| {
//!     let x = b.witness(F::from_u64(3));
//!     let y = b.mul(&x, &x);
//!     let out = b.public_input(F::from_u64(9));
//!     b.assert_equal(&y, &out);
//! });
//! let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
//! let relaxed = RelaxedWitness::embed(&witness);
//! // The chart u = 1, E = 0: any coefficients.
//! let coefficients = [F::from_u64(7)];
//! assert!(circuit.is_relaxed_satisfied(&relaxed, witness.public_inputs(), F::ONE, &coefficients, 2));
//! // Scaling the witness, the slack and the error together keeps it
//! // satisfied (homogeneity), here with E = 0.
//! let t = F::from_u64(5);
//! let scaled = RelaxedWitness::new(
//!     relaxed.columns().clone().map(|c| c.into_iter().map(|v| t * v).collect()),
//!     relaxed.error().to_vec(),
//! );
//! let public: Vec<F> = witness.public_inputs().iter().map(|v| t * *v).collect();
//! assert!(circuit.is_relaxed_satisfied(&scaled, &public, t, &coefficients, 2));
//! ```

use alloc::{vec, vec::Vec};

use arrabbiata_field::Field;
use arrabbiata_poly::{Scalars, evaluate_homogeneous};

use crate::{
    circuit::{Circuit, RowView, Witness},
    gate::GateSet,
};

/// A relaxed witness: `W` witness columns and an error column, all of `n`
/// values.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::RelaxedWitness;
/// let w = RelaxedWitness::new([vec![F::ONE, F::ZERO]], vec![F::ZERO, F::ONE]);
/// assert_eq!(w.error(), &[F::ZERO, F::ONE]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedWitness<F, const W: usize> {
    columns: [Vec<F>; W],
    error: Vec<F>,
}

impl<F, const W: usize> RelaxedWitness<F, W> {
    /// The relaxed witness with the given columns and error column.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::RelaxedWitness;
    /// let w = RelaxedWitness::new([vec![F::ONE]], vec![F::ZERO]);
    /// assert_eq!(w.columns()[0], vec![F::ONE]);
    /// ```
    #[must_use]
    pub const fn new(columns: [Vec<F>; W], error: Vec<F>) -> Self {
        Self { columns, error }
    }

    /// The witness columns.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::RelaxedWitness;
    /// let w = RelaxedWitness::new([vec![F::ONE], vec![F::ZERO]], vec![F::ZERO]);
    /// assert_eq!(w.columns().len(), 2);
    /// ```
    #[must_use]
    pub const fn columns(&self) -> &[Vec<F>; W] {
        &self.columns
    }

    /// The error column.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::RelaxedWitness;
    /// assert_eq!(RelaxedWitness::new([vec![F::ONE]], vec![F::ONE]).error(), &[F::ONE]);
    /// ```
    #[must_use]
    pub fn error(&self) -> &[F] {
        &self.error
    }
}

impl<F, const W: usize> RelaxedWitness<F, W>
where
    F: Field,
{
    /// A witness of the circuit as a relaxed witness: the same columns and
    /// a zero error column, the chart `E = 0`.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::{RelaxedWitness, Witness};
    /// let w = Witness::new([vec![F::ONE, F::from_u64(2)]], vec![]);
    /// let r = RelaxedWitness::embed(&w);
    /// assert_eq!(r.columns(), w.columns());
    /// assert_eq!(r.error(), &[F::ZERO, F::ZERO]);
    /// ```
    #[must_use]
    pub fn embed(witness: &Witness<F, W>) -> Self {
        let n = if W == 0 {
            0
        } else {
            witness.columns()[0].len()
        };
        Self {
            columns: witness.columns().clone(),
            error: vec![F::ZERO; n],
        }
    }
}

impl<F, G, const W: usize, const X: usize, const Q: usize> Circuit<F, G, W, X, Q>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    /// The left-hand side of the relaxed gate condition on `row`:
    /// `sum_k c_k f^D_(g,k)(u, w)` for the gadget `g` the row turns on, `0`
    /// for a row with no gadget.
    ///
    /// `None` when `degree` is below the degree of a constraint of `g`, or
    /// when `coefficients` has fewer entries than `g` has constraints; the
    /// columns must have `n` values each.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::{Builder, vanilla::{Arithmetic, Vanilla}};
    /// // a * b - c, with q_M = 1, q_O = -1.
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// let one = F::ONE;
    /// let row = b.row::<Arithmetic, _>([F::from_u64(2), F::from_u64(3), F::from_u64(7)], [F::ZERO, F::ZERO, -one, one, F::ZERO]);
    /// let (circuit, w) = b.build().unwrap();
    /// // a b - u c at u = 2: 6 - 14.
    /// let lhs = circuit.relaxed_row(w.columns(), row, F::from_u64(2), &[one], 2);
    /// assert_eq!(lhs, Some(F::from_u64(6) - F::from_u64(14)));
    /// // Degree 1 cannot hold a b.
    /// assert_eq!(circuit.relaxed_row(w.columns(), row, one, &[one], 1), None);
    /// ```
    #[must_use]
    pub fn relaxed_row(
        &self,
        columns: &[Vec<F>; W],
        row: usize,
        u: F,
        coefficients: &[F],
        degree: usize,
    ) -> Option<F> {
        let Some(gadget) = self.gadgets()[row] else {
            return Some(F::ZERO);
        };
        let constraints = G::constraints(gadget.index());
        if coefficients.len() < constraints.len() {
            return None;
        }
        let view = RowView {
            witness: columns,
            fixed: self.fixed(),
            row,
            num_rows: self.num_rows(),
        };
        let ring = Scalars::new();
        let mut value = F::ZERO;
        for (c, k) in constraints.iter().zip(coefficients) {
            value = value + *k * evaluate_homogeneous(&ring, c, &view, &u, degree)?;
        }
        Some(value)
    }

    /// Whether `witness` satisfies the relaxation of the circuit of degree
    /// `degree`, for the slack `u`, the coefficients `coefficients` and the
    /// public inputs `public`: see the [module documentation](self).
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::{Builder, RelaxedWitness, vanilla::{Arithmetic, Vanilla}};
    /// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    /// let one = F::ONE;
    /// b.row::<Arithmetic, _>([F::from_u64(2), F::from_u64(3), F::from_u64(7)], [F::ZERO, F::ZERO, -one, one, F::ZERO]);
    /// let (circuit, w) = b.build().unwrap();
    /// // 2 * 3 != 7, but the error column can absorb the difference.
    /// assert!(!circuit.is_satisfied(&w));
    /// let error = vec![-one, F::ZERO];
    /// let relaxed = RelaxedWitness::new(w.columns().clone(), error);
    /// assert!(circuit.is_relaxed_satisfied(&relaxed, &[], one, &[one], 2));
    /// ```
    #[must_use]
    pub fn is_relaxed_satisfied(
        &self,
        witness: &RelaxedWitness<F, W>,
        public: &[F],
        u: F,
        coefficients: &[F],
        degree: usize,
    ) -> bool {
        let n = self.num_rows();
        if witness.columns.iter().any(|c| c.len() != n)
            || witness.error.len() != n
            || public.len() != self.public_rows().len()
        {
            return false;
        }
        for row in 0..n {
            match self.relaxed_row(&witness.columns, row, u, coefficients, degree) {
                Some(value) if value == witness.error[row] => {}
                _ => return false,
            }
        }
        if !witness.columns.is_empty() {
            for (row, value) in self.public_rows().iter().zip(public) {
                if witness.columns[0][*row] != *value {
                    return false;
                }
            }
        }
        for (cell, target) in self.sigma().iter().enumerate() {
            if witness.columns[cell / n][cell % n] != witness.columns[target / n][target % n] {
                return false;
            }
        }
        true
    }
}
