//! Constraint systems, traces, and the relations they define.

use alloc::vec::Vec;
use core::{marker::PhantomData, num::NonZeroUsize};

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::Field;
use arrabbiata_poly::{
    Algebra, Assignment, Expr, FixedColumn, Rotation, Scalars, Variable, WitnessColumn, evaluate,
    evaluate_homogeneous,
};
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, tag, typed::Label};

use crate::{
    aggregation::Aggregation,
    error::{RelationError, ShapeError, Slot},
    instance::{Instance, RelaxedInstance, RelaxedWitness},
};

/// A circuit: `M` polynomial constraints of degree at most `D` over `W`
/// witness and `X` fixed columns, imposed on every row of a trace of
/// `num_rows` rows, and aggregated by the strategy `A`.
///
/// The constraint system defines three relations:
///
/// - the *strict* relation on traces: every constraint vanishes on every row
///   ([`ConstraintSystem::check_trace`]);
/// - the *relaxed* relation on `(u, c, w, E)`: on every row `i`,
///   `sum_j c_j f_j^D(u, w_i) = E_i` ([`ConstraintSystem::check_relaxed_witness`]);
/// - the *committed* relaxed relation, which adds that the instance's
///   commitments open to the witness ([`ConstraintSystem::check_relaxed`]).
///
/// The relaxed relation is homogeneous of degree
/// [`ConstraintSystem::DEGREE`] in `(u, c, w)`, which is what makes it
/// closed under folding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintSystem<F, A, const W: usize, const X: usize, const M: usize, const D: usize> {
    num_rows: NonZeroUsize,
    fixed: [Vec<F>; X],
    constraints: [Expr<F, W, X>; M],
    aggregation: PhantomData<A>,
}

/// A trace that has one value per row in every witness column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace<F, const W: usize> {
    columns: [Vec<F>; W],
}

impl<F, const W: usize> Trace<F, W> {
    /// The witness columns.
    #[must_use]
    pub const fn columns(&self) -> &[Vec<F>; W] {
        &self.columns
    }
}

/// One row of a trace, as an assignment of the circuit variables in `F`.
#[derive(Clone, Copy, Debug)]
pub struct RowView<'a, F, const W: usize, const X: usize> {
    witness: &'a [Vec<F>; W],
    fixed: &'a [Vec<F>; X],
    row: usize,
    num_rows: usize,
}

impl<'a, F, const W: usize, const X: usize> RowView<'a, F, W, X> {
    /// The view of row `row` of a trace of `num_rows` rows. Every column
    /// must have `num_rows` values.
    #[must_use]
    pub const fn new(
        witness: &'a [Vec<F>; W],
        fixed: &'a [Vec<F>; X],
        row: usize,
        num_rows: usize,
    ) -> Self {
        Self {
            witness,
            fixed,
            row,
            num_rows,
        }
    }
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

impl<F, A, const W: usize, const X: usize, const M: usize, const D: usize>
    ConstraintSystem<F, A, W, X, M, D>
where
    F: Field,
    A: Aggregation<F, M>,
{
    /// The degree of the relaxed relation in `(u, c, w)`: `D`, plus one when
    /// the aggregation coefficients are variables.
    pub const DEGREE: usize = D + A::EXTRA_DEGREE;

    /// The number of cross-terms a fold produces: `DEGREE - 1`.
    pub const CROSS_TERMS: usize = D + A::EXTRA_DEGREE - 1;

    /// A constraint system with `num_rows` rows, the given fixed columns
    /// (one value per row each) and constraints.
    ///
    /// The bounds `M >= 1`, `D >= 1` and `W >= 1` are checked at compile
    /// time; the lengths of the fixed columns and the degrees of the
    /// constraints, which are data, at run time.
    ///
    /// # Errors
    ///
    /// [`ShapeError::FixedColumnLength`] if a fixed column does not have
    /// `num_rows` values, and [`ShapeError::ConstraintDegree`] if a
    /// constraint has degree above `D`.
    pub fn new(
        num_rows: NonZeroUsize,
        fixed: [Vec<F>; X],
        constraints: [Expr<F, W, X>; M],
    ) -> Result<Self, ShapeError> {
        const {
            assert!(M >= 1, "a constraint system needs a constraint");
            assert!(D >= 1, "the homogenization degree must be at least 1");
            assert!(W >= 1, "a constraint system needs a witness column");
        };
        let mut column = 0;
        while column < X {
            if fixed[column].len() != num_rows.get() {
                return Err(ShapeError::FixedColumnLength {
                    column,
                    expected: num_rows.get(),
                    found: fixed[column].len(),
                });
            }
            column += 1;
        }
        let mut j = 0;
        while j < M {
            let degree = constraints[j].degree();
            if degree > D {
                return Err(ShapeError::ConstraintDegree {
                    constraint: j,
                    degree,
                    maximum: D,
                });
            }
            j += 1;
        }
        Ok(Self {
            num_rows,
            fixed,
            constraints,
            aggregation: PhantomData,
        })
    }

    /// The number of rows.
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows.get()
    }

    /// The fixed columns.
    #[must_use]
    pub const fn fixed_columns(&self) -> &[Vec<F>; X] {
        &self.fixed
    }

    /// The constraints.
    #[must_use]
    pub const fn constraints(&self) -> &[Expr<F, W, X>; M] {
        &self.constraints
    }

    /// Check that `columns` has one value per row in each column, and wrap
    /// it as a [`Trace`].
    ///
    /// # Errors
    ///
    /// [`ShapeError::WitnessColumnLength`] on the first column of the wrong
    /// length.
    pub fn trace(&self, columns: [Vec<F>; W]) -> Result<Trace<F, W>, ShapeError> {
        self.check_witness_columns(&columns)?;
        Ok(Trace { columns })
    }

    fn check_witness_columns(&self, columns: &[Vec<F>; W]) -> Result<(), ShapeError> {
        let mut column = 0;
        while column < W {
            if columns[column].len() != self.num_rows() {
                return Err(ShapeError::WitnessColumnLength {
                    column,
                    expected: self.num_rows(),
                    found: columns[column].len(),
                });
            }
            column += 1;
        }
        Ok(())
    }

    /// The left-hand side of the relaxed relation on one row,
    /// `sum_j c_j f_j^D(u, w)`, evaluated in the algebra `ring`.
    ///
    /// `coefficients` holds the lifted `c_j`, one per constraint, or is
    /// `None` when the aggregation has no coefficients. This is the single
    /// definition of the relation's polynomial: the relaxed check evaluates
    /// it in `F`, and the folding prover in `F[r]` to get the cross-terms.
    pub fn evaluate_row<R, V>(
        &self,
        ring: &R,
        row: &V,
        u: &R::Elem,
        coefficients: Option<&[R::Elem]>,
    ) -> R::Elem
    where
        R: Algebra<F>,
        V: Assignment<F, R, W, X>,
    {
        let mut acc = ring.zero();
        let mut j = 0;
        while j < M {
            let value = match evaluate_homogeneous(ring, &self.constraints[j], row, u, D) {
                Some(value) => value,
                // Every constraint has degree at most D, checked in `new`.
                None => unreachable!("constraint degree exceeds D"),
            };
            let term = match coefficients {
                Some(c) => ring.mul(&c[j], &value),
                None => value,
            };
            acc = ring.add(&acc, &term);
            j += 1;
        }
        acc
    }

    /// Check the strict relation: every constraint vanishes on every row.
    ///
    /// # Errors
    ///
    /// [`RelationError::Unsatisfied`] naming the first failing row and
    /// constraint.
    pub fn check_trace(&self, trace: &Trace<F, W>) -> Result<(), RelationError> {
        let ring = Scalars::new();
        let mut row = 0;
        while row < self.num_rows() {
            let view = RowView::new(&trace.columns, &self.fixed, row, self.num_rows());
            let mut j = 0;
            while j < M {
                if !evaluate(&ring, &self.constraints[j], &view).is_zero() {
                    return Err(RelationError::Unsatisfied {
                        row,
                        constraint: Some(j),
                    });
                }
                j += 1;
            }
            row += 1;
        }
        Ok(())
    }

    /// Check the relaxed relation `sum_j c_j f_j^D(u, w_i) = E_i` on every
    /// row, without commitments.
    ///
    /// # Errors
    ///
    /// [`RelationError::Shape`] if the witness has the wrong shape, and
    /// [`RelationError::Unsatisfied`] on the first failing row.
    pub fn check_relaxed_witness(
        &self,
        u: F,
        coefficients: &A::Coefficients,
        witness: &RelaxedWitness<F, W>,
    ) -> Result<(), RelationError> {
        self.check_witness_columns(witness.columns())?;
        if witness.error().len() != self.num_rows() {
            return Err(ShapeError::ErrorLength {
                expected: self.num_rows(),
                found: witness.error().len(),
            }
            .into());
        }
        let ring = Scalars::new();
        let lifted = lift_scalars::<F, A, M>(coefficients);
        let lifted_ref = lifted.as_deref();
        let mut row = 0;
        while row < self.num_rows() {
            let view = RowView::new(witness.columns(), &self.fixed, row, self.num_rows());
            let value = self.evaluate_row(&ring, &view, &u, lifted_ref);
            if value != witness.error()[row] {
                return Err(RelationError::Unsatisfied {
                    row,
                    constraint: None,
                });
            }
            row += 1;
        }
        Ok(())
    }

    /// Check the committed relaxed relation: the commitments of `instance`
    /// open to `witness` under `ck`, and the relaxed relation holds.
    ///
    /// # Errors
    ///
    /// [`RelationError::CommitmentMismatch`] on the first commitment that
    /// does not open, otherwise as [`ConstraintSystem::check_relaxed_witness`].
    pub fn check_relaxed<PC>(
        &self,
        ck: &PC,
        instance: &RelaxedInstance<F, PC::Commitment, A::Coefficients, W>,
        witness: &RelaxedWitness<F, W>,
    ) -> Result<(), RelationError>
    where
        PC: HomomorphicCommitment<F>,
    {
        self.check_witness_columns(witness.columns())?;
        let mut column = 0;
        while column < W {
            if ck.commit(&witness.columns()[column]) != instance.columns()[column] {
                return Err(RelationError::CommitmentMismatch(Slot::Column(column)));
            }
            column += 1;
        }
        if ck.commit(witness.error()) != *instance.error() {
            return Err(RelationError::CommitmentMismatch(Slot::Error));
        }
        self.check_relaxed_witness(instance.u(), instance.coefficients(), witness)
    }

    /// Commit to a trace and draw the aggregation coefficients: the fresh
    /// instance of a strict witness.
    ///
    /// The column commitments are absorbed before the coefficients are
    /// sampled, so the coefficients bind the witness.
    pub fn commit<PC, T>(
        &self,
        ck: &PC,
        trace: &Trace<F, W>,
        transcript: &mut T,
    ) -> Instance<PC::Commitment, A::Coefficients, W>
    where
        PC: HomomorphicCommitment<F>,
        T: Transcript<F>,
    {
        let first = ck.commit(&trace.columns[0]);
        let mut columns: [PC::Commitment; W] = core::array::from_fn(|_| first.clone());
        let mut column = 1;
        while column < W {
            columns[column] = ck.commit(&trace.columns[column]);
            column += 1;
        }
        let mut c = 0;
        while c < W {
            columns[c].absorb_into(transcript);
            c += 1;
        }
        let coefficients = A::sample(transcript);
        Instance::new(columns, coefficients)
    }

    /// The relaxed instance and witness of a fresh instance: `u = 1` and a
    /// zero error vector.
    pub fn relax<PC>(
        &self,
        ck: &PC,
        instance: Instance<PC::Commitment, A::Coefficients, W>,
        trace: Trace<F, W>,
    ) -> (
        RelaxedInstance<F, PC::Commitment, A::Coefficients, W>,
        RelaxedWitness<F, W>,
    )
    where
        PC: HomomorphicCommitment<F>,
    {
        let error = alloc::vec![F::ZERO; self.num_rows()];
        let error_commitment = ck.commit(&error);
        let (columns, coefficients) = instance.into_parts();
        (
            RelaxedInstance::new(F::ONE, coefficients, columns, error_commitment),
            RelaxedWitness::new(trace.columns, error),
        )
    }
}

/// The aggregation coefficients as field elements, or `None` when the
/// strategy has none.
fn lift_scalars<F, A, const M: usize>(coefficients: &A::Coefficients) -> Option<Vec<F>>
where
    F: Field,
    A: Aggregation<F, M>,
{
    A::coefficient(coefficients, 0)?;
    let mut out = Vec::with_capacity(M);
    let mut j = 0;
    while j < M {
        out.push(match A::coefficient(coefficients, j) {
            Some(c) => c,
            None => F::ONE,
        });
        j += 1;
    }
    Some(out)
}

impl<F, A, const W: usize, const X: usize, const M: usize, const D: usize> Label
    for ConstraintSystem<F, A, W, X, M, D>
{
    const LABEL: u64 = tag::CONSTRAINT_SYSTEM;
}

impl<F, A, const W: usize, const X: usize, const M: usize, const D: usize> Absorb<F>
    for ConstraintSystem<F, A, W, X, M, D>
where
    F: Field,
    A: Aggregation<F, M>,
{
    /// Absorb the whole circuit: dimensions, degree, aggregation, fixed
    /// columns and constraints. Expressions are absorbed in prefix order with
    /// a tag per node, which is an injective (prefix-free) encoding.
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::CONSTRAINT_SYSTEM));
        absorb_length(transcript, self.num_rows());
        absorb_length(transcript, W);
        absorb_length(transcript, X);
        absorb_length(transcript, M);
        absorb_length(transcript, D);
        absorb_length(transcript, A::EXTRA_DEGREE);
        let mut column = 0;
        while column < X {
            let mut row = 0;
            while row < self.num_rows() {
                transcript.absorb(self.fixed[column][row]);
                row += 1;
            }
            column += 1;
        }
        let mut j = 0;
        while j < M {
            absorb_expr(&self.constraints[j], transcript);
            j += 1;
        }
    }
}

fn absorb_expr<F, T, const W: usize, const X: usize>(expr: &Expr<F, W, X>, transcript: &mut T)
where
    F: Field,
    T: Transcript<F>,
{
    match expr {
        Expr::Constant(c) => {
            transcript.absorb(F::from_u64(0));
            transcript.absorb(*c);
        }
        Expr::Variable(Variable::Witness(column, rotation)) => {
            transcript.absorb(F::from_u64(1));
            absorb_length(transcript, column.index());
            transcript.absorb(F::from_u64(Rotation::code(*rotation)));
        }
        Expr::Variable(Variable::Fixed(column, rotation)) => {
            transcript.absorb(F::from_u64(2));
            absorb_length(transcript, column.index());
            transcript.absorb(F::from_u64(Rotation::code(*rotation)));
        }
        Expr::Neg(a) => {
            transcript.absorb(F::from_u64(3));
            absorb_expr(a, transcript);
        }
        Expr::Add(a, b) => {
            transcript.absorb(F::from_u64(4));
            absorb_expr(a, transcript);
            absorb_expr(b, transcript);
        }
        Expr::Mul(a, b) => {
            transcript.absorb(F::from_u64(5));
            absorb_expr(a, transcript);
            absorb_expr(b, transcript);
        }
    }
}
