//! Errors.

use core::fmt;

/// A value does not have the shape the constraint system requires.
///
/// ```
/// use arrabbiata_relation::error::ShapeError;
/// let e = ShapeError::ConstraintDegree { constraint: 0, degree: 3, maximum: 2 };
/// assert_eq!(e.to_string(), "constraint 0 has degree 3, above the maximum 2");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// A fixed column does not have one value per row.
    FixedColumnLength {
        /// The column.
        column: usize,
        /// The number of rows.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// A witness column does not have one value per row.
    WitnessColumnLength {
        /// The column.
        column: usize,
        /// The number of rows.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// The error vector does not have one value per row.
    ErrorLength {
        /// The number of rows.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// A constraint has a degree above the declared maximum `D`.
    ConstraintDegree {
        /// The constraint.
        constraint: usize,
        /// Its degree.
        degree: usize,
        /// The declared maximum.
        maximum: usize,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedColumnLength {
                column,
                expected,
                found,
            } => write!(
                f,
                "fixed column {column} has {found} values, expected {expected}"
            ),
            Self::WitnessColumnLength {
                column,
                expected,
                found,
            } => write!(
                f,
                "witness column {column} has {found} values, expected {expected}"
            ),
            Self::ErrorLength { expected, found } => {
                write!(f, "error vector has {found} values, expected {expected}")
            }
            Self::ConstraintDegree {
                constraint,
                degree,
                maximum,
            } => write!(
                f,
                "constraint {constraint} has degree {degree}, above the maximum {maximum}"
            ),
        }
    }
}

impl core::error::Error for ShapeError {}

/// Which committed vector of an instance a commitment check failed on.
///
/// ```
/// use arrabbiata_relation::error::Slot;
/// assert_ne!(Slot::Column(0), Slot::Error);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// A witness column.
    Column(usize),
    /// The error vector.
    Error,
}

/// A witness does not satisfy a relation.
///
/// ```
/// use arrabbiata_relation::error::RelationError;
/// let e = RelationError::Unsatisfied { row: 3, constraint: Some(1) };
/// assert_eq!(e.to_string(), "constraint 1 is not satisfied on row 3");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationError {
    /// The witness has the wrong shape.
    Shape(ShapeError),
    /// A row does not satisfy the relation. `constraint` names the failing
    /// constraint when it can be isolated (strict relation), and is `None`
    /// for the aggregated relaxed equation.
    Unsatisfied {
        /// The row.
        row: usize,
        /// The constraint, when known.
        constraint: Option<usize>,
    },
    /// A commitment of the instance does not open to the witness.
    CommitmentMismatch(Slot),
}

impl fmt::Display for RelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(e) => write!(f, "shape error: {e}"),
            Self::Unsatisfied {
                row,
                constraint: Some(j),
            } => write!(f, "constraint {j} is not satisfied on row {row}"),
            Self::Unsatisfied {
                row,
                constraint: None,
            } => write!(f, "the relaxed relation is not satisfied on row {row}"),
            Self::CommitmentMismatch(Slot::Column(c)) => {
                write!(f, "the commitment to witness column {c} does not match")
            }
            Self::CommitmentMismatch(Slot::Error) => {
                write!(f, "the commitment to the error vector does not match")
            }
        }
    }
}

impl core::error::Error for RelationError {}

impl From<ShapeError> for RelationError {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}
