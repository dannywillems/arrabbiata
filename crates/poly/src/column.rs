//! Typed column handles.
//!
//! A circuit with `W` witness columns and `X` fixed columns can only name
//! columns `0..W` and `0..X`. The bound is part of the type: a
//! [`WitnessColumn<W>`] cannot be built out of range, and a column of a
//! 3-column circuit cannot be used in an expression of a 4-column circuit.
//!
//! When the index is a literal, [`WitnessColumn::new`] checks the bound at
//! compile time.
//!
//! ```
//! use arrabbiata_poly::{Rotation, Variable, WitnessColumn};
//!
//! let a = WitnessColumn::<2>::new::<1>();
//! let v: Variable<2, 0> = Variable::Witness(a, Rotation::Next);
//! assert_eq!(a.index(), 1);
//! # let _ = v;
//! ```
//!
//! ```compile_fail
//! use arrabbiata_poly::WitnessColumn;
//! // A 2-column circuit has no column 2: this is rejected by the compiler.
//! let _ = WitnessColumn::<2>::new::<2>();
//! ```

/// A witness (advice) column of a circuit with `W` witness columns.
///
/// Witness columns are private, committed, and folded linearly: they carry
/// degree one in the relation.
///
/// ```
/// use arrabbiata_poly::WitnessColumn;
/// assert_eq!(WitnessColumn::<3>::new::<2>().index(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WitnessColumn<const W: usize>(usize);

impl<const W: usize> WitnessColumn<W> {
    /// The column with index `I`, checked against `W` at compile time.
    ///
    /// ```
    /// use arrabbiata_poly::WitnessColumn;
    /// let c = WitnessColumn::<4>::new::<3>();
    /// assert_eq!(c.index(), 3);
    /// ```
    #[must_use]
    pub const fn new<const I: usize>() -> Self {
        const { assert!(I < W, "witness column index out of range") };
        Self(I)
    }

    /// The column with index `index`, or `None` when `index >= W`.
    ///
    /// ```
    /// use arrabbiata_poly::WitnessColumn;
    /// assert!(WitnessColumn::<2>::try_new(1).is_some());
    /// assert!(WitnessColumn::<2>::try_new(2).is_none());
    /// ```
    #[must_use]
    pub const fn try_new(index: usize) -> Option<Self> {
        if index < W { Some(Self(index)) } else { None }
    }

    /// The index, guaranteed to be below `W`.
    ///
    /// ```
    /// use arrabbiata_poly::WitnessColumn;
    /// assert_eq!(WitnessColumn::<5>::new::<0>().index(), 0);
    /// ```
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A fixed (preprocessed) column of a circuit with `X` fixed columns.
///
/// Fixed columns are part of the circuit, identical in every instance, and
/// therefore never folded: they carry degree zero, like constants. Selectors
/// are fixed columns.
///
/// ```
/// use arrabbiata_poly::FixedColumn;
/// assert_eq!(FixedColumn::<1>::new::<0>().index(), 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FixedColumn<const X: usize>(usize);

impl<const X: usize> FixedColumn<X> {
    /// The column with index `I`, checked against `X` at compile time.
    ///
    /// ```
    /// use arrabbiata_poly::FixedColumn;
    /// assert_eq!(FixedColumn::<2>::new::<1>().index(), 1);
    /// ```
    #[must_use]
    pub const fn new<const I: usize>() -> Self {
        const { assert!(I < X, "fixed column index out of range") };
        Self(I)
    }

    /// The column with index `index`, or `None` when `index >= X`.
    ///
    /// ```
    /// use arrabbiata_poly::FixedColumn;
    /// assert!(FixedColumn::<0>::try_new(0).is_none());
    /// ```
    #[must_use]
    pub const fn try_new(index: usize) -> Option<Self> {
        if index < X { Some(Self(index)) } else { None }
    }

    /// The index, guaranteed to be below `X`.
    ///
    /// ```
    /// use arrabbiata_poly::FixedColumn;
    /// assert_eq!(FixedColumn::<3>::new::<2>().index(), 2);
    /// ```
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Which row a variable refers to, relative to the row being constrained.
///
/// Rows wrap around: on the last row, [`Rotation::Next`] is the first row.
///
/// ```
/// use arrabbiata_poly::Rotation;
/// assert_eq!(Rotation::Next.apply(7, 8), 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rotation {
    /// The constrained row.
    Curr,
    /// The row after it.
    Next,
}

impl Rotation {
    /// The row index this rotation designates from `row`, in a trace of
    /// `num_rows` rows. `num_rows` must be non-zero.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::Curr.apply(3, 8), 3);
    /// assert_eq!(Rotation::Next.apply(3, 8), 4);
    /// assert_eq!(Rotation::Next.apply(7, 8), 0);
    /// ```
    #[must_use]
    pub const fn apply(self, row: usize, num_rows: usize) -> usize {
        match self {
            Self::Curr => row,
            Self::Next => {
                if row + 1 == num_rows {
                    0
                } else {
                    row + 1
                }
            }
        }
    }
}

/// A variable of a constraint: a column read at a rotation.
///
/// ```
/// use arrabbiata_poly::{FixedColumn, Rotation, Variable};
/// let selector: Variable<1, 1> = Variable::Fixed(FixedColumn::new::<0>(), Rotation::Curr);
/// # let _ = selector;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variable<const W: usize, const X: usize> {
    /// A witness column. Degree one.
    Witness(WitnessColumn<W>, Rotation),
    /// A fixed column. Degree zero.
    Fixed(FixedColumn<X>, Rotation),
}
