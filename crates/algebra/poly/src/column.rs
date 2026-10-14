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
//! let v: Variable<2, 0> = Variable::Witness(a, Rotation::NEXT);
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

/// Which row a variable refers to, relative to the row being constrained:
/// an offset `k` in `Z`, acting on the rows `Z/n` by translation,
/// `i |-> i + k mod n`.
///
/// Rows wrap around: on the last row, [`Rotation::NEXT`] is the first row,
/// and on the first row [`Rotation::PREV`] is the last. The action is a
/// group action of `Z` (through `Z -> Z/n`): `apply` of a composite is the
/// composite of the `apply`s, and `apply` of `n` is the identity.
///
/// ```
/// use arrabbiata_poly::Rotation;
/// assert_eq!(Rotation::NEXT.apply(7, 8), 0);
/// assert_eq!(Rotation::PREV.apply(0, 8), 7);
/// assert_eq!(Rotation::new(10).apply(3, 8), 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rotation(i32);

impl Rotation {
    /// The constrained row: the identity.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::CURR.offset(), 0);
    /// ```
    pub const CURR: Self = Self(0);

    /// The row after the constrained row.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::NEXT.offset(), 1);
    /// ```
    pub const NEXT: Self = Self(1);

    /// The row before the constrained row.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::PREV.offset(), -1);
    /// ```
    pub const PREV: Self = Self(-1);

    /// The rotation by `offset` rows.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::new(1), Rotation::NEXT);
    /// ```
    #[must_use]
    pub const fn new(offset: i32) -> Self {
        Self(offset)
    }

    /// The offset in rows.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::new(-3).offset(), -3);
    /// ```
    #[must_use]
    pub const fn offset(self) -> i32 {
        self.0
    }

    /// The rotation by `self` then `other`, or `None` if the offset
    /// overflows an `i32`.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::NEXT.compose(Rotation::PREV), Some(Rotation::CURR));
    /// assert_eq!(Rotation::new(i32::MAX).compose(Rotation::NEXT), None);
    /// ```
    #[must_use]
    pub const fn compose(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(offset) => Some(Self(offset)),
            None => None,
        }
    }

    /// The inverse rotation, or `None` for `i32::MIN`, whose negation is
    /// not an `i32`.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::NEXT.inverse(), Some(Rotation::PREV));
    /// assert_eq!(Rotation::new(i32::MIN).inverse(), None);
    /// ```
    #[must_use]
    pub const fn inverse(self) -> Option<Self> {
        match self.0.checked_neg() {
            Some(offset) => Some(Self(offset)),
            None => None,
        }
    }

    /// The zigzag code of the offset, `0, -1, 1, -2, 2, ... |-> 0, 1, 2, 3,
    /// 4, ...`: an injection of `i32` into `u64` (into `0..2^32`), used to
    /// absorb a rotation into a transcript.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::CURR.code(), 0);
    /// assert_eq!(Rotation::PREV.code(), 1);
    /// assert_eq!(Rotation::NEXT.code(), 2);
    /// assert_eq!(Rotation::new(i32::MIN).code(), u32::MAX as u64);
    /// ```
    #[must_use]
    pub const fn code(self) -> u64 {
        let k = self.0 as i64;
        ((k << 1) ^ (k >> 63)) as u64
    }

    /// The row this rotation designates from `row`, in a trace of
    /// `num_rows` rows: `row + offset mod num_rows`. `num_rows` must be
    /// non-zero.
    ///
    /// ```
    /// use arrabbiata_poly::Rotation;
    /// assert_eq!(Rotation::CURR.apply(3, 8), 3);
    /// assert_eq!(Rotation::NEXT.apply(3, 8), 4);
    /// assert_eq!(Rotation::NEXT.apply(7, 8), 0);
    /// assert_eq!(Rotation::new(-9).apply(0, 8), 7);
    /// ```
    #[must_use]
    pub const fn apply(self, row: usize, num_rows: usize) -> usize {
        let n = num_rows as i128;
        (row as i128 + self.0 as i128).rem_euclid(n) as usize
    }
}

/// A variable of a constraint: a column read at a rotation.
///
/// ```
/// use arrabbiata_poly::{FixedColumn, Rotation, Variable};
/// let selector: Variable<1, 1> = Variable::Fixed(FixedColumn::new::<0>(), Rotation::CURR);
/// # let _ = selector;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variable<const W: usize, const X: usize> {
    /// A witness column. Degree one.
    Witness(WitnessColumn<W>, Rotation),
    /// A fixed column. Degree zero.
    Fixed(FixedColumn<X>, Rotation),
}
