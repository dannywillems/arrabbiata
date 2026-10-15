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
//! ```compile_fail
//! use arrabbiata_poly::WitnessColumn;
//! // A 2-column circuit has no column 2: this is rejected by the compiler.
//! let _ = WitnessColumn::<2>::new::<2>();
//! ```

/// A witness (advice) column of a circuit with `W` witness columns.
///
/// Witness columns are private, committed, and folded linearly: they carry
/// degree one in the relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WitnessColumn<const W: usize>(usize);

impl<const W: usize> WitnessColumn<W> {
    /// The column with index `I`, checked against `W` at compile time.
    #[must_use]
    pub const fn new<const I: usize>() -> Self {
        const { assert!(I < W, "witness column index out of range") };
        Self(I)
    }

    /// The column with index `index`, or `None` when `index >= W`.
    #[must_use]
    pub const fn try_new(index: usize) -> Option<Self> {
        if index < W { Some(Self(index)) } else { None }
    }

    /// The index, guaranteed to be below `W`.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FixedColumn<const X: usize>(usize);

impl<const X: usize> FixedColumn<X> {
    /// The column with index `I`, checked against `X` at compile time.
    #[must_use]
    pub const fn new<const I: usize>() -> Self {
        const { assert!(I < X, "fixed column index out of range") };
        Self(I)
    }

    /// The column with index `index`, or `None` when `index >= X`.
    #[must_use]
    pub const fn try_new(index: usize) -> Option<Self> {
        if index < X { Some(Self(index)) } else { None }
    }

    /// The index, guaranteed to be below `X`.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rotation(i32);

impl Rotation {
    /// The constrained row: the identity.
    pub const CURR: Self = Self(0);

    /// The row after the constrained row.
    pub const NEXT: Self = Self(1);

    /// The row before the constrained row.
    pub const PREV: Self = Self(-1);

    /// The rotation by `offset` rows.
    #[must_use]
    pub const fn new(offset: i32) -> Self {
        Self(offset)
    }

    /// The offset in rows.
    #[must_use]
    pub const fn offset(self) -> i32 {
        self.0
    }

    /// The rotation by `self` then `other`, or `None` if the offset
    /// overflows an `i32`.
    #[must_use]
    pub const fn compose(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(offset) => Some(Self(offset)),
            None => None,
        }
    }

    /// The inverse rotation, or `None` for `i32::MIN`, whose negation is
    /// not an `i32`.
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
    #[must_use]
    pub const fn code(self) -> u64 {
        let k = self.0 as i64;
        ((k << 1) ^ (k >> 63)) as u64
    }

    /// The row this rotation designates from `row`, in a trace of
    /// `num_rows` rows: `row + offset mod num_rows`. `num_rows` must be
    /// non-zero.
    #[must_use]
    pub const fn apply(self, row: usize, num_rows: usize) -> usize {
        let n = num_rows as i128;
        (row as i128 + self.0 as i128).rem_euclid(n) as usize
    }
}

/// A variable of a constraint: a column read at a rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variable<const W: usize, const X: usize> {
    /// A witness column. Degree one.
    Witness(WitnessColumn<W>, Rotation),
    /// A fixed column. Degree zero.
    Fixed(FixedColumn<X>, Rotation),
}
