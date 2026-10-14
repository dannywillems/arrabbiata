//! Small example circuits over [`Goldilocks`], used throughout the
//! documentation and the tests of the workspace.
//!
//! - [`squaring`]: one constraint of degree 2, aggregated with
//!   [`Single`]. Its relaxed relation is exactly Nova's relaxed R1CS for the
//!   row-local constraint `next = curr^2`: the relation has degree 2 and a
//!   fold has one cross-term.
//! - [`fibonacci`]: two constraints of degree 2 (the "multiplicative
//!   Fibonacci" sequence `(x, y) -> (y, x * y)`), aggregated with
//!   [`Powers`]. The relation has degree 3 and a fold has two cross-terms.
//!
//! Both gate their constraints with a selector that is off on the last row,
//! since [`Rotation::Next`] wraps around.
//!
//! ```
//! use arrabbiata_relation::examples;
//!
//! let cs = examples::fibonacci(6);
//! let trace = examples::fibonacci_trace(&cs, 2, 3);
//! assert!(cs.check_trace(&trace).is_ok());
//! ```

use alloc::{vec, vec::Vec};
use core::num::NonZeroUsize;

use arrabbiata_field::{Field, Goldilocks};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};

use crate::{
    aggregation::{Powers, Single},
    system::{ConstraintSystem, Trace},
};

/// The squaring circuit: one witness column, one selector, one constraint
/// `q * (next - curr^2)` of degree 2.
///
/// ```
/// use arrabbiata_relation::examples::{Squaring, squaring};
/// let cs: Squaring = squaring(4);
/// assert_eq!(Squaring::CROSS_TERMS, 1);
/// # let _ = cs;
/// ```
pub type Squaring = ConstraintSystem<Goldilocks, Single, 1, 1, 1, 2>;

/// The multiplicative Fibonacci circuit: two witness columns, one selector,
/// constraints `q * (x_next - y)` and `q * (y_next - x * y)`.
///
/// ```
/// use arrabbiata_relation::examples::{Fibonacci, fibonacci};
/// let cs: Fibonacci = fibonacci(4);
/// assert_eq!(Fibonacci::DEGREE, 3);
/// # let _ = cs;
/// ```
pub type Fibonacci = ConstraintSystem<Goldilocks, Powers, 2, 1, 2, 2>;

/// The selector `1, ..., 1, 0` of `num_rows` rows.
fn selector(num_rows: usize) -> Vec<Goldilocks> {
    let mut q = vec![Goldilocks::ONE; num_rows];
    q[num_rows - 1] = Goldilocks::ZERO;
    q
}

fn rows(num_rows: usize) -> NonZeroUsize {
    match NonZeroUsize::new(num_rows) {
        Some(n) => n,
        None => panic!("an example circuit needs at least one row"),
    }
}

/// The squaring circuit on `num_rows` rows.
///
/// # Panics
///
/// If `num_rows` is zero.
///
/// ```
/// use arrabbiata_relation::examples::squaring;
/// assert_eq!(squaring(16).num_rows(), 16);
/// ```
#[must_use]
pub fn squaring(num_rows: usize) -> Squaring {
    type E = Expr<Goldilocks, 1, 1>;
    let q = E::fixed(FixedColumn::new::<0>(), Rotation::Curr);
    let curr = E::witness(WitnessColumn::new::<0>(), Rotation::Curr);
    let next = E::witness(WitnessColumn::new::<0>(), Rotation::Next);
    let constraint = q * (next - curr.clone() * curr);
    match Squaring::new(rows(num_rows), [selector(num_rows)], [constraint]) {
        Ok(cs) => cs,
        Err(e) => panic!("the squaring circuit is well-formed: {e}"),
    }
}

/// The trace `x0, x0^2, x0^4, ...` of the squaring circuit.
///
/// ```
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_relation::examples::{squaring, squaring_trace};
/// let cs = squaring(3);
/// assert_eq!(squaring_trace(&cs, 3).columns()[0], vec![F::new(3), F::new(9), F::new(81)]);
/// ```
#[must_use]
pub fn squaring_trace(cs: &Squaring, x0: u64) -> Trace<Goldilocks, 1> {
    let mut x = Vec::with_capacity(cs.num_rows());
    x.push(Goldilocks::new(x0));
    let mut row = 1;
    while row < cs.num_rows() {
        x.push(x[row - 1].square());
        row += 1;
    }
    match cs.trace([x]) {
        Ok(trace) => trace,
        Err(e) => panic!("the squaring trace has one value per row: {e}"),
    }
}

/// The multiplicative Fibonacci circuit on `num_rows` rows.
///
/// # Panics
///
/// If `num_rows` is zero.
///
/// ```
/// use arrabbiata_relation::examples::fibonacci;
/// assert_eq!(fibonacci(5).constraints().len(), 2);
/// ```
#[must_use]
pub fn fibonacci(num_rows: usize) -> Fibonacci {
    type E = Expr<Goldilocks, 2, 1>;
    let q = E::fixed(FixedColumn::new::<0>(), Rotation::Curr);
    let x = E::witness(WitnessColumn::new::<0>(), Rotation::Curr);
    let y = E::witness(WitnessColumn::new::<1>(), Rotation::Curr);
    let x_next = E::witness(WitnessColumn::new::<0>(), Rotation::Next);
    let y_next = E::witness(WitnessColumn::new::<1>(), Rotation::Next);
    let shift = q.clone() * (x_next - y.clone());
    let product = q * (y_next - x * y);
    match Fibonacci::new(rows(num_rows), [selector(num_rows)], [shift, product]) {
        Ok(cs) => cs,
        Err(e) => panic!("the Fibonacci circuit is well-formed: {e}"),
    }
}

/// The trace `(x0, y0), (y0, x0 y0), ...` of the multiplicative Fibonacci
/// circuit.
///
/// ```
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_relation::examples::{fibonacci, fibonacci_trace};
/// let cs = fibonacci(3);
/// let t = fibonacci_trace(&cs, 2, 3);
/// assert_eq!(t.columns()[1], vec![F::new(3), F::new(6), F::new(18)]);
/// ```
#[must_use]
pub fn fibonacci_trace(cs: &Fibonacci, x0: u64, y0: u64) -> Trace<Goldilocks, 2> {
    let mut x = Vec::with_capacity(cs.num_rows());
    let mut y = Vec::with_capacity(cs.num_rows());
    x.push(Goldilocks::new(x0));
    y.push(Goldilocks::new(y0));
    let mut row = 1;
    while row < cs.num_rows() {
        let (px, py) = (x[row - 1], y[row - 1]);
        x.push(py);
        y.push(px * py);
        row += 1;
    }
    match cs.trace([x, y]) {
        Ok(trace) => trace,
        Err(e) => panic!("the Fibonacci trace has one value per row: {e}"),
    }
}
