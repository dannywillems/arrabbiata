//! Small example circuits over any field, used throughout the documentation
//! and the tests of the workspace.
//!
//! - [`squaring`]: one constraint of degree 2, aggregated with [`Single`].
//!   Its relaxed relation is exactly Nova's relaxed R1CS for the row-local
//!   constraint `next = curr^2`: the relation has degree 2 and a fold has one
//!   cross-term.
//! - [`fibonacci`]: two constraints of degree 2 (the "multiplicative
//!   Fibonacci" sequence `(x, y) -> (y, x * y)`), aggregated with
//!   [`Powers`]. The relation has degree 3 and a fold has two cross-terms.
//!
//! Both gate their constraints with a selector that is off on the last row,
//! since [`Rotation::Next`] wraps around.
//!
//! ```
//! use arrabbiata_pasta::Fq;
//! use arrabbiata_relation::examples;
//!
//! let cs = examples::fibonacci::<Fq>(6);
//! let trace = examples::fibonacci_trace(&cs, 2, 3);
//! assert!(cs.check_trace(&trace).is_ok());
//! ```

use alloc::{vec, vec::Vec};
use core::num::NonZeroUsize;

use arrabbiata_field::Field;
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};

use crate::{
    aggregation::{Powers, Single},
    system::{ConstraintSystem, Trace},
};

/// The squaring circuit over `F`: one witness column, one selector, one
/// constraint `q * (next - curr^2)` of degree 2.
///
/// ```
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_relation::examples::{Squaring, squaring};
/// let cs: Squaring<Fq> = squaring(4);
/// assert_eq!(Squaring::<Fq>::CROSS_TERMS, 1);
/// # let _ = cs;
/// ```
pub type Squaring<F> = ConstraintSystem<F, Single, 1, 1, 1, 2>;

/// The multiplicative Fibonacci circuit over `F`: two witness columns, one
/// selector, constraints `q * (x_next - y)` and `q * (y_next - x * y)`.
///
/// ```
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_relation::examples::{Fibonacci, fibonacci};
/// let cs: Fibonacci<Fq> = fibonacci(4);
/// assert_eq!(Fibonacci::<Fq>::DEGREE, 3);
/// # let _ = cs;
/// ```
pub type Fibonacci<F> = ConstraintSystem<F, Powers, 2, 1, 2, 2>;

/// The selector `1, ..., 1, 0` of `num_rows` rows.
fn selector<F>(num_rows: usize) -> Vec<F>
where
    F: Field,
{
    let mut q = vec![F::ONE; num_rows];
    q[num_rows - 1] = F::ZERO;
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
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_relation::examples::squaring;
/// assert_eq!(squaring::<Fq>(16).num_rows(), 16);
/// ```
#[must_use]
pub fn squaring<F>(num_rows: usize) -> Squaring<F>
where
    F: Field,
{
    let q = Expr::<F, 1, 1>::fixed(FixedColumn::new::<0>(), Rotation::Curr);
    let curr = Expr::<F, 1, 1>::witness(WitnessColumn::new::<0>(), Rotation::Curr);
    let next = Expr::<F, 1, 1>::witness(WitnessColumn::new::<0>(), Rotation::Next);
    let constraint = q * (next - curr.clone() * curr);
    match Squaring::new(rows(num_rows), [selector(num_rows)], [constraint]) {
        Ok(cs) => cs,
        Err(e) => panic!("the squaring circuit is well-formed: {e}"),
    }
}

/// The trace `x0, x0^2, x0^4, ...` of the squaring circuit.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_relation::examples::{squaring, squaring_trace};
/// let cs = squaring::<Fq>(3);
/// let expected = vec![Fq::from_u64(3), Fq::from_u64(9), Fq::from_u64(81)];
/// assert_eq!(squaring_trace(&cs, 3).columns()[0], expected);
/// ```
#[must_use]
pub fn squaring_trace<F>(cs: &Squaring<F>, x0: u64) -> Trace<F, 1>
where
    F: Field,
{
    let mut x = Vec::with_capacity(cs.num_rows());
    x.push(F::from_u64(x0));
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
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_relation::examples::fibonacci;
/// assert_eq!(fibonacci::<Fq>(5).constraints().len(), 2);
/// ```
#[must_use]
pub fn fibonacci<F>(num_rows: usize) -> Fibonacci<F>
where
    F: Field,
{
    type E<F> = Expr<F, 2, 1>;
    let q = E::<F>::fixed(FixedColumn::new::<0>(), Rotation::Curr);
    let x = E::<F>::witness(WitnessColumn::new::<0>(), Rotation::Curr);
    let y = E::<F>::witness(WitnessColumn::new::<1>(), Rotation::Curr);
    let x_next = E::<F>::witness(WitnessColumn::new::<0>(), Rotation::Next);
    let y_next = E::<F>::witness(WitnessColumn::new::<1>(), Rotation::Next);
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
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_relation::examples::{fibonacci, fibonacci_trace};
/// let cs = fibonacci::<Fq>(3);
/// let t = fibonacci_trace(&cs, 2, 3);
/// assert_eq!(t.columns()[1], vec![Fq::from_u64(3), Fq::from_u64(6), Fq::from_u64(18)]);
/// ```
#[must_use]
pub fn fibonacci_trace<F>(cs: &Fibonacci<F>, x0: u64, y0: u64) -> Trace<F, 2>
where
    F: Field,
{
    let mut x = Vec::with_capacity(cs.num_rows());
    let mut y = Vec::with_capacity(cs.num_rows());
    x.push(F::from_u64(x0));
    y.push(F::from_u64(y0));
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
