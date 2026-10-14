//! Operations on vectors of field elements.
//!
//! ```
//! use arrabbiata_field::{Goldilocks as F, inner_product, powers};
//! // <(1, z, z^2), (a0, a1, a2)> is the polynomial a0 + a1 X + a2 X^2 at z.
//! let a = [F::new(1), F::new(2), F::new(3)];
//! assert_eq!(inner_product(&a, &powers(F::new(2), 3)), F::new(17));
//! ```

use alloc::vec::Vec;

use crate::Field;

/// `(1, z, z^2, ..., z^(n-1))`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F, powers};
/// assert_eq!(powers(F::new(3), 4), vec![F::ONE, F::new(3), F::new(9), F::new(27)]);
/// assert!(powers(F::new(3), 0).is_empty());
/// ```
#[must_use]
pub fn powers<F>(z: F, n: usize) -> Vec<F>
where
    F: Field,
{
    let mut out = Vec::with_capacity(n);
    let mut acc = F::ONE;
    let mut i = 0;
    while i < n {
        out.push(acc);
        acc = acc * z;
        i += 1;
    }
    out
}

/// `sum_i a_i b_i`, over the common prefix of `a` and `b`.
///
/// ```
/// use arrabbiata_field::{Goldilocks as F, inner_product};
/// let a = [F::new(1), F::new(2)];
/// let b = [F::new(3), F::new(4), F::new(5)];
/// assert_eq!(inner_product(&a, &b), F::new(11));
/// ```
#[must_use]
pub fn inner_product<F>(a: &[F], b: &[F]) -> F
where
    F: Field,
{
    let mut acc = F::ZERO;
    let mut i = 0;
    while i < a.len() && i < b.len() {
        acc = acc + a[i] * b[i];
        i += 1;
    }
    acc
}
