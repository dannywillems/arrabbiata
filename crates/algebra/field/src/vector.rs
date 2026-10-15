//! Operations on vectors of field elements.

use alloc::vec::Vec;

use crate::Field;

/// `(1, z, z^2, ..., z^(n-1))`.
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
