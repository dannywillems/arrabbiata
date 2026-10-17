//! Chunking: a polynomial of degree below `m k` as `k` polynomials of degree
//! below `m`, the way a commitment key of `m` generators commits to a
//! polynomial too large for it (Kimchi's chunks).
//!
//! With `Y = X^m`, `F[X]` is a free `F[Y]`-module with basis `1, X, ..,
//! X^(m - 1)`: every `p` is uniquely `sum_i Y^i p_i` with `deg p_i < m`.
//! Truncated to `k` chunks this is a linear isomorphism
//!
//! ```text
//! split: F[X]_{<m k}  ->  (F[X]_{<m})^k,    p  ->  (p_0, .., p_{k-1}),
//! ```
//!
//! and the operations on chunks are evaluations of the `Y` factor:
//!
//! - [`Chunking::linearize`] at `y` is `ev_{Y = y}`: `(p_i) -> sum y^i p_i`,
//!   a polynomial of degree below `m`;
//! - [`Chunking::evaluate`] at `x` is `ev_{X = x}` on each chunk;
//! - [`combine`] at `y` is `ev_{Y = y}` on any module the chunks are mapped
//!   into: their values, their commitments, their blinders.
//!
//! The two evaluations commute, which is the identity a chunked protocol
//! rests on: `p(x) = combine(evaluate(p, x), x^m)`, and for every linear
//! map `L` (a Pedersen commitment), `L(linearize(p, y)) = combine(L(p_i),
//! y)`. The verifier combines the chunks' commitments, the prover opens the
//! linearized polynomial, and both mean the same element.
//!
//! The number of chunks `k` is part of the shape, fixed by the protocol (by
//! the domain, not by the degree): a polynomial of smaller degree is padded
//! with zero chunks, which still take their place, and their blinders.

use alloc::vec::Vec;
use core::ops::{Add, Mul};

use arrabbiata_field::Field;

use crate::UniPoly;

/// Polynomials of degree below `size * count`, as `count` chunks of `size`
/// coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunking {
    size: usize,
    count: usize,
}

impl Chunking {
    /// The chunking into `count` chunks of `size` coefficients; `None` if
    /// either is zero.
    #[must_use]
    pub const fn new(size: usize, count: usize) -> Option<Self> {
        if size == 0 || count == 0 {
            None
        } else {
            Some(Self { size, count })
        }
    }

    /// The coefficients per chunk, `m`.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// The number of chunks, `k`.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// The degree bound, `m k`: the polynomials it splits have fewer
    /// coefficients (trailing zeros aside).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.size * self.count
    }

    /// The number of coefficients of `p` once its trailing zeros are left
    /// out.
    fn length<F: Field>(p: &UniPoly<F>) -> usize {
        let coefficients = p.coefficients();
        let mut len = coefficients.len();
        while len > 0 && coefficients[len - 1].is_zero() {
            len -= 1;
        }
        len
    }

    /// The `count` chunks of `p`, padded with zero chunks; `None` when `p`
    /// has degree `m k` or more.
    #[must_use]
    pub fn split<F: Field>(&self, p: &UniPoly<F>) -> Option<Vec<UniPoly<F>>> {
        if Self::length(p) > self.capacity() {
            return None;
        }
        let mut chunks = Vec::with_capacity(self.count);
        let mut i = 0;
        while i < self.count {
            let mut chunk = Vec::with_capacity(self.size);
            let mut j = 0;
            while j < self.size {
                chunk.push(p.coefficient(i * self.size + j));
                j += 1;
            }
            chunks.push(UniPoly::new(chunk));
            i += 1;
        }
        Some(chunks)
    }

    /// The value of each chunk of `p` at `x`; `None` as [`Chunking::split`].
    #[must_use]
    pub fn evaluate<F: Field>(&self, p: &UniPoly<F>, x: F) -> Option<Vec<F>> {
        let chunks = self.split(p)?;
        let mut values = Vec::with_capacity(self.count);
        let mut i = 0;
        while i < chunks.len() {
            values.push(chunks[i].evaluate(x));
            i += 1;
        }
        Some(values)
    }

    /// `sum_i y^i p_i`, of degree below `m`; `None` as [`Chunking::split`].
    #[must_use]
    pub fn linearize<F: Field>(&self, p: &UniPoly<F>, y: F) -> Option<UniPoly<F>> {
        let chunks = self.split(p)?;
        let mut sum = Vec::with_capacity(self.size);
        let mut j = 0;
        while j < self.size {
            let mut column = Vec::with_capacity(self.count);
            let mut i = 0;
            while i < chunks.len() {
                column.push(chunks[i].coefficient(j));
                i += 1;
            }
            sum.push(combine_or(&column, y, F::ZERO));
            j += 1;
        }
        Some(UniPoly::new(sum))
    }
}

/// `sum_i y^i c_i` by Horner's rule, in any module over `F` (field elements,
/// curve points, blinders): the chunks' images recombined. `None` for no
/// chunk.
#[must_use]
pub fn combine<M, F>(chunks: &[M], y: F) -> Option<M>
where
    M: Copy + Add<Output = M> + Mul<F, Output = M>,
    F: Copy,
{
    let mut i = chunks.len();
    if i == 0 {
        return None;
    }
    i -= 1;
    let mut acc = chunks[i];
    while i > 0 {
        i -= 1;
        acc = acc * y + chunks[i];
    }
    Some(acc)
}

/// [`combine`], `zero` for no chunk.
fn combine_or<M, F>(chunks: &[M], y: F, zero: M) -> M
where
    M: Copy + Add<Output = M> + Mul<F, Output = M>,
    F: Copy,
{
    match combine(chunks, y) {
        Some(sum) => sum,
        None => zero,
    }
}
