//! Lookup arguments: plookup and LogUp.
//!
//! A [`Lookup`] asks that on every row its selector turns on, the tuple of
//! its input polynomials (in the gates' variables) is a row of a table made
//! of fixed columns. With a challenge `theta` drawn after the witness, a
//! tuple `v` is compressed to `v_0 + theta v_1 + theta^2 v_2 + ...`; the
//! query on a row is `f = q c(inputs) + (1 - q) c(t[0])`, the selector `q`
//! being boolean, so a row with the selector off queries the table's first
//! row, which is in the table. The table is `t = c(table columns)`.
//!
//! # Plookup
//!
//! Gabizon and Williamson, *plookup: A simplified polynomial protocol for
//! lookup tables*, ePrint 2020/315. The paper works on a domain of size
//! `n + 1` for `n` queries; here (our adaptation) the domain has `n` rows
//! and the last one is reserved: the queries are the rows `0, ..., n - 2`,
//! the table all `n` rows. The prover sorts `s = (f, t)` by `t` (each query
//! placed after the first occurrence of its value in `t`), of length
//! `2n - 1`, and commits to its halves `h_1 = s[0..n]` and
//! `h_2 = s[n-1..2n-1]`, which overlap in one element. The paper shows
//! that, for random `beta, gamma`, `f` is in `t` and `s` is so sorted iff
//!
//! ```text
//! (1 + beta)^(n-1) prod_i (gamma + f_i) prod_i (gamma (1 + beta) + t_i + beta t_(i+1))
//!   = prod_i (gamma (1 + beta) + s_i + beta s_(i+1))
//! ```
//!
//! over the `n - 1` consecutive pairs, which a grand product `Z` checks:
//! `L_0 (Z - 1)`, `L_(n-1) (Z - 1)`, `L_(n-1) (h_1 - h_2(w X))`, and on every
//! row but the last, `Z (1 + beta)(gamma + f)(gamma (1 + beta) + t + beta t(w X))
//! = Z(w X)(gamma (1 + beta) + h_1 + beta h_1(w X))(gamma (1 + beta) + h_2 +
//! beta h_2(w X))`.
//!
//! # LogUp
//!
//! Haboeck, *Multivariate lookups based on logarithmic derivatives*,
//! ePrint 2022/1530, in its univariate form: `f` is in `t` iff there are
//! multiplicities `m` with
//!
//! ```text
//! sum_i 1 / (gamma - f_i) = sum_j m_j / (gamma - t_j)
//! ```
//!
//! as rational functions of `gamma`, which the paper shows provided the
//! characteristic exceeds the number of queries. The prover commits to
//! `m` (the count of each table row among the queries, on its first
//! occurrence), then to the running sum `phi`, `phi_0 = 0`,
//! `phi_(i+1) = phi_i + 1 / (gamma - f_i) - m_i / (gamma - t_i)`. The sum
//! is zero exactly when the running sum closes on the cycle:
//! `(phi(w X) - phi)(gamma - f)(gamma - t) = (gamma - t) - m (gamma - f)` on
//! every row, and `L_0 phi = 0`. All `n` rows are queries.
//!
//! Both arguments run on range checks, a XOR table, polynomial queries and
//! repeated rows in `tests/lookups.rs`.

use alloc::{collections::BTreeMap, vec, vec::Vec};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_mvpoly::{MVPoly, Sparse};

use crate::Error;

/// Which lookup argument proves a [`Lookup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Argument {
    /// Plookup: a sorted vector and a grand product. Queries on every row
    /// but the last.
    Plookup,
    /// LogUp: multiplicities and a running sum of inverses. Queries on
    /// every row.
    LogUp,
}

impl Argument {
    /// The number of polynomials the prover commits to before `beta,
    /// gamma` (`h_1, h_2`, or `m`) and after (`Z`, or `phi`).
    pub(crate) const fn columns(self) -> (usize, usize) {
        match self {
            Self::Plookup => (2, 1),
            Self::LogUp => (1, 1),
        }
    }
}

/// A lookup: on every row its selector turns on (every row if `None`),
/// the tuple of `inputs` is a row of the fixed columns `table`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup<F, const N: usize, const D: usize>
where
    F: Field,
{
    argument: Argument,
    selector: Option<usize>,
    inputs: Vec<Sparse<F, N, D>>,
    table: Vec<usize>,
    default: Vec<F>,
}

impl<F, const N: usize, const D: usize> Lookup<F, N, D>
where
    F: Field,
{
    /// The lookup of `inputs` into the fixed columns `table`, one input per
    /// column, turned on by `selector`, proved with `argument`.
    #[must_use]
    pub const fn new(
        argument: Argument,
        selector: Option<usize>,
        inputs: Vec<Sparse<F, N, D>>,
        table: Vec<usize>,
    ) -> Self {
        Self {
            argument,
            selector,
            inputs,
            table,
            default: Vec::new(),
        }
    }

    /// The argument proving the lookup.
    #[must_use]
    pub const fn argument(&self) -> Argument {
        self.argument
    }

    /// The selector turning the lookup on, if any.
    #[must_use]
    pub const fn selector(&self) -> Option<usize> {
        self.selector
    }

    /// The input polynomials, one per table column.
    #[must_use]
    pub fn inputs(&self) -> &[Sparse<F, N, D>] {
        &self.inputs
    }

    /// The fixed columns of the table.
    #[must_use]
    pub fn table(&self) -> &[usize] {
        &self.table
    }

    /// The table's first row, which a row with the selector off queries;
    /// empty until the lookup is attached to a circuit.
    #[must_use]
    pub fn default(&self) -> &[F] {
        &self.default
    }

    pub(crate) fn set_default(&mut self, default: Vec<F>) {
        self.default = default;
    }
}

/// `v_0 + theta v_1 + theta^2 v_2 + ...`.
pub(crate) fn compress<F>(values: impl IntoIterator<Item = F>, theta: F) -> F
where
    F: Field,
{
    let mut acc = F::ZERO;
    let mut power = F::ONE;
    for v in values {
        acc = acc + power * v;
        power = power * theta;
    }
    acc
}

/// The query of a lookup at a point: `q c(inputs) + (1 - q) c(default)`,
/// or `c(inputs)` without a selector.
pub(crate) fn query<F, const N: usize, const D: usize>(
    lookup: &Lookup<F, N, D>,
    vars: &[F; N],
    selector: F,
    theta: F,
) -> F
where
    F: Field,
{
    let input = compress(lookup.inputs.iter().map(|p| p.eval(vars)), theta);
    match lookup.selector {
        None => input,
        Some(_) => {
            selector * input + (F::ONE - selector) * compress(lookup.default.iter().copied(), theta)
        }
    }
}

/// A key ordering field elements by their canonical representative.
fn key<F>(v: F) -> Vec<u64>
where
    F: PrimeField,
{
    (0..F::LIMBS).rev().map(|i| v.limb(i)).collect()
}

/// For each query, the index of the first occurrence of its value in the
/// table, or [`Error::Unsatisfied`] if it is not there.
fn positions<F>(queries: &[F], table: &[F]) -> Result<Vec<usize>, Error>
where
    F: PrimeField,
{
    let mut first: BTreeMap<Vec<u64>, usize> = BTreeMap::new();
    for (j, t) in table.iter().enumerate() {
        first.entry(key(*t)).or_insert(j);
    }
    queries
        .iter()
        .map(|f| first.get(&key(*f)).copied().ok_or(Error::Unsatisfied))
        .collect()
}

/// Plookup's `h_1, h_2`: the halves of `s`, the queries of the rows
/// `0..n-1` merged into the table after the first occurrence of their
/// value.
pub(crate) fn sorted_halves<F>(f: &[F], t: &[F]) -> Result<(Vec<F>, Vec<F>), Error>
where
    F: PrimeField,
{
    let n = t.len();
    let mut count = vec![0usize; n];
    for j in positions(&f[..n - 1], t)? {
        count[j] += 1;
    }
    let mut s = Vec::with_capacity(2 * n - 1);
    for (j, value) in t.iter().enumerate() {
        s.push(*value);
        for _ in 0..count[j] {
            s.push(*value);
        }
    }
    Ok((s[..n].to_vec(), s[n - 1..].to_vec()))
}

/// LogUp's multiplicities: the number of queries equal to each table row,
/// counted on its first occurrence.
pub(crate) fn multiplicities<F>(f: &[F], t: &[F]) -> Result<Vec<F>, Error>
where
    F: PrimeField,
{
    let mut m = vec![F::ZERO; t.len()];
    for j in positions(f, t)? {
        m[j] = m[j] + F::ONE;
    }
    Ok(m)
}

/// Plookup's grand product `Z`, `Z_0 = 1`, over the rows `0..n-1`.
pub(crate) fn plookup_product<F>(
    f: &[F],
    t: &[F],
    h1: &[F],
    h2: &[F],
    beta: F,
    gamma: F,
) -> Result<Vec<F>, Error>
where
    F: Field,
{
    let n = t.len();
    let g = gamma * (F::ONE + beta);
    let mut z = vec![F::ONE; n];
    for i in 0..n - 1 {
        let num = (F::ONE + beta) * (gamma + f[i]) * (g + t[i] + beta * t[i + 1]);
        let den = (g + h1[i] + beta * h1[i + 1]) * (g + h2[i] + beta * h2[i + 1]);
        z[i + 1] = z[i] * num * den.inverse().ok_or(Error::Unsatisfied)?;
    }
    Ok(z)
}

/// LogUp's running sum `phi`, `phi_0 = 0`.
pub(crate) fn logup_sum<F>(f: &[F], t: &[F], m: &[F], gamma: F) -> Result<Vec<F>, Error>
where
    F: Field,
{
    let n = t.len();
    let mut phi = vec![F::ZERO; n];
    for i in 0..n - 1 {
        let a = (gamma - f[i]).inverse().ok_or(Error::Unsatisfied)?;
        let b = (gamma - t[i]).inverse().ok_or(Error::Unsatisfied)?;
        phi[i + 1] = phi[i] + a - m[i] * b;
    }
    Ok(phi)
}

/// The values one lookup's identity terms read at a point.
pub(crate) struct At<F> {
    /// The query.
    pub(crate) f: F,
    /// The table, here and one row later.
    pub(crate) t: (F, F),
    /// The committed columns, each here and one row later: `h_1, h_2, Z`
    /// for plookup, `m, phi` for LogUp.
    pub(crate) columns: Vec<(F, F)>,
    /// `L_0`, `L_(n-1)` and `X - w^(n-1)` at the point.
    pub(crate) boundary: (F, F, F),
}

/// The identity terms of a lookup at a point, each to vanish on `H`.
pub(crate) fn terms<F>(argument: Argument, at: &At<F>, beta: F, gamma: F) -> Vec<F>
where
    F: Field,
{
    let (l0, last, not_last) = at.boundary;
    let f = at.f;
    let (t, t_next) = at.t;
    match argument {
        Argument::Plookup => {
            let ((h1, h1n), (h2, h2n), (z, zn)) = (at.columns[0], at.columns[1], at.columns[2]);
            let g = gamma * (F::ONE + beta);
            let left = z * (F::ONE + beta) * (gamma + f) * (g + t + beta * t_next);
            let right = zn * (g + h1 + beta * h1n) * (g + h2 + beta * h2n);
            vec![
                l0 * (z - F::ONE),
                last * (z - F::ONE),
                last * (h1 - h2n),
                not_last * (left - right),
            ]
        }
        Argument::LogUp => {
            let ((m, _), (phi, phin)) = (at.columns[0], at.columns[1]);
            vec![
                (phin - phi) * (gamma - f) * (gamma - t) - (gamma - t) + m * (gamma - f),
                l0 * phi,
            ]
        }
    }
}
