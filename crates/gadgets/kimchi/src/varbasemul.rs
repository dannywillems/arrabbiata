//! Kimchi's VarBaseMul gate: five steps of a variable-base scalar
//! multiplication on two rows.
//!
//! Each step reads a bit `b` and maps an accumulator `A` to
//! `2 A + (2 b - 1) T`, for the base `T`, as `(A + Q) + A` with
//! `Q = (2 b - 1) T`, without the `y` coordinate of `A + Q` (Eisentraeger,
//! Lauter, Montgomery, section 3.1 of arXiv math/0208038, as Kimchi cites
//! it). Starting from `A = 2 T` and reading the bits of `x` from the most
//! significant, `m` steps give `(1 + 2^m + 2 x) T` (Kimchi's
//! `tests/varbasemul.rs`).
//!
//! The two rows, as Kimchi lays them out (the second is a Zero gate):
//!
//! | row | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   |
//! | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
//! | `i` | `xT` | `yT` | `x0` | `y0` | `n`  | `n'` |      | `x1` | `y1` | `x2` | `y2` | `x3` | `y3` | `x4` | `y4` |
//! | `i+1` | `x5` | `y5` | `b0` | `b1` | `b2` | `b3` | `b4` | `s0` | `s1` | `s2` | `s3` | `s4` |   |      |      |
//!
//! The 21 constraints: `n' = 32 n + sum 2^(4 - k) b_k`, and for each step
//! `k`, with input `(xI, yI)`, output `(xO, yO)` and slope `s = s_k`:
//!
//! - `b (b - 1) = 0`;
//! - `(xI - xT) s - (yI - (2 b - 1) yT) = 0`;
//! - `u^2 - t^2 (xO - xT + s^2) = 0` and `(yO + yI) t - (xI - xO) u = 0`,
//!   where `t = 2 xI + xT - s^2` and `u = 2 yI - t s`.
//!
//! The witness is computed as Kimchi's `varbasemul::witness`.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget};

/// The steps, that is bits, of one gate.
pub const BITS_PER_GATE: usize = 5;

/// The rows of one gate.
pub const ROWS_PER_GATE: usize = 2;

/// The cells of the accumulators `(x0, y0) .. (x5, y5)`: (row, column of
/// `x`), `y` being in the next column.
pub const ACCUMULATORS: [(usize, usize); BITS_PER_GATE + 1] =
    [(0, 2), (0, 7), (0, 9), (0, 11), (0, 13), (1, 0)];

/// The columns, on the second row, of the bits `b0 .. b4`.
pub const BITS: [usize; BITS_PER_GATE] = [2, 3, 4, 5, 6];

/// The columns, on the second row, of the slopes `s0 .. s4`.
pub const SLOPES: [usize; BITS_PER_GATE] = [7, 8, 9, 10, 11];

/// The column of `n` (first row).
pub const N: usize = 4;

/// The column of `n'` (first row).
pub const N_NEXT: usize = 5;

/// One step of the witness: the slope and the next accumulator, as
/// Kimchi's `single_bit_witness` computes them (zero where a division by
/// zero would be).
fn step<F: Field>(b: F, [xt, yt]: [F; 2], [xi, yi]: [F; 2]) -> (F, [F; 2]) {
    let inverse = |x: F| x.inverse().unwrap_or(F::ZERO);
    let sign = b + b - F::ONE;
    let s1 = (yi - sign * yt) * inverse(xi - xt);
    let s1_squared = s1 * s1;
    let s2 = (yi + yi) * inverse(xi + xi + xt - s1_squared) - s1;
    let xo = xt + s2 * s2 - s1_squared;
    let yo = (xi - xo) * s2 - yi;
    (s1, [xo, yo])
}

/// The witness of one gate on `values` (`[xT, yT, x0, y0, b0, .., b4]`):
/// the five slopes and the six accumulators.
fn trace<F: Field>(values: &[F]) -> ([F; BITS_PER_GATE], [[F; 2]; BITS_PER_GATE + 1]) {
    let base = [values[0], values[1]];
    let mut slopes = [F::ZERO; BITS_PER_GATE];
    let mut accumulators = [[values[2], values[3]]; BITS_PER_GATE + 1];
    for k in 0..BITS_PER_GATE {
        let (slope, next) = step(values[4 + k], base, accumulators[k]);
        slopes[k] = slope;
        accumulators[k + 1] = next;
    }
    (slopes, accumulators)
}

/// One gate: five steps. Input `(T, A, n, [b0, .., b4])`, output
/// `(A', n')` with `A' = 32 A + (2 sum 2^(4 - k) b_k - 31) T` and
/// `n' = 32 n + sum 2^(4 - k) b_k`.
///
/// `T`, `A` and `n` are wired by copies to their cells. The bits are
/// written from their values and constrained by the gate (each boolean,
/// and `n'` their combination with `n`), so they need not be cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarBaseMulGate;

impl<F> TypedGadget<F> for VarBaseMulGate
where
    F: Field,
{
    type Input = (
        Array<Scalar, 2>,
        Array<Scalar, 2>,
        Scalar,
        Array<Scalar, BITS_PER_GATE>,
    );
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-varbasemul";
    const ROWS: usize = ROWS_PER_GATE;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (base, acc, n, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        gate(env, &base, &acc, &n, |env, k, at| {
            env.advice(at, [&bits[k]], |[v]| v)
        })
    }
}

/// One gate on `base`, `acc` and `n`, its bit `k` placed at its cell by
/// `bit`: the constraints and the witness, whichever gives the bits.
fn gate<F, E>(
    env: &mut E,
    base: &[E::Var; 2],
    acc: &[E::Var; 2],
    n: &E::Var,
    mut bit: impl FnMut(&mut E, usize, Cell) -> E::Var,
) -> ([E::Var; 2], E::Var)
where
    F: Field,
    E: Env<F>,
{
    let xt = env.copy(Cell::new(0, 0), &base[0]);
    let yt = env.copy(Cell::new(1, 0), &base[1]);
    let x0 = env.copy(Cell::new(ACCUMULATORS[0].1, 0), &acc[0]);
    let y0 = env.copy(Cell::new(ACCUMULATORS[0].1 + 1, 0), &acc[1]);
    let n = env.copy(Cell::new(N, 0), n);
    let b: [E::Var; BITS_PER_GATE] = core::array::from_fn(|k| bit(env, k, Cell::new(BITS[k], 1)));
    // The values the witness needs: T, A0, and the bits.
    let known: [&E::Var; 4 + BITS_PER_GATE] =
        [&xt, &yt, &x0, &y0, &b[0], &b[1], &b[2], &b[3], &b[4]];
    let s: [E::Var; BITS_PER_GATE] =
        core::array::from_fn(|k| env.advice(Cell::new(SLOPES[k], 1), known, |v| trace(&v).0[k]));
    let accumulators: [(E::Var, E::Var); BITS_PER_GATE + 1] = core::array::from_fn(|k| {
        if k == 0 {
            return (x0.clone(), y0.clone());
        }
        let (row, column) = ACCUMULATORS[k];
        let x = env.advice(Cell::new(column, row), known, |v| trace(&v).1[k][0]);
        let y = env.advice(Cell::new(column + 1, row), known, |v| trace(&v).1[k][1]);
        (x, y)
    });
    let n_next = env.advice(
        Cell::new(N_NEXT, 0),
        [&n, &b[0], &b[1], &b[2], &b[3], &b[4]],
        |v| v[1..].iter().fold(v[0], |acc, bit| acc + acc + *bit),
    );

    // n' = 32 n + bits, each step a scaling by 2 (not `acc + acc`, which
    // copies the term at every bit).
    let two = env.constant(F::from_u64(2));
    let combined = b.iter().fold(n, |acc, bit| acc * two.clone() + bit.clone());
    env.assert_zero(n_next.clone() - combined);
    let one = env.constant(F::ONE);
    for k in 0..BITS_PER_GATE {
        let (xi, yi) = accumulators[k].clone();
        let (xo, yo) = accumulators[k + 1].clone();
        let bit = b[k].clone();
        let slope = s[k].clone();
        let sign = bit.clone() + bit.clone() - one.clone();
        let slope_squared = slope.clone() * slope.clone();
        let t = xi.clone() + xi.clone() + xt.clone() - slope_squared.clone();
        let u = yi.clone() + yi.clone() - t.clone() * slope.clone();
        env.assert_zero(bit.clone() * bit.clone() - bit);
        env.assert_zero((xi.clone() - xt.clone()) * slope - (yi.clone() - sign * yt.clone()));
        env.assert_zero(
            u.clone() * u.clone()
                - t.clone() * t.clone() * (xo.clone() - xt.clone() + slope_squared),
        );
        env.assert_zero((yo + yi) * t - (xi - xo) * u);
    }
    let (x5, y5) = accumulators[BITS_PER_GATE].clone();
    ([x5, y5], n_next)
}

/// The bit of `scalar` at `index`, the least significant `0`.
fn bit_of<F: PrimeField>(scalar: F, index: usize) -> F {
    F::from_u64((scalar.limb(index / 64) >> (index % 64)) & 1)
}

/// One gate of [`VarBaseMulOf`]: its five bits those of the scalar it
/// reads, taken from the scalar's value, from bit `top` down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarBaseMulGateOf {
    /// The index of the gate's most significant bit.
    pub top: usize,
}

impl<F> TypedGadget<F> for VarBaseMulGateOf
where
    F: PrimeField,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar, Scalar);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-varbasemul";
    const ROWS: usize = ROWS_PER_GATE;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (base, acc, n, scalar): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let top = self.top;
        gate(env, &base, &acc, &n, |env, k, at| {
            env.advice(at, [&scalar], |[v]| bit_of(v, top - k))
        })
    }
}

/// A variable-base scalar multiplication by `BITS` bits, most significant
/// first, on `2 BITS / 5` rows: input `(T, A, [b])`, output `(A', n)` with
/// `n` the bits as an integer and `A'` after `BITS` steps. From `A = 2 T`,
/// `A' = (1 + 2^BITS + 2 n) T`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarBaseMul<const BITS: usize>;

impl<F, const B: usize> TypedGadget<F> for VarBaseMul<B>
where
    F: Field,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Array<Scalar, B>);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-varbasemul-scalar";
    const ROWS: usize = {
        assert!(
            B.is_multiple_of(BITS_PER_GATE),
            "the bits come in gates of five"
        );
        ROWS_PER_GATE * (B / BITS_PER_GATE)
    };

    fn synthesize<E>(
        &self,
        env: &mut E,
        (base, acc, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let zero = env.constant(F::ZERO);
        let mut n = env.define(Cell::new(N, 0), zero);
        let mut acc = acc;
        // Each gate copies T from the cells it is given; after the first
        // gate, those are the previous gate's, as in Kimchi's wiring.
        for chunk in bits.chunks(BITS_PER_GATE) {
            let five: [E::Var; BITS_PER_GATE] = core::array::from_fn(|k| chunk[k].clone());
            let (next, n_next) = env.call(&VarBaseMulGate, (base.clone(), acc, n, five));
            acc = next;
            n = n_next;
        }
        (acc, n)
    }
}

/// [`VarBaseMul`] of the scalar it reads, its bits taken from the scalar's
/// value, most significant first: input `(T, A, x)`, output `(A', n)`. The
/// gates pin `n` to the bits, so a caller asserting `n` equal to `x` checks
/// that `x` has `BITS` bits and that the bits are its own; from `A = 2 T`,
/// `A' = (1 + 2^BITS + 2 x) T`, Pickles' `scale_fast` of the shifted value
/// `x` (`Shifted_value.Type1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarBaseMulOf<const BITS: usize>;

impl<F, const B: usize> TypedGadget<F> for VarBaseMulOf<B>
where
    F: PrimeField,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-varbasemul-of";
    const ROWS: usize = <VarBaseMul<B> as TypedGadget<F>>::ROWS;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (base, acc, scalar): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let zero = env.constant(F::ZERO);
        let mut n = env.define(Cell::new(N, 0), zero);
        let mut acc = acc;
        for g in 0..B / BITS_PER_GATE {
            let gate = VarBaseMulGateOf {
                top: B - 1 - g * BITS_PER_GATE,
            };
            let (next, n_next) = env.call(&gate, (base.clone(), acc, n, scalar.clone()));
            acc = next;
            n = n_next;
        }
        (acc, n)
    }
}
