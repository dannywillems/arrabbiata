//! Kimchi's EndoMul gate: four bits of an endomorphism-optimized scalar
//! multiplication per row.
//!
//! On a curve with the endomorphism `phi(x, y) = (endo x, y)` (the GLV
//! endomorphism of `arrabbiata_curve::glv`), two bits `(b1, b2)` choose
//! `Q` among `T`, `-T`, `phi(T)`, `-phi(T)`: `xq = (1 + (endo - 1) b1) xT`,
//! `yq = (2 b2 - 1) yT`. A row applies `P -> (P + Q1) + P` then
//! `R -> (R + Q2) + R` for the pairs `(b1, b2)` and `(b3, b4)`, and
//! accumulates `n' = 16 n + 8 b1 + 4 b2 + 2 b3 + b4`. Over 128 bits from
//! `P = 2 (phi(T) + T)`, the result is `[k] T` for the scalar Kimchi's
//! `ScalarChallenge::to_field` maps the bits to.
//!
//! The row, as Kimchi lays it out (`S`, `n'` on the next row, where the next
//! gate reads them as its `P`, `n`):
//!
//! | 0    | 1    | 2     | 3 | 4    | 5    | 6   | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   |
//! | ---- | ---- | ----- | - | ---- | ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
//! | `xT` | `yT` | `inv` |   | `xP` | `yP` | `n` | `xR` | `yR` | `s1` | `s3` | `b1` | `b2` | `b3` | `b4` |
//!
//! The 12 constraints, in Kimchi's order: the four bits boolean; for each
//! half, the slope `s` of `P + Q`, then the two equations of `(P + Q) + P`
//! with its slope eliminated; `n'`; and `(xP - xR)(xR - xS) inv = 1`. The
//! witness is computed as Kimchi's `endosclmul::gen_witness`.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget};

use self::column::{BITS, INV, N, S1, S3, XP, XR, XT, YP, YR, YT};

/// The bits of one row.
pub const BITS_PER_ROW: usize = 4;

/// The columns of the row.
pub mod column {
    /// `xT`.
    pub const XT: usize = 0;
    /// `yT`.
    pub const YT: usize = 1;
    /// `inv`, the inverse of `(xP - xR)(xR - xS)`.
    pub const INV: usize = 2;
    /// `xP` (and `xS` on the next row).
    pub const XP: usize = 4;
    /// `yP` (and `yS` on the next row).
    pub const YP: usize = 5;
    /// `n` (and `n'` on the next row).
    pub const N: usize = 6;
    /// `xR`.
    pub const XR: usize = 7;
    /// `yR`.
    pub const YR: usize = 8;
    /// `s1`.
    pub const S1: usize = 9;
    /// `s3`.
    pub const S3: usize = 10;
    /// `b1 .. b4`.
    pub const BITS: [usize; super::BITS_PER_ROW] = [11, 12, 13, 14];
}

/// The witness of one row from `[xT, yT, xP, yP, b1, b2, b3, b4]`: `xR`,
/// `yR`, `s1`, `s3`, `xS`, `yS` and `inv`, as Kimchi's `gen_witness`
/// computes them (zero where it would divide by zero).
fn row<F: Field>(endo: F, v: &[F]) -> [F; 7] {
    let inverse = |x: F| x.inverse().unwrap_or(F::ZERO);
    let one = F::ONE;
    let (xt, yt, xp, yp) = (v[0], v[1], v[2], v[3]);
    let [b1, b2, b3, b4] = [v[4], v[5], v[6], v[7]];
    let xq1 = (one + (endo - one) * b1) * xt;
    let yq1 = (b2 + b2 - one) * yt;
    let s1 = (yq1 - yp) * inverse(xq1 - xp);
    let s1_squared = s1 * s1;
    let s2 = (yp + yp) * inverse(xp + xp + xq1 - s1_squared) - s1;
    let xr = xq1 + s2 * s2 - s1_squared;
    let yr = (xp - xr) * s2 - yp;
    let xq2 = (one + (endo - one) * b3) * xt;
    let yq2 = (b4 + b4 - one) * yt;
    let s3 = (yq2 - yr) * inverse(xq2 - xr);
    let s3_squared = s3 * s3;
    let s4 = (yr + yr) * inverse(xr + xr + xq2 - s3_squared) - s3;
    let xs = xq2 + s4 * s4 - s3_squared;
    let ys = (xr - xs) * s4 - yr;
    let inv = inverse((xp - xr) * (xr - xs));
    [xr, yr, s1, s3, xs, ys, inv]
}

/// One row. Input `(T, P, n, [b1, b2, b3, b4])`, output `(S, n')`.
///
/// `T`, `P` and `n` are wired by copies; the bits are written from their
/// values and constrained by the gate (boolean, and `n'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoMulGate<F> {
    /// `endo`, the cube root of unity of the endomorphism: the curve's GLV
    /// `beta`.
    pub endo: F,
}

impl<F> TypedGadget<F> for EndoMulGate<F>
where
    F: Field,
{
    type Input = (
        Array<Scalar, 2>,
        Array<Scalar, 2>,
        Scalar,
        Array<Scalar, BITS_PER_ROW>,
    );
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-endomul";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (t, p, n, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        gate(env, self.endo, &t, &p, &n, |env, k, at| {
            env.advice(at, [&bits[k]], |[v]| v)
        })
    }
}

/// One row on `t`, `p` and `n`, its bit `k` placed at its cell by `bit`:
/// the constraints and the witness, whichever gives the bits.
fn gate<F, E>(
    env: &mut E,
    endo: F,
    t: &[E::Var; 2],
    p: &[E::Var; 2],
    n: &E::Var,
    mut bit: impl FnMut(&mut E, usize, Cell) -> E::Var,
) -> ([E::Var; 2], E::Var)
where
    F: Field,
    E: Env<F>,
{
    let xt = env.copy(Cell::new(XT, 0), &t[0]);
    let yt = env.copy(Cell::new(YT, 0), &t[1]);
    let xp = env.copy(Cell::new(XP, 0), &p[0]);
    let yp = env.copy(Cell::new(YP, 0), &p[1]);
    let n = env.copy(Cell::new(N, 0), n);
    let b: [E::Var; BITS_PER_ROW] = core::array::from_fn(|k| bit(env, k, Cell::new(BITS[k], 0)));
    let known = [&xt, &yt, &xp, &yp, &b[0], &b[1], &b[2], &b[3]];
    let at = |env: &mut E, column: usize, row_offset: usize, index: usize| {
        env.advice(Cell::new(column, row_offset), known, |v| {
            row(endo, &v)[index]
        })
    };
    let xr = at(env, XR, 0, 0);
    let yr = at(env, YR, 0, 1);
    let s1 = at(env, S1, 0, 2);
    let s3 = at(env, S3, 0, 3);
    let xs = at(env, XP, 1, 4);
    let ys = at(env, YP, 1, 5);
    let inv = at(env, INV, 0, 6);
    let n_next = env.advice(Cell::new(N, 1), [&n, &b[0], &b[1], &b[2], &b[3]], |v| {
        v[1..].iter().fold(v[0], |acc, bit| acc + acc + *bit)
    });

    let one = env.constant(F::ONE);
    let endo_minus_one = env.constant(endo - F::ONE);
    let xq1 = (one.clone() + b[0].clone() * endo_minus_one.clone()) * xt.clone();
    let xq2 = (one.clone() + b[2].clone() * endo_minus_one) * xt;
    let yq1 = (b[1].clone() + b[1].clone() - one.clone()) * yt.clone();
    let yq2 = (b[3].clone() + b[3].clone() - one.clone()) * yt;
    let s1_squared = s1.clone() * s1.clone();
    let s3_squared = s3.clone() * s3.clone();
    let xp_xr = xp.clone() - xr.clone();
    let xr_xs = xr.clone() - xs.clone();
    let ys_yr = ys.clone() + yr.clone();
    let yr_yp = yr.clone() + yp.clone();
    for bit in &b {
        env.assert_zero(bit.clone() * bit.clone() - bit.clone());
    }
    env.assert_zero((xq1.clone() - xp.clone()) * s1.clone() - (yq1 - yp.clone()));
    env.assert_zero(
        (xp.clone() + xp.clone() - s1_squared.clone() + xq1.clone())
            * (xp_xr.clone() * s1 + yr_yp.clone())
            - (yp.clone() + yp) * xp_xr.clone(),
    );
    env.assert_zero(
        yr_yp.clone() * yr_yp - xp_xr.clone() * xp_xr.clone() * (s1_squared - xq1 + xr.clone()),
    );
    env.assert_zero((xq2.clone() - xr.clone()) * s3.clone() - (yq2 - yr.clone()));
    env.assert_zero(
        (xr.clone() + xr.clone() - s3_squared.clone() + xq2.clone())
            * (xr_xs.clone() * s3 + ys_yr.clone())
            - (yr.clone() + yr) * xr_xs.clone(),
    );
    env.assert_zero(
        ys_yr.clone() * ys_yr - xr_xs.clone() * xr_xs.clone() * (s3_squared - xq2 + xs.clone()),
    );
    let combined = b.iter().fold(n, |acc, bit| acc.clone() + acc + bit.clone());
    env.assert_zero(combined - n_next.clone());
    env.assert_zero(xp_xr * xr_xs * inv - one);
    ([xs, ys], n_next)
}

/// One row of [`EndoMulOf`]: its four bits those of the scalar it reads,
/// taken from the scalar's value, from bit `top` down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoMulGateOf<F> {
    /// `endo`: the curve's GLV `beta`.
    pub endo: F,
    /// The index of the row's most significant bit.
    pub top: usize,
}

impl<F> TypedGadget<F> for EndoMulGateOf<F>
where
    F: PrimeField,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar, Scalar);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-endomul";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (t, p, n, scalar): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let top = self.top;
        gate(env, self.endo, &t, &p, &n, |env, k, at| {
            env.advice(at, [&scalar], |[v]| {
                let i = top - k;
                F::from_u64((v.limb(i / 64) >> (i % 64)) & 1)
            })
        })
    }
}

/// An endomorphism-optimized scalar multiplication by `BITS` bits, most
/// significant first, on `BITS / 4` rows, the result on the row after:
/// input `(T, P, [b])`, output `(P', n)` with `n` the bits as an integer.
/// From `P = 2 (phi(T) + T)` and 128 bits, `P'` is `[k] T` for Kimchi's
/// `ScalarChallenge::to_field` of the bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoMul<F, const BITS: usize> {
    /// `endo`: the curve's GLV `beta`.
    pub endo: F,
}

impl<F, const B: usize> TypedGadget<F> for EndoMul<F, B>
where
    F: Field,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Array<Scalar, B>);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-endomul-scalar";
    const ROWS: usize = {
        assert!(
            B.is_multiple_of(BITS_PER_ROW),
            "the bits come in rows of four"
        );
        B / BITS_PER_ROW
    };

    fn synthesize<E>(
        &self,
        env: &mut E,
        (t, p, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let zero = env.constant(F::ZERO);
        let mut n = env.define(Cell::new(N, 0), zero);
        let mut acc = p;
        let gate = EndoMulGate { endo: self.endo };
        // Each row copies T from the cells it is given; after the first row,
        // those are the previous row's, as in Kimchi's wiring.
        for chunk in bits.chunks(BITS_PER_ROW) {
            let four: [E::Var; BITS_PER_ROW] = core::array::from_fn(|k| chunk[k].clone());
            let (next, n_next) = env.call(&gate, (t.clone(), acc, n, four));
            acc = next;
            n = n_next;
        }
        (acc, n)
    }
}

/// [`EndoMul`] of the scalar it reads, its bits taken from the scalar's
/// value, most significant first: input `(T, P, x)`, output `(P', n)`. The
/// rows pin `n` to the bits, so a caller asserting `n` equal to `x` checks
/// that `x` has `BITS` bits and that the bits are its own; from
/// `P = 2 (phi(T) + T)`, `P'` is `[k] T` for `ScalarChallenge::to_field` of
/// `x` (Pickles' `endo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoMulOf<F, const BITS: usize> {
    /// `endo`: the curve's GLV `beta`.
    pub endo: F,
}

impl<F, const B: usize> TypedGadget<F> for EndoMulOf<F, B>
where
    F: PrimeField,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-endomul-of";
    const ROWS: usize = <EndoMul<F, B> as TypedGadget<F>>::ROWS;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (t, p, scalar): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let zero = env.constant(F::ZERO);
        let mut n = env.define(Cell::new(N, 0), zero);
        let mut acc = p;
        for r in 0..B / BITS_PER_ROW {
            let gate = EndoMulGateOf {
                endo: self.endo,
                top: B - 1 - r * BITS_PER_ROW,
            };
            let (next, n_next) = env.call(&gate, (t.clone(), acc, n, scalar.clone()));
            acc = next;
            n = n_next;
        }
        (acc, n)
    }
}
