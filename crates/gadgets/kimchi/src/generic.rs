//! Kimchi's Generic gate: two PlonK half-gates on one row.
//!
//! The row reads columns 0-5 and its 10 coefficients `c`, and constrains
//!
//! - `c0 w0 + c1 w1 + c2 w2 + c3 w0 w1 + c4 = 0`,
//! - `c5 w3 + c6 w4 + c7 w5 + c8 w3 w4 + c9 = 0`,
//!
//! the coefficients being fixed columns 0-9 ([`GenericRow`]). A half-gate is
//! usually given by what it computes ([`GenericSpec`], Kimchi's
//! `GenericGateSpec`), which fixes its five coefficients as Kimchi's
//! `create_generic_gadget` does. [`Generic`] computes one operation on a row
//! and [`GenericPair`] two; both write their output in the half's third
//! cell (column 2, or 5), which the constraint then pins.
//!
//! Inputs are wired by copies, as in Kimchi: they must be cells of the
//! permuted columns, or be already where the gate reads them.
//!
//! Kimchi's `Pub` half-gate (`w0 - p = 0`, with the public input `p` added
//! by the verifier) has no counterpart here: a public input is a binding of
//! generalized PlonK ([`arrabbiata_gadget::Env::public`]).

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};

/// The coefficients of a half-gate: left, right, output, multiplication,
/// constant.
pub const HALF_COEFFICIENTS: usize = 5;

/// The cells of a half-gate: left, right, output.
pub const HALF_REGISTERS: usize = 3;

/// A half-gate, by what it computes (Kimchi's `GenericGateSpec`, without
/// `Pub`). The coefficients `None` takes are Kimchi's defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenericSpec<F> {
    /// `left l + right r + output o = 0`; by default `l = r = 1`, `o = -1`,
    /// so the output is the sum.
    Add {
        /// `l`.
        left: Option<F>,
        /// `r`.
        right: Option<F>,
        /// `o`.
        output: Option<F>,
    },
    /// `left right m + output o = 0`; by default `m = 1`, `o = -1`, so the
    /// output is the product.
    Mul {
        /// `o`.
        output: Option<F>,
        /// `m`.
        mul: Option<F>,
    },
    /// `left = c`.
    Const(F),
    /// `output = left + c`.
    Plus(F),
}

impl<F> GenericSpec<F>
where
    F: Field,
{
    /// The five coefficients `[l, r, o, m, c]`, as Kimchi's
    /// `create_generic_gadget` sets them.
    #[must_use]
    pub fn coefficients(&self) -> [F; HALF_COEFFICIENTS] {
        match *self {
            Self::Add {
                left,
                right,
                output,
            } => [
                left.unwrap_or(F::ONE),
                right.unwrap_or(F::ONE),
                output.unwrap_or(-F::ONE),
                F::ZERO,
                F::ZERO,
            ],
            Self::Mul { output, mul } => [
                F::ZERO,
                F::ZERO,
                output.unwrap_or(-F::ONE),
                mul.unwrap_or(F::ONE),
                F::ZERO,
            ],
            Self::Const(c) => [F::ONE, F::ZERO, F::ZERO, F::ZERO, -c],
            Self::Plus(c) => [F::ONE, F::ZERO, -F::ONE, F::ZERO, c],
        }
    }

    /// The output the half-gate pins, for the inputs `left` and `right`:
    /// the solution of its equation in the output cell, when the output has
    /// a non-zero coefficient. `None` for [`GenericSpec::Const`], which
    /// constrains its left input only.
    #[must_use]
    pub fn output(&self, left: F, right: F) -> Option<F> {
        let [l, r, o, m, c] = self.coefficients();
        let inverse = (-o).inverse()?;
        Some((l * left + r * right + m * left * right + c) * inverse)
    }
}

/// One Generic row, by its 10 coefficients: the gate itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenericRow<F> {
    /// The coefficients of the half-gate on columns 0-2, then of the one on
    /// columns 3-5.
    pub coefficients: [F; 2 * HALF_COEFFICIENTS],
}

impl<F> GenericRow<F>
where
    F: Field,
{
    /// The row of the half-gates `first` and, if any, `second`, as Kimchi's
    /// `create_generic_gadget` builds it.
    #[must_use]
    pub fn new(first: GenericSpec<F>, second: Option<GenericSpec<F>>) -> Self {
        let first = first.coefficients();
        let second = second.map_or([F::ZERO; HALF_COEFFICIENTS], |s| s.coefficients());
        let coefficients = core::array::from_fn(|i| {
            if i < HALF_COEFFICIENTS {
                first[i]
            } else {
                second[i - HALF_COEFFICIENTS]
            }
        });
        Self { coefficients }
    }
}

impl<F> TypedGadget<F> for GenericRow<F>
where
    F: Field,
{
    type Input = Array<Scalar, { 2 * HALF_REGISTERS }>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-generic";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Unit, E::Var>
    where
        E: Env<F>,
    {
        let w: [E::Var; 2 * HALF_REGISTERS] =
            core::array::from_fn(|i| env.copy(Cell::new(i, 0), &input[i]));
        let c: [E::Var; 2 * HALF_COEFFICIENTS] =
            core::array::from_fn(|i| env.fixed(Cell::new(i, 0), self.coefficients[i]));
        for half in 0..2 {
            let [left, right, output] =
                core::array::from_fn(|i| w[HALF_REGISTERS * half + i].clone());
            let [cl, cr, co, cm, cc] =
                core::array::from_fn(|i| c[HALF_COEFFICIENTS * half + i].clone());
            let product = left.clone() * right.clone();
            env.assert_zero(cl * left + cr * right + co * output + cm * product + cc);
        }
    }
}

/// The cells of one half's operation: its inputs where the half reads them,
/// and its output, computed from them, in the half's third cell.
fn half<F, E>(
    env: &mut E,
    spec: GenericSpec<F>,
    offset: usize,
    input: &[E::Var; 2],
) -> [E::Var; HALF_REGISTERS]
where
    F: Field,
    E: Env<F>,
{
    let left = env.copy(Cell::new(offset, 0), &input[0]);
    let right = env.copy(Cell::new(offset + 1, 0), &input[1]);
    let output = env.advice(Cell::new(offset + 2, 0), [&left, &right], |[a, b]| {
        spec.output(a, b).unwrap_or(F::ZERO)
    });
    [left, right, output]
}

/// The output of a half: its output cell, or its left input for
/// [`GenericSpec::Const`], whose output cell is not constrained.
fn result<F, V>(spec: GenericSpec<F>, cells: &[V; HALF_REGISTERS]) -> V
where
    F: Field,
    V: Clone,
{
    match spec {
        GenericSpec::Const(_) => cells[0].clone(),
        _ => cells[2].clone(),
    }
}

/// One operation on a Generic row: its inputs `[left, right]`, its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generic<F> {
    /// The operation.
    pub spec: GenericSpec<F>,
}

impl<F> TypedGadget<F> for Generic<F>
where
    F: Field,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-generic-operation";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Array<Scalar, 2>, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let cells = half(env, self.spec, 0, &input);
        // The second half is off: its coefficients are zero, so its cells
        // are not constrained; they hold zero.
        let unused: [E::Var; HALF_REGISTERS] = core::array::from_fn(|i| {
            env.advice(Cell::new(HALF_REGISTERS + i, 0), [], |[]| F::ZERO)
        });
        let row = GenericRow::new(self.spec, None);
        let [l, r, o] = cells.clone();
        let [a, b, c] = unused;
        env.call(&row, [l, r, o, a, b, c]);
        result(self.spec, &cells)
    }
}

/// Two operations on one Generic row, on columns 0-2 and 3-5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenericPair<F> {
    /// The operation on columns 0-2.
    pub first: GenericSpec<F>,
    /// The operation on columns 3-5.
    pub second: GenericSpec<F>,
}

impl<F> TypedGadget<F> for GenericPair<F>
where
    F: Field,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>);
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "kimchi-generic-pair";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (x, y): Of<Self::Input, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var>
    where
        E: Env<F>,
    {
        let first = half(env, self.first, 0, &x);
        let second = half(env, self.second, HALF_REGISTERS, &y);
        let row = GenericRow::new(self.first, Some(self.second));
        let [l1, r1, o1] = first.clone();
        let [l2, r2, o2] = second.clone();
        env.call(&row, [l1, r1, o1, l2, r2, o2]);
        [result(self.first, &first), result(self.second, &second)]
    }
}
