//! The circuit model of the signature of fields
//! ([`arrabbiata_field::FieldModel`]): a computation written once against
//! the signature is, in [`arrabbiata_field::Native`], the field's values,
//! and in [`Circuit`], a circuit of Kimchi's Generic gate, each operation
//! laying out the gate that constrains its result, so the same computation
//! is a circuit and, with values, its witness.
//!
//! In [`Circuit`] a value is an affine form `s x + d` in at most one cell
//! `x` ([`Affine`]): constants, negations, scalings and additions of
//! constants cost nothing, and an addition or a product of two cells is one
//! Generic half-gate `l x + r y + o z + m x y + c = 0` (snarky's `Basic`
//! constraint), which absorbs both affine forms. That is what the Generic
//! gate can express in one row, so the forms are the gate's own normal
//! form.

use alloc::vec::Vec;
use core::marker::PhantomData;

use arrabbiata_field::{Field, FieldModel};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi_gadgets::generic::{GenericRow, HALF_COEFFICIENTS, HALF_REGISTERS};

/// `scale cell + offset`, or `offset` alone without a cell: a value of
/// [`Circuit`].
#[derive(Clone, Debug)]
pub struct Affine<V, F> {
    /// The cell.
    pub cell: Option<V>,
    /// Its scale.
    pub scale: F,
    /// The constant term.
    pub offset: F,
}

impl<V, F> Affine<V, F>
where
    F: Field,
{
    /// The cell `x` itself.
    #[must_use]
    pub const fn of(x: V) -> Self {
        Self {
            cell: Some(x),
            scale: F::ONE,
            offset: F::ZERO,
        }
    }

    /// The constant `c`.
    #[must_use]
    pub const fn constant(c: F) -> Self {
        Self {
            cell: None,
            scale: F::ZERO,
            offset: c,
        }
    }

    /// `k self + c`.
    fn affine(&self, k: F, c: F) -> Self
    where
        V: Clone,
    {
        match &self.cell {
            Some(x) => Self {
                cell: Some(x.clone()),
                scale: self.scale * k,
                offset: self.offset * k + c,
            },
            None => Self::constant(self.offset * k + c),
        }
    }
}

/// A Generic half-gate with any coefficients `[l, r, o, m, c]` on columns
/// 0-2, its inputs `[x, y]` copied in, its output `z` solving `l x + r y + o
/// z + m x y + c = 0` (with `o = -1`, `z = l x + r y + m x y + c`), and the
/// other half off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Basic<F> {
    /// `[l, r, o, m, c]`.
    pub coefficients: [F; HALF_COEFFICIENTS],
}

impl<F> TypedGadget<F> for Basic<F>
where
    F: Field,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "pickles-basic";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Array<Scalar, 2>, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let [l, r, o, m, c] = self.coefficients;
        let x = env.copy(Cell::new(0, 0), &input[0]);
        let y = env.copy(Cell::new(1, 0), &input[1]);
        let z = env.advice(Cell::new(2, 0), [&x, &y], |[a, b]| {
            let numerator = l * a + r * b + m * a * b + c;
            (-o).inverse().map_or(F::ZERO, |i| numerator * i)
        });
        row(env, self.coefficients, [x, y, z.clone()]);
        z
    }
}

/// `x^(-1)` on one Generic row: `x` copied into column 0, its inverse in
/// column 1, and `x x^(-1) - 1 = 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inverse;

impl<F> TypedGadget<F> for Inverse
where
    F: Field,
{
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "pickles-inverse";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let x = env.copy(Cell::new(0, 0), &input);
        let inverse = env.advice(Cell::new(1, 0), [&x], |[a]| a.inverse().unwrap_or(F::ZERO));
        let unused = env.advice(Cell::new(2, 0), [], |[]| F::ZERO);
        let coefficients = [F::ZERO, F::ZERO, F::ZERO, F::ONE, -F::ONE];
        row(env, coefficients, [x, inverse.clone(), unused]);
        inverse
    }
}

/// The constant `c` in a cell, column 0 of a Generic row: `x - c = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constant<F> {
    /// The constant.
    pub value: F,
}

impl<F> TypedGadget<F> for Constant<F>
where
    F: Field,
{
    type Input = Unit;
    type Output = Scalar;
    const NAME: &'static str = "pickles-constant";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, (): ()) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let value = self.value;
        let x = env.advice(Cell::new(0, 0), [], |[]| value);
        let a = env.advice(Cell::new(1, 0), [], |[]| F::ZERO);
        let b = env.advice(Cell::new(2, 0), [], |[]| F::ZERO);
        let coefficients = [F::ONE, F::ZERO, F::ZERO, F::ZERO, -value];
        row(env, coefficients, [x.clone(), a, b]);
        x
    }
}

/// The Generic row of `coefficients` on the cells `first` (columns 0-2),
/// its second half off, holding zeros.
fn row<F, E>(env: &mut E, coefficients: [F; HALF_COEFFICIENTS], first: [E::Var; HALF_REGISTERS])
where
    F: Field,
    E: Env<F>,
{
    let unused: [E::Var; HALF_REGISTERS] =
        core::array::from_fn(|i| env.advice(Cell::new(HALF_REGISTERS + i, 0), [], |[]| F::ZERO));
    let row = GenericRow {
        coefficients: core::array::from_fn(|i| {
            if i < HALF_COEFFICIENTS {
                coefficients[i]
            } else {
                F::ZERO
            }
        }),
    };
    let [a, b, c] = first;
    let [d, e, f] = unused;
    env.call(&row, [a, b, c, d, e, f]);
}

/// The cells a row of permuted columns holds: Kimchi copies between its
/// first seven columns.
pub const LOAD: usize = 7;

/// Up to [`LOAD`] values written in the permuted cells of one row, where
/// the gates after copy them from. A circuit's inputs come in this way; the
/// row has no gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Load;

impl<F> TypedGadget<F> for Load
where
    F: Field,
{
    type Input = Array<Scalar, LOAD>;
    type Output = Array<Scalar, LOAD>;
    const NAME: &'static str = "pickles-load";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&input[i]], |[v]| v))
    }
}

/// `values` loaded into cells, [`LOAD`] per row.
pub fn load<F, E>(env: &mut E, values: &[E::Var]) -> Vec<E::Var>
where
    F: Field,
    E: Env<F>,
{
    let mut cells = Vec::with_capacity(values.len());
    for chunk in values.chunks(LOAD) {
        let row: [E::Var; LOAD] = core::array::from_fn(|i| {
            chunk
                .get(i)
                .cloned()
                .unwrap_or_else(|| env.constant(F::ZERO))
        });
        let placed = env.call(&Load, row);
        cells.extend(placed.into_iter().take(chunk.len()));
    }
    cells
}

/// The circuit model: each operation on two cells lays out one Generic row
/// in `env` (see the module documentation).
#[derive(Debug)]
pub struct Circuit<'e, F, E> {
    env: &'e mut E,
    field: PhantomData<F>,
}

impl<'e, F, E> Circuit<'e, F, E>
where
    F: Field,
    E: Env<F>,
{
    /// The model laying its gates out in `env`, from its current row.
    pub const fn new(env: &'e mut E) -> Self {
        Self {
            env,
            field: PhantomData,
        }
    }

    /// The environment.
    pub const fn env(&mut self) -> &mut E {
        self.env
    }

    /// A cell holding `a`: its own cell when it is one, else a row computing
    /// it.
    pub fn cell(&mut self, a: &Affine<E::Var, F>) -> E::Var {
        match &a.cell {
            Some(x) if a.scale == F::ONE && a.offset == F::ZERO => x.clone(),
            Some(x) => {
                let coefficients = [a.scale, F::ZERO, -F::ONE, F::ZERO, a.offset];
                self.env
                    .call(&Basic { coefficients }, [x.clone(), x.clone()])
            }
            None => self.env.call(&Constant { value: a.offset }, ()),
        }
    }

    /// The Basic row of `coefficients` on `x` and `y`, as a value.
    fn basic(
        &mut self,
        coefficients: [F; HALF_COEFFICIENTS],
        x: &E::Var,
        y: &E::Var,
    ) -> Affine<E::Var, F> {
        let z = self
            .env
            .call(&Basic { coefficients }, [x.clone(), y.clone()]);
        Affine::of(z)
    }
}

impl<F, E> FieldModel<F> for Circuit<'_, F, E>
where
    F: Field,
    E: Env<F>,
{
    type Value = Affine<E::Var, F>;

    fn constant(&mut self, c: F) -> Self::Value {
        Affine::constant(c)
    }

    fn add(&mut self, a: &Self::Value, b: &Self::Value) -> Self::Value {
        match (&a.cell, &b.cell) {
            (None, _) => b.affine(F::ONE, a.offset),
            (_, None) => a.affine(F::ONE, b.offset),
            (Some(x), Some(y)) => {
                let coefficients = [a.scale, b.scale, -F::ONE, F::ZERO, a.offset + b.offset];
                self.basic(coefficients, x, y)
            }
        }
    }

    fn mul(&mut self, a: &Self::Value, b: &Self::Value) -> Self::Value {
        match (&a.cell, &b.cell) {
            (None, _) => b.affine(a.offset, F::ZERO),
            (_, None) => a.affine(b.offset, F::ZERO),
            (Some(x), Some(y)) => {
                // (s x + d)(t y + e) = s t x y + s e x + t d y + d e.
                let coefficients = [
                    a.scale * b.offset,
                    b.scale * a.offset,
                    -F::ONE,
                    a.scale * b.scale,
                    a.offset * b.offset,
                ];
                self.basic(coefficients, x, y)
            }
        }
    }

    fn neg(&mut self, a: &Self::Value) -> Self::Value {
        a.affine(-F::ONE, F::ZERO)
    }

    fn inverse(&mut self, a: &Self::Value) -> Self::Value {
        if a.cell.is_none() {
            return Affine::constant(a.offset.inverse().unwrap_or(F::ZERO));
        }
        let x = self.cell(a);
        let inverse = self.env.call(&Inverse, x);
        Affine::of(inverse)
    }
}
