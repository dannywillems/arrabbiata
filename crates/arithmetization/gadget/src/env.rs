//! The signature a gadget is written against, and the gadgets themselves.
//!
//! [`Env`] is a signature: a ring of variables (`Add`, `Sub`, `Mul`, `Neg`
//! on [`Env::Var`]) and the operations that place variables in the table
//! and constrain them. A gadget's [`TypedGadget::synthesize`] is generic in
//! the environment, so it is one term of that signature, and every
//! environment is a model of it: the builder reads it as constraints and,
//! with values, as a witness, from the same code.

use crate::shape::{Of, Shape};
use arrabbiata_field::Field;
use core::ops::{Add, Mul, Neg, Sub};

/// A cell of the table, relative to the first row of the gadget placing it:
/// `row` 0 is that row, `row` 1 the next one, and so on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    /// The column.
    pub column: usize,
    /// The row, relative to the gadget.
    pub row: usize,
}

impl Cell {
    /// The cell of `column` on the gadget's `row`.
    #[must_use]
    pub const fn new(column: usize, row: usize) -> Self {
        Self { column, row }
    }
}

/// The operations a gadget uses.
///
/// Every operation that writes a cell either constrains it or says that it
/// does not: [`Env::define`] and [`Env::place`] constrain the cell to a
/// value, [`Env::copy`] wires it to another cell, and only [`Env::advice`]
/// leaves it free, to be constrained by a later [`Env::assert_zero`].
pub trait Env<F>
where
    F: Field,
{
    /// A variable: a polynomial in the cells, with its value when a
    /// witness is computed.
    type Var: Clone
        + Add<Output = Self::Var>
        + Sub<Output = Self::Var>
        + Mul<Output = Self::Var>
        + Neg<Output = Self::Var>;

    /// The constant `value`.
    fn constant(&mut self, value: F) -> Self::Var;

    /// A cell of fixed column `at.column`, holding `value`: the parameters
    /// of one instance of a gadget (round constants, coefficients) go here,
    /// so that every instance has the same constraints.
    fn fixed(&mut self, at: Cell, value: F) -> Self::Var;

    /// The witness cell `at`, constrained to equal `value`.
    fn define(&mut self, at: Cell, value: Self::Var) -> Self::Var;

    /// The witness cell `at`, holding `hint` of the values of `inputs`, and
    /// not constrained: a gadget using it must constrain it itself.
    fn advice<const K: usize>(
        &mut self,
        at: Cell,
        inputs: [&Self::Var; K],
        hint: impl FnOnce([F; K]) -> F,
    ) -> Self::Var;

    /// The witness cell `at`, wired to the cell `value` is: a copy
    /// constraint of the permutation argument. Both cells must be in
    /// permuted columns.
    fn copy(&mut self, at: Cell, value: &Self::Var) -> Self::Var;

    /// The witness cell `at` holding `value`: `value` itself when it is that
    /// cell, a copy when it is a cell of a permuted column and `at` is in
    /// one, a definition otherwise. Placing twice is placing once.
    fn place(&mut self, at: Cell, value: &Self::Var) -> Self::Var;

    /// Make the cell `value` is a public input. It must be a cell of the
    /// first column.
    fn public(&mut self, value: &Self::Var);

    /// Constrain `value` to be zero, on the rows of this gadget's instances.
    fn assert_zero(&mut self, value: Self::Var);

    /// Constrain `a` and `b` to be equal.
    fn assert_equal(&mut self, a: Self::Var, b: Self::Var) {
        self.assert_zero(a - b);
    }

    /// Run `gadget` from the current row.
    fn call<G>(&mut self, gadget: &G, input: Of<G::Input, Self::Var>) -> Of<G::Output, Self::Var>
    where
        G: TypedGadget<F>;
}

/// A gadget: a typed map from its input to its output, written once for
/// every environment.
///
/// Its constraints are the same for every instance: what changes from one
/// instance to the next (constants, coefficients) is written in fixed
/// columns with [`Env::fixed`]. The builder checks it, which is what lets
/// all instances share one selector.
pub trait TypedGadget<F>
where
    F: Field,
{
    /// The input.
    type Input: Shape;

    /// The output.
    type Output: Shape;

    /// The name, which identifies the gadget's gates.
    const NAME: &'static str;

    /// The number of rows the gadget takes; its cells may also reach into
    /// the rows after, which the next gadget then reads.
    const ROWS: usize;

    /// The constraints and the witness of one instance.
    fn synthesize<E>(
        &self,
        env: &mut E,
        input: Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>;
}

/// `first`, then `second` on its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chain<A, B> {
    /// The first gadget.
    pub first: A,
    /// The second gadget.
    pub second: B,
}

impl<F, A, B> TypedGadget<F> for Chain<A, B>
where
    F: Field,
    A: TypedGadget<F>,
    B: TypedGadget<F, Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;
    const NAME: &'static str = "chain";
    const ROWS: usize = A::ROWS + B::ROWS;

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let middle = env.call(&self.first, input);
        env.call(&self.second, middle)
    }
}

/// `gadget`, `N` times, each on the previous output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Repeat<G, const N: usize> {
    /// The repeated gadget.
    pub gadget: G,
}

impl<F, G, const N: usize> TypedGadget<F> for Repeat<G, N>
where
    F: Field,
    G: TypedGadget<F, Output = <G as TypedGadget<F>>::Input>,
{
    type Input = G::Input;
    type Output = G::Output;
    const NAME: &'static str = "repeat";
    const ROWS: usize = G::ROWS * N;

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let mut state = input;
        for _ in 0..N {
            state = env.call(&self.gadget, state);
        }
        state
    }
}
