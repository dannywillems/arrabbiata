//! Typed gadgets, written once, compiled to generalized PlonK.
//!
//! A gadget ([`TypedGadget`]) has a typed input and output ([`Shape`]) and
//! one method, [`TypedGadget::synthesize`], generic in the environment
//! ([`Env`]). The environment is a signature: a ring of variables, and the
//! operations that place them in the table and constrain them. Because
//! `synthesize` is generic, it cannot look at what a variable is; it is one
//! term of the signature, and the builder interprets it in two models at
//! once:
//!
//! - the constraints: each variable is a polynomial in the cells, and each
//!   [`Env::assert_zero`] a gate of generalized PlonK;
//! - the witness: each variable is a field element, and each
//!   [`Env::assert_zero`] a check, reported on the row that fails.
//!
//! A variable is the pair of the two ([`Wire`]), so the witness is the
//! constraints evaluated, by construction: [`evaluate`] is the function the
//! gadget computes, and there is no second implementation of it to keep in
//! sync.
//!
//! The API is shaped against the usual mistakes:
//!
//! - every write says how the cell is constrained: [`Env::define`] (to a
//!   value), [`Env::copy`] (to another cell), [`Env::place`] (whichever
//!   applies), or [`Env::advice`] (not at all, until an
//!   [`Env::assert_zero`]);
//! - cells are addressed relative to the gadget ([`Cell`]), and the builder
//!   lays gadgets out row after row ([`Env::call`]); a cell written twice is
//!   an error;
//! - instances of a gadget share their gates and one selector; what differs
//!   between instances goes in fixed columns ([`Env::fixed`]), and an
//!   instance whose constraints differ is rejected
//!   ([`Error::Inconsistent`]);
//! - gadgets compose by type ([`Chain`], [`Repeat`]): the output shape of
//!   one is the input shape of the next.
//!
//! The circuit is [`circuit`] (no witness: what the setup needs), the
//! witness [`witness`], and its figures (per gate: the constraints, their
//! degrees, the instances, the cells read) [`figures`]. Examples, from a squaring map to a Poseidon hash,
//! are in `tests/`.

#![no_std]

extern crate alloc;

pub mod builder;
pub mod env;
pub mod expr;
pub mod shape;

pub use crate::{
    builder::{
        Builder, Compiled, Error, Figures, GateConstraints, GateFigures, Layout, Rom, Violation,
        Witnessed, assign, circuit, compile, evaluate, figures, gates, measure, witness,
    },
    env::{Cell, Chain, Env, Placed, Repeat, TypedGadget},
    expr::Wire,
    shape::{Array, Of, Scalar, Shape, Unit},
};
