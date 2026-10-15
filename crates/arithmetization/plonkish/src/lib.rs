//! PlonKish circuits: gadgets, one-hot selectors, rotations, copy
//! constraints and public inputs, and their relaxation.
//!
//! A circuit is a table of `W` witness columns and `X` fixed columns over
//! `n` rows, with a [gate set](gate::GateSet) of gadgets each turned on by
//! its own selector, at most one gadget per row (one-hot by construction),
//! constraints reading any row at a fixed offset from the constrained one
//! ([`Rotation`](arrabbiata_poly::Rotation)), copy constraints between any
//! two witness cells, and public inputs in witness column 0
//! ([`circuit`]). The relation it defines, *a witness satisfies the
//! circuit*, is what a SNARK proves ([`circuit::Circuit::is_satisfied`]).
//!
//! Its [relaxation](relaxed) is the relation an accumulation scheme folds:
//! every constraint homogenized with a slack scalar `u`, the constraints
//! of a row combined by coefficients `c`, and an error column `E` on the
//! right-hand side. The circuit relation is the chart `u = 1, E = 0` of it.
//!
//! This crate is the arithmetization only. The PlonK SNARK lives in
//! `arrabbiata-plonk`, the accumulation scheme in
//! `arrabbiata-plonkish-folding` and the decider in
//! `arrabbiata-plonk-decider`; all three read circuits from here.

#![no_std]

extern crate alloc;

pub mod circuit;
pub mod gate;
pub mod relaxed;
pub mod vanilla;

use core::fmt;

pub use crate::{
    circuit::{Builder, Cell, Circuit, Witness},
    gate::{Gadget, GateSet, Member, Selector},
    relaxed::RelaxedWitness,
};

/// Why a circuit could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitError {
    /// A circuit or a witness has the wrong shape: a number of rows that is
    /// not a power of two, a column of the wrong length, a copy
    /// permutation that is not one, a public row out of range.
    Shape,
    /// A gadget reads a row outside the rows of a path (see
    /// [`Builder::new`]).
    Rotation,
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => write!(f, "a circuit or a witness has the wrong shape"),
            Self::Rotation => write!(f, "a gadget reads a row outside the circuit"),
        }
    }
}

impl core::error::Error for CircuitError {}
