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
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonkish::{Builder, Cell, Gadget};
//! use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
//!
//! /// The Poseidon S-box, `b = a^5`.
//! struct SBox;
//! impl Gadget<F, 2, 0> for SBox {
//!     const NAME: &'static str = "sbox";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
//!         let a2 = a.clone() * a.clone();
//!         vec![b - a2.clone() * a2 * a]
//!     }
//! }
//!
//! let mut b = Builder::<F, (SBox, ()), 2, 0, 1>::new();
//! let x = F::from_u64(3);
//! let r0 = b.row::<SBox, _>([x, x.pow(5)], []);
//! let out = b.plain_row([x.pow(5), F::ZERO], []);
//! b.copy(Cell::new(WitnessColumn::new::<1>(), r0), Cell::new(WitnessColumn::new::<0>(), out));
//! b.public(out);
//! let (circuit, witness) = b.build().unwrap();
//! assert!(circuit.is_satisfied(&witness));
//! ```

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
///
/// ```
/// use arrabbiata_plonkish::CircuitError;
/// assert_eq!(CircuitError::Shape.to_string(), "a circuit or a witness has the wrong shape");
/// ```
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
