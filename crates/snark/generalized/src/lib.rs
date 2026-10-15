//! Generalized PlonK: any number of witness columns, fixed columns,
//! selectors and public inputs, with gates that are any multivariate
//! polynomials of degree at most `D`.
//!
//! A gate is a polynomial in `N` variables ([`arrabbiata_mvpoly::Sparse`]),
//! each variable a witness or fixed column read at a rotation of the
//! constrained row, or a public input ([`Variable`]); a selector turns it
//! on ([`Gate`]). Public inputs are read by gates as scalars, and some are
//! also bound to cells of witness column 0. Copy constraints range over all
//! witness cells. See [`circuit`] for the relation and [`protocol`] for the
//! proof system, PlonK's polynomial IOP with this gate identity.
//!
//! The PlonKish circuits of [`arrabbiata_plonkish`] are generalized
//! circuits ([`plonkish::from_plonkish`]), and so is the relaxation an
//! accumulation scheme folds: its slack and coefficients are public inputs
//! the gates read, its error a witness column. That is how the decider of
//! `arrabbiata-plonkish-folding` is built.
//!
//! Examples, each a complete run from a circuit to a verified proof:
//! `tests/examples.rs` (an iterated map with a public parameter, a bound
//! public input), `tests/lookups.rs` (range checks, a XOR table, both
//! lookup arguments) and `tests/permutation.rs` (a sum of 4-bit numbers,
//! copies on some columns only).

#![no_std]

extern crate alloc;

pub mod circuit;
pub mod lookup;
pub mod plonkish;
pub mod protocol;

use core::fmt;

pub use crate::{
    circuit::{Circuit, Gate, Variable},
    lookup::{Argument, Lookup},
    protocol::{
        Commitments, GeneralizedPlonk, GeneralizedTranscript, Keys, Opened, Preprocessed, Proof,
        ProverKey, PublicInputs, Tagged, VerifierKey, max_coefficients, prove, quotient_factor,
        rotated, setup, verify,
    },
};

/// Why building a circuit, setup, proving or verification failed.
///
/// ```
/// use arrabbiata_generalized_plonk::Error;
/// assert_eq!(Error::IdentityFailed.to_string(), "the identity does not hold at zeta");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A circuit, a witness or a proof has the wrong shape.
    Shape,
    /// The number of public inputs does not match the circuit.
    PublicInputs,
    /// The field has no domain of the size the circuit needs.
    DomainTooLarge,
    /// The coset shifts of the permutation argument are not disjoint.
    CosetShifts,
    /// The witness does not satisfy the circuit.
    Unsatisfied,
    /// An opening proof does not verify; the index is in the order of
    /// [`Opened::iter`].
    Opening(usize),
    /// The identity does not hold at `zeta`.
    IdentityFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => write!(f, "a circuit, witness or proof has the wrong shape"),
            Self::PublicInputs => write!(f, "wrong number of public inputs"),
            Self::DomainTooLarge => write!(f, "the field has no domain of the circuit size"),
            Self::CosetShifts => write!(f, "the coset shifts are not in distinct cosets"),
            Self::Unsatisfied => write!(f, "the witness does not satisfy the circuit"),
            Self::Opening(i) => write!(f, "opening proof {i} does not verify"),
            Self::IdentityFailed => write!(f, "the identity does not hold at zeta"),
        }
    }
}

impl core::error::Error for Error {}
