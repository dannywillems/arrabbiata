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
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_generalized_plonk::{Circuit, Gate, Variable, prove, setup, verify};
//! use arrabbiata_ipa::Ipa;
//! use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
//! use arrabbiata_pasta::{Fq as F, Pallas};
//! use arrabbiata_poly::Rotation;
//! use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
//! // Sponge domain tags: each transcript gets its own, and prover and
//! // verifier start from the same one.
//! let proof_domain = 2;
//!
//! // w(next) = s * w^3, a cubic recurrence scaled by the public input s,
//! // on rows 0 to 2; the last value is bound to public input 1.
//! let v = |i| Sparse::<F, 3, 4>::variable(Var::try_new(i).unwrap()).unwrap();
//! let cube = v(0).checked_mul(&v(0)).unwrap().checked_mul(&v(0)).unwrap();
//! let selector = Some(0);
//! let polynomial = v(1) - v(2).checked_mul(&cube).unwrap();
//! let gate = Gate::new(selector, polynomial);
//! let variables = vec![
//!     Variable::Witness { column: 0, rotation: Rotation::CURR },
//!     Variable::Witness { column: 0, rotation: Rotation::NEXT },
//!     Variable::Public(0),
//! ];
//! let selector = vec![F::ONE, F::ONE, F::ONE, F::ZERO];
//! let num_rows = 4;
//! let witness_columns = 1;
//! let fixed = vec![];
//! let selectors = vec![selector];
//! let gates = vec![gate];
//! let sigma = (0..4).collect();
//! let public_inputs = 2;
//! let bindings = vec![(1, 3)];
//! let circuit = Circuit::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
//!
//! let s = F::from_u64(2);
//! let mut column = vec![F::from_u64(3)];
//! for i in 0..3 {
//!     let x: F = column[i];
//!     column.push(s * x * x * x);
//! }
//! let public = [s, column[3]];
//! let mut generators = pallas_sponge(0);
//! let size = 16;
//! let ipa_transcript = vesta_sponge(1);
//! let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let (proof, _) = prove(&pk, &pcs, vesta_sponge(proof_domain), &public, &[column]).unwrap();
//! verify(&vk, &pcs, vesta_sponge(proof_domain), &public, proof).unwrap();
//! ```

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
