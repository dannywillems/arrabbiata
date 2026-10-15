//! Accumulation of PlonKish circuits of any degree: the folding scheme of
//! the Arrabbiata note, on the circuits of [`arrabbiata_plonkish`], with
//! the cross-terms computed from multivariate polynomials
//! ([`arrabbiata_mvpoly`]).
//!
//! # The morphisms
//!
//! An accumulation scheme for a relation `R` is a relaxation `R'` with
//! three reductions of knowledge ([`arrabbiata_reduction`]):
//!
//! ```text
//! Embed : R -> R'              the chart u = 1, E = 0       (embed)
//! Fold  : R' x R' -> R'        a 2-folding scheme           (fold)
//! Dec   : R' -> R_top          an argument for R'           (the decider)
//! ```
//!
//! `R` is [`Satisfiability`]: a witness satisfies a circuit, for the given
//! public inputs. `R'` is [`Relaxed`]: commitments to the witness columns
//! and to an error column, opened by a relaxed witness satisfying the
//! relaxation of the circuit of degree `D`
//! ([`arrabbiata_plonkish::relaxed`]). The decider is in another crate
//! (`arrabbiata-plonk-decider`), because it depends on `R'` only.
//!
//! Composites come from the category: accumulating a fresh step into an
//! accumulator, `R x R' -> R'`, is `Fold . (Embed x Id)` ([`Accumulate`]),
//! and `Dec . Embed` is a SNARK for `R`.
//!
//! # The fold
//!
//! On a row turning on gadget `g`, the relaxed condition is
//! `P(u, c, w) = sum_k c_k f^D_(g,k)(u, w) = E`, homogeneous of degree
//! `D + 1` in `(u, c, w)`. On the line through two relaxed instances,
//!
//! ```text
//! P(u1 + r u2, c1 + r c2, w1 + r w2) = E1 + r T_1 + ... + r^D T_D + r^(D+1) E2,
//! ```
//!
//! where homogeneity pins the two ends. The cross-terms `T_k` are computed
//! per row from the constraints in normal form (Hasse derivatives of their
//! homogenization, [`arrabbiata_mvpoly::combined_cross_terms_with`]); only
//! the gadget a row turns on is expanded, since the selectors are one-hot
//! (the branched-gate observation of Bunz and Chen, *Protostar*, ePrint
//! 2023/620, section 3.6.1). The prover commits to them, the challenge `r`
//! is drawn after, and every component folds linearly:
//!
//! ```text
//! u = u1 + r u2,  c = c1 + r c2,  x = x1 + r x2,  [w] = [w1] + r [w2],
//! [E] = [E1] + sum_k r^k [T_k] + r^(D+1) [E2].
//! ```
//!
//! # Commitments: the Lagrange basis
//!
//! A column is committed as the polynomial interpolating it on the domain
//! of the circuit: `commit(interpolate(column))`. Interpolation is linear,
//! so this is a homomorphic commitment to the column, which is all the
//! fold needs; and it is the commitment a polynomial IOP makes to the same
//! column, which is what lets the decider open the accumulator's
//! commitments directly ([`Params::commit_column`]).
//!
//! Complete runs: `tests/examples.rs` (an incremental computation, two
//! accumulators folded); laws: `tests/accumulation.rs`.

#![no_std]

extern crate alloc;

mod embed;
mod fold;
mod instance;
mod params;

pub use crate::{
    embed::Embed,
    fold::{Accumulate, CrossTerms, Fold, accumulate},
    instance::{Columns, Digest, PublicInputs, RelaxedInstance},
    params::{Params, Relaxed, Satisfiability},
};

use core::fmt;

/// Why setting up, proving, verifying or checking failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The degree `D` is below the degree of the gate set in the witness.
    Degree,
    /// `N` variables cannot hold the `W` columns at every rotation the gate
    /// set reads.
    Layout,
    /// The field has no domain of the circuit size, or the commitment key
    /// is too short for a column.
    Domain,
    /// An instance or a witness has the wrong shape.
    Shape,
    /// The witness does not carry the instance's public inputs.
    PublicInputs,
    /// Commitment `k` of the instance (a witness column for `k < W`, the
    /// error column for `k = W`) is not the commitment of the witness.
    Commitment(usize),
    /// The witness does not satisfy the (relaxed) circuit.
    Unsatisfied,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Degree => write!(f, "the degree is below the degree of the gate set"),
            Self::Layout => write!(f, "too few variables for the columns and rotations"),
            Self::Domain => write!(f, "no domain, or no commitment key, of the circuit size"),
            Self::Shape => write!(f, "an instance or a witness has the wrong shape"),
            Self::PublicInputs => write!(f, "the public inputs do not match"),
            Self::Commitment(k) => write!(f, "commitment {k} does not open to the witness"),
            Self::Unsatisfied => write!(f, "the witness does not satisfy the circuit"),
        }
    }
}

impl core::error::Error for Error {}
