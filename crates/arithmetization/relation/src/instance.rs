//! Instances and witnesses.
//!
//! A fresh [`Instance`] is the commitment to a strict witness (a trace) plus
//! its aggregation coefficients; implicitly `u = 1` and `E = 0`. A
//! [`RelaxedInstance`] makes `u` and the error commitment explicit; it is
//! what folding produces and consumes. The two are distinct types so that a
//! function expecting an accumulator cannot be handed a fresh instance, or
//! the converse, without going through [`crate::ConstraintSystem::relax`].

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, tag, typed::Label};

/// A fresh instance: column commitments and aggregation coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance<C, K, const W: usize> {
    columns: [C; W],
    coefficients: K,
}

impl<C, K, const W: usize> Instance<C, K, W> {
    /// The instance with the given column commitments and coefficients.
    pub const fn new(columns: [C; W], coefficients: K) -> Self {
        Self {
            columns,
            coefficients,
        }
    }

    /// The column commitments.
    pub const fn columns(&self) -> &[C; W] {
        &self.columns
    }

    /// The aggregation coefficients.
    pub const fn coefficients(&self) -> &K {
        &self.coefficients
    }

    /// The column commitments and the coefficients.
    pub fn into_parts(self) -> ([C; W], K) {
        (self.columns, self.coefficients)
    }
}

/// A relaxed instance `(u, c, [w], [E])`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedInstance<F, C, K, const W: usize> {
    u: F,
    coefficients: K,
    columns: [C; W],
    error: C,
}

impl<F, C, K, const W: usize> RelaxedInstance<F, C, K, W>
where
    F: Field,
{
    /// The relaxed instance with the given parts.
    ///
    /// No invariant ties the parts together; the relation they satisfy is
    /// checked by [`crate::ConstraintSystem::check_relaxed`].
    pub const fn new(u: F, coefficients: K, columns: [C; W], error: C) -> Self {
        Self {
            u,
            coefficients,
            columns,
            error,
        }
    }

    /// The homogenizing scalar `u`.
    pub const fn u(&self) -> F {
        self.u
    }

    /// The aggregation coefficients.
    pub const fn coefficients(&self) -> &K {
        &self.coefficients
    }

    /// The column commitments.
    pub const fn columns(&self) -> &[C; W] {
        &self.columns
    }

    /// The commitment to the error vector.
    pub const fn error(&self) -> &C {
        &self.error
    }
}

impl<F, C, K, const W: usize> Label for RelaxedInstance<F, C, K, W> {
    const LABEL: u64 = tag::RELAXED_INSTANCE;
}

impl<F, C, K, const W: usize> Absorb<F> for RelaxedInstance<F, C, K, W>
where
    F: Field,
    C: Absorb<F>,
    K: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::RELAXED_INSTANCE));
        transcript.absorb(self.u);
        self.coefficients.absorb_into(transcript);
        absorb_length(transcript, W);
        let mut column = 0;
        while column < W {
            self.columns[column].absorb_into(transcript);
            column += 1;
        }
        self.error.absorb_into(transcript);
    }
}

/// A relaxed witness: the witness columns and the error vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedWitness<F, const W: usize> {
    columns: [Vec<F>; W],
    error: Vec<F>,
}

impl<F, const W: usize> RelaxedWitness<F, W> {
    /// The witness with the given columns and error vector. Shapes are
    /// checked by the relation, not here.
    pub const fn new(columns: [Vec<F>; W], error: Vec<F>) -> Self {
        Self { columns, error }
    }

    /// The witness columns.
    pub const fn columns(&self) -> &[Vec<F>; W] {
        &self.columns
    }

    /// The error vector.
    pub fn error(&self) -> &[F] {
        &self.error
    }
}
