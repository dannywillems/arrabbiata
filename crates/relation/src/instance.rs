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
///
/// ```
/// use arrabbiata_commit::IdentityCommitment;
/// use arrabbiata_relation::examples;
/// use arrabbiata_transcript::toy_sponge;
///
/// let cs = examples::squaring(4);
/// let trace = examples::squaring_trace(&cs, 3);
/// let instance = cs.commit(&IdentityCommitment, &trace, &mut toy_sponge(0));
/// assert_eq!(instance.columns()[0].values()[0], arrabbiata_field::Goldilocks::new(3));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance<C, K, const W: usize> {
    columns: [C; W],
    coefficients: K,
}

impl<C, K, const W: usize> Instance<C, K, W> {
    /// The instance with the given column commitments and coefficients.
    ///
    /// ```
    /// use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    /// let i = Instance::new([7u8], NoCoefficients);
    /// assert_eq!(i.columns(), &[7]);
    /// ```
    pub const fn new(columns: [C; W], coefficients: K) -> Self {
        Self {
            columns,
            coefficients,
        }
    }

    /// The column commitments.
    ///
    /// ```
    /// use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    /// assert_eq!(Instance::new([1u8, 2], NoCoefficients).columns().len(), 2);
    /// ```
    pub const fn columns(&self) -> &[C; W] {
        &self.columns
    }

    /// The aggregation coefficients.
    ///
    /// ```
    /// use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    /// assert_eq!(Instance::new([1u8], NoCoefficients).coefficients(), &NoCoefficients);
    /// ```
    pub const fn coefficients(&self) -> &K {
        &self.coefficients
    }

    /// The column commitments and the coefficients.
    ///
    /// ```
    /// use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    /// let (columns, _) = Instance::new([1u8], NoCoefficients).into_parts();
    /// assert_eq!(columns, [1]);
    /// ```
    pub fn into_parts(self) -> ([C; W], K) {
        (self.columns, self.coefficients)
    }
}

/// A relaxed instance `(u, c, [w], [E])`.
///
/// ```
/// use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
/// use arrabbiata_field::{Field, Goldilocks as F};
/// let i = RelaxedInstance::new(F::ONE, NoCoefficients, [10u8], 0u8);
/// assert_eq!(i.u(), F::ONE);
/// ```
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
    ///
    /// ```
    /// use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    /// use arrabbiata_field::Goldilocks as F;
    /// let i = RelaxedInstance::new(F::new(2), NoCoefficients, [1u8, 2], 3u8);
    /// assert_eq!(i.error(), &3);
    /// ```
    pub const fn new(u: F, coefficients: K, columns: [C; W], error: C) -> Self {
        Self {
            u,
            coefficients,
            columns,
            error,
        }
    }

    /// The homogenizing scalar `u`.
    ///
    /// ```
    /// use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    /// use arrabbiata_field::Goldilocks as F;
    /// assert_eq!(RelaxedInstance::new(F::new(4), NoCoefficients, [0u8], 0u8).u(), F::new(4));
    /// ```
    pub const fn u(&self) -> F {
        self.u
    }

    /// The aggregation coefficients.
    ///
    /// ```
    /// use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    /// use arrabbiata_field::Goldilocks as F;
    /// let i = RelaxedInstance::new(F::new(1), NoCoefficients, [0u8], 0u8);
    /// assert_eq!(i.coefficients(), &NoCoefficients);
    /// ```
    pub const fn coefficients(&self) -> &K {
        &self.coefficients
    }

    /// The column commitments.
    ///
    /// ```
    /// use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    /// use arrabbiata_field::Goldilocks as F;
    /// let i = RelaxedInstance::new(F::new(1), NoCoefficients, [5u8], 0u8);
    /// assert_eq!(i.columns(), &[5]);
    /// ```
    pub const fn columns(&self) -> &[C; W] {
        &self.columns
    }

    /// The commitment to the error vector.
    ///
    /// ```
    /// use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    /// use arrabbiata_field::Goldilocks as F;
    /// let i = RelaxedInstance::new(F::new(1), NoCoefficients, [5u8], 6u8);
    /// assert_eq!(i.error(), &6);
    /// ```
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
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_relation::RelaxedWitness;
/// let w = RelaxedWitness::new([vec![F::ONE]], vec![F::ZERO]);
/// assert_eq!(w.error(), &[F::ZERO]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedWitness<F, const W: usize> {
    columns: [Vec<F>; W],
    error: Vec<F>,
}

impl<F, const W: usize> RelaxedWitness<F, W> {
    /// The witness with the given columns and error vector. Shapes are
    /// checked by the relation, not here.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::RelaxedWitness;
    /// let w = RelaxedWitness::new([vec![F::ONE, F::ONE]], vec![F::ZERO; 2]);
    /// assert_eq!(w.columns()[0].len(), 2);
    /// ```
    pub const fn new(columns: [Vec<F>; W], error: Vec<F>) -> Self {
        Self { columns, error }
    }

    /// The witness columns.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::RelaxedWitness;
    /// assert_eq!(RelaxedWitness::new([vec![F::ONE]], vec![]).columns()[0], vec![F::ONE]);
    /// ```
    pub const fn columns(&self) -> &[Vec<F>; W] {
        &self.columns
    }

    /// The error vector.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::RelaxedWitness;
    /// assert!(RelaxedWitness::<F, 1>::new([vec![]], vec![]).error().is_empty());
    /// ```
    pub fn error(&self) -> &[F] {
        &self.error
    }
}
