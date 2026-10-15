//! Instances and messages, with their transcript encodings.

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, absorb_slice, tag, typed::Label};

/// A relaxed instance: the slack `u`, the coefficients `c` combining the
/// constraints of a row, the public inputs `x`, the commitments to the `W`
/// witness columns and the commitment to the error column.
///
/// Every component is folded linearly. A fresh instance (the image of the
/// embedding) has `u = 1`, `c = (1, a, a^2, ...)` and a commitment to the
/// zero column as error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedInstance<F, C, const W: usize> {
    u: F,
    coefficients: Vec<F>,
    public: Vec<F>,
    columns: [C; W],
    error: C,
}

impl<F, C, const W: usize> RelaxedInstance<F, C, W> {
    /// The instance with the given components.
    pub const fn new(
        u: F,
        coefficients: Vec<F>,
        public: Vec<F>,
        columns: [C; W],
        error: C,
    ) -> Self {
        Self {
            u,
            coefficients,
            public,
            columns,
            error,
        }
    }

    /// The coefficients combining the constraints of a row.
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    /// The public inputs.
    pub fn public(&self) -> &[F] {
        &self.public
    }

    /// The commitments to the witness columns.
    pub const fn columns(&self) -> &[C; W] {
        &self.columns
    }

    /// The commitment to the error column.
    pub const fn error(&self) -> &C {
        &self.error
    }
}

impl<F, C, const W: usize> RelaxedInstance<F, C, W>
where
    F: Copy,
{
    /// The slack `u`.
    pub const fn u(&self) -> F {
        self.u
    }
}

impl<F, C, const W: usize> Label for RelaxedInstance<F, C, W> {
    const LABEL: u64 = tag::RELAXED_INSTANCE;
}

impl<F, C, const W: usize> Absorb<F> for RelaxedInstance<F, C, W>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::RELAXED_INSTANCE));
        transcript.absorb(self.u);
        absorb_slice(transcript, &self.coefficients);
        absorb_slice(transcript, &self.public);
        absorb_length(transcript, W);
        for c in &self.columns {
            c.absorb_into(transcript);
        }
        self.error.absorb_into(transcript);
    }
}

/// The digest of a circuit: what the transcripts of the scheme absorb in
/// place of the circuit itself, so the verifiers stay sublinear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest<F>(F);

impl<F> Digest<F>
where
    F: Copy,
{
    /// The digest with the given value.
    pub const fn new(value: F) -> Self {
        Self(value)
    }

    /// The value.
    pub const fn value(&self) -> F {
        self.0
    }
}

impl<F> Label for Digest<F> {
    const LABEL: u64 = 0x6467_7374;
}

impl<F> Absorb<F> for Digest<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(self.0);
    }
}

/// The public inputs of a fresh instance, as a statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInputs<F>(pub Vec<F>);

impl<F> Label for PublicInputs<F> {
    const LABEL: u64 = 0x7075_626c;
}

impl<F> Absorb<F> for PublicInputs<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.0);
    }
}

/// The commitments to the `W` witness columns of a fresh instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Columns<C, const W: usize>(pub [C; W]);

impl<C, const W: usize> Label for Columns<C, W> {
    const LABEL: u64 = 0x636f_6c73;
}

impl<F, C, const W: usize> Absorb<F> for Columns<C, W>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_length(transcript, W);
        for c in &self.0 {
            c.absorb_into(transcript);
        }
    }
}
