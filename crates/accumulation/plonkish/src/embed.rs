//! The embedding `R -> R'`: a fresh instance in the chart `u = 1, E = 0`.

use alloc::{vec, vec::Vec};

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::{TwoAdicField, powers};
use arrabbiata_plonkish::{RelaxedWitness, Witness, gate::GateSet};
use arrabbiata_reduction::{Output, OutputOf, Reduction, Relation};
use arrabbiata_transcript::{
    Transcript,
    typed::{Challenge, End, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape},
};

use crate::{
    Columns, Digest, Error, Params, PublicInputs, Relaxed, RelaxedInstance, Satisfiability,
};

/// The embedding of the circuit relation into its relaxation, as a
/// reduction of knowledge `R -> R'`.
///
/// The prover commits to the witness columns; the verifier draws the
/// combiner `a` after them; the output instance is
/// `(u = 1, c = (1, a, ..., a^(m-1)), x, [w], [0])`, with `m` the
/// [combiner length](Params::combiner_length).
///
/// Completeness is the chart `u = 1, E = 0` of the relaxation, which holds
/// for every `c`. Knowledge soundness: the commitments are fixed before
/// `a`, so a relaxed witness for the output, with `u = 1` and `E = 0`, has
/// on each row `sum_k a^k f_k(w) = 0` for a random `a`, hence every
/// `f_k(w) = 0` except with probability `(m - 1) / |F|` per row.
///
/// Complete runs: `tests/examples.rs` (an incremental computation, two
/// accumulators folded); laws: `tests/accumulation.rs`.
pub struct Embed<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(pub &'a Params<F, PC, G, W, X, Q, N, D>)
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>;

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Protocol for Embed<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    type Transcript = Public<
        Digest<F>,
        Public<PublicInputs<F>, Message<Columns<PC::Commitment, W>, Challenge<End>>>,
    >;
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Embed<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    /// The output instance, from the public inputs, the column commitments
    /// and the combiner. Shared by prover and verifier.
    fn instance(
        &self,
        public: &[F],
        columns: [PC::Commitment; W],
        combiner: F,
    ) -> Result<RelaxedInstance<F, PC::Commitment, W>, Error> {
        let p = self.0;
        let error = match columns.first() {
            Some(c) => p.zero_like(c),
            None => p.commit_column(&vec![F::ZERO; p.circuit().num_rows()])?,
        };
        Ok(RelaxedInstance::new(
            F::ONE,
            powers(combiner, p.combiner_length()),
            public.to_vec(),
            columns,
            error,
        ))
    }
}

impl<'a, F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Reduction<F> for Embed<'a, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    type Source = Satisfiability<'a, F, PC, G, W, X, Q, N, D>;
    type Target = Relaxed<'a, F, PC, G, W, X, Q, N, D>;
    type Proof = ProofOf<Self>;
    type Error = Error;

    fn source(&self) -> Self::Source {
        self.0.satisfiability()
    }

    fn target(&self) -> Self::Target {
        self.0.relaxed()
    }

    fn prove<T>(
        &self,
        transcript: T,
        public: &Vec<F>,
        witness: &Witness<F, W>,
    ) -> Result<OutputOf<F, Self, T>, Error>
    where
        T: Transcript<F>,
    {
        let p = self.0;
        if public.len() != p.circuit().public_rows().len() {
            return Err(Error::Shape);
        }
        let mut commitments = Vec::with_capacity(W);
        for column in witness.columns() {
            commitments.push(p.commit_column(column)?);
        }
        let columns: [PC::Commitment; W] = match commitments.try_into() {
            Ok(c) => c,
            Err(_) => unreachable!("one commitment per column"),
        };
        let (combiner, tape) = prover_tape::<F, T, Self>(transcript)
            .public(p.digest())
            .public(&PublicInputs(public.clone()))
            .send(Columns(columns.clone()))
            .challenge();
        let (transcript, proof) = tape.finish();
        Ok(Output {
            instance: self.instance(public, columns, combiner)?,
            witness: RelaxedWitness::embed(witness),
            proof,
            transcript,
        })
    }

    fn verify<T>(
        &self,
        transcript: T,
        public: &Vec<F>,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, T), Error>
    where
        T: Transcript<F>,
    {
        let p = self.0;
        if public.len() != p.circuit().public_rows().len() {
            return Err(Error::Shape);
        }
        let (columns, tape) = verifier_tape::<F, T, Self>(transcript, proof)
            .public(p.digest())
            .public(&PublicInputs(public.clone()))
            .read();
        let (combiner, tape) = tape.challenge();
        let transcript = tape.finish();
        Ok((self.instance(public, columns.0, combiner)?, transcript))
    }
}
