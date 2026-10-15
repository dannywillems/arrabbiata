//! The fold `R' x R' -> R'`, and accumulation `R x R' -> R'`.

use alloc::{vec, vec::Vec};

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::{Field, TwoAdicField, powers};
use arrabbiata_mvpoly::combined_cross_terms_with;
use arrabbiata_plonkish::{RelaxedWitness, gate::GateSet};
use arrabbiata_reduction::{
    Identity, Output, OutputOf, Parallel, Product, Reduction, Relation, Then,
};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, tag,
    typed::{
        Challenge, End, Label, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape,
    },
};

use crate::{Digest, Embed, Error, Params, Relaxed, RelaxedInstance};

/// The commitments to the `K` cross-terms of a fold, `T_1` first.
///
/// ```
/// use arrabbiata_plonkish_folding::CrossTerms;
/// let t = CrossTerms([1u8, 2]);
/// assert_eq!(t.0[0], 1);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossTerms<C, const K: usize>(pub [C; K]);

impl<C, const K: usize> Label for CrossTerms<C, K> {
    const LABEL: u64 = tag::FOLDING_PROOF;
}

impl<F, C, const K: usize> Absorb<F> for CrossTerms<C, K>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::FOLDING_PROOF));
        absorb_length(transcript, K);
        for c in &self.0 {
            c.absorb_into(transcript);
        }
    }
}

/// The fold of two relaxed instances, as a reduction of knowledge
/// `R' x R' -> R'` (a 2-folding scheme, Kothapalli and Parno,
/// Definition 4).
///
/// The prover commits to the `D` cross-terms; the verifier draws `r`
/// after them, having absorbed the digest of the circuit and both
/// instances; both fold every component linearly (see the
/// [crate documentation](crate)). The verifier does no polynomial
/// evaluation: `W + D + 2` scalar multiplications of commitments.
///
/// Completeness is the expansion of the relaxed condition on the line,
/// which is exact for every `r`. Knowledge soundness: the folded error is
/// fixed by the cross-terms before `r`, so a relaxed witness for the
/// output at `D + 2` distinct values of `r` interpolates witnesses for
/// both inputs (the scheme is `(D + 2)`-special-sound).
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_plonkish::{Builder, vanilla::{Arithmetic, Vanilla}};
/// use arrabbiata_plonkish_folding::{Embed, Fold, Params};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// use arrabbiata_reduction::{Reduction, Relation};
/// // a * b = c on one row.
/// let witness = |a: u64, b: u64| {
///     let mut builder = Builder::<F, Vanilla, 3, 5, 1>::new();
///     let one = F::ONE;
///     builder.row::<Arithmetic, _>([F::from_u64(a), F::from_u64(b), F::from_u64(a * b)], [F::ZERO, F::ZERO, -one, one, F::ZERO]);
///     builder.build().unwrap()
/// };
/// let (circuit, w1) = witness(2, 3);
/// let (_, w2) = witness(4, 5);
/// let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
/// let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
/// let a = Embed(&p).prove(vesta_sponge(3), &vec![], &w1).unwrap();
/// let b = Embed(&p).prove(vesta_sponge(3), &vec![], &w2).unwrap();
/// let inputs = (a.instance, b.instance);
/// let out = Fold(&p).prove(vesta_sponge(4), &inputs, &(a.witness, b.witness)).unwrap();
/// let (folded, _) = Fold(&p).verify(vesta_sponge(4), &inputs, out.proof).unwrap();
/// assert_eq!(folded, out.instance);
/// p.relaxed().check(&folded, &out.witness).unwrap();
/// // The fold of two fresh instances has a nonzero error.
/// assert!(out.witness.error().iter().any(|e| !e.is_zero()));
/// ```
pub struct Fold<
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
    Protocol for Fold<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    type Transcript = Public<
        Digest<F>,
        Public<
            RelaxedInstance<F, PC::Commitment, W>,
            Public<
                RelaxedInstance<F, PC::Commitment, W>,
                Message<CrossTerms<PC::Commitment, D>, Challenge<End>>,
            >,
        >,
    >;
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Fold<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    /// The shape of an instance: the number of coefficients and of public
    /// inputs of the circuit.
    fn check_instance(
        &self,
        instance: &RelaxedInstance<F, PC::Commitment, W>,
    ) -> Result<(), Error> {
        let p = self.0;
        if instance.coefficients().len() == p.combiner_length()
            && instance.public().len() == p.circuit().public_rows().len()
        {
            Ok(())
        } else {
            Err(Error::Shape)
        }
    }

    /// The shape of a witness: `n` values in every column.
    fn check_witness(&self, witness: &RelaxedWitness<F, W>) -> Result<(), Error> {
        let n = self.0.circuit().num_rows();
        if witness.error().len() == n && witness.columns().iter().all(|c| c.len() == n) {
            Ok(())
        } else {
            Err(Error::Shape)
        }
    }

    /// The cross-terms `T_1, ..., T_D` of the fold, one value per row each:
    /// on a row turning on gadget `g`, the coefficients of `r^1` to `r^D` of
    /// `sum_k (c1_k + r c2_k) f^D_(g,k)(u1 + r u2, w1 + r w2)`.
    fn cross_terms(
        &self,
        acc: (
            &RelaxedInstance<F, PC::Commitment, W>,
            &RelaxedWitness<F, W>,
        ),
        fresh: (
            &RelaxedInstance<F, PC::Commitment, W>,
            &RelaxedWitness<F, W>,
        ),
    ) -> Result<Vec<Vec<F>>, Error> {
        let p = self.0;
        let n = p.circuit().num_rows();
        let mut terms = vec![vec![F::ZERO; n]; D];
        for (row, gadget) in p.circuit().gadgets().iter().enumerate() {
            let Some(g) = gadget else { continue };
            let polys = p.polynomials(g.index(), row)?;
            let x1 = p.point(acc.1.columns(), row);
            let x2 = p.point(fresh.1.columns(), row);
            let t = combined_cross_terms_with(
                &polys,
                &x1,
                &x2,
                acc.0.u(),
                fresh.0.u(),
                acc.0.coefficients(),
                fresh.0.coefficients(),
            );
            for (k, term) in terms.iter_mut().enumerate() {
                term[row] = t[k + 1];
            }
        }
        Ok(terms)
    }

    /// The folded instance. Shared by prover and verifier, so the prover's
    /// instance is the verifier's by construction.
    fn fold_instances(
        &self,
        acc: &RelaxedInstance<F, PC::Commitment, W>,
        fresh: &RelaxedInstance<F, PC::Commitment, W>,
        cross_terms: &CrossTerms<PC::Commitment, D>,
        r: F,
    ) -> RelaxedInstance<F, PC::Commitment, W> {
        let ck = self.0.ck();
        let line =
            |a: &[F], b: &[F]| -> Vec<F> { a.iter().zip(b).map(|(x, y)| *x + r * *y).collect() };
        let columns =
            core::array::from_fn(|k| ck.add(&acc.columns()[k], &ck.scale(&fresh.columns()[k], r)));
        let r_powers = powers(r, D + 2);
        let mut error = acc.error().clone();
        for (k, t) in cross_terms.0.iter().enumerate() {
            error = ck.add(&error, &ck.scale(t, r_powers[k + 1]));
        }
        error = ck.add(&error, &ck.scale(fresh.error(), r_powers[D + 1]));
        RelaxedInstance::new(
            acc.u() + r * fresh.u(),
            line(acc.coefficients(), fresh.coefficients()),
            line(acc.public(), fresh.public()),
            columns,
            error,
        )
    }
}

/// `w = w1 + r w2` and `E = E1 + sum_k r^k T_k + r^(D+1) E2`.
fn fold_witness<F, const W: usize, const D: usize>(
    acc: &RelaxedWitness<F, W>,
    fresh: &RelaxedWitness<F, W>,
    cross_terms: &[Vec<F>],
    r: F,
) -> RelaxedWitness<F, W>
where
    F: TwoAdicField,
{
    let columns = core::array::from_fn(|k| {
        acc.columns()[k]
            .iter()
            .zip(&fresh.columns()[k])
            .map(|(a, b)| *a + r * *b)
            .collect()
    });
    let r_powers = powers(r, D + 2);
    let error = (0..acc.error().len())
        .map(|row| {
            let mut e = acc.error()[row];
            for (k, t) in cross_terms.iter().enumerate() {
                e = e + r_powers[k + 1] * t[row];
            }
            e + r_powers[D + 1] * fresh.error()[row]
        })
        .collect();
    RelaxedWitness::new(columns, error)
}

impl<'a, F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Reduction<F> for Fold<'a, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    type Source =
        Product<Relaxed<'a, F, PC, G, W, X, Q, N, D>, Relaxed<'a, F, PC, G, W, X, Q, N, D>>;
    type Target = Relaxed<'a, F, PC, G, W, X, Q, N, D>;
    type Proof = ProofOf<Self>;
    type Error = Error;

    fn source(&self) -> Self::Source {
        Product(self.0.relaxed(), self.0.relaxed())
    }

    fn target(&self) -> Self::Target {
        self.0.relaxed()
    }

    fn prove<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &<Self::Source as Relation>::Witness,
    ) -> Result<OutputOf<F, Self, T>, Error>
    where
        T: Transcript<F>,
    {
        let (acc, fresh) = instance;
        self.check_instance(acc)?;
        self.check_instance(fresh)?;
        self.check_witness(&witness.0)?;
        self.check_witness(&witness.1)?;
        let terms = self.cross_terms((acc, &witness.0), (fresh, &witness.1))?;
        let mut commitments = Vec::with_capacity(D);
        for t in &terms {
            commitments.push(self.0.commit_column(t)?);
        }
        let cross_terms = CrossTerms(match commitments.try_into() {
            Ok(c) => c,
            Err(_) => unreachable!("D cross-terms"),
        });
        let (r, tape) = prover_tape::<F, T, Self>(transcript)
            .public(self.0.digest())
            .public(acc)
            .public(fresh)
            .send(cross_terms.clone())
            .challenge();
        let (transcript, proof) = tape.finish();
        Ok(Output {
            instance: self.fold_instances(acc, fresh, &cross_terms, r),
            witness: fold_witness::<F, W, D>(&witness.0, &witness.1, &terms, r),
            proof,
            transcript,
        })
    }

    fn verify<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<(<Self::Target as Relation>::Instance, T), Error>
    where
        T: Transcript<F>,
    {
        let (acc, fresh) = instance;
        self.check_instance(acc)?;
        self.check_instance(fresh)?;
        let (cross_terms, tape) = verifier_tape::<F, T, Self>(transcript, proof)
            .public(self.0.digest())
            .public(acc)
            .public(fresh)
            .read();
        let (r, tape) = tape.challenge();
        let transcript = tape.finish();
        Ok((self.fold_instances(acc, fresh, &cross_terms, r), transcript))
    }
}

/// Accumulation of a fresh step into an accumulator, `R x R' -> R'`: the
/// composite `Fold . (Embed x Id)` of the category of reductions, built by
/// [`accumulate`].
pub type Accumulate<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
> = Then<
    Parallel<Embed<'a, F, PC, G, W, X, Q, N, D>, Identity<Relaxed<'a, F, PC, G, W, X, Q, N, D>>>,
    Fold<'a, F, PC, G, W, X, Q, N, D>,
>;

/// The accumulation reduction `R x R' -> R'` of the scheme with parameters
/// `params`: embed the fresh step, keep the accumulator, fold.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
/// use arrabbiata_plonkish_folding::{Embed, Params, accumulate};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// use arrabbiata_reduction::{Reduction, Relation};
/// let (circuit, w) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
/// let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
/// let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
/// let acc = Embed(&p).prove(vesta_sponge(3), &vec![], &w).unwrap();
/// let input = (vec![], acc.instance);
/// let out = accumulate(&p).prove(vesta_sponge(3), &input, &(w, acc.witness)).unwrap();
/// let (verified, _) = accumulate(&p).verify(vesta_sponge(3), &input, out.proof).unwrap();
/// assert_eq!(verified, out.instance);
/// p.relaxed().check(&verified, &out.witness).unwrap();
/// ```
pub const fn accumulate<
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(
    params: &Params<F, PC, G, W, X, Q, N, D>,
) -> Accumulate<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    Then(
        Parallel(Embed(params), Identity(params.relaxed())),
        Fold(params),
    )
}
