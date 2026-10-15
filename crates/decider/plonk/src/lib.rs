//! A PlonK decider for accumulated PlonKish circuits.
//!
//! A *decider* for an accumulation scheme is an argument of knowledge for
//! its relaxed relation `R'`: a reduction `R' -> R_top` in the category of
//! reductions ([`arrabbiata_reduction`]). It depends on `R'` only, not on
//! how an accumulator was reached, which is why it lives apart from the
//! accumulation scheme.
//!
//! [`Decider`] is the one for `arrabbiata_plonkish_folding`: relaxed PlonK
//! ([`arrabbiata_plonk::relaxed`]) on the statement an accumulator is. The
//! accumulator's commitments are the statement's: the scheme commits to a
//! column as the polynomial interpolating it, which is the commitment PlonK
//! opens. The link is the law
//!
//! ```text
//! HomomorphicCommitment::commit(p) = PolynomialCommitment::commit(p)
//! ```
//!
//! for the coefficients `p` of a polynomial, which the commitment schemes
//! of this repository satisfy (one key, one map; see the tests).
//!
//! Composed with the embedding `R -> R'` of the accumulation scheme, the
//! decider is an argument for the circuit relation itself
//! ([`snark`]): the precomposition of the category, `Dec . Embed`.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::ipa;
//! use arrabbiata_plonk_decider::Decider;
//! use arrabbiata_plonkish::vanilla::{self, Vanilla};
//! use arrabbiata_plonkish_folding::{Embed, Params, accumulate};
//! use arrabbiata_poseidon::vesta_sponge;
//! use arrabbiata_reduction::Reduction;
//!
//! let step = |v: u64| {
//!     let (c, w, ()) = build::<F, _, _>(|b| {
//!         let x = b.witness(F::from_u64(v));
//!         let y = b.mul(&x, &x);
//!         let out = b.public_input(F::from_u64(v * v));
//!         b.assert_equal(&y, &out);
//!     });
//!     (vanilla::circuit(&c).unwrap(), vanilla::witness(&w))
//! };
//! let (circuit, w0) = step(3);
//! // One key for the fold (vectors) and the decider (polynomials).
//! let pcs = ipa::pallas(&circuit);
//! let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
//!
//! // Accumulate three steps.
//! let fresh = Embed(&params).prove(vesta_sponge(3), &w0.public_inputs().to_vec(), &w0).unwrap();
//! let (mut instance, mut witness) = (fresh.instance, fresh.witness);
//! for v in [4, 5] {
//!     let (_, w) = step(v);
//!     let out = accumulate(&params).prove(vesta_sponge(3), &(w.public_inputs().to_vec(), instance), &(w, witness)).unwrap();
//!     (instance, witness) = (out.instance, out.witness);
//! }
//!
//! // Decide the accumulator: one PlonK proof for all three steps.
//! let decider = Decider::new(&params).unwrap();
//! let out = decider.prove(vesta_sponge(4), &instance, &witness).unwrap();
//! decider.verify(vesta_sponge(4), &instance, out.proof).unwrap();
//! ```

#![no_std]

extern crate alloc;

use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment, Univariate};
use arrabbiata_field::TwoAdicField;
use arrabbiata_plonk::{
    PlonkError, ProverKey, VerifierKey,
    relaxed::{self, Relaxation, RelaxedPlonk},
    setup,
};
use arrabbiata_plonkish::{RelaxedWitness, gate::GateSet};
use arrabbiata_plonkish_folding::{Embed, Params, Relaxed, RelaxedInstance};
use arrabbiata_reduction::{Output, OutputOf, Reduction, Relation, Then, Top};
use arrabbiata_transcript::{
    Transcript,
    typed::{Label, ProofOf},
};

/// A commitment scheme that commits to vectors homomorphically and to
/// polynomials with evaluation proofs, with one commitment type: what the
/// fold and the decider share. The law that the two `commit` agree on the
/// same coefficients is what links an accumulator to its decider; it is a
/// property of the scheme, tested for IPA and KZG.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::ipa::PallasIpa;
/// use arrabbiata_plonk_decider::Shared;
/// fn shared<PC: Shared<F>>() {}
/// shared::<PallasIpa>();
/// ```
pub trait Shared<F>:
    HomomorphicCommitment<F>
    + PolynomialCommitment<
        F,
        Family = Univariate,
        Commitment = <Self as HomomorphicCommitment<F>>::Commitment,
    > + Label
where
    F: TwoAdicField,
{
}

impl<F, PC> Shared<F> for PC
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>
        + PolynomialCommitment<
            F,
            Family = Univariate,
            Commitment = <PC as HomomorphicCommitment<F>>::Commitment,
        > + Label,
{
}

/// The decider: relaxed PlonK on the circuit of `params`, keyed once.
///
/// As a reduction it goes from the relaxed relation of `params` to
/// [`Top`]: the prover proves the accumulator, the verifier checks the
/// proof against the accumulator's instance and outputs `()`.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{PlonkError, ipa};
/// use arrabbiata_plonk_decider::Decider;
/// use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
/// use arrabbiata_plonkish_folding::{Embed, Params};
/// use arrabbiata_poseidon::vesta_sponge;
/// use arrabbiata_reduction::Reduction;
/// let (circuit, w) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
/// let pcs = ipa::pallas(&circuit);
/// let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
/// let fresh = Embed(&params).prove(vesta_sponge(3), &vec![], &w).unwrap();
/// let decider = Decider::new(&params).unwrap();
/// let out = decider.prove(vesta_sponge(4), &fresh.instance, &fresh.witness).unwrap();
/// // The proof is bound to the instance: here, to its slack.
/// let other = arrabbiata_plonkish_folding::RelaxedInstance::new(
///     F::from_u64(2), fresh.instance.coefficients().to_vec(), vec![],
///     fresh.instance.columns().clone(), fresh.instance.error().clone(),
/// );
/// assert!(decider.verify(vesta_sponge(4), &other, out.proof.clone()).is_err());
/// assert!(decider.verify(vesta_sponge(4), &fresh.instance, out.proof).is_ok());
/// ```
pub struct Decider<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
> where
    F: TwoAdicField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    params: &'a Params<F, PC, G, W, X, Q, N, D>,
    pk: ProverKey<F, PC, G, W, X, Q>,
    vk: VerifierKey<F, PC, G, W, X, Q>,
}

impl<'a, F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Decider<'a, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    /// The decider for the scheme with parameters `params`: PlonK's setup
    /// of the circuit, with the scheme's key.
    ///
    /// # Errors
    ///
    /// The errors of [`arrabbiata_plonk::setup`].
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::ipa;
    /// use arrabbiata_plonk_decider::Decider;
    /// use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    /// use arrabbiata_plonkish_folding::Params;
    /// use arrabbiata_poseidon::vesta_sponge;
    /// let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// let pcs = ipa::pallas(&circuit);
    /// let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
    /// assert!(Decider::new(&params).is_ok());
    /// ```
    pub fn new(params: &'a Params<F, PC, G, W, X, Q, N, D>) -> Result<Self, PlonkError> {
        let (pk, vk) = setup(params.circuit(), params.ck())?;
        Ok(Self { params, pk, vk })
    }

    /// The verifier key of the decider.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::ipa;
    /// use arrabbiata_plonk_decider::Decider;
    /// use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    /// use arrabbiata_plonkish_folding::Params;
    /// use arrabbiata_poseidon::vesta_sponge;
    /// let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// let pcs = ipa::pallas(&circuit);
    /// let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
    /// let decider = Decider::new(&params).unwrap();
    /// assert_eq!(decider.verifier_key().value.num_rows(), 2);
    /// ```
    pub const fn verifier_key(&self) -> &VerifierKey<F, PC, G, W, X, Q> {
        &self.vk
    }

    /// The statement of relaxed PlonK an accumulator is.
    fn statement(
        instance: &RelaxedInstance<F, <PC as HomomorphicCommitment<F>>::Commitment, W>,
    ) -> Relaxation<F, <PC as HomomorphicCommitment<F>>::Commitment, W> {
        Relaxation::new(
            D,
            instance.u(),
            instance.coefficients().to_vec(),
            instance.public().to_vec(),
            instance.columns().clone(),
            instance.error().clone(),
        )
    }
}

impl<'a, F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Reduction<F> for Decider<'a, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    type Source = Relaxed<'a, F, PC, G, W, X, Q, N, D>;
    type Target = Top;
    type Proof = ProofOf<RelaxedPlonk<F, PC, G, W, X, Q>>;
    type Error = PlonkError;

    fn source(&self) -> Self::Source {
        self.params.relaxed()
    }

    fn target(&self) -> Top {
        Top
    }

    fn prove<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        witness: &RelaxedWitness<F, W>,
    ) -> Result<OutputOf<F, Self, T>, PlonkError>
    where
        T: Transcript<F>,
    {
        let (proof, transcript) = relaxed::prove(
            &self.pk,
            self.params.ck(),
            transcript,
            &Self::statement(instance),
            witness,
        )?;
        Ok(Output {
            instance: (),
            witness: (),
            proof,
            transcript,
        })
    }

    fn verify<T>(
        &self,
        transcript: T,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<((), T), PlonkError>
    where
        T: Transcript<F>,
    {
        let transcript = relaxed::verify(
            &self.vk,
            self.params.ck(),
            transcript,
            &Self::statement(instance),
            proof,
        )?;
        Ok(((), transcript))
    }
}

/// A SNARK for the circuit relation: the decider after the embedding,
/// `Dec . Embed : R -> R_top`.
pub type Snark<
    'a,
    'd,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
> = Then<Embed<'a, F, PC, G, W, X, Q, N, D>, &'d Decider<'a, F, PC, G, W, X, Q, N, D>>;

/// The SNARK `Dec . Embed` for the circuit of `decider`: an argument of
/// knowledge for the circuit relation, obtained from the category by
/// composition, with no protocol of its own.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::ipa;
/// use arrabbiata_plonk_decider::{Decider, snark};
/// use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
/// use arrabbiata_plonkish_folding::Params;
/// use arrabbiata_poseidon::vesta_sponge;
/// use arrabbiata_reduction::{Argument, Reduction};
/// let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
/// let r = b.plain_row([F::from_u64(7), F::ZERO, F::ZERO], [F::ZERO; 5]);
/// b.public(r);
/// let (circuit, w) = b.build().unwrap();
/// let pcs = ipa::pallas(&circuit);
/// let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
/// let decider = Decider::new(&params).unwrap();
/// let s = snark(&decider);
/// fn argument<A: Argument<F>>(_: &A) {}
/// argument(&s);
/// let out = s.prove(vesta_sponge(5), &vec![F::from_u64(7)], &w).unwrap();
/// assert!(s.verify(vesta_sponge(5), &vec![F::from_u64(7)], out.proof.clone()).is_ok());
/// assert!(s.verify(vesta_sponge(5), &vec![F::from_u64(8)], out.proof).is_err());
/// ```
pub const fn snark<
    'a,
    'd,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(
    decider: &'d Decider<'a, F, PC, G, W, X, Q, N, D>,
) -> Snark<'a, 'd, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    Then(Embed(decider.params), decider)
}
