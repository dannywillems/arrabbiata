//! The decider: completeness on accumulators, soundness probes, the
//! commitment law linking it to the accumulator, and the SNARK obtained by
//! composing it with the embedding.

#[path = "../../../accumulation/plonkish/tests/common/mod.rs"]
mod common;

use arrabbiata_bn254::Fr;
use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment};
use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_frontend::vanilla::build;
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonk::{PlonkError, QuotientSizing, ipa, kzg};
use arrabbiata_plonk_decider::{Decider, Shared, snark};
use arrabbiata_plonkish::{
    GateSet, RelaxedWitness, Witness,
    vanilla::{self, Vanilla},
};
use arrabbiata_plonkish_folding::{Embed, Params, RelaxedInstance, accumulate};
use arrabbiata_poseidon::{bn254_scalar_sponge, vesta_sponge};
use arrabbiata_reduction::{Reduction, Relation, Then, laws};
use arrabbiata_transcript::Transcript;
use common::{Gates, custom, vanilla as vanilla_step};
use proptest::prelude::*;

/// Accumulate the witnesses with the scheme of `params`.
#[allow(clippy::type_complexity)]
fn accumulated<
    Fd,
    PC,
    G,
    T,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(
    params: &Params<Fd, PC, G, W, X, Q, N, D>,
    witnesses: &[Witness<Fd, W>],
    sponge: impl Fn() -> T,
) -> (
    RelaxedInstance<Fd, <PC as HomomorphicCommitment<Fd>>::Commitment, W>,
    RelaxedWitness<Fd, W>,
)
where
    Fd: TwoAdicField,
    PC: HomomorphicCommitment<Fd>,
    G: GateSet<Fd, W, X>,
    T: Transcript<Fd>,
{
    let first = &witnesses[0];
    let out = Embed(params)
        .prove(sponge(), &first.public_inputs().to_vec(), first)
        .unwrap();
    let (mut instance, mut witness) = (out.instance, out.witness);
    for w in &witnesses[1..] {
        let input = (w.public_inputs().to_vec(), instance);
        let out = accumulate(params)
            .prove(sponge(), &input, &(w.clone(), witness))
            .unwrap();
        (instance, witness) = (out.instance, out.witness);
    }
    (instance, witness)
}

/// The decider is complete on the accumulator, and its proof does not
/// verify for any other instance in `others`.
fn decides<
    Fd,
    PC,
    G,
    T,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(
    decider: &Decider<'_, Fd, PC, G, W, X, Q, N, D>,
    instance: &RelaxedInstance<Fd, <PC as HomomorphicCommitment<Fd>>::Commitment, W>,
    witness: &RelaxedWitness<Fd, W>,
    others: &[RelaxedInstance<Fd, <PC as HomomorphicCommitment<Fd>>::Commitment, W>],
    sponge: impl Fn() -> T,
) -> Result<(), TestCaseError>
where
    Fd: TwoAdicField,
    PC: Shared<Fd>,
    G: GateSet<Fd, W, X>,
    T: Transcript<Fd>,
{
    let out = laws::complete::<Fd, _, _>(decider, &sponge, instance, witness)
        .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
    for other in others {
        prop_assert!(decider.verify(sponge(), other, out.proof.clone()).is_err());
    }
    Ok(())
}

/// An instance with one component changed: the slack, a coefficient, a
/// public input, a column commitment, the error commitment.
fn perturbations<Fd, C, const W: usize>(
    i: &RelaxedInstance<Fd, C, W>,
    shift: impl Fn(&C) -> C,
) -> Vec<RelaxedInstance<Fd, C, W>>
where
    Fd: Field,
    C: Clone,
{
    let rebuild =
        |u: Fd, c: Vec<Fd>, x: Vec<Fd>, w: [C; W], e: C| RelaxedInstance::new(u, c, x, w, e);
    let mut out = vec![rebuild(
        i.u() + Fd::ONE,
        i.coefficients().to_vec(),
        i.public().to_vec(),
        i.columns().clone(),
        i.error().clone(),
    )];
    if !i.coefficients().is_empty() {
        let mut c = i.coefficients().to_vec();
        c[0] = c[0] + Fd::ONE;
        out.push(rebuild(
            i.u(),
            c,
            i.public().to_vec(),
            i.columns().clone(),
            i.error().clone(),
        ));
    }
    if !i.public().is_empty() {
        let mut x = i.public().to_vec();
        x[0] = x[0] + Fd::ONE;
        out.push(rebuild(
            i.u(),
            i.coefficients().to_vec(),
            x,
            i.columns().clone(),
            i.error().clone(),
        ));
    }
    if W > 0 {
        let mut w = i.columns().clone();
        w[0] = shift(&w[0]);
        out.push(rebuild(
            i.u(),
            i.coefficients().to_vec(),
            i.public().to_vec(),
            w,
            i.error().clone(),
        ));
    }
    out.push(rebuild(
        i.u(),
        i.coefficients().to_vec(),
        i.public().to_vec(),
        i.columns().clone(),
        shift(i.error()),
    ));
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// The two `commit` of IPA agree: the accumulator's commitments are
    /// the decider's.
    #[test]
    fn ipa_commitments_agree(p in prop::collection::vec(any::<u64>(), 0..16)) {
        let (circuit, _) = vanilla_step(1, 1);
        let pcs = ipa::pallas(&circuit);
        let p: Vec<F> = p.into_iter().map(F::from_u64).collect();
        prop_assert_eq!(HomomorphicCommitment::commit(&pcs, &p), PolynomialCommitment::commit(&pcs, &p));
    }

    /// The same for KZG.
    #[test]
    fn kzg_commitments_agree(p in prop::collection::vec(any::<u64>(), 0..16), tau in 2u64..1000) {
        let srs = arrabbiata_kzg::Kzg::<arrabbiata_bn254::Bn254>::insecure_setup(16, Fr::from_u64(tau));
        let p: Vec<Fr> = p.into_iter().map(Fr::from_u64).collect();
        prop_assert_eq!(HomomorphicCommitment::commit(&srs, &p), PolynomialCommitment::commit(&srs, &p));
    }

    /// Vanilla: 1 to 4 steps, decided by one proof, which no perturbed
    /// instance accepts.
    #[test]
    fn vanilla_accumulators_are_decided(steps in prop::collection::vec((any::<u64>(), any::<u64>()), 1..5)) {
        let (circuit, _) = vanilla_step(0, 0);
        let pcs = ipa::pallas(&circuit);
        let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
        let witnesses: Vec<_> = steps.iter().map(|(x, k)| vanilla_step(*x, *k).1).collect();
        let (instance, witness) = accumulated(&params, &witnesses, || vesta_sponge(3));
        let decider = Decider::new(&params).unwrap();
        let ck = params.ck();
        let one = HomomorphicCommitment::commit(ck, &[F::ONE]);
        let others = perturbations(&instance, |c| ck.add(c, &one));
        decides(&decider, &instance, &witness, &others, || vesta_sponge(4))?;
    }

    /// The custom gate set: degree 5, rotations, a fixed column, two
    /// constraints in a gadget.
    #[test]
    fn custom_accumulators_are_decided(xs in prop::collection::vec(any::<u64>(), 1..4), q in any::<u64>()) {
        let (circuit, _) = custom(0, q);
        let pcs = ipa::pallas(&circuit);
        let params = Params::<F, _, Gates, 2, 1, 3, 6, 5>::new(circuit, pcs, vesta_sponge(2)).unwrap();
        let witnesses: Vec<_> = xs.iter().map(|x| custom(*x, q).1).collect();
        let (instance, witness) = accumulated(&params, &witnesses, || vesta_sponge(3));
        let decider = Decider::new(&params).unwrap();
        let ck = params.ck();
        let one = HomomorphicCommitment::commit(ck, &[F::ONE]);
        let others = perturbations(&instance, |c| ck.add(c, &one));
        decides(&decider, &instance, &witness, &others, || vesta_sponge(4))?;
    }

    /// The decider composed with accumulation, `Dec . Fold . (Embed x Id)`:
    /// one reduction from a fresh step and an accumulator to `R_top`.
    #[test]
    fn accumulate_then_decide(xs in prop::collection::vec(any::<u64>(), 2..4), k in any::<u64>()) {
        let (circuit, _) = vanilla_step(0, 0);
        let pcs = ipa::pallas(&circuit);
        let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
        let witnesses: Vec<_> = xs.iter().map(|x| vanilla_step(*x, k).1).collect();
        let (instance, witness) = accumulated(&params, &witnesses[..witnesses.len() - 1], || vesta_sponge(3));
        let last = witnesses.last().unwrap();
        let decider = Decider::new(&params).unwrap();
        let both = Then(accumulate(&params), &decider);
        let input = (last.public_inputs().to_vec(), instance);
        let out = laws::complete::<F, _, _>(&both, || vesta_sponge(5), &input, &(last.clone(), witness));
        prop_assert!(out.is_ok(), "{:?}", out.err());
    }

    /// The SNARK `Dec . Embed` is complete and rejects other statements.
    #[test]
    fn the_snark_is_an_argument(x in any::<u64>(), k in any::<u64>()) {
        let (circuit, w) = vanilla_step(x, k);
        let pcs = ipa::pallas(&circuit);
        let params = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
        let decider = Decider::new(&params).unwrap();
        let s = snark(&decider);
        let public = w.public_inputs().to_vec();
        let out = laws::complete::<F, _, _>(&s, || vesta_sponge(6), &public, &w);
        prop_assert!(out.is_ok(), "{:?}", out.err());
        let proof = out.unwrap().proof;
        let wrong = vec![public[0] + F::ONE];
        prop_assert!(s.verify(vesta_sponge(6), &wrong, proof).is_err());
    }
}

/// KZG on BN254: the same scheme and decider, over another field and
/// another commitment.
#[test]
fn kzg_accumulators_are_decided() {
    let step = |v: u64| {
        let (c, w, ()) = build::<Fr, _, _>(|b| {
            let x = b.witness(Fr::from_u64(v));
            let y = b.mul(&x, &x);
            let z = b.mul(&y, &x);
            let out = b.public_input(Fr::from_u64(v * v * v));
            b.assert_equal(&z, &out);
        });
        (vanilla::circuit(&c).unwrap(), vanilla::witness(&w))
    };
    let (circuit, _) = step(0);
    let srs = kzg::insecure_srs(&circuit, Fr::from_u64(123_456_789));
    assert!(srs.size() >= circuit.max_coefficients());
    let params =
        Params::<Fr, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, srs, bn254_scalar_sponge(2)).unwrap();
    let witnesses: Vec<_> = [2, 3, 4].iter().map(|v| step(*v).1).collect();
    let (instance, witness) = accumulated(&params, &witnesses, || bn254_scalar_sponge(3));
    let decider = Decider::new(&params).unwrap();
    let out = decider
        .prove(bn254_scalar_sponge(4), &instance, &witness)
        .unwrap();
    decider
        .verify(bn254_scalar_sponge(4), &instance, out.proof)
        .unwrap();
}

/// The honest prover refuses a witness outside `R'`, or one that does not
/// open the instance.
#[test]
fn the_prover_refuses_an_unsatisfied_accumulator() {
    let (circuit, _) = vanilla_step(0, 0);
    let pcs = ipa::pallas(&circuit);
    let params =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
    let witnesses: Vec<_> = [(2, 3), (4, 5)]
        .iter()
        .map(|(x, k)| vanilla_step(*x, *k).1)
        .collect();
    let (instance, witness) = accumulated(&params, &witnesses, || vesta_sponge(3));
    let decider = Decider::new(&params).unwrap();
    // An error column off on one row, recommitted so it opens.
    let mut error = witness.error().to_vec();
    error[0] = error[0] + F::ONE;
    let bad = RelaxedWitness::new(witness.columns().clone(), error);
    let bad_instance = RelaxedInstance::new(
        instance.u(),
        instance.coefficients().to_vec(),
        instance.public().to_vec(),
        *instance.columns(),
        params.commit_column(bad.error()).unwrap(),
    );
    assert!(params.relaxed().check(&bad_instance, &bad).is_err());
    assert_eq!(
        decider
            .prove(vesta_sponge(4), &bad_instance, &bad)
            .unwrap_err(),
        PlonkError::Unsatisfied
    );
    // A satisfied witness against an instance it does not open.
    assert_eq!(
        decider
            .prove(vesta_sponge(4), &bad_instance, &witness)
            .unwrap_err(),
        PlonkError::Unsatisfied
    );
}

/// A proof with a changed evaluation of the error polynomial fails, at
/// its opening.
#[test]
fn the_error_opening_is_checked() {
    let (circuit, _) = vanilla_step(0, 0);
    let pcs = ipa::pallas(&circuit);
    let params =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, pcs, vesta_sponge(2)).unwrap();
    let witnesses: Vec<_> = [(2, 3), (4, 5)]
        .iter()
        .map(|(x, k)| vanilla_step(*x, *k).1)
        .collect();
    let (instance, witness) = accumulated(&params, &witnesses, || vesta_sponge(3));
    let decider = Decider::new(&params).unwrap();
    let out = decider.prove(vesta_sponge(4), &instance, &witness).unwrap();
    let mut proof = out.proof;
    proof.tail.tail.head.error = proof.tail.tail.head.error + F::ONE;
    let result = decider.verify(vesta_sponge(4), &instance, proof);
    assert!(matches!(result, Err(PlonkError::Opening(_))), "{result:?}");
}
