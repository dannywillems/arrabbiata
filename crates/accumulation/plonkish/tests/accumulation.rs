//! The laws of the accumulation scheme, on a vanilla circuit and on a
//! custom gate set of degree 5 with rotations, a fixed column and a gadget
//! of two constraints.
//!
//! - the embedding, the fold and accumulation are complete reductions: the
//!   verifier's instance is the prover's, and the output is in `R'`
//!   (every `r`, since the sponge domain varies the challenges);
//! - the embedding lands in the chart `u = 1, E = 0`, `c = (1, a, ...)`;
//! - soundness probes: an unsatisfied step, a corrupted accumulator, a
//!   corrupted cross-term, a wrong instance or another circuit's digest
//!   each yield an output outside `R'`;
//! - category laws: `Then(Identity, A)` and `Then(A, Identity)` verify to
//!   the instance of `A`.

mod common;

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::{Field, powers};
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonkish::{GateSet, RelaxedWitness, Witness};
use arrabbiata_plonkish_folding::{Embed, Error, Fold, Params, RelaxedInstance, accumulate};
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_reduction::{Identity, Reduction, Relation, Then, laws};
use common::{Pcs, custom, custom_params, vanilla, vanilla_params};
use proptest::prelude::*;

type Instance<const W: usize> =
    RelaxedInstance<F, <Pcs as HomomorphicCommitment<F>>::Commitment, W>;

/// Embed every witness, then accumulate them one by one; check the laws at
/// each step and return the final accumulator.
fn chain<G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>(
    params: &Params<F, Pcs, G, W, X, Q, N, D>,
    witnesses: &[Witness<F, W>],
    domain: u64,
) -> Result<(Instance<W>, RelaxedWitness<F, W>), TestCaseError>
where
    G: GateSet<F, W, X>,
{
    let sponge = || vesta_sponge(domain);
    let first = &witnesses[0];
    let out = laws::complete::<F, _, _>(
        &Embed(params),
        sponge,
        &first.public_inputs().to_vec(),
        first,
    )
    .map_err(|e| TestCaseError::fail(format!("embed: {e:?}")))?;
    let (mut instance, mut witness) = (out.instance, out.witness);
    for w in &witnesses[1..] {
        let input = (w.public_inputs().to_vec(), instance);
        let out =
            laws::complete::<F, _, _>(&accumulate(params), sponge, &input, &(w.clone(), witness))
                .map_err(|e| TestCaseError::fail(format!("accumulate: {e:?}")))?;
        (instance, witness) = (out.instance, out.witness);
    }
    Ok((instance, witness))
}

/// The chart: `u = 1`, `c` the powers of the combiner, the zero error.
fn in_chart<G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>(
    params: &Params<F, Pcs, G, W, X, Q, N, D>,
    witness: &Witness<F, W>,
) -> Result<(), TestCaseError>
where
    G: GateSet<F, W, X>,
{
    let out = Embed(params)
        .prove(vesta_sponge(7), &witness.public_inputs().to_vec(), witness)
        .unwrap();
    let i = &out.instance;
    prop_assert_eq!(i.u(), F::ONE);
    let n = params.circuit().num_rows();
    prop_assert_eq!(i.error(), &params.commit_column(&vec![F::ZERO; n]).unwrap());
    prop_assert!(out.witness.error().iter().all(|e| e.is_zero()));
    let c = i.coefficients();
    prop_assert_eq!(c.len(), params.combiner_length());
    if c.len() > 1 {
        prop_assert_eq!(c.to_vec(), powers(c[1], c.len()));
    }
    prop_assert_eq!(i.public(), witness.public_inputs());
    Ok(())
}

/// A witness with one cell changed, on a row a gadget reads.
fn break_cell<const W: usize>(witness: &Witness<F, W>, column: usize, row: usize) -> Witness<F, W> {
    let mut columns = witness.columns().clone();
    columns[column][row] = columns[column][row] + F::ONE;
    Witness::new(columns, witness.public_inputs().to_vec())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Accumulating 1 to 4 vanilla steps: every step complete.
    #[test]
    fn vanilla_accumulation_is_complete(
        steps in prop::collection::vec((any::<u64>(), any::<u64>()), 1..5),
        domain in 0u64..1000,
    ) {
        let params = vanilla_params(2);
        let witnesses: Vec<_> = steps.iter().map(|(x, k)| vanilla(*x, *k).1).collect();
        let (i, w) = chain(&params, &witnesses, domain)?;
        prop_assert_eq!(params.relaxed().check(&i, &w), Ok(()));
    }

    /// The same for the custom gate set: degree 5 (five cross-terms),
    /// rotations -1 and +1, a fixed column, two constraints in a gadget.
    #[test]
    fn custom_accumulation_is_complete(
        xs in prop::collection::vec(any::<u64>(), 1..5),
        q in any::<u64>(),
        domain in 0u64..1000,
    ) {
        let params = custom_params(q, 2);
        let witnesses: Vec<_> = xs.iter().map(|x| custom(*x, q).1).collect();
        let (i, w) = chain(&params, &witnesses, domain)?;
        prop_assert_eq!(params.relaxed().check(&i, &w), Ok(()));
    }

    /// Two accumulators fold too (`R' x R' -> R'`, not only `R x R'`).
    #[test]
    fn accumulators_fold(xs in prop::collection::vec(any::<u64>(), 4), q in any::<u64>()) {
        let params = custom_params(q, 2);
        let w: Vec<_> = xs.iter().map(|x| custom(*x, q).1).collect();
        let left = chain(&params, &w[..2], 1)?;
        let right = chain(&params, &w[2..], 2)?;
        let out = laws::complete::<F, _, _>(&Fold(&params), || vesta_sponge(3), &(left.0, right.0), &(left.1, right.1));
        prop_assert!(out.is_ok(), "{:?}", out.err());
    }

    /// The embedding lands in the chart `u = 1, E = 0`.
    #[test]
    fn embedding_is_the_chart(x in any::<u64>(), k in any::<u64>(), q in any::<u64>()) {
        in_chart(&vanilla_params(2), &vanilla(x, k).1)?;
        in_chart(&custom_params(q, 2), &custom(x, q).1)?;
    }

    /// An unsatisfied step is outside `R'` once embedded, and stays out
    /// after being accumulated into a valid accumulator.
    #[test]
    fn an_unsatisfied_step_is_caught(x in any::<u64>(), y in any::<u64>(), q in any::<u64>(), column in 0usize..2, row in 0usize..3) {
        let params = custom_params(q, 2);
        let good = custom(x, q).1;
        let bad = break_cell(&custom(y, q).1, column, row);
        prop_assert!(params.satisfiability().check(&bad.public_inputs().to_vec(), &bad).is_err());
        let embedded = Embed(&params).prove(vesta_sponge(3), &bad.public_inputs().to_vec(), &bad).unwrap();
        prop_assert!(params.relaxed().check(&embedded.instance, &embedded.witness).is_err());
        let (acc, acc_w) = chain(&params, &[good], 4)?;
        let input = (bad.public_inputs().to_vec(), acc);
        let out = accumulate(&params).prove(vesta_sponge(5), &input, &(bad, acc_w)).unwrap();
        prop_assert!(params.relaxed().check(&out.instance, &out.witness).is_err());
    }

    /// An accumulator whose error column is off on one row folds into an
    /// instance outside `R'`.
    #[test]
    fn a_corrupted_accumulator_is_caught(xs in prop::collection::vec(any::<u64>(), 3), k in any::<u64>(), row in 0usize..8) {
        let params = vanilla_params(2);
        let w: Vec<_> = xs.iter().map(|x| vanilla(*x, k).1).collect();
        let (acc, acc_w) = chain(&params, &w[..2], 1)?;
        let n = params.circuit().num_rows();
        let mut error = acc_w.error().to_vec();
        error[row % n] = error[row % n] + F::ONE;
        let corrupted = RelaxedWitness::new(acc_w.columns().clone(), error);
        // Recommit the error, so the corrupted pair is self-consistent.
        let acc = RelaxedInstance::new(acc.u(), acc.coefficients().to_vec(), acc.public().to_vec(), *acc.columns(), params.commit_column(corrupted.error()).unwrap());
        prop_assert_eq!(params.relaxed().check(&acc, &corrupted), Err(Error::Unsatisfied));
        let input = (w[2].public_inputs().to_vec(), acc);
        let out = accumulate(&params).prove(vesta_sponge(5), &input, &(w[2].clone(), corrupted)).unwrap();
        prop_assert!(params.relaxed().check(&out.instance, &out.witness).is_err());
    }

    /// A cross-term commitment changed in the proof: the verifier's folded
    /// error commitment no longer opens to the prover's folded error.
    #[test]
    fn a_corrupted_cross_term_is_caught(xs in prop::collection::vec(any::<u64>(), 2), q in any::<u64>(), k in 0usize..5) {
        let params = custom_params(q, 2);
        let e: Vec<_> = xs.iter().map(|x| {
            let w = custom(*x, q).1;
            Embed(&params).prove(vesta_sponge(3), &w.public_inputs().to_vec(), &w).unwrap()
        }).collect();
        let inputs = (e[0].instance.clone(), e[1].instance.clone());
        let witnesses = (e[0].witness.clone(), e[1].witness.clone());
        let out = Fold(&params).prove(vesta_sponge(4), &inputs, &witnesses).unwrap();
        let mut proof = out.proof.clone();
        proof.head.0[k] = params.ck().add(&proof.head.0[k], &proof.head.0[(k + 1) % 5]);
        let (verified, _) = Fold(&params).verify(vesta_sponge(4), &inputs, proof).unwrap();
        prop_assert_ne!(&verified, &out.instance);
        prop_assert!(params.relaxed().check(&verified, &out.witness).is_err());
    }

    /// The verifier folding other instances than the prover's (swapped)
    /// gets an instance the prover's witness does not open.
    #[test]
    fn instances_are_bound(xs in prop::collection::vec(any::<u64>(), 2), k in any::<u64>()) {
        let params = vanilla_params(2);
        let e: Vec<_> = xs.iter().map(|x| {
            let w = vanilla(*x, k).1;
            Embed(&params).prove(vesta_sponge(3), &w.public_inputs().to_vec(), &w).unwrap()
        }).collect();
        prop_assume!(e[0].instance != e[1].instance);
        let inputs = (e[0].instance.clone(), e[1].instance.clone());
        let out = Fold(&params).prove(vesta_sponge(4), &inputs, &(e[0].witness.clone(), e[1].witness.clone())).unwrap();
        let swapped = (inputs.1, inputs.0);
        let (verified, _) = Fold(&params).verify(vesta_sponge(4), &swapped, out.proof).unwrap();
        prop_assert!(params.relaxed().check(&verified, &out.witness).is_err());
    }

    /// The category's unit laws: composing with an identity changes
    /// nothing the verifier outputs.
    #[test]
    fn identity_is_a_unit(x in any::<u64>(), k in any::<u64>()) {
        let params = vanilla_params(2);
        let w = vanilla(x, k).1;
        let public = w.public_inputs().to_vec();
        let embed = Embed(&params);
        let out = embed.prove(vesta_sponge(3), &public, &w).unwrap();
        let left = Then(Identity(params.satisfiability()), Embed(&params));
        let right = Then(Embed(&params), Identity(params.relaxed()));
        prop_assert!(laws::same_instance::<F, _, _, _>(&embed, &left, || vesta_sponge(3), &public, out.proof.clone(), ((), out.proof.clone())).unwrap());
        prop_assert!(laws::same_instance::<F, _, _, _>(&embed, &right, || vesta_sponge(3), &public, out.proof.clone(), (out.proof, ())).unwrap());
    }
}

/// The transcript is bound to the circuit through the digest: the same
/// fold under the parameters of another circuit draws another `r`, so its
/// folded instance differs.
#[test]
fn the_digest_binds_the_circuit() {
    let (a, b) = (custom_params(3, 2), custom_params(4, 2));
    assert_ne!(a.digest(), b.digest());
    let w: Vec<_> = [5, 6].iter().map(|x| custom(*x, 3).1).collect();
    let e: Vec<_> = w
        .iter()
        .map(|w| {
            Embed(&a)
                .prove(vesta_sponge(3), &w.public_inputs().to_vec(), w)
                .unwrap()
        })
        .collect();
    let inputs = (e[0].instance.clone(), e[1].instance.clone());
    let out = Fold(&a)
        .prove(
            vesta_sponge(4),
            &inputs,
            &(e[0].witness.clone(), e[1].witness.clone()),
        )
        .unwrap();
    let (other, _) = Fold(&b)
        .verify(vesta_sponge(4), &inputs, out.proof)
        .unwrap();
    assert_ne!(other, out.instance);
}

/// Shapes the prover and the verifier refuse.
#[test]
fn shapes_are_checked() {
    let params = vanilla_params(2);
    let w = vanilla(2, 3).1;
    let public = w.public_inputs().to_vec();
    assert_eq!(
        Embed(&params)
            .prove(vesta_sponge(0), &vec![], &w)
            .unwrap_err(),
        Error::Shape
    );
    let out = Embed(&params).prove(vesta_sponge(0), &public, &w).unwrap();
    let wide = RelaxedInstance::new(
        out.instance.u(),
        vec![F::ONE; 2],
        out.instance.public().to_vec(),
        *out.instance.columns(),
        *out.instance.error(),
    );
    assert_eq!(
        params.relaxed().check(&wide, &out.witness),
        Err(Error::Shape)
    );
    let inputs = (wide, out.instance.clone());
    let witnesses = (out.witness.clone(), out.witness.clone());
    assert_eq!(
        Fold(&params)
            .prove(vesta_sponge(0), &inputs, &witnesses)
            .unwrap_err(),
        Error::Shape
    );
    let short = RelaxedWitness::new(out.witness.columns().clone(), vec![F::ZERO]);
    let inputs = (out.instance.clone(), out.instance.clone());
    assert_eq!(
        Fold(&params)
            .prove(vesta_sponge(0), &inputs, &(short, out.witness))
            .unwrap_err(),
        Error::Shape
    );
}

/// Column commitments in the wrong order do not open the witness.
#[test]
fn commitments_are_checked() {
    let params = vanilla_params(2);
    let w = vanilla(2, 3).1;
    let out = Embed(&params)
        .prove(vesta_sponge(0), &w.public_inputs().to_vec(), &w)
        .unwrap();
    let mut columns = *out.instance.columns();
    columns.swap(0, 1);
    let swapped = RelaxedInstance::new(
        out.instance.u(),
        out.instance.coefficients().to_vec(),
        out.instance.public().to_vec(),
        columns,
        *out.instance.error(),
    );
    assert_eq!(
        params.relaxed().check(&swapped, &out.witness),
        Err(Error::Commitment(0))
    );
}
