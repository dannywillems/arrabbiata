//! Complete runs of the accumulation scheme. The rustdoc of the crate,
//! `Embed`, `Fold` and `accumulate` points here.
//!
//! - an incremental computation: steps proving `y = x^2` for successive
//!   `x`, each embedded and accumulated into one running accumulator, which
//!   stays in the relaxed relation;
//! - two accumulators built separately, folded into one.

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::build;
use arrabbiata_ipa::Ipa;
use arrabbiata_pasta::{Fq as F, Pallas};
use arrabbiata_plonkish::{
    Circuit, Witness,
    vanilla::{self, Vanilla},
};
use arrabbiata_plonkish_folding::{Embed, Fold, Params, accumulate};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use arrabbiata_reduction::{Reduction, Relation};

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the sponge squeezing the circuit's digest.
const DIGEST_DOMAIN: u64 = 2;
/// The domain tag of the accumulation transcripts: the prover's and the
/// verifier's fold start from the same one.
const ACCUMULATION_DOMAIN: u64 = 3;

/// One step: `x^2 = y`, with `y` public.
fn step(x: u64) -> (Circuit<F, Vanilla, 3, 5, 1>, Witness<F, 3>) {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let input = b.witness(F::from_u64(x));
        let square = b.mul(&input, &input);
        let claimed = b.public_input(F::from_u64(x * x));
        b.assert_equal(&square, &claimed);
    });
    (vanilla::circuit(&c).unwrap(), vanilla::witness(&w))
}

/// The scheme for the step circuit. Vanilla reads its 3 witness columns on
/// the current row (3 variables) and has degree 2.
fn params() -> Params<F, Ipa<Pallas, PoseidonSponge<F>>, Vanilla, 3, 5, 1, 3, 2> {
    let (circuit, _) = step(0);
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = circuit.num_rows();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    let ck = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    let digest = vesta_sponge(DIGEST_DOMAIN);
    Params::new(circuit, ck, digest).unwrap()
}

#[test]
fn an_incremental_computation() {
    let params = params();
    let (_, first) = step(3);
    let public = first.public_inputs().to_vec();
    let fresh = Embed(&params)
        .prove_on(vesta_sponge(ACCUMULATION_DOMAIN), &public, &first)
        .unwrap();
    let (mut instance, mut witness) = (fresh.instance, fresh.witness);
    for x in [4, 5, 6] {
        let (_, next) = step(x);
        let input = (next.public_inputs().to_vec(), instance);
        let out = accumulate(&params)
            .prove_on(vesta_sponge(ACCUMULATION_DOMAIN), &input, &(next, witness))
            .unwrap();
        // The verifier folds from the instances and the proof alone.
        let (verified, _) = accumulate(&params)
            .verify_on(vesta_sponge(ACCUMULATION_DOMAIN), &input, out.proof)
            .unwrap();
        assert_eq!(verified, out.instance);
        (instance, witness) = (verified, out.witness);
    }
    // The accumulator is relaxed (u is no longer 1), and in R'.
    assert_ne!(instance.u(), F::ONE);
    params.relaxed().check(&instance, &witness).unwrap();
}

#[test]
fn two_accumulators_fold_into_one() {
    let params = params();
    let accumulator = |xs: [u64; 2]| {
        let (_, first) = step(xs[0]);
        let fresh = Embed(&params)
            .prove_on(
                vesta_sponge(ACCUMULATION_DOMAIN),
                &first.public_inputs().to_vec(),
                &first,
            )
            .unwrap();
        let (_, second) = step(xs[1]);
        let input = (second.public_inputs().to_vec(), fresh.instance);
        let out = accumulate(&params)
            .prove_on(
                vesta_sponge(ACCUMULATION_DOMAIN),
                &input,
                &(second, fresh.witness),
            )
            .unwrap();
        (out.instance, out.witness)
    };
    let left = accumulator([2, 3]);
    let right = accumulator([7, 11]);
    let instances = (left.0, right.0);
    let witnesses = (left.1, right.1);
    let out = Fold(&params)
        .prove_on(vesta_sponge(ACCUMULATION_DOMAIN), &instances, &witnesses)
        .unwrap();
    let (folded, _) = Fold(&params)
        .verify_on(vesta_sponge(ACCUMULATION_DOMAIN), &instances, out.proof)
        .unwrap();
    assert_eq!(folded, out.instance);
    params.relaxed().check(&folded, &out.witness).unwrap();
}
