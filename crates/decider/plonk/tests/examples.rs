//! Complete runs of the decider. The rustdoc of the crate, `Decider` and
//! `snark` points here.
//!
//! - incremental verification: steps proving `y = x^2` accumulated into one
//!   accumulator, then a single generalized-PlonK proof for all of them;
//! - the SNARK `Dec . Embed` on one step, obtained by composition.

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::build;
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonk::ipa;
use arrabbiata_plonk_decider::{Decider, snark};
use arrabbiata_plonkish::{
    Circuit, Witness,
    vanilla::{self, Vanilla},
};
use arrabbiata_plonkish_folding::{Embed, Params, accumulate};
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_reduction::{Argument, Reduction};

/// The domain tag of the sponge squeezing the circuit's digest.
const DIGEST_DOMAIN: u64 = 2;
/// The domain tag of the accumulation transcripts.
const ACCUMULATION_DOMAIN: u64 = 3;
/// The domain tag of the decider's transcripts.
const DECIDER_DOMAIN: u64 = 4;

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

/// The key serves both the fold (vector commitments) and the decider
/// (polynomial commitments): `ipa::pallas` sizes it for PlonK's quotient.
type Scheme = Params<F, ipa::PallasIpa, Vanilla, 3, 5, 1, 3, 2>;

fn params() -> Scheme {
    let (circuit, _) = step(0);
    let pcs = ipa::pallas(&circuit);
    Params::new(circuit, pcs, vesta_sponge(DIGEST_DOMAIN)).unwrap()
}

/// Vanilla's relaxation: its 8 columns read, the error, `u` and one
/// coefficient are 11 variables; `c q_M a b` has degree 4.
type VanillaDecider<'a> = Decider<'a, F, ipa::PallasIpa, Vanilla, 3, 5, 1, 3, 2, 11, 4>;

#[test]
fn incremental_verification() {
    let params = params();
    let (_, first) = step(3);
    let fresh = Embed(&params)
        .prove_on(
            vesta_sponge(ACCUMULATION_DOMAIN),
            &first.public_inputs().to_vec(),
            &first,
        )
        .unwrap();
    let (mut instance, mut witness) = (fresh.instance, fresh.witness);
    for x in [4, 5, 6, 7] {
        let (_, next) = step(x);
        let input = (next.public_inputs().to_vec(), instance);
        let out = accumulate(&params)
            .prove_on(vesta_sponge(ACCUMULATION_DOMAIN), &input, &(next, witness))
            .unwrap();
        (instance, witness) = (out.instance, out.witness);
    }
    // One proof for the five steps.
    let decider = VanillaDecider::new(&params).unwrap();
    let out = decider
        .prove_on(vesta_sponge(DECIDER_DOMAIN), &instance, &witness)
        .unwrap();
    decider
        .verify_on(vesta_sponge(DECIDER_DOMAIN), &instance, out.proof)
        .unwrap();
}

#[test]
fn a_snark_by_composition() {
    let params = params();
    let decider = VanillaDecider::new(&params).unwrap();
    let s = snark(&decider);
    fn is_argument<A: Argument<F>>(_: &A) {}
    is_argument(&s);
    let (_, w) = step(9);
    let statement = w.public_inputs().to_vec();
    let out = s
        .prove_on(vesta_sponge(DECIDER_DOMAIN), &statement, &w)
        .unwrap();
    assert!(
        s.verify_on(vesta_sponge(DECIDER_DOMAIN), &statement, out.proof.clone())
            .is_ok()
    );
    let other = vec![statement[0] + F::ONE];
    assert!(
        s.verify_on(vesta_sponge(DECIDER_DOMAIN), &other, out.proof)
            .is_err()
    );
}
