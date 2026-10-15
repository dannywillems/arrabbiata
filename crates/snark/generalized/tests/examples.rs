//! Complete runs of generalized PlonK, from a circuit to a verified proof.
//! The rustdoc of `setup`, `prove` and `verify` points here.
//!
//! - an iterated map with a public parameter: `x_(i+1) = s x_i^3`, the
//!   scale `s` a public input the gate reads, the last iterate bound to a
//!   second public input; a claim with another scale or another result is
//!   rejected;
//! - a public input bound to a witness cell, the smallest statement there
//!   is.
//!
//! Lookups and the permutation on some columns have their own examples:
//! `tests/lookups.rs` (range checks, a XOR table) and `tests/permutation.rs`
//! (a sum of 4-bit numbers).

use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::{
    Circuit, Error, Gate, Variable, max_coefficients, prove, setup, verify,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_pasta::{Fq as F, Pallas};
use arrabbiata_poly::Rotation;
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts: prover and verifier start
/// from the same one.
const PROOF_DOMAIN: u64 = 2;

fn ipa(size: usize) -> Ipa<Pallas, PoseidonSponge<F>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript)
}

/// The iterates of `x -> s x^3` on `n` rows: the gate
/// `w(next) - s w^3`, on every row but the last; `s` is public input 0, and
/// the last iterate is bound to public input 1.
fn scaled_cubes(n: usize) -> Circuit<F, 3, 4> {
    let v = |i| Sparse::<F, 3, 4>::variable(Var::try_new(i).unwrap()).unwrap();
    let (current, next, scale) = (v(0), v(1), v(2));
    let cube = current
        .checked_mul(&current)
        .unwrap()
        .checked_mul(&current)
        .unwrap();
    let step = next - scale.checked_mul(&cube).unwrap();
    let on_every_row_but_the_last = Some(0);
    let gate = Gate::new(on_every_row_but_the_last, step);
    let mut selector = vec![F::ONE; n];
    selector[n - 1] = F::ZERO;

    let num_rows = n;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![selector];
    let variables = vec![
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Witness {
            column: 0,
            rotation: Rotation::NEXT,
        },
        Variable::Public(0),
    ];
    let gates = vec![gate];
    let sigma = (0..n).collect();
    let public_inputs = 2;
    let last_iterate = (1, n - 1);
    let bindings = vec![last_iterate];
    Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    )
    .unwrap()
}

#[test]
fn iterates_with_a_public_scale() {
    let n = 8;
    let circuit = scaled_cubes(n);
    let scale = F::from_u64(2);
    let start = F::from_u64(3);
    let mut iterates = vec![start];
    for i in 0..n - 1 {
        let x = iterates[i];
        iterates.push(scale * x * x * x);
    }
    let result = iterates[n - 1];
    let statement = [scale, result];

    let pcs = ipa(max_coefficients(&circuit));
    let (prover_key, verifier_key) = setup(&circuit, &pcs).unwrap();
    let columns = [iterates];
    let (proof, _) = prove(
        &prover_key,
        &pcs,
        vesta_sponge(PROOF_DOMAIN),
        &statement,
        &columns,
    )
    .unwrap();
    verify(
        &verifier_key,
        &pcs,
        vesta_sponge(PROOF_DOMAIN),
        &statement,
        proof.clone(),
    )
    .unwrap();

    // The proof is for this scale and this result only.
    let other_scale = [scale + F::ONE, result];
    let other_result = [scale, result + F::ONE];
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &other_scale,
            proof.clone()
        )
        .is_err()
    );
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &other_result,
            proof
        )
        .is_err()
    );
    // And the prover cannot claim another result.
    assert_eq!(
        prove(
            &prover_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &other_result,
            &columns
        )
        .unwrap_err(),
        Error::Unsatisfied
    );
}

#[test]
fn a_public_input_bound_to_a_cell() {
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![0, 1];
    let public_inputs = 1;
    let row_1_holds_input_0 = (0, 1);
    let bindings = vec![row_1_holds_input_0];
    let circuit = Circuit::<F, 1, 1>::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    )
    .unwrap();

    let pcs = ipa(max_coefficients(&circuit));
    let (prover_key, verifier_key) = setup(&circuit, &pcs).unwrap();
    let seven = F::from_u64(7);
    let columns = [vec![F::ZERO, seven]];
    let (proof, _) = prove(
        &prover_key,
        &pcs,
        vesta_sponge(PROOF_DOMAIN),
        &[seven],
        &columns,
    )
    .unwrap();
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &[seven],
            proof.clone()
        )
        .is_ok()
    );
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &[F::from_u64(8)],
            proof
        )
        .is_err()
    );
    assert_eq!(
        prove(
            &prover_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &[F::from_u64(8)],
            &columns
        )
        .unwrap_err(),
        Error::Unsatisfied
    );
}
