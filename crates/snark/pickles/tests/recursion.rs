//! Recursion: one step circuit and one wrap circuit, iterated from a dummy
//! base case. Each level's step proof verifies the previous wrap proof and
//! applies the rule (the state plus one); each wrap proof wraps a step
//! proof and finalizes the wrap proof that step proof verified. The head of
//! the chain verifies, at every level, with the same two circuits.

use arrabbiata_field::Field;
use arrabbiata_gadget::TypedGadget;
use arrabbiata_kimchi::{
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_pickles::{
    recursion::{
        Error, Params, RecStep, RecWrap, StepInput, WrapInput, domains_needed, dummy, prove_step,
        prove_wrap, step_index, step_statement, wrap_index, wrap_statement,
    },
    statement::Word,
    wrap::StepEvaluations,
    wrap_one,
};
use arrabbiata_poseidon::kimchi::{pallas_base, vesta_base};

/// How many levels the chain test proves.
const LEVELS: u64 = 3;

/// The domains, fixed in advance.
const PARAMS: Params = Params {
    step_log2: 14,
    wrap_log2: 14,
};
const SS: usize = PARAMS.step_statement();
const SW: usize = PARAMS.step_private();
const WS: usize = PARAMS.wrap_statement();
const WW: usize = PARAMS.wrap_private();

#[test]
fn the_domains_are_a_fixed_point() {
    // Each circuit, built on the domains fixed in advance, fits them.
    let step_srs = Srs::<Vesta>::create(PARAMS.step_domain());
    let wrap_srs = Srs::<Pallas>::create(PARAMS.wrap_domain());
    let fp_poseidon = pallas_base();
    let fq_poseidon = vesta_base();
    let step = RecStep::<SS, SW>::new(PARAMS, &wrap_srs, &fp_poseidon).unwrap();
    let step_idx = step_index(&step, &step_srs);
    let Ok(step_idx) = step_idx else {
        panic!("step: {step_idx:?}, rows {}", step.rows())
    };
    let padding = vec![Fq::from_u64(7); PARAMS.wrap_rounds()];
    let wrap = RecWrap::<WS, WW>::new(
        PARAMS,
        &step_idx.verifier_index,
        &step_srs,
        &fq_poseidon,
        padding,
    )
    .unwrap();
    let (step_needs, wrap_needs) = domains_needed(&step, &wrap);
    assert_eq!(
        (step_needs, wrap_needs),
        (PARAMS.step_domain(), PARAMS.wrap_domain()),
        "rows: step {}, wrap {}",
        step.rows(),
        wrap.rows()
    );
}

#[test]
fn a_chain_of_one_rule_verifies_at_every_level() {
    let step_srs = Srs::<Vesta>::create(PARAMS.step_domain());
    let wrap_srs = Srs::<Pallas>::create(PARAMS.wrap_domain());
    let fp_poseidon = pallas_base();
    let fq_poseidon = vesta_base();
    let step = RecStep::<SS, SW>::new(PARAMS, &wrap_srs, &fp_poseidon).unwrap();
    let step_idx = step_index(&step, &step_srs).unwrap();
    let padding = vec![Fq::from_u64(7); PARAMS.wrap_rounds()];
    let wrap = RecWrap::<WS, WW>::new(
        PARAMS,
        &step_idx.verifier_index,
        &step_srs,
        &fq_poseidon,
        padding.clone(),
    )
    .unwrap();
    let wrap_idx = wrap_index(&wrap, &wrap_srs).unwrap();
    let wrap_side = (&wrap_idx.verifier_index, &wrap_idx.srs);

    // The dummy cannot stand for a real predecessor: from a state that is
    // not the base case, its IPA must hold, and does not.
    let (mut input, mut previous, mut evaluations) =
        dummy(PARAMS, wrap_side, &step_srs, &padding).unwrap();
    let mut forged = (input.clone(), previous.clone());
    forged.1.app_state = vec![Fp::from_u64(5)];
    forged.0.statement.messages_for_next_step_proof = Word::of(forged.1.digest()).reduce();
    assert_eq!(
        step_statement(PARAMS, wrap_side, &forged.0, &forged.1, &evaluations).err(),
        Some(Error::Opening)
    );

    let mut rng = Arkworks(SplitMix64(101));
    let mut wrap_rng = Arkworks(SplitMix64(103));
    for level in 0..LEVELS {
        // The step proof of the next state, verifying the previous wrap
        // proof (the dummy at the base case).
        let (statement, messages, private, carried) =
            step_statement(PARAMS, wrap_side, &input, &previous, &evaluations).unwrap();
        assert_eq!(messages.app_state, vec![Fp::from_u64(level + 1)]);
        let flag = if level == 0 { Fp::ZERO } else { Fp::ONE };
        assert_eq!(statement.unfinalized.should_finalize, flag);
        let step_proof =
            prove_step(&step, &step_idx, &statement, &private, &carried, &mut rng).unwrap();

        // Its wrap proof, finalizing the wrap proof it verified.
        let step_input = StepInput {
            index: &step_idx.verifier_index,
            srs: &step_idx.srs,
            proof: &step_proof,
            statement: &statement,
            messages: &messages,
        };
        let (wrap_stmt, wrap_messages, wrap_private, wrap_carried) =
            wrap_statement(PARAMS, &step_input, &input, &padding).unwrap();
        let wrap_proof = prove_wrap(
            &wrap,
            &wrap_idx,
            &wrap_stmt,
            &wrap_private,
            &wrap_carried,
            &mut wrap_rng,
        )
        .unwrap();

        // The head verifies: the wrap proof, the step proof's deferred
        // values, both messages and the accumulators.
        let step_evaluations = StepEvaluations::of(&step_proof);
        let mut verify_rng = Arkworks(SplitMix64(107 + level));
        assert_eq!(
            wrap_one::verify(
                &wrap_idx.verifier_index,
                &wrap_idx.srs,
                &wrap_proof,
                &wrap_stmt,
                &wrap_messages,
                (&step_idx.verifier_index, &step_idx.srs, &step_evaluations),
                &messages,
                wrap_side,
                &mut verify_rng,
            ),
            Ok(()),
            "level {level}"
        );

        input = WrapInput::of(
            &wrap_idx.verifier_index,
            &wrap_idx.srs,
            &wrap_proof,
            &wrap_stmt,
            &wrap_messages,
        )
        .unwrap();
        previous = messages;
        evaluations = step_evaluations;
    }
}
