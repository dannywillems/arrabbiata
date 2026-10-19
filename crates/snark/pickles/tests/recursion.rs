//! Recursion: one step circuit and one wrap circuit, iterated from a dummy
//! base case. Each level's step proof verifies the previous wrap proof and
//! applies the rule (the state plus one); each wrap proof wraps a step
//! proof and finalizes the wrap proof that step proof verified. The head of
//! the chain verifies, at every level, with the same two circuits.

use arrabbiata_field::Field;
use arrabbiata_gadget::{TypedGadget, witness};
use arrabbiata_kimchi::{
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_kimchi_gadgets::LAYOUT;
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_pickles::{
    recursion::{
        Error, Params, RecStep, RecWrap, StepInput, WrapInput, domains_needed, dummy, prove_step,
        prove_wrap, step_index, step_statement, verify, wrap_index, wrap_statement,
    },
    statement::Word,
    wrap::StepEvaluations,
};
use arrabbiata_poseidon::kimchi::{pallas_base, vesta_base};

/// How many levels the chain test proves.
const LEVELS: u64 = 3;

/// The position of `z1`'s high part in the step circuit's private input:
/// after the wrap key, the wrap statement and the wrap proof's points.
const Z1: usize = 2 * 28 + PARAMS.wrap_statement() + 2 * (15 + 1 + 7 + 2 * 14 + 2);

/// Whether the step circuit has a witness for `statement` and `private`.
fn satisfied(step: &RecStep<'_, SS, SW>, statement: &[Fp; SS], private: &[Fp; SW]) -> bool {
    witness::<Fp, _, 64, 7>(step, LAYOUT, &(*statement, *private)).is_ok()
}

/// The step circuit against a prover who lies, at `level`: a broken IPA
/// of the wrap proof verified (accepted only at the base case, where the
/// predecessor is the dummy and must not verify), the base case claimed at
/// a later state, and `should_finalize` flipped.
fn attacks(
    step: &RecStep<'_, SS, SW>,
    statement: &arrabbiata_pickles::recursion::step::Statement<Fp>,
    private: &[Fp],
    level: u64,
) {
    let public: [Fp; SS] = statement.to_vec().try_into().unwrap();
    let honest: [Fp; SW] = private.try_into().unwrap();
    assert!(satisfied(step, &public, &honest), "level {level}");

    let mut broken = honest;
    broken[Z1] = broken[Z1] + Fp::ONE;
    assert_eq!(
        satisfied(step, &public, &broken),
        level == 0,
        "level {level}"
    );

    if level > 0 {
        let mut base = honest;
        base[SW - 1] = Fp::ZERO;
        assert!(!satisfied(step, &public, &base), "level {level}");
    }

    let mut flipped = statement.clone();
    flipped.unfinalized.should_finalize = Fp::ONE - flipped.unfinalized.should_finalize;
    let flipped: [Fp; SS] = flipped.to_vec().try_into().unwrap();
    assert!(!satisfied(step, &flipped, &honest), "level {level}");
}

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
        attacks(&step, &statement, &private, level);
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
            verify(
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

        if level == 1 {
            refusals(&step_input, &input, &padding);
            // Step messages over another key, even consistently hashed:
            // the step circuit took the key as a witness, so the verifier
            // pins it.
            let mut other_key = messages.clone();
            other_key.index[0] = other_key.index[0] + Fp::ONE;
            let mut verify_rng = Arkworks(SplitMix64(211));
            assert_eq!(
                verify(
                    &wrap_idx.verifier_index,
                    &wrap_idx.srs,
                    &wrap_proof,
                    &wrap_stmt,
                    &wrap_messages,
                    (&step_idx.verifier_index, &step_idx.srs, &step_evaluations),
                    &other_key,
                    wrap_side,
                    &mut verify_rng,
                ),
                Err(Error::Accumulator)
            );
        }

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

/// The wrap side natively refuses, before proving: an unfinalized proof
/// that does not finalize, and a predecessor whose messages are not those
/// the step statement passes.
fn refusals(step: &StepInput<'_>, predecessor: &WrapInput, padding: &[Fq]) {
    let mut false_perm = step.statement.clone();
    false_perm.unfinalized.perm[1] = Fp::ONE - false_perm.unfinalized.perm[1];
    let false_step = StepInput {
        statement: &false_perm,
        ..*step
    };
    assert_eq!(
        wrap_statement(PARAMS, &false_step, predecessor, padding).err(),
        Some(Error::Deferred("perm"))
    );
    let mut other = predecessor.clone();
    other.messages.challenge_polynomial_commitment.swap(0, 1);
    assert_eq!(
        wrap_statement(PARAMS, step, &other, padding).err(),
        Some(Error::Messages)
    );
}
