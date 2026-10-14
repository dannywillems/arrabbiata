//! The typed transcript: patterns, tapes, prover and verifier.

use arrabbiata_field::Goldilocks as F;

use arrabbiata_poseidon::goldilocks_sponge;
use arrabbiata_transcript::{
    Absorb, Transcript,
    hlist::{HCons, HNil},
    typed::{Challenge, End, Label, Message, Pattern, Prover, Tape, Then, Verifier},
};

type Sigma = Message<F, Challenge<End>>;
type TwoRounds = Then<Sigma, Sigma>;

#[test]
fn counts_are_known_at_compile_time() {
    const { assert!(<TwoRounds as Pattern>::MESSAGES == 2) };
    const { assert!(<TwoRounds as Pattern>::CHALLENGES == 2) };
}

#[test]
fn composition_is_the_concatenation() {
    // Then<Sigma, Sigma> is literally Message<F, Challenge<Message<F,
    // Challenge<End>>>>; the assignment only type-checks if so.
    type Expected = Message<F, Challenge<Message<F, Challenge<End>>>>;
    let tape: Tape<F, _, Expected> = Tape::<F, _, TwoRounds>::start(goldilocks_sponge(0));
    let (c1, tape) = tape.send(&F::new(1)).challenge();
    let (c2, tape) = tape.send(&F::new(2)).challenge();
    let _ = tape.finish();
    assert_ne!(c1, c2);
}

#[test]
fn prover_and_verifier_agree() {
    let prover = Tape::<F, _, Sigma>::start(goldilocks_sponge(7)).send(&F::new(3));
    let verifier = Tape::<F, _, Sigma>::start(goldilocks_sponge(7)).receive(&F::new(3));
    assert_eq!(prover.challenge().0, verifier.challenge().0);
}

#[test]
fn prover_proof_replays_on_the_verifier() {
    type P = Then<Sigma, Message<u64Label, Challenge<End>>>;
    let prover = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(1));
    let (c1, prover) = prover.send(F::new(10)).challenge();
    let (c2, prover) = prover.send(u64Label(5)).challenge();
    let (_, proof) = prover.finish();
    let _: HCons<F, HCons<u64Label, HNil>> = proof;

    let verifier = Verifier::<F, _, P>::start(goldilocks_sponge(1), proof);
    let (m1, verifier) = verifier.read();
    let (d1, verifier) = verifier.challenge();
    let (m2, verifier) = verifier.read();
    let (d2, verifier) = verifier.challenge();
    let _ = verifier.finish();
    assert_eq!((m1, m2, d1, d2), (F::new(10), u64Label(5), c1, c2));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
struct u64Label(u64);

impl Label for u64Label {
    const LABEL: u64 = 99;
}

impl Absorb<F> for u64Label {
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::new(self.0));
    }
}

#[test]
fn the_pattern_separates_protocols() {
    // Same messages, different declared protocol: different challenges.
    type Other = Message<F, Challenge<Message<F, End>>>;
    let a = Tape::<F, _, Sigma>::start(goldilocks_sponge(0)).send(&F::new(3));
    let b = Tape::<F, _, Other>::start(goldilocks_sponge(0)).send(&F::new(3));
    assert_ne!(a.challenge().0, b.challenge().0);
}
