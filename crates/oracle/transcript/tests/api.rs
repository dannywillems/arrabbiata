//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::{Field, Goldilocks as F};
use arrabbiata_poseidon::{goldilocks, goldilocks_sponge};
use arrabbiata_transcript::{
    Absorb, Duplex, Permutation, Transcript, absorb_length, absorb_slice,
    hlist::{HCons, HList, HNil, Reverse},
    typed::{
        Challenge, End, Label, Labeled, Message, Pattern, ProofOf, Protocol, Prover, Public, Tape,
        Then, Verifier, prover_tape, verifier_tape,
    },
};

#[test]
fn hlist_h_nil() {
    assert_eq!(<HNil as HList>::LEN, 0);
}

#[test]
fn hlist_h_cons() {
    let l = HCons::new(1u8, HCons::new("two", HNil));
    assert_eq!(l.tail.head, "two");
}

#[test]
fn hlist_h_cons_new() {
    let l = HCons::new(true, HNil);
    assert!(l.head);
}

#[test]
fn hlist_h_list() {
    assert_eq!(<HCons<u8, HCons<bool, HNil>> as HList>::LEN, 2);
}

#[test]
fn hlist_reverse() {
    let r = HCons::new(1u8, HCons::new(2u16, HNil)).reverse_onto(HNil);
    assert_eq!((r.head, r.tail.head), (2u16, 1u8));
}

#[test]
fn hlist_h_cons_reverse_onto() {
    let r = HCons::new(1u8, HNil).reverse_onto(HCons::new(9u8, HNil));
    assert_eq!((r.head, r.tail.head), (1, 9));
}

#[test]
fn transcript() {
    fn challenge_after<T>(t: &mut T, x: F) -> F
    where
        T: Transcript<F>,
    {
        t.absorb(x);
        t.squeeze()
    }
    let c = challenge_after(&mut goldilocks_sponge(0), F::new(1));
    assert_eq!(c, challenge_after(&mut goldilocks_sponge(0), F::new(1)));
}

#[test]
fn absorb() {
    let mut t = goldilocks_sponge(0);
    t.absorb(F::new(1));
}

#[test]
fn squeeze() {
    let mut t = goldilocks_sponge(0);
    let a = t.squeeze();
    let b = t.squeeze();
    assert_ne!(a, b);
}

#[test]
fn absorb_2() {
    struct Point {
        x: F,
        y: F,
    }
    impl Absorb<F> for Point {
        fn absorb_into<T>(&self, t: &mut T)
        where
            T: Transcript<F>,
        {
            t.absorb(self.x);
            t.absorb(self.y);
        }
    }
    let mut t = goldilocks_sponge(0);
    Point {
        x: F::ONE,
        y: F::ZERO,
    }
    .absorb_into(&mut t);
}

#[test]
fn absorb_into() {
    let mut t = goldilocks_sponge(0);
    F::new(5).absorb_into(&mut t);
    let _ = t.squeeze();
}

#[test]
fn absorb_length_works() {
    let mut t = goldilocks_sponge(0);
    absorb_length(&mut t, 3);
}

#[test]
fn absorb_slice_works() {
    let mut t = goldilocks_sponge(0);
    absorb_slice(&mut t, &[F::ONE, F::ONE]);
    let _ = t.squeeze();
}

#[test]
fn permutation() {
    /// Rotate the state: a (useless) permutation of F^2.
    struct Swap;
    impl Permutation<F, 2> for Swap {
        fn permute(&self, state: &mut [F; 2]) {
            state.swap(0, 1);
        }
    }
    let mut sponge = Duplex::<F, Swap, 2, 1>::new(Swap, 0);
    sponge.absorb(F::new(3));
    assert_eq!(sponge.squeeze(), F::ZERO);
}

#[test]
fn f_permute() {
    let mut state = [F::ZERO; 12];
    goldilocks().permute(&mut state);
    assert_ne!(state, [F::ZERO; 12]);
}

#[test]
fn duplex() {
    let mut sponge = Duplex::<F, _, 12, 8>::new(goldilocks(), 42);
    sponge.absorb(F::ONE);
    let _challenge = sponge.squeeze();
}

#[test]
fn duplex_new() {
    let mut a = Duplex::<F, _, 12, 8>::new(goldilocks(), 1);
    let mut b = Duplex::<F, _, 12, 8>::new(goldilocks(), 2);
    assert_ne!(a.squeeze(), b.squeeze());
}

#[test]
fn typed_overview() {
    struct Commitment(F);
    impl Label for Commitment {
        const LABEL: u64 = 1;
    }
    impl arrabbiata_transcript::Absorb<F> for Commitment {
        fn absorb_into<T>(&self, t: &mut T)
        where
            T: arrabbiata_transcript::Transcript<F>,
        {
            t.absorb(self.0);
        }
    }

    // Sigma-protocol shape: commitment, challenge, end.
    type Sigma = Message<Commitment, Challenge<End>>;

    let tape = Tape::<F, _, Sigma>::start(goldilocks_sponge(0));
    let tape = tape.send(&Commitment(F::new(42)));
    let (challenge, tape) = tape.challenge();
    let _sponge = tape.finish();
    let _ = challenge;
}

#[test]
fn typed_label() {
    struct Opening;
    impl Label for Opening {
        const LABEL: u64 = 0x4f50;
    }
    assert_eq!(Opening::LABEL, 0x4f50);
}

#[test]
fn typed_labeled() {
    type Beta = Labeled<F, 0x6265>;
    assert_eq!(Beta::LABEL, 0x6265);
    let b: Beta = Labeled(F::new(1));
    assert_eq!(b.0, F::new(1));
}

#[test]
fn typed_end() {
    assert_eq!(<End as Pattern>::MESSAGES, 0);
}

#[test]
fn typed_message() {
    type P = Message<F, Message<F, End>>;
    assert_eq!(<P as Pattern>::MESSAGES, 2);
    let _: <P as Pattern>::Proof = HCons::new(F::ONE, HCons::new(F::ZERO, HNil));
}

#[test]
fn typed_challenge() {
    assert_eq!(<Challenge<Challenge<End>> as Pattern>::CHALLENGES, 2);
}

#[test]
fn typed_public() {
    // A public value contributes nothing to the proof.
    let _: <Public<F, End> as Pattern>::Proof = HNil;
}

#[test]
fn typed_pattern() {
    type Sumcheck2 = Message<F, Challenge<Message<F, Challenge<End>>>>;
    assert_eq!(<Sumcheck2 as Pattern>::MESSAGES, 2);
    assert_eq!(<Sumcheck2 as Pattern>::CHALLENGES, 2);
}

#[test]
fn typed_labeled_describe() {
    let mut a = goldilocks_sponge(0);
    let mut b = goldilocks_sponge(0);
    <Message<F, End> as Pattern>::describe(&mut a);
    <Challenge<End> as Pattern>::describe(&mut b);
    assert_ne!(a.squeeze(), b.squeeze());
}

#[test]
fn typed_concat() {
    type A = Message<F, End>;
    type B = Challenge<End>;
    let _same: Option<Then<A, B>> = None::<Message<F, Challenge<End>>>;
}

#[test]
fn typed_tape() {
    let tape = Tape::<F, _, Challenge<End>>::start(goldilocks_sponge(0));
    let (_c, tape) = tape.challenge();
    let _sponge = tape.finish();
}

#[test]
fn typed_tape_start() {
    let _tape = Tape::<F, _, End>::start(goldilocks_sponge(0));
}

#[test]
fn typed_tape_send() {
    let tape = Tape::<F, _, Message<F, End>>::start(goldilocks_sponge(0));
    let _ = tape.send(&F::ONE).finish();
}

#[test]
fn typed_tape_receive() {
    let tape = Tape::<F, _, Message<F, End>>::start(goldilocks_sponge(0));
    let _ = tape.receive(&F::ONE).finish();
}

#[test]
fn typed_tape_public() {
    let tape = Tape::<F, _, Public<F, End>>::start(goldilocks_sponge(0));
    let _ = tape.public(&F::new(1)).finish();
}

#[test]
fn typed_tape_challenge() {
    let (c, _) = Tape::<F, _, Challenge<End>>::start(goldilocks_sponge(0)).challenge();
    let (d, _) = Tape::<F, _, Challenge<End>>::start(goldilocks_sponge(0)).challenge();
    assert_eq!(c, d);
}

#[test]
fn typed_tape_finish() {
    let mut sponge = Tape::<F, _, End>::start(goldilocks_sponge(0)).finish();
    let _ = sponge.squeeze();
}

#[test]
fn typed_prover() {
    type P = Message<F, Challenge<End>>;
    let (_c, prover) = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(0))
        .send(F::ONE)
        .challenge();
    let (_, proof) = prover.finish();
    assert_eq!(proof, HCons::new(F::ONE, HNil));
}

#[test]
fn typed_prover_start() {
    type P = End;
    let (_, proof) = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(0)).finish();
    assert_eq!(proof, HNil);
}

#[test]
fn typed_prover_send() {
    type P = Message<F, End>;
    let (_, proof) = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(0))
        .send(F::new(3))
        .finish();
    assert_eq!(proof.head, F::new(3));
}

#[test]
fn typed_prover_public() {
    type P = Public<F, End>;
    let (_, proof) = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(0))
        .public(&F::new(1))
        .finish();
    assert_eq!(proof, HNil);
}

#[test]
fn typed_prover_challenge() {
    type P = Challenge<End>;
    let (_c, prover) = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(0)).challenge();
    let _ = prover.finish();
}

#[test]
fn typed_prover_finish() {
    type P = Message<F, Message<F, End>>;
    let prover = Prover::<F, _, P, P, HNil>::start(goldilocks_sponge(0))
        .send(F::new(1))
        .send(F::new(2));
    let (_, proof) = prover.finish();
    // In protocol order, not in reverse.
    assert_eq!((proof.head, proof.tail.head), (F::new(1), F::new(2)));
}

#[test]
fn typed_verifier() {
    type P = Message<F, Challenge<End>>;
    let verifier = Verifier::<F, _, P>::start(goldilocks_sponge(0), HCons::new(F::ONE, HNil));
    let (m, verifier) = verifier.read();
    let (_c, verifier) = verifier.challenge();
    let _ = verifier.finish();
    assert_eq!(m, F::ONE);
}

#[test]
fn typed_verifier_start() {
    let _ = Verifier::<F, _, End>::start(goldilocks_sponge(0), HNil).finish();
}

#[test]
fn typed_verifier_read() {
    let verifier =
        Verifier::<F, _, Message<F, End>>::start(goldilocks_sponge(0), HCons::new(F::new(8), HNil));
    let (m, _) = verifier.read();
    assert_eq!(m, F::new(8));
}

#[test]
fn typed_verifier_public() {
    let verifier = Verifier::<F, _, Public<F, End>>::start(goldilocks_sponge(0), HNil);
    let _ = verifier.public(&F::new(1)).finish();
}

#[test]
fn typed_verifier_challenge() {
    let (_c, v) = Verifier::<F, _, Challenge<End>>::start(goldilocks_sponge(0), HNil).challenge();
    let _ = v.finish();
}

#[test]
fn typed_verifier_finish() {
    let _sponge = Verifier::<F, _, End>::start(goldilocks_sponge(0), HNil).finish();
}

#[test]
fn typed_protocol() {
    /// Prove knowledge of nothing: absorb a statement and a message, squeeze.
    struct Sigma;
    impl Protocol for Sigma {
        type Transcript = Public<F, Message<F, Challenge<End>>>;
    }

    let (c, prover) = prover_tape::<F, _, Sigma>(goldilocks_sponge(0))
        .public(&F::new(1))
        .send(F::new(2))
        .challenge();
    let (_, proof): (_, ProofOf<Sigma>) = prover.finish();

    let (d, verifier) = verifier_tape::<F, _, Sigma>(goldilocks_sponge(0), proof)
        .public(&F::new(1))
        .read()
        .1
        .challenge();
    let _ = verifier.finish();
    assert_eq!(c, d);
}

#[test]
fn typed_prover_tape() {
    struct Empty;
    impl Protocol for Empty {
        type Transcript = End;
    }
    let (_, _proof) = prover_tape::<F, _, Empty>(goldilocks_sponge(0)).finish();
}

#[test]
fn typed_verifier_tape() {
    struct Empty;
    impl Protocol for Empty {
        type Transcript = End;
    }
    let _ = verifier_tape::<F, _, Empty>(goldilocks_sponge(0), HNil).finish();
}
