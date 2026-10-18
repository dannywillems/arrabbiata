//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::{Field, Goldilocks as F, Native};
use arrabbiata_poseidon::{goldilocks, goldilocks_sponge};
use arrabbiata_transcript::{
    Absorb, Duplex, Permutation, PermutationIn, Transcript, absorb_length, absorb_slice,
    hlist::{HCons, HList, HNil, Reverse},
    typed::{
        Branch, Challenge, Draw, Either, End, Label, Labeled, Message, Pattern, Peek, ProofOf,
        Protocol, Prover, Public, Rounds, Send, Squeeze, Statement, Tape, Then, Verifier, Write,
        prover_tape, verifier_tape,
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

/// A transcript that records what is written and counts what is drawn: the
/// nodes below write and draw through it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Log {
    written: Vec<u64>,
    drawn: u64,
}

/// A value written as itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Word(u64);

impl Label for Word {
    const LABEL: u64 = 0x77;
}

impl Write<Log> for Word {
    fn write(&self, log: &mut Log) {
        log.written.push(self.0);
    }
}

/// A challenge: how many were drawn before, counting this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Count;

impl Label for Count {
    const LABEL: u64 = 0x63;
}

impl Draw<Log> for Count {
    type Value = u64;

    fn draw(log: &mut Log) -> u64 {
        log.drawn += 1;
        log.drawn
    }
}

#[test]
fn typed_write() {
    let mut log = Log::default();
    Word(3).write(&mut log);
    assert_eq!(log.written, vec![3]);
}

#[test]
fn typed_draw() {
    let mut log = Log::default();
    assert_eq!(Count::draw(&mut log), 1);
    assert_eq!(Count::draw(&mut log), 2);
}

#[test]
fn typed_statement() {
    // A statement is not part of the proof.
    assert_eq!(<Statement<Word, End> as Pattern>::MESSAGES, 0);
    let tape = Tape::<F, Log, Statement<Word, End>>::undescribed(Log::default());
    assert_eq!(tape.statement(&Word(5)).into_inner().written, vec![5]);
}

#[test]
fn typed_send() {
    assert_eq!(<Send<Word, End> as Pattern>::MESSAGES, 1);
    let tape = Tape::<F, Log, Send<Word, Send<Word, End>>>::undescribed(Log::default());
    let log = tape.send(&Word(1)).receive(&Word(2)).into_inner();
    assert_eq!(log.written, vec![1, 2]);
}

#[test]
fn typed_squeeze() {
    let tape = Tape::<F, Log, Squeeze<Count, Squeeze<Count, End>>>::undescribed(Log::default());
    let (first, tape) = tape.squeeze();
    let (second, _) = tape.squeeze();
    assert_eq!((first, second), (1, 2));
}

#[test]
fn typed_peek() {
    // A peek draws from a copy: the next draw is the same.
    let tape = Tape::<F, Log, Peek<Count, Squeeze<Count, End>>>::undescribed(Log::default());
    let (peeked, tape) = tape.peek();
    let (drawn, _) = tape.squeeze();
    assert_eq!(peeked, drawn);
}

#[test]
fn typed_rounds() {
    // Two rounds of a message and a challenge, then a last message.
    type Round = Send<Word, Squeeze<Count, End>>;
    type P = Rounds<Round, Send<Word, End>>;
    let mut tape = Tape::<F, Log, P>::undescribed(Log::default());
    for i in 0..2 {
        let (x, next) = tape.round().send(&Word(i)).squeeze();
        assert_eq!(x, i + 1);
        tape = next;
    }
    let log = tape.done().send(&Word(9)).into_inner();
    assert_eq!(log.written, vec![0, 1, 9]);
    assert_eq!(<P as Pattern>::MESSAGES, 1);
}

#[test]
fn typed_branch() {
    type P = Branch<Send<Word, End>, End>;
    let left = Tape::<F, Log, P>::undescribed(Log::default());
    assert_eq!(left.left().send(&Word(1)).into_inner().written, vec![1]);
    let right = Tape::<F, Log, P>::undescribed(Log::default());
    assert!(right.right().into_inner().written.is_empty());
    assert_eq!(<P as Pattern>::MESSAGES, 0);
}

#[test]
fn typed_either() {
    let taken: Either<u8, ()> = Either::Left(1);
    assert_ne!(taken, Either::Right(()));
}

#[test]
fn typed_tape_undescribed() {
    // Nothing absorbed at the start: a described tape absorbs its pattern.
    type P = Message<F, End>;
    let plain = Tape::<F, _, P>::undescribed(goldilocks_sponge(0)).send(&F::ONE);
    let described = Tape::<F, _, P>::start(goldilocks_sponge(0)).send(&F::ONE);
    let mut reference = goldilocks_sponge(0);
    reference.absorb(F::ONE);
    let (mut a, mut b) = (plain.finish(), described.finish());
    let expected = reference.squeeze();
    assert_eq!(a.squeeze(), expected);
    assert_ne!(b.squeeze(), expected);
}

#[test]
fn typed_tape_into_inner() {
    let tape = Tape::<F, Log, End>::undescribed(Log::default());
    assert_eq!(tape.into_inner(), Log::default());
}

#[test]
fn typed_new_nodes_describe() {
    // The new nodes are part of a described pattern too: two patterns that
    // differ only in a challenge's kind are separated.
    type A = Squeeze<Count, End>;
    type B = Squeeze<Word, End>;
    let mut a = goldilocks_sponge(0);
    let mut b = goldilocks_sponge(0);
    <A as Pattern>::describe::<F, _>(&mut a);
    <B as Pattern>::describe::<F, _>(&mut b);
    assert_ne!(a.squeeze(), b.squeeze());
}

#[test]
fn permutation_in() {
    // In the native model, the permutation itself; through a borrowed
    // model, the same.
    let p = goldilocks();
    let start: [F; 12] = core::array::from_fn(|i| F::new(i as u64));
    let native = <_ as PermutationIn<Native, F, 12>>::permute_in(&p, &mut Native, start);
    let mut state = start;
    p.permute(&mut state);
    assert_eq!(native, state);
    let mut model = Native;
    let borrowed = <_ as PermutationIn<&mut Native, F, 12>>::permute_in(&p, &mut &mut model, start);
    assert_eq!(borrowed, state);
}

/// Absorb `x`, squeeze: on any transcript.
fn absorb_then_squeeze<T: Transcript<F>>(transcript: &mut T, x: F) -> F {
    transcript.absorb(x);
    transcript.squeeze()
}

#[test]
fn transcript_borrowed() {
    // A borrowed transcript is the transcript: the same challenge, and its
    // state is the owner's.
    let mut owned = goldilocks_sponge(0);
    let mut direct = goldilocks_sponge(0);
    let through = absorb_then_squeeze(&mut &mut owned, F::new(7));
    assert_eq!(through, absorb_then_squeeze(&mut direct, F::new(7)));
    assert_eq!(owned.squeeze(), direct.squeeze());
}
