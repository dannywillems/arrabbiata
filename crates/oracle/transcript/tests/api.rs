//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::{Field, Goldilocks as F, Native};
use arrabbiata_poseidon::{goldilocks, goldilocks_sponge};
use arrabbiata_transcript::{
    Absorb, Duplex, Permutation, PermutationIn, Transcript, absorb_length, absorb_slice,
    hlist::{HCons, HList, HNil, Reverse},
    typed::{
        Agreed, Branch, Challenge, Draw, Either, End, Exactly, Label, Labeled, Message, Pattern,
        Peek, ProofOf, Protocol, Prover, Public, Rounds, Send, Seq, Sequence, Squeeze, Statement,
        Tail, Tape, Then, Verifier, Write, prove_within, prover_tape, verifier_tape, verify_within,
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

/// `Self` then `R`, at the type level: the steps of `P`, then those of `R`.
fn steps_then<P: Sequence, R>(log: Log) -> Tape<F, Log, P::Tail<R>> {
    Tape::undescribed(log)
}

#[test]
fn typed_sequence() {
    // A pattern's tail replaced by another pattern: its steps, then the
    // other's, on one tape.
    type P = Send<Word, Squeeze<Count, End>>;
    let tape = steps_then::<P, Send<Word, End>>(Log::default());
    let (x, tape) = tape.send(&Word(1)).squeeze();
    assert_eq!(x, 1);
    assert_eq!(tape.send(&Word(2)).into_inner().written, vec![1, 2]);
}

#[test]
fn typed_tail() {
    // `Tail<P, R>` is `P` followed by `R`; `Tail<P, End>` is `P`.
    type P = Send<Word, End>;
    let tape = Tape::<F, Log, Tail<P, Send<Word, End>>>::undescribed(Log::default());
    let log = tape.send(&Word(1)).send(&Word(2)).into_inner();
    assert_eq!(log.written, vec![1, 2]);
    let _: Tape<F, Log, P> = Tape::<F, Log, Tail<P, End>>::undescribed(Log::default());
}

#[test]
fn typed_seq() {
    // A sequence of two patterns: their steps in order, their messages
    // summed, and the description of the sequence it denotes, so however a
    // composite is bracketed it is separated alike.
    type A = Message<F, End>;
    type B = Challenge<End>;
    type S = Seq<A, B>;
    assert_eq!(<S as Pattern>::MESSAGES, 1);
    assert_eq!(<S as Pattern>::CHALLENGES, 1);
    let tape = Tape::<F, _, Tail<S, End>>::undescribed(goldilocks_sponge(0));
    let (x, _) = tape.send(&F::ONE).challenge();
    let flat = Tape::<F, _, Message<F, Challenge<End>>>::undescribed(goldilocks_sponge(0));
    let (y, _) = flat.send(&F::ONE).challenge();
    assert_eq!(x, y);
    let mut a = goldilocks_sponge(0);
    let mut b = goldilocks_sponge(0);
    <S as Pattern>::describe::<F, _>(&mut a);
    <Message<F, Challenge<End>> as Pattern>::describe::<F, _>(&mut b);
    assert_eq!(a.squeeze(), b.squeeze());
    type Left = Seq<Seq<A, B>, A>;
    type Right = Seq<A, Seq<B, A>>;
    let mut left = goldilocks_sponge(0);
    let mut right = goldilocks_sponge(0);
    <Left as Pattern>::describe::<F, _>(&mut left);
    <Right as Pattern>::describe::<F, _>(&mut right);
    assert_eq!(left.squeeze(), right.squeeze());
}

#[test]
fn typed_tape_transport() {
    // The transcript replaced by a view of it, the position kept.
    let tape = Tape::<F, Log, Send<Word, End>>::undescribed(Log::default());
    let moved = tape.transport(|mut log| {
        log.written.push(7);
        log
    });
    assert_eq!(moved.send(&Word(1)).into_inner().written, vec![7, 1]);
}

#[test]
fn typed_tape_through() {
    // Steps run through a borrowed view are steps of this tape: the same
    // challenge as stepping directly, and the tape moves on.
    type P = Message<F, Challenge<End>>;
    let tape = Tape::<F, _, P>::undescribed(goldilocks_sponge(0));
    let (x, after) = tape.through(
        |sponge| sponge,
        |inner| {
            let (x, inner) = inner.send(&F::ONE).challenge();
            (x, inner)
        },
        |sponge| sponge,
    );
    let (y, direct) = Tape::<F, _, P>::undescribed(goldilocks_sponge(0))
        .send(&F::ONE)
        .challenge();
    assert_eq!(x, y);
    assert_eq!(after.finish().squeeze(), direct.finish().squeeze());
}

#[test]
fn typed_tape_try_through() {
    // A refusal inside the view ends the protocol.
    type P = Message<F, End>;
    let tape = Tape::<F, _, P>::undescribed(goldilocks_sponge(0));
    let refused: Option<((), Tape<F, _, End>)> =
        tape.try_through(|sponge| sponge, |_| None, |sponge| sponge);
    assert!(refused.is_none());
    let tape = Tape::<F, _, P>::undescribed(goldilocks_sponge(0));
    let accepted = tape.try_through(
        |sponge| sponge,
        |inner| Some(((), inner.send(&F::ONE))),
        |sponge| sponge,
    );
    assert!(accepted.is_some());
}

/// A sub-protocol: a message, then a challenge.
type Sub = Message<F, Challenge<End>>;

#[test]
fn typed_prove_within() {
    // The sub-protocol proved on a longer tape: its proof, its challenge,
    // and the tape handed back at the rest.
    let tape = Tape::<F, _, Tail<Sub, Message<F, End>>>::undescribed(goldilocks_sponge(0));
    let proved = prove_within::<F, _, Sub, _, _, _, ()>(tape, |prover| {
        Ok(prover.send(F::new(3)).challenge())
    });
    let (x, proof, rest) = proved.unwrap();
    assert_eq!(proof.head, F::new(3));
    let (y, alone) = Tape::<F, _, Sub>::undescribed(goldilocks_sponge(0))
        .send(&F::new(3))
        .challenge();
    assert_eq!(x, y);
    let rest = rest.send(&F::ONE).finish();
    let alone = Tape::<F, _, Message<F, End>>::undescribed(alone.finish());
    assert_eq!(rest, alone.send(&F::ONE).finish());
}

#[test]
fn typed_verify_within() {
    // The verifier of the sub-protocol on a longer tape reads the proof and
    // draws the prover's challenge.
    let tape = Tape::<F, _, Tail<Sub, End>>::undescribed(goldilocks_sponge(0));
    let proved = prove_within::<F, _, Sub, _, _, _, ()>(tape, |prover| {
        Ok(prover.send(F::new(3)).challenge())
    });
    let (x, proof, _) = proved.unwrap();
    let tape = Tape::<F, _, Tail<Sub, End>>::undescribed(goldilocks_sponge(0));
    let verified = verify_within::<F, _, Sub, _, _, ()>(tape, proof, |verifier| {
        let (m, verifier) = verifier.read();
        let (y, verifier) = verifier.challenge();
        Ok(((m, y), verifier))
    });
    let ((m, y), _) = verified.unwrap();
    assert_eq!((m, y), (F::new(3), x));
}

/// A round of the tests below: a word sent, a count drawn.
type Round = Send<Word, Squeeze<Count, End>>;

#[test]
fn typed_tape_repeat() {
    // Exactly three rounds, in order, then the rest.
    type P = Rounds<Round, Send<Word, End>, Exactly<3>>;
    let tape = Tape::<F, Log, P>::undescribed(Log::default());
    let (drawn, tape) = tape.repeat(|i, round| round.send(&Word(i as u64)).squeeze());
    assert_eq!(drawn, [1, 2, 3]);
    let log = tape.send(&Word(9)).into_inner();
    assert_eq!(log.written, vec![0, 1, 2, 9]);
}

#[test]
fn typed_tape_try_repeat() {
    // The first refusal ends the rounds: later rounds do not run.
    type P = Rounds<Round, End, Exactly<3>>;
    let tape = Tape::<F, Log, P>::undescribed(Log::default());
    let mut ran = 0;
    let refused = tape.try_repeat(|i, round| {
        ran += 1;
        if i == 1 {
            Err(i)
        } else {
            Ok(round.send(&Word(0)).squeeze())
        }
    });
    assert_eq!(refused.err(), Some(1));
    assert_eq!(ran, 2);
    let tape = Tape::<F, Log, P>::undescribed(Log::default());
    let accepted: Result<_, ()> = tape.try_repeat(|_, round| Ok(round.send(&Word(0)).squeeze()));
    assert_eq!(accepted.map(|(drawn, _)| drawn).ok(), Some([1, 2, 3]));
}

#[test]
fn typed_agreed() {
    // The default count: `Rounds<B, N>` is `Rounds<B, N, Agreed>`, stepped
    // round by round, its counts those of what follows.
    let tape = Tape::<F, Log, Rounds<Round, End, Agreed>>::undescribed(Log::default());
    let _: Tape<F, Log, Rounds<Round, End>> = tape;
    assert_eq!(<Rounds<Round, End> as Pattern>::CHALLENGES, 0);
}

#[test]
fn typed_exactly() {
    // A fixed count is in the counts and in the description: two counts
    // are two protocols, and neither is the agreed one.
    type Body = Message<F, Challenge<End>>;
    assert_eq!(<Rounds<Body, End, Exactly<4>> as Pattern>::MESSAGES, 4);
    assert_eq!(<Rounds<Body, End, Exactly<4>> as Pattern>::CHALLENGES, 4);
    let describe = |which: u8| {
        let mut sponge = goldilocks_sponge(0);
        match which {
            0 => <Rounds<Body, End, Exactly<4>> as Pattern>::describe::<F, _>(&mut sponge),
            1 => <Rounds<Body, End, Exactly<5>> as Pattern>::describe::<F, _>(&mut sponge),
            _ => <Rounds<Body, End> as Pattern>::describe::<F, _>(&mut sponge),
        }
        sponge.squeeze()
    };
    assert_ne!(describe(0), describe(1));
    assert_ne!(describe(0), describe(2));
}
