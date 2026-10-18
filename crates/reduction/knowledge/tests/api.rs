//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_pasta::Fq as F;
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_reduction::{
    Argument, Batch, Concat, Each, Either, Identity, Indexed, Output, OutputOf, Parallel, Product,
    Reduction, Relation, Single, Then, Top, Unit,
    laws::{Failure, complete, same_instance},
    tape,
};
use arrabbiata_transcript::{
    Transcript,
    typed::{Challenge, End, Seq, Tail, Tape},
};

#[test]
fn relation() {
    /// `w * w = x` over the integers below 2^32.
    struct Square;
    impl Relation for Square {
        type Instance = u64;
        type Witness = u32;
        type Error = ();
        fn check(&self, x: &u64, w: &u32) -> Result<(), ()> {
            if u64::from(*w) * u64::from(*w) == *x {
                Ok(())
            } else {
                Err(())
            }
        }
    }
    assert!(Square.check(&9, &3).is_ok());
    assert!(Square.check(&8, &3).is_err());
}

#[test]
fn check() {
    assert!(Top.check(&(), &()).is_ok());
}

#[test]
fn top() {
    assert_eq!(Top.check(&(), &()), Ok(()));
}

#[test]
fn unit() {
    assert_eq!(Unit.check(&(), &()), Ok(()));
}

#[test]
fn either() {
    let e: Either<u8, &str> = Either::Right("second");
    assert_eq!(e.to_string(), "right: second");
}

#[test]
fn product() {
    /// `w = x` over `u8`.
    struct Equal;
    impl Relation for Equal {
        type Instance = u8;
        type Witness = u8;
        type Error = ();
        fn check(&self, x: &u8, w: &u8) -> Result<(), ()> {
            if x == w { Ok(()) } else { Err(()) }
        }
    }
    let both = Product(Equal, Equal);
    assert!(both.check(&(1, 2), &(1, 2)).is_ok());
    assert_eq!(both.check(&(1, 2), &(1, 3)), Err(Either::Right(())));
}

#[test]
fn output() {
    let out = Output {
        instance: 1u8,
        witness: 2u8,
        proof: (),
        transcript: (),
    };
    assert_eq!((out.instance, out.witness), (1, 2));
}

#[test]
fn reduction() {
    let id = Identity(Top);
    // Top has one instance, `()`: proving and verifying both succeed.
    let proved = Reduction::<F>::prove_on(&id, vesta_sponge(0), &(), &());
    let verified = Reduction::<F>::verify_on(&id, vesta_sponge(0), &(), ());
    assert!(proved.is_ok());
    assert!(verified.is_ok());
}

#[test]
fn product_source() {
    assert_eq!(<Identity<Top> as Reduction<F>>::source(&Identity(Top)), Top);
}

#[test]
fn product_target() {
    assert_eq!(<Identity<Top> as Reduction<F>>::target(&Identity(Top)), Top);
}

#[test]
fn product_prove() {
    let out = Reduction::<F>::prove_on(&Identity(Top), vesta_sponge(0), &(), &()).unwrap();
    assert_eq!(out.witness, ());
}

#[test]
fn product_verify() {
    let (u, _) = Reduction::<F>::verify_on(&Identity(Top), vesta_sponge(0), &(), ()).unwrap();
    assert_eq!(u, ());
}

#[test]
fn r_impl() {
    let id = Identity(Top);
    let both = Then(&id, &id);
    assert!(Reduction::<F>::prove_on(&both, vesta_sponge(0), &(), &()).is_ok());
}

#[test]
fn argument() {
    fn is_argument<A: Argument<F>>(_: &A) {}
    is_argument(&Identity(arrabbiata_reduction::Top));
}

#[test]
fn identity() {
    let out = Reduction::<F>::prove_on(&Identity(Unit), vesta_sponge(0), &(), &()).unwrap();
    assert_eq!((out.instance, out.witness, out.proof), ((), (), ()));
}

#[test]
fn then() {
    let both = Then(Identity(Top), Identity(Top));
    let out = Reduction::<F>::prove_on(&both, vesta_sponge(0), &(), &()).unwrap();
    assert_eq!(out.proof, ((), ()));
}

#[test]
fn parallel() {
    let both = Parallel(Identity(Top), Identity(Unit));
    assert_eq!(Reduction::<F>::source(&both), Product(Top, Unit));
    let out = Reduction::<F>::prove_on(&both, vesta_sponge(0), &((), ()), &((), ())).unwrap();
    assert_eq!(out.instance, ((), ()));
}

#[test]
fn failure() {
    let f: Failure<(), (), ()> = Failure::Disagree;
    assert_eq!(f, Failure::Disagree);
}

#[test]
fn parallel_complete() {
    let r = Then(Identity(Top), Identity(Top));
    assert!(complete::<F, _, _>(&r, || vesta_sponge(0), &(), &()).is_ok());
}

#[test]
fn parallel_same_instance() {
    let a = Identity(Top);
    let b = Then(Identity(Top), Identity(Top));
    let (pa, pb) = ((), ((), ()));
    assert!(same_instance::<F, _, _, _>(&a, &b, || vesta_sponge(0), &(), pa, pb).unwrap());
}

#[test]
fn indexed() {
    // A list error names the failing index, or the length mismatch.
    assert_eq!(format!("{}", Indexed::At(2, "bad")), "at 2: bad");
    assert_eq!(
        format!("{}", Indexed::<&str>::Length),
        "lengths do not match"
    );
}

#[test]
fn batch() {
    // `Top^*`: any list of units is in it, if the lengths agree.
    assert!(Batch(Top).check(&vec![(), ()], &vec![(), ()]).is_ok());
    assert_eq!(Batch(Top).check(&vec![()], &vec![]), Err(Indexed::Length));
}

#[test]
fn each() {
    // The identity on each element of a list of three.
    let each = Each(Identity(Top));
    let (instances, _) =
        Reduction::<F>::verify_on(&each, vesta_sponge(0), &vec![(); 3], vec![(); 3]).unwrap();
    assert_eq!(instances.len(), 3);
    assert_eq!(
        Reduction::<F>::verify_on(&each, vesta_sponge(0), &vec![(); 3], vec![(); 2]).err(),
        Some(Indexed::Length)
    );
}

#[test]
fn single() {
    let (list, _) = Reduction::<F>::verify_on(&Single(Top), vesta_sponge(0), &(), ()).unwrap();
    assert_eq!(list, vec![()]);
}

#[test]
fn concat() {
    let (list, _) =
        Reduction::<F>::verify_on(&Concat(Top), vesta_sponge(0), &(vec![()], vec![(), ()]), ())
            .unwrap();
    assert_eq!(list.len(), 3);
}

/// Every field element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Anything;

impl Relation for Anything {
    type Instance = F;
    type Witness = ();
    type Error = ();
    fn check(&self, _: &F, (): &()) -> Result<(), ()> {
        Ok(())
    }
}

/// From `Top` to a challenge: one step, a draw.
struct Draw;

impl Reduction<F> for Draw {
    type Source = Top;
    type Target = Anything;
    type Proof = ();
    type Error = ();
    type Pattern = Challenge<End>;

    fn source(&self) -> Top {
        Top
    }

    fn target(&self) -> Anything {
        Anything
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Challenge<End>, R>>,
        (): &(),
        (): &(),
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, ()>
    where
        T: Transcript<F>,
    {
        let (x, tape) = tape.challenge();
        Ok(Output {
            instance: x,
            witness: (),
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Challenge<End>, R>>,
        (): &(),
        (): (),
    ) -> Result<(F, Tape<F, T, R>), ()>
    where
        T: Transcript<F>,
    {
        Ok(tape.challenge())
    }
}

#[test]
fn reduction_pattern() {
    // A composite's pattern is its parts' in sequence.
    fn pattern_of<R: Reduction<F, Pattern = P>, P>(_: &R) {}
    pattern_of::<_, Seq<Challenge<End>, Challenge<End>>>(&Then(Draw, Draw2));
    pattern_of::<_, Seq<Challenge<End>, End>>(&Then(Draw, Identity(Anything)));
    pattern_of::<_, End>(&Identity(Top));
}

#[test]
fn reduction_prove() {
    // On a longer tape: the reduction's steps, then the tape handed back
    // at the rest of the protocol.
    let tape = Tape::<F, _, Tail<Challenge<End>, Challenge<End>>>::undescribed(vesta_sponge(0));
    let out = Draw.prove(tape, &(), &()).unwrap();
    let (next, _) = out.transcript.challenge();
    let (a, b) = {
        let mut sponge = vesta_sponge(0);
        (sponge.squeeze(), sponge.squeeze())
    };
    assert_eq!((out.instance, next), (a, b));
}

#[test]
fn reduction_verify() {
    let tape = Tape::<F, _, Tail<Challenge<End>, End>>::undescribed(vesta_sponge(0));
    let (x, _) = Draw.verify(tape, &(), ()).unwrap();
    assert_eq!(x, vesta_sponge(0).squeeze());
}

#[test]
fn reduction_prove_on() {
    // On its own tape: the pattern described first, so the challenge is not
    // the bare sponge's.
    let out = Draw.prove_on(vesta_sponge(0), &(), &()).unwrap();
    assert_ne!(out.instance, vesta_sponge(0).squeeze());
    let (x, _) = Draw.verify_on(vesta_sponge(0), &(), ()).unwrap();
    assert_eq!(out.instance, x);
}

#[test]
fn reduction_verify_on() {
    let (x, after) = Draw.verify_on(vesta_sponge(0), &(), ()).unwrap();
    let (y, again) = Draw.verify_on(vesta_sponge(0), &(), ()).unwrap();
    assert_eq!(x, y);
    assert_eq!(after.finish().squeeze(), again.finish().squeeze());
}

#[test]
fn reduction_tape() {
    // The tape `prove_on` starts from.
    let started = tape::<F, Draw, _>(vesta_sponge(0));
    let out = Draw.prove(started, &(), &()).unwrap();
    let direct = Draw.prove_on(vesta_sponge(0), &(), &()).unwrap();
    assert_eq!(out.instance, direct.instance);
}

#[test]
fn then_splits_its_tape() {
    // `Then(A, B)`'s tape is `A`'s steps then `B`'s: its parts run on it in
    // turn reach the composite's instance.
    let both = Then(Draw, Draw2);
    let whole = both.prove_on(vesta_sponge(0), &(), &()).unwrap();
    let first = Draw
        .prove(tape::<F, Then<Draw, Draw2>, _>(vesta_sponge(0)), &(), &())
        .unwrap();
    let second = Draw2.prove(first.transcript, &first.instance, &()).unwrap();
    assert_eq!(whole.instance, second.instance);
}

/// From a field element to a challenge drawn after it.
struct Draw2;

impl Reduction<F> for Draw2 {
    type Source = Anything;
    type Target = Anything;
    type Proof = ();
    type Error = ();
    type Pattern = Challenge<End>;

    fn source(&self) -> Anything {
        Anything
    }

    fn target(&self) -> Anything {
        Anything
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Challenge<End>, R>>,
        x: &F,
        (): &(),
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, ()>
    where
        T: Transcript<F>,
    {
        let (y, tape) = tape.challenge();
        Ok(Output {
            instance: *x + y,
            witness: (),
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Challenge<End>, R>>,
        x: &F,
        (): (),
    ) -> Result<(F, Tape<F, T, R>), ()>
    where
        T: Transcript<F>,
    {
        let (y, tape) = tape.challenge();
        Ok((*x + y, tape))
    }
}
