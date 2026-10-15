//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_pasta::Fq as F;
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_reduction::{
    Argument, Either, Identity, Output, Parallel, Product, Reduction, Relation, Then, Top, Unit,
    laws::{Failure, complete, same_instance},
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
    let proved = Reduction::<F>::prove(&id, vesta_sponge(0), &(), &());
    let verified = Reduction::<F>::verify(&id, vesta_sponge(0), &(), ());
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
    let out = Reduction::<F>::prove(&Identity(Top), vesta_sponge(0), &(), &()).unwrap();
    assert_eq!(out.witness, ());
}

#[test]
fn product_verify() {
    let (u, _) = Reduction::<F>::verify(&Identity(Top), vesta_sponge(0), &(), ()).unwrap();
    assert_eq!(u, ());
}

#[test]
fn r_impl() {
    let id = Identity(Top);
    let both = Then(&id, &id);
    assert!(Reduction::<F>::prove(&both, vesta_sponge(0), &(), &()).is_ok());
}

#[test]
fn argument() {
    fn is_argument<A: Argument<F>>(_: &A) {}
    is_argument(&Identity(arrabbiata_reduction::Top));
}

#[test]
fn identity() {
    let out = Reduction::<F>::prove(&Identity(Unit), vesta_sponge(0), &(), &()).unwrap();
    assert_eq!((out.instance, out.witness, out.proof), ((), (), ()));
}

#[test]
fn then() {
    let both = Then(Identity(Top), Identity(Top));
    let out = Reduction::<F>::prove(&both, vesta_sponge(0), &(), &()).unwrap();
    assert_eq!(out.proof, ((), ()));
}

#[test]
fn parallel() {
    let both = Parallel(Identity(Top), Identity(Unit));
    assert_eq!(Reduction::<F>::source(&both), Product(Top, Unit));
    let out = Reduction::<F>::prove(&both, vesta_sponge(0), &((), ()), &((), ())).unwrap();
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
