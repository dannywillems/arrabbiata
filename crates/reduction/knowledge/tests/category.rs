//! The laws of the category of reductions, on a genuine reduction: the
//! 2-folding scheme of linear relations `A x = y` (`y1 + r y2`,
//! `x1 + r x2`), which is 2-special-sound.
//!
//! - completeness of the fold, of compositions and of products;
//! - unit laws: `Then(Identity, A)` and `Then(A, Identity)` verify to the
//!   instance of `A`;
//! - associativity: `Then(Then(A, B), C)` and `Then(A, Then(B, C))` verify
//!   to the same instance;
//! - a pair outside the source is reported as such, on the right side;
//! - lists: [`Each`] is a functor (it preserves identities and
//!   composition), [`Concat`] is associative with the empty list as unit,
//!   [`Single`] is a list of one, and a list outside `R^*` names its index.

use arrabbiata_field::Field;
use arrabbiata_pasta::Fq as F;
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_reduction::{
    Batch, Concat, Each, Either, Identity, Indexed, Output, OutputOf, Parallel, Product, Reduction,
    Relation, Single, Then, laws,
};
use arrabbiata_transcript::{
    Absorb, Transcript,
    typed::{Challenge, End, Label, Public, Tail, Tape},
};
use proptest::prelude::*;

/// `A x = y` for a fixed matrix `A` with `rows` rows: instance `y`,
/// witness `x`.
#[derive(Clone, Debug, PartialEq)]
struct Linear(Vec<Vec<F>>);

impl Relation for Linear {
    type Instance = Vec<F>;
    type Witness = Vec<F>;
    type Error = ();

    fn check(&self, y: &Vec<F>, x: &Vec<F>) -> Result<(), ()> {
        let ok = y.len() == self.0.len()
            && self.0.iter().zip(y).all(|(row, yi)| {
                row.len() == x.len()
                    && row.iter().zip(x).fold(F::ZERO, |s, (a, b)| s + *a * *b) == *yi
            });
        if ok { Ok(()) } else { Err(()) }
    }
}

/// The fold of two linear instances.
#[derive(Clone)]
struct Fold(Linear);

fn line(a: &[F], b: &[F], r: F) -> Vec<F> {
    a.iter().zip(b).map(|(x, y)| *x + r * *y).collect()
}

/// The two instances, as the public value the fold's challenge binds.
struct Instances(Vec<F>, Vec<F>);

impl Label for Instances {
    const LABEL: u64 = 0x6c69_6e65;
}

impl Absorb<F> for Instances {
    fn absorb_into<T: Transcript<F>>(&self, t: &mut T) {
        for v in self.0.iter().chain(&self.1) {
            t.absorb(*v);
        }
    }
}

/// The fold's tape: the instances, then the challenge.
type FoldPattern = Public<Instances, Challenge<End>>;

/// The challenge, after the instances, on `tape`.
fn challenge<T: Transcript<F>, R>(
    tape: Tape<F, T, Tail<FoldPattern, R>>,
    y: &(Vec<F>, Vec<F>),
) -> (F, Tape<F, T, R>) {
    tape.public(&Instances(y.0.clone(), y.1.clone()))
        .challenge()
}

impl Reduction<F> for Fold {
    type Source = Product<Linear, Linear>;
    type Target = Linear;
    type Proof = ();
    type Error = ();
    type Pattern = FoldPattern;

    fn source(&self) -> Self::Source {
        Product(self.0.clone(), self.0.clone())
    }

    fn target(&self) -> Linear {
        self.0.clone()
    }

    fn prove<T: Transcript<F>, R>(
        &self,
        tape: Tape<F, T, Tail<FoldPattern, R>>,
        y: &(Vec<F>, Vec<F>),
        x: &(Vec<F>, Vec<F>),
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, ()> {
        let (r, tape) = challenge(tape, y);
        Ok(Output {
            instance: line(&y.0, &y.1, r),
            witness: line(&x.0, &x.1, r),
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T: Transcript<F>, R>(
        &self,
        tape: Tape<F, T, Tail<FoldPattern, R>>,
        y: &(Vec<F>, Vec<F>),
        (): (),
    ) -> Result<(Vec<F>, Tape<F, T, R>), ()> {
        let (r, tape) = challenge(tape, y);
        Ok((line(&y.0, &y.1, r), tape))
    }
}

/// `R -> R x R`: duplicate a pair.
#[derive(Clone)]
struct Duplicate(Linear);

impl Reduction<F> for Duplicate {
    type Source = Linear;
    type Target = Product<Linear, Linear>;
    type Proof = ();
    type Error = ();
    type Pattern = End;

    fn source(&self) -> Linear {
        self.0.clone()
    }

    fn target(&self) -> Self::Target {
        Product(self.0.clone(), self.0.clone())
    }

    fn prove<T: Transcript<F>, R>(
        &self,
        tape: Tape<F, T, R>,
        y: &Vec<F>,
        x: &Vec<F>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, ()> {
        Ok(Output {
            instance: (y.clone(), y.clone()),
            witness: (x.clone(), x.clone()),
            proof: (),
            transcript: tape,
        })
    }

    fn verify<T: Transcript<F>, R>(
        &self,
        tape: Tape<F, T, R>,
        y: &Vec<F>,
        (): (),
    ) -> Result<((Vec<F>, Vec<F>), Tape<F, T, R>), ()> {
        Ok(((y.clone(), y.clone()), tape))
    }
}

fn arb_f() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(F::from_u64)
}

/// A matrix and two witnesses, with their instances.
fn arb_statement() -> impl Strategy<Value = (Linear, Vec<F>, Vec<F>)> {
    (1usize..4, 1usize..5).prop_flat_map(|(rows, cols)| {
        (
            prop::collection::vec(prop::collection::vec(arb_f(), cols), rows).prop_map(Linear),
            prop::collection::vec(arb_f(), cols),
            prop::collection::vec(arb_f(), cols),
        )
    })
}

fn apply(a: &Linear, x: &[F]) -> Vec<F> {
    a.0.iter()
        .map(|row| row.iter().zip(x).fold(F::ZERO, |s, (a, b)| s + *a * *b))
        .collect()
}

proptest! {
    #[test]
    fn fold_is_complete((a, x1, x2) in arb_statement(), domain in any::<u64>()) {
        let (y1, y2) = (apply(&a, &x1), apply(&a, &x2));
        let out = laws::complete::<F, _, _>(&Fold(a), || vesta_sponge(domain), &(y1, y2), &(x1, x2));
        prop_assert!(out.is_ok());
    }

    #[test]
    fn composition_and_products_are_complete((a, x1, x2) in arb_statement()) {
        let (y1, y2) = (apply(&a, &x1), apply(&a, &x2));
        // R x R -> R x R -> R.
        let par = Then(Parallel(Identity(a.clone()), Identity(a.clone())), Fold(a.clone()));
        prop_assert!(laws::complete::<F, _, _>(&par, || vesta_sponge(0), &(y1.clone(), y2), &(x1.clone(), x2)).is_ok());
        // R -> R x R -> R, folding a pair with itself.
        let dup = Then(Duplicate(a.clone()), Fold(a));
        prop_assert!(laws::complete::<F, _, _>(&dup, || vesta_sponge(0), &y1, &x1).is_ok());
    }

    #[test]
    fn identity_is_a_unit((a, x1, x2) in arb_statement()) {
        let y = (apply(&a, &x1), apply(&a, &x2));
        let fold = Fold(a.clone());
        let left = Then(Identity(Product(a.clone(), a.clone())), fold.clone());
        let right = Then(fold.clone(), Identity(a));
        prop_assert!(laws::same_instance::<F, _, _, _>(&fold, &left, || vesta_sponge(1), &y, (), ((), ())).unwrap());
        prop_assert!(laws::same_instance::<F, _, _, _>(&fold, &right, || vesta_sponge(1), &y, (), ((), ())).unwrap());
    }

    #[test]
    fn composition_is_associative((a, x1, _x2) in arb_statement()) {
        let y = apply(&a, &x1);
        let (d, f) = (Duplicate(a.clone()), Fold(a.clone()));
        let i = Identity(a);
        let left = Then(Then(d.clone(), f.clone()), i.clone());
        let right = Then(d, Then(f, i));
        prop_assert!(laws::same_instance::<F, _, _, _>(&left, &right, || vesta_sponge(2), &y, (((), ()), ()), ((), ((), ()))).unwrap());
    }

    #[test]
    fn a_pair_outside_the_source_is_reported((a, x1, x2) in arb_statement()) {
        let (y1, mut y2) = (apply(&a, &x1), apply(&a, &x2));
        y2[0] = y2[0] + F::ONE;
        let source = Product(a.clone(), a.clone());
        prop_assert_eq!(source.check(&(y1.clone(), y2.clone()), &(x1.clone(), x2.clone())), Err(Either::Right(())));
        let out = laws::complete::<F, _, _>(&Fold(a), || vesta_sponge(0), &(y1, y2), &(x1, x2));
        prop_assert!(matches!(out, Err(laws::Failure::Source(Either::Right(())))));
    }
}

/// Witnesses and their instances, for a list.
fn arb_list() -> impl Strategy<Value = (Linear, Vec<Vec<F>>)> {
    (1usize..4, 1usize..5, 0usize..4).prop_flat_map(|(rows, cols, len)| {
        (
            prop::collection::vec(prop::collection::vec(arb_f(), cols), rows).prop_map(Linear),
            prop::collection::vec(prop::collection::vec(arb_f(), cols), len),
        )
    })
}

proptest! {
    #[test]
    fn each_is_complete((a, xs) in arb_list()) {
        let ys: Vec<Vec<F>> = xs.iter().map(|x| apply(&a, x)).collect();
        // R^* -> (R x R)^* -> R^*: duplicate then fold each element.
        let each = Then(Each(Duplicate(a.clone())), Each(Fold(a)));
        prop_assert!(laws::complete::<F, _, _>(&each, || vesta_sponge(3), &ys, &xs).is_ok());
    }

    #[test]
    fn each_is_a_functor((a, xs) in arb_list()) {
        let ys: Vec<Vec<F>> = xs.iter().map(|x| apply(&a, x)).collect();
        let (d, f) = (Duplicate(a.clone()), Fold(a.clone()));
        // Each(Then(D, F)) = Then(Each(D), Each(F)), on the instances. The
        // two are not one protocol: their patterns are (D;F)* and D*;F*,
        // different languages of traces, so their described tapes
        // diverge. The law is the reductions', not the transcripts': here
        // D takes no step, so on undescribed tapes both absorb the same
        // sequence and the instances agree.
        let whole = Each(Then(d.clone(), f.clone()));
        let parts = Then(Each(d), Each(f));
        let whole_proof = vec![((), ()); ys.len()];
        let parts_proof = (vec![(); ys.len()], vec![(); ys.len()]);
        let (left, _) = whole.verify::<_, End>(Tape::undescribed(vesta_sponge(4)), &ys, whole_proof).unwrap();
        let (right, _) = parts.verify::<_, End>(Tape::undescribed(vesta_sponge(4)), &ys, parts_proof).unwrap();
        prop_assert_eq!(left, right);
        // Each(Identity) = Identity.
        let each_identity = Each(Identity(a.clone()));
        let identity = Identity(Batch(a));
        prop_assert!(laws::same_instance::<F, _, _, _>(&each_identity, &identity, || vesta_sponge(4), &ys, vec![(); ys.len()], ()).unwrap());
    }

    #[test]
    fn concat_is_a_monoid((a, xs) in arb_list(), cut in any::<usize>()) {
        let ys: Vec<Vec<F>> = xs.iter().map(|x| apply(&a, x)).collect();
        let cut = if ys.is_empty() { 0 } else { cut % (ys.len() + 1) };
        let (left, right) = (ys[..cut].to_vec(), ys[cut..].to_vec());
        let concat = Concat(a.clone());
        let (joined, _) = concat.verify_on(vesta_sponge(5), &(left.clone(), right.clone()), ()).unwrap();
        prop_assert_eq!(&joined, &ys);
        // The empty list is a unit on both sides.
        let (unit_left, _) = concat.verify_on(vesta_sponge(5), &(vec![], ys.clone()), ()).unwrap();
        let (unit_right, _) = concat.verify_on(vesta_sponge(5), &(ys.clone(), vec![]), ()).unwrap();
        prop_assert_eq!(&unit_left, &ys);
        prop_assert_eq!(&unit_right, &ys);
        // Associativity: (l ++ m) ++ r = l ++ (m ++ r), with m a singleton.
        let witness_left = xs[..cut].to_vec();
        let witness_right = xs[cut..].to_vec();
        let out = laws::complete::<F, _, _>(&concat, || vesta_sponge(5), &(left.clone(), right.clone()), &(witness_left, witness_right));
        prop_assert!(out.is_ok());
        if let Some(first) = ys.first() {
            let (single, _) = Single(a.clone()).verify_on(vesta_sponge(5), first, ()).unwrap();
            prop_assert_eq!(&single, &vec![first.clone()]);
            let rest = ys[1..].to_vec();
            let (lm, _) = concat.verify_on(vesta_sponge(5), &(single.clone(), rest[..0].to_vec()), ()).unwrap();
            let (lm_r, _) = concat.verify_on(vesta_sponge(5), &(lm, rest.clone()), ()).unwrap();
            let (m_r, _) = concat.verify_on(vesta_sponge(5), &(rest[..0].to_vec(), rest), ()).unwrap();
            let (l_mr, _) = concat.verify_on(vesta_sponge(5), &(single, m_r), ()).unwrap();
            prop_assert_eq!(lm_r, l_mr);
        }
    }

    #[test]
    fn a_list_outside_the_power_names_its_index((a, xs) in arb_list(), which in any::<usize>()) {
        prop_assume!(!xs.is_empty());
        let mut ys: Vec<Vec<F>> = xs.iter().map(|x| apply(&a, x)).collect();
        let which = which % ys.len();
        ys[which][0] = ys[which][0] + F::ONE;
        prop_assert_eq!(Batch(a.clone()).check(&ys, &xs), Err(Indexed::At(which, ())));
        prop_assert_eq!(Batch(a).check(&ys, &xs[1..].to_vec()), Err(Indexed::Length));
    }
}
