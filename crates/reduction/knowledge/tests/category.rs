//! The laws of the category of reductions, on a genuine reduction: the
//! 2-folding scheme of linear relations `A x = y` (`y1 + r y2`,
//! `x1 + r x2`), which is 2-special-sound.
//!
//! - completeness of the fold, of compositions and of products;
//! - unit laws: `Then(Identity, A)` and `Then(A, Identity)` verify to the
//!   instance of `A`;
//! - associativity: `Then(Then(A, B), C)` and `Then(A, Then(B, C))` verify
//!   to the same instance;
//! - a pair outside the source is reported as such, on the right side.

use arrabbiata_field::Field;
use arrabbiata_pasta::Fq as F;
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_reduction::{
    Either, Identity, Output, OutputOf, Parallel, Product, Reduction, Relation, Then, laws,
};
use arrabbiata_transcript::Transcript;
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

fn challenge<T: Transcript<F>>(t: &mut T, y: &(Vec<F>, Vec<F>)) -> F {
    for v in y.0.iter().chain(&y.1) {
        t.absorb(*v);
    }
    t.squeeze()
}

impl Reduction<F> for Fold {
    type Source = Product<Linear, Linear>;
    type Target = Linear;
    type Proof = ();
    type Error = ();

    fn source(&self) -> Self::Source {
        Product(self.0.clone(), self.0.clone())
    }

    fn target(&self) -> Linear {
        self.0.clone()
    }

    fn prove<T: Transcript<F>>(
        &self,
        mut t: T,
        y: &(Vec<F>, Vec<F>),
        x: &(Vec<F>, Vec<F>),
    ) -> Result<OutputOf<F, Self, T>, ()> {
        let r = challenge(&mut t, y);
        Ok(Output {
            instance: line(&y.0, &y.1, r),
            witness: line(&x.0, &x.1, r),
            proof: (),
            transcript: t,
        })
    }

    fn verify<T: Transcript<F>>(
        &self,
        mut t: T,
        y: &(Vec<F>, Vec<F>),
        (): (),
    ) -> Result<(Vec<F>, T), ()> {
        let r = challenge(&mut t, y);
        Ok((line(&y.0, &y.1, r), t))
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

    fn source(&self) -> Linear {
        self.0.clone()
    }

    fn target(&self) -> Self::Target {
        Product(self.0.clone(), self.0.clone())
    }

    fn prove<T: Transcript<F>>(
        &self,
        t: T,
        y: &Vec<F>,
        x: &Vec<F>,
    ) -> Result<OutputOf<F, Self, T>, ()> {
        Ok(Output {
            instance: (y.clone(), y.clone()),
            witness: (x.clone(), x.clone()),
            proof: (),
            transcript: t,
        })
    }

    fn verify<T: Transcript<F>>(
        &self,
        t: T,
        y: &Vec<F>,
        (): (),
    ) -> Result<((Vec<F>, Vec<F>), T), ()> {
        Ok(((y.clone(), y.clone()), t))
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
