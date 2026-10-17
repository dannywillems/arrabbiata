//! The laws of chunking (`arrabbiata_poly::chunk`): `split` is a linear
//! isomorphism `F[X]_{<m k} -> (F[X]_{<m})^k`, and the evaluations of the
//! two factors commute, for values and for any linear image (a commitment).

use arrabbiata_field::{Field, Goldilocks};
use arrabbiata_poly::{Chunking, UniPoly, combine};
use proptest::prelude::*;

type F = Goldilocks;

fn arb_f() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(F::from_u64)
}

/// A chunking and a polynomial it splits: `size` 1..8, `count` 1..6, and up
/// to `size * count` coefficients.
fn arb_chunked() -> impl Strategy<Value = (Chunking, UniPoly<F>)> {
    (1usize..8, 1usize..6).prop_flat_map(|(size, count)| {
        let chunking = Chunking::new(size, count).unwrap();
        prop::collection::vec(arb_f(), 0..=size * count)
            .prop_map(move |c| (chunking, UniPoly::new(c)))
    })
}

/// `x^k`.
fn pow(x: F, k: usize) -> F {
    (0..k).fold(F::ONE, |acc, _| acc * x)
}

/// A linear functional on `F[X]_{<m}`: `p -> sum_j w_j p_j`, the stand-in
/// for a commitment (a Pedersen commitment is such a map into a group).
fn functional(weights: &[F], p: &UniPoly<F>) -> F {
    weights
        .iter()
        .enumerate()
        .fold(F::ZERO, |acc, (j, w)| acc + *w * p.coefficient(j))
}

proptest! {
    #[test]
    fn split_is_a_bijection_onto_its_chunks((chunking, p) in arb_chunked()) {
        let chunks = chunking.split(&p).unwrap();
        prop_assert_eq!(chunks.len(), chunking.count());
        for chunk in &chunks {
            prop_assert_eq!(chunk.len(), chunking.size());
        }
        let joined: Vec<F> = chunks.iter().flat_map(|c| c.coefficients().to_vec()).collect();
        prop_assert_eq!(joined.len(), chunking.capacity());
        for (k, c) in joined.iter().enumerate() {
            prop_assert_eq!(*c, p.coefficient(k));
        }
    }

    #[test]
    fn evaluation_factors_through_the_chunks((chunking, p) in arb_chunked(), x in arb_f()) {
        let values = chunking.evaluate(&p, x).unwrap();
        let y = pow(x, chunking.size());
        prop_assert_eq!(combine(&values, y).unwrap(), p.evaluate(x));
    }

    #[test]
    fn linearize_is_the_evaluation_of_the_other_factor(
        (chunking, p) in arb_chunked(),
        x in arb_f(),
        y in arb_f(),
    ) {
        let linear = chunking.linearize(&p, y).unwrap();
        prop_assert!(linear.len() <= chunking.size());
        let values = chunking.evaluate(&p, x).unwrap();
        prop_assert_eq!(linear.evaluate(x), combine(&values, y).unwrap());
    }

    #[test]
    fn combine_commutes_with_a_linear_map(
        (chunking, p) in arb_chunked(),
        y in arb_f(),
        seed in prop::collection::vec(arb_f(), 8),
    ) {
        // L(linearize(p, y)) = combine(L(p_i), y): the square the verifier
        // relies on to build a commitment from the chunks' commitments.
        let weights = &seed[..chunking.size()];
        let images: Vec<F> = chunking
            .split(&p)
            .unwrap()
            .iter()
            .map(|chunk| functional(weights, chunk))
            .collect();
        let linear = chunking.linearize(&p, y).unwrap();
        prop_assert_eq!(functional(weights, &linear), combine(&images, y).unwrap());
    }

    #[test]
    fn split_is_linear(
        (chunking, p) in arb_chunked(),
        q in prop::collection::vec(arb_f(), 0..8),
        c in arb_f(),
    ) {
        let q: Vec<F> = q.into_iter().take(chunking.capacity()).collect();
        let q = UniPoly::new(q);
        let len = chunking.capacity();
        let sum = UniPoly::new((0..len).map(|k| p.coefficient(k) + c * q.coefficient(k)).collect());
        let (sp, sq, ss) = (
            chunking.split(&p).unwrap(),
            chunking.split(&q).unwrap(),
            chunking.split(&sum).unwrap(),
        );
        for i in 0..chunking.count() {
            for j in 0..chunking.size() {
                let expected = sp[i].coefficient(j) + c * sq[i].coefficient(j);
                prop_assert_eq!(ss[i].coefficient(j), expected);
            }
        }
    }

    #[test]
    fn too_large_a_degree_is_refused(size in 1usize..8, count in 1usize..6, c in arb_f()) {
        prop_assume!(!c.is_zero());
        let chunking = Chunking::new(size, count).unwrap();
        let mut coefficients = vec![F::ZERO; size * count];
        coefficients.push(c);
        let p = UniPoly::new(coefficients);
        prop_assert!(chunking.split(&p).is_none());
        prop_assert!(chunking.evaluate(&p, c).is_none());
        prop_assert!(chunking.linearize(&p, c).is_none());
    }
}

#[test]
fn trailing_zeros_do_not_count_against_the_capacity() {
    // Four stored coefficients, the last two zero: degree 1 fits two chunks
    // of one.
    let chunking = Chunking::new(1, 2).unwrap();
    let p = UniPoly::new(vec![F::ONE, F::ONE, F::ZERO, F::ZERO]);
    assert_eq!(chunking.split(&p).unwrap().len(), 2);
}

#[test]
fn a_small_polynomial_takes_every_chunk() {
    // The count is the protocol's: a constant split in three is the
    // constant, then two zero chunks, which combine to nothing.
    let chunking = Chunking::new(4, 3).unwrap();
    let p = UniPoly::constant(F::from_u64(7));
    let values = chunking.evaluate(&p, F::from_u64(5)).unwrap();
    assert_eq!(values, vec![F::from_u64(7), F::ZERO, F::ZERO]);
}
