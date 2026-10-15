//! The behaviour of each public item of the challenge module, one test per
//! item.

use arrabbiata_commit::PolynomialCommitment;
use arrabbiata_field::Field;
use arrabbiata_ipa::{
    Ipa, RecursionChallenge, Statement, b_poly, b_poly_coefficients, combined_inner_product,
    evaluation_vector,
};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};

#[test]
fn b_poly_works() {
    // No rounds: the constant 1. One round with challenge 2: 1 + 2 x.
    assert_eq!(b_poly(&[], Fq::from_u64(5)), Fq::ONE);
    assert_eq!(
        b_poly(&[Fq::from_u64(2)], Fq::from_u64(5)),
        Fq::from_u64(11)
    );
}

#[test]
fn b_poly_coefficients_works() {
    // Two rounds (2, 3): (1 + 2 x^2)(1 + 3 x) = 1 + 3 x + 2 x^2 + 6 x^3.
    let chals = [Fq::from_u64(2), Fq::from_u64(3)];
    let expected = [1u64, 3, 2, 6].map(Fq::from_u64);
    assert_eq!(b_poly_coefficients(&chals), expected.to_vec());
}

#[test]
fn combined_inner_product_works() {
    // One polynomial at two points: v_0 + evalscale v_1.
    let values = vec![vec![Fq::from_u64(4), Fq::from_u64(5)]];
    let polyscale = Fq::from_u64(7);
    let evalscale = Fq::from_u64(3);
    assert_eq!(
        combined_inner_product(polyscale, evalscale, &values),
        Fq::from_u64(19)
    );
}

#[test]
fn evaluation_vector_works() {
    // Points 2 and 3, scale 10, length 3: (1 + 10, 2 + 30, 4 + 90).
    let points = [Fq::from_u64(2), Fq::from_u64(3)];
    let evalscale = Fq::from_u64(10);
    let expected = [11u64, 32, 94].map(Fq::from_u64);
    assert_eq!(evaluation_vector(&points, evalscale, 3), expected.to_vec());
}

#[test]
fn recursion_challenge() {
    // An accumulator evaluates its challenge polynomial in O(k), and its
    // coefficients are the witness of its claim.
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let chals = vec![Fq::from_u64(2)];
    let comm = ipa.commit(&b_poly_coefficients(&chals));
    let accumulator = RecursionChallenge::<Pallas> { chals, comm };
    assert_eq!(accumulator.evaluate(Fq::from_u64(5)), Fq::from_u64(11));
    assert_eq!(accumulator.coefficients(), vec![Fq::ONE, Fq::from_u64(2)]);
    assert!(ipa.decide(&accumulator));
}

#[test]
fn statement_prove_succinct_decide() {
    // Open 1 + x at 2 and 3 with scale 1: the combined value is 3 + 4.
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let p = [Fq::ONE, Fq::ONE];
    let commitment = ipa.commit(&p);
    let points = [Fq::from_u64(2), Fq::from_u64(3)];
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale: Fq::ONE,
        value: Fq::from_u64(7),
    };
    let (proof, accumulator) = ipa.prove(&mut vesta_sponge(2), &statement, &p);
    assert_eq!(ipa.rounds(), 2);
    assert_eq!(
        ipa.succinct(&mut vesta_sponge(2), &statement, &proof),
        Some(accumulator.clone())
    );
    assert!(ipa.decide(&accumulator));
    assert!(ipa.decide_all(&[accumulator.clone(), accumulator], Fq::from_u64(9)));
}
