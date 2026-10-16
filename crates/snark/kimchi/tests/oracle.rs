//! Kimchi's Fiat-Shamir against Kimchi: the same scripts as the `oracles`
//! tool of the Kimchi figures notes (proof-systems `9227444890`) produce
//! the same values (`vectors/kimchi_oracles.txt`):
//!
//! - the base-field sponge over Vesta's base field: points absorbed, 128-bit
//!   challenges, scalars absorbed, a full base-field challenge absorbed
//!   back, the identity, the digests;
//! - the scalar-field sponge: absorptions, challenges, the digest;
//! - the endomorphism map of 128-bit challenges, with Vesta's `lambda`,
//!   which is Kimchi's `endo_r`.

use arrabbiata_curve::{Affine, Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::oracle::{FqSponge, FrSponge, ScalarChallenge};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// Kimchi's output.
const VECTORS: &str = include_str!("vectors/kimchi_oracles.txt");

/// A field element from 64 hex digits, big-endian.
fn element<F: PrimeField>(hex: &str) -> F {
    let limbs: Vec<u64> = (0..4)
        .rev()
        .map(|i| u64::from_str_radix(&hex[16 * i..16 * (i + 1)], 16).unwrap())
        .collect();
    F::from_limbs(&limbs).unwrap()
}

/// The lines whose tag is `tag`, in order, without the tag and the field.
fn lines<'a>(tag: &'a str) -> impl Iterator<Item = Vec<&'a str>> + 'a {
    VECTORS.lines().filter_map(move |l| {
        let words: Vec<&str> = l.split_whitespace().collect();
        (words[0] == tag).then(|| words[1..].to_vec())
    })
}

/// The last word of each line tagged `tag`, as an element.
fn values<F: PrimeField>(tag: &str) -> Vec<F> {
    lines(tag).map(|w| element(w[w.len() - 1])).collect()
}

fn points() -> Vec<Projective<Vesta>> {
    lines("point")
        .map(|w| {
            let x: Fq = element(w[1]);
            let y: Fq = element(w[2]);
            Affine::<Vesta>::new(x, y).unwrap().to_projective()
        })
        .collect()
}

#[test]
fn lambda_is_kimchis_endo_r() {
    assert_eq!(values::<Fp>("endo_r"), vec![Vesta::lambda()]);
}

#[test]
fn the_base_field_sponge_is_kimchis() {
    let p = points();
    let mut fq = FqSponge::<Vesta, _>::new(KimchiSponge::new(vesta_base()));
    let mut challenges = Vec::new();
    fq.absorb_g(&p[..2]);
    challenges.push(fq.challenge());
    fq.absorb_fr(&[Fp::from_u64(7), -Fp::ONE]);
    challenges.push(fq.challenge());
    let f = fq.challenge_fq();
    fq.absorb_fq(&[f]);
    challenges.push(fq.challenge());
    challenges.push(fq.challenge());
    fq.absorb_g(&[Projective::identity(), p[2]]);
    let digest = fq.clone().digest();
    let digest_fq = fq.digest_fq();
    assert_eq!(challenges, values::<Fp>("fq.challenge"));
    assert_eq!(vec![f], values::<Fq>("fq.challenge_fq"));
    assert_eq!(vec![digest], values::<Fp>("fq.digest"));
    assert_eq!(vec![digest_fq], values::<Fq>("fq.digest_fq"));
}

#[test]
fn the_scalar_field_sponge_is_kimchis() {
    let mut fr = FrSponge::<Fp, _>::new(KimchiSponge::new(pallas_base()));
    fr.absorb(Fp::from_u64(11));
    fr.absorb_multiple(&[Fp::from_u64(12), -Fp::from_u64(13)]);
    let first = fr.challenge();
    let second = fr.challenge();
    let digest = fr.digest();
    assert_eq!(vec![first.0, second.0], values::<Fp>("fr.challenge"));
    assert_eq!(
        vec![first.to_field(Vesta::lambda())],
        values::<Fp>("fr.challenge.to_field")
    );
    assert_eq!(vec![digest], values::<Fp>("fr.digest"));
}

#[test]
fn the_endomorphism_map_is_kimchis() {
    let cases: Vec<([u64; 2], Fp)> = lines("to_field")
        .map(|w| {
            let limbs = [
                u64::from_str_radix(w[0], 16).unwrap(),
                u64::from_str_radix(w[1], 16).unwrap(),
            ];
            (limbs, element(w[3]))
        })
        .collect();
    assert_eq!(cases.len(), 4);
    for (limbs, expected) in cases {
        let found = ScalarChallenge::<Fp>::from_limbs(limbs).to_field(Vesta::lambda());
        assert_eq!(found, expected, "{limbs:x?}");
    }
}

/// On Pallas the scalars (Fq) have the larger modulus, so `absorb_fr`
/// absorbs each scalar as its high bits then its low bit.
#[test]
fn absorbing_a_larger_scalar_is_kimchis() {
    let point = lines("pallas.point")
        .map(|w| {
            let x: Fp = element(w[1]);
            let y: Fp = element(w[2]);
            Affine::<Pallas>::new(x, y).unwrap().to_projective()
        })
        .next()
        .unwrap();
    let mut sponge = FqSponge::<Pallas, _>::new(KimchiSponge::new(pallas_base()));
    sponge.absorb_g(&[point]);
    sponge.absorb_fr(&[Fq::from_u64(7), -Fq::ONE, Fq::ONE]);
    let challenge = sponge.challenge();
    let digest = sponge.digest();
    assert_eq!(vec![challenge], values::<Fq>("pallas.challenge"));
    assert_eq!(vec![digest], values::<Fq>("pallas.digest"));
}
