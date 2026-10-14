//! The IPA commitment on Pallas, with Poseidon transcripts.

use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment};
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_ipa::{Ipa, IpaProof};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;
use std::sync::OnceLock;

type Scheme = Ipa<Pallas, PoseidonSponge<Fq>>;

fn ipa() -> &'static Scheme {
    static IPA: OnceLock<Scheme> = OnceLock::new();
    IPA.get_or_init(|| Ipa::setup(&mut pallas_sponge(0), 16, vesta_sponge(1)))
}

fn arb_f() -> impl Strategy<Value = Fq> {
    prop_oneof![
        Just(Fq::ZERO),
        Just(Fq::ONE),
        any::<[u64; 4]>().prop_map(|l| {
            l.iter().fold(Fq::ZERO, |acc, x| {
                acc * Fq::from_u64(u64::MAX) + Fq::from_u64(*x)
            })
        }),
    ]
}

fn arb_poly() -> impl Strategy<Value = Vec<Fq>> {
    prop::collection::vec(arb_f(), 0..=16)
}

fn eval(p: &[Fq], x: Fq) -> Fq {
    p.iter().rev().fold(Fq::ZERO, |acc, c| acc * x + *c)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Completeness: an honest opening verifies, at the polynomial's value.
    #[test]
    fn completeness(p in arb_poly(), z in arb_f()) {
        let c = PolynomialCommitment::commit(ipa(), &p);
        let (v, proof) = ipa().open(&p, &z);
        prop_assert_eq!(v, eval(&p, z));
        prop_assert!(ipa().verify(&c, &z, v, &proof));
    }

    /// Evaluation binding: the proof does not verify for another value,
    /// another point at which the polynomial takes another value, or another
    /// commitment. (For the zero polynomial every point is a true statement.)
    #[test]
    fn binding(p in arb_poly(), z in arb_f(), delta in 1u64..1000) {
        let d = Fq::from_u64(delta);
        let c = PolynomialCommitment::commit(ipa(), &p);
        let (v, proof) = ipa().open(&p, &z);
        prop_assert!(!ipa().verify(&c, &z, v + d, &proof));
        if eval(&p, z + d) != v {
            prop_assert!(!ipa().verify(&c, &(z + d), v, &proof));
        }
        let other = c + Projective::<Pallas>::generator();
        prop_assert!(!ipa().verify(&other, &z, v, &proof));
    }

    /// Tampering with any round point or the final scalar is rejected.
    #[test]
    fn tampered_proof(p in arb_poly(), z in arb_f(), round in 0usize..4, side in any::<bool>()) {
        let c = PolynomialCommitment::commit(ipa(), &p);
        let (v, proof) = ipa().open(&p, &z);
        let g = Projective::<Pallas>::generator();
        let mut rounds = proof.rounds().to_vec();
        if side { rounds[round].0 = rounds[round].0 + g } else { rounds[round].1 = rounds[round].1 + g }
        prop_assert!(!ipa().verify(&c, &z, v, &IpaProof::from_parts(rounds, proof.scalar())));
        let bumped = IpaProof::from_parts(proof.rounds().to_vec(), proof.scalar() + Fq::ONE);
        prop_assert!(!ipa().verify(&c, &z, v, &bumped));
    }

    /// The commitment is a module morphism.
    #[test]
    fn commitment_is_a_morphism(a in prop::collection::vec(arb_f(), 16), b in prop::collection::vec(arb_f(), 16), s in arb_f()) {
        let folded: Vec<Fq> = a.iter().zip(&b).map(|(x, y)| *x + s * *y).collect();
        let ck = ipa();
        prop_assert_eq!(
            HomomorphicCommitment::commit(ck, &folded),
            ck.add(&HomomorphicCommitment::commit(ck, &a), &ck.scale(&HomomorphicCommitment::commit(ck, &b), s))
        );
    }
}

#[test]
fn wrong_number_of_rounds_is_rejected() {
    let p = [Fq::ONE, Fq::ONE];
    let c = PolynomialCommitment::commit(ipa(), &p);
    let (v, proof) = ipa().open(&p, &Fq::ONE);
    let mut rounds = proof.rounds().to_vec();
    rounds.pop();
    assert!(!ipa().verify(
        &c,
        &Fq::ONE,
        v,
        &IpaProof::from_parts(rounds, proof.scalar())
    ));
}

#[test]
fn both_traits_commit_identically() {
    let p = [Fq::from_u64(3), Fq::from_u64(4)];
    assert_eq!(
        PolynomialCommitment::commit(ipa(), &p),
        HomomorphicCommitment::commit(ipa(), &p)
    );
}

#[test]
#[should_panic(expected = "more coefficients than generators")]
fn too_many_coefficients() {
    let _ = PolynomialCommitment::commit(ipa(), &[Fq::ONE; 17]);
}
