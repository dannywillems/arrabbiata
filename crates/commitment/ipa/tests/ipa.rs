//! The IPA commitment on Pallas, with Poseidon transcripts.
//!
//! Besides the scheme's laws: the challenge polynomial `b` is the
//! polynomial with coefficients `s`; an opening at several points combined
//! by `evalscale` is complete and sound; the verifier is its succinct half
//! then the decider, `sg` being the only thing the succinct half trusts;
//! and `decide_all` is `decide` for each, with one MSM.

use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment};
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_ipa::{
    Ipa, IpaProof, RecursionChallenge, Statement, b_poly, b_poly_coefficients, evaluation_vector,
};
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

    /// Tampering with any round point, the final scalar or `sg` is rejected.
    #[test]
    fn tampered_proof(p in arb_poly(), z in arb_f(), round in 0usize..4, side in any::<bool>()) {
        let c = PolynomialCommitment::commit(ipa(), &p);
        let (v, proof) = ipa().open(&p, &z);
        let g = Projective::<Pallas>::generator();
        let mut rounds = proof.rounds().to_vec();
        if side { rounds[round].0 = rounds[round].0 + g } else { rounds[round].1 = rounds[round].1 + g }
        prop_assert!(!ipa().verify(&c, &z, v, &IpaProof::from_parts(rounds, proof.scalar(), proof.sg())));
        let bumped = IpaProof::from_parts(proof.rounds().to_vec(), proof.scalar() + Fq::ONE, proof.sg());
        prop_assert!(!ipa().verify(&c, &z, v, &bumped));
        let moved = IpaProof::from_parts(proof.rounds().to_vec(), proof.scalar(), proof.sg() + g);
        prop_assert!(!ipa().verify(&c, &z, v, &moved));
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
        &IpaProof::from_parts(rounds, proof.scalar(), proof.sg())
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

fn arb_points() -> impl Strategy<Value = Vec<Fq>> {
    prop::collection::vec(arb_f(), 1..4)
}

/// The value of `p` on the combined evaluation vector.
fn combined(p: &[Fq], points: &[Fq], evalscale: Fq) -> Fq {
    points
        .iter()
        .rev()
        .fold(Fq::ZERO, |acc, x| acc * evalscale + eval(p, *x))
}

/// An opening of `p` at `points`: its statement's parts and the proof.
fn open_points(
    p: &[Fq],
    points: &[Fq],
    evalscale: Fq,
) -> (Projective<Pallas>, Fq, IpaProof<Pallas>) {
    let commitment = PolynomialCommitment::commit(ipa(), p);
    let value = combined(p, points, evalscale);
    let statement = Statement {
        commitment: &commitment,
        points,
        evalscale,
        value,
    };
    let (proof, accumulator) = ipa().prove(&mut vesta_sponge(2), &statement, p);
    assert_eq!(accumulator.comm, proof.sg());
    (commitment, value, proof)
}

fn succinct(
    commitment: &Projective<Pallas>,
    points: &[Fq],
    evalscale: Fq,
    value: Fq,
    proof: &IpaProof<Pallas>,
) -> Option<RecursionChallenge<Pallas>> {
    let statement = Statement {
        commitment,
        points,
        evalscale,
        value,
    };
    ipa().succinct(&mut vesta_sponge(2), &statement, proof)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// `b(x) = sum_i s_i x^i`, and `<s, (1, x, x^2, ...)>` is the same
    /// value: the challenge polynomial's two descriptions agree.
    #[test]
    fn b_poly_is_its_coefficients(chals in prop::collection::vec(arb_f(), 0..6), x in arb_f()) {
        let s = b_poly_coefficients(&chals);
        prop_assert_eq!(s.len(), 1 << chals.len());
        prop_assert_eq!(b_poly(&chals, x), eval(&s, x));
    }

    /// The evaluation vector is linear: its inner product with `p` is the
    /// combination of the values of `p`.
    #[test]
    fn evaluation_vector_combines_the_values(p in arb_poly(), points in arb_points(), evalscale in arb_f()) {
        let b = evaluation_vector(&points, evalscale, 16);
        let inner = p.iter().zip(&b).fold(Fq::ZERO, |acc, (a, b)| acc + *a * *b);
        prop_assert_eq!(inner, combined(&p, &points, evalscale));
    }

    /// An opening at several points is complete: the succinct half accepts
    /// it, and its accumulator is decided true; `sg` is the commitment to
    /// the challenge polynomial.
    #[test]
    fn multi_point_completeness(p in arb_poly(), points in arb_points(), evalscale in arb_f()) {
        let (c, v, proof) = open_points(&p, &points, evalscale);
        let accumulator = succinct(&c, &points, evalscale, v, &proof);
        prop_assert!(accumulator.is_some());
        let accumulator = accumulator.unwrap();
        // The prover's accumulator is the verifier's.
        let statement = Statement { commitment: &c, points: &points, evalscale, value: v };
        let (_, proved) = ipa().prove(&mut vesta_sponge(2), &statement, &p);
        prop_assert_eq!(&proved, &accumulator);
        prop_assert_eq!(accumulator.comm, proof.sg());
        prop_assert_eq!(accumulator.comm, PolynomialCommitment::commit(ipa(), &accumulator.coefficients()));
        prop_assert!(ipa().decide(&accumulator));
    }

    /// A wrong combined value, a wrong point or a wrong scale is rejected
    /// by the succinct half.
    #[test]
    fn multi_point_soundness(p in arb_poly(), points in arb_points(), evalscale in arb_f(), delta in 1u64..1000) {
        let d = Fq::from_u64(delta);
        let (c, v, proof) = open_points(&p, &points, evalscale);
        prop_assert!(succinct(&c, &points, evalscale, v + d, &proof).is_none());
        // Moving a point or the scale is caught when it makes the claim
        // false (the zero polynomial takes the same value everywhere).
        let mut moved = points.clone();
        moved[0] = moved[0] + d;
        if combined(&p, &moved, evalscale) != v {
            prop_assert!(succinct(&c, &moved, evalscale, v, &proof).is_none());
        }
        if combined(&p, &points, evalscale + d) != v {
            prop_assert!(succinct(&c, &points, evalscale + d, v, &proof).is_none());
        }
    }

    /// The decider catches a wrong `comm`; and `sg` is not free in the
    /// succinct check either, which binds it through `a sg`.
    #[test]
    fn the_decider_checks_sg(p in arb_poly(), z in arb_f()) {
        let (c, v, proof) = open_points(&p, &[z], Fq::ONE);
        let accumulator = succinct(&c, &[z], Fq::ONE, v, &proof).unwrap();
        let g = Projective::<Pallas>::generator();
        let wrong = RecursionChallenge { chals: accumulator.chals.clone(), comm: accumulator.comm + g };
        prop_assert!(!ipa().decide(&wrong));
        // `sg` enters the check as `a sg`: with `a = 0` (the zero
        // polynomial) it is not used at all.
        if proof.scalar() != Fq::ZERO {
            let moved = IpaProof::from_parts(proof.rounds().to_vec(), proof.scalar(), proof.sg() + g);
            prop_assert!(succinct(&c, &[z], Fq::ONE, v, &moved).is_none());
        }
    }

    /// `decide_all` with one MSM agrees with `decide` on each: it accepts
    /// true accumulators and rejects a list with one false.
    #[test]
    fn decide_all_is_decide_for_each(ps in prop::collection::vec(arb_poly(), 1..4), z in arb_f(), r in arb_f(), which in any::<usize>()) {
        let mut accumulators = Vec::new();
        for p in &ps {
            let (c, v, proof) = open_points(p, &[z], Fq::ONE);
            accumulators.push(succinct(&c, &[z], Fq::ONE, v, &proof).unwrap());
        }
        prop_assert!(ipa().decide_all(&accumulators, r));
        let which = which % accumulators.len();
        accumulators[which].comm = accumulators[which].comm + Projective::<Pallas>::generator();
        // A false accumulator passes only for the few `r` that are roots;
        // a random `r` is not one.
        prop_assume!(r != Fq::ZERO || which == 0);
        prop_assert!(!ipa().decide_all(&accumulators, r));
    }
}
