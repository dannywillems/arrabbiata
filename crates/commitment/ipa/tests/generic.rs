//! The laws of the generic IPA, for every combination of its parameters
//! that runs on a scalar-field oracle (binding × challenges × hiding; the
//! hashed value base needs a base-field oracle, and is tested with
//! Kimchi's in `arrabbiata-kimchi`, bit for bit against Kimchi):
//!
//! - completeness: the verifier's accumulator is the prover's, and the
//!   decider accepts it;
//! - soundness of the opening: a wrong value, a wrong polynomial, a
//!   tampered round, a tampered final step or a tampered `sg` is rejected
//!   (by the succinct check, or by the decider for `sg`);
//! - the instances agree with the scheme of the crate: `Ipa` is the
//!   (BindStatement, Full, ScaledGenerator, NotHiding) instance.

use arrabbiata_curve::{Curve, Projective, glv::Glv, msm, points_from_transcript};
use arrabbiata_field::{
    Field,
    rng::{Arkworks, SplitMix64},
};
use arrabbiata_ipa::{
    Ipa, Statement,
    generic::{
        BindStatement, BindValue, Binding, Blinded, ChallengeMap, Endo, Full, GenericIpa, Hiding,
        NotHiding, Opening, ScalarOracle, ScaledGenerator, ValueBase, Witness,
    },
};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};

type F = <Pallas as Curve>::Scalar;

/// The scheme on 16 generators, `u` and `h` from Poseidon over the base
/// field.
fn scheme<B, Ch, H>(
    binding: B,
    challenges: Ch,
    hiding: H,
) -> GenericIpa<Pallas, B, Ch, ScaledGenerator, H> {
    let mut points = points_from_transcript::<Pallas, _>(&mut pallas_sponge(3), 18);
    let h = points.pop().unwrap();
    let u = points.pop().unwrap();
    GenericIpa {
        generators: points,
        u,
        h,
        binding,
        challenges,
        value_base: ScaledGenerator,
        hiding,
    }
}

/// `p(x)`.
fn eval(p: &[F], x: F) -> F {
    p.iter().rev().fold(F::ZERO, |acc, c| acc * x + *c)
}

/// The laws, for one instance.
fn laws<B, Ch, H>(ipa: &GenericIpa<Pallas, B, Ch, ScaledGenerator, H>, hiding: bool)
where
    B: for<'a> Binding<Pallas, ScalarOracle<&'a mut PoseidonSponge<Fq>>>,
    ScaledGenerator: for<'a> ValueBase<Pallas, ScalarOracle<&'a mut PoseidonSponge<Fq>>>,
    Ch: ChallengeMap<F>,
    H: Hiding<Pallas>,
{
    let p: Vec<F> = (1..=11u64).map(F::from_u64).collect();
    let points = [F::from_u64(5), F::from_u64(7)];
    let evalscale = F::from_u64(3);
    let value = eval(&p, points[0]) + evalscale * eval(&p, points[1]);
    let blinder = if hiding { F::from_u64(13) } else { F::ZERO };
    let commitment = msm(&ipa.generators, &p) + ipa.h * blinder;
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale,
        value,
    };
    let witness = Witness {
        coefficients: &p,
        blinder,
    };
    let prove = |statement: &Statement<'_, Pallas>, witness: &Witness<'_, F>| {
        let mut transcript = vesta_sponge(1);
        let mut rng = Arkworks(SplitMix64(17));
        ipa.prove(
            &mut ScalarOracle(&mut transcript),
            statement,
            witness,
            &mut rng,
        )
    };
    let check = |statement: &Statement<'_, Pallas>, opening: &Opening<Pallas, H::Final>| {
        let mut transcript = vesta_sponge(1);
        ipa.succinct(&mut ScalarOracle(&mut transcript), statement, opening)
    };
    let (opening, accumulator) = prove(&statement, &witness);
    assert_eq!(opening.rounds.len(), 4);
    // Completeness.
    assert_eq!(check(&statement, &opening), Some(accumulator.clone()));
    assert!(ipa.decide(&accumulator));
    // A wrong value.
    let wrong = Statement {
        value: value + F::ONE,
        ..statement
    };
    assert_eq!(check(&wrong, &opening), None);
    // A wrong polynomial for the statement.
    let other: Vec<F> = (2..=12u64).map(F::from_u64).collect();
    let bad = Witness {
        coefficients: &other,
        blinder,
    };
    let (forged, _) = prove(&statement, &bad);
    assert_eq!(check(&statement, &forged), None);
    // A tampered round.
    let mut tampered = opening.clone();
    tampered.rounds[1].0 = tampered.rounds[1].0 + Projective::generator();
    assert_eq!(check(&statement, &tampered), None);
    // A tampered final step: change sg and let the decider catch what the
    // succinct check does not see.
    let mut moved = opening.clone();
    moved.sg = moved.sg + Projective::generator();
    match check(&statement, &moved) {
        None => {}
        Some(acc) => assert!(!ipa.decide(&acc)),
    }
}

#[test]
fn every_instance_satisfies_the_laws() {
    let lambda = Pallas::lambda();
    laws(&scheme(BindStatement, Full, NotHiding), false);
    laws(&scheme(BindStatement, Full, Blinded), true);
    laws(&scheme(BindStatement, Endo { lambda }, NotHiding), false);
    laws(&scheme(BindStatement, Endo { lambda }, Blinded), true);
    laws(&scheme(BindValue, Full, NotHiding), false);
    laws(&scheme(BindValue, Full, Blinded), true);
    laws(&scheme(BindValue, Endo { lambda }, NotHiding), false);
    laws(&scheme(BindValue, Endo { lambda }, Blinded), true);
}

#[test]
fn a_tampered_schnorr_step_is_rejected() {
    let ipa = scheme(
        BindValue,
        Endo {
            lambda: Pallas::lambda(),
        },
        Blinded,
    );
    let p: Vec<F> = (1..=5u64).map(F::from_u64).collect();
    let points = [F::from_u64(9)];
    let blinder = F::from_u64(4);
    let commitment = msm(&ipa.generators, &p) + ipa.h * blinder;
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale: F::ONE,
        value: eval(&p, points[0]),
    };
    let witness = Witness {
        coefficients: &p,
        blinder,
    };
    let mut transcript = vesta_sponge(1);
    let mut rng = Arkworks(SplitMix64(2));
    let (opening, _) = ipa.prove(
        &mut ScalarOracle(&mut transcript),
        &statement,
        &witness,
        &mut rng,
    );
    for which in 0..3 {
        let mut bad = opening.clone();
        match which {
            0 => bad.last.z1 = bad.last.z1 + F::ONE,
            1 => bad.last.z2 = bad.last.z2 + F::ONE,
            _ => bad.last.delta = bad.last.delta + Projective::generator(),
        }
        let mut transcript = vesta_sponge(1);
        assert_eq!(
            ipa.succinct(&mut ScalarOracle(&mut transcript), &statement, &bad),
            None
        );
    }
    // A wrong blinder is a wrong opening of the commitment.
    let mut transcript = vesta_sponge(1);
    let mut rng = Arkworks(SplitMix64(2));
    let off = Witness {
        coefficients: &p,
        blinder: blinder + F::ONE,
    };
    let (forged, _) = ipa.prove(
        &mut ScalarOracle(&mut transcript),
        &statement,
        &off,
        &mut rng,
    );
    let mut transcript = vesta_sponge(1);
    assert_eq!(
        ipa.succinct(&mut ScalarOracle(&mut transcript), &statement, &forged),
        None
    );
}

#[test]
fn ipa_is_the_default_instance() {
    // The scheme of the crate and the (BindStatement, Full,
    // ScaledGenerator, NotHiding) instance on the same key give the same
    // proof and the same accumulator.
    let generic = scheme(BindStatement, Full, NotHiding);
    let ipa =
        Ipa::<Pallas, _>::from_generators(generic.generators.clone(), generic.u, vesta_sponge(1));
    let p: Vec<F> = (1..=9u64).map(F::from_u64).collect();
    let points = [F::from_u64(2)];
    let commitment = msm(&generic.generators, &p);
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale: F::ONE,
        value: eval(&p, points[0]),
    };
    let (proof, accumulator) = ipa.prove(&mut vesta_sponge(1), &statement, &p);
    let witness = Witness {
        coefficients: &p,
        blinder: F::ZERO,
    };
    let mut transcript = vesta_sponge(1);
    let mut rng = Arkworks(SplitMix64(0));
    let (opening, generic_accumulator) = generic.prove(
        &mut ScalarOracle(&mut transcript),
        &statement,
        &witness,
        &mut rng,
    );
    assert_eq!(accumulator, generic_accumulator);
    assert_eq!(proof.rounds(), opening.rounds.as_slice());
    assert_eq!(proof.scalar(), opening.last);
    assert_eq!(proof.sg(), opening.sg);
}
