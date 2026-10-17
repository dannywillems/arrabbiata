//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_commit::PolynomialCommitment;
use arrabbiata_curve::{Projective, msm, points_from_transcript};
use arrabbiata_field::{
    Field, PrimeField,
    rng::{Arkworks, Sampler, SplitMix64},
};
use arrabbiata_ipa::{
    Ipa, IpaProof, RecursionChallenge, Statement,
    generic::{
        BaseDraw, BaseOracle, BindStatement, BindValue, Binding, Blinded, BoundStatement,
        CHALLENGE_BITS, ChallengeMap, DOMAIN, Delta, Endo, FinalChallenge, Folded, Full,
        GenericIpa, HashToCurve, HashedToCurve, Hiding, IpaOracle, NoRandomness, NotHiding,
        Opening, OpeningTranscript, Round, RoundChallenge, RoundPoints, ScalarDraw, ScalarOracle,
        ScaledGenerator, Schnorr, ShiftedValue, ValueBase, ValueChallenge, Witness, shift_scalar,
    },
};
use arrabbiata_pasta::{Fp, Fq, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use arrabbiata_transcript::{
    Transcript,
    typed::{Draw, End, Label, Pattern, Tape, Write},
};

#[test]
fn crate_overview() {
    // 8 generators on Pallas from Poseidon over Fp; challenges from Poseidon
    // over Fq, the field the polynomials live in.
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let p = [Fq::from_u64(1), Fq::from_u64(2), Fq::from_u64(3)]; // 1 + 2X + 3X^2
    let c = ipa.commit(&p);
    let (v, proof) = ipa.open(&p, &Fq::from_u64(2));
    assert_eq!(v, Fq::from_u64(17));
    assert!(ipa.verify(&c, &Fq::from_u64(2), v, &proof));
    assert!(!ipa.verify(&c, &Fq::from_u64(2), v + Fq::ONE, &proof));
}

#[test]
fn ipa() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    assert_eq!(ipa.size(), 4);
}

#[test]
fn ipa_proof() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let (_, proof) = ipa.open(&[Fq::ONE], &Fq::ZERO);
    assert_eq!(proof.rounds().len(), 3);
}

#[test]
fn ipa_proof_rounds() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    assert_eq!(ipa.open(&[Fq::ONE], &Fq::ONE).1.rounds().len(), 1);
}

#[test]
fn ipa_proof_from_parts() {
    let sg = Projective::<Pallas>::generator();
    let proof = IpaProof::<Pallas>::from_parts(vec![], Fq::ONE, sg);
    assert!(proof.rounds().is_empty());
}

#[test]
fn ipa_proof_scalar() {
    let sg = Projective::<Pallas>::generator();
    let proof = IpaProof::<Pallas>::from_parts(vec![], Fq::ONE, sg);
    assert_eq!(proof.scalar(), Fq::ONE);
}

#[test]
fn ipa_proof_sg() {
    // The folded generator of a size-1 key is the generator itself.
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 1, vesta_sponge(1));
    let (_, proof) = ipa.open(&[Fq::ONE], &Fq::ZERO);
    assert_eq!(proof.sg(), PolynomialCommitment::commit(&ipa, &[Fq::ONE]));
}

#[test]
fn ipa_setup() {
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(7), 16, vesta_sponge(1));
    assert_eq!(ipa.size(), 16);
}

#[test]
fn ipa_from_generators() {
    // The same key, rebuilt from its generators, commits alike.
    let ipa = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    let g = Projective::<Pallas>::generator();
    let generators: Vec<Projective<Pallas>> = (1..=4u64).map(|k| g * Fq::from_u64(k)).collect();
    let rebuilt = Ipa::<Pallas, _>::from_generators(generators.clone(), g, vesta_sponge(1));
    assert_eq!(rebuilt.size(), 4);
    let p = [Fq::ONE, Fq::ONE];
    assert_eq!(rebuilt.commit(&p), generators[0] + generators[1]);
    assert_eq!(ipa.size(), rebuilt.size());
}

#[test]
fn ipa_size() {
    assert_eq!(
        Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1)).size(),
        2
    );
}

/// The generic scheme on 8 generators from Poseidon, with `binding`,
/// `challenges` and `hiding`.
fn scheme<B, Ch, H>(
    binding: B,
    challenges: Ch,
    hiding: H,
) -> GenericIpa<Pallas, B, Ch, ScaledGenerator, H> {
    let mut points = points_from_transcript::<Pallas, _>(&mut pallas_sponge(3), 10);
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

/// `p = 1 + 2X + 3X^2` and its statement at 2 (value 17), unhidden.
fn opened(generators: &[Projective<Pallas>]) -> (Vec<Fq>, Projective<Pallas>) {
    let p: Vec<Fq> = (1..=3u64).map(Fq::from_u64).collect();
    let commitment = msm(&generators[..3], &p);
    (p, commitment)
}

/// A base-field oracle: a scalar oracle that also squeezes the base field
/// from a second sponge.
struct TwoSponges(PoseidonSponge<Fq>, PoseidonSponge<Fp>);

impl IpaOracle<Pallas> for TwoSponges {
    fn absorb_scalar(&mut self, x: Fq) {
        self.0.absorb(x);
    }

    fn absorb_point(&mut self, p: &Projective<Pallas>) {
        IpaOracle::<Pallas>::absorb_point(&mut ScalarOracle(&mut self.0), p);
    }

    fn squeeze_scalar(&mut self) -> Fq {
        self.0.squeeze()
    }
}

impl BaseOracle<Pallas> for TwoSponges {
    fn squeeze_base(&mut self) -> Fp {
        self.1.squeeze()
    }
}

/// `t G`, a (non-random) map from the base field.
struct ScaleGenerator;

impl HashToCurve<Pallas> for ScaleGenerator {
    fn to_group(&self, t: Fp) -> Projective<Pallas> {
        Projective::<Pallas>::generator() * Fq::from_u64(t.limb(0))
    }
}

#[test]
fn generic_ipa_oracle() {
    // A scalar oracle squeezes what its transcript squeezes.
    let mut oracle = ScalarOracle(&mut vesta_sponge(1));
    IpaOracle::<Pallas>::absorb_scalar(&mut oracle, Fq::ONE);
    let mut reference = vesta_sponge(1);
    reference.absorb(Fq::ONE);
    let squeezed = IpaOracle::<Pallas>::squeeze_scalar(&mut oracle);
    assert_eq!(squeezed, reference.squeeze());
}

#[test]
fn generic_base_oracle() {
    let mut oracle = TwoSponges(vesta_sponge(1), pallas_sponge(2));
    assert_eq!(oracle.squeeze_base(), pallas_sponge(2).squeeze());
}

#[test]
fn generic_scalar_oracle() {
    // Points are absorbed by their coordinates: two points, two streams.
    let g = Projective::<Pallas>::generator();
    let mut a = vesta_sponge(1);
    let mut b = vesta_sponge(1);
    IpaOracle::<Pallas>::absorb_point(&mut ScalarOracle(&mut a), &g);
    IpaOracle::<Pallas>::absorb_point(&mut ScalarOracle(&mut b), &(g + g));
    assert_ne!(a.squeeze(), b.squeeze());
}

#[test]
fn generic_binding() {
    // Binding changes the oracle's state with the statement.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (_, commitment) = opened(&ipa.generators);
    let statement = Statement {
        commitment: &commitment,
        points: &[Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let mut bound = vesta_sponge(1);
    bind(&BindStatement, &mut ScalarOracle(&mut bound), &statement);
    assert_ne!(bound.squeeze(), vesta_sponge(1).squeeze());
}

#[test]
fn generic_domain() {
    // "ipa_open" in ASCII.
    assert_eq!(DOMAIN.to_be_bytes(), *b"ipa_open");
}

#[test]
fn generic_bind_statement() {
    // The statement's value is bound: another value, another stream.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (_, commitment) = opened(&ipa.generators);
    let squeeze = |value: Fq| {
        let statement = Statement {
            commitment: &commitment,
            points: &[Fq::from_u64(2)],
            evalscale: Fq::ONE,
            value,
        };
        let mut sponge = vesta_sponge(1);
        bind(&BindStatement, &mut ScalarOracle(&mut sponge), &statement);
        sponge.squeeze()
    };
    assert_ne!(squeeze(Fq::from_u64(17)), squeeze(Fq::from_u64(18)));
}

#[test]
fn generic_bind_value() {
    // Only the shifted value is absorbed: the commitment does not count.
    let g = Projective::<Pallas>::generator();
    let other = g + g;
    let squeeze = |commitment: &Projective<Pallas>| {
        let statement = Statement {
            commitment,
            points: &[Fq::from_u64(2)],
            evalscale: Fq::ONE,
            value: Fq::from_u64(17),
        };
        let mut sponge = vesta_sponge(1);
        bind(&BindValue, &mut ScalarOracle(&mut sponge), &statement);
        sponge.squeeze()
    };
    assert_eq!(squeeze(&g), squeeze(&other));
}

#[test]
fn generic_shift_scalar() {
    // The shift is affine with slope 1 or 1/2: differences are kept or
    // halved.
    let one = shift_scalar::<Pallas>(Fq::ONE) - shift_scalar::<Pallas>(Fq::ZERO);
    assert!(one == Fq::ONE || one + one == Fq::ONE);
}

#[test]
fn generic_challenge_map() {
    let raw = Fq::from_u64(42);
    assert_eq!(ChallengeMap::map(&Full, raw), raw);
}

#[test]
fn generic_full() {
    assert_eq!(Full.map(Fq::ONE), Fq::ONE);
}

#[test]
fn generic_endo() {
    // A challenge maps to `a lambda + b` with `a` nonzero: the image
    // depends on lambda.
    let raw = Fq::ZERO;
    let one = Endo { lambda: Fq::ONE }.map(raw);
    let two = Endo {
        lambda: Fq::from_u64(2),
    }
    .map(raw);
    assert_ne!(one, two);
}

#[test]
fn generic_challenge_bits() {
    assert_eq!(CHALLENGE_BITS, 128);
}

#[test]
fn generic_value_base() {
    let u = Projective::<Pallas>::generator();
    let mut sponge = vesta_sponge(1);
    let base = base(&ScaledGenerator, &mut ScalarOracle(&mut sponge), &u);
    assert_eq!(base, u * vesta_sponge(1).squeeze());
}

#[test]
fn generic_scaled_generator() {
    // A different `u`, a different base.
    let u = Projective::<Pallas>::generator();
    let a = base(
        &ScaledGenerator,
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &u,
    );
    let b = base(
        &ScaledGenerator,
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &(u + u),
    );
    assert_eq!(b, a + a);
}

#[test]
fn generic_hash_to_curve() {
    let g = Projective::<Pallas>::generator();
    assert_eq!(
        ScaleGenerator.to_group(Fp::from_u64(3)),
        g * Fq::from_u64(3)
    );
}

#[test]
fn generic_hashed_to_curve() {
    // The base ignores `u`: it is the map of a squeezed base element.
    let u = Projective::<Pallas>::generator();
    let mut oracle = TwoSponges(vesta_sponge(1), pallas_sponge(2));
    let base = base(&HashedToCurve(ScaleGenerator), &mut oracle, &u);
    let t: Fp = pallas_sponge(2).squeeze();
    assert_eq!(base, ScaleGenerator.to_group(t));
}

#[test]
fn generic_folded() {
    let g = Projective::<Pallas>::generator();
    let folded = Folded::<Pallas> {
        a: Fq::ONE,
        b: Fq::from_u64(2),
        sg: g,
        u: g,
        h: g,
        blinder: Fq::ZERO,
    };
    assert_eq!(folded.a * folded.b, Fq::from_u64(2));
}

#[test]
fn generic_hiding() {
    // Without hiding the round blinders are zero and nothing is drawn.
    let (l, r) = Hiding::<Pallas>::round_blinders(&NotHiding, &mut NoRandomness);
    assert_eq!((l, r), (Fq::ZERO, Fq::ZERO));
}

#[test]
fn generic_not_hiding() {
    // The final step is the folded scalar `a`.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (p, commitment) = opened(&ipa.generators);
    let statement = Statement {
        commitment: &commitment,
        points: &[Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let witness = Witness {
        coefficients: &p,
        blinder: Fq::ZERO,
    };
    let (opening, _) = ipa.prove(
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &statement,
        &witness,
        &mut NoRandomness,
    );
    let last: Fq = opening.last;
    assert_ne!(last, Fq::ZERO);
}

#[test]
fn generic_blinded() {
    // Hiding draws the round blinders from the sampler.
    let mut rng = Arkworks(SplitMix64(1));
    let (l, r) = Hiding::<Pallas>::round_blinders(&Blinded, &mut rng);
    let mut reference = Arkworks(SplitMix64(1));
    let expected: (Fq, Fq) = (reference.sample(), reference.sample());
    assert_eq!((l, r), expected);
}

#[test]
fn generic_schnorr() {
    // A hidden opening of a blinded commitment ends in (delta, z1, z2).
    let ipa = scheme(BindStatement, Full, Blinded);
    let (p, commitment) = opened(&ipa.generators);
    let blinder = Fq::from_u64(5);
    let commitment = commitment + ipa.h * blinder;
    let statement = Statement {
        commitment: &commitment,
        points: &[Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let witness = Witness {
        coefficients: &p,
        blinder,
    };
    let mut rng = Arkworks(SplitMix64(1));
    let (opening, _) = ipa.prove(
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &statement,
        &witness,
        &mut rng,
    );
    let Schnorr { delta, z1, z2 } = opening.last;
    assert_ne!(delta, Projective::<Pallas>::identity());
    assert_ne!(z1, z2);
}

#[test]
fn generic_opening() {
    // 8 generators, 3 rounds.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (p, commitment) = opened(&ipa.generators);
    let statement = Statement {
        commitment: &commitment,
        points: &[Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let witness = Witness {
        coefficients: &p,
        blinder: Fq::ZERO,
    };
    let (opening, accumulator) = ipa.prove(
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &statement,
        &witness,
        &mut NoRandomness,
    );
    let Opening { rounds, sg, .. } = opening;
    assert_eq!(rounds.len(), 3);
    assert_eq!(sg, accumulator.comm);
}

#[test]
fn generic_generic_ipa() {
    // The crate's IPA is the instance on the same generators.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let crate_ipa =
        Ipa::<Pallas, _>::from_generators(ipa.generators.clone(), ipa.u, vesta_sponge(1));
    let (p, _) = opened(&ipa.generators);
    assert_eq!(crate_ipa.commit(&p), msm(&ipa.generators[..3], &p));
}

#[test]
fn generic_no_randomness() {
    // Never drawn from by an opening without hiding: a NotHiding opening
    // runs on it (see generic_not_hiding).
    assert_eq!(NoRandomness, NoRandomness);
}

#[test]
fn generic_witness() {
    let p = [Fq::ONE];
    let witness = Witness {
        coefficients: &p,
        blinder: Fq::ZERO,
    };
    assert_eq!(witness.coefficients.len(), 1);
}

#[test]
fn generic_ipa_rounds() {
    assert_eq!(scheme(BindStatement, Full, NotHiding).rounds(), 3);
}

/// An opening and its statement, for the check and the decider.
fn checked(value: Fq) -> Option<RecursionChallenge<Pallas>> {
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (p, commitment) = opened(&ipa.generators);
    let points = [Fq::from_u64(2)];
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let witness = Witness {
        coefficients: &p,
        blinder: Fq::ZERO,
    };
    let (opening, _) = ipa.prove(
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &statement,
        &witness,
        &mut NoRandomness,
    );
    let claimed = Statement { value, ..statement };
    ipa.succinct(&mut ScalarOracle(&mut vesta_sponge(1)), &claimed, &opening)
}

#[test]
fn generic_ipa_prove() {
    // The prover's accumulator is the verifier's.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (p, commitment) = opened(&ipa.generators);
    let points = [Fq::from_u64(2)];
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let witness = Witness {
        coefficients: &p,
        blinder: Fq::ZERO,
    };
    let (_, accumulator) = ipa.prove(
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &statement,
        &witness,
        &mut NoRandomness,
    );
    assert_eq!(checked(Fq::from_u64(17)), Some(accumulator));
}

#[test]
fn generic_ipa_check() {
    // The check gives the accumulator and the point that must vanish.
    let ipa = scheme(BindStatement, Full, NotHiding);
    let (p, commitment) = opened(&ipa.generators);
    let points = [Fq::from_u64(2)];
    let statement = Statement {
        commitment: &commitment,
        points: &points,
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let witness = Witness {
        coefficients: &p,
        blinder: Fq::ZERO,
    };
    let (opening, _) = ipa.prove(
        &mut ScalarOracle(&mut vesta_sponge(1)),
        &statement,
        &witness,
        &mut NoRandomness,
    );
    let (_, expression) = ipa
        .check(
            &mut ScalarOracle(&mut vesta_sponge(1)),
            &statement,
            &opening,
        )
        .unwrap();
    assert_eq!(expression, Projective::<Pallas>::identity());
}

#[test]
fn generic_ipa_succinct() {
    assert!(checked(Fq::from_u64(17)).is_some());
    assert!(checked(Fq::from_u64(18)).is_none());
}

#[test]
fn generic_ipa_decide() {
    let ipa = scheme(BindStatement, Full, NotHiding);
    let mut accumulator = checked(Fq::from_u64(17)).unwrap();
    assert!(ipa.decide(&accumulator));
    accumulator.comm = accumulator.comm + Projective::<Pallas>::generator();
    assert!(!ipa.decide(&accumulator));
}

/// `binding`'s step alone, on `oracle`.
fn bind<B, O>(binding: &B, oracle: &mut O, statement: &Statement<'_, Pallas>)
where
    B: Binding<Pallas, O>,
{
    let tape = Tape::<Fq, &mut O, B::Step<End>>::undescribed(oracle);
    let _ = binding.bind(tape, statement).into_inner();
}

/// `value_base`'s step alone, on `oracle`.
fn base<V, O>(value_base: &V, oracle: &mut O, u: &Projective<Pallas>) -> Projective<Pallas>
where
    V: ValueBase<Pallas, O>,
{
    let tape = Tape::<Fq, &mut O, V::Step<End>>::undescribed(oracle);
    let (point, tape) = value_base.base(tape, u);
    let _ = tape.into_inner();
    point
}

#[test]
fn generic_bound_statement() {
    // Written as BindStatement binds: the domain first.
    let g = Projective::<Pallas>::generator();
    let bound = BoundStatement::<Pallas> {
        commitment: g,
        points: vec![Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let mut sponge = vesta_sponge(1);
    bound.write(&mut &mut ScalarOracle(&mut sponge));
    let statement = Statement {
        commitment: &g,
        points: &[Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let mut reference = vesta_sponge(1);
    bind(
        &BindStatement,
        &mut ScalarOracle(&mut reference),
        &statement,
    );
    assert_eq!(sponge.squeeze(), reference.squeeze());
}

#[test]
fn generic_shifted_value() {
    let mut sponge = vesta_sponge(1);
    ShiftedValue::<Pallas>(Fq::ONE).write(&mut &mut ScalarOracle(&mut sponge));
    let mut reference = vesta_sponge(1);
    reference.absorb(Fq::ONE);
    assert_eq!(sponge.squeeze(), reference.squeeze());
}

#[test]
fn generic_round_points() {
    // L then R.
    let g = Projective::<Pallas>::generator();
    let mut a = vesta_sponge(1);
    let mut b = vesta_sponge(1);
    RoundPoints(g, g + g).write(&mut &mut ScalarOracle(&mut a));
    RoundPoints(g + g, g).write(&mut &mut ScalarOracle(&mut b));
    assert_ne!(a.squeeze(), b.squeeze());
}

#[test]
fn generic_delta() {
    let g = Projective::<Pallas>::generator();
    let mut sponge = vesta_sponge(1);
    Delta(g).write(&mut &mut ScalarOracle(&mut sponge));
    assert_ne!(sponge.squeeze(), vesta_sponge(1).squeeze());
}

#[test]
fn generic_scalar_draw() {
    let mut sponge = vesta_sponge(1);
    let x = <ScalarDraw<Pallas, RoundChallenge> as Draw<&mut ScalarOracle<'_, _>>>::draw(
        &mut &mut ScalarOracle(&mut sponge),
    );
    assert_eq!(x, vesta_sponge(1).squeeze());
}

#[test]
fn generic_base_draw() {
    let mut oracle = TwoSponges(vesta_sponge(1), pallas_sponge(2));
    let t = <BaseDraw<Pallas> as Draw<&mut TwoSponges>>::draw(&mut &mut oracle);
    assert_eq!(t, pallas_sponge(2).squeeze());
}

#[test]
fn generic_step_names() {
    let labels = [
        <ValueChallenge as Label>::LABEL,
        <RoundChallenge as Label>::LABEL,
        <FinalChallenge as Label>::LABEL,
    ];
    assert!(labels[0] != labels[1] && labels[1] != labels[2]);
}

#[test]
fn generic_round() {
    // One message and one challenge.
    assert_eq!(<Round<Pallas> as Pattern>::MESSAGES, 1);
    assert_eq!(<Round<Pallas> as Pattern>::CHALLENGES, 1);
}

#[test]
fn generic_opening_transcript() {
    // This crate's IPA: the bound statement, xi, the rounds, nothing after;
    // Kimchi's: the shifted value, the hashed base, the rounds, delta and c.
    type Ours<'a> = OpeningTranscript<
        Pallas,
        BindStatement,
        ScaledGenerator,
        NotHiding,
        ScalarOracle<'a, PoseidonSponge<Fq>>,
    >;
    type Blinding = OpeningTranscript<Pallas, BindValue, ScaledGenerator, Blinded, TwoSponges>;
    assert_eq!(<Ours<'_> as Pattern>::MESSAGES, 0);
    assert_eq!(<Ours<'_> as Pattern>::CHALLENGES, 1);
    assert_eq!(<Blinding as Pattern>::MESSAGES, 1);
    assert_eq!(<Blinding as Pattern>::CHALLENGES, 2);
}

#[test]
fn generic_opening_tape() {
    // The tape starts at the binding's step: binding first, then xi.
    let g = Projective::<Pallas>::generator();
    let statement = Statement {
        commitment: &g,
        points: &[Fq::from_u64(2)],
        evalscale: Fq::ONE,
        value: Fq::from_u64(17),
    };
    let mut sponge = vesta_sponge(1);
    let mut oracle = ScalarOracle(&mut sponge);
    let tape: OpeningTape<'_, Pallas, BindStatement, ScaledGenerator, NotHiding, _> =
        Tape::undescribed(&mut oracle);
    let tape = BindStatement.bind(tape, &statement);
    let (u, tape) = ScaledGenerator.base(tape, &g);
    let _ = tape.done().into_inner();
    let mut reference = vesta_sponge(1);
    bind(
        &BindStatement,
        &mut ScalarOracle(&mut reference),
        &statement,
    );
    assert_eq!(u, g * reference.squeeze());
}
