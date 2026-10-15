//! The laws of [`Accumulation`], as generic functions run on every scheme
//! and on more than one oracle:
//!
//! - IPA over Pallas, with the Poseidon duplex sponge and with Kimchi's
//!   sponge;
//! - KZG over BN254, with the Poseidon duplex sponge.
//!
//! The laws:
//!
//! - completeness of `Succinct`, `Fold`, `Decide` and of the composites
//!   `Open`, `Accumulate ; Decide` and `DecideAll`;
//! - `Accumulate` is `Each(Succinct)` then `Fold`;
//! - a wrong value is rejected by `Open`;
//! - a false accumulator, which no succinct check sees, makes the fold
//!   fail or its output false: the decider of the last accumulator covers
//!   every accumulated one.

use arrabbiata_bn254::{Bn254, Fr, G1};
use arrabbiata_commit::Univariate;
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_halo::{
    Accumulation, Claims, ClaimsOf, Decide, Evaluations, Fold, Succinct, accumulate, decide_all,
    open,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{
    PoseidonSponge, bn254_scalar_sponge,
    kimchi::{KimchiSponge, vesta_base as kimchi_vesta_base},
    pallas_sponge, vesta_sponge,
};
use arrabbiata_reduction::{Each, Reduction, Relation, Then, laws};
use arrabbiata_transcript::Transcript;
use proptest::prelude::*;
use std::sync::OnceLock;

/// The key size.
const SIZE: usize = 16;
/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of the IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the reductions' transcripts.
const PROOF_DOMAIN: u64 = 2;
/// The trapdoor of the test KZG setup.
const TRAPDOOR: u64 = 1234;

fn ipa() -> &'static Ipa<Pallas, PoseidonSponge<Fq>> {
    static IPA: OnceLock<Ipa<Pallas, PoseidonSponge<Fq>>> = OnceLock::new();
    IPA.get_or_init(|| {
        let mut generators = pallas_sponge(GENERATORS_DOMAIN);
        let transcript = vesta_sponge(IPA_DOMAIN);
        Ipa::setup(&mut generators, SIZE, transcript)
    })
}

fn kzg() -> &'static Kzg<Bn254> {
    static KZG: OnceLock<Kzg<Bn254>> = OnceLock::new();
    KZG.get_or_init(|| {
        let trapdoor = Fr::from_u64(TRAPDOOR);
        Kzg::insecure_setup(SIZE, trapdoor)
    })
}

fn poseidon_vesta() -> PoseidonSponge<Fq> {
    vesta_sponge(PROOF_DOMAIN)
}

fn kimchi_vesta() -> KimchiSponge<Fq> {
    KimchiSponge::new(kimchi_vesta_base())
}

fn poseidon_bn254() -> PoseidonSponge<Fr> {
    bn254_scalar_sponge(PROOF_DOMAIN)
}

/// A batch of claims as integers: polynomials and points.
#[derive(Clone, Debug)]
struct Raw {
    polynomials: Vec<Vec<u64>>,
    points: Vec<u64>,
}

fn arb_raw() -> impl Strategy<Value = Raw> {
    let polynomial = prop::collection::vec(any::<u64>(), 0..=SIZE);
    let polynomials = prop::collection::vec(polynomial, 1..4);
    let points = prop::collection::vec(any::<u64>(), 1..3);
    (polynomials, points).prop_map(|(polynomials, points)| Raw {
        polynomials,
        points,
    })
}

fn eval<F: Field>(p: &[F], x: F) -> F {
    p.iter().rev().fold(F::ZERO, |acc, c| acc * x + *c)
}

/// The claims of a raw batch over `pcs`, and their polynomials.
fn build<F, PC>(pcs: &PC, raw: &Raw) -> (ClaimsOf<F, PC>, Vec<Vec<F>>)
where
    F: Field,
    PC: Accumulation<F, Family = Univariate>,
{
    let polynomials: Vec<Vec<F>> = raw
        .polynomials
        .iter()
        .map(|p| p.iter().map(|c| F::from_u64(*c)).collect())
        .collect();
    let points: Vec<F> = raw.points.iter().map(|x| F::from_u64(*x)).collect();
    let claims = Claims {
        commitments: polynomials.iter().map(|p| pcs.commit(p)).collect(),
        values: polynomials
            .iter()
            .map(|p| points.iter().map(|x| eval(p, *x)).collect())
            .collect(),
        points,
    };
    (claims, polynomials)
}

/// Completeness of a batched opening, and a wrong value rejected.
fn opening_laws<F, PC, T>(
    pcs: &PC,
    transcript: impl Fn() -> T,
    raw: &Raw,
    which: usize,
) -> Result<(), TestCaseError>
where
    F: Field,
    PC: Accumulation<F, Family = Univariate>,
    T: Transcript<F>,
{
    let (claims, polynomials) = build::<F, PC>(pcs, raw);
    prop_assert!(Evaluations::new(pcs).check(&claims, &polynomials).is_ok());
    let succinct = Succinct::new(pcs);
    let out = laws::complete::<F, _, _>(&succinct, &transcript, &claims, &polynomials);
    prop_assert!(out.is_ok(), "succinct: {:?}", out.err());
    let opened = laws::complete::<F, _, _>(&open(pcs), &transcript, &claims, &polynomials);
    prop_assert!(opened.is_ok(), "open: {:?}", opened.err());
    let proof = open(pcs)
        .prove(transcript(), &claims, &polynomials)
        .unwrap()
        .proof;
    let mut wrong = claims.clone();
    let row = which % wrong.values.len();
    let column = (which / 7) % wrong.points.len();
    wrong.values[row][column] = wrong.values[row][column] + F::ONE;
    prop_assert!(open(pcs).verify(transcript(), &wrong, proof).is_err());
    Ok(())
}

/// The accumulators of honest batches.
fn accumulators<F, PC, T>(
    pcs: &PC,
    transcript: &impl Fn() -> T,
    raws: &[Raw],
) -> Vec<PC::Accumulator>
where
    F: Field,
    PC: Accumulation<F, Family = Univariate>,
    T: Transcript<F>,
{
    raws.iter()
        .map(|raw| {
            let (claims, polynomials) = build::<F, PC>(pcs, raw);
            Succinct::new(pcs)
                .prove(transcript(), &claims, &polynomials)
                .unwrap()
                .instance
        })
        .collect()
}

/// Completeness of the fold and of the composites, and `Accumulate` as
/// `Each(Succinct)` then `Fold`.
fn accumulation_laws<F, PC, T>(
    pcs: &PC,
    transcript: impl Fn() -> T,
    raws: &[Raw],
) -> Result<(), TestCaseError>
where
    F: Field,
    PC: Accumulation<F, Family = Univariate>,
    T: Transcript<F>,
{
    let accumulators = accumulators(pcs, &transcript, raws);
    let units = vec![(); accumulators.len()];
    let folded = laws::complete::<F, _, _>(&Fold::new(pcs), &transcript, &accumulators, &units);
    prop_assert!(folded.is_ok(), "fold: {:?}", folded.err());
    let decided = laws::complete::<F, _, _>(&decide_all(pcs), &transcript, &accumulators, &units);
    prop_assert!(decided.is_ok(), "decide all: {:?}", decided.err());

    let (claims, polynomials): (Vec<_>, Vec<_>) =
        raws.iter().map(|raw| build::<F, PC>(pcs, raw)).unzip();
    let halo = Then(accumulate(pcs), Decide::new(pcs));
    let out = laws::complete::<F, _, _>(&halo, &transcript, &claims, &polynomials);
    prop_assert!(out.is_ok(), "accumulate: {:?}", out.err());

    let whole = accumulate(pcs)
        .prove(transcript(), &claims, &polynomials)
        .unwrap();
    let middle = Each(Succinct::new(pcs))
        .prove(transcript(), &claims, &polynomials)
        .unwrap();
    let rest = Fold::new(pcs)
        .prove(middle.transcript, &middle.instance, &middle.witness)
        .unwrap();
    prop_assert_eq!(whole.instance, rest.instance);
    Ok(())
}

/// A false accumulator among honest ones: the fold's verifier rejects it,
/// or its output is false.
fn a_false_accumulator_is_caught<F, PC, T>(
    pcs: &PC,
    transcript: impl Fn() -> T,
    raws: &[Raw],
    which: usize,
    tamper: impl Fn(&mut PC::Accumulator),
) -> Result<(), TestCaseError>
where
    F: Field,
    PC: Accumulation<F, Family = Univariate>,
    T: Transcript<F>,
{
    let mut accumulators = accumulators(pcs, &transcript, raws);
    let which = which % accumulators.len();
    tamper(&mut accumulators[which]);
    prop_assert!(!pcs.decide(&accumulators[which]));
    let units = vec![(); accumulators.len()];
    let fold = Fold::new(pcs);
    let proof = fold
        .prove(transcript(), &accumulators, &units)
        .unwrap()
        .proof;
    match fold.verify(transcript(), &accumulators, proof) {
        Err(_) => {}
        Ok((folded, _)) => prop_assert!(!pcs.decide(&folded)),
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    #[test]
    fn ipa_openings(raw in arb_raw(), which in any::<usize>()) {
        opening_laws(ipa(), poseidon_vesta, &raw, which)?;
        opening_laws(ipa(), kimchi_vesta, &raw, which)?;
    }

    #[test]
    fn kzg_openings(raw in arb_raw(), which in any::<usize>()) {
        opening_laws(kzg(), poseidon_bn254, &raw, which)?;
    }

    #[test]
    fn ipa_accumulation(raws in prop::collection::vec(arb_raw(), 1..4)) {
        accumulation_laws(ipa(), poseidon_vesta, &raws)?;
        accumulation_laws(ipa(), kimchi_vesta, &raws)?;
    }

    #[test]
    fn kzg_accumulation(raws in prop::collection::vec(arb_raw(), 1..4)) {
        accumulation_laws(kzg(), poseidon_bn254, &raws)?;
    }

    #[test]
    fn ipa_false_accumulators(raws in prop::collection::vec(arb_raw(), 1..4), which in any::<usize>()) {
        let g = Projective::<Pallas>::generator();
        a_false_accumulator_is_caught(ipa(), poseidon_vesta, &raws, which, |a| a.comm = a.comm + g)?;
    }

    #[test]
    fn kzg_false_accumulators(raws in prop::collection::vec(arb_raw(), 1..4), which in any::<usize>()) {
        let g = Projective::<G1>::generator();
        a_false_accumulator_is_caught(kzg(), poseidon_bn254, &raws, which, |a| a.p = a.p + g)?;
    }
}
