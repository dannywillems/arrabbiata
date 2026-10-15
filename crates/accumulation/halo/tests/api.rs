//! The behaviour of each public item, one test per item.

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_commit::PolynomialCommitment;
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_halo::{
    Accumulated, Accumulation, Claims, ClaimsOf, Combine, Decide, Error, Evaluations, Fold, Lift,
    Opened, PairingAccumulator, PointClaim, Succinct, accumulate, decide_all, ipa::Accumulators,
    open,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, bn254_scalar_sponge, pallas_sponge, vesta_sponge};
use arrabbiata_reduction::{Reduction, Relation};

type Scheme = Ipa<Pallas, PoseidonSponge<Fq>>;

fn ipa() -> Scheme {
    Ipa::setup(&mut pallas_sponge(0), 4, vesta_sponge(1))
}

fn kzg() -> Kzg<Bn254> {
    Kzg::insecure_setup(4, Fr::from_u64(1234))
}

/// `1 + x` at 2 and 3.
fn claims<F: Field, PC: PolynomialCommitment<F, Family = arrabbiata_commit::Univariate>>(
    pcs: &PC,
) -> (ClaimsOf<F, PC>, Vec<Vec<F>>) {
    let p = vec![F::ONE, F::ONE];
    let claims = Claims {
        commitments: vec![pcs.commit(&p)],
        points: vec![F::from_u64(2), F::from_u64(3)],
        values: vec![vec![F::from_u64(3), F::from_u64(4)]],
    };
    (claims, vec![p])
}

#[test]
fn error() {
    let error = Error::Value(1, 2);
    assert_eq!(
        format!("{error}"),
        "value of polynomial 1 at point 2 is wrong"
    );
}

#[test]
fn claims_and_evaluations() {
    let ipa = ipa();
    let (mut claims, polynomials) = claims::<Fq, _>(&ipa);
    assert!(claims.has_shape());
    assert!(Evaluations::new(&ipa).check(&claims, &polynomials).is_ok());
    claims.values[0][1] = Fq::ZERO;
    let found = Evaluations::new(&ipa).check(&claims, &polynomials);
    assert_eq!(found, Err(Error::Value(0, 1)));
}

#[test]
fn point_claim_and_opened() {
    let ipa = ipa();
    let p = vec![Fq::ONE, Fq::ONE];
    let claim = PointClaim {
        commitment: ipa.commit(&p),
        points: vec![Fq::from_u64(2)],
        values: vec![Fq::from_u64(3)],
    };
    assert!(Opened::new(&ipa).check(&claim, &p).is_ok());
}

#[test]
fn combine() {
    // A linear scheme's batch: the polynomials combined into one.
    let ipa = ipa();
    let (claims, polynomials) = claims::<Fq, _>(&ipa);
    let out = Combine::new(&ipa)
        .prove(vesta_sponge(2), &claims, &polynomials)
        .unwrap();
    assert!(Opened::new(&ipa).check(&out.instance, &out.witness).is_ok());
}

#[test]
fn succinct_and_accumulated() {
    let ipa = ipa();
    let (claims, polynomials) = claims::<Fq, _>(&ipa);
    let out = Succinct::new(&ipa)
        .prove(vesta_sponge(2), &claims, &polynomials)
        .unwrap();
    assert!(Accumulated::new(&ipa).check(&out.instance, &()).is_ok());
}

#[test]
fn accumulation() {
    // The trait directly, on KZG: open, check, fold, check, decide.
    let kzg = kzg();
    let (claims, polynomials) = claims::<Fr, _>(&kzg);
    let (opening, accumulator, _) = kzg.open_claims(bn254_scalar_sponge(2), &claims, &polynomials);
    let (checked, _) = kzg
        .check_claims(bn254_scalar_sponge(2), &claims, opening)
        .unwrap();
    assert_eq!(checked, accumulator);
    let list = [accumulator.clone(), accumulator];
    let (proof, folded, _) = kzg.fold(bn254_scalar_sponge(3), &list);
    let (checked, _) = kzg
        .check_fold(bn254_scalar_sponge(3), &list, proof)
        .unwrap();
    assert_eq!(checked, folded);
    assert!(kzg.decide(&folded));
}

#[test]
fn pairing_accumulator() {
    // (P, W) = (C - [v] G + [x] W, W) for an opening of 1 + x at 2.
    let kzg = kzg();
    let p = [Fr::ONE, Fr::ONE];
    let (value, w) = PolynomialCommitment::open(&kzg, &p, &Fr::from_u64(2));
    let g = Projective::<arrabbiata_bn254::G1>::generator();
    let pair = PairingAccumulator::<Bn254> {
        p: kzg.commit(&p) - g * value + w * Fr::from_u64(2),
        w,
    };
    assert!(kzg.decide(&pair));
}

#[test]
fn decide() {
    let ipa = ipa();
    let (claims, polynomials) = claims::<Fq, _>(&ipa);
    let accumulator = Succinct::new(&ipa)
        .prove(vesta_sponge(2), &claims, &polynomials)
        .unwrap()
        .instance;
    assert!(
        Decide::new(&ipa)
            .verify(vesta_sponge(3), &accumulator, ())
            .is_ok()
    );
}

#[test]
fn fold_and_decide_all() {
    let ipa = ipa();
    let (claims, polynomials) = claims::<Fq, _>(&ipa);
    let accumulator = Succinct::new(&ipa)
        .prove(vesta_sponge(2), &claims, &polynomials)
        .unwrap()
        .instance;
    let list = vec![accumulator.clone(), accumulator];
    let units = vec![(); 2];
    let folded = Fold::new(&ipa)
        .prove(vesta_sponge(3), &list, &units)
        .unwrap();
    assert!(ipa.decide(&folded.instance));
    let out = decide_all(&ipa)
        .prove(vesta_sponge(3), &list, &units)
        .unwrap();
    assert!(
        decide_all(&ipa)
            .verify(vesta_sponge(3), &list, out.proof)
            .is_ok()
    );
}

#[test]
fn lift_and_accumulators() {
    // One IPA accumulator becomes one claim at one fresh point.
    let ipa = ipa();
    let (claims, polynomials) = claims::<Fq, _>(&ipa);
    let accumulator = Succinct::new(&ipa)
        .prove(vesta_sponge(2), &claims, &polynomials)
        .unwrap()
        .instance;
    assert_eq!(Accumulators(vec![accumulator.clone()]).0.len(), 1);
    let out = Lift(&ipa)
        .prove(vesta_sponge(3), &vec![accumulator], &vec![()])
        .unwrap();
    assert_eq!(out.instance.points.len(), 1);
    assert!(
        Evaluations::new(&ipa)
            .check(&out.instance, &out.witness)
            .is_ok()
    );
}

#[test]
fn open_works() {
    let kzg = kzg();
    let (claims, polynomials) = claims::<Fr, _>(&kzg);
    let out = open(&kzg)
        .prove(bn254_scalar_sponge(2), &claims, &polynomials)
        .unwrap();
    assert!(
        open(&kzg)
            .verify(bn254_scalar_sponge(2), &claims, out.proof)
            .is_ok()
    );
}

#[test]
fn accumulate_works() {
    let kzg = kzg();
    let (claims, polynomials) = claims::<Fr, _>(&kzg);
    let batches = vec![claims.clone(), claims];
    let witnesses = vec![polynomials.clone(), polynomials];
    let out = accumulate(&kzg)
        .prove(bn254_scalar_sponge(2), &batches, &witnesses)
        .unwrap();
    assert!(kzg.decide(&out.instance));
}
