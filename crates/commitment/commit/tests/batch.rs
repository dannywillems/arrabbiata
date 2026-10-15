//! The laws of batched openings, for IPA, KZG, and both unbatched:
//!
//! - evaluation is linear: the combination of the values is the value of
//!   the combination;
//! - a linear scheme's `combine` is `commit` of the combination;
//! - an honest batch verifies (completeness);
//! - one wrong value, or a value too many or too few, is rejected
//!   (soundness, for the random challenge).

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_commit::{
    BatchOpening, LinearCommitment, PolynomialCommitment, PolynomialFamily, Unbatched, Univariate,
    combine_polynomials, combine_values,
};
use arrabbiata_field::Field;
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The schemes' size: polynomials of at most 8 coefficients.
const SIZE: usize = 8;
/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The trapdoor of the test KZG setup.
const TRAPDOOR: u64 = 1234;

fn ipa() -> Ipa<Pallas, PoseidonSponge<Fq>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, SIZE, transcript)
}

fn kzg() -> Kzg<Bn254> {
    let trapdoor = Fr::from_u64(TRAPDOOR);
    Kzg::<Bn254>::insecure_setup(SIZE, trapdoor)
}

/// The polynomials, the point and the challenge, as integers.
#[derive(Clone, Debug)]
struct Instance {
    polynomials: Vec<Vec<u64>>,
    point: u64,
    nu: u64,
}

fn arb_instance() -> impl Strategy<Value = Instance> {
    let polynomial = prop::collection::vec(any::<u64>(), 0..=SIZE);
    let polynomials = prop::collection::vec(polynomial, 1..6);
    (polynomials, any::<u64>(), any::<u64>()).prop_map(|(polynomials, point, nu)| Instance {
        polynomials,
        point,
        nu,
    })
}

fn to_field<F: Field>(instance: &Instance) -> (Vec<Vec<F>>, F, F) {
    let polynomials = instance
        .polynomials
        .iter()
        .map(|p| p.iter().map(|c| F::from_u64(*c)).collect())
        .collect();
    (
        polynomials,
        F::from_u64(instance.point),
        F::from_u64(instance.nu),
    )
}

fn evaluation_is_linear<F: Field>(instance: &Instance) -> bool {
    let (polynomials, point, nu) = to_field::<F>(instance);
    let slices: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
    let values: Vec<F> = slices
        .iter()
        .map(|p| <Univariate as PolynomialFamily<F>>::evaluate(p, &point))
        .collect();
    let combined = combine_polynomials(&slices, nu);
    combine_values(&values, nu) == <Univariate as PolynomialFamily<F>>::evaluate(&combined, &point)
}

fn combine_is_commit_of_the_combination<F, PC>(pcs: &PC, instance: &Instance) -> bool
where
    F: Field,
    PC: LinearCommitment<F, Family = Univariate>,
{
    let (polynomials, _, nu) = to_field::<F>(instance);
    let slices: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
    let commitments: Vec<PC::Commitment> = slices.iter().map(|p| pcs.commit(p)).collect();
    let mut power = F::ONE;
    let mut terms = Vec::new();
    for c in &commitments {
        terms.push((power, c));
        power = power * nu;
    }
    pcs.combine(&terms) == pcs.commit(&combine_polynomials(&slices, nu))
}

/// Completeness, and the rejection of a wrong value at `position`, of an
/// extra value, and of a missing one.
fn batch_laws<F, PC>(pcs: &PC, instance: &Instance, position: usize) -> Result<(), TestCaseError>
where
    F: Field,
    PC: BatchOpening<F, Family = Univariate>,
{
    let (polynomials, point, nu) = to_field::<F>(instance);
    let slices: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
    let commitments: Vec<PC::Commitment> = slices.iter().map(|p| pcs.commit(p)).collect();
    let commitments: Vec<&PC::Commitment> = commitments.iter().collect();
    let (values, proof) = pcs.open_batch(&slices, &point, nu);
    for (p, v) in slices.iter().zip(&values) {
        prop_assert_eq!(*v, <Univariate as PolynomialFamily<F>>::evaluate(p, &point));
    }
    prop_assert!(pcs.verify_batch(&commitments, &point, &values, nu, &proof));

    let mut wrong = values.clone();
    let position = position % wrong.len();
    wrong[position] = wrong[position] + F::ONE;
    prop_assert!(!pcs.verify_batch(&commitments, &point, &wrong, nu, &proof));

    let mut extra = values.clone();
    extra.push(F::ZERO);
    prop_assert!(!pcs.verify_batch(&commitments, &point, &extra, nu, &proof));
    let missing = &values[..values.len() - 1];
    prop_assert!(!pcs.verify_batch(&commitments, &point, missing, nu, &proof));
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn evaluation_is_linear_works(instance in arb_instance()) {
        prop_assert!(evaluation_is_linear::<Fq>(&instance));
        prop_assert!(evaluation_is_linear::<Fr>(&instance));
    }

    #[test]
    fn combine_is_commit_of_the_combination_works(instance in arb_instance()) {
        prop_assert!(combine_is_commit_of_the_combination(&ipa(), &instance));
        prop_assert!(combine_is_commit_of_the_combination(&kzg(), &instance));
    }

    #[test]
    fn ipa_batches(instance in arb_instance(), position in any::<usize>()) {
        batch_laws(&ipa(), &instance, position)?;
    }

    #[test]
    fn kzg_batches(instance in arb_instance(), position in any::<usize>()) {
        batch_laws(&kzg(), &instance, position)?;
    }

    #[test]
    fn unbatched_ipa_batches(instance in arb_instance(), position in any::<usize>()) {
        batch_laws(&Unbatched(ipa()), &instance, position)?;
    }

    #[test]
    fn unbatched_kzg_batches(instance in arb_instance(), position in any::<usize>()) {
        batch_laws(&Unbatched(kzg()), &instance, position)?;
    }
}

/// A batch proof is one proof for a linear scheme, and one per polynomial
/// unbatched.
#[test]
fn proof_sizes() {
    let pcs = ipa();
    let polynomials: [&[Fq]; 3] = [&[Fq::ONE], &[Fq::ZERO, Fq::ONE], &[Fq::ONE, Fq::ONE]];
    let point = Fq::from_u64(3);
    let nu = Fq::from_u64(5);
    let (_, proof) = pcs.open_batch(&polynomials, &point, nu);
    let (_, single) = pcs.open(polynomials[0], &point);
    assert_eq!(proof.rounds().len(), single.rounds().len());
    let unbatched = Unbatched(pcs);
    let (_, proofs) = unbatched.open_batch(&polynomials, &point, nu);
    assert_eq!(proofs.0.len(), polynomials.len());
}
