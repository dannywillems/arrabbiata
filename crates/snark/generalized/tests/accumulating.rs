//! Generalized PlonK compiled with an accumulation, as a reduction
//! `Satisfiability -> Accumulated`, on IPA (Poseidon, Kimchi's sponge and
//! the SHA-256 hash transcript) and KZG (Poseidon and the hash transcript):
//!
//! - completeness: the verifier's accumulator is the prover's, and it is
//!   decided true; `Then(Accumulating, Decide)` is a complete SNARK;
//! - soundness at the boundary: a wrong public input, or one changed
//!   evaluation, is rejected;
//! - accumulation: `Then(Each(Accumulating), Fold)` folds the proofs of
//!   several statements into one accumulator, decided once;
//! - the layout: every rotation a gate reads (here `PREV` and `NEXT`) is
//!   opened, and lookups (plookup and LogUp) go through the same path.

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_generalized_plonk::{
    Argument, Circuit, Gate, Lookup, Variable,
    accumulating::{Accumulating, rotations},
    max_coefficients, setup,
};
use arrabbiata_halo::{Accumulation, Decide, Fold};
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poly::Rotation;
use arrabbiata_poseidon::{
    PoseidonSponge, bn254_scalar_sponge,
    kimchi::{KimchiSponge, vesta_base as kimchi_vesta_base},
    pallas_sponge, vesta_sponge,
};
use arrabbiata_reduction::{Each, Reduction, Then, laws};
use arrabbiata_transcript::{
    Transcript,
    hash::{ByteHash, HashTranscript},
    typed::Label,
};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;
/// The trapdoor of the test KZG setup.
const TRAPDOOR: u64 = 1234;
/// The rows of the circuits.
const ROWS: usize = 8;

fn ipa(size: usize) -> Ipa<Pallas, PoseidonSponge<Fq>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::setup(&mut generators, size, transcript)
}

fn kzg(size: usize) -> Kzg<Bn254> {
    let trapdoor = Fr::from_u64(TRAPDOOR);
    Kzg::insecure_setup(size.next_power_of_two(), trapdoor)
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

/// SHA-256 as a `ByteHash`: the hash oracle, next to the sponges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Sha;

impl ByteHash for Sha {
    fn hash(&self, parts: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part);
        }
        hasher.finalize().into()
    }
}

fn hash_vesta() -> HashTranscript<Fq, Sha> {
    HashTranscript::new(Sha, PROOF_DOMAIN)
}

fn hash_bn254() -> HashTranscript<Fr, Sha> {
    HashTranscript::new(Sha, PROOF_DOMAIN)
}

fn witness(column: usize, rotation: Rotation) -> Variable {
    Variable::Witness { column, rotation }
}

/// `w(next) = s w^3` on every row but the last, `s` public input 0, and
/// `w(prev) + w = u` on every row but the first, `u` in column 1: the
/// gates read the rows before and after.
fn circuit<F: TwoAdicField>(n: usize) -> Circuit<F, 5, 4> {
    let v = |i| Sparse::<F, 5, 4>::variable(Var::try_new(i).unwrap()).unwrap();
    let (current, next, scale, previous, sum) = (v(0), v(1), v(2), v(3), v(4));
    let cube = current
        .checked_mul(&current)
        .unwrap()
        .checked_mul(&current)
        .unwrap();
    let step = next - scale.checked_mul(&cube).unwrap();
    let pairs = previous + current - sum;
    let gates = vec![Gate::new(Some(0), step), Gate::new(Some(1), pairs)];
    let mut but_last = vec![F::ONE; n];
    but_last[n - 1] = F::ZERO;
    let mut but_first = vec![F::ONE; n];
    but_first[0] = F::ZERO;
    let variables = vec![
        witness(0, Rotation::CURR),
        witness(0, Rotation::NEXT),
        Variable::Public(0),
        witness(0, Rotation::PREV),
        witness(1, Rotation::CURR),
    ];
    let num_rows = n;
    let witness_columns = 2;
    let fixed = vec![];
    let selectors = vec![but_last, but_first];
    let sigma = (0..2 * n).collect();
    let public_inputs = 1;
    let bindings = vec![];
    Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    )
    .unwrap()
}

/// The witness of [`circuit`] from `start`, and the statement `[s]`.
fn witness_of<F: Field>(n: usize, scale: u64, start: u64) -> (Vec<F>, Vec<Vec<F>>) {
    let s = F::from_u64(scale);
    let mut w = vec![F::from_u64(start)];
    for i in 0..n - 1 {
        let x = w[i];
        w.push(s * x * x * x);
    }
    let mut u = vec![F::ZERO; n];
    for i in 1..n {
        u[i] = w[i - 1] + w[i];
    }
    (vec![s], vec![w, u])
}

/// The laws on one scheme and one oracle, for the statements of `starts`.
fn laws_on<F, PC, T>(
    circuit: &Circuit<F, 5, 4>,
    pcs: &PC,
    transcript: impl Fn() -> T,
    scale: u64,
    starts: &[u64],
) -> Result<(), TestCaseError>
where
    F: TwoAdicField + arrabbiata_field::PrimeField,
    PC: Accumulation<F, Family = arrabbiata_commit::Univariate> + Label,
    T: Transcript<F>,
{
    let (pk, _) = setup(circuit, pcs).unwrap();
    let reduction = Accumulating::new(&pk, pcs);
    let statements: Vec<_> = starts
        .iter()
        .map(|s| witness_of::<F>(ROWS, scale, *s))
        .collect();
    let (public, columns) = &statements[0];

    let out = laws::complete::<F, _, _>(&reduction, &transcript, public, columns);
    prop_assert!(out.is_ok(), "accumulating: {:?}", out.err());
    let snark = Then(reduction, Decide::new(pcs));
    prop_assert!(laws::complete::<F, _, _>(&snark, &transcript, public, columns).is_ok());

    // Another statement, or a changed evaluation, is rejected.
    let proof = reduction
        .prove(transcript(), public, columns)
        .unwrap()
        .proof;
    let wrong = vec![public[0] + F::ONE];
    prop_assert!(
        reduction
            .verify(transcript(), &wrong, proof.clone())
            .is_err()
    );
    let mut tampered = proof;
    let values = &mut tampered.messages.tail.tail.tail.tail.tail.head.0;
    values[0][0] = values[0][0] + F::ONE;
    prop_assert!(reduction.verify(transcript(), public, tampered).is_err());

    // Many proofs, one accumulator, one decider.
    let (publics, witnesses): (Vec<_>, Vec<_>) = statements.into_iter().unzip();
    let halo = Then(Then(Each(reduction), Fold::new(pcs)), Decide::new(pcs));
    let out = laws::complete::<F, _, _>(&halo, &transcript, &publics, &witnesses);
    prop_assert!(out.is_ok(), "accumulate: {:?}", out.err());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3))]

    #[test]
    fn on_ipa(scale in 1u64..1000, starts in prop::collection::vec(any::<u64>(), 1..3)) {
        let c = circuit::<Fq>(ROWS);
        let pcs = ipa(max_coefficients(&c));
        laws_on(&c, &pcs, poseidon_vesta, scale, &starts)?;
        laws_on(&c, &pcs, kimchi_vesta, scale, &starts)?;
        laws_on(&c, &pcs, hash_vesta, scale, &starts)?;
    }

    #[test]
    fn on_kzg(scale in 1u64..1000, starts in prop::collection::vec(any::<u64>(), 1..3)) {
        let c = circuit::<Fr>(ROWS);
        let pcs = kzg(max_coefficients(&c));
        laws_on(&c, &pcs, poseidon_bn254, scale, &starts)?;
        laws_on(&c, &pcs, hash_bn254, scale, &starts)?;
    }
}

#[test]
fn the_rotations_read_are_opened() {
    let c = circuit::<Fq>(ROWS);
    let found = rotations(c.variables());
    assert_eq!(found, vec![Rotation::CURR, Rotation::PREV, Rotation::NEXT]);
}

/// Witness column 0 in `[0, n)` on every row but the last (plookup needs
/// its selector off there), the table being fixed column 0.
fn range_check(argument: Argument, n: usize) -> Circuit<Fq, 1, 1> {
    let num_rows = n;
    let witness_columns = 1;
    let fixed = vec![(0..n as u64).map(Fq::from_u64).collect()];
    let mut but_last = vec![Fq::ONE; n];
    but_last[n - 1] = Fq::ZERO;
    let selectors = vec![but_last];
    let variables = vec![witness(0, Rotation::CURR)];
    let gates = vec![];
    let sigma = (0..n).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let circuit = Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    )
    .unwrap();
    let inputs = vec![Sparse::variable(Var::try_new(0).unwrap()).unwrap()];
    let table = vec![0];
    let lookup = Lookup::new(argument, Some(0), inputs, table);
    circuit.with_lookups(vec![lookup]).unwrap()
}

#[test]
fn lookups_accumulate() {
    for argument in [Argument::Plookup, Argument::LogUp] {
        let c = range_check(argument, ROWS);
        let pcs = ipa(max_coefficients(&c));
        let (pk, _) = setup(&c, &pcs).unwrap();
        let reduction = Accumulating::new(&pk, &pcs);
        let inside = vec![
            vec![3u64, 1, 4, 1, 5, 2, 6, 5]
                .into_iter()
                .map(Fq::from_u64)
                .collect(),
        ];
        let snark = Then(reduction, Decide::new(&pcs));
        let out = laws::complete::<Fq, _, _>(&snark, poseidon_vesta, &vec![], &inside);
        assert!(out.is_ok(), "{argument:?}: {:?}", out.err());
    }
}
