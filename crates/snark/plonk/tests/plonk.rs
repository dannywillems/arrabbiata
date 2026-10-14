//! Tests of vanilla PlonK.
//!
//! Organised by what they establish:
//!
//! - completeness: honest proofs of satisfied circuits verify, at every size;
//! - the prover is exactly as strong as the relation: `prove` succeeds iff
//!   the witness satisfies the circuit, on witnesses corrupted outside the
//!   builder;
//! - the verifier rejects every tampered component of a proof, a proof for
//!   another circuit, another statement, or another transcript;
//! - proofs are bound to the commitment scheme, and are deterministic.

use arrabbiata_commit::{PolynomialCommitment, Univariate};
use arrabbiata_field::Field;
use arrabbiata_frontend::{Builder, Circuit, Wire, Witness, build};
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonk::{PlonkError, ipa, ipa::PallasIpa, prove, setup, verify};
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_transcript::typed::Label;
use proptest::prelude::*;

type Point = <PallasIpa as PolynomialCommitment<F>>::Commitment;
type IpaProof = <PallasIpa as PolynomialCommitment<F>>::Proof;

/// One step of a random straight-line program over the wires built so far.
#[derive(Clone, Debug)]
enum Op {
    Witness(u64),
    Constant(u64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    AddConstant(usize, u64),
    Scale(usize, u64),
    /// Expose a wire as a public output.
    Output(usize),
}

fn arb_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        any::<u64>().prop_map(Op::Witness),
        any::<u64>().prop_map(Op::Constant),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Add(i, j)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Sub(i, j)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Mul(i, j)),
        (any::<usize>(), any::<u64>()).prop_map(|(i, c)| Op::AddConstant(i, c)),
        (any::<usize>(), any::<u64>()).prop_map(|(i, c)| Op::Scale(i, c)),
        any::<usize>().prop_map(Op::Output),
    ]
}

fn arb_program() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(arb_op(), 1..24)
}

/// Run the program; the last wire is always a public output. When `lie` is
/// set, that output claims a different value.
fn run<'id>(b: &mut Builder<'id, F>, program: &[Op], lie: bool) {
    let mut wires: Vec<Wire<'id, F>> = vec![b.witness(F::from_u64(1))];
    for op in program {
        let pick = |i: usize| wires[i % wires.len()];
        let w = match op {
            Op::Witness(v) => b.witness(F::from_u64(*v)),
            Op::Constant(c) => b.constant(F::from_u64(*c)),
            Op::Add(i, j) => b.add(&pick(*i), &pick(*j)),
            Op::Sub(i, j) => b.sub(&pick(*i), &pick(*j)),
            Op::Mul(i, j) => b.mul(&pick(*i), &pick(*j)),
            Op::AddConstant(i, c) => b.add_constant(&pick(*i), F::from_u64(*c)),
            Op::Scale(i, c) => b.scale(&pick(*i), F::from_u64(*c)),
            Op::Output(i) => {
                let w = pick(*i);
                let out = b.public_input(w.value());
                b.assert_equal(&w, &out);
                out
            }
        };
        wires.push(w);
    }
    let last = *wires.last().unwrap();
    let claimed = if lie {
        last.value() + F::ONE
    } else {
        last.value()
    };
    let output = b.public_input(claimed);
    b.assert_equal(&last, &output);
}

fn circuit_of(program: &[Op], lie: bool) -> (Circuit<F>, Witness<F>) {
    let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, program, lie));
    (circuit, witness)
}

fn honest(program: &[Op]) -> (Circuit<F>, Witness<F>) {
    circuit_of(program, false)
}

/// IPA on Pallas under another label: same commitment and proof types,
/// different scheme.
struct Twin(PallasIpa);

impl Label for Twin {
    const LABEL: u64 = 0x7477;
}

impl PolynomialCommitment<F> for Twin {
    type Family = Univariate;
    type Commitment = Point;
    type Proof = IpaProof;

    fn commit(&self, p: &[F]) -> Point {
        self.0.commit(p)
    }

    fn open(&self, p: &[F], x: &F) -> (F, IpaProof) {
        self.0.open(p, x)
    }

    fn verify(&self, c: &Point, x: &F, v: F, proof: &IpaProof) -> bool {
        self.0.verify(c, x, v, proof)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Completeness: an honest proof of a satisfied circuit verifies.
    #[test]
    fn completeness(program in arb_program(), seed in any::<u64>()) {
        let (circuit, witness) = honest(&program);
        prop_assert!(circuit.is_satisfied(&witness));
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, vesta_sponge(seed), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, vesta_sponge(seed), witness.public_inputs(), proof).is_ok());
    }

    /// The prover is exactly as strong as the relation: on a witness with
    /// one cell (or one public input) changed outside the builder, `prove`
    /// succeeds iff the witness still satisfies the circuit. A cell that no
    /// gate and no copy constraint touches can change freely.
    #[test]
    fn prover_succeeds_iff_satisfied(
        program in arb_program(),
        column in 0usize..3,
        row in any::<usize>(),
        value in any::<u64>(),
        corrupt_public in any::<bool>(),
    ) {
        let (circuit, honest_witness) = honest(&program);
        let mut wires = honest_witness.wires().clone();
        let mut public = honest_witness.public_inputs().to_vec();
        if corrupt_public {
            let i = row % public.len();
            public[i] = F::from_u64(value);
        } else {
            let row = row % circuit.num_rows();
            wires[column][row] = F::from_u64(value);
        }
        let witness = Witness::new(wires, public);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let satisfied = circuit.is_satisfied(&witness);
        match prove(&pk, &pcs, vesta_sponge(0), &witness) {
            Ok(proof) => {
                prop_assert!(satisfied);
                prop_assert!(verify(&vk, &pcs, vesta_sponge(0), witness.public_inputs(), proof).is_ok());
            }
            Err(e) => {
                prop_assert!(!satisfied);
                prop_assert_eq!(e, PlonkError::Unsatisfied);
            }
        }
    }

    /// A false public output has no proof.
    #[test]
    fn false_statement_has_no_proof(program in arb_program()) {
        let (circuit, witness) = circuit_of(&program, true);
        prop_assert!(!circuit.is_satisfied(&witness));
        let pcs = ipa::pallas(&circuit);
        let (pk, _) = setup(&circuit, &pcs).unwrap();
        prop_assert_eq!(prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap_err(), PlonkError::Unsatisfied);
    }

    /// Changing any claimed evaluation is caught by its opening.
    #[test]
    fn tampered_evaluation_is_rejected(program in arb_program(), index in 0usize..13, delta in 1u64..1000) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        let evaluations = &mut proof.tail.tail.tail.head;
        evaluations.values[index] = evaluations.values[index] + F::from_u64(delta);
        prop_assert_eq!(
            verify(&vk, &pcs, vesta_sponge(0), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// Changing the claimed z(zeta w) is caught by its opening.
    #[test]
    fn tampered_shifted_evaluation_is_rejected(program in arb_program(), delta in 1u64..1000) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        let evaluations = &mut proof.tail.tail.tail.head;
        evaluations.z_shifted = evaluations.z_shifted + F::from_u64(delta);
        prop_assert_eq!(
            verify(&vk, &pcs, vesta_sponge(0), witness.public_inputs(), proof),
            Err(PlonkError::Opening(13))
        );
    }

    /// Replacing a committed polynomial (a wire, z or t) by another one
    /// changes the challenges after it, and the proof no longer verifies.
    #[test]
    fn tampered_commitment_is_rejected(
        program in arb_program(),
        which in 0usize..5,
        delta in 1u64..1000,
    ) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        // The commitment to `p + delta` in place of the one to `p`.
        let bump = |c: &Point| *c + pcs.commit(&[F::from_u64(delta)]);
        match which {
            0..=2 => {
                let wires = &mut proof.head.value.0;
                wires[which] = bump(&wires[which]);
            }
            3 => proof.tail.head.value = bump(&proof.tail.head.value),
            _ => proof.tail.tail.head.value = bump(&proof.tail.tail.head.value),
        }
        prop_assert!(verify(&vk, &pcs, vesta_sponge(0), witness.public_inputs(), proof).is_err());
    }

    /// A proof is bound to its statement.
    #[test]
    fn wrong_public_input_is_rejected(program in arb_program(), delta in 1u64..1000) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        let mut wrong = witness.public_inputs().to_vec();
        let last = wrong.len() - 1;
        wrong[last] = wrong[last] + F::from_u64(delta);
        prop_assert!(verify(&vk, &pcs, vesta_sponge(0), &wrong, proof).is_err());
    }

    /// Public inputs are positional: swapping two distinct ones is another
    /// statement.
    #[test]
    fn reordered_public_inputs_are_rejected(program in arb_program()) {
        let (circuit, witness) = honest(&program);
        let public = witness.public_inputs();
        let pair = (0..public.len())
            .flat_map(|i| (i + 1..public.len()).map(move |j| (i, j)))
            .find(|&(i, j)| public[i] != public[j]);
        prop_assume!(pair.is_some());
        let (i, j) = pair.unwrap();
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        let mut swapped = public.to_vec();
        swapped.swap(i, j);
        prop_assert!(verify(&vk, &pcs, vesta_sponge(0), &swapped, proof).is_err());
    }

    /// A wrong number of public inputs is rejected before anything else.
    #[test]
    fn public_input_count_is_checked(program in arb_program()) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        let mut more = witness.public_inputs().to_vec();
        more.push(F::ONE);
        prop_assert_eq!(verify(&vk, &pcs, vesta_sponge(0), &more, proof), Err(PlonkError::PublicInputs));
    }

    /// A proof is bound to its transcript.
    #[test]
    fn transcript_mismatch_is_rejected(program in arb_program()) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, vesta_sponge(1), witness.public_inputs(), proof).is_err());
    }

    /// Proving is deterministic.
    #[test]
    fn proving_is_deterministic(program in arb_program(), seed in any::<u64>()) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, _) = setup(&circuit, &pcs).unwrap();
        let a = prove(&pk, &pcs, vesta_sponge(seed), &witness).unwrap();
        let b = prove(&pk, &pcs, vesta_sponge(seed), &witness).unwrap();
        prop_assert_eq!(a, b);
    }

    /// The scheme is part of the transcript: proving the same witness under
    /// two schemes with identical types draws different challenges, so the
    /// claimed evaluations differ.
    #[test]
    fn the_scheme_separates_transcripts(program in arb_program()) {
        let (circuit, witness) = honest(&program);
        let pcs = ipa::pallas(&circuit);
        let twin = Twin(ipa::pallas(&circuit));
        let (pk, _) = setup(&circuit, &pcs).unwrap();
        let (twin_pk, _) = setup(&circuit, &twin).unwrap();
        let a = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
        let b = prove(&twin_pk, &twin, vesta_sponge(0), &witness).unwrap();
        // Same commitments (same polynomials, same commitment function)...
        prop_assert_eq!(&a.head.value, &b.head.value);
        // ...but different challenges from the first one on.
        prop_assert_ne!(&a.tail.head.value, &b.tail.head.value);
    }
}

/// Completeness at every size from 2 to 128 rows.
#[test]
fn every_size_verifies() {
    for rows in 1..=127u64 {
        let (circuit, witness, ()) = build::<F, _, _>(|b| {
            let mut acc = b.witness(F::from_u64(rows));
            for i in 1..rows {
                acc = b.add_constant(&acc, F::from_u64(i));
            }
            let out = b.public_input(acc.value());
            b.assert_equal(&acc, &out);
        });
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, vesta_sponge(rows), &witness).unwrap();
        assert!(
            verify(
                &vk,
                &pcs,
                vesta_sponge(rows),
                witness.public_inputs(),
                proof
            )
            .is_ok(),
            "{rows} gates, {} rows",
            circuit.num_rows()
        );
    }
}

/// A proof for one circuit does not verify against another circuit of the
/// same size and the same statement.
#[test]
fn proof_is_bound_to_its_circuit() {
    let make = |multiply: bool| {
        build::<F, _, _>(move |b| {
            let x = b.witness(F::from_u64(2));
            let y = b.witness(F::from_u64(2));
            let z = if multiply {
                b.mul(&x, &y)
            } else {
                b.add(&x, &y)
            };
            let out = b.public_input(F::from_u64(4));
            b.assert_equal(&z, &out);
        })
    };
    let (mul_circuit, mul_witness, ()) = make(true);
    let (add_circuit, _, ()) = make(false);
    assert_eq!(mul_circuit.num_rows(), add_circuit.num_rows());
    let pcs = ipa::pallas(&mul_circuit);
    let (mul_pk, mul_vk) = setup(&mul_circuit, &pcs).unwrap();
    let (_, add_vk) = setup(&add_circuit, &pcs).unwrap();
    let proof = prove(&mul_pk, &pcs, vesta_sponge(0), &mul_witness).unwrap();
    assert!(
        verify(
            &mul_vk,
            &pcs,
            vesta_sponge(0),
            &[F::from_u64(4)],
            proof.clone()
        )
        .is_ok()
    );
    assert!(verify(&add_vk, &pcs, vesta_sponge(0), &[F::from_u64(4)], proof).is_err());
}

/// Known answer: x^3 + x + 5 = 35 for x = 3, the example of the PlonK
/// literature.
#[test]
fn cubic_known_answer() {
    let statement = |x: u64, y: u64| {
        build::<F, _, _>(move |b| {
            let x = b.witness(F::from_u64(x));
            let x2 = b.mul(&x, &x);
            let x3 = b.mul(&x2, &x);
            let sum = b.add(&x3, &x);
            let lhs = b.add_constant(&sum, F::from_u64(5));
            let rhs = b.public_input(F::from_u64(y));
            b.assert_equal(&lhs, &rhs);
        })
    };
    let (circuit, witness, ()) = statement(3, 35);
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap();
    assert!(verify(&vk, &pcs, vesta_sponge(0), &[F::from_u64(35)], proof).is_ok());
    // x = 2 gives 15, not 35.
    let (circuit, witness, ()) = statement(2, 35);
    let (pk, _) = setup(&circuit, &pcs).unwrap();
    assert_eq!(
        prove(&pk, &pcs, vesta_sponge(0), &witness).unwrap_err(),
        PlonkError::Unsatisfied
    );
}
