//! The behaviour of each public item, one test per item.

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_ipa::generic::{BaseOracle, ChallengeMap, HashToCurve, IpaOracle, Opening, Schnorr};
use arrabbiata_kimchi::{
    accumulation::{
        self, AccumulateProofs, Assignment, KimchiSuccinct, Satisfied, Separate, Statement, Verify,
        accumulate_proofs, ipa_of, separate,
    },
    constraints::{
        self, Builder, COLUMNS, CircuitGate, CircuitGateError, Connect, ConstraintSystem,
        EvaluationDomains, GateError, GateTable, GateType, LookupTable, RANGE_CHECK_TABLE_SIZE,
        RuntimeTableCfg, Wire, XOR_TABLE_SIZE, zk_rows_strict_lower_bound,
    },
    gates::{Gates, KimchiGates},
    ipa::{KimchiIpa, kimchi_ipa, kimchi_ipa_with},
    lookup::{
        self as lookups, JointLookup, LookupError, LookupIndex, LookupInfo, LookupPattern,
        LookupPoint, LookupTableId, PATTERNS, RANGE_CHECK_TABLE_ID, RuntimeTable, XOR_TABLE_ID,
        combine_entry, constraint_count, id_to_field, lookup_value, range_check_table, xor_table,
    },
    oracle::{CHALLENGE_LIMBS, FqSponge, FrSponge, ScalarChallenge},
    permutation::{PERMUTS, shifts},
    polynomials::{
        self as gadgets,
        and::{create_and_witness, extend_and_witness},
        bit_length,
        generic::{
            GenericGateSpec,
            testing::{create_circuit, fill_in_witness},
        },
        not::{extend_not_witness_checked_length, extend_not_witness_unchecked_length},
        range_check::{
            create as create_range_check_witness, create_multi, create_multi_compact,
            create_multi_compact_limbs, create_multi_limbs, extend as extend_range_check_witness,
            extend_multi, extend_multi_compact, extend_multi_compact_limbs, extend_multi_limbs,
        },
        rot::{RotMode, extend_rot},
        xor::{create_xor_witness, extend_xor_witness, num_xors},
    },
    prover::{
        self, EvaluationRound, Hidden, LOOKUP_ALPHA, LookupCommitments, LookupEvaluations,
        LookupRound, OpeningProof, PERMUTATION_ALPHA, PermutationRound, PointEvaluations,
        ProofEvaluations, ProverIndex, ProverProof, QuotientRound, WitnessRound, evaluation_round,
        opening_round, permutation_round, prove, quotient_round, shift_scalar, witness_round,
    },
    rng::{Arkworks, RandomSource, Sampler, SplitMix64},
    srs::{GroupMap, PolyComm, Srs, packed, point_of_random_bytes},
    transcript::{
        self, AfterGamma, Base, Beta, Challenge, Digest, DigestOf, FqTape, Message, Piop, Scalar,
        VerifierIndexDigest,
    },
    verifier::{Claim, Oracles, VerifyError, oracles, succinct, verify},
    verifier_index::{LookupVerifierIndex, VerifierIndex},
};
use arrabbiata_kimchi_gadgets::{generic::GenericSpec, zero::Zero};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};
use arrabbiata_reduction::{Reduction, Relation};
use arrabbiata_transcript::typed::Label;

#[test]
fn challenge_limbs() {
    assert_eq!(CHALLENGE_LIMBS, 2);
}

#[test]
fn scalar_challenge() {
    let lambda = Vesta::lambda();
    let zero = ScalarChallenge::<Fp>::from_limbs([0, 0]);
    assert_eq!(
        zero.to_field(lambda),
        zero.to_field_with_length(128, lambda)
    );
    // Two zero bits: from a = b = 2, double both, then -1 on b.
    let two = zero.to_field_with_length(2, lambda);
    assert_eq!(two, Fp::from_u64(4) * lambda + Fp::from_u64(3));
}

#[test]
fn fq_sponge() {
    // A challenge is 128 bits: its limbs past the second are zero.
    let mut sponge = FqSponge::<Vesta, _>::new(KimchiSponge::new(vesta_base()));
    sponge.absorb_g(&[Projective::<Vesta>::generator()]);
    let c = sponge.challenge();
    assert_eq!((c.limb(2), c.limb(3)), (0, 0));
    assert_ne!(sponge.digest(), Fp::ZERO);
}

#[test]
fn fr_sponge() {
    let mut sponge = FrSponge::<Fp, _>::new(KimchiSponge::new(pallas_base()));
    sponge.absorb(Fp::ONE);
    let c = sponge.challenge();
    assert_eq!((c.0.limb(2), c.0.limb(3)), (0, 0));
    assert_ne!(sponge.digest(), Fp::ZERO);
}

#[test]
fn permuts() {
    assert_eq!(PERMUTS, 7);
}

#[test]
fn shifts_of_a_domain() {
    // The first shift is 1, the others are non-residues, and the sampling
    // depends on the domain only through the check `k^n != 1`, which a
    // non-residue passes for every power-of-two `n` below 2^32: so the
    // shifts are the same for every size Kimchi uses.
    let small = shifts::<Fp>(4);
    let large = shifts::<Fp>(10);
    assert_eq!(small[0], Fp::ONE);
    assert_eq!(small, large);
}

/// Two Generic rows, wired to themselves.
fn two_generic() -> Vec<CircuitGate<Fp>> {
    (0..2)
        .map(|row| CircuitGate::new(GateType::Generic, Wire::for_row(row), vec![Fp::ONE]))
        .collect()
}

#[test]
fn columns() {
    assert_eq!(COLUMNS, 15);
}

#[test]
fn table_sizes() {
    assert_eq!(RANGE_CHECK_TABLE_SIZE, 4096);
    assert_eq!(XOR_TABLE_SIZE, 256);
}

#[test]
fn gate_type() {
    assert_eq!(GateType::default(), GateType::Zero);
    assert_eq!(GateType::Rot64.table(), Some(GateTable::RangeCheck));
    assert_eq!(
        GateType::ForeignFieldMul.table(),
        Some(GateTable::RangeCheck)
    );
    assert_eq!(GateType::Xor16.table(), Some(GateTable::Xor));
    assert_eq!(GateType::Lookup.table(), None);
    assert!(GateType::Rot64.is_optional());
    assert!(!GateType::Poseidon.is_optional());
}

#[test]
fn gate_table() {
    assert_ne!(GateTable::RangeCheck, GateTable::Xor);
}

#[test]
fn wire() {
    let wires = Wire::for_row(5);
    assert_eq!(wires[3], Wire { row: 5, col: 3 });
}

#[test]
fn circuit_gate() {
    let gate = CircuitGate::new(GateType::Zero, Wire::for_row(0), vec![Fp::ONE]);
    assert_eq!(gate.typ, GateType::Zero);
    assert_eq!(gate.coeffs, vec![Fp::ONE]);
}

#[test]
fn connect() {
    // Two fixed points become a 2-cycle.
    let mut gates = two_generic();
    let first = (0, 1);
    let second = (1, 4);
    gates.connect_cell_pair(first, second);
    assert_eq!(gates[0].wires[1], Wire { row: 1, col: 4 });
    assert_eq!(gates[1].wires[4], Wire { row: 0, col: 1 });
}

#[test]
fn lookup_tables_count_in_the_domain() {
    // 8 Lookup rows; 40 + 30 table rows and the dummy row, plus one: 72,
    // plus 3 zk rows: 75, so 128 rows.
    let gates = (0..8)
        .map(|row| CircuitGate::new(GateType::Lookup, Wire::for_row(row), vec![]))
        .collect();
    let table = LookupTable::<Fp> {
        id: 1,
        data: vec![vec![Fp::ZERO; 40]],
    };
    let runtime = RuntimeTableCfg::<Fp> {
        id: 2,
        first_column: vec![Fp::ZERO; 30],
    };
    let cs = ConstraintSystem::create(gates)
        .lookup(vec![table])
        .runtime(Some(vec![runtime]))
        .build()
        .unwrap();
    assert_eq!(cs.domain.d1.size(), 128);
}

#[test]
fn evaluation_domains() {
    let d = EvaluationDomains::<Fp>::create(9).unwrap();
    assert_eq!(d.d1.size(), 16);
    assert_eq!(d.d8.size(), 128);
    assert_eq!(d.d2.generator() * d.d2.generator(), d.d1.generator());
    assert_eq!(d.d8.generator() * d.d8.generator(), d.d4.generator());
}

#[test]
fn error() {
    let one = vec![CircuitGate::<Fp>::new(
        GateType::Zero,
        Wire::for_row(0),
        vec![],
    )];
    let too_few = ConstraintSystem::create(one).build().unwrap_err();
    assert_eq!(too_few, constraints::Error::TooFewGates);
    assert_eq!(too_few.to_string(), "a circuit needs at least two gates");
    let mut wired_out = two_generic();
    wired_out[0].wires[0] = Wire { row: 2, col: 0 };
    assert_eq!(
        ConstraintSystem::create(wired_out).build().unwrap_err(),
        constraints::Error::InvalidWire { row: 2, col: 0 }
    );
    let mut crowded = two_generic();
    crowded[1].coeffs = vec![Fp::ONE; COLUMNS + 1];
    assert_eq!(
        ConstraintSystem::create(crowded).build().unwrap_err(),
        constraints::Error::TooManyCoefficients { row: 1 }
    );
}

#[test]
fn zk_rows_bound() {
    // Kimchi's comment: one chunk needs more than 2 rows, so 3.
    assert_eq!(zk_rows_strict_lower_bound(1), 2);
    assert_eq!(zk_rows_strict_lower_bound(4), 8);
}

#[test]
fn builder() {
    let builder: Builder<Fp> = ConstraintSystem::create(two_generic());
    let cs = builder
        .public(1)
        .prev_challenges(2)
        .max_poly_size(Some(4))
        .build()
        .unwrap();
    assert_eq!((cs.public, cs.prev_challenges), (1, 2));
    // Chunks of 4 rows: 2 gates and 3 rows give n = 8, so c = 2 chunks and
    // zk_rows = 30 / 7 + 1 = 5; the 7 rows fit in 8, the fixpoint.
    assert_eq!((cs.domain.d1.size(), cs.zk_rows), (8, 5));
    assert_eq!(cs.max_poly_size, Some(4));
}

#[test]
fn constraint_system() {
    let cs = ConstraintSystem::create(two_generic()).build().unwrap();
    // Two gates and three zk rows fit in 8 rows; the rest are Zero gates.
    assert_eq!(cs.gates.len(), 8);
    assert_eq!(cs.gates[5].typ, GateType::Zero);
    assert!(cs.uses(GateType::Generic));
    assert!(!cs.uses(GateType::Zero));
    let label = cs.cell_to_field(Wire { row: 1, col: 2 });
    assert_eq!(label, cs.shift[2] * cs.domain.d1.generator());
    assert_eq!(cs.sigmas()[2][1], label);
    assert_eq!(
        cs.selector(GateType::Generic)[..3],
        [Fp::ONE, Fp::ONE, Fp::ZERO]
    );
    assert_eq!(cs.coefficient(0)[..3], [Fp::ONE, Fp::ONE, Fp::ZERO]);
    // Extending a column keeps its values on d1, every eighth point of d8.
    let column = cs.coefficient(0);
    let wide = cs.extend(&column, &cs.domain.d8).unwrap();
    assert_eq!(wide[8], column[1]);
    assert!(cs.extend(&column[1..], &cs.domain.d8).is_none());
}

#[test]
fn group_map() {
    // Every image is on the curve (Affine::new checks it), and for each t
    // at least one candidate is.
    let map = GroupMap::<Vesta>::setup();
    for k in 0..20 {
        let t = Fq::from_u64(k);
        let p = map.to_group(t);
        assert!(p.to_projective().is_on_curve());
        let xs = map.potential_xs(t);
        assert!(xs.contains(&p.x().unwrap()));
    }
}

#[test]
fn packing() {
    // Bit 0 of byte 0 is the top bit, 2^247; bit 7 of byte 30 is 2^0.
    let mut bytes = [0u8; 31];
    bytes[0] = 1;
    let top: Fq = packed(&bytes);
    assert_eq!(top.limb(3), 1 << (247 - 192));
    let mut bytes = [0u8; 31];
    bytes[30] = 0x80;
    assert_eq!(packed::<Fq>(&bytes), Fq::ONE);
}

#[test]
fn point_of_bytes() {
    let map = GroupMap::<Vesta>::setup();
    let bytes = [7u8; 64];
    let p = point_of_random_bytes(&map, &bytes);
    assert_eq!(p, map.to_group(packed(&bytes)).to_projective());
}

#[test]
fn srs_and_commitments() {
    let srs = Srs::<Vesta>::create(4);
    assert_eq!(srs.max_poly_size(), 4);
    assert_eq!(srs.g.len(), 4);
    assert!(!srs.g.contains(&srs.h));
    // Six coefficients over 8 rows: two chunks, the second the high two.
    let coefficients: Vec<Fp> = (1..=6).map(Fp::from_u64).collect();
    let rows = 8;
    let c: PolyComm<Vesta> = srs.commit_non_hiding(&coefficients, rows);
    assert_eq!(c.chunks.len(), 2);
    assert_eq!(
        c.chunks[1],
        srs.g[0] * Fp::from_u64(5) + srs.g[1] * Fp::from_u64(6)
    );
    // A zero polynomial still has one chunk per block of rows.
    let zero = srs.commit_non_hiding(&[], rows);
    assert!(zero.chunks.iter().all(Projective::is_identity));
    let masked = srs.mask_fixed(zero);
    assert_eq!(masked.chunks, vec![srs.h, srs.h]);
}

#[test]
fn verifier_index() {
    let cs = ConstraintSystem::create(two_generic()).build().unwrap();
    let srs = Srs::<Vesta>::create(cs.domain.d1.size());
    let index = VerifierIndex::create(&cs, &srs);
    assert_eq!(index.domain_size, 8);
    assert_eq!(index.shift, cs.shift);
    // No Poseidon row: the masked selector is h alone.
    assert_eq!(index.psm_comm.chunks, vec![srs.h]);
    assert!(index.rot_comm.is_none());
}

#[test]
fn split_mix_64() {
    // The first output of SplitMix64(0) (Steele, Lea, Flood).
    let mut rng = SplitMix64(0);
    assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
}

#[test]
fn arkworks_sampler() {
    // A zero stream is the zero element; samples are in the field.
    struct Zeros;
    impl RandomSource for Zeros {
        fn next_u64(&mut self) -> u64 {
            0
        }
    }
    let zero: Fp = Arkworks(Zeros).sample();
    assert_eq!(zero, Fp::ZERO);
    let mut rng = Arkworks(SplitMix64(1));
    let a: Fp = rng.sample();
    let b: Fp = rng.sample();
    assert_ne!(a, b);
}

#[test]
fn verifier_index_digest() {
    // The digest depends on the commitments.
    let srs = Srs::<Vesta>::create(8);
    let one = ConstraintSystem::create(two_generic()).build().unwrap();
    let mut other_gates = two_generic();
    other_gates[1].coeffs = vec![Fp::ONE + Fp::ONE];
    let other = ConstraintSystem::create(other_gates).build().unwrap();
    let sponge = KimchiSponge::new(vesta_base());
    let a = VerifierIndex::create(&one, &srs).digest(sponge.clone());
    let b = VerifierIndex::create(&other, &srs).digest(sponge);
    assert_ne!(a, b);
}

#[test]
fn prover_index() {
    let cs = ConstraintSystem::create(two_generic()).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(4));
    // 8 rows in chunks of 4.
    assert_eq!(index.num_chunks(), 2);
    assert_eq!(index.verifier_index.max_poly_size, 4);
}

#[test]
fn prover_errors() {
    let cs = ConstraintSystem::create(two_generic()).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let sponge = KimchiSponge::new(vesta_base());
    let mut rng = Arkworks(SplitMix64(3));
    let mut ragged: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    ragged[4].push(Fp::ZERO);
    let error = witness_round(&index, ragged, &[], &[], &sponge, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::WitnessCsInconsistent);
    assert_eq!(error.to_string(), "witness columns of different lengths");
    // Six rows leave two of eight free, fewer than the three zk rows.
    let long: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 6]);
    let error = witness_round(&index, long, &[], &[], &sponge, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::NoRoomForZkInWitness);
    // Chunks of 4 rows need more zk rows than the 3 built for one chunk.
    let one_chunk = ConstraintSystem::create(two_generic()).build().unwrap();
    let chunked = ProverIndex::create(one_chunk, Srs::<Vesta>::create(4));
    let short: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    let error = witness_round(&chunked, short, &[], &[], &sponge, &mut rng).unwrap_err();
    assert_eq!(
        error,
        prover::Error::NotZeroKnowledge {
            needed: 5,
            found: 3
        }
    );
}

#[test]
fn witness_round_state() {
    let cs = ConstraintSystem::create(two_generic())
        .public(1)
        .build()
        .unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let sponge = KimchiSponge::new(vesta_base());
    let mut rng = Arkworks(SplitMix64(3));
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    witness[0][0] = Fp::ONE;
    let (round, _): (WitnessRound<Vesta>, _) =
        witness_round(&index, witness, &[], &[], &sponge, &mut rng).unwrap();
    assert_eq!(round.public, vec![Fp::ONE]);
    // Padded to 8 rows, the last three random, the others zero.
    assert_eq!(round.witness[3].len(), 8);
    assert_eq!(round.witness[3][4], Fp::ZERO);
    assert_ne!(round.witness[3][7], Fp::ZERO);
    // A commitment is the column's plus h times its blinder.
    let blinded = round.w_comm[3].chunks[0];
    assert_ne!(blinded, index.srs.h * round.blinders[3][0]);
    assert_ne!(round.beta, round.gamma);
}

#[test]
fn permutation_round_state() {
    // Two Generic rows with cell (0, 0) copied to (1, 0).
    let mut gates = two_generic();
    gates.connect_cell_pair((0, 0), (1, 0));
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let sponge = KimchiSponge::new(vesta_base());
    let equal: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ONE; 2]);
    let mut rng = Arkworks(SplitMix64(5));
    let (first, tape) = witness_round(&index, equal, &[], &[], &sponge, &mut rng).unwrap();
    let (round, _): (PermutationRound<Vesta>, _) =
        permutation_round(&index, first, tape, &mut rng).unwrap();
    // z starts at 1 and closes at row n - zk_rows = 5.
    assert_eq!(round.z[0], Fp::ONE);
    assert_eq!(round.z[5], Fp::ONE);
    assert_eq!(round.z_blinders.len(), 1);
    assert_eq!(round.alpha, round.alpha_chal.to_field(Vesta::lambda()));
    // Breaking the copy breaks the product.
    let mut broken: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ONE; 2]);
    broken[0][1] = Fp::ZERO;
    let (first, tape) = witness_round(&index, broken, &[], &[], &sponge, &mut rng).unwrap();
    let error = permutation_round(&index, first, tape, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::PermutationFinalValue);
}

#[test]
fn kimchi_gates() {
    // The Generic gate: two constraints, the first l c0 + r c1 + o c2 +
    // l r c3 + c4 with the coefficients 0-4 and the cells 0-2.
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let witness = |c: usize, _: usize| Fp::from_u64(c as u64 + 1);
    let coefficient = |c: usize, _: usize| if c == 4 { Fp::ONE } else { Fp::ZERO };
    let values = gates.evaluate(GateType::Generic, &witness, &coefficient);
    assert_eq!(values.len(), 2);
    assert_eq!(values[0], Fp::ONE);
    assert!(
        gates
            .evaluate(GateType::Zero, &witness, &coefficient)
            .is_empty()
    );
    assert_eq!(
        gates
            .evaluate(GateType::VarBaseMul, &witness, &coefficient)
            .len(),
        PERMUTATION_ALPHA
    );
}

#[test]
fn quotient_round_state() {
    // Two Generic rows asserting l + r - o = 0 on (1, 2, 3).
    let gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            let coeffs = vec![Fp::ONE, Fp::ONE, -Fp::ONE];
            CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs)
        })
        .collect();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let sponge = KimchiSponge::new(vesta_base());
    let good: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        let value = if c < 3 {
            Fp::from_u64(c as u64 + 1)
        } else {
            Fp::ZERO
        };
        vec![value; 2]
    });
    let mut rng = Arkworks(SplitMix64(9));
    let (first, tape) = witness_round(&index, good.clone(), &[], &[], &sponge, &mut rng).unwrap();
    let (second, tape) = permutation_round(&index, first, tape, &mut rng).unwrap();
    let round: QuotientRound<Vesta, _> =
        quotient_round(&index, second, tape, &kimchi, &mut rng).unwrap();
    // 7 chunks of 8 rows, each hidden by its blinder.
    assert_eq!(round.t_comm.chunks.len(), 7);
    assert_eq!(round.t_blinders.len(), 7);
    assert!(round.t.coefficients().len() <= 7 * 8);
    assert_eq!(round.zeta, round.zeta_chal.to_field(Vesta::lambda()));
    // A broken addition leaves a remainder.
    let mut bad = good;
    bad[2][1] = Fp::from_u64(4);
    let (first, tape) = witness_round(&index, bad, &[], &[], &sponge, &mut rng).unwrap();
    let (second, tape) = permutation_round(&index, first, tape, &mut rng).unwrap();
    let error = quotient_round(&index, second, tape, &kimchi, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::QuotientRemainder);
    assert_eq!(
        error.to_string(),
        "the constraints do not vanish on the domain"
    );
}

/// A proof's rounds up to the evaluations, on two satisfied Generic rows.
fn evaluated() -> (
    ProverIndex<Vesta>,
    EvaluationRound<Vesta, KimchiSponge<Fq>, KimchiSponge<Fp>>,
) {
    let gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            let coeffs = vec![Fp::ONE, Fp::ONE, -Fp::ONE];
            CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs)
        })
        .collect();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let sponge = KimchiSponge::new(vesta_base());
    let scalar_sponge = KimchiSponge::new(pallas_base());
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        let value = if c < 3 {
            Fp::from_u64(c as u64 + 1)
        } else {
            Fp::ZERO
        };
        vec![value; 2]
    });
    let mut rng = Arkworks(SplitMix64(11));
    let (first, tape) = witness_round(&index, witness, &[], &[], &sponge, &mut rng).unwrap();
    let (second, tape) = permutation_round(&index, first, tape, &mut rng).unwrap();
    let third = quotient_round(&index, second, tape, &kimchi, &mut rng).unwrap();
    let round = evaluation_round(&index, third, &scalar_sponge);
    (index, round)
}

#[test]
fn point_and_proof_evaluations() {
    let (_, round) = evaluated();
    let e: &ProofEvaluations<Fp> = &round.evals;
    // One chunk each; the witness value 1 of column 0 on every row but
    // the zk rows, so not 1 at zeta.
    let w0: &PointEvaluations<Fp> = &e.w[0];
    assert_eq!(w0.zeta.len(), 1);
    assert_eq!(w0.zeta_omega.len(), 1);
    // The absorption order: z, six selectors, 15 + 15 + 6 columns.
    let absorbed = e.absorbed();
    assert_eq!(absorbed.len(), 7 + 15 + 15 + 6);
    assert_eq!(absorbed[0], &e.z);
    assert_eq!(absorbed[7], &e.w[0]);
}

#[test]
fn evaluation_round_state() {
    let (index, round) = evaluated();
    let zeta = round.previous.zeta;
    assert_eq!(round.zeta_omega, zeta * index.cs.domain.d1.generator());
    assert_eq!(round.ft_eval1, round.ft.evaluate(round.zeta_omega));
    // ft has degree below the chunk size.
    assert!(round.ft.coefficients().len() <= index.srs.max_poly_size());
    assert_eq!(round.v, round.v_chal.to_field(Vesta::lambda()));
    assert_eq!(round.u, round.u_chal.to_field(Vesta::lambda()));
    assert_ne!(round.blinding_ft, Fp::ZERO);
}

#[test]
fn shift_scalar_inverts_kimchis_unshift() {
    // On Vesta the scalar modulus is the smaller: shift(x) = (x - 2^255 -
    // 1) / 2, so 2 shift(x) + 2^255 + 1 = x.
    let x = Fp::from_u64(12345);
    let two = Fp::from_u64(2);
    let shifted = shift_scalar::<Vesta>(x);
    assert_eq!(shifted * two + two.pow(255) + Fp::ONE, x);
}

#[test]
fn opening_round_and_proof() {
    let (index, round) = evaluated();
    let map = GroupMap::<Vesta>::setup();
    let mut rng = Arkworks(SplitMix64(13));
    let proof: ProverProof<Vesta> = opening_round(&index, &round, &map, &mut rng);
    let opening: &OpeningProof<Vesta> = &proof.proof;
    // log2(8) rounds; sg on the curve; the proof carries the rounds' parts.
    assert_eq!(opening.lr.len(), 3);
    assert!(opening.sg.is_on_curve());
    assert_eq!(proof.ft_eval1, round.ft_eval1);
    assert_eq!(proof.t_comm, round.previous.t_comm);
}

#[test]
fn prove_runs_every_round() {
    let gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            let coeffs = vec![Fp::ONE, Fp::ONE, -Fp::ONE];
            CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs)
        })
        .collect();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        let value = if c < 3 {
            Fp::from_u64(c as u64 + 1)
        } else {
            Fp::ZERO
        };
        vec![value; 2]
    });
    let sponges = (&base, &scalar);
    let mut one = Arkworks(SplitMix64(17));
    let mut two = Arkworks(SplitMix64(17));
    let a = prove(
        &index,
        witness.clone(),
        &[],
        &[],
        &kimchi,
        sponges,
        &mut one,
    )
    .unwrap();
    let b = prove(
        &index,
        witness.clone(),
        &[],
        &[],
        &kimchi,
        sponges,
        &mut two,
    )
    .unwrap();
    // Reproducible from the stream; another stream hides differently.
    assert_eq!(a, b);
    let mut three = Arkworks(SplitMix64(18));
    let c = prove(
        &index,
        witness.clone(),
        &[],
        &[],
        &kimchi,
        sponges,
        &mut three,
    )
    .unwrap();
    assert_ne!(a.w_comm, c.w_comm);
    assert!(a.prev_challenges.is_empty());
    // A previous accumulator is carried, and changes the transcript.
    let chals = vec![Fp::from_u64(2), Fp::from_u64(3), Fp::from_u64(5)];
    let b = arrabbiata_kimchi::srs::Srs::<Vesta>::create(8)
        .commit_non_hiding(&arrabbiata_ipa::b_poly_coefficients(&chals), 8);
    let previous = arrabbiata_ipa::RecursionChallenge {
        chals,
        comm: b.chunks[0],
    };
    let mut four = Arkworks(SplitMix64(17));
    let previous = [previous];
    let d = prove(&index, witness, &[], &previous, &kimchi, sponges, &mut four).unwrap();
    assert_eq!(d.prev_challenges, previous.to_vec());
    assert_eq!(d.w_comm, a.w_comm);
    assert_ne!(d.proof, a.proof);
}

#[test]
fn lookup_patterns() {
    assert_eq!(PATTERNS[0], LookupPattern::Xor);
    assert_eq!(
        LookupPattern::from_gate(GateType::RangeCheck1, true),
        Some(LookupPattern::RangeCheck)
    );
    assert_eq!(LookupPattern::from_gate(GateType::Xor16, true), None);
    assert_eq!(LookupPattern::Xor.table_id(), Some(XOR_TABLE_ID));
    assert_eq!(LookupPattern::Lookup.table_id(), None);
    assert_eq!(LookupPattern::Lookup.max_lookups_per_row(), 3);
    assert_eq!(LookupPattern::Xor.max_joint_size(), 3);
    let xor: Vec<JointLookup> = LookupPattern::Xor.lookups();
    assert_eq!(xor[1].entry, vec![4, 8, 12]);
    assert_eq!(xor[1].table_id, LookupTableId::Constant(XOR_TABLE_ID));
    let lookup = LookupPattern::Lookup.lookups();
    assert_eq!(lookup[2].table_id, LookupTableId::WitnessColumn(0));
}

#[test]
fn builtin_tables() {
    let xor = xor_table::<Fp>();
    assert_eq!(xor.data[0].len(), 256);
    // Reversed: the last row is the zero row, the first (15, 15, 0).
    assert_eq!(xor.data[2][255], Fp::ZERO);
    assert_eq!(xor.data[0][0], Fp::from_u64(15));
    let range = range_check_table::<Fp>();
    assert_eq!(range.id, RANGE_CHECK_TABLE_ID);
    assert_eq!(range.data[0][4095], Fp::from_u64(4095));
    assert_eq!(id_to_field::<Fp>(-3), -Fp::from_u64(3));
}

#[test]
fn lookup_info_and_index() {
    let gates: Vec<CircuitGate<Fp>> = (0..3)
        .map(|row| CircuitGate::new(GateType::Lookup, Wire::for_row(row), vec![]))
        .collect();
    let info = LookupInfo::from_gates(&gates, false).unwrap();
    assert_eq!(info.patterns, vec![LookupPattern::Lookup]);
    assert!(info.joint_lookup_used());
    assert_eq!(info.by_row(&gates)[1].len(), 3);
    assert!(LookupInfo::from_gates(&two_generic(), false).is_none());
    // A table of id 2: no zero row needed; its ids are a column.
    let table = LookupTable {
        id: 2,
        data: vec![vec![Fp::ONE, Fp::from_u64(2)]],
    };
    let index = LookupIndex::create(&gates, std::slice::from_ref(&table), None, 16, 3)
        .unwrap()
        .unwrap();
    assert_eq!(index.table.len(), 2);
    assert_eq!(index.table_ids.as_ref().unwrap()[0], Fp::from_u64(2));
    assert_eq!(index.selector(LookupPattern::Lookup).unwrap()[2], Fp::ONE);
    assert!(index.selector(LookupPattern::Xor).is_none());
    let colliding = LookupIndex::create(&gates, &[table.clone(), table], None, 16, 3);
    assert_eq!(colliding.unwrap_err(), LookupError::LookupTableIdCollision);
    let bad_zero = LookupTable {
        id: 0,
        data: vec![vec![Fp::ONE]],
    };
    let error = LookupIndex::create(&gates, &[bad_zero], None, 16, 3).unwrap_err();
    assert_eq!(error, LookupError::TableIdZeroMustHaveZeroEntry);
    assert_eq!(
        error.to_string(),
        "the table with id 0 must have a row of zeros"
    );
}

#[test]
fn lookup_verifier_index() {
    let gates: Vec<CircuitGate<Fp>> = (0..3)
        .map(|row| CircuitGate::new(GateType::Lookup, Wire::for_row(row), vec![]))
        .collect();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let srs = Srs::<Vesta>::create(cs.domain.d1.size());
    let index = VerifierIndex::create(&cs, &srs);
    let lookup: &LookupVerifierIndex<Vesta> = index.lookup_index.as_ref().unwrap();
    // Only the dummy zero row: no table of nonzero id, so no id column.
    assert!(lookup.table_ids.is_none());
    assert!(lookup.lookup_selectors[1].is_some());
    assert!(lookup.lookup_selectors[0].is_none());
}

/// A proof of two satisfied Generic rows, with its index.
fn proved() -> (ProverIndex<Vesta>, ProverProof<Vesta>, KimchiGates<Fp>) {
    let gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            let coeffs = vec![Fp::ONE, Fp::ONE, -Fp::ONE];
            CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs)
        })
        .collect();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        let value = if c < 3 {
            Fp::from_u64(c as u64 + 1)
        } else {
            Fp::ZERO
        };
        vec![value; 2]
    });
    let mut rng = Arkworks(SplitMix64(19));
    let proof = prove(
        &index,
        witness,
        &[],
        &[],
        &kimchi,
        (&base, &scalar),
        &mut rng,
    )
    .unwrap();
    (index, proof, kimchi)
}

#[test]
fn verifier_oracles() {
    let (index, proof, kimchi) = proved();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let o: Oracles<Vesta, _> = oracles(
        &index.verifier_index,
        &index.srs,
        &proof,
        &[],
        &kimchi,
        (&base, &scalar),
    )
    .unwrap();
    // ft(zeta w) is the prover's; ft(zeta) the verifier recomputes.
    assert_eq!(o.zeta_omega, o.zeta * index.cs.domain.d1.generator());
    let claims: &[Claim<Vesta>] = &o.evaluations;
    // public, ft, z, 6 selectors, 15 + 15 + 6 columns.
    assert_eq!(claims.len(), 3 + 6 + 15 + 15 + 6);
    assert_eq!(claims[1].1[1], vec![proof.ft_eval1]);
    // No public input: the public commitment is h.
    assert_eq!(o.public_comm.chunks, vec![index.srs.h]);
}

#[test]
fn verifier_accepts_and_refuses() {
    let (index, proof, kimchi) = proved();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let sponges = (&base, &scalar);
    let vi = &index.verifier_index;
    let mut rng = Arkworks(SplitMix64(23));
    assert_eq!(
        verify(vi, &index.srs, &proof, &[], &kimchi, sponges, &mut rng),
        Ok(())
    );
    let mut bad = proof.clone();
    bad.proof.z2 = bad.proof.z2 + Fp::ONE;
    let error = verify(vi, &index.srs, &bad, &[], &kimchi, sponges, &mut rng).unwrap_err();
    assert_eq!(error, VerifyError::OpeningProof);
    assert_eq!(error.to_string(), "the opening does not verify");
    let public = [Fp::ONE];
    let error = verify(vi, &index.srs, &proof, &public, &kimchi, sponges, &mut rng).unwrap_err();
    assert_eq!(error, VerifyError::IncorrectPublicInputLength);
    let mut short = proof.clone();
    short.evals.z.zeta.push(Fp::ZERO);
    let error = verify(vi, &index.srs, &short, &[], &kimchi, sponges, &mut rng).unwrap_err();
    assert_eq!(error, VerifyError::IncorrectLength);
    let mut extra = proof;
    extra.prev_challenges = vec![arrabbiata_ipa::RecursionChallenge {
        chals: vec![Fp::ONE; 3],
        comm: index.srs.h,
    }];
    let error = verify(vi, &index.srs, &extra, &[], &kimchi, sponges, &mut rng).unwrap_err();
    assert_eq!(error, VerifyError::IncorrectPrevChallengesLength);
}

/// Three Lookup rows over a table of id 1: rows (index, value) = (i, 10 i).
fn lookup_circuit() -> (ProverIndex<Vesta>, [Vec<Fp>; COLUMNS]) {
    let gates: Vec<CircuitGate<Fp>> = (0..3)
        .map(|row| CircuitGate::new(GateType::Lookup, Wire::for_row(row), vec![]))
        .collect();
    let table = LookupTable {
        id: 1,
        data: vec![
            (0..4).map(Fp::from_u64).collect(),
            (0..4).map(|i| Fp::from_u64(10 * i)).collect(),
        ],
    };
    let cs = ConstraintSystem::create(gates)
        .lookup(vec![table])
        .build()
        .unwrap();
    let srs = Srs::<Vesta>::create(cs.domain.d1.size());
    // Column 0 the table id; columns 2i + 1, 2i + 2 the index (row + i)
    // mod 4 and its value.
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        (0..3u64)
            .map(|row| match c {
                0 => Fp::ONE,
                1..=6 => {
                    let index = (row + (c as u64 - 1) / 2) % 4;
                    if c % 2 == 1 {
                        Fp::from_u64(index)
                    } else {
                        Fp::from_u64(10 * index)
                    }
                }
                _ => Fp::ZERO,
            })
            .collect()
    });
    (ProverIndex::create(cs, srs), witness)
}

#[test]
fn lookup_values() {
    // sum_i jc^i entry_i + tid id.
    let (jc, tid) = (Fp::from_u64(3), Fp::from_u64(5));
    let entry = [Fp::from_u64(2), Fp::from_u64(7)];
    assert_eq!(
        combine_entry(&entry, Fp::ONE, jc, tid),
        Fp::from_u64(2 + 21 + 5)
    );
    let lookup = &LookupPattern::Lookup.lookups()[0];
    let cell = |c: usize| Fp::from_u64(c as u64 + 1);
    // Table id from column 0 (1), entry columns 1, 2 (2, 3).
    assert_eq!(
        lookup_value(lookup, &cell, jc, tid),
        Fp::from_u64(2 + 9 + 5)
    );
}

#[test]
fn lookup_algorithms() {
    let (index, witness) = lookup_circuit();
    let cs = &index.cs;
    let l = cs.lookup.as_ref().unwrap();
    let n = cs.domain.d1.size();
    let (jc, tid) = (Fp::from_u64(3), Fp::from_u64(9));
    let joint = l.joint_table(jc, tid, None);
    // Row 2 of the table of id 1: (2, 20, id 1).
    assert_eq!(joint[2], Fp::from_u64(2 + 60 + 9));
    let mut padded: Vec<Vec<Fp>> = witness.to_vec();
    for column in &mut padded {
        column.resize(n, Fp::ZERO);
    }
    let sorted = lookups::sorted(
        &l.info,
        &cs.gates,
        &padded,
        &joint,
        jc,
        tid,
        Fp::ZERO,
        cs.zk_rows,
    )
    .unwrap();
    // max_per_row + 1 columns of n - zk_rows values, joined end to end.
    assert_eq!(sorted.len(), 4);
    assert!(sorted.iter().all(|c| c.len() == n - cs.zk_rows));
    let (beta, gamma) = (Fp::from_u64(11), Fp::from_u64(13));
    let z = lookups::aggregation(
        &l.info,
        &cs.gates,
        &padded,
        &joint,
        &sorted,
        (beta, gamma, jc, tid),
        Fp::ZERO,
        cs.zk_rows,
    );
    // The product closes: the sorted multiset is the table and the lookups.
    assert_eq!(z[0], Fp::ONE);
    assert_eq!(z[n - cs.zk_rows - 1], Fp::ONE);
    // A value in no table.
    let mut bad = padded.clone();
    bad[2][0] = Fp::from_u64(11);
    let error = lookups::sorted(
        &l.info,
        &cs.gates,
        &bad,
        &joint,
        jc,
        tid,
        Fp::ZERO,
        cs.zk_rows,
    )
    .unwrap_err();
    assert_eq!(error, LookupError::ValueNotInTable { row: 0 });
    // Runtime tables the index does not configure.
    let runtime = [RuntimeTable {
        id: 2,
        data: vec![Fp::ONE],
    }];
    assert_eq!(
        l.runtime_contribution(&runtime, n).unwrap_err(),
        LookupError::RuntimeTablesInconsistent
    );
}

#[test]
fn lookup_constraints_at_a_point() {
    let (index, _) = lookup_circuit();
    let info = &index.cs.lookup.as_ref().unwrap().info;
    assert_eq!(constraint_count(info), 7);
    // z = 1 and equal sorted values: the boundary and join constraints
    // vanish whatever the weights.
    let point = LookupPoint {
        witness: vec![Fp::ZERO; COLUMNS],
        sorted: vec![(Fp::ONE, Fp::ONE); 4],
        aggregation: (Fp::ONE, Fp::ONE),
        table: (Fp::ZERO, Fp::ZERO),
        selectors: vec![(LookupPattern::Lookup, Fp::ZERO)],
        runtime: None,
        vanishing: Fp::ZERO,
        first: Fp::from_u64(5),
        last: Fp::from_u64(7),
    };
    let values = lookups::constraints(info, &point, Fp::ONE, Fp::ONE, Fp::ONE);
    assert_eq!(values.len(), 7);
    assert!(values.iter().all(|v| *v == Fp::ZERO));
}

#[test]
fn lookup_proofs() {
    let (index, witness) = lookup_circuit();
    assert_eq!(LOOKUP_ALPHA, PERMUTATION_ALPHA + 3);
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(29));
    let (first, tape) = witness_round(&index, witness.clone(), &[], &[], &base, &mut rng).unwrap();
    let round: &LookupRound<Vesta> = first.lookup.as_ref().unwrap();
    // One lookup reads two columns: the joint combiner is squeezed.
    assert_ne!(round.joint_combiner_chal.0, Fp::ZERO);
    let column: &Hidden<Vesta> = &round.sorted[0];
    assert_eq!(column.blinders.len(), 1);
    let (second, _) = permutation_round(&index, first, tape, &mut rng).unwrap();
    assert!(second.aggregation.is_some());
    let mut rng = Arkworks(SplitMix64(29));
    let proof = prove(
        &index,
        witness.clone(),
        &[],
        &[],
        &kimchi,
        (&base, &scalar),
        &mut rng,
    )
    .unwrap();
    let comms: &LookupCommitments<Vesta> = proof.lookup.as_ref().unwrap();
    assert_eq!(comms.sorted.len(), 4);
    assert!(comms.runtime.is_none());
    let evals: &LookupEvaluations<Fp> = proof.evals.lookup.as_ref().unwrap();
    assert_eq!(evals.selectors.len(), 1);
    let mut rng = Arkworks(SplitMix64(31));
    let vi = &index.verifier_index;
    let sponges = (&base, &scalar);
    assert_eq!(
        verify(vi, &index.srs, &proof, &[], &kimchi, sponges, &mut rng),
        Ok(())
    );
    let mut bad = proof.clone();
    bad.evals.lookup = None;
    let error = verify(vi, &index.srs, &bad, &[], &kimchi, sponges, &mut rng).unwrap_err();
    assert_eq!(error, VerifyError::LookupMismatch);
    // A value in no table is refused by the prover.
    let mut wrong = witness;
    wrong[2][1] = Fp::from_u64(12);
    let error = prove(&index, wrong, &[], &[], &kimchi, sponges, &mut rng).unwrap_err();
    assert_eq!(
        error,
        prover::Error::Lookup(LookupError::ValueNotInTable { row: 1 })
    );
}

/// The statement and assignment of [`proved`]'s circuit.
fn statement_of_two_generic() -> (Statement<Vesta>, Assignment<Fp>) {
    let columns: [Vec<Fp>; COLUMNS] = core::array::from_fn(|c| {
        let value = if c < 3 {
            Fp::from_u64(c as u64 + 1)
        } else {
            Fp::ZERO
        };
        vec![value; 2]
    });
    (
        Statement {
            public: vec![],
            prev_challenges: vec![],
        },
        Assignment {
            columns,
            runtime_tables: vec![],
        },
    )
}

#[test]
fn succinct_verifier() {
    let (index, proof, kimchi) = proved();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let vi = &index.verifier_index;
    let accumulator = succinct(vi, &index.srs, &proof, &[], &kimchi, (&base, &scalar)).unwrap();
    assert_eq!(accumulator.comm, proof.proof.sg);
    let mut bad = proof;
    bad.proof.delta = bad.proof.delta + index.srs.h;
    let error = succinct(vi, &index.srs, &bad, &[], &kimchi, (&base, &scalar)).unwrap_err();
    assert_eq!(error, VerifyError::OpeningProof);
}

#[test]
fn accumulation_reductions() {
    let (index, _, kimchi) = proved();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let ipa = ipa_of(&index.srs, scalar.clone());
    assert_eq!(ipa.size(), index.srs.max_poly_size());
    let relation = Satisfied::new(&index, &kimchi, (&base, &scalar), &ipa);
    let (statement, assignment) = statement_of_two_generic();
    assert_eq!(relation.check(&statement, &assignment), Ok(()));
    let mut wrong = assignment.clone();
    wrong.columns[2][1] = Fp::from_u64(4);
    let error = relation.check(&statement, &wrong).unwrap_err();
    assert_eq!(
        error,
        accumulation::Error::Prove(prover::Error::QuotientRemainder)
    );
    assert_eq!(
        error.to_string(),
        "prover: the constraints do not vanish on the domain"
    );
    let public = Statement {
        public: vec![Fp::ONE],
        prev_challenges: vec![],
    };
    assert_eq!(
        relation.check(&public, &assignment),
        Err(accumulation::Error::Public)
    );
    let reduction = KimchiSuccinct::new(relation, Arkworks(SplitMix64(1)));
    assert!(reduction.source().check(&statement, &assignment).is_ok());
    let transcript = || KimchiSponge::new(pallas_base());
    let out = reduction
        .prove(transcript(), &statement, &assignment)
        .unwrap();
    assert!(reduction.target().check(&out.instance, &()).is_ok());
    let carrying = Statement {
        public: vec![],
        prev_challenges: vec![out.instance.clone()],
    };
    let error = reduction
        .verify(transcript(), &carrying, out.proof.clone())
        .unwrap_err();
    assert_eq!(error, accumulation::Error::PrevChallenges);
    let whole: Verify<'_, Vesta, _, _, _, _> = accumulation::verify(reduction);
    assert!(whole.prove(transcript(), &statement, &assignment).is_ok());
    let relation = Satisfied::new(&index, &kimchi, (&base, &scalar), &ipa);
    let reduction = KimchiSuccinct::new(relation, Arkworks(SplitMix64(2)));
    let fold: AccumulateProofs<'_, Vesta, _, _, _, _> = accumulate_proofs(reduction);
    let out = fold
        .prove(transcript(), &vec![statement], &vec![assignment])
        .unwrap();
    assert!(ipa.decide(&out.instance));
}

#[test]
fn accumulation_separate() {
    // A proof without previous accumulators, folded with one: the composite
    // proves, its verifier agrees, and the accumulator is decided.
    let (index, _, kimchi) = proved();
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let ipa = ipa_of(&index.srs, scalar.clone());
    let relation = Satisfied::new(&index, &kimchi, (&base, &scalar), &ipa);
    let (statement, assignment) = statement_of_two_generic();
    let transcript = || KimchiSponge::new(pallas_base());
    let first = KimchiSuccinct::new(relation, Arkworks(SplitMix64(3)));
    let previous = first
        .prove(transcript(), &statement, &assignment)
        .unwrap()
        .instance;
    let folded: Separate<'_, Vesta, _, _, _, _> =
        separate(KimchiSuccinct::new(relation, Arkworks(SplitMix64(4))));
    let instance = (statement, vec![previous]);
    let witness = (assignment, vec![()]);
    let out = folded.prove(transcript(), &instance, &witness).unwrap();
    assert!(ipa.decide(&out.instance));
    let (accumulator, _) = folded.verify(transcript(), &instance, out.proof).unwrap();
    assert_eq!(accumulator, out.instance);
}

#[test]
fn ipa_kimchi_ipa() {
    // On the SRS's generators and blinding point, and its challenges are
    // Kimchi's scalar challenges.
    let srs = Srs::<Vesta>::create(4);
    let ipa: KimchiIpa<Vesta> = kimchi_ipa(&srs);
    assert_eq!(ipa.generators, srs.g);
    assert_eq!(ipa.h, srs.h);
    let mut rng = SplitMix64(7);
    for _ in 0..16 {
        let limbs = [rng.next_u64(), rng.next_u64()];
        let raw = ScalarChallenge::<Fp>::from_limbs(limbs);
        assert_eq!(ipa.challenges.map(raw.0), raw.to_field(Vesta::lambda()));
    }
}

#[test]
fn ipa_kimchi_ipa_with() {
    let srs = Srs::<Vesta>::create(4);
    let with = kimchi_ipa_with(&srs, GroupMap::setup());
    assert_eq!(with.generators, kimchi_ipa(&srs).generators);
    assert_eq!(with.challenges, kimchi_ipa(&srs).challenges);
}

#[test]
fn ipa_fq_sponge_oracle() {
    // The oracle squeezes the sponge's 128-bit challenges and its base
    // field elements.
    let new = || FqSponge::<Vesta, _>::new(KimchiSponge::new(vesta_base()));
    let mut oracle = new();
    let mut reference = new();
    IpaOracle::<Vesta>::absorb_scalar(&mut oracle, Fp::ONE);
    reference.absorb_fr(&[Fp::ONE]);
    let squeezed = IpaOracle::<Vesta>::squeeze_scalar(&mut oracle);
    let expected: Fp = reference.challenge();
    assert_eq!(squeezed, expected);
    let base = BaseOracle::<Vesta>::squeeze_base(&mut oracle);
    assert_eq!(base, reference.challenge_fq());
}

#[test]
fn ipa_group_map_hash_to_curve() {
    let map = GroupMap::<Vesta>::setup();
    let t = Fq::from_u64(3);
    assert_eq!(
        HashToCurve::to_group(&map, t),
        map.to_group(t).to_projective()
    );
}

#[test]
fn ipa_opening_conversions() {
    // Kimchi's opening proof and the generic opening carry the same data.
    let (_, proof, _) = proved();
    let opening: Opening<Vesta, Schnorr<Vesta>> = (&proof.proof).into();
    assert_eq!(opening.last.z1, proof.proof.z1);
    let back: OpeningProof<Vesta> = opening.into();
    assert_eq!(back, proof.proof);
}

/// A 16-bit XOR's rows and witness.
fn xor16() -> (ConstraintSystem<Fp>, [Vec<Fp>; COLUMNS]) {
    let mut gates = Vec::new();
    CircuitGate::<Fp>::extend_xor_gadget(&mut gates, 16);
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let in1 = Fp::from_u64(0x1234);
    let in2 = Fp::from_u64(0x00ff);
    (cs, create_xor_witness(in1, in2, 16).unwrap())
}

#[test]
fn constraints_circuit_gate_error() {
    let error = CircuitGateError::Constraint(GateType::Xor16, 2);
    assert_eq!(error.to_string(), "Invalid Xor16 constraint: 2");
}

#[test]
fn constraints_circuit_gate_verify_witness() {
    // A wrong nybble breaks the first constraint, from 1 as in Kimchi.
    let (cs, mut witness) = xor16();
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    assert_eq!(cs.gates[0].verify_witness(0, &witness, &kimchi), Ok(()));
    witness[3][0] = witness[3][0] + Fp::ONE;
    assert_eq!(
        cs.gates[0].verify_witness(0, &witness, &kimchi),
        Err(CircuitGateError::Constraint(GateType::Xor16, 1))
    );
}

#[test]
fn constraints_circuit_gate_verify_generic() {
    // w0 = 3 is `Const(3)`; w0 = 4 is not.
    let spec = GenericGateSpec::Spec(GenericSpec::Const(Fp::from_u64(3)));
    let gate = CircuitGate::create_generic_gadget(Wire::for_row(0), spec, None);
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO]);
    witness[0][0] = Fp::from_u64(3);
    assert_eq!(gate.verify_generic(0, &witness, &[]), Ok(()));
    witness[0][0] = Fp::from_u64(4);
    let error = gate.verify_generic(0, &witness, &[]).unwrap_err();
    assert_eq!(error, "generic: incorrect gate");
}

#[test]
fn constraints_circuit_gate_verify() {
    // The Xor16 row through verify_witness, as a message.
    let (cs, mut witness) = xor16();
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    witness[1][0] = witness[1][0] + Fp::ONE;
    let error = cs.gates[0].verify(0, &witness, &kimchi, &[]).unwrap_err();
    assert_eq!(error, "Invalid Xor16 constraint: 2");
}

#[test]
fn constraints_gate_error() {
    let error = GateError::IncorrectPublic(3);
    assert_eq!(error.to_string(), "public row 3 has the wrong gate");
}

#[test]
fn constraints_constraint_system_verify_witness() {
    // Row 1 is also the Xor16 row's next row: changing it fails row 0
    // first, as Kimchi's row-by-row check does.
    let (cs, mut witness) = xor16();
    let kimchi = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    assert_eq!(cs.verify_witness(&witness, &[], &kimchi), Ok(()));
    witness[1][1] = Fp::ONE;
    let found = cs.verify_witness(&witness, &[], &kimchi);
    let err = String::from("Invalid Xor16 constraint: 2");
    assert!(found == Err(GateError::Custom { row: 0, err }), "{found:?}");
}

#[test]
fn polynomials_columns() {
    // The witness of a gadget, cut to the rows asked for: the Zero row,
    // empty.
    let witness = gadgets::columns::<Fp, Zero, 8, 2>(&Zero, &(), 1).unwrap();
    assert!(witness.iter().all(|c| c == &vec![Fp::ZERO]));
}

#[test]
fn polynomials_generic_generic_gate_spec() {
    let public = GenericGateSpec::<Fp>::Pub;
    assert_eq!(public.coefficients()[0], Fp::ONE);
}

#[test]
fn polynomials_generic_create_generic_gadget() {
    // Kimchi's coefficients: [1, 0, 0, 0, -c] for Const(c).
    let spec = GenericGateSpec::Spec(GenericSpec::Const(Fp::from_u64(5)));
    let gate = CircuitGate::create_generic_gadget(Wire::for_row(0), spec, None);
    assert_eq!(gate.typ, GateType::Generic);
    assert_eq!(gate.coeffs[0], Fp::ONE);
    assert_eq!(gate.coeffs[4], -Fp::from_u64(5));
    assert_eq!(gate.coeffs.len(), 10);
}

#[test]
fn polynomials_xor_num_xors() {
    assert_eq!(num_xors(8), 1);
    assert_eq!(num_xors(64), 4);
    assert_eq!(num_xors(65), 5);
}

#[test]
fn polynomials_xor_create_xor_gadget() {
    let (next, gates) = CircuitGate::<Fp>::create_xor_gadget(3, 32);
    assert_eq!(next, 6);
    assert_eq!(gates[0].typ, GateType::Xor16);
    assert_eq!(gates[2].typ, GateType::Generic);
}

#[test]
fn polynomials_xor_extend_xor_gadget() {
    // The zero row's three first cells on one cycle.
    let mut gates = Vec::new();
    let next = CircuitGate::<Fp>::extend_xor_gadget(&mut gates, 16);
    assert_eq!(next, 2);
    let mut cycle = vec![Wire { row: 1, col: 0 }];
    let mut at = gates[1].wires[0];
    while at != cycle[0] {
        cycle.push(at);
        at = gates[at.row].wires[at.col];
    }
    cycle.sort();
    let expected: Vec<Wire> = (0..3).map(|col| Wire { row: 1, col }).collect();
    assert_eq!(cycle, expected);
}

#[test]
fn polynomials_xor_create_xor_witness() {
    let witness = create_xor_witness(Fp::from_u64(0xf0f0), Fp::from_u64(0x0ff0), 16).unwrap();
    assert_eq!(witness[2][0], Fp::from_u64(0xff00));
    assert!(create_xor_witness(Fp::from_u64(1 << 20), Fp::ZERO, 16).is_err());
}

#[test]
fn polynomials_xor_extend_xor_witness() {
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO]);
    extend_xor_witness(&mut witness, Fp::ONE, Fp::ONE, 16).unwrap();
    assert_eq!(witness[0].len(), 3);
}

#[test]
fn polynomials_rot_create_rot64() {
    // Rot64 with 2^rot, then two RangeCheck0 rows with coefficient 0.
    let gates = CircuitGate::<Fp>::create_rot64(1, 3);
    assert_eq!(gates[0].typ, GateType::Rot64);
    assert_eq!(gates[0].coeffs, vec![Fp::from_u64(8)]);
    assert_eq!(gates[2].typ, GateType::RangeCheck0);
    assert_eq!(gates[2].wires, Wire::for_row(3));
}

#[test]
fn polynomials_rot_create_rot() {
    // A right rotation by 3 is a left one by 61.
    let (next, gates) = CircuitGate::<Fp>::create_rot(0, 3, RotMode::Right);
    assert_eq!(next, 3);
    assert_eq!(gates[0].coeffs, vec![Fp::from_u64(2).pow(61)]);
}

#[test]
fn polynomials_rot_extend_rot() {
    // The excess on the Rot64 row is wired to the second range check.
    let mut gates = vec![CircuitGate::<Fp>::new(
        GateType::Zero,
        Wire::for_row(0),
        vec![],
    )];
    let next = CircuitGate::extend_rot(&mut gates, 5, RotMode::Left, 0);
    assert_eq!(next, 4);
    assert_eq!(gates[1].wires[2], Wire { row: 3, col: 0 });
}

#[test]
fn constraints_connect_connect_64bit() {
    // Kimchi's three swaps: the third joins two cells already on one
    // cycle and splits it, so (1, 1) is left alone and only (1, 2) is
    // wired to (0, 0) (see kimchi_rot::connect_64bit_forgery).
    let mut gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| CircuitGate::new(GateType::Zero, Wire::for_row(row), vec![]))
        .collect();
    gates.connect_64bit(0, 1);
    assert_eq!(gates[1].wires[1], Wire { row: 1, col: 1 });
    assert_eq!(gates[1].wires[2], Wire { row: 0, col: 0 });
    assert_eq!(gates[0].wires[0], Wire { row: 1, col: 2 });
}

#[test]
fn polynomials_rot_extend_rot_witness() {
    // Three rows; the rotated word in column 1 of the first.
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO]);
    extend_rot(&mut witness, 0x8000_0000_0000_0001, 1, RotMode::Left).unwrap();
    assert_eq!(witness[0].len(), 4);
    assert_eq!(witness[1][1], Fp::from_u64(3));
    assert!(extend_rot(&mut witness, 1, 65, RotMode::Left).is_err());
}

#[test]
fn polynomials_rot_rot_mode() {
    assert_ne!(RotMode::Left, RotMode::Right);
}

/// A commitment of one chunk.
fn one_chunk(k: u64) -> PolyComm<Vesta> {
    PolyComm {
        chunks: vec![Projective::<Vesta>::generator() * Fp::from_u64(k)],
    }
}

#[test]
fn transcript_piop() {
    // Without lookups, the tape absorbs and squeezes what Kimchi's sponge
    // does, in Kimchi's order.
    let digest = Fq::from_u64(7);
    let w_comm: [PolyComm<Vesta>; COLUMNS] = core::array::from_fn(|i| one_chunk(i as u64 + 1));
    let tape: FqTape<Vesta, _, Piop<Vesta>> = transcript::piop(KimchiSponge::new(vesta_base()));
    let tape = tape
        .statement(&Message::new(Base(digest)))
        .statement(&Message::new(vec![]))
        .statement(&Message::new(one_chunk(20)))
        .send(&Message::new(w_comm.clone()));
    let (beta, tape) = tape.left().squeeze();
    let (gamma, tape) = tape.squeeze();
    let tape = tape.send(&Message::new(one_chunk(21)));
    let (alpha, tape) = tape.squeeze();
    let tape = tape.send(&Message::new(one_chunk(22)));
    let (zeta, tape) = tape.squeeze();
    let (fq_digest, tape) = tape.peek();
    let mut sponge = FqSponge::<Vesta, _>::new(KimchiSponge::new(vesta_base()));
    sponge.absorb_fq(&[digest]);
    sponge.absorb_g(&one_chunk(20).chunks);
    for c in &w_comm {
        sponge.absorb_g(&c.chunks);
    }
    assert_eq!(beta, sponge.challenge());
    assert_eq!(gamma, sponge.challenge());
    sponge.absorb_g(&one_chunk(21).chunks);
    assert_eq!(alpha, sponge.challenge());
    sponge.absorb_g(&one_chunk(22).chunks);
    assert_eq!(zeta, sponge.challenge());
    assert_eq!(fq_digest, sponge.clone().digest());
    // The peek left the sponge as it was.
    let mut after = tape.into_inner();
    assert_eq!(after.challenge_fq(), sponge.challenge_fq());
}

#[test]
fn transcript_evaluation() {
    // The two digests, ft(zeta w) and the evaluations, then v and u.
    let (_, proof, _) = proved();
    let tape = transcript::evaluation::<Fp, _>(KimchiSponge::new(pallas_base()))
        .statement(&Message::new(Scalar(Fp::ONE)))
        .statement(&Message::new(Scalar(Fp::from_u64(2))))
        .send(&Message::new(Scalar(proof.ft_eval1)))
        .send(&Message::new(proof.evals.clone()));
    let (v, tape) = tape.squeeze();
    let (u, _) = tape.squeeze();
    let mut sponge = FrSponge::<Fp, _>::new(KimchiSponge::new(pallas_base()));
    sponge.absorb(Fp::ONE);
    sponge.absorb(Fp::from_u64(2));
    sponge.absorb(proof.ft_eval1);
    sponge.absorb_multiple(&proof.evals.public.zeta);
    sponge.absorb_multiple(&proof.evals.public.zeta_omega);
    for e in proof.evals.absorbed() {
        sponge.absorb_multiple(&e.zeta);
        sponge.absorb_multiple(&e.zeta_omega);
    }
    assert_eq!(v, sponge.challenge());
    assert_eq!(u, sponge.challenge());
}

#[test]
fn transcript_previous_challenges_digest() {
    // The digest of a sponge that absorbed the challenges, in order.
    let chals = vec![vec![Fp::ONE, Fp::from_u64(2)], vec![Fp::from_u64(3)]];
    let digest = transcript::previous_challenges_digest(KimchiSponge::new(pallas_base()), chals);
    let mut sponge = FrSponge::<Fp, _>::new(KimchiSponge::new(pallas_base()));
    sponge.absorb_multiple(&[Fp::ONE, Fp::from_u64(2)]);
    sponge.absorb_multiple(&[Fp::from_u64(3)]);
    assert_eq!(digest, sponge.digest());
}

#[test]
fn transcript_message() {
    let m: Message<Base<Fq>, VerifierIndexDigest> = Message::new(Base(Fq::ONE));
    assert_eq!(m.0, Base(Fq::ONE));
    assert_eq!(
        <Message<Base<Fq>, VerifierIndexDigest> as Label>::LABEL,
        <VerifierIndexDigest as Label>::LABEL
    );
}

#[test]
fn transcript_challenge_and_digest_kinds() {
    // Each challenge and digest is named by its step.
    assert_eq!(<Challenge<Beta> as Label>::LABEL, <Beta as Label>::LABEL);
    assert_eq!(<DigestOf<Digest> as Label>::LABEL, <Digest as Label>::LABEL);
    assert_ne!(<Beta as Label>::LABEL, <Digest as Label>::LABEL);
}

#[test]
fn transcript_after_gamma() {
    // The witness round hands on the tape at z, without lookups.
    let cs = ConstraintSystem::create(two_generic()).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let sponge = KimchiSponge::new(vesta_base());
    let witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    let mut rng = Arkworks(SplitMix64(3));
    let (_, tape) = witness_round(&index, witness, &[], &[], &sponge, &mut rng).unwrap();
    assert!(matches!(tape, AfterGamma::WithoutLookup(_)));
}

#[test]
fn polynomials_bit_length() {
    assert_eq!(bit_length(Fp::ZERO), 1);
    assert_eq!(bit_length(Fp::from_u64(1 << 40)), 41);
}

#[test]
fn polynomials_range_check_create_multi_range_check() {
    // RC0, RC0, RC1, Zero; the high limbs wired to the Zero row's 3-6.
    let (next, gates) = CircuitGate::<Fp>::create_multi_range_check(2);
    assert_eq!(next, 6);
    let types: Vec<GateType> = gates.iter().map(|g| g.typ).collect();
    let expected = [
        GateType::RangeCheck0,
        GateType::RangeCheck0,
        GateType::RangeCheck1,
        GateType::Zero,
    ];
    assert_eq!(types, expected);
    assert_eq!(gates[1].coeffs, vec![Fp::ZERO]);
}

#[test]
fn polynomials_range_check_create_compact_multi_range_check() {
    let (_, gates) = CircuitGate::<Fp>::create_compact_multi_range_check(0);
    assert_eq!(gates[1].coeffs, vec![Fp::ONE]);
}

#[test]
fn polynomials_range_check_extend_multi_range_check() {
    let mut gates = Vec::new();
    let mut row = 0;
    CircuitGate::<Fp>::extend_multi_range_check(&mut gates, &mut row);
    CircuitGate::<Fp>::extend_compact_multi_range_check(&mut gates, &mut row);
    assert_eq!((row, gates.len()), (8, 8));
}

#[test]
fn polynomials_range_check_create_range_check() {
    let mut gates = Vec::new();
    let mut row = 0;
    CircuitGate::<Fp>::extend_range_check(&mut gates, &mut row);
    let (next, single) = CircuitGate::<Fp>::create_range_check(1);
    assert_eq!((row, next), (1, 2));
    assert_eq!(single[0].typ, GateType::RangeCheck0);
}

#[test]
fn polynomials_range_check_witnesses() {
    // Four rows for the multi checks, one for the single; the values in
    // column 0 as Kimchi lays them out (compact: v2, v0, v1).
    let (v0, v1, v2) = (Fp::from_u64(3), Fp::from_u64(5), Fp::from_u64(7));
    let multi = create_multi(v0, v1, v2).unwrap();
    assert_eq!([multi[0][0], multi[0][1], multi[0][2]], [v0, v1, v2]);
    assert_eq!(create_multi_limbs(&[v0, v1, v2]).unwrap(), multi);
    let two_88 = Fp::from_u64(1 << 44) * Fp::from_u64(1 << 44);
    let compact = create_multi_compact(v0 + two_88 * v1, v2).unwrap();
    assert_eq!([compact[0][0], compact[0][1], compact[0][2]], [v2, v0, v1]);
    assert_eq!(
        create_multi_compact_limbs(&[v0 + two_88 * v1, v2]).unwrap(),
        compact
    );
    assert_eq!(create_range_check_witness(v0).unwrap()[0], vec![v0]);
    let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| Vec::new());
    extend_multi(&mut w, v0, v1, v2).unwrap();
    extend_multi_limbs(&mut w, &[v0, v1, v2]).unwrap();
    extend_multi_compact(&mut w, v0, v2).unwrap();
    extend_multi_compact_limbs(&mut w, &[v0, v2]).unwrap();
    extend_range_check_witness(&mut w, v0).unwrap();
    assert_eq!(w[0].len(), 17);
}

#[test]
fn polynomials_generic_testing_create_circuit() {
    // 2 public rows, 10 add/mul rows, 10 constant rows.
    let gates = create_circuit::<Fp>(0, 2);
    assert_eq!(gates.len(), 22);
    assert_eq!(gates[2].coeffs[1], Fp::from_u64(3));
    assert_eq!(gates[2].coeffs[8], Fp::from_u64(2));
    assert_eq!(gates[21].coeffs[9], -Fp::from_u64(5));
}

#[test]
fn polynomials_generic_testing_fill_in_witness() {
    // 11 + 3 23 = 80 and 2 11 23 = 506; every row satisfies its gate.
    let gates = create_circuit::<Fp>(0, 1);
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; gates.len()]);
    fill_in_witness(0, &mut witness, &[Fp::from_u64(9)]);
    assert_eq!(witness[2][1], Fp::from_u64(80));
    assert_eq!(witness[5][1], Fp::from_u64(506));
    for (row, gate) in gates.iter().enumerate() {
        assert_eq!(
            gate.verify_generic(row, &witness, &[Fp::from_u64(9)]),
            Ok(())
        );
    }
}

#[test]
fn polynomials_not_extend_not_gadget_unchecked_length() {
    // Three NOTs: a double row and a single one, wired to the all-ones cell.
    let mut gates = vec![CircuitGate::<Fp>::new(
        GateType::Generic,
        Wire::for_row(0),
        vec![],
    )];
    let next = CircuitGate::extend_not_gadget_unchecked_length(&mut gates, 3, 0);
    assert_eq!(next, 3);
    assert_eq!(gates[1].coeffs[1], -Fp::ONE);
    assert_eq!(gates[1].coeffs[6], -Fp::ONE);
    assert_eq!(gates[2].coeffs[6], Fp::ZERO);
}

#[test]
fn polynomials_not_extend_not_gadget_checked_length() {
    // A XOR of 16 bits whose second input is the all-ones cell.
    let mut gates = vec![CircuitGate::<Fp>::new(
        GateType::Generic,
        Wire::for_row(0),
        vec![],
    )];
    let next = CircuitGate::extend_not_gadget_checked_length(&mut gates, 0, 16);
    assert_eq!(next, 3);
    assert_eq!(gates[1].typ, GateType::Xor16);
    assert_eq!(gates[0].wires[0], Wire { row: 1, col: 1 });
}

#[test]
fn polynomials_not_extend_not_witness_checked_length() {
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| Vec::new());
    extend_not_witness_checked_length(&mut witness, Fp::from_u64(0b1010), Some(4)).unwrap();
    assert_eq!(witness[2][0], Fp::from_u64(0b0101));
}

#[test]
fn polynomials_not_extend_not_witness_unchecked_length() {
    // ones - x, two per row; an input wider than `bits` is refused.
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| Vec::new());
    let inputs = [Fp::from_u64(1), Fp::from_u64(2), Fp::from_u64(3)];
    extend_not_witness_unchecked_length(&mut witness, &inputs, 4).unwrap();
    assert_eq!(witness[0].len(), 2);
    assert_eq!(witness[5][0], Fp::from_u64(13));
    let error = extend_not_witness_unchecked_length(&mut witness, &[Fp::from_u64(32)], 4);
    assert_eq!(
        error,
        Err(String::from(
            "Bits must be greater or equal than the inputs length: 4 < 6"
        ))
    );
}

#[test]
fn polynomials_and_extend_and() {
    // The XOR of 16 bits, its zero row, then the Generic AND row.
    let mut gates = Vec::new();
    let next = CircuitGate::<Fp>::extend_and(&mut gates, 2);
    assert_eq!(next, 3);
    assert_eq!(gates[2].typ, GateType::Generic);
    assert_eq!(gates[2].coeffs[7], -Fp::from_u64(2));
}

#[test]
fn polynomials_and_create_and_witness() {
    // a + b = xor + 2 and: [a, b, sum, sum, xor, and] on the last row.
    let (a, b) = (Fp::from_u64(0b1100), Fp::from_u64(0b1010));
    let witness = create_and_witness(a, b, 1).unwrap();
    let row = witness[0].len() - 1;
    assert_eq!(witness[4][row], Fp::from_u64(0b0110));
    assert_eq!(witness[5][row], Fp::from_u64(0b1000));
    assert!(create_and_witness(Fp::from_u64(1 << 9), b, 1).is_err());
}

#[test]
fn polynomials_and_extend_and_witness() {
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO]);
    extend_and_witness(&mut witness, Fp::ONE, Fp::ONE, 2).unwrap();
    assert_eq!(witness[0].len(), 4);
    assert_eq!(witness[5][3], Fp::ONE);
}
