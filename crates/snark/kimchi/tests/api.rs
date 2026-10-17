//! The behaviour of each public item, one test per item.

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::{
    constraints::{
        self, Builder, COLUMNS, CircuitGate, Connect, ConstraintSystem, EvaluationDomains,
        GateTable, GateType, LookupTable, RANGE_CHECK_TABLE_SIZE, RuntimeTableCfg, Wire,
        XOR_TABLE_SIZE, zk_rows_strict_lower_bound,
    },
    gates::{Gates, KimchiGates},
    oracle::{CHALLENGE_LIMBS, FqSponge, FrSponge, ScalarChallenge},
    permutation::{PERMUTS, shifts},
    prover::{
        self, PERMUTATION_ALPHA, PermutationRound, ProverIndex, QuotientRound, WitnessRound,
        permutation_round, quotient_round, witness_round,
    },
    rng::{Arkworks, RandomSource, Sampler, SplitMix64},
    srs::{GroupMap, PolyComm, Srs, packed, point_of_random_bytes},
    verifier_index::VerifierIndex,
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

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
    let error = witness_round(&index, ragged, &sponge, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::WitnessCsInconsistent);
    assert_eq!(error.to_string(), "witness columns of different lengths");
    // Six rows leave two of eight free, fewer than the three zk rows.
    let long: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 6]);
    let error = witness_round(&index, long, &sponge, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::NoRoomForZkInWitness);
    // Chunks of 4 rows need more zk rows than the 3 built for one chunk.
    let one_chunk = ConstraintSystem::create(two_generic()).build().unwrap();
    let chunked = ProverIndex::create(one_chunk, Srs::<Vesta>::create(4));
    let short: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 2]);
    let error = witness_round(&chunked, short, &sponge, &mut rng).unwrap_err();
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
    let round: WitnessRound<Vesta, _> = witness_round(&index, witness, &sponge, &mut rng).unwrap();
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
    let first = witness_round(&index, equal, &sponge, &mut rng).unwrap();
    let round: PermutationRound<Vesta, _> = permutation_round(&index, first, &mut rng).unwrap();
    // z starts at 1 and closes at row n - zk_rows = 5.
    assert_eq!(round.z[0], Fp::ONE);
    assert_eq!(round.z[5], Fp::ONE);
    assert_eq!(round.z_blinders.len(), 1);
    assert_eq!(round.alpha, round.alpha_chal.to_field(Vesta::lambda()));
    // Breaking the copy breaks the product.
    let mut broken: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ONE; 2]);
    broken[0][1] = Fp::ZERO;
    let first = witness_round(&index, broken, &sponge, &mut rng).unwrap();
    let error = permutation_round(&index, first, &mut rng).unwrap_err();
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
    let first = witness_round(&index, good.clone(), &sponge, &mut rng).unwrap();
    let second = permutation_round(&index, first, &mut rng).unwrap();
    let round: QuotientRound<Vesta, _> = quotient_round(&index, second, &kimchi, &mut rng).unwrap();
    // 7 chunks of 8 rows, each hidden by its blinder.
    assert_eq!(round.t_comm.chunks.len(), 7);
    assert_eq!(round.t_blinders.len(), 7);
    assert!(round.t.coefficients().len() <= 7 * 8);
    assert_eq!(round.zeta, round.zeta_chal.to_field(Vesta::lambda()));
    // A broken addition leaves a remainder.
    let mut bad = good;
    bad[2][1] = Fp::from_u64(4);
    let first = witness_round(&index, bad, &sponge, &mut rng).unwrap();
    let second = permutation_round(&index, first, &mut rng).unwrap();
    let error = quotient_round(&index, second, &kimchi, &mut rng).unwrap_err();
    assert_eq!(error, prover::Error::QuotientRemainder);
    assert_eq!(
        error.to_string(),
        "the constraints do not vanish on the domain"
    );
}
