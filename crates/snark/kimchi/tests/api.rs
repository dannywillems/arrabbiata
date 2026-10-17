//! The behaviour of each public item, one test per item.

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::{
    constraints::{
        self, Builder, COLUMNS, CircuitGate, Connect, ConstraintSystem, EvaluationDomains,
        GateTable, GateType, LookupTable, RANGE_CHECK_TABLE_SIZE, RuntimeTableCfg, Wire,
        XOR_TABLE_SIZE, zk_rows_strict_lower_bound,
    },
    oracle::{CHALLENGE_LIMBS, FqSponge, FrSponge, ScalarChallenge},
    permutation::{PERMUTS, shifts},
};
use arrabbiata_pasta::{Fp, Vesta};
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
