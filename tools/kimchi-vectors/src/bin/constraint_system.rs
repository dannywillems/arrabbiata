//! Kimchi's constraint system (`ConstraintSystem::build` and the prover
//! index's `column_evaluations`) on four circuits over `Fp`: the domain
//! size, `zk_rows`, the generators of d1 and d8, and a Blake2b512 digest of
//! every column's evaluations (each element as 32 little-endian bytes).
use std::sync::Arc;

use ark_ff::{BigInteger, PrimeField};
use blake2::{Blake2b512, Digest};
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        domains::EvaluationDomains,
        gate::{CircuitGate, Connect, GateType},
        lookup::{runtime_tables::RuntimeTableCfg, tables::LookupTable},
        wires::Wire,
    },
    curve::KimchiCurve,
    prover_index::ProverIndex,
};
use ark_poly::{EvaluationDomain, Evaluations, Radix2EvaluationDomain as D};
use mina_curves::pasta::{Fp, Vesta};
use poly_commitment::{ipa::SRS, SRS as _};

fn hex<F: PrimeField>(x: F) -> String {
    x.into_bigint().to_bytes_be().iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(e: &Evaluations<Fp, D<Fp>>) -> String {
    let mut h = Blake2b512::new();
    for x in &e.evals {
        h.update(x.into_bigint().to_bytes_le());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn f(k: u64) -> Fp {
    Fp::from(k)
}

/// Rows of `typ` with identity wiring and coefficients `seed + i`.
fn gate(typ: GateType, row: usize, coeffs: usize, seed: u64) -> CircuitGate<Fp> {
    CircuitGate::new(
        typ,
        Wire::for_row(row),
        (0..coeffs).map(|i| f(seed + i as u64)).collect(),
    )
}

fn report(
    name: &str,
    gates: Vec<CircuitGate<Fp>>,
    public: usize,
    max_poly_size: Option<usize>,
    tables: Vec<LookupTable<Fp>>,
    runtime: Option<Vec<RuntimeTableCfg<Fp>>>,
) {
    let cs = ConstraintSystem::<Fp>::create(gates)
        .public(public)
        .max_poly_size(max_poly_size)
        .lookup(tables)
        .runtime(runtime)
        .build()
        .unwrap();
    let domain: EvaluationDomains<Fp> = cs.domain;
    println!(
        "cs {name} {} {} {} {}",
        domain.d1.size,
        cs.zk_rows,
        hex(domain.d1.group_gen),
        hex(domain.d8.group_gen)
    );
    let srs = SRS::<Vesta>::create(domain.d1.size());
    let endo = *Vesta::other_curve_endo();
    let index: ProverIndex<55, Vesta, SRS<Vesta>> =
        ProverIndex::create(cs, endo, Arc::new(srs), false);
    let c = index.column_evaluations.get();
    for (i, e) in c.permutation_coefficients8.iter().enumerate() {
        println!("col {name} sigma{i} 8 {}", digest(e));
    }
    for (i, e) in c.coefficients8.iter().enumerate() {
        println!("col {name} coefficient{i} 8 {}", digest(e));
    }
    let mut selector = |label: &str, d: usize, e: Option<&Evaluations<Fp, D<Fp>>>| match e {
        Some(e) => println!("col {name} {label} {d} {}", digest(e)),
        None => println!("col {name} {label} {d} none"),
    };
    selector("Generic", 4, Some(&c.generic_selector4));
    selector("Poseidon", 8, Some(&c.poseidon_selector8));
    selector("CompleteAdd", 4, Some(&c.complete_add_selector4));
    selector("VarBaseMul", 8, Some(&c.mul_selector8));
    selector("EndoMul", 8, Some(&c.emul_selector8));
    selector("EndoMulScalar", 8, Some(&c.endomul_scalar_selector8));
    selector("RangeCheck0", 8, c.range_check0_selector8.as_ref());
    selector("RangeCheck1", 8, c.range_check1_selector8.as_ref());
    selector("ForeignFieldAdd", 8, c.foreign_field_add_selector8.as_ref());
    selector("ForeignFieldMul", 8, c.foreign_field_mul_selector8.as_ref());
    selector("Xor16", 8, c.xor_selector8.as_ref());
    selector("Rot64", 8, c.rot_selector8.as_ref());
}

fn main() {
    // 1. Ten Generic gates, two public inputs, three copies (one a cycle
    //    of three cells).
    let mut generic: Vec<CircuitGate<Fp>> =
        (0..10).map(|row| gate(GateType::Generic, row, 10, 100 * row as u64)).collect();
    generic.connect_cell_pair((0, 0), (3, 1));
    generic.connect_cell_pair((3, 1), (7, 2));
    generic.connect_cell_pair((1, 0), (9, 6));
    report("generic", generic, 2, None, vec![], None);

    // 2. One row of every gate type (the gate checks are not run when
    //    building), wired across rows.
    let kinds = [
        (GateType::Generic, 10),
        (GateType::Poseidon, 15),
        (GateType::CompleteAdd, 0),
        (GateType::VarBaseMul, 0),
        (GateType::Zero, 0),
        (GateType::EndoMul, 0),
        (GateType::EndoMulScalar, 0),
        (GateType::RangeCheck0, 1),
        (GateType::RangeCheck1, 0),
        (GateType::Zero, 0),
        (GateType::ForeignFieldAdd, 4),
        (GateType::ForeignFieldMul, 4),
        (GateType::Zero, 0),
        (GateType::Xor16, 0),
        (GateType::Rot64, 1),
        (GateType::Zero, 0),
    ];
    let mut every: Vec<CircuitGate<Fp>> = kinds
        .iter()
        .enumerate()
        .map(|(row, (typ, n))| gate(*typ, row, *n, 7 + 31 * row as u64))
        .collect();
    every.connect_cell_pair((0, 2), (14, 5));
    every.connect_cell_pair((8, 0), (11, 3));
    report("every", every, 0, None, vec![], None);

    // 3. A hundred Generic gates in chunks of 32 rows.
    let chunked: Vec<CircuitGate<Fp>> =
        (0..100).map(|row| gate(GateType::Generic, row, 10, row as u64)).collect();
    report("chunked", chunked, 1, Some(32), vec![], None);

    // 4. Lookup gates over a fixed table (id 1, five rows of three) and a
    //    runtime table (id 2, three rows).
    let mut lookup: Vec<CircuitGate<Fp>> =
        (0..4).map(|row| gate(GateType::Generic, row, 10, 3 * row as u64)).collect();
    lookup.extend((4..8).map(|row| gate(GateType::Lookup, row, 0, 0)));
    let table = LookupTable {
        id: 1,
        data: (0..3).map(|c| (0..5).map(|r| f(10 * c + r)).collect()).collect(),
    };
    let runtime = RuntimeTableCfg { id: 2, first_column: vec![f(1), f(2), f(3)] };
    report("lookup", lookup, 0, None, vec![table], Some(vec![runtime]));
}
