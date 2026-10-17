//! Kimchi's SRS (`SRS::create`, Blake2b512 then the group map) and verifier
//! index (`ProverIndex::verifier_index`) on the four circuits of
//! `constraint_system`, over Vesta: the first and last SRS points and `h`,
//! then every chunk of every commitment, as affine coordinates.
use std::sync::Arc;

use ark_ec::AffineRepr;
use ark_poly::EvaluationDomain;
use ark_ff::{BigInteger, PrimeField};
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        gate::{CircuitGate, Connect, GateType},
        lookup::{runtime_tables::RuntimeTableCfg, tables::LookupTable},
        wires::Wire,
    },
    curve::KimchiCurve,
    prover_index::ProverIndex,
};
use mina_curves::pasta::{Fp, Vesta};
use mina_curves::pasta::VestaParameters;
use mina_poseidon::{constants::PlonkSpongeConstantsKimchi as SC, sponge::DefaultFqSponge};
use poly_commitment::{commitment::PolyComm, ipa::SRS, SRS as _};

type BaseSponge = DefaultFqSponge<VestaParameters, SC, 55>;

fn hex<F: PrimeField>(x: F) -> String {
    x.into_bigint().to_bytes_be().iter().map(|b| format!("{b:02x}")).collect()
}

fn point(p: &Vesta) -> String {
    match p.xy() {
        Some((x, y)) => format!("{} {}", hex(x), hex(y)),
        None => String::from("inf"),
    }
}

fn comm(name: &str, label: &str, c: Option<&PolyComm<Vesta>>) {
    match c {
        None => println!("comm {name} {label} none"),
        Some(c) => {
            let chunks: Vec<String> = c.chunks.iter().map(point).collect();
            println!("comm {name} {label} {} {}", c.chunks.len(), chunks.join(" "));
        }
    }
}

fn f(k: u64) -> Fp {
    Fp::from(k)
}

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
    let depth = max_poly_size.unwrap_or(cs.domain.d1.size());
    let srs = SRS::<Vesta>::create(depth);
    println!("srs {name} {depth} {} {} {}", point(&srs.g[0]), point(&srs.g[depth - 1]), point(&srs.h));
    let endo = *Vesta::other_curve_endo();
    let index: ProverIndex<55, Vesta, SRS<Vesta>> =
        ProverIndex::create(cs, endo, Arc::new(srs), false);
    let v = index.verifier_index();
    println!("vi {name} {} {} {} {} {}", v.domain.size, v.max_poly_size, v.zk_rows, v.public, v.prev_challenges);
    for (i, c) in v.sigma_comm.iter().enumerate() {
        comm(name, &format!("sigma{i}"), Some(c));
    }
    for (i, c) in v.coefficients_comm.iter().enumerate() {
        comm(name, &format!("coefficient{i}"), Some(c));
    }
    comm(name, "Generic", Some(&v.generic_comm));
    comm(name, "Poseidon", Some(&v.psm_comm));
    comm(name, "CompleteAdd", Some(&v.complete_add_comm));
    comm(name, "VarBaseMul", Some(&v.mul_comm));
    comm(name, "EndoMul", Some(&v.emul_comm));
    comm(name, "EndoMulScalar", Some(&v.endomul_scalar_comm));
    comm(name, "RangeCheck0", v.range_check0_comm.as_ref());
    comm(name, "RangeCheck1", v.range_check1_comm.as_ref());
    comm(name, "ForeignFieldAdd", v.foreign_field_add_comm.as_ref());
    comm(name, "ForeignFieldMul", v.foreign_field_mul_comm.as_ref());
    comm(name, "Xor16", v.xor_comm.as_ref());
    comm(name, "Rot64", v.rot_comm.as_ref());
    match &v.lookup_index {
        None => println!("lookup {name} none"),
        Some(l) => {
            println!(
                "lookup {name} info {} {} {}",
                l.lookup_info.max_per_row, l.lookup_info.max_joint_size, l.joint_lookup_used
            );
            for (i, c) in l.lookup_table.iter().enumerate() {
                comm(name, &format!("lookup_table{i}"), Some(c));
            }
            comm(name, "table_ids", l.table_ids.as_ref());
            comm(name, "runtime_selector", l.runtime_tables_selector.as_ref());
            let s = &l.lookup_selectors;
            comm(name, "selector_xor", s.xor.as_ref());
            comm(name, "selector_lookup", s.lookup.as_ref());
            comm(name, "selector_range_check", s.range_check.as_ref());
            comm(name, "selector_ffmul", s.ffmul.as_ref());
        }
    }
    println!("digest {name} {}", hex(v.digest::<BaseSponge>()));
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
