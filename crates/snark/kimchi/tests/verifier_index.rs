//! Kimchi's SRS and verifier index against Kimchi: for the four circuits
//! of the figures tools, on Vesta, the first and last SRS points and `h`,
//! the parameters of the index, every chunk of every commitment (the
//! lookup part included), and the digest
//! (`vectors/kimchi_verifier_index.txt`, from the `verifier_index` tool).

mod common;

use std::collections::BTreeMap;

use arrabbiata_curve::Projective;
use arrabbiata_field::PrimeField;
use arrabbiata_kimchi::{
    srs::{PolyComm, Srs},
    verifier_index::VerifierIndex,
};
use arrabbiata_pasta::Vesta;
use arrabbiata_poseidon::kimchi::{KimchiSponge, vesta_base};
use common::circuits;

/// Kimchi's output.
const VECTORS: &str = include_str!("vectors/kimchi_verifier_index.txt");

fn hex<F: PrimeField>(x: F) -> String {
    (0..F::LIMBS)
        .rev()
        .map(|i| format!("{:016x}", x.limb(i)))
        .collect()
}

fn point(p: &Projective<Vesta>) -> String {
    let a = p.to_affine();
    match (a.x(), a.y()) {
        (Some(x), Some(y)) => format!("{} {}", hex(x), hex(y)),
        _ => String::from("inf"),
    }
}

fn comm(c: Option<&PolyComm<Vesta>>) -> String {
    match c {
        None => String::from("none"),
        Some(c) => {
            let chunks: Vec<String> = c.chunks.iter().map(point).collect();
            format!("{} {}", c.chunks.len(), chunks.join(" "))
        }
    }
}

/// Our lines, keyed by their first three words (two for `srs` and `vi`).
fn ours(name: &str, index: &VerifierIndex<Vesta>, srs: &Srs<Vesta>) -> BTreeMap<String, String> {
    let mut lines = BTreeMap::new();
    let depth = srs.max_poly_size();
    lines.insert(
        format!("srs {name}"),
        format!(
            "{depth} {} {} {}",
            point(&srs.g[0]),
            point(&srs.g[depth - 1]),
            point(&srs.h)
        ),
    );
    lines.insert(
        format!("vi {name}"),
        format!(
            "{} {} {} {} {}",
            index.domain_size,
            index.max_poly_size,
            index.zk_rows,
            index.public,
            index.prev_challenges
        ),
    );
    let mut put = |label: &str, c: Option<&PolyComm<Vesta>>| {
        lines.insert(format!("comm {name} {label}"), comm(c));
    };
    for (i, c) in index.sigma_comm.iter().enumerate() {
        put(&format!("sigma{i}"), Some(c));
    }
    for (i, c) in index.coefficients_comm.iter().enumerate() {
        put(&format!("coefficient{i}"), Some(c));
    }
    put("Generic", Some(&index.generic_comm));
    put("Poseidon", Some(&index.psm_comm));
    put("CompleteAdd", Some(&index.complete_add_comm));
    put("VarBaseMul", Some(&index.mul_comm));
    put("EndoMul", Some(&index.emul_comm));
    put("EndoMulScalar", Some(&index.endomul_scalar_comm));
    put("RangeCheck0", index.range_check0_comm.as_ref());
    put("RangeCheck1", index.range_check1_comm.as_ref());
    put("ForeignFieldAdd", index.foreign_field_add_comm.as_ref());
    put("ForeignFieldMul", index.foreign_field_mul_comm.as_ref());
    put("Xor16", index.xor_comm.as_ref());
    put("Rot64", index.rot_comm.as_ref());
    match &index.lookup_index {
        None => {
            lines.insert(format!("lookup {name} none"), String::new());
        }
        Some(l) => {
            let info = &l.lookup_info;
            lines.insert(
                format!("lookup {name} info"),
                format!(
                    "{} {} {}",
                    info.max_per_row, info.max_joint_size, l.joint_lookup_used
                ),
            );
            for (i, c) in l.lookup_table.iter().enumerate() {
                lines.insert(format!("comm {name} lookup_table{i}"), comm(Some(c)));
            }
            let mut put = |label: &str, c: Option<&PolyComm<Vesta>>| {
                lines.insert(format!("comm {name} {label}"), comm(c));
            };
            put("table_ids", l.table_ids.as_ref());
            put("runtime_selector", l.runtime_tables_selector.as_ref());
            for (label, c) in ["xor", "lookup", "range_check", "ffmul"]
                .iter()
                .zip(&l.lookup_selectors)
            {
                put(&format!("selector_{label}"), c.as_ref());
            }
        }
    }
    let digest = index.digest(KimchiSponge::new(vesta_base()));
    lines.insert(format!("digest {name}"), hex(digest));
    lines
}

/// Kimchi's lines, keyed as [`ours`] keys them.
fn expected() -> BTreeMap<String, String> {
    VECTORS
        .lines()
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let key = if words[0] == "comm" || words[0] == "lookup" {
                3
            } else {
                2
            };
            (words[..key].join(" "), words[key..].join(" "))
        })
        .collect()
}

#[test]
fn the_verifier_indexes_are_kimchis() {
    let expected = expected();
    assert_eq!(expected.len(), 170);
    let mut found = BTreeMap::new();
    for (name, cs) in circuits() {
        // The tools take the SRS as large as the domain, except with chunks.
        let depth = cs.max_poly_size.unwrap_or(cs.domain.d1.size());
        let srs = Srs::<Vesta>::create(depth);
        let index = VerifierIndex::create(&cs, &srs);
        found.extend(ours(name, &index, &srs));
    }
    for (key, value) in &expected {
        assert_eq!(found.get(key), Some(value), "{key}");
    }
    assert_eq!(found.len(), expected.len());
}
