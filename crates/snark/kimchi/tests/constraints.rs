//! Kimchi's constraint system against Kimchi: the four circuits of the
//! `constraint_system` tool of `tools/kimchi-vectors`, built here the
//! same way, give the same domain, zero-knowledge rows and generators, and
//! every column (sigmas over d8, coefficients over d8, selectors over d4
//! or d8) has the same Blake2b512 digest (`vectors/kimchi_constraint_system.txt`).

mod common;
mod kimchi_runs;

use std::{collections::BTreeMap, sync::LazyLock};

use common::circuits;

use arrabbiata_field::PrimeField;
use arrabbiata_kimchi::{
    constraints::{ConstraintSystem, GateType, Wire},
    permutation::PERMUTS,
};
use arrabbiata_pasta::Fp;
use blake2::{Blake2b512, Digest};

/// Kimchi's output.
static VECTORS: LazyLock<String> =
    LazyLock::new(|| kimchi_runs::read("kimchi_constraint_system.txt"));

fn hex_be(x: Fp) -> String {
    (0..4)
        .rev()
        .map(|i| format!("{:016x}", x.limb(i)))
        .collect()
}

fn digest(values: &[Fp]) -> String {
    let mut h = Blake2b512::new();
    for x in values {
        for i in 0..4 {
            h.update(x.limb(i).to_le_bytes());
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The selectors Kimchi stores, with the domain it stores each on.
const SELECTORS: [(&str, GateType, usize); 12] = [
    ("Generic", GateType::Generic, 4),
    ("Poseidon", GateType::Poseidon, 8),
    ("CompleteAdd", GateType::CompleteAdd, 4),
    ("VarBaseMul", GateType::VarBaseMul, 8),
    ("EndoMul", GateType::EndoMul, 8),
    ("EndoMulScalar", GateType::EndoMulScalar, 8),
    ("RangeCheck0", GateType::RangeCheck0, 8),
    ("RangeCheck1", GateType::RangeCheck1, 8),
    ("ForeignFieldAdd", GateType::ForeignFieldAdd, 8),
    ("ForeignFieldMul", GateType::ForeignFieldMul, 8),
    ("Xor16", GateType::Xor16, 8),
    ("Rot64", GateType::Rot64, 8),
];

/// Our line for each of Kimchi's, keyed as `the_constraint_systems_are_kimchis` keys them.
fn ours(name: &str, cs: &ConstraintSystem<Fp>) -> BTreeMap<String, String> {
    let mut lines = BTreeMap::new();
    let d = cs.domain;
    lines.insert(
        format!("cs {name}"),
        format!(
            "{} {} {} {}",
            d.d1.size(),
            cs.zk_rows,
            hex_be(d.d1.generator()),
            hex_be(d.d8.generator())
        ),
    );
    let over = |column: &[Fp], k: usize| {
        let to = if k == 4 { d.d4 } else { d.d8 };
        digest(&cs.extend(column, &to).unwrap())
    };
    for (i, sigma) in cs.sigmas().iter().enumerate() {
        lines.insert(
            format!("col {name} sigma{i}"),
            format!("8 {}", over(sigma, 8)),
        );
    }
    for i in 0..15 {
        lines.insert(
            format!("col {name} coefficient{i}"),
            format!("8 {}", over(&cs.coefficient(i), 8)),
        );
    }
    for (label, typ, k) in SELECTORS {
        let value = if typ.is_optional() && !cs.uses(typ) {
            String::from("none")
        } else {
            over(&cs.selector(typ), k)
        };
        lines.insert(format!("col {name} {label}"), format!("{k} {value}"));
    }
    lines
}

#[test]
fn the_constraint_systems_are_kimchis() {
    let mut expected = BTreeMap::new();
    for line in VECTORS.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        // `cs <name> ...` lines are keyed by two words, `col` lines by three.
        let key = if words[0] == "cs" { 2 } else { 3 };
        expected.insert(words[..key].join(" "), words[key..].join(" "));
    }
    let mut found = BTreeMap::new();
    for (name, cs) in circuits() {
        found.extend(ours(name, &cs));
    }
    assert_eq!(expected.len(), 140);
    for (key, value) in &expected {
        assert_eq!(found.get(key), Some(value), "{key}");
    }
    assert_eq!(found.len(), expected.len());
}

/// The sigmas are a permutation of the cell labels: each label appears
/// once, outside the zeroed zero-knowledge rows.
#[test]
fn the_sigmas_permute_the_labels() {
    for (name, cs) in circuits() {
        let n = cs.domain.d1.size();
        let sigmas = cs.sigmas();
        let mut labels = Vec::new();
        let mut images = Vec::new();
        for row in 0..n {
            if (n + 2 - cs.zk_rows..n - 1).contains(&row) {
                continue;
            }
            for (col, sigma) in sigmas.iter().enumerate() {
                labels.push(hex_be(cs.cell_to_field(Wire { row, col })));
                images.push(hex_be(sigma[row]));
            }
        }
        labels.sort();
        images.sort();
        assert_eq!(labels, images, "{name}");
        assert_eq!(labels.len(), PERMUTS * (n - (cs.zk_rows - 3)), "{name}");
    }
}
