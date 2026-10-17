//! What the ports of Kimchi's gate tests share: Kimchi's runs of them
//! (`vectors/kimchi_gate_tests.txt`, from the `gate_tests` tool of the
//! Kimchi figures notes, each test with a fixed seed for the inputs Kimchi
//! draws at random), and Kimchi's test framework, `prove_and_verify`, on
//! our prover and verifier.

#![allow(dead_code)]

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::PrimeField;
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem},
    gates::KimchiGates,
    prover::{ProverIndex, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
    verifier::verify,
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// Kimchi's runs.
const VECTORS: &str = include_str!("../vectors/kimchi_gate_tests.txt");

/// One of Kimchi's tests as Kimchi ran it.
#[derive(Debug, Default)]
pub struct Case {
    /// The inputs, by key, as printed.
    pub inputs: Vec<(String, Vec<String>)>,
    /// Kimchi's witnesses, in order, each by rows of 15 hex cells.
    pub witnesses: Vec<Vec<Vec<String>>>,
    /// The outcomes, by label, as Kimchi's `Debug` prints them.
    pub verdicts: Vec<(String, String)>,
}

impl Case {
    /// The values of input `key`.
    pub fn input(&self, key: &str) -> &[String] {
        &self
            .inputs
            .iter()
            .find(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("no input {key}"))
            .1
    }

    /// Input `key` as field elements.
    pub fn elements<F: PrimeField>(&self, key: &str) -> Vec<F> {
        self.input(key).iter().map(|h| field(h)).collect()
    }

    /// Input `key` as a number.
    pub fn number(&self, key: &str) -> usize {
        self.input(key)[0].parse().unwrap()
    }

    /// The outcome labelled `label`.
    pub fn verdict(&self, label: &str) -> &str {
        &self
            .verdicts
            .iter()
            .find(|(l, _)| l == label)
            .unwrap_or_else(|| panic!("no verdict {label}"))
            .1
    }
}

/// Kimchi's run of the test `name`.
pub fn case(name: &str) -> Case {
    let mut found = None;
    let mut lines = VECTORS.lines();
    while let Some(line) = lines.next() {
        if let Some(n) = line.strip_prefix("test ") {
            if found.is_some() {
                break;
            }
            if n == name {
                found = Some(Case::default());
            }
            continue;
        }
        let Some(case) = found.as_mut() else { continue };
        let mut words = line.split_whitespace();
        match words.next() {
            Some("input") => {
                let key = words.next().unwrap().to_string();
                case.inputs.push((key, words.map(String::from).collect()));
            }
            Some("witness") => {
                let rows: usize = words.next().unwrap().parse().unwrap();
                let witness = (0..rows)
                    .map(|_| {
                        let row = lines.next().unwrap();
                        row.split_whitespace().skip(1).map(String::from).collect()
                    })
                    .collect();
                case.witnesses.push(witness);
            }
            Some("verdict") => {
                let label = words.next().unwrap().to_string();
                let rest = line.splitn(3, ' ').nth(2).unwrap_or("").to_string();
                case.verdicts.push((label, rest));
            }
            _ => {}
        }
    }
    found.unwrap_or_else(|| panic!("no test {name}"))
}

/// A field element from 64 hex digits, big-endian.
pub fn field<F: PrimeField>(hex: &str) -> F {
    let limbs: Vec<u64> = (0..4)
        .rev()
        .map(|i| u64::from_str_radix(&hex[16 * i..16 * (i + 1)], 16).unwrap())
        .collect();
    F::from_limbs(&limbs).unwrap()
}

/// 64 hex digits, big-endian.
pub fn hex<F: PrimeField>(x: F) -> String {
    (0..F::LIMBS)
        .rev()
        .map(|i| format!("{:016x}", x.limb(i)))
        .collect()
}

/// A witness by rows of hex cells, as the vectors print it.
pub fn rows<F: PrimeField>(witness: &[Vec<F>; COLUMNS]) -> Vec<Vec<String>> {
    (0..witness[0].len())
        .map(|r| (0..COLUMNS).map(|c| hex(witness[c][r])).collect())
        .collect()
}

/// Kimchi's gates over Vesta's scalar field.
pub fn vesta_gates() -> KimchiGates<Fp> {
    KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap()
}

/// Kimchi's gates over Pallas's scalar field.
pub fn pallas_gates() -> KimchiGates<Fq> {
    KimchiGates::new(&vesta_base(), Vesta::beta()).unwrap()
}

/// Kimchi's `TestFramework::prove_and_verify` on Vesta: the constraint
/// system of `gates` with `public` public inputs, the witness checked
/// against it, a proof, and its verification, each failure as Kimchi's
/// framework reports it.
pub fn prove_and_verify(
    gates: Vec<CircuitGate<Fp>>,
    witness: [Vec<Fp>; COLUMNS],
    public: Vec<Fp>,
) -> Result<(), String> {
    let kimchi = vesta_gates();
    let cs = ConstraintSystem::create(gates)
        .public(public.len())
        .build()
        .map_err(|e| format!("{e:?}"))?;
    cs.verify_witness(&witness, &public, &kimchi)
        .map_err(|e| format!("{e:?}"))?;
    let depth = cs.domain.d1.size();
    let index: ProverIndex<Vesta> = ProverIndex::create(cs, Srs::create(depth));
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(250));
    let proof = prove(
        &index,
        witness,
        &[],
        &[],
        &kimchi,
        (&base, &scalar),
        &mut rng,
    )
    .map_err(|e| e.to_string())?;
    verify(
        &index.verifier_index,
        &index.srs,
        &proof,
        &public,
        &kimchi,
        (&base, &scalar),
        &mut rng,
    )
    .map_err(|e| e.to_string())
}

/// The `Debug` of a result, as the vectors print verdicts.
pub fn debug<T: core::fmt::Debug>(x: &T) -> String {
    format!("{x:?}")
}
