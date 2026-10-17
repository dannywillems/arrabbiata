//! What the ports of Kimchi's gate tests share: Kimchi's runs of them
//! (`vectors/kimchi_gate_tests.txt`, from the `gate_tests` tool of
//! `tools/kimchi-vectors`, each test with a fixed seed for the inputs Kimchi
//! draws at random), and Kimchi's test framework, `prove_and_verify`, on
//! our prover and verifier.

#![allow(dead_code)]

use std::{collections::BTreeSet, sync::LazyLock};

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{PrimeField, Sqrt, TwoAdicField};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, GateType, Wire},
    gates::{Gates, KimchiGates},
    lookup::LookupPattern,
    prover::{PointEvaluations, ProverIndex, ProverProof, prove},
    rng::{Arkworks, RandomSource, SplitMix64},
    srs::{PolyComm, Srs},
    verifier::verify,
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};
use arrabbiata_transcript::Transcript;

/// Kimchi's runs.
static VECTORS: LazyLock<String> =
    LazyLock::new(|| crate::kimchi_runs::read("kimchi_gate_tests.txt"));

/// One of Kimchi's tests as Kimchi ran it.
#[derive(Debug, Default)]
pub struct Case {
    /// The inputs, by key, as printed.
    pub inputs: Vec<(String, Vec<String>)>,
    /// Kimchi's witnesses, in order, each by rows of 15 hex cells.
    pub witnesses: Vec<Vec<Vec<String>>>,
    /// The outcomes, by label, as Kimchi's `Debug` prints them.
    pub verdicts: Vec<(String, String)>,
    /// Kimchi's proof, in the prover tool's lines.
    pub proof: Vec<String>,
    /// Kimchi's rows, each block as printed, padding left out.
    pub gates: Vec<Vec<KimchiRow>>,
}

/// One of Kimchi's rows as the tool prints it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KimchiRow {
    /// The gate type, as `Debug` prints it.
    pub typ: String,
    /// The wires, `(row, col)`.
    pub wires: Vec<(usize, usize)>,
    /// The coefficients, in hex.
    pub coeffs: Vec<String>,
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
            Some("proof") => case.proof.push(line["proof ".len()..].to_string()),
            Some("gates") => {
                let count: usize = words.next().unwrap().parse().unwrap();
                let block = (0..count)
                    .map(|_| {
                        let row: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                        let wires = row[2..9]
                            .iter()
                            .map(|w| {
                                let (r, c) = w.split_once(':').unwrap();
                                (r.parse().unwrap(), c.parse().unwrap())
                            })
                            .collect();
                        KimchiRow {
                            typ: row[1].to_string(),
                            wires,
                            coeffs: row[10..].iter().map(|c| (*c).to_string()).collect(),
                        }
                    })
                    .collect();
                case.gates.push(block);
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

/// A field element from hex digits read as little-endian bytes, as
/// o1-utils' `FieldHelpers::from_hex` reads them.
pub fn from_hex_le<F: PrimeField>(hex: &str) -> F {
    let bytes: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    let limbs: Vec<u64> = bytes
        .chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u64, |acc, (i, b)| acc | (u64::from(*b) << (8 * i)))
        })
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

/// Kimchi's `TestFramework::prove_and_verify` on the curve `C`: the
/// constraint system of `gates` with `public` public inputs, the witness
/// checked against it, a proof, and its verification, each failure as
/// Kimchi's framework reports it.
pub fn prove_and_verify_on<C, G, S, T>(
    kimchi: &G,
    (base, scalar): (&S, &T),
    gates: Vec<CircuitGate<C::Scalar>>,
    witness: [Vec<C::Scalar>; COLUMNS],
    public: Vec<C::Scalar>,
) -> Result<(), String>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
    G: Gates<C::Scalar>,
    S: Transcript<C::Base> + Clone,
    T: Transcript<C::Scalar> + Clone,
{
    let cs = ConstraintSystem::create(gates)
        .public(public.len())
        .build()
        .map_err(|e| format!("{e:?}"))?;
    cs.verify_witness(&witness, &public, kimchi)
        .map_err(|e| format!("{e:?}"))?;
    let depth = cs.domain.d1.size();
    let index: ProverIndex<C> = ProverIndex::create(cs, Srs::create(depth));
    let mut rng = Arkworks(SplitMix64(250));
    let proof = prove(&index, witness, &[], &[], kimchi, (base, scalar), &mut rng)
        .map_err(|e| e.to_string())?;
    verify(
        &index.verifier_index,
        &index.srs,
        &proof,
        &public,
        kimchi,
        (base, scalar),
        &mut rng,
    )
    .map_err(|e| e.to_string())
}

/// [`prove_and_verify_on`] Vesta.
pub fn prove_and_verify(
    gates: Vec<CircuitGate<Fp>>,
    witness: [Vec<Fp>; COLUMNS],
    public: Vec<Fp>,
) -> Result<(), String> {
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    prove_and_verify_on::<Vesta, _, _, _>(&vesta_gates(), (&base, &scalar), gates, witness, public)
}

/// [`prove_and_verify_on`] Pallas.
pub fn prove_and_verify_pallas(
    gates: Vec<CircuitGate<Fq>>,
    witness: [Vec<Fq>; COLUMNS],
    public: Vec<Fq>,
) -> Result<(), String> {
    let base = KimchiSponge::new(pallas_base());
    let scalar = KimchiSponge::new(vesta_base());
    prove_and_verify_on::<Pallas, _, _, _>(
        &pallas_gates(),
        (&base, &scalar),
        gates,
        witness,
        public,
    )
}

/// The `Debug` of a result, as the vectors print verdicts.
pub fn debug<T: core::fmt::Debug>(x: &T) -> String {
    format!("{x:?}")
}

/// A stream of words replayed: the words Kimchi's prover drew.
#[derive(Clone, Debug)]
pub struct Replay(pub Vec<u64>, pub usize);

impl RandomSource for Replay {
    fn next_u64(&mut self) -> u64 {
        let word = self.0[self.1];
        self.1 += 1;
        word
    }
}

fn point(p: &Projective<Vesta>) -> String {
    let a = p.to_affine();
    match (a.x(), a.y()) {
        (Some(x), Some(y)) => format!("{} {}", hex(x), hex(y)),
        _ => String::from("inf"),
    }
}

fn comm(c: &PolyComm<Vesta>) -> String {
    let chunks: Vec<String> = c.chunks.iter().map(point).collect();
    format!("{} {}", c.chunks.len(), chunks.join(" "))
}

fn values(v: &[Fp]) -> String {
    let hexes: Vec<String> = v.iter().map(|x| hex(*x)).collect();
    format!("{} {}", v.len(), hexes.join(" "))
}

fn evaluation(e: &PointEvaluations<Fp>) -> String {
    format!("{} {}", values(&e.zeta), values(&e.zeta_omega))
}

/// Our proof in the prover tool's lines, the absent evaluations left out,
/// sorted (the tool lists the optional selectors in Kimchi's order).
pub fn proof_lines(name: &str, proof: &ProverProof<Vesta>) -> Vec<String> {
    let mut out = Vec::new();
    for (i, c) in proof.w_comm.iter().enumerate() {
        out.push(format!("comm {name} w{i} {}", comm(c)));
    }
    out.push(format!("comm {name} z {}", comm(&proof.z_comm)));
    out.push(format!("comm {name} t {}", comm(&proof.t_comm)));
    if let Some(l) = &proof.lookup {
        for (i, c) in l.sorted.iter().enumerate() {
            out.push(format!("comm {name} sorted{i} {}", comm(c)));
        }
        out.push(format!("comm {name} aggreg {}", comm(&l.aggreg)));
        if let Some(r) = &l.runtime {
            out.push(format!("comm {name} runtime {}", comm(r)));
        }
    }
    let e = &proof.evals;
    let mut put = |label: String, p: &PointEvaluations<Fp>| {
        out.push(format!("eval {name} {label} {}", evaluation(p)));
    };
    put("public".into(), &e.public);
    for (i, w) in e.w.iter().enumerate() {
        put(format!("w{i}"), w);
    }
    put("z".into(), &e.z);
    for (i, s) in e.s.iter().enumerate() {
        put(format!("s{i}"), s);
    }
    for (i, c) in e.coefficients.iter().enumerate() {
        put(format!("coefficient{i}"), c);
    }
    for (label, p) in [
        ("Generic", &e.generic_selector),
        ("Poseidon", &e.poseidon_selector),
        ("CompleteAdd", &e.complete_add_selector),
        ("VarBaseMul", &e.mul_selector),
        ("EndoMul", &e.emul_selector),
        ("EndoMulScalar", &e.endomul_scalar_selector),
    ] {
        put(label.into(), p);
    }
    for (typ, p) in &e.optional_selectors {
        put(format!("{typ:?}"), p);
    }
    if let Some(l) = &e.lookup {
        put("lookup_aggregation".into(), &l.aggregation);
        put("lookup_table".into(), &l.table);
        for (i, s) in l.sorted.iter().enumerate() {
            put(format!("lookup_sorted{i}"), s);
        }
        if let Some(r) = &l.runtime_table {
            put("runtime_lookup_table".into(), r);
        }
        if let Some(r) = &l.runtime_selector {
            put("runtime_lookup_table_selector".into(), r);
        }
        for (label, pattern) in [
            ("xor_lookup_selector", LookupPattern::Xor),
            ("lookup_gate_lookup_selector", LookupPattern::Lookup),
            ("range_check_lookup_selector", LookupPattern::RangeCheck),
            (
                "foreign_field_mul_lookup_selector",
                LookupPattern::ForeignFieldMul,
            ),
        ] {
            if let Some((_, p)) = l.selectors.iter().find(|(q, _)| *q == pattern) {
                put(label.into(), p);
            }
        }
    }
    out.push(format!("ft_eval1 {name} {}", hex(proof.ft_eval1)));
    let o = &proof.proof;
    for (i, (l, r)) in o.lr.iter().enumerate() {
        out.push(format!("opening {name} lr{i} {} {}", point(l), point(r)));
    }
    out.push(format!("opening {name} delta {}", point(&o.delta)));
    out.push(format!("opening {name} z1 {}", hex(o.z1)));
    out.push(format!("opening {name} z2 {}", hex(o.z2)));
    out.push(format!("opening {name} sg {}", point(&o.sg)));
    out.sort();
    out
}

/// Our proof of `witness` for `gates` on Vesta, drawing `words` as Kimchi's
/// prover drew them.
pub fn prove_replayed(
    gates: Vec<CircuitGate<Fp>>,
    witness: [Vec<Fp>; COLUMNS],
    words: Vec<u64>,
) -> ProverProof<Vesta> {
    let kimchi = vesta_gates();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    cs.verify_witness(&witness, &[], &kimchi).unwrap();
    let depth = cs.domain.d1.size();
    let index: ProverIndex<Vesta> = ProverIndex::create(cs, Srs::create(depth));
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(Replay(words, 0));
    prove(
        &index,
        witness,
        &[],
        &[],
        &kimchi,
        (&base, &scalar),
        &mut rng,
    )
    .unwrap()
}

/// The gate type `Debug` prints as `name`.
pub fn gate_type(name: &str) -> GateType {
    let all = [
        GateType::Zero,
        GateType::Generic,
        GateType::Poseidon,
        GateType::CompleteAdd,
        GateType::VarBaseMul,
        GateType::EndoMul,
        GateType::EndoMulScalar,
        GateType::Lookup,
        GateType::RangeCheck0,
        GateType::RangeCheck1,
        GateType::ForeignFieldAdd,
        GateType::ForeignFieldMul,
        GateType::Xor16,
        GateType::Rot64,
    ];
    *all.iter().find(|t| format!("{t:?}") == name).unwrap()
}

/// Kimchi's rows as ours, for a proof that must be Kimchi's bit for bit
/// (the sigmas follow the order Kimchi's swaps leave in each cycle).
pub fn kimchi_gates<F: PrimeField>(rows: &[KimchiRow]) -> Vec<CircuitGate<F>> {
    rows.iter()
        .map(|r| {
            let wires = core::array::from_fn(|c| Wire {
                row: r.wires[c].0,
                col: r.wires[c].1,
            });
            CircuitGate::new(
                gate_type(&r.typ),
                wires,
                r.coeffs.iter().map(|h| field(h)).collect(),
            )
        })
        .collect()
}

/// The copy cycles of rows, as sorted sets of cells: what the wiring
/// means, whatever order each cycle is written in.
pub fn partition(wires: &[[(usize, usize); 7]]) -> Vec<Vec<(usize, usize)>> {
    let mut seen = BTreeSet::new();
    let mut cycles = Vec::new();
    for (row, w) in wires.iter().enumerate() {
        for (col, &first) in w.iter().enumerate() {
            if !seen.insert((row, col)) {
                continue;
            }
            let mut cycle = vec![(row, col)];
            let mut at = first;
            while at != (row, col) {
                seen.insert(at);
                cycle.push(at);
                at = wires[at.0][at.1];
            }
            cycle.sort_unstable();
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
    }
    cycles.sort();
    cycles
}

/// Our rows are Kimchi's: the same gate types and coefficients (trailing
/// zeros aside), and the same copy cycles as sets of cells.
pub fn assert_kimchi_rows<F: PrimeField>(ours: &[CircuitGate<F>], kimchi: &[KimchiRow]) {
    assert!(
        ours.len() >= kimchi.len(),
        "{} rows, Kimchi has {}",
        ours.len(),
        kimchi.len()
    );
    let trim = |mut c: Vec<String>| {
        let zero = hex(F::ZERO);
        while c.last() == Some(&zero) {
            c.pop();
        }
        c
    };
    for (row, (g, k)) in ours.iter().zip(kimchi).enumerate() {
        assert_eq!(format!("{:?}", g.typ), k.typ, "row {row}");
        let coeffs: Vec<String> = g.coeffs.iter().map(|c| hex(*c)).collect();
        assert_eq!(
            trim(coeffs),
            trim(k.coeffs.clone()),
            "coefficients of row {row}"
        );
    }
    let n = kimchi.len();
    let ours_wires: Vec<[(usize, usize); 7]> = ours[..n]
        .iter()
        .map(|g| core::array::from_fn(|c| (g.wires[c].row, g.wires[c].col)))
        .collect();
    let theirs: Vec<[(usize, usize); 7]> = kimchi
        .iter()
        .map(|k| core::array::from_fn(|c| k.wires[c]))
        .collect();
    assert_eq!(partition(&ours_wires), partition(&theirs), "copy cycles");
}
