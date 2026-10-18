//! Kimchi's prover against Kimchi, round by round: the circuits of the
//! `prover` tool of `tools/kimchi-vectors`, proved with SplitMix64(7) over
//! Vesta and over Pallas (`vectors/kimchi_prover.txt`). The tests rebuild the circuit and
//! the witness from the vectors, run our rounds on the same stream, and
//! compare each value the round produces.

mod kimchi_runs;

use std::{collections::BTreeMap, sync::LazyLock};

use arrabbiata_codec::laws as codec_laws;
use arrabbiata_curve::{Affine, Curve, Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_halo::Accumulated;
use arrabbiata_ipa::RecursionChallenge;
use arrabbiata_kimchi::{
    accumulation::{
        self, Assignment, KimchiSuccinct, Satisfied, Statement, accumulate_proofs, ipa_of, separate,
    },
    constraints::{CircuitGate, ConstraintSystem, GateType, LookupTable, RuntimeTableCfg, Wire},
    gates::KimchiGates,
    lookup::{LookupPattern, RuntimeTable},
    prover::{
        LookupCommitments, LookupEvaluations, OpeningProof, PointEvaluations, ProofEvaluations,
        ProverIndex, ProverProof, evaluation_round, permutation_round, prove, quotient_round,
        witness_round,
    },
    rng::{Arkworks, Sampler, SplitMix64},
    srs::{PolyComm, Srs},
    verifier::{VerifyError, oracles, succinct, verify},
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};
use arrabbiata_reduction::{Reduction, Relation, laws};
use blake2::{Blake2b512, Digest};

/// Kimchi's output.
static VECTORS: LazyLock<String> = LazyLock::new(|| kimchi_runs::read("kimchi_prover.txt"));

/// The circuits of the tool, their chunk sizes and public inputs: Kimchi's
/// generic test circuit unchunked, in chunks of 16 rows, and on a key of 64
/// generators, wider than its domain (one chunk, padded), one block of
/// each gate without lookups (Poseidon, CompleteAdd, VarBaseMul, EndoMul,
/// EndoMulScalar), and the generic circuit carrying one previous
/// accumulator (as Kimchi's `test_recursion`); then four circuits with
/// lookups: a 64-bit Xor, a multi-range check, Lookup gates over two fixed
/// tables, over two runtime tables, a 64-bit word rotated left and right
/// (Rot64), and a foreign-field multiplication modulo secp256k1's base
/// field with all its external checks (ForeignFieldMul), and a chain of
/// two foreign-field operations with every range check (ForeignFieldAdd);
/// then features together: lookups in chunks (Lookup gates, the Xor), two
/// previous accumulators, and the generic circuit's public inputs followed
/// by a Xor.
const CIRCUITS: [(&str, Option<usize>, usize); 16] = [
    ("generic", None, 5),
    ("chunked", Some(16), 5),
    ("wide", Some(64), 5),
    ("gadgets", None, 0),
    ("recursion", None, 5),
    ("xor", None, 0),
    ("range_check", None, 0),
    ("lookup", None, 0),
    ("runtime", None, 0),
    ("rot", None, 0),
    ("ffmul", None, 0),
    ("ffadd", None, 1),
    ("lookup_chunked", Some(8), 0),
    ("xor_chunked", Some(16), 0),
    ("recursion2", None, 5),
    ("public_lookup", None, 5),
];

/// The tool's circuits over Pallas: Kimchi's generic test circuit,
/// unchunked and in chunks of 16 rows.
const PALLAS_CIRCUITS: [(&str, Option<usize>, usize); 2] =
    [("pallas", None, 5), ("pallas_chunked", Some(16), 5)];

/// A curve Kimchi proves over, with what the tests need of it: Kimchi's
/// gates over its scalar field, its two sponges, and the tool's circuits on
/// it.
trait Kimchi:
    Glv + Curve<Base: PrimeField + Sqrt, Scalar: TwoAdicField + arrabbiata_codec::Codec>
{
    const CIRCUITS: &'static [(&'static str, Option<usize>, usize)];
    fn gates() -> KimchiGates<Self::Scalar>;
    fn base() -> KimchiSponge<Self::Base>;
    fn scalar() -> KimchiSponge<Self::Scalar>;
}

impl Kimchi for Vesta {
    const CIRCUITS: &'static [(&'static str, Option<usize>, usize)] = &CIRCUITS;
    fn gates() -> KimchiGates<Fp> {
        KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap()
    }
    fn base() -> KimchiSponge<Fq> {
        KimchiSponge::new(vesta_base())
    }
    fn scalar() -> KimchiSponge<Fp> {
        KimchiSponge::new(pallas_base())
    }
}

impl Kimchi for Pallas {
    const CIRCUITS: &'static [(&'static str, Option<usize>, usize)] = &PALLAS_CIRCUITS;
    fn gates() -> KimchiGates<Fq> {
        KimchiGates::new(&vesta_base(), Vesta::beta()).unwrap()
    }
    fn base() -> KimchiSponge<Fp> {
        KimchiSponge::new(pallas_base())
    }
    fn scalar() -> KimchiSponge<Fq> {
        KimchiSponge::new(vesta_base())
    }
}

/// The entry of [`CIRCUITS`] named `name`.
fn named(name: &str) -> (&'static str, Option<usize>, usize) {
    match CIRCUITS.iter().find(|(n, _, _)| *n == name) {
        Some(c) => *c,
        None => panic!("no circuit {name}"),
    }
}

/// Values from `k x1 ... xk`.
fn values_of<F: PrimeField>(words: &[&str]) -> Vec<F> {
    let k: usize = words[0].parse().unwrap();
    words[1..=k].iter().map(|h| field(h)).collect()
}

/// The fixed tables of the tool's circuit `name`.
fn tables<F: PrimeField>(name: &str) -> Vec<LookupTable<F>> {
    let mut tables: Vec<LookupTable<F>> = Vec::new();
    for line in VECTORS.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words[0] != "table" || words[1] != name {
            continue;
        }
        let id: i32 = words[2].parse().unwrap();
        let column = values_of(&words[4..]);
        match tables.iter_mut().find(|t| t.id == id) {
            Some(t) => t.data.push(column),
            None => tables.push(LookupTable {
                id,
                data: vec![column],
            }),
        }
    }
    tables
}

/// The runtime table configuration of the tool's circuit `name`.
fn runtime_cfgs<F: PrimeField>(name: &str) -> Option<Vec<RuntimeTableCfg<F>>> {
    let cfgs: Vec<RuntimeTableCfg<F>> = VECTORS
        .lines()
        .filter(|l| l.starts_with(&format!("runtime_cfg {name} ")))
        .map(|l| {
            let words: Vec<&str> = l.split_whitespace().collect();
            RuntimeTableCfg {
                id: words[2].parse().unwrap(),
                first_column: values_of(&words[3..]),
            }
        })
        .collect();
    (!cfgs.is_empty()).then_some(cfgs)
}

/// The runtime tables' data of the tool's circuit `name`.
fn runtime_tables<F: PrimeField>(name: &str) -> Vec<RuntimeTable<F>> {
    VECTORS
        .lines()
        .filter(|l| l.starts_with(&format!("runtime_data {name} ")))
        .map(|l| {
            let words: Vec<&str> = l.split_whitespace().collect();
            RuntimeTable {
                id: words[2].parse().unwrap(),
                data: values_of(&words[3..]),
            }
        })
        .collect()
}

fn hex<F: PrimeField>(x: F) -> String {
    (0..F::LIMBS)
        .rev()
        .map(|i| format!("{:016x}", x.limb(i)))
        .collect()
}

fn field<F: PrimeField>(hex: &str) -> F {
    let limbs: Vec<u64> = (0..4)
        .rev()
        .map(|i| u64::from_str_radix(&hex[16 * i..16 * (i + 1)], 16).unwrap())
        .collect();
    F::from_limbs(&limbs).unwrap()
}

/// The previous accumulators of the tool's circuit `name`.
fn prev_challenges<C: Kimchi>(name: &str) -> Vec<RecursionChallenge<C>> {
    let prev = lines("prev", name);
    // Accumulator 0 is labelled `chal<i>`, `comm`; accumulator j > 0 with
    // the suffix `_<j>`.
    (0..)
        .map(|j: usize| {
            if j == 0 {
                String::new()
            } else {
                format!("_{j}")
            }
        })
        .map_while(|suffix| {
            let words: Vec<&str> = prev
                .get(&format!("comm{suffix}"))?
                .split_whitespace()
                .collect();
            assert_eq!(words[0], "1");
            let chals = (0..)
                .map_while(|i| prev.get(&format!("chal{i}{suffix}")).map(|h| field(h)))
                .collect();
            let comm = Affine::<C>::new(field(words[1]), field(words[2]))
                .unwrap()
                .to_projective();
            Some(RecursionChallenge { chals, comm })
        })
        .collect()
}

fn point<C: Kimchi>(p: &Projective<C>) -> String {
    let a = p.to_affine();
    match (a.x(), a.y()) {
        (Some(x), Some(y)) => format!("{} {}", hex(x), hex(y)),
        _ => String::from("inf"),
    }
}

fn comm<C: Kimchi>(c: &PolyComm<C>) -> String {
    let chunks: Vec<String> = c.chunks.iter().map(point).collect();
    format!("{} {}", c.chunks.len(), chunks.join(" "))
}

/// Kimchi's lines of the given kind and circuit, by label.
fn lines(kind: &str, name: &str) -> BTreeMap<String, String> {
    VECTORS
        .lines()
        .filter_map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            (words[0] == kind && words[1] == name)
                .then(|| (words[2].to_string(), words[3..].join(" ")))
        })
        .collect()
}

fn gate_type(name: &str) -> GateType {
    match name {
        "Zero" => GateType::Zero,
        "Generic" => GateType::Generic,
        "Poseidon" => GateType::Poseidon,
        "CompleteAdd" => GateType::CompleteAdd,
        "VarBaseMul" => GateType::VarBaseMul,
        "EndoMul" => GateType::EndoMul,
        "EndoMulScalar" => GateType::EndoMulScalar,
        "Lookup" => GateType::Lookup,
        "RangeCheck0" => GateType::RangeCheck0,
        "RangeCheck1" => GateType::RangeCheck1,
        "ForeignFieldAdd" => GateType::ForeignFieldAdd,
        "ForeignFieldMul" => GateType::ForeignFieldMul,
        "Xor16" => GateType::Xor16,
        "Rot64" => GateType::Rot64,
        other => panic!("gate {other} not in the vectors"),
    }
}

/// The circuit and the (unpadded) witness of the tool's circuit `name`.
fn circuit<F: PrimeField>(name: &str) -> (Vec<CircuitGate<F>>, [Vec<F>; 15]) {
    let mut gates = Vec::new();
    for line in VECTORS.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words[0] != "gate" || words[1] != name {
            continue;
        }
        assert_eq!(words[2].parse::<usize>().unwrap(), gates.len());
        let wires: [Wire; 7] = core::array::from_fn(|i| Wire {
            row: words[4 + 2 * i].parse().unwrap(),
            col: words[5 + 2 * i].parse().unwrap(),
        });
        let count: usize = words[18].parse().unwrap();
        let coeffs = words[19..19 + count].iter().map(|h| field(h)).collect();
        gates.push(CircuitGate::new(gate_type(words[3]), wires, coeffs));
    }
    let mut witness: [Vec<F>; 15] = core::array::from_fn(|_| vec![F::ZERO; gates.len()]);
    for line in VECTORS.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words[0] == "witness" && words[1] == name {
            let col: usize = words[2].parse().unwrap();
            let row: usize = words[3].parse().unwrap();
            witness[col][row] = field(words[4]);
        }
    }
    (gates, witness)
}

fn index<C: Kimchi>(
    name: &str,
    max_poly_size: Option<usize>,
    public: usize,
) -> (ProverIndex<C>, [Vec<C::Scalar>; 15]) {
    let (gates, witness) = circuit(name);
    let cs = ConstraintSystem::create(gates)
        .public(public)
        .prev_challenges(prev_challenges::<C>(name).len())
        .lookup(tables(name))
        .runtime(runtime_cfgs(name))
        .max_poly_size(max_poly_size)
        .build()
        .unwrap();
    let depth = max_poly_size.unwrap_or(cs.domain.d1.size());
    (ProverIndex::create(cs, Srs::create(depth)), witness)
}

#[test]
fn the_samples_are_arkworks() {
    let all: Vec<String> = VECTORS
        .lines()
        .filter(|l| l.starts_with("rand "))
        .map(|l| l.split_whitespace().nth(2).unwrap().to_string())
        .collect();
    let mut rng = Arkworks(SplitMix64(42));
    let ours: Vec<String> = (0..all.len()).map(|_| hex::<Fp>(rng.sample())).collect();
    assert_eq!(all.len(), 8);
    assert_eq!(ours, all);
}

#[test]
fn the_digests_are_kimchis() {
    the_digests_are_kimchis_on::<Vesta>();
    the_digests_are_kimchis_on::<Pallas>();
}

fn the_digests_are_kimchis_on<C: Kimchi>() {
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, _) = index::<C>(name, max_poly_size, public);
        let sponge = C::base();
        let digest = index.verifier_index.digest(sponge);
        let expected = VECTORS
            .lines()
            .find(|l| l.starts_with(&format!("digest {name} ")))
            .unwrap();
        assert_eq!(format!("digest {name} {}", hex(digest)), expected);
    }
}

#[test]
fn the_witness_round_is_kimchis() {
    the_witness_round_is_kimchis_on::<Vesta>();
    the_witness_round_is_kimchis_on::<Pallas>();
}

fn the_witness_round_is_kimchis_on<C: Kimchi>() {
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, witness) = index::<C>(name, max_poly_size, public);
        let sponge = C::base();
        let mut rng = Arkworks(SplitMix64(7));
        let (round, _) = witness_round(
            &index,
            witness,
            &runtime_tables::<C::Scalar>(name),
            &prev_challenges::<C>(name),
            &sponge,
            &mut rng,
        )
        .unwrap();
        let comms = lines("comm", name);
        assert_eq!(comms["public"], comm(&round.public_comm), "{name} public");
        for (i, c) in round.w_comm.iter().enumerate() {
            assert_eq!(comms[&format!("w{i}")], comm(c), "{name} w{i}");
        }
        let oracles = lines("oracle", name);
        if let Some(l) = &round.lookup {
            for (i, c) in l.sorted.iter().enumerate() {
                assert_eq!(
                    comms[&format!("sorted{i}")],
                    comm(&c.comm),
                    "{name} sorted{i}"
                );
            }
            assert_eq!(
                comms.contains_key("runtime"),
                l.runtime.is_some(),
                "{name} runtime"
            );
            if let Some(r) = &l.runtime {
                assert_eq!(comms["runtime"], comm(&r.comm), "{name} runtime");
            }
            assert_eq!(
                oracles["joint_combiner_chal"],
                hex(l.joint_combiner_chal.0),
                "{name}"
            );
            assert_eq!(oracles["joint_combiner"], hex(l.joint_combiner), "{name}");
        } else {
            assert!(!comms.contains_key("sorted0"), "{name}");
        }
        assert_eq!(oracles["beta"], hex(round.beta), "{name} beta");
        assert_eq!(oracles["gamma"], hex(round.gamma), "{name} gamma");
    }
}

#[test]
fn the_permutation_round_is_kimchis() {
    the_permutation_round_is_kimchis_on::<Vesta>();
    the_permutation_round_is_kimchis_on::<Pallas>();
}

fn the_permutation_round_is_kimchis_on<C: Kimchi>() {
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, witness) = index::<C>(name, max_poly_size, public);
        let sponge = C::base();
        let mut rng = Arkworks(SplitMix64(7));
        let (first, tape) = witness_round(
            &index,
            witness,
            &runtime_tables::<C::Scalar>(name),
            &prev_challenges::<C>(name),
            &sponge,
            &mut rng,
        )
        .unwrap();
        let (round, _) = permutation_round(&index, first, tape, &mut rng).unwrap();
        let comms = lines("comm", name);
        assert_eq!(comms["z"], comm(&round.z_comm), "{name} z");
        match &round.aggregation {
            Some(a) => assert_eq!(comms["aggreg"], comm(&a.comm), "{name} aggreg"),
            None => assert!(!comms.contains_key("aggreg"), "{name}"),
        }
        let oracles = lines("oracle", name);
        assert_eq!(oracles["alpha_chal"], hex(round.alpha_chal.0), "{name}");
        assert_eq!(oracles["alpha"], hex(round.alpha), "{name} alpha");
    }
}

#[test]
fn the_quotient_round_is_kimchis() {
    the_quotient_round_is_kimchis_on::<Vesta>();
    the_quotient_round_is_kimchis_on::<Pallas>();
}

fn the_quotient_round_is_kimchis_on<C: Kimchi>() {
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, witness) = index::<C>(name, max_poly_size, public);
        let sponge = C::base();
        let mut rng = Arkworks(SplitMix64(7));
        let (first, tape) = witness_round(
            &index,
            witness,
            &runtime_tables::<C::Scalar>(name),
            &prev_challenges::<C>(name),
            &sponge,
            &mut rng,
        )
        .unwrap();
        let (second, tape) = permutation_round(&index, first, tape, &mut rng).unwrap();
        let round = quotient_round(&index, second, tape, &gates, &mut rng).unwrap();
        assert_eq!(lines("comm", name)["t"], comm(&round.t_comm), "{name} t");
        let oracles = lines("oracle", name);
        assert_eq!(oracles["zeta_chal"], hex(round.zeta_chal.0), "{name}");
        assert_eq!(oracles["zeta"], hex(round.zeta), "{name} zeta");
    }
}

fn values<F: PrimeField>(v: &[F]) -> String {
    let hexes: Vec<String> = v.iter().map(|x| hex(*x)).collect();
    format!("{} {}", v.len(), hexes.join(" "))
}

fn evaluation<F: PrimeField>(e: &PointEvaluations<F>) -> String {
    format!("{} {}", values(&e.zeta), values(&e.zeta_omega))
}

#[test]
fn the_evaluation_round_is_kimchis() {
    the_evaluation_round_is_kimchis_on::<Vesta>();
    the_evaluation_round_is_kimchis_on::<Pallas>();
}

fn the_evaluation_round_is_kimchis_on<C: Kimchi>() {
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, witness) = index::<C>(name, max_poly_size, public);
        let sponge = C::base();
        let scalar_sponge = C::scalar();
        let mut rng = Arkworks(SplitMix64(7));
        let (first, tape) = witness_round(
            &index,
            witness,
            &runtime_tables::<C::Scalar>(name),
            &prev_challenges::<C>(name),
            &sponge,
            &mut rng,
        )
        .unwrap();
        let (second, tape) = permutation_round(&index, first, tape, &mut rng).unwrap();
        let third = quotient_round(&index, second, tape, &gates, &mut rng).unwrap();
        let round = evaluation_round(&index, third, &scalar_sponge);
        let expected = lines("eval", name);
        let e = &round.evals;
        let mut ours = vec![("public".to_string(), evaluation(&e.public))];
        for (i, w) in e.w.iter().enumerate() {
            ours.push((format!("w{i}"), evaluation(w)));
        }
        ours.push(("z".to_string(), evaluation(&e.z)));
        for (i, s) in e.s.iter().enumerate() {
            ours.push((format!("s{i}"), evaluation(s)));
        }
        for (i, c) in e.coefficients.iter().enumerate() {
            ours.push((format!("coefficient{i}"), evaluation(c)));
        }
        for (label, p) in [
            ("Generic", &e.generic_selector),
            ("Poseidon", &e.poseidon_selector),
            ("CompleteAdd", &e.complete_add_selector),
            ("VarBaseMul", &e.mul_selector),
            ("EndoMul", &e.emul_selector),
            ("EndoMulScalar", &e.endomul_scalar_selector),
        ] {
            ours.push((label.to_string(), evaluation(p)));
        }
        // The optional selectors, and the lookup evaluations (none here).
        for (label, typ) in [
            ("RangeCheck0", GateType::RangeCheck0),
            ("RangeCheck1", GateType::RangeCheck1),
            ("ForeignFieldAdd", GateType::ForeignFieldAdd),
            ("ForeignFieldMul", GateType::ForeignFieldMul),
            ("Xor16", GateType::Xor16),
            ("Rot64", GateType::Rot64),
        ] {
            let value = e
                .optional_selectors
                .iter()
                .find(|(t, _)| *t == typ)
                .map_or_else(|| String::from("none"), |(_, p)| evaluation(p));
            ours.push((label.to_string(), value));
        }
        ours.extend(lookup_lines(e));
        assert_eq!(ours.len(), expected.len(), "{name}");
        for (label, value) in &ours {
            assert_eq!(&expected[label], value, "{name} {label}");
        }
        let ft_eval1 = VECTORS
            .lines()
            .find(|l| l.starts_with(&format!("ft_eval1 {name} ")))
            .unwrap();
        assert_eq!(ft_eval1, format!("ft_eval1 {name} {}", hex(round.ft_eval1)));
        let oracles = lines("oracle", name);
        assert_eq!(oracles["v_chal"], hex(round.v_chal.0), "{name}");
        assert_eq!(oracles["v"], hex(round.v), "{name} v");
        assert_eq!(oracles["u_chal"], hex(round.u_chal.0), "{name}");
        assert_eq!(oracles["u"], hex(round.u), "{name} u");
    }
}

#[test]
fn the_proofs_are_kimchis() {
    the_proofs_are_kimchis_on::<Vesta>();
    the_proofs_are_kimchis_on::<Pallas>();
}

fn the_proofs_are_kimchis_on<C: Kimchi>() {
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, witness) = index::<C>(name, max_poly_size, public);
        let base = C::base();
        let scalar = C::scalar();
        let mut rng = Arkworks(SplitMix64(7));
        let proof = prove(
            &index,
            witness,
            &runtime_tables::<C::Scalar>(name),
            &prev_challenges::<C>(name),
            &gates,
            (&base, &scalar),
            &mut rng,
        )
        .unwrap();
        let opening = lines("opening", name);
        let o = &proof.proof;
        assert_eq!(opening.len(), o.lr.len() + 4, "{name}");
        for (i, (l, r)) in o.lr.iter().enumerate() {
            assert_eq!(
                opening[&format!("lr{i}")],
                format!("{} {}", point(l), point(r)),
                "{name} lr{i}"
            );
        }
        assert_eq!(opening["delta"], point(&o.delta), "{name} delta");
        assert_eq!(opening["z1"], hex(o.z1), "{name} z1");
        assert_eq!(opening["z2"], hex(o.z2), "{name} z2");
        assert_eq!(opening["sg"], point(&o.sg), "{name} sg");
        // The rest of the proof is the rounds', checked above.
        let comms = lines("comm", name);
        assert_eq!(comms["t"], comm(&proof.t_comm), "{name}");
    }
}

/// A point from its affine coordinates, or the identity for `inf`.
fn parse_point<C: Kimchi>(words: &[&str]) -> (Projective<C>, usize) {
    if words[0] == "inf" {
        (Projective::identity(), 1)
    } else {
        let p = Affine::<C>::new(field(words[0]), field(words[1])).unwrap();
        (p.to_projective(), 2)
    }
}

/// A commitment from `k x y x y ...`.
fn parse_comm<C: Kimchi>(value: &str) -> PolyComm<C> {
    let words: Vec<&str> = value.split_whitespace().collect();
    let k: usize = words[0].parse().unwrap();
    let mut at = 1;
    let mut chunks = Vec::with_capacity(k);
    for _ in 0..k {
        let (p, used) = parse_point(&words[at..]);
        chunks.push(p);
        at += used;
    }
    PolyComm { chunks }
}

/// Evaluations from `nz z... nw w...`.
fn parse_evaluations<F: PrimeField>(value: &str) -> PointEvaluations<F> {
    let words: Vec<&str> = value.split_whitespace().collect();
    let nz: usize = words[0].parse().unwrap();
    let zeta = words[1..=nz].iter().map(|h| field(h)).collect();
    let nw: usize = words[nz + 1].parse().unwrap();
    let zeta_omega = words[nz + 2..nz + 2 + nw]
        .iter()
        .map(|h| field(h))
        .collect();
    PointEvaluations { zeta, zeta_omega }
}

/// Kimchi's proof of the tool's circuit `name`, read from the vectors.
fn kimchis_proof<C: Kimchi>(name: &str) -> ProverProof<C> {
    let comms = lines("comm", name);
    let evals = lines("eval", name);
    let e = |label: &str| parse_evaluations(&evals[label]);
    let optional_selectors = [
        ("RangeCheck0", GateType::RangeCheck0),
        ("RangeCheck1", GateType::RangeCheck1),
        ("ForeignFieldAdd", GateType::ForeignFieldAdd),
        ("ForeignFieldMul", GateType::ForeignFieldMul),
        ("Xor16", GateType::Xor16),
        ("Rot64", GateType::Rot64),
    ]
    .iter()
    .filter(|(label, _)| evals[*label] != "none")
    .map(|(label, typ)| (*typ, e(label)))
    .collect();
    let opening = lines("opening", name);
    let rounds = opening.keys().filter(|k| k.starts_with("lr")).count();
    let lr = (0..rounds)
        .map(|i| {
            let words: Vec<&str> = opening[&format!("lr{i}")].split_whitespace().collect();
            let (l, used) = parse_point(&words);
            let (r, _) = parse_point(&words[used..]);
            (l, r)
        })
        .collect();
    let point = |label: &str| {
        let words: Vec<&str> = opening[label].split_whitespace().collect();
        parse_point(&words).0
    };
    let ft_eval1 = VECTORS
        .lines()
        .find(|l| l.starts_with(&format!("ft_eval1 {name} ")))
        .unwrap()
        .split_whitespace()
        .nth(2)
        .unwrap();
    let lookup = comms.contains_key("aggreg").then(|| LookupCommitments {
        sorted: (0..)
            .map_while(|i| comms.get(&format!("sorted{i}")).map(|c| parse_comm(c)))
            .collect(),
        aggreg: parse_comm(&comms["aggreg"]),
        runtime: comms.get("runtime").map(|c| parse_comm(c)),
    });
    let some = |label: &str| (evals[label] != "none").then(|| e(label));
    let lookup_evals = (evals["lookup_aggregation"] != "none").then(|| LookupEvaluations {
        aggregation: e("lookup_aggregation"),
        table: e("lookup_table"),
        sorted: (0..5)
            .filter_map(|i| some(&format!("lookup_sorted{i}")))
            .collect(),
        runtime_table: some("runtime_lookup_table"),
        runtime_selector: some("runtime_lookup_table_selector"),
        selectors: [
            ("xor_lookup_selector", LookupPattern::Xor),
            ("lookup_gate_lookup_selector", LookupPattern::Lookup),
            ("range_check_lookup_selector", LookupPattern::RangeCheck),
            (
                "foreign_field_mul_lookup_selector",
                LookupPattern::ForeignFieldMul,
            ),
        ]
        .iter()
        .filter_map(|(label, p)| some(label).map(|v| (*p, v)))
        .collect(),
    });
    ProverProof {
        w_comm: core::array::from_fn(|i| parse_comm(&comms[&format!("w{i}")])),
        z_comm: parse_comm(&comms["z"]),
        t_comm: parse_comm(&comms["t"]),
        lookup,
        evals: ProofEvaluations {
            public: e("public"),
            w: core::array::from_fn(|i| e(&format!("w{i}"))),
            z: e("z"),
            s: core::array::from_fn(|i| e(&format!("s{i}"))),
            coefficients: core::array::from_fn(|i| e(&format!("coefficient{i}"))),
            generic_selector: e("Generic"),
            poseidon_selector: e("Poseidon"),
            complete_add_selector: e("CompleteAdd"),
            mul_selector: e("VarBaseMul"),
            emul_selector: e("EndoMul"),
            endomul_scalar_selector: e("EndoMulScalar"),
            optional_selectors,
            lookup: lookup_evals,
        },
        ft_eval1: field(ft_eval1),
        proof: OpeningProof {
            lr,
            delta: point("delta"),
            z1: field(&opening["z1"]),
            z2: field(&opening["z2"]),
            sg: point("sg"),
        },
        prev_challenges: prev_challenges::<C>(name),
    }
}

/// The public inputs of the tool's circuit `name`: the first rows of column
/// 0 of its witness.
fn public_inputs<F: PrimeField>(name: &str, count: usize) -> Vec<F> {
    circuit(name).1[0][..count].to_vec()
}

#[test]
fn kimchis_proofs_are_ours() {
    kimchis_proofs_are_ours_on::<Vesta>();
    kimchis_proofs_are_ours_on::<Pallas>();
}

fn kimchis_proofs_are_ours_on<C: Kimchi>() {
    // Parsing is faithful: Kimchi's proof read back is the proof we make.
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, witness) = index::<C>(name, max_poly_size, public);
        let base = C::base();
        let scalar = C::scalar();
        let mut rng = Arkworks(SplitMix64(7));
        let ours = prove(
            &index,
            witness,
            &runtime_tables::<C::Scalar>(name),
            &prev_challenges::<C>(name),
            &gates,
            (&base, &scalar),
            &mut rng,
        )
        .unwrap();
        assert_eq!(kimchis_proof::<C>(name), ours, "{name}");
    }
}

#[test]
fn the_verifier_oracles_are_kimchis() {
    the_verifier_oracles_are_kimchis_on::<Vesta>();
    the_verifier_oracles_are_kimchis_on::<Pallas>();
}

fn the_verifier_oracles_are_kimchis_on<C: Kimchi>() {
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, _) = index::<C>(name, max_poly_size, public);
        let base = C::base();
        let scalar = C::scalar();
        let proof = kimchis_proof::<C>(name);
        let o = oracles(
            &index.verifier_index,
            &index.srs,
            &proof,
            &public_inputs::<C::Scalar>(name, public),
            &gates,
            (&base, &scalar),
        )
        .unwrap();
        let expected = lines("oracle", name);
        for (label, value) in [
            ("beta", o.beta),
            ("gamma", o.gamma),
            ("alpha", o.alpha),
            ("zeta", o.zeta),
            ("v", o.v),
            ("u", o.u),
            ("ft_eval0", o.ft_eval0),
            ("combined_inner_product", o.combined_inner_product),
            ("digest", o.digest),
        ] {
            assert_eq!(expected[label], hex(value), "{name} {label}");
        }
        assert_eq!(
            lines("comm", name)["public"],
            comm(&o.public_comm),
            "{name} public"
        );
    }
}

#[test]
fn kimchis_proofs_verify() {
    kimchis_proofs_verify_on::<Vesta>();
    kimchis_proofs_verify_on::<Pallas>();
}

fn kimchis_proofs_verify_on<C: Kimchi>() {
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, _) = index::<C>(name, max_poly_size, public);
        let base = C::base();
        let scalar = C::scalar();
        let mut rng = Arkworks(SplitMix64(1));
        let result = verify(
            &index.verifier_index,
            &index.srs,
            &kimchis_proof::<C>(name),
            &public_inputs::<C::Scalar>(name, public),
            &gates,
            (&base, &scalar),
            &mut rng,
        );
        assert_eq!(result, Ok(()), "{name}");
    }
}

#[test]
fn tampered_proofs_are_refused() {
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let (name, max_poly_size, public) = named("generic");
    let (index, _) = index(name, max_poly_size, public);
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let inputs = public_inputs(name, public);
    let check = |proof: &ProverProof<Vesta>, inputs: &[Fp]| {
        let mut rng = Arkworks(SplitMix64(1));
        verify(
            &index.verifier_index,
            &index.srs,
            proof,
            inputs,
            &gates,
            (&base, &scalar),
            &mut rng,
        )
    };
    let good = kimchis_proof(name);
    assert_eq!(check(&good, &inputs), Ok(()));
    let mut bad = good.clone();
    bad.proof.z1 = bad.proof.z1 + Fp::ONE;
    assert_eq!(check(&bad, &inputs), Err(VerifyError::OpeningProof), "z1");
    let mut bad = good.clone();
    bad.ft_eval1 = bad.ft_eval1 + Fp::ONE;
    assert_eq!(
        check(&bad, &inputs),
        Err(VerifyError::OpeningProof),
        "ft_eval1"
    );
    let mut bad = good.clone();
    bad.evals.w[3].zeta[0] = bad.evals.w[3].zeta[0] + Fp::ONE;
    assert_eq!(
        check(&bad, &inputs),
        Err(VerifyError::OpeningProof),
        "an evaluation"
    );
    let mut bad = good.clone();
    bad.w_comm[0].chunks[0] = bad.w_comm[0].chunks[0] + Projective::generator();
    assert_eq!(
        check(&bad, &inputs),
        Err(VerifyError::OpeningProof),
        "a commitment"
    );
    let mut bad = good.clone();
    bad.proof.sg = bad.proof.sg + Projective::generator();
    assert_eq!(check(&bad, &inputs), Err(VerifyError::OpeningProof), "sg");
    // Another public input is another statement.
    let mut other = inputs.clone();
    other[0] = other[0] + Fp::ONE;
    assert_eq!(
        check(&good, &other),
        Err(VerifyError::OpeningProof),
        "public input"
    );
    assert_eq!(
        check(&good, &inputs[1..]),
        Err(VerifyError::IncorrectPublicInputLength)
    );
}

/// Our proofs, in the format of the vectors, for Kimchi's verifier (the
/// `prover` tool of `tools/kimchi-vectors`).
const OURS: &str = include_str!("vectors/arrabbiata_proofs.txt");

/// The lines of our proof of `name`.
fn dump<C: Kimchi>(name: &str, proof: &ProverProof<C>) -> Vec<String> {
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
    let mut put = |label: String, p: &PointEvaluations<C::Scalar>| {
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
    for (label, value) in lookup_lines(e) {
        if value != "none" {
            out.push(format!("eval {name} {label} {value}"));
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
    out
}

/// Our proofs of the tool's circuits, from another stream than Kimchi's
/// (SplitMix64(1000)), so that they are ours and not copies of Kimchi's.
fn our_proofs<C: Kimchi>() -> Vec<(&'static str, ProverProof<C>)> {
    let gates = C::gates();
    C::CIRCUITS
        .iter()
        .map(|(name, max_poly_size, public)| {
            let (index, witness) = index::<C>(name, *max_poly_size, *public);
            let base = C::base();
            let scalar = C::scalar();
            let mut rng = Arkworks(SplitMix64(1000));
            let proof = prove(
                &index,
                witness,
                &runtime_tables(name),
                &prev_challenges::<C>(name),
                &gates,
                (&base, &scalar),
                &mut rng,
            )
            .unwrap();
            (*name, proof)
        })
        .collect()
}

#[test]
fn our_proofs_verify_and_are_the_committed_ones() {
    let mut lines = our_proofs_verify::<Vesta>();
    lines.extend(our_proofs_verify::<Pallas>());
    let ours = lines.join("\n") + "\n";
    if std::env::var_os("REGENERATE").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/vectors/arrabbiata_proofs.txt"
        );
        std::fs::write(path, &ours).unwrap();
    }
    assert_eq!(
        OURS, ours,
        "rerun with REGENERATE=1 after changing the prover"
    );
}

/// Our proofs over `C` verify with our verifier and are not Kimchi's (they
/// come from another stream); their lines in the dump.
fn our_proofs_verify<C: Kimchi>() -> Vec<String> {
    let gates = C::gates();
    let mut lines = Vec::new();
    for (name, proof) in our_proofs::<C>() {
        let (max_poly_size, public) = C::CIRCUITS
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, m, p)| (*m, *p))
            .unwrap();
        let (index, _) = index::<C>(name, max_poly_size, public);
        let base = C::base();
        let scalar = C::scalar();
        let mut rng = Arkworks(SplitMix64(2));
        let inputs = public_inputs::<C::Scalar>(name, public);
        let result = verify(
            &index.verifier_index,
            &index.srs,
            &proof,
            &inputs,
            &gates,
            (&base, &scalar),
            &mut rng,
        );
        assert_eq!(result, Ok(()), "{name}");
        assert_ne!(
            proof,
            kimchis_proof::<C>(name),
            "{name}: another stream, another proof"
        );
        // Written as Kimchi writes proofs, and read back to itself; so is
        // Kimchi's proof of the same circuit.
        assert_eq!(codec_laws::round_trip(&proof, ()), Ok(()), "{name}");
        assert_eq!(
            codec_laws::round_trip(&kimchis_proof::<C>(name), ()),
            Ok(()),
            "{name}"
        );
        lines.extend(dump(name, &proof));
    }
    lines
}

/// What Kimchi's verifier said of our proofs (the `prover` tool run on
/// `arrabbiata_proofs.txt`): the Blake2b512 digest of the file it read,
/// then, per circuit, its verdict on our proof, on our proof with `z1 + 1`,
/// and on its own.
static KIMCHI_ON_OURS: LazyLock<String> =
    LazyLock::new(|| kimchi_runs::read("kimchi_verifies_arrabbiata.txt"));

#[test]
fn kimchi_verifies_our_proofs() {
    // The verdicts are about the proofs we make now: the digest of the
    // file Kimchi read is the digest of our current dump.
    let digest: String = Blake2b512::digest(OURS.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let first = KIMCHI_ON_OURS.lines().next().unwrap();
    assert_eq!(
        first,
        format!("file {digest}"),
        "rerun the tool on the new dump"
    );
    for (name, _, _) in CIRCUITS.iter().chain(&PALLAS_CIRCUITS) {
        for (who, verdict) in [
            ("arrabbiata", "Ok(\"accepted\")"),
            ("arrabbiata-tampered", "refused"),
            ("kimchi", "Ok(\"accepted\")"),
        ] {
            let line = format!("verify {name} {who} {verdict}");
            assert!(KIMCHI_ON_OURS.lines().any(|l| l == line), "{line}");
        }
    }
}

/// The lookup evaluation lines of `e`, labelled as the vectors label them.
fn lookup_lines<F: PrimeField>(e: &ProofEvaluations<F>) -> Vec<(String, String)> {
    let none = || String::from("none");
    let shown = |p: Option<&PointEvaluations<F>>| p.map_or_else(none, evaluation);
    let l = e.lookup.as_ref();
    let mut out = vec![
        (
            "lookup_aggregation".to_string(),
            shown(l.map(|l| &l.aggregation)),
        ),
        ("lookup_table".to_string(), shown(l.map(|l| &l.table))),
    ];
    for i in 0..5 {
        out.push((
            format!("lookup_sorted{i}"),
            shown(l.and_then(|l| l.sorted.get(i))),
        ));
    }
    out.push((
        "runtime_lookup_table".to_string(),
        shown(l.and_then(|l| l.runtime_table.as_ref())),
    ));
    out.push((
        "runtime_lookup_table_selector".to_string(),
        shown(l.and_then(|l| l.runtime_selector.as_ref())),
    ));
    for (label, pattern) in [
        ("xor_lookup_selector", LookupPattern::Xor),
        ("lookup_gate_lookup_selector", LookupPattern::Lookup),
        ("range_check_lookup_selector", LookupPattern::RangeCheck),
        (
            "foreign_field_mul_lookup_selector",
            LookupPattern::ForeignFieldMul,
        ),
    ] {
        let p = l.and_then(|l| {
            l.selectors
                .iter()
                .find(|(q, _)| *q == pattern)
                .map(|(_, p)| p)
        });
        out.push((label.to_string(), shown(p)));
    }
    out
}

#[test]
fn tampered_lookup_proofs_are_refused() {
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let (name, max_poly_size, public) = named("runtime");
    assert_eq!(name, "runtime");
    let (index, _) = index(name, max_poly_size, public);
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let check = |proof: &ProverProof<Vesta>| {
        let mut rng = Arkworks(SplitMix64(1));
        let sponges = (&base, &scalar);
        verify(
            &index.verifier_index,
            &index.srs,
            proof,
            &[],
            &gates,
            sponges,
            &mut rng,
        )
    };
    let good = kimchis_proof(name);
    assert_eq!(check(&good), Ok(()));
    let shift = Projective::<Vesta>::generator();
    let mut bad = good.clone();
    bad.lookup.as_mut().unwrap().sorted[1].chunks[0] =
        bad.lookup.as_ref().unwrap().sorted[1].chunks[0] + shift;
    assert_eq!(
        check(&bad),
        Err(VerifyError::OpeningProof),
        "a sorted column"
    );
    let mut bad = good.clone();
    let l = bad.evals.lookup.as_mut().unwrap();
    l.aggregation.zeta_omega[0] = l.aggregation.zeta_omega[0] + Fp::ONE;
    assert_eq!(
        check(&bad),
        Err(VerifyError::OpeningProof),
        "the aggregation"
    );
    let mut bad = good.clone();
    let l = bad.evals.lookup.as_mut().unwrap();
    l.table.zeta[0] = l.table.zeta[0] + Fp::ONE;
    assert_eq!(
        check(&bad),
        Err(VerifyError::OpeningProof),
        "the joint table"
    );
    let mut bad = good.clone();
    let r = bad.lookup.as_mut().unwrap().runtime.as_mut().unwrap();
    r.chunks[0] = r.chunks[0] + shift;
    assert_eq!(
        check(&bad),
        Err(VerifyError::OpeningProof),
        "the runtime table"
    );
    let mut bad = good.clone();
    bad.lookup = None;
    assert_eq!(
        check(&bad),
        Err(VerifyError::LookupMismatch),
        "no lookup part"
    );
}

/// The statement and the assignment of the tool's circuit `name`.
fn statement(name: &str, public: usize) -> (Statement<Vesta>, Assignment<Fp>) {
    let (_, columns) = circuit(name);
    let statement = Statement {
        public: columns[0][..public].to_vec(),
        prev_challenges: prev_challenges(name),
    };
    let assignment = Assignment {
        columns,
        runtime_tables: runtime_tables(name),
    };
    (statement, assignment)
}

#[test]
fn kimchis_proofs_reduce_to_decided_accumulators() {
    kimchis_proofs_reduce_to_decided_accumulators_on::<Vesta>();
    kimchis_proofs_reduce_to_decided_accumulators_on::<Pallas>();
}

fn kimchis_proofs_reduce_to_decided_accumulators_on<C: Kimchi>() {
    let gates = C::gates();
    for &(name, max_poly_size, public) in C::CIRCUITS {
        let (index, _) = index::<C>(name, max_poly_size, public);
        let base = C::base();
        let scalar = C::scalar();
        let proof = kimchis_proof::<C>(name);
        let accumulator = succinct(
            &index.verifier_index,
            &index.srs,
            &proof,
            &public_inputs::<C::Scalar>(name, public),
            &gates,
            (&base, &scalar),
        )
        .unwrap();
        // The accumulator is (u, sg): sg is the proof's, u has log2(|G|)
        // challenges, and Halo's decider (the IPA on Kimchi's key) accepts.
        assert_eq!(accumulator.comm, proof.proof.sg, "{name}");
        let ipa = ipa_of(&index.srs, scalar.clone());
        assert_eq!(accumulator.chals.len(), ipa.rounds(), "{name}");
        assert!(ipa.decide(&accumulator), "{name}");
        let mut wrong = accumulator.clone();
        wrong.chals[0] = wrong.chals[0] + C::Scalar::ONE;
        assert!(!ipa.decide(&wrong), "{name}");
    }
}

#[test]
fn the_reduction_is_complete_and_its_composites_verify() {
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    for (name, max_poly_size, public) in [
        named("generic"),
        named("recursion"),
        named("lookup"),
        named("runtime"),
    ] {
        let (index, _) = index(name, max_poly_size, public);
        let base = KimchiSponge::new(vesta_base());
        let scalar = KimchiSponge::new(pallas_base());
        let ipa = ipa_of(&index.srs, scalar.clone());
        let relation = Satisfied::new(&index, &gates, (&base, &scalar), &ipa);
        let (statement, assignment) = statement(name, public);
        assert_eq!(relation.check(&statement, &assignment), Ok(()), "{name}");
        let transcript = || KimchiSponge::new(pallas_base());
        let reduction = KimchiSuccinct::new(relation, Arkworks(SplitMix64(5)));
        // The verifier's accumulator is the prover's, and is decided.
        let out = laws::complete(&reduction, transcript, &statement, &assignment).unwrap();
        assert_eq!(
            Accumulated::new(&ipa).check(&out.instance, &()),
            Ok(()),
            "{name}"
        );
        // KimchiSuccinct ; Decide, Kimchi's whole verifier.
        let whole = accumulation::verify(reduction);
        let proved = whole.prove(transcript(), &statement, &assignment).unwrap();
        let mut tampered = proved.proof.clone();
        tampered.0.proof.z1 = tampered.0.proof.z1 + Fp::ONE;
        let accepted = whole.verify(transcript(), &statement, proved.proof);
        assert!(accepted.is_ok(), "{name}");
        let refused = whole.verify(transcript(), &statement, tampered);
        assert!(refused.is_err(), "{name}");
    }
}

#[test]
fn two_proofs_fold_into_one_decided_accumulator() {
    // Each(KimchiSuccinct) ; Fold, then one Decide for both proofs.
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let (name, max_poly_size, public) = named("generic");
    let (index, _) = index(name, max_poly_size, public);
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let ipa = ipa_of(&index.srs, scalar.clone());
    let relation = Satisfied::new(&index, &gates, (&base, &scalar), &ipa);
    let (statement, assignment) = statement(name, public);
    let statements = vec![statement.clone(), statement];
    let assignments = vec![assignment.clone(), assignment];
    let fold = accumulate_proofs(KimchiSuccinct::new(relation, Arkworks(SplitMix64(9))));
    let transcript = || KimchiSponge::new(pallas_base());
    let out = laws::complete(&fold, transcript, &statements, &assignments).unwrap();
    assert!(ipa.decide(&out.instance));
    // The two proofs differ (the prover's stream moves on), and both
    // accumulators are folded into the one decided.
    let (proofs, _) = &out.proof;
    assert_ne!(proofs[0], proofs[1]);
}

#[test]
fn a_false_previous_accumulator_is_caught() {
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let (name, max_poly_size, public) = named("recursion");
    assert_eq!(name, "recursion");
    let (index, _) = index(name, max_poly_size, public);
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let ipa = ipa_of(&index.srs, scalar.clone());
    let relation = Satisfied::new(&index, &gates, (&base, &scalar), &ipa);
    let (mut statement, assignment) = statement(name, public);
    statement.prev_challenges[0].comm =
        statement.prev_challenges[0].comm + Projective::<Vesta>::generator();
    // Outside the relation: its previous accumulator is not decided.
    assert_eq!(
        relation.check(&statement, &assignment),
        Err(accumulation::Error::PreviousAccumulator(0))
    );
    // And a proof carrying it does not reduce: the lift of sg at zeta
    // fails inside the opening.
    let reduction = KimchiSuccinct::new(relation, Arkworks(SplitMix64(5)));
    let transcript = || KimchiSponge::new(pallas_base());
    let result = reduction.prove(transcript(), &statement, &assignment);
    assert_eq!(
        result.err(),
        Some(accumulation::Error::Verify(VerifyError::OpeningProof))
    );
}

#[test]
fn merged_and_separate_folding_decide_the_same_statements() {
    // Merged: the recursion circuit's proof opens its previous accumulator
    // in its own IPA. Separate: the same circuit without it (the generic
    // circuit), then Halo's fold of the two accumulators.
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    let transcript = || KimchiSponge::new(pallas_base());
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());

    let (merged_name, max_poly_size, public) = named("recursion");
    let (merged_index, _) = index(merged_name, max_poly_size, public);
    let merged_ipa = ipa_of(&merged_index.srs, scalar.clone());
    let merged_relation = Satisfied::new(&merged_index, &gates, (&base, &scalar), &merged_ipa);
    let (merged_statement, merged_assignment) = statement(merged_name, public);
    let previous = merged_statement.prev_challenges.clone();
    assert!(!previous.is_empty());

    let (separate_name, max_poly_size, public) = named("generic");
    let (separate_index, _) = index(separate_name, max_poly_size, public);
    let separate_ipa = ipa_of(&separate_index.srs, scalar.clone());
    let separate_relation =
        Satisfied::new(&separate_index, &gates, (&base, &scalar), &separate_ipa);
    let (separate_statement, separate_assignment) = statement(separate_name, public);
    assert!(separate_statement.prev_challenges.is_empty());

    // A valid previous accumulator: both accept.
    let merged = KimchiSuccinct::new(merged_relation, Arkworks(SplitMix64(11)));
    let out = merged
        .prove(transcript(), &merged_statement, &merged_assignment)
        .unwrap();
    assert!(merged_ipa.decide(&out.instance));
    let folded = separate(KimchiSuccinct::new(
        separate_relation,
        Arkworks(SplitMix64(11)),
    ));
    let witnesses = vec![(); previous.len()];
    let instance = (separate_statement.clone(), previous.clone());
    let assignment = (separate_assignment.clone(), witnesses.clone());
    let out = laws::complete(&folded, transcript, &instance, &assignment).unwrap();
    assert!(separate_ipa.decide(&out.instance));

    // A false one: merged refuses to prove; separate's prover still folds
    // (it computes the new accumulator from its own coefficients), and its
    // verifier, which opens the claimed commitments, refuses.
    let mut false_previous = previous;
    false_previous[0].comm = false_previous[0].comm + Projective::<Vesta>::generator();
    let mut false_statement = merged_statement;
    false_statement.prev_challenges = false_previous.clone();
    assert!(
        merged
            .prove(transcript(), &false_statement, &merged_assignment)
            .is_err()
    );
    let instance = (separate_statement, false_previous);
    let assignment = (separate_assignment, witnesses);
    let out = folded.prove(transcript(), &instance, &assignment).unwrap();
    let verified = folded.verify(transcript(), &instance, out.proof);
    assert!(verified.is_err());
}
