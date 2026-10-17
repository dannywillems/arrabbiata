//! Kimchi's prover against Kimchi, round by round: Kimchi's generic test
//! circuit with five public inputs, proved by the `prover` tool of the
//! Kimchi figures notes with SplitMix64(7), unchunked and in chunks of 16
//! rows (`vectors/kimchi_prover.txt`). The tests rebuild the circuit and
//! the witness from the vectors, run our rounds on the same stream, and
//! compare each value the round produces.

use std::collections::BTreeMap;

use arrabbiata_curve::{Affine, Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_ipa::RecursionChallenge;
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    gates::KimchiGates,
    prover::{
        PointEvaluations, ProverIndex, evaluation_round, permutation_round, prove, quotient_round,
        witness_round,
    },
    rng::{Arkworks, Sampler, SplitMix64},
    srs::{PolyComm, Srs},
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// Kimchi's output.
const VECTORS: &str = include_str!("vectors/kimchi_prover.txt");

/// The circuits of the tool, their chunk sizes and public inputs: Kimchi's
/// generic test circuit unchunked and in chunks of 16 rows, one block of
/// each gate without lookups (Poseidon, CompleteAdd, VarBaseMul, EndoMul,
/// EndoMulScalar), and the generic circuit carrying one previous
/// accumulator (as Kimchi's `test_recursion`).
const CIRCUITS: [(&str, Option<usize>, usize); 4] = [
    ("generic", None, 5),
    ("chunked", Some(16), 5),
    ("gadgets", None, 0),
    ("recursion", None, 5),
];

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

fn element(hex: &str) -> Fp {
    field(hex)
}

/// The previous accumulators of the tool's circuit `name`.
fn prev_challenges(name: &str) -> Vec<RecursionChallenge<Vesta>> {
    let prev = lines("prev", name);
    if prev.is_empty() {
        return vec![];
    }
    let chals = (0..prev.len() - 1)
        .map(|i| element(&prev[&format!("chal{i}")]))
        .collect();
    let words: Vec<&str> = prev["comm"].split_whitespace().collect();
    assert_eq!(words[0], "1");
    let comm = Affine::<Vesta>::new(field::<Fq>(words[1]), field::<Fq>(words[2]))
        .unwrap()
        .to_projective();
    vec![RecursionChallenge { chals, comm }]
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
        other => panic!("gate {other} not in the vectors"),
    }
}

/// The circuit and the (unpadded) witness of the tool's circuit `name`.
fn circuit(name: &str) -> (Vec<CircuitGate<Fp>>, [Vec<Fp>; 15]) {
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
        let coeffs = words[19..19 + count].iter().map(|h| element(h)).collect();
        gates.push(CircuitGate::new(gate_type(words[3]), wires, coeffs));
    }
    let mut witness: [Vec<Fp>; 15] = core::array::from_fn(|_| vec![Fp::ZERO; gates.len()]);
    for line in VECTORS.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words[0] == "witness" && words[1] == name {
            let col: usize = words[2].parse().unwrap();
            let row: usize = words[3].parse().unwrap();
            witness[col][row] = element(words[4]);
        }
    }
    (gates, witness)
}

fn index(
    name: &str,
    max_poly_size: Option<usize>,
    public: usize,
) -> (ProverIndex<Vesta>, [Vec<Fp>; 15]) {
    let (gates, witness) = circuit(name);
    let cs = ConstraintSystem::create(gates)
        .public(public)
        .prev_challenges(prev_challenges(name).len())
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
    for (name, max_poly_size, public) in CIRCUITS {
        let (index, _) = index(name, max_poly_size, public);
        let sponge = KimchiSponge::new(vesta_base());
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
    for (name, max_poly_size, public) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size, public);
        let sponge = KimchiSponge::new(vesta_base());
        let mut rng = Arkworks(SplitMix64(7));
        let round =
            witness_round(&index, witness, &prev_challenges(name), &sponge, &mut rng).unwrap();
        let comms = lines("comm", name);
        assert_eq!(comms["public"], comm(&round.public_comm), "{name} public");
        for (i, c) in round.w_comm.iter().enumerate() {
            assert_eq!(comms[&format!("w{i}")], comm(c), "{name} w{i}");
        }
        let oracles = lines("oracle", name);
        assert_eq!(oracles["beta"], hex(round.beta), "{name} beta");
        assert_eq!(oracles["gamma"], hex(round.gamma), "{name} gamma");
    }
}

#[test]
fn the_permutation_round_is_kimchis() {
    for (name, max_poly_size, public) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size, public);
        let sponge = KimchiSponge::new(vesta_base());
        let mut rng = Arkworks(SplitMix64(7));
        let first =
            witness_round(&index, witness, &prev_challenges(name), &sponge, &mut rng).unwrap();
        let round = permutation_round(&index, first, &mut rng).unwrap();
        assert_eq!(lines("comm", name)["z"], comm(&round.z_comm), "{name} z");
        let oracles = lines("oracle", name);
        assert_eq!(oracles["alpha_chal"], hex(round.alpha_chal.0), "{name}");
        assert_eq!(oracles["alpha"], hex(round.alpha), "{name} alpha");
    }
}

#[test]
fn the_quotient_round_is_kimchis() {
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    for (name, max_poly_size, public) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size, public);
        let sponge = KimchiSponge::new(vesta_base());
        let mut rng = Arkworks(SplitMix64(7));
        let first =
            witness_round(&index, witness, &prev_challenges(name), &sponge, &mut rng).unwrap();
        let second = permutation_round(&index, first, &mut rng).unwrap();
        let round = quotient_round(&index, second, &gates, &mut rng).unwrap();
        assert_eq!(lines("comm", name)["t"], comm(&round.t_comm), "{name} t");
        let oracles = lines("oracle", name);
        assert_eq!(oracles["zeta_chal"], hex(round.zeta_chal.0), "{name}");
        assert_eq!(oracles["zeta"], hex(round.zeta), "{name} zeta");
    }
}

fn values(v: &[Fp]) -> String {
    let hexes: Vec<String> = v.iter().map(|x| hex(*x)).collect();
    format!("{} {}", v.len(), hexes.join(" "))
}

fn evaluation(e: &PointEvaluations<Fp>) -> String {
    format!("{} {}", values(&e.zeta), values(&e.zeta_omega))
}

#[test]
fn the_evaluation_round_is_kimchis() {
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    for (name, max_poly_size, public) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size, public);
        let sponge = KimchiSponge::new(vesta_base());
        let scalar_sponge = KimchiSponge::new(pallas_base());
        let mut rng = Arkworks(SplitMix64(7));
        let first =
            witness_round(&index, witness, &prev_challenges(name), &sponge, &mut rng).unwrap();
        let second = permutation_round(&index, first, &mut rng).unwrap();
        let third = quotient_round(&index, second, &gates, &mut rng).unwrap();
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
        assert_eq!(ours.len(), expected.len(), "{name}");
        for (label, value) in &ours {
            assert_eq!(&expected[label], value, "{name} {label}");
        }
        assert!(e.optional_selectors.is_empty());
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
    let gates = KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap();
    for (name, max_poly_size, public) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size, public);
        let base = KimchiSponge::new(vesta_base());
        let scalar = KimchiSponge::new(pallas_base());
        let mut rng = Arkworks(SplitMix64(7));
        let proof = prove(
            &index,
            witness,
            &prev_challenges(name),
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
