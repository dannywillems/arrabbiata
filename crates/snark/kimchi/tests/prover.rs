//! Kimchi's prover against Kimchi, round by round: Kimchi's generic test
//! circuit with five public inputs, proved by the `prover` tool of the
//! Kimchi figures notes with SplitMix64(7), unchunked and in chunks of 16
//! rows (`vectors/kimchi_prover.txt`). The tests rebuild the circuit and
//! the witness from the vectors, run our rounds on the same stream, and
//! compare each value the round produces.

use std::collections::BTreeMap;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    gates::KimchiGates,
    prover::{
        PointEvaluations, ProverIndex, evaluation_round, permutation_round, quotient_round,
        witness_round,
    },
    rng::{Arkworks, Sampler, SplitMix64},
    srs::{PolyComm, Srs},
};
use arrabbiata_pasta::{Fp, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// Kimchi's output.
const VECTORS: &str = include_str!("vectors/kimchi_prover.txt");

/// The circuits of the tool, and their chunk sizes.
const CIRCUITS: [(&str, Option<usize>); 2] = [("generic", None), ("chunked", Some(16))];

fn hex<F: PrimeField>(x: F) -> String {
    (0..F::LIMBS)
        .rev()
        .map(|i| format!("{:016x}", x.limb(i)))
        .collect()
}

fn element(hex: &str) -> Fp {
    let limbs: Vec<u64> = (0..4)
        .rev()
        .map(|i| u64::from_str_radix(&hex[16 * i..16 * (i + 1)], 16).unwrap())
        .collect();
    Fp::from_limbs(&limbs).unwrap()
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

fn index(name: &str, max_poly_size: Option<usize>) -> (ProverIndex<Vesta>, [Vec<Fp>; 15]) {
    let (gates, witness) = circuit(name);
    let cs = ConstraintSystem::create(gates)
        .public(5)
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
    for (name, max_poly_size) in CIRCUITS {
        let (index, _) = index(name, max_poly_size);
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
    for (name, max_poly_size) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size);
        let sponge = KimchiSponge::new(vesta_base());
        let mut rng = Arkworks(SplitMix64(7));
        let round = witness_round(&index, witness, &sponge, &mut rng).unwrap();
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
    for (name, max_poly_size) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size);
        let sponge = KimchiSponge::new(vesta_base());
        let mut rng = Arkworks(SplitMix64(7));
        let first = witness_round(&index, witness, &sponge, &mut rng).unwrap();
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
    for (name, max_poly_size) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size);
        let sponge = KimchiSponge::new(vesta_base());
        let mut rng = Arkworks(SplitMix64(7));
        let first = witness_round(&index, witness, &sponge, &mut rng).unwrap();
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
    for (name, max_poly_size) in CIRCUITS {
        let (index, witness) = index(name, max_poly_size);
        let sponge = KimchiSponge::new(vesta_base());
        let scalar_sponge = KimchiSponge::new(pallas_base());
        let mut rng = Arkworks(SplitMix64(7));
        let first = witness_round(&index, witness, &sponge, &mut rng).unwrap();
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
