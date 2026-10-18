//! A whole Kimchi proof, reproducibly: Kimchi's generic test circuit
//! (`generic::testing::create_circuit` with 5 public inputs, then
//! `fill_in_witness`), proved over Vesta with a SplitMix64 RNG, unchunked
//! and in chunks of 16 rows, and over Pallas too. Prints the circuit, the witness, the
//! verifier-index digest, every part of the proof, the public commitment
//! and every oracle; also the first field elements arkworks samples from
//! SplitMix64(42), to pin the sampling.
use std::sync::Arc;

use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup};
use ark_ff::{BigInteger, BitIteratorLE, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use groupmap::GroupMap;
use num_bigint::BigUint;
use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        gate::{CircuitGate, Connect, GateType},
        polynomials::{
            endomul_scalar, endosclmul,
            generic::testing::{create_circuit, fill_in_witness},
            foreign_field_common::BigUintForeignFieldHelpers,
            foreign_field_mul,
            generic::GenericGateSpec,
            poseidon, range_check,
            rot::{self, RotMode},
            varbasemul, xor,
        },
        lookup::{
            runtime_tables::{RuntimeTable, RuntimeTableCfg},
            tables::LookupTable,
        },
        wires::{Wire, COLUMNS},
    },
    curve::KimchiCurve,
    proof::{
        LookupCommitments, PointEvaluations, ProofEvaluations, ProverCommitments, ProverProof,
        RecursionChallenge,
    },
    verifier::verify,
    prover_index::ProverIndex,
};
use kimchi::{
    circuits::polynomials::foreign_field_add::witness::{self as ffadd_witness, FFOps},
    plonk_sponge::FrSponge,
};
use mina_curves::pasta::{Fp, Fq, Fq as PallasScalar, Pallas, PallasParameters, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi as SC,
    poseidon::ArithmeticSpongeParams,
    sponge::{DefaultFqSponge, DefaultFrSponge},
    FqSponge,
};
use poly_commitment::{
    commitment::{b_poly_coefficients, CommitmentCurve, PolyComm},
    ipa::{endos, OpeningProof, SRS},
    SRS as _,
};
use blake2::Digest;
use rand::{CryptoRng, RngCore};

type BaseSponge = DefaultFqSponge<VestaParameters, SC, 55>;
type ScalarSponge = DefaultFrSponge<Fp, SC, 55>;
type PallasBaseSponge = DefaultFqSponge<PallasParameters, SC, 55>;
type PallasScalarSponge = DefaultFrSponge<Fq, SC, 55>;

/// SplitMix64 (Steele, Lea, Flood 2014), the stream both sides sample from.
struct SplitMix64(u64);

impl RngCore for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for SplitMix64 {}

fn hex<F: PrimeField>(x: F) -> String {
    x.into_bigint().to_bytes_be().iter().map(|b| format!("{b:02x}")).collect()
}

fn point<G: AffineRepr>(p: &G) -> String
where
    G::BaseField: PrimeField,
{
    match p.xy() {
        Some((x, y)) => format!("{} {}", hex(x), hex(y)),
        None => String::from("inf"),
    }
}

fn comm<G: AffineRepr>(c: &PolyComm<G>) -> String
where
    G::BaseField: PrimeField,
{
    let chunks: Vec<String> = c.chunks.iter().map(point).collect();
    format!("{} {}", c.chunks.len(), chunks.join(" "))
}

fn values<F: PrimeField>(v: &[F]) -> String {
    let hexes: Vec<String> = v.iter().map(|x| hex(*x)).collect();
    format!("{} {}", v.len(), hexes.join(" "))
}

fn eval<F: PrimeField>(name: &str, label: &str, e: Option<&PointEvaluations<Vec<F>>>) {
    match e {
        None => println!("eval {name} {label} none"),
        Some(e) => println!("eval {name} {label} {} {}", values(&e.zeta), values(&e.zeta_omega)),
    }
}

fn typ(t: GateType) -> String {
    format!("{t:?}")
}

/// A circuit to prove: gates, witness, public inputs, tables.
struct Circuit<F: PrimeField = Fp> {
    gates: Vec<CircuitGate<F>>,
    witness: [Vec<F>; COLUMNS],
    public: Vec<F>,
    tables: Vec<LookupTable<F>>,
    runtime_cfgs: Option<Vec<RuntimeTableCfg<F>>>,
    runtime: Vec<RuntimeTable<F>>,
}

impl<F: PrimeField> From<(Vec<CircuitGate<F>>, [Vec<F>; COLUMNS], Vec<F>)> for Circuit<F> {
    fn from((gates, witness, public): (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS], Vec<F>)) -> Self {
        Self { gates, witness, public, tables: vec![], runtime_cfgs: None, runtime: vec![] }
    }
}

/// A 64-bit Xor (`extend_xor_gadget`, `create_xor_witness`).
fn xor_circuit() -> Circuit {
    let mut rng = SplitMix64(31);
    let mut gates = vec![];
    CircuitGate::<Fp>::extend_xor_gadget(&mut gates, 64);
    let witness = xor::create_xor_witness(Fp::from(rng.next_u64()), Fp::from(rng.next_u64()), 64);
    (gates, witness, vec![]).into()
}

/// A multi-range check of three 88-bit values (`create_multi_range_check`,
/// `range_check::witness::create_multi`).
fn range_check_circuit() -> Circuit {
    let mut rng = SplitMix64(37);
    let (_, gates) = CircuitGate::<Fp>::create_multi_range_check(0);
    let mut value = || {
        let low = Fp::from(rng.next_u64());
        let high = Fp::from(rng.next_u64() >> 40);
        low + high * Fp::from(2u64).pow([64])
    };
    let (v0, v1, v2) = (value(), value(), value());
    let witness = range_check::witness::create_multi(v0, v1, v2);
    (gates, witness, vec![]).into()
}

/// Lookup gates over two fixed tables of ids 0 and 1, as Kimchi's
/// `setup_lookup_proof` (an index column and a value column each).
fn lookup_circuit() -> Circuit {
    let mut rng = SplitMix64(41);
    // 5 + 7 rows would hit Kimchi's off-by-one: the domain is sized for
    // table + 1 + zk_rows = 16 rows, the index needs table < n - zk_rows - 1.
    let sizes = [5usize, 6];
    let values: Vec<Vec<Fp>> = sizes
        .iter()
        .enumerate()
        .map(|(t, size)| (0..*size).map(|i| if t == 0 && i == 0 { Fp::zero() } else { Fp::from(rng.next_u64()) }).collect())
        .collect();
    let tables: Vec<LookupTable<Fp>> = values
        .iter()
        .enumerate()
        .map(|(id, v)| LookupTable { id: id as i32, data: vec![(0..v.len() as u64).map(Fp::from).collect(), v.clone()] })
        .collect();
    let rows = 6;
    let gates = (0..rows).map(|i| CircuitGate::new(GateType::Lookup, Wire::for_row(i), vec![])).collect();
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); rows]);
    for row in 0..rows {
        let table = (rng.next_u64() % 2) as usize;
        witness[0][row] = Fp::from(table as u64);
        for i in 0..3 {
            let index = (rng.next_u64() % sizes[table] as u64) as usize;
            witness[2 * i + 1][row] = Fp::from(index as u64);
            witness[2 * i + 2][row] = values[table][index];
        }
    }
    Circuit { gates, witness, public: vec![], tables, runtime_cfgs: None, runtime: vec![] }
}

/// Lookup gates over two runtime tables, as Kimchi's `test_runtime_table`.
fn runtime_circuit() -> Circuit {
    let mut rng = SplitMix64(43);
    let first_column: Vec<Fp> = [8u64, 9, 8, 7, 1].into_iter().map(Fp::from).collect();
    let data: Vec<Fp> = [0u64, 2, 3, 4, 5].into_iter().map(Fp::from).collect();
    let cfgs: Vec<RuntimeTableCfg<Fp>> = (1..=2).map(|id| RuntimeTableCfg { id, first_column: first_column.clone() }).collect();
    let runtime: Vec<RuntimeTable<Fp>> = cfgs.iter().map(|c| RuntimeTable { id: c.id, data: data.clone() }).collect();
    let rows = 8;
    let gates = (0..rows).map(|i| CircuitGate::new(GateType::Lookup, Wire::for_row(i), vec![])).collect();
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); rows]);
    for row in 0..rows {
        witness[0][row] = Fp::from(1 + rng.next_u64() % 2);
        for i in 0..3 {
            // Index 1, as Kimchi's test, avoids index 0 (the value 8 appears
            // twice in the first column, with different data).
            let idx = 1 + (rng.next_u64() % 4) as usize;
            witness[2 * i + 1][row] = first_column[idx];
            witness[2 * i + 2][row] = data[idx];
        }
    }
    Circuit { gates, witness, public: vec![], tables: vec![], runtime_cfgs: Some(cfgs), runtime }
}

/// A 64-bit word rotated left and right (Kimchi's `create_rot_gadget`: a
/// Generic zero row, then `extend_rot` per rotation, wired to it).
fn rot_circuit() -> Circuit {
    let mut rng = SplitMix64(47);
    let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero()]);
    for side in [RotMode::Left, RotMode::Right] {
        let word = rng.next_u64();
        let offset = (1 + rng.next_u64() % 63) as u32;
        CircuitGate::<Fp>::extend_rot(&mut gates, offset, side, 0);
        rot::extend_rot(&mut witness, word, offset, side);
    }
    (gates, witness, vec![]).into()
}

/// The secp256k1 base field modulus.
fn secp256k1_modulus() -> BigUint {
    BigUint::parse_bytes(b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f", 16).unwrap()
}

/// A foreign-field multiplication modulo secp256k1's base field with every
/// external check, as Kimchi's `foreign_field_mul` `run_test` with
/// `external_gates` (compact multi-range check of the remainder, high
/// bounds, multi-range checks of the quotient, of the inputs and of the
/// result bound, limb checks), inputs from SplitMix64(53).
fn ffmul_circuit() -> Circuit {
    let mut rng = SplitMix64(53);
    let modulus = secp256k1_modulus();
    let mut below = || {
        let words: Vec<u32> = (0..8).map(|_| rng.next_u64() as u32).collect();
        BigUint::new(words) % &modulus
    };
    let (left_input, right_input) = (below(), below());
    let foreign_field_modulus = &modulus;
    let (mut next_row, mut gates) = CircuitGate::<Fp>::create_foreign_field_mul(0, foreign_field_modulus);
    let (mut witness, mut external_checks) =
        foreign_field_mul::witness::create(&left_input, &right_input, foreign_field_modulus);
    CircuitGate::extend_compact_multi_range_check(&mut gates, &mut next_row);
    gates.connect_cell_pair((1, 0), (4, 1));
    gates.connect_cell_pair((1, 1), (2, 0));
    external_checks.extend_witness_compact_multi_range_checks(&mut witness);
    CircuitGate::extend_high_bounds(&mut gates, &mut next_row, foreign_field_modulus);
    gates.connect_cell_pair((6, 0), (1, 1));
    external_checks.extend_witness_high_bounds_computation(&mut witness, foreign_field_modulus);
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    gates.connect_cell_pair((1, 2), (7, 0));
    gates.connect_cell_pair((1, 3), (8, 0));
    gates.connect_cell_pair((1, 4), (9, 0));
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    gates.connect_cell_pair((1, 5), (11, 0));
    gates.connect_cell_pair((0, 6), (12, 0));
    gates.connect_cell_pair((1, 6), (13, 0));
    external_checks.extend_witness_multi_range_checks(&mut witness);
    let left_limbs = left_input.to_field_limbs();
    let right_limbs = right_input.to_field_limbs();
    external_checks.add_high_bound_computation(&left_limbs[2]);
    external_checks.add_high_bound_computation(&right_limbs[2]);
    CircuitGate::extend_high_bounds(&mut gates, &mut next_row, foreign_field_modulus);
    gates.connect_cell_pair((15, 0), (0, 2));
    gates.connect_cell_pair((15, 3), (0, 5));
    external_checks.extend_witness_high_bounds_computation(&mut witness, foreign_field_modulus);
    external_checks.add_multi_range_check(&left_limbs);
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    gates.connect_cell_pair((0, 0), (16, 0));
    gates.connect_cell_pair((0, 1), (17, 0));
    gates.connect_cell_pair((0, 2), (18, 0));
    external_checks.add_multi_range_check(&right_limbs);
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    gates.connect_cell_pair((0, 3), (20, 0));
    gates.connect_cell_pair((0, 4), (21, 0));
    gates.connect_cell_pair((0, 5), (22, 0));
    external_checks.extend_witness_multi_range_checks(&mut witness);
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    gates.connect_cell_pair((6, 2), (24, 0));
    let left_hi_bound = foreign_field_mul::witness::compute_high_bound(&left_input, foreign_field_modulus);
    let right_hi_bound = foreign_field_mul::witness::compute_high_bound(&right_input, foreign_field_modulus);
    external_checks.add_limb_check(&left_hi_bound.into());
    external_checks.add_limb_check(&right_hi_bound.into());
    gates.connect_cell_pair((15, 2), (25, 0));
    gates.connect_cell_pair((15, 5), (26, 0));
    external_checks.extend_witness_limb_checks(&mut witness);
    (gates, witness, vec![]).into()
}

/// A chain of two foreign-field operations modulo secp256k1's base field
/// (an addition, then a subtraction), its bound, and the multi-range checks
/// of its inputs, results and bound, as Kimchi's `foreign_field_add`
/// `full_circuit` and `long_witness`: the public 1 on row 0, the chain from
/// row 1, inputs from SplitMix64(59).
fn ffadd_circuit() -> Circuit {
    let mut rng = SplitMix64(59);
    let modulus = secp256k1_modulus();
    let mut below = || {
        let words: Vec<u32> = (0..8).map(|_| rng.next_u64() as u32).collect();
        BigUint::new(words) % &modulus
    };
    let inputs = vec![below(), below(), below()];
    let opcodes = [FFOps::Add, FFOps::Sub];
    let num = opcodes.len();
    let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
    let mut next_row = 1;
    CircuitGate::<Fp>::extend_chain_ffadd(&mut gates, 0, &mut next_row, &opcodes, &modulus);
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    for _ in 0..num {
        for _ in 0..2 {
            CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
        }
    }
    CircuitGate::extend_multi_range_check(&mut gates, &mut next_row);
    for i in 0..num {
        let ffadd_row = i + 1;
        let left_rc = num + 3 + 8 * i;
        let right_rc = num + 7 + 8 * i;
        let out_rc = num + 11 + 8 * i;
        gates.connect_ffadd_range_checks(ffadd_row, Some(left_rc), Some(right_rc), out_rc);
    }
    gates.connect_ffadd_range_checks(num + 1, None, None, 9 * num + 7);
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 1]);
    witness[0][0] = Fp::one();
    let chain = ffadd_witness::create_chain::<Fp>(&inputs, &opcodes, modulus);
    for col in 0..COLUMNS {
        witness[col].extend(chain[col].iter());
    }
    let left = (witness[0][1], witness[1][1], witness[2][1]);
    range_check::witness::extend_multi(&mut witness, left.0, left.1, left.2);
    for i in 1..num + 1 {
        let right = (witness[3][i], witness[4][i], witness[5][i]);
        let output = (witness[0][i + 1], witness[1][i + 1], witness[2][i + 1]);
        range_check::witness::extend_multi(&mut witness, right.0, right.1, right.2);
        range_check::witness::extend_multi(&mut witness, output.0, output.1, output.2);
    }
    let bound = (witness[0][num + 2], witness[1][num + 2], witness[2][num + 2]);
    range_check::witness::extend_multi(&mut witness, bound.0, bound.1, bound.2);
    (gates, witness, vec![Fp::one()]).into()
}

/// Kimchi's generic test circuit with its 5 public inputs, then a 64-bit
/// Xor: public inputs and lookups in one circuit.
fn public_lookup_circuit() -> Circuit {
    let (mut gates, mut witness, public) = generic::<Fp>();
    let xor = xor_circuit();
    append(&mut gates, &mut witness, xor.gates, xor.witness);
    (gates, witness, public).into()
}

/// Kimchi's generic test circuit with 5 public inputs.
fn generic<F: PrimeField>() -> (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS], Vec<F>) {
    let public: Vec<F> = (1..=5u64).map(|k| F::from(1000 + k)).collect();
    let gates = create_circuit::<F>(0, public.len());
    let mut witness: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![F::zero(); gates.len()]);
    fill_in_witness(0, &mut witness, &public);
    (gates, witness, public)
}

/// Append `block` (gates whose wires name rows from 0) and its witness.
fn append(
    gates: &mut Vec<CircuitGate<Fp>>,
    witness: &mut [Vec<Fp>; COLUMNS],
    block: Vec<CircuitGate<Fp>>,
    rows: [Vec<Fp>; COLUMNS],
) {
    let offset = gates.len();
    for mut g in block {
        for w in g.wires.iter_mut() {
            w.row += offset;
        }
        gates.push(g);
    }
    for (c, r) in witness.iter_mut().zip(rows) {
        c.extend(r);
    }
}

/// One block of each gate without lookups, built and filled as Kimchi's
/// tests do (poseidon.rs, ec.rs, varbasemul.rs, endomul.rs,
/// endomul_scalar.rs), with inputs from SplitMix64(99).
fn gadgets() -> (Vec<CircuitGate<Fp>>, [Vec<Fp>; COLUMNS], Vec<Fp>) {
    let mut rng = SplitMix64(99);
    let mut gates = Vec::new();
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);

    // Poseidon: 11 rows and the output row.
    let params = Vesta::sponge_params();
    let (block, _) = CircuitGate::<Fp>::create_poseidon_gadget(
        0,
        [Wire::for_row(0), Wire::for_row(11)],
        &params.round_constants,
    );
    let mut rows: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 12]);
    poseidon::generate_witness(0, params, &mut rows, [Fp::from(1u32), Fp::from(2u32), Fp::from(3u32)]);
    append(&mut gates, &mut witness, block, rows);

    // CompleteAdd: a doubling, an addition, P + (-P).
    let p = (Pallas::generator() * PallasScalar::rand(&mut rng)).into_affine();
    let q = (Pallas::generator() * PallasScalar::rand(&mut rng)).into_affine();
    let block = (0..3).map(|row| CircuitGate::new(GateType::CompleteAdd, Wire::for_row(row), vec![])).collect();
    let mut rows: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 3]);
    let p2 = (p + p).into_affine();
    let s = (p.x.square() * Fp::from(3u32)) / p.y.double();
    for (c, v) in [p.x, p.y, p.x, p.y, p2.x, p2.y, Fp::zero(), Fp::one(), s, Fp::zero(), Fp::zero()].iter().enumerate() {
        rows[c][0] = *v;
    }
    let pq = (p + q).into_affine();
    let s = (q.y - p.y) / (q.x - p.x);
    for (c, v) in [p.x, p.y, q.x, q.y, pq.x, pq.y, Fp::zero(), Fp::zero(), s, Fp::zero(), (q.x - p.x).inverse().unwrap()].iter().enumerate() {
        rows[c][1] = *v;
    }
    let neg = -p;
    let s = (p.x.square() * Fp::from(3u32)) / p.y.double();
    for (c, v) in [p.x, p.y, neg.x, neg.y, p2.x, p2.y, Fp::one(), Fp::one(), s, (neg.y - p.y).inverse().unwrap(), Fp::zero()].iter().enumerate() {
        rows[c][2] = *v;
    }
    append(&mut gates, &mut witness, block, rows);

    // VarBaseMul: one 255-bit scalar, 51 pairs of rows.
    let bits = 255usize;
    let block = (0..bits / 5)
        .flat_map(|i| {
            [
                CircuitGate::new(GateType::VarBaseMul, Wire::for_row(2 * i), vec![]),
                CircuitGate::new(GateType::Zero, Wire::for_row(2 * i + 1), vec![]),
            ]
        })
        .collect();
    let mut rows: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 2 * (bits / 5)]);
    let x = Fp::rand(&mut rng);
    let bits_lsb: Vec<bool> = BitIteratorLE::new(x.into_bigint()).take(bits).collect();
    let base = Pallas::generator();
    let acc = (base + base).into_affine();
    let bits_msb: Vec<bool> = bits_lsb.iter().copied().rev().collect();
    varbasemul::witness(&mut rows, 0, (base.x, base.y), &bits_msb, (acc.x, acc.y));
    append(&mut gates, &mut witness, block, rows);

    // EndoMul: one 128-bit scalar, 32 rows and a Zero row.
    let chunks = 32;
    let mut block: Vec<CircuitGate<Fp>> = (0..chunks)
        .map(|row| CircuitGate::new(GateType::EndoMul, Wire::for_row(row), vec![]))
        .collect();
    block.push(CircuitGate::new(GateType::Zero, Wire::for_row(chunks), vec![]));
    let mut rows: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); chunks + 1]);
    let (endo_q, _) = endos::<Pallas>();
    let bits_lsb: Vec<bool> = BitIteratorLE::new(Fp::rand(&mut rng).into_bigint()).take(128).collect();
    let acc0 = {
        let t = Pallas::new_unchecked(endo_q * base.x, base.y);
        let p = t + base;
        let acc = (p + p).into_affine();
        (acc.x, acc.y)
    };
    let bits_msb: Vec<bool> = bits_lsb.iter().copied().rev().collect();
    endosclmul::gen_witness(&mut rows, 0, endo_q, (base.x, base.y), &bits_msb, acc0);
    append(&mut gates, &mut witness, block, rows);

    // EndoMulScalar: one 128-bit scalar, 8 rows.
    let block = (0..8)
        .map(|row| CircuitGate::new(GateType::EndoMulScalar, Wire::for_row(row), vec![]))
        .collect();
    let mut rows: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);
    let (_, endo_r) = endos::<Vesta>();
    let bits_lsb: Vec<bool> = BitIteratorLE::new(Fp::rand(&mut rng).into_bigint()).take(128).collect();
    let x = Fp::from_bigint(<Fp as PrimeField>::BigInt::from_bits_le(&bits_lsb)).unwrap();
    endomul_scalar::gen_witness(&mut rows, x, endo_r, 128);
    append(&mut gates, &mut witness, block, rows);

    (gates, witness, vec![])
}

/// A field element from big-endian hex.
fn parse_field<F: PrimeField>(h: &str) -> F {
    let bytes: Vec<u8> = (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect();
    F::from_be_bytes_mod_order(&bytes)
}

/// A point from `x y` or `inf`, and the words it took.
fn parse_point<G: KimchiCurve<55>>(w: &[&str]) -> (G, usize)
where
    G::BaseField: PrimeField,
{
    if w[0] == "inf" {
        (G::zero(), 1)
    } else {
        (G::of_coordinates(parse_field(w[0]), parse_field(w[1])), 2)
    }
}

/// A commitment from `k x y ...`.
fn parse_comm<G: KimchiCurve<55>>(value: &str) -> PolyComm<G>
where
    G::BaseField: PrimeField,
{
    let w: Vec<&str> = value.split_whitespace().collect();
    let k: usize = w[0].parse().unwrap();
    let mut at = 1;
    let chunks = (0..k)
        .map(|_| {
            let (p, used) = parse_point(&w[at..]);
            at += used;
            p
        })
        .collect();
    PolyComm::new(chunks)
}

/// Evaluations from `nz z... nw w...`.
fn parse_evals<F: PrimeField>(value: &str) -> PointEvaluations<Vec<F>> {
    let w: Vec<&str> = value.split_whitespace().collect();
    let nz: usize = w[0].parse().unwrap();
    let nw: usize = w[nz + 1].parse().unwrap();
    PointEvaluations {
        zeta: w[1..=nz].iter().map(|h| parse_field(h)).collect(),
        zeta_omega: w[nz + 2..nz + 2 + nw].iter().map(|h| parse_field(h)).collect(),
    }
}

/// arrabbiata's proof of `name`, from its dump, as a Kimchi proof.
fn ours<G: KimchiCurve<55>>(
    file: &str,
    name: &str,
    prev_challenges: Vec<RecursionChallenge<G>>,
) -> ProverProof<G, OpeningProof<G, 55>, 55>
where
    G::BaseField: PrimeField,
{
    let lines = |kind: &str| -> std::collections::BTreeMap<String, String> {
        file.lines()
            .filter_map(|l| {
                let w: Vec<&str> = l.split_whitespace().collect();
                (w[0] == kind && w[1] == name).then(|| (w[2].to_string(), w[3..].join(" ")))
            })
            .collect()
    };
    let comms = lines("comm");
    let evals = lines("eval");
    let opening = lines("opening");
    let e = |label: &str| parse_evals(&evals[label]);
    let o = |label: &str| evals.get(label).map(|v| parse_evals(v));
    let rounds = opening.keys().filter(|k| k.starts_with("lr")).count();
    let lr = (0..rounds)
        .map(|i| {
            let w: Vec<&str> = opening[&format!("lr{i}")].split_whitespace().collect();
            let (l, used) = parse_point(&w);
            (l, parse_point(&w[used..]).0)
        })
        .collect();
    let point = |label: &str| {
        let w: Vec<&str> = opening[label].split_whitespace().collect();
        parse_point(&w).0
    };
    let ft_eval1 = file
        .lines()
        .find(|l| l.starts_with(&format!("ft_eval1 {name} ")))
        .unwrap()
        .split_whitespace()
        .nth(2)
        .unwrap();
    ProverProof {
        commitments: ProverCommitments {
            w_comm: core::array::from_fn(|i| parse_comm(&comms[&format!("w{i}")])),
            z_comm: parse_comm(&comms["z"]),
            t_comm: parse_comm(&comms["t"]),
            lookup: comms.contains_key("aggreg").then(|| LookupCommitments {
                sorted: (0..).map_while(|i| comms.get(&format!("sorted{i}")).map(|c| parse_comm(c))).collect(),
                aggreg: parse_comm(&comms["aggreg"]),
                runtime: comms.get("runtime").map(|c| parse_comm(c)),
            }),
        },
        proof: OpeningProof {
            lr,
            delta: point("delta"),
            z1: parse_field(&opening["z1"]),
            z2: parse_field(&opening["z2"]),
            sg: point("sg"),
        },
        evals: ProofEvaluations {
            public: Some(e("public")),
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
            range_check0_selector: o("RangeCheck0"),
            range_check1_selector: o("RangeCheck1"),
            foreign_field_add_selector: o("ForeignFieldAdd"),
            foreign_field_mul_selector: o("ForeignFieldMul"),
            xor_selector: o("Xor16"),
            rot_selector: o("Rot64"),
            lookup_aggregation: o("lookup_aggregation"),
            lookup_table: o("lookup_table"),
            lookup_sorted: core::array::from_fn(|i| o(&format!("lookup_sorted{i}"))),
            runtime_lookup_table: o("runtime_lookup_table"),
            runtime_lookup_table_selector: o("runtime_lookup_table_selector"),
            xor_lookup_selector: o("xor_lookup_selector"),
            lookup_gate_lookup_selector: o("lookup_gate_lookup_selector"),
            range_check_lookup_selector: o("range_check_lookup_selector"),
            foreign_field_mul_lookup_selector: o("foreign_field_mul_lookup_selector"),
        },
        ft_eval1: parse_field(ft_eval1),
        prev_challenges,
    }
}

fn run<G, EFq, EFr>(name: &str, circuit: Circuit<G::ScalarField>, max_poly_size: Option<usize>, accumulators: usize)
where
    G: KimchiCurve<55>,
    G::BaseField: PrimeField,
    EFq: Clone + FqSponge<G::BaseField, G, G::ScalarField, 55>,
    EFr: FrSponge<G::ScalarField> + From<&'static ArithmeticSpongeParams<G::ScalarField, 55>>,
{
    let Circuit { gates, witness, public, tables, runtime_cfgs, runtime } = circuit;
    for t in &tables {
        for (c, column) in t.data.iter().enumerate() {
            println!("table {name} {} {c} {}", t.id, values(column));
        }
    }
    if let Some(cfgs) = &runtime_cfgs {
        for cfg in cfgs {
            println!("runtime_cfg {name} {} {}", cfg.id, values(&cfg.first_column));
        }
    }
    for t in &runtime {
        println!("runtime_data {name} {} {}", t.id, values(&t.data));
    }
    for (row, g) in gates.iter().enumerate() {
        let wires: Vec<String> = g.wires.iter().map(|w| format!("{} {}", w.row, w.col)).collect();
        let coeffs: Vec<String> = g.coeffs.iter().map(|c| hex(*c)).collect();
        println!(
            "gate {name} {row} {} {} {} {}",
            typ(g.typ),
            wires.join(" "),
            coeffs.len(),
            coeffs.join(" ")
        );
    }
    for (col, w) in witness.iter().enumerate() {
        for (row, x) in w.iter().enumerate() {
            if !x.is_zero() {
                println!("witness {name} {col} {row} {}", hex(*x));
            }
        }
    }
    let cs = ConstraintSystem::<G::ScalarField>::create(gates)
        .public(public.len())
        .prev_challenges(accumulators)
        .lookup(tables)
        .runtime(runtime_cfgs)
        .max_poly_size(max_poly_size)
        .build()
        .unwrap();
    let depth = max_poly_size.unwrap_or(cs.domain.d1.size as usize);
    let srs = Arc::new(SRS::<G>::create(depth));
    let endo = *G::other_curve_endo();
    let index: ProverIndex<55, G, SRS<G>> = ProverIndex::create(cs, endo, srs, false);
    let vi = index.verifier_index();
    println!("digest {name} {}", hex(vi.digest::<EFq>()));

    // Previous accumulators as Kimchi's test_recursion makes one: random
    // challenges (accumulator j from SplitMix64(123 + j)) and the
    // commitment to their b.
    let prev_challenges: Vec<RecursionChallenge<G>> = (0..accumulators)
        .map(|j| {
            let mut chal_rng = SplitMix64(123 + j as u64);
            let k = index.srs.g.len().trailing_zeros() as usize;
            let chals: Vec<G::ScalarField> = (0..k).map(|_| G::ScalarField::rand(&mut chal_rng)).collect();
            let b = DensePolynomial::from_coefficients_vec(b_poly_coefficients(&chals));
            let b_comm = index.srs.commit_non_hiding(&b, 1);
            let suffix = if j == 0 { String::new() } else { format!("_{j}") };
            for (i, c) in chals.iter().enumerate() {
                println!("prev {name} chal{i}{suffix} {}", hex(*c));
            }
            println!("prev {name} comm{suffix} {}", comm(&b_comm));
            RecursionChallenge::new(chals, b_comm)
        })
        .collect();
    let group_map = <G as CommitmentCurve>::Map::setup();
    let mut rng = SplitMix64(7);
    let proof = ProverProof::<G, OpeningProof<G, 55>, 55>::create_recursive::<EFq, EFr, _>(&group_map, witness, &runtime, &index, prev_challenges.clone(), None, &mut rng)
    .unwrap();

    let c = &proof.commitments;
    for (i, w) in c.w_comm.iter().enumerate() {
        println!("comm {name} w{i} {}", comm(w));
    }
    println!("comm {name} z {}", comm(&c.z_comm));
    println!("comm {name} t {}", comm(&c.t_comm));
    if let Some(l) = &c.lookup {
        for (i, s) in l.sorted.iter().enumerate() {
            println!("comm {name} sorted{i} {}", comm(s));
        }
        println!("comm {name} aggreg {}", comm(&l.aggreg));
        if let Some(r) = &l.runtime {
            println!("comm {name} runtime {}", comm(r));
        }
    }

    let e = &proof.evals;
    eval(name, "public", e.public.as_ref());
    for (i, w) in e.w.iter().enumerate() {
        eval(name, &format!("w{i}"), Some(w));
    }
    eval(name, "z", Some(&e.z));
    for (i, s) in e.s.iter().enumerate() {
        eval(name, &format!("s{i}"), Some(s));
    }
    for (i, s) in e.coefficients.iter().enumerate() {
        eval(name, &format!("coefficient{i}"), Some(s));
    }
    eval(name, "Generic", Some(&e.generic_selector));
    eval(name, "Poseidon", Some(&e.poseidon_selector));
    eval(name, "CompleteAdd", Some(&e.complete_add_selector));
    eval(name, "VarBaseMul", Some(&e.mul_selector));
    eval(name, "EndoMul", Some(&e.emul_selector));
    eval(name, "EndoMulScalar", Some(&e.endomul_scalar_selector));
    eval(name, "RangeCheck0", e.range_check0_selector.as_ref());
    eval(name, "RangeCheck1", e.range_check1_selector.as_ref());
    eval(name, "ForeignFieldAdd", e.foreign_field_add_selector.as_ref());
    eval(name, "ForeignFieldMul", e.foreign_field_mul_selector.as_ref());
    eval(name, "Xor16", e.xor_selector.as_ref());
    eval(name, "Rot64", e.rot_selector.as_ref());
    eval(name, "lookup_aggregation", e.lookup_aggregation.as_ref());
    eval(name, "lookup_table", e.lookup_table.as_ref());
    for (i, s) in e.lookup_sorted.iter().enumerate() {
        eval(name, &format!("lookup_sorted{i}"), s.as_ref());
    }
    eval(name, "runtime_lookup_table", e.runtime_lookup_table.as_ref());
    eval(name, "runtime_lookup_table_selector", e.runtime_lookup_table_selector.as_ref());
    eval(name, "xor_lookup_selector", e.xor_lookup_selector.as_ref());
    eval(name, "lookup_gate_lookup_selector", e.lookup_gate_lookup_selector.as_ref());
    eval(name, "range_check_lookup_selector", e.range_check_lookup_selector.as_ref());
    eval(name, "foreign_field_mul_lookup_selector", e.foreign_field_mul_lookup_selector.as_ref());
    println!("ft_eval1 {name} {}", hex(proof.ft_eval1));
    let o = &proof.proof;
    for (i, (l, r)) in o.lr.iter().enumerate() {
        println!("opening {name} lr{i} {} {}", point(l), point(r));
    }
    println!("opening {name} delta {}", point(&o.delta));
    println!("opening {name} z1 {}", hex(o.z1));
    println!("opening {name} z2 {}", hex(o.z2));
    println!("opening {name} sg {}", point(&o.sg));

    // The public commitment, as Kimchi's verifier computes it (with no
    // public input, `h` in each chunk).
    let d1_size = vi.domain.size as usize;
    let chunk_size = if d1_size < vi.max_poly_size { 1 } else { d1_size / vi.max_poly_size };
    let public_comm = if public.is_empty() {
        PolyComm::new(vec![vi.srs().blinding_commitment(); chunk_size])
    } else {
        let lgr = vi.srs().get_lagrange_basis(vi.domain);
        let bases: Vec<_> = lgr.iter().take(vi.public).collect();
        let negated: Vec<G::ScalarField> = public.iter().map(|s| -*s).collect();
        let public_comm = PolyComm::<G>::multi_scalar_mul(&bases, &negated);
        vi.srs()
            .mask_custom(public_comm.clone(), &public_comm.map(|_| G::ScalarField::one()))
            .unwrap()
            .commitment
    };
    println!("comm {name} public {}", comm(&public_comm));

    // With a dump of arrabbiata's proofs as argument, Kimchi verifies ours.
    if let Some(path) = std::env::args().nth(1) {
        let file = std::fs::read_to_string(path).unwrap();
        if file.lines().any(|l| l.starts_with(&format!("comm {name} "))) {
            let theirs = ours::<G>(&file, name, prev_challenges.clone());
            let ok = verify::<55, G, EFq, EFr, OpeningProof<G, 55>>(&group_map, &vi, &theirs, &public);
            eprintln!("verify {name} arrabbiata {:?}", ok.as_ref().map(|_| "accepted"));
            let mut tampered = theirs.clone();
            tampered.proof.z1 += G::ScalarField::one();
            let bad = verify::<55, G, EFq, EFr, OpeningProof<G, 55>>(&group_map, &vi, &tampered, &public);
            eprintln!("verify {name} arrabbiata-tampered {}", if bad.is_err() { "refused" } else { "ACCEPTED" });
            let own = verify::<55, G, EFq, EFr, OpeningProof<G, 55>>(&group_map, &vi, &proof, &public);
            eprintln!("verify {name} kimchi {:?}", own.as_ref().map(|_| "accepted"));
        }
    }
    let r = proof
        .oracles::<EFq, EFr, _>(&vi, &public_comm, Some(&public))
        .unwrap();
    let k = &r.oracles;
    if let Some((chal, jc)) = &k.joint_combiner {
        println!("oracle {name} joint_combiner_chal {}", hex(chal.inner()));
        println!("oracle {name} joint_combiner {}", hex(*jc));
    }
    println!("oracle {name} beta {}", hex(k.beta));
    println!("oracle {name} gamma {}", hex(k.gamma));
    println!("oracle {name} alpha_chal {}", hex(k.alpha_chal.inner()));
    println!("oracle {name} alpha {}", hex(k.alpha));
    println!("oracle {name} zeta_chal {}", hex(k.zeta_chal.inner()));
    println!("oracle {name} zeta {}", hex(k.zeta));
    println!("oracle {name} v_chal {}", hex(k.v_chal.inner()));
    println!("oracle {name} v {}", hex(k.v));
    println!("oracle {name} u_chal {}", hex(k.u_chal.inner()));
    println!("oracle {name} u {}", hex(k.u));
    println!("oracle {name} ft_eval0 {}", hex(r.ft_eval0));
    println!("oracle {name} combined_inner_product {}", hex(r.combined_inner_product));
    println!("oracle {name} digest {}", hex(r.digest));
}

fn main() {
    if let Some(path) = std::env::args().nth(1) {
        let bytes = std::fs::read(path).unwrap();
        let digest: String = blake2::Blake2b512::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        eprintln!("file {digest}");
    }
    let mut rng = SplitMix64(42);
    for i in 0..8 {
        println!("rand {i} {}", hex(Fp::rand(&mut rng)));
    }
    type V = Vesta;
    type B = BaseSponge;
    type S = ScalarSponge;
    run::<V, B, S>("generic", generic().into(), None, 0);
    run::<V, B, S>("chunked", generic().into(), Some(16), 0);
    run::<V, B, S>("wide", generic().into(), Some(64), 0);
    run::<V, B, S>("gadgets", gadgets().into(), None, 0);
    run::<V, B, S>("recursion", generic().into(), None, 1);
    run::<V, B, S>("xor", xor_circuit(), None, 0);
    run::<V, B, S>("range_check", range_check_circuit(), None, 0);
    run::<V, B, S>("lookup", lookup_circuit(), None, 0);
    run::<V, B, S>("runtime", runtime_circuit(), None, 0);
    run::<V, B, S>("rot", rot_circuit(), None, 0);
    run::<V, B, S>("ffmul", ffmul_circuit(), None, 0);
    run::<V, B, S>("ffadd", ffadd_circuit(), None, 0);
    // Features together: lookups in chunks, two previous accumulators,
    // public inputs with lookups.
    run::<V, B, S>("lookup_chunked", lookup_circuit(), Some(8), 0);
    run::<V, B, S>("xor_chunked", xor_circuit(), Some(16), 0);
    run::<V, B, S>("recursion2", generic().into(), None, 2);
    run::<V, B, S>("public_lookup", public_lookup_circuit(), None, 0);
    // Over Pallas: Kimchi's generic test circuit, unchunked and in chunks.
    run::<Pallas, PallasBaseSponge, PallasScalarSponge>("pallas", generic().into(), None, 0);
    run::<Pallas, PallasBaseSponge, PallasScalarSponge>("pallas_chunked", generic().into(), Some(16), 0);
}
