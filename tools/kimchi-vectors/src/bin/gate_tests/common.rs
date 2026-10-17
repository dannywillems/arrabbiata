use std::sync::Arc;

use ark_ff::{BigInteger, PrimeField, Zero};
use groupmap::GroupMap;
use kimchi::{
    circuits::{constraints::ConstraintSystem, gate::CircuitGate, wires::{Wire, COLUMNS}},
    curve::KimchiCurve,
    proof::ProverProof,
    prover_index::ProverIndex,
    verifier::verify,
};
use mina_curves::pasta::{Fp, Fq, Pallas, PallasParameters, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi as SC,
    pasta::FULL_ROUNDS,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use poly_commitment::{
    commitment::CommitmentCurve,
    ipa::{OpeningProof, SRS},
    SRS as _,
};
use rand::{rngs::StdRng, SeedableRng};

type BaseSponge = DefaultFqSponge<VestaParameters, SC, FULL_ROUNDS>;
type ScalarSponge = DefaultFrSponge<Fp, SC, FULL_ROUNDS>;
type PallasBaseSponge = DefaultFqSponge<PallasParameters, SC, FULL_ROUNDS>;
type PallasScalarSponge = DefaultFrSponge<Fq, SC, FULL_ROUNDS>;

pub fn rng(k: u8) -> StdRng {
    StdRng::from_seed([k; 32])
}

pub fn hex<F: PrimeField>(x: F) -> String {
    x.into_bigint()
        .to_bytes_be()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn test(name: &str) {
    println!("test {name}");
}

pub fn input<F: PrimeField>(key: &str, xs: &[F]) {
    let items: Vec<String> = xs.iter().map(|x| hex(*x)).collect();
    println!("input {key} {}", items.join(" "));
}

pub fn number(key: &str, n: usize) {
    println!("input {key} {n}");
}

pub fn witness<F: PrimeField>(w: &[Vec<F>; COLUMNS]) {
    println!("witness {}", w[0].len());
    for r in 0..w[0].len() {
        let items: Vec<String> = (0..COLUMNS).map(|c| hex(w[c][r])).collect();
        println!("row {}", items.join(" "));
    }
}

pub fn verdict<T: core::fmt::Debug>(label: &str, result: &T) {
    println!("verdict {label} {result:?}");
}

/// Kimchi's rows: per row its type, coefficients and wires.
pub fn gates_line<F: PrimeField>(gates: &[CircuitGate<F>]) {
    use kimchi::circuits::gate::GateType;
    let padding = |g: &CircuitGate<F>, r: usize| {
        g.typ == GateType::Zero && g.coeffs.iter().all(|c| c.is_zero()) && g.wires == Wire::for_row(r)
    };
    let mut len = gates.len();
    while len > 0 && padding(&gates[len - 1], len - 1) {
        len -= 1;
    }
    let gates = &gates[..len];
    println!("gates {}", gates.len());
    for g in gates {
        let coeffs: Vec<String> = g.coeffs.iter().map(|c| hex(*c)).collect();
        let wires: Vec<String> = g.wires.iter().map(|w| format!("{}:{}", w.row, w.col)).collect();
        println!("gate {:?} {} {} {}", g.typ, wires.join(" "), coeffs.len(), coeffs.join(" "));
    }
}

/// Every row's `verify_witness`, as `verdict row<r>`.
pub fn rows<const R: usize, G: KimchiCurve<R>>(
    cs: &ConstraintSystem<G::ScalarField>,
    w: &[Vec<G::ScalarField>; COLUMNS],
) where
    G::BaseField: PrimeField,
{
    gates_line(&cs.gates);
    for row in 0..w[0].len() {
        let result = cs.gates[row].verify_witness::<R, G>(row, w, cs, &w[0][0..cs.public]);
        verdict(&format!("row{row}"), &result);
    }
}

/// Kimchi's `TestFramework::prove_and_verify` on `G`, with the public
/// prover API: the witness check, the proof, the verification.
pub fn prove_and_verify_on<G, EFq, EFr>(
    gates: Vec<CircuitGate<G::ScalarField>>,
    witness: [Vec<G::ScalarField>; COLUMNS],
    public: Vec<G::ScalarField>,
) -> Result<(), String>
where
    G: KimchiCurve<FULL_ROUNDS>,
    G::BaseField: PrimeField,
    EFq: Clone + mina_poseidon::FqSponge<G::BaseField, G, G::ScalarField, FULL_ROUNDS>,
    EFr: kimchi::plonk_sponge::FrSponge<G::ScalarField>
        + From<&'static mina_poseidon::poseidon::ArithmeticSpongeParams<G::ScalarField, FULL_ROUNDS>>,
{
    gates_line(&gates);
    let cs = ConstraintSystem::<G::ScalarField>::create(gates)
        .public(public.len())
        .lookup(vec![])
        .build()
        .map_err(|e| format!("{e:?}"))?;
    let srs = Arc::new(SRS::<G>::create(cs.domain.d1.size as usize));
    let endo = *G::other_curve_endo();
    let index: ProverIndex<FULL_ROUNDS, G, SRS<G>> = ProverIndex::create(cs, endo, srs, false);
    index.verify(&witness, &public).map_err(|e| format!("{e:?}"))?;
    let group_map = <G as CommitmentCurve>::Map::setup();
    let mut rng = rng(250);
    let proof = ProverProof::<G, OpeningProof<G, FULL_ROUNDS>, FULL_ROUNDS>::create_recursive::<
        EFq,
        EFr,
        _,
    >(&group_map, witness, &[], &index, vec![], None, &mut rng)
    .map_err(|e| e.to_string())?;
    let vi = index.verifier_index();
    verify::<FULL_ROUNDS, G, EFq, EFr, OpeningProof<G, FULL_ROUNDS>>(&group_map, &vi, &proof, &public)
        .map_err(|e| e.to_string())
}

/// [`prove_and_verify_on`] Vesta.
pub fn prove_and_verify(
    gates: Vec<CircuitGate<Fp>>,
    witness: [Vec<Fp>; COLUMNS],
    public: Vec<Fp>,
) -> Result<(), String> {
    prove_and_verify_on::<Vesta, BaseSponge, ScalarSponge>(gates, witness, public)
}

/// [`prove_and_verify_on`] Pallas.
pub fn prove_and_verify_pallas(
    gates: Vec<CircuitGate<Fq>>,
    witness: [Vec<Fq>; COLUMNS],
    public: Vec<Fq>,
) -> Result<(), String> {
    prove_and_verify_on::<Pallas, PallasBaseSponge, PallasScalarSponge>(gates, witness, public)
}

/// The words a prover draws, recorded as it draws them: the stream our
/// prover replays.
pub struct Recording<R> {
    pub inner: R,
    pub words: Vec<u64>,
    pub other: usize,
}

impl<R: rand::RngCore> rand::RngCore for Recording<R> {
    fn next_u32(&mut self) -> u32 {
        self.other += 1;
        self.inner.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        let w = self.inner.next_u64();
        self.words.push(w);
        w
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.other += 1;
        self.inner.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.other += 1;
        self.inner.try_fill_bytes(dest)
    }
}

impl<R: rand::CryptoRng> rand::CryptoRng for Recording<R> {}

fn point_hex(p: &Vesta) -> String {
    use ark_ec::AffineRepr;
    match p.xy() {
        Some((x, y)) => format!("{} {}", hex(x), hex(y)),
        None => String::from("inf"),
    }
}

fn comm_hex(c: &poly_commitment::commitment::PolyComm<Vesta>) -> String {
    let chunks: Vec<String> = c.chunks.iter().map(point_hex).collect();
    format!("{} {}", c.chunks.len(), chunks.join(" "))
}

fn values(v: &[Fp]) -> String {
    let hexes: Vec<String> = v.iter().map(|x| hex(*x)).collect();
    format!("{} {}", v.len(), hexes.join(" "))
}

/// A proof in the prover tool's lines, the absent evaluations left out.
pub fn print_proof(name: &str, proof: &ProverProof<Vesta, OpeningProof<Vesta, FULL_ROUNDS>, FULL_ROUNDS>) {
    let c = &proof.commitments;
    for (i, w) in c.w_comm.iter().enumerate() {
        println!("proof comm {name} w{i} {}", comm_hex(w));
    }
    println!("proof comm {name} z {}", comm_hex(&c.z_comm));
    println!("proof comm {name} t {}", comm_hex(&c.t_comm));
    if let Some(l) = &c.lookup {
        for (i, s) in l.sorted.iter().enumerate() {
            println!("proof comm {name} sorted{i} {}", comm_hex(s));
        }
        println!("proof comm {name} aggreg {}", comm_hex(&l.aggreg));
        if let Some(r) = &l.runtime {
            println!("proof comm {name} runtime {}", comm_hex(r));
        }
    }
    let e = &proof.evals;
    let eval = |label: &str, p: Option<&kimchi::proof::PointEvaluations<Vec<Fp>>>| {
        if let Some(p) = p {
            println!("proof eval {name} {label} {} {}", values(&p.zeta), values(&p.zeta_omega));
        }
    };
    eval("public", e.public.as_ref());
    for (i, w) in e.w.iter().enumerate() {
        eval(&format!("w{i}"), Some(w));
    }
    eval("z", Some(&e.z));
    for (i, s) in e.s.iter().enumerate() {
        eval(&format!("s{i}"), Some(s));
    }
    for (i, s) in e.coefficients.iter().enumerate() {
        eval(&format!("coefficient{i}"), Some(s));
    }
    eval("Generic", Some(&e.generic_selector));
    eval("Poseidon", Some(&e.poseidon_selector));
    eval("CompleteAdd", Some(&e.complete_add_selector));
    eval("VarBaseMul", Some(&e.mul_selector));
    eval("EndoMul", Some(&e.emul_selector));
    eval("EndoMulScalar", Some(&e.endomul_scalar_selector));
    eval("RangeCheck0", e.range_check0_selector.as_ref());
    eval("RangeCheck1", e.range_check1_selector.as_ref());
    eval("ForeignFieldAdd", e.foreign_field_add_selector.as_ref());
    eval("ForeignFieldMul", e.foreign_field_mul_selector.as_ref());
    eval("Xor16", e.xor_selector.as_ref());
    eval("Rot64", e.rot_selector.as_ref());
    eval("lookup_aggregation", e.lookup_aggregation.as_ref());
    eval("lookup_table", e.lookup_table.as_ref());
    for (i, s) in e.lookup_sorted.iter().enumerate() {
        eval(&format!("lookup_sorted{i}"), s.as_ref());
    }
    eval("runtime_lookup_table", e.runtime_lookup_table.as_ref());
    eval("runtime_lookup_table_selector", e.runtime_lookup_table_selector.as_ref());
    eval("xor_lookup_selector", e.xor_lookup_selector.as_ref());
    eval("lookup_gate_lookup_selector", e.lookup_gate_lookup_selector.as_ref());
    eval("range_check_lookup_selector", e.range_check_lookup_selector.as_ref());
    eval("foreign_field_mul_lookup_selector", e.foreign_field_mul_lookup_selector.as_ref());
    println!("proof ft_eval1 {name} {}", hex(proof.ft_eval1));
    let o = &proof.proof;
    for (i, (l, r)) in o.lr.iter().enumerate() {
        println!("proof opening {name} lr{i} {} {}", point_hex(l), point_hex(r));
    }
    println!("proof opening {name} delta {}", point_hex(&o.delta));
    println!("proof opening {name} z1 {}", hex(o.z1));
    println!("proof opening {name} z2 {}", hex(o.z2));
    println!("proof opening {name} sg {}", point_hex(&o.sg));
}

/// Kimchi's `prove_and_check_serialization_regression` up to the proof:
/// the witness check, then the proof with `rng`, recorded.
pub fn prove_recorded<R: rand::RngCore + rand::CryptoRng>(
    gates: Vec<CircuitGate<Fp>>,
    witness: [Vec<Fp>; COLUMNS],
    rng: &mut R,
) -> (ProverProof<Vesta, OpeningProof<Vesta, FULL_ROUNDS>, FULL_ROUNDS>, Vec<u64>, usize) {
    gates_line(&gates);
    let cs = ConstraintSystem::<Fp>::create(gates).lookup(vec![]).build().unwrap();
    let srs = Arc::new(SRS::<Vesta>::create(cs.domain.d1.size as usize));
    let endo = *Vesta::other_curve_endo();
    let index: ProverIndex<FULL_ROUNDS, Vesta, SRS<Vesta>> = ProverIndex::create(cs, endo, srs, false);
    index.verify(&witness, &[]).unwrap();
    let group_map = <Vesta as CommitmentCurve>::Map::setup();
    let mut recording = Recording { inner: rng, words: vec![], other: 0 };
    let proof = ProverProof::<Vesta, OpeningProof<Vesta, FULL_ROUNDS>, FULL_ROUNDS>::create_recursive::<
        BaseSponge,
        ScalarSponge,
        _,
    >(&group_map, witness, &[], &index, vec![], None, &mut recording)
    .unwrap();
    (proof, recording.words, recording.other)
}

/// A large witness as the Blake2b-512 digest of its rows, each row its 15
/// cells in hex joined by spaces, rows joined by newlines.
pub fn witness_digest<F: PrimeField>(w: &[Vec<F>; COLUMNS]) {
    use blake2::{Blake2b512, Digest};
    let rows: Vec<String> = (0..w[0].len())
        .map(|r| (0..COLUMNS).map(|c| hex(w[c][r])).collect::<Vec<_>>().join(" "))
        .collect();
    let digest = Blake2b512::digest(rows.join("\n").as_bytes());
    let hexed: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    println!("input digest {} {hexed}", w[0].len());
}
