//! Kimchi's `kimchi/src/tests/endomul.rs`, test by test, with the same
//! inputs and outputs: 100 scalar challenges multiplied by our EndoMul gates
//! (each result `[k] T` for the challenge's scalar, the witness's Blake2b512
//! digest against Kimchi's, the proof's verdict), and the 16-bit regression
//! with Kimchi's result and witness (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::{Affine, Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, GateType, Wire},
    oracle::ScalarChallenge,
    polynomials::endosclmul::gen_witness,
};
use arrabbiata_pasta::{Fp, Fq, Pallas};
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, prove_and_verify, rows};

/// The coordinates of `p`.
fn coordinates(p: &Projective<Pallas>) -> (Fp, Fp) {
    let affine: Affine<Pallas> = p.to_affine();
    (affine.x().unwrap(), affine.y().unwrap())
}

/// Kimchi's base point `T`, its Pallas generator `(1, y)` (mina-curves),
/// which is not ours, `(-1, 2)` (as in zcash's pasta_curves).
fn base(case: &gate_tests::Case) -> Projective<Pallas> {
    let [x, y]: [Fp; 2] = case.elements("base").try_into().unwrap();
    assert_eq!(x, Fp::ONE);
    Affine::<Pallas>::new(x, y).unwrap().to_projective()
}

/// Kimchi's starting accumulator: `2 (phi(T) + T)`.
fn acc0(t: &Projective<Pallas>) -> (Fp, Fp) {
    let sum = t.endomorphism() + *t;
    coordinates(&(sum + sum))
}

/// The low 128 bits of `x`, most significant first.
fn bits_msb(x: Fq) -> Vec<bool> {
    (0..128)
        .rev()
        .map(|i| (x.limb(i / 64) >> (i % 64)) & 1 == 1)
        .collect()
}

/// `x`, a 128-bit integer, in `Fp`.
fn integer(x: Fq) -> Fp {
    let two_64 = Fp::from_u64(2).pow(64);
    Fp::from_u64(x.limb(1)) * two_64 + Fp::from_u64(x.limb(0))
}

/// The Blake2b512 digest of the witness as the tool prints it.
fn digest(witness: &[Vec<Fp>; COLUMNS]) -> String {
    let lines: Vec<String> = rows(witness).iter().map(|row| row.join(" ")).collect();
    Blake2b512::digest(lines.join("\n").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn endomul_test() {
    let case = case("endomul_test");
    let chunks = 32;
    let rows_per_scalar = 1 + chunks;
    let endo_q: Fp = case.elements("endo_q")[0];
    let endo_r: Fq = case.elements("endo_r")[0];
    assert_eq!(endo_q, Pallas::beta(), "Kimchi's endo_q is Pallas's beta");
    assert_eq!(
        endo_r,
        Pallas::lambda(),
        "Kimchi's endo_r is Pallas's lambda"
    );
    let xs: Vec<Fq> = case.elements("x");
    let accs: Vec<Fp> = case.elements("acc");
    let ns: Vec<Fp> = case.elements("n");
    let mut gates = Vec::new();
    for s in 0..xs.len() {
        for i in 0..chunks {
            let row = rows_per_scalar * s + i;
            gates.push(CircuitGate::new(
                GateType::EndoMul,
                Wire::for_row(row),
                vec![],
            ));
        }
        let row = rows_per_scalar * s + chunks;
        gates.push(CircuitGate::new(GateType::Zero, Wire::for_row(row), vec![]));
    }
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let mut witness: [Vec<Fp>; COLUMNS] =
        core::array::from_fn(|_| vec![Fp::ZERO; rows_per_scalar * xs.len()]);
    let t = base(&case);
    let start = acc0(&t);
    for (i, x) in xs.iter().enumerate() {
        let bits = bits_msb(*x);
        let row0 = i * rows_per_scalar;
        let res = gen_witness(&mut witness, row0, endo_q, coordinates(&t), &bits, start).unwrap();
        let scalar = ScalarChallenge(*x).to_field(endo_r);
        let expected = coordinates(&(t * scalar));
        assert_eq!(res.acc, expected, "scalar {i}");
        assert_eq!(
            res.acc,
            (accs[2 * i], accs[2 * i + 1]),
            "Kimchi's scalar {i}"
        );
        assert_eq!(res.n, integer(*x));
        assert_eq!(res.n, ns[i]);
    }
    let kimchi_digest = case.input("digest");
    assert_eq!(kimchi_digest[0], witness[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&witness));
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_endomul_regression() {
    // 0b1010_0011_1100_0101 = 41925, most significant first.
    let case = case("test_endomul_regression");
    let bits = [
        true, false, true, false, false, false, true, true, true, true, false, false, false, true,
        false, true,
    ];
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::ZERO; 5]);
    let t = base(&case);
    let res = gen_witness(
        &mut witness,
        0,
        Pallas::beta(),
        coordinates(&t),
        &bits,
        acc0(&t),
    )
    .unwrap();
    let acc: Vec<Fp> = case.elements("acc");
    assert_eq!(res.acc, (acc[0], acc[1]));
    assert_eq!(res.n, Fp::from_u64(41925));
    assert_eq!(res.n, case.elements::<Fp>("n")[0]);
    assert_eq!(rows(&witness), case.witnesses[0]);
    // Row 0 reads the bits 1, 0, 1, 0, and n is 10 after it.
    let first = [
        witness[11][0],
        witness[12][0],
        witness[13][0],
        witness[14][0],
    ];
    assert_eq!(first, [Fp::ONE, Fp::ZERO, Fp::ONE, Fp::ZERO]);
    assert_eq!(witness[6][1], Fp::from_u64(10));
}
