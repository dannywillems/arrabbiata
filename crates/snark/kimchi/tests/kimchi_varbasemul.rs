//! Kimchi's `kimchi/src/tests/varbasemul.rs`, with the same inputs and
//! outputs: ten scalars of 255 bits multiplied by our VarBaseMul gates, each
//! result `(1 + 2^255 + 2 x) T` and equal to Kimchi's, the witness's
//! Blake2b512 digest Kimchi's, and the proof's verdict (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::{Affine, Projective};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, GateType, Wire},
    polynomials::varbasemul::witness,
};
use arrabbiata_pasta::{Fp, Fq, Pallas};
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, prove_and_verify, rows};

/// The coordinates of `p`.
fn coordinates(p: &Projective<Pallas>) -> (Fp, Fp) {
    let affine: Affine<Pallas> = p.to_affine();
    (affine.x().unwrap(), affine.y().unwrap())
}

/// The Blake2b512 digest of the witness as the tool prints it.
fn digest(witness: &[Vec<Fp>; COLUMNS]) -> String {
    let lines: Vec<String> = rows(witness).iter().map(|row| row.join(" ")).collect();
    Blake2b512::digest(lines.join("\n").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `x`, an integer below both moduli, in `Fp`.
fn integer(x: Fq) -> Fp {
    let two_64 = Fp::from_u64(2).pow(64);
    (0..Fq::LIMBS)
        .rev()
        .fold(Fp::ZERO, |acc, i| acc * two_64 + Fp::from_u64(x.limb(i)))
}

#[test]
fn varbase_mul_test() {
    let case = case("varbase_mul_test");
    let num_bits = 255;
    let chunks = num_bits / 5;
    let rows_per_scalar = 2 * chunks;
    // Kimchi's T, mina-curves' Pallas generator (1, y), not ours.
    let [x, y]: [Fp; 2] = case.elements("base").try_into().unwrap();
    let t = Affine::<Pallas>::new(x, y).unwrap().to_projective();
    let xs: Vec<Fq> = case.elements("x");
    let accs: Vec<Fp> = case.elements("acc");
    let ns: Vec<Fp> = case.elements("n");
    let mut gates = Vec::new();
    for i in 0..chunks * xs.len() {
        let row = 2 * i;
        gates.push(CircuitGate::new(
            GateType::VarBaseMul,
            Wire::for_row(row),
            vec![],
        ));
        gates.push(CircuitGate::new(
            GateType::Zero,
            Wire::for_row(row + 1),
            vec![],
        ));
    }
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let mut w: [Vec<Fp>; COLUMNS] =
        core::array::from_fn(|_| vec![Fp::ZERO; rows_per_scalar * xs.len()]);
    let acc0 = coordinates(&(t + t));
    for (i, x) in xs.iter().enumerate() {
        let bits: Vec<bool> = (0..num_bits)
            .rev()
            .map(|k| (x.limb(k / 64) >> (k % 64)) & 1 == 1)
            .collect();
        let res = witness(&mut w, i * rows_per_scalar, coordinates(&t), &bits, acc0).unwrap();
        let shift = Fq::from_u64(2).pow(num_bits as u64);
        let expected = coordinates(&(t * (Fq::ONE + shift + *x + *x)));
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
    assert_eq!(kimchi_digest[0], w[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&w));
    let result = prove_and_verify(gates, w, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
