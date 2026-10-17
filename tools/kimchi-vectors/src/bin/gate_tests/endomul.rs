//! kimchi/src/tests/endomul.rs
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInteger, BitIteratorLE, PrimeField, UniformRand, Zero};
use core::ops::Mul;
use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    polynomials::endosclmul,
    wires::{Wire, COLUMNS},
};
use mina_curves::pasta::{Fp, Pallas as Other};
use mina_poseidon::sponge::ScalarChallenge;
use poly_commitment::ipa::endos;

use crate::common::*;

/// Kimchi's starting accumulator, `2 (phi(T) + T)` for the generator `T`.
fn acc0(endo_q: Fp) -> (Fp, Fp) {
    let base = Other::generator();
    let t = Other::new_unchecked(endo_q * base.x, base.y);
    let p = t + base;
    let acc: Other = (p + p).into();
    (acc.x, acc.y)
}

pub fn run() {
    {
        test("endomul_test");
        let (num_bits, chunks, num_scalars) = (128, 32, 100);
        let rows_per_scalar = 1 + chunks;
        let mut gates = vec![];
        for s in 0..num_scalars {
            for i in 0..chunks {
                gates.push(CircuitGate::new(GateType::EndoMul, Wire::for_row(rows_per_scalar * s + i), vec![]));
            }
            gates.push(CircuitGate::new(GateType::Zero, Wire::for_row(rows_per_scalar * s + chunks), vec![]));
        }
        let (endo_q, endo_r) = endos::<Other>();
        input("endo_q", &[endo_q]);
        input("endo_r", &[endo_r]);
        let generator = Other::generator();
        input("base", &[generator.x, generator.y]);
        let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); rows_per_scalar * num_scalars]);
        let rng = &mut rng(72);
        let (mut xs, mut accs, mut ns) = (vec![], vec![], vec![]);
        for i in 0..num_scalars {
            let bits_lsb: Vec<_> = BitIteratorLE::new(Fp::rand(rng).into_bigint()).take(num_bits).collect();
            let x = <Other as AffineRepr>::ScalarField::from_bigint(<Fp as PrimeField>::BigInt::from_bits_le(&bits_lsb[..])).unwrap();
            let x_scalar = ScalarChallenge::new(x).to_field(&endo_r);
            let start = acc0(endo_q);
            let bits_msb: Vec<_> = bits_lsb.iter().take(num_bits).copied().rev().collect();
            let base = Other::generator();
            let res = endosclmul::gen_witness(&mut w, i * rows_per_scalar, endo_q, (base.x, base.y), &bits_msb, start);
            let expected = Other::generator().into_group().mul(x_scalar).into_affine();
            assert_eq!((expected.x, expected.y), res.acc);
            assert_eq!(x.into_bigint(), res.n.into_bigint());
            xs.push(x);
            accs.push(res.acc.0);
            accs.push(res.acc.1);
            ns.push(res.n);
        }
        input("x", &xs);
        input("acc", &accs);
        input("n", &ns);
        witness_digest(&w);
        verdict("prove", &prove_and_verify(gates, w, vec![]));
    }
    {
        test("test_endomul_regression");
        let (endo_q, _) = endos::<Other>();
        let base = Other::generator();
        input("base", &[base.x, base.y]);
        let bits_msb = [
            true, false, true, false, false, false, true, true, true, true, false, false, false, true, false, true,
        ];
        let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 5]);
        let res = endosclmul::gen_witness(&mut w, 0, endo_q, (base.x, base.y), &bits_msb, acc0(endo_q));
        input("acc", &[res.acc.0, res.acc.1]);
        input("n", &[res.n]);
        witness(&w);
        assert_eq!(w[6][1], Fp::from(10u64));
    }
}
