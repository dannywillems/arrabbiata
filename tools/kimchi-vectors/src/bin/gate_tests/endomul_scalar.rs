//! kimchi/src/tests/endomul_scalar.rs
use ark_ff::{BigInteger, BitIteratorLE, Field, One, PrimeField, UniformRand, Zero};
use kimchi::circuits::{
    constraints::ConstraintSystem,
    gate::{CircuitGate, GateType},
    polynomials::endomul_scalar,
    wires::{Wire, COLUMNS},
};
use mina_curves::pasta::{Fp, Pallas as Other, Vesta};
use mina_poseidon::sponge::ScalarChallenge;
use poly_commitment::ipa::endos;

use crate::common::*;

pub fn run() {
    {
        test("endomul_scalar_test");
        let (num_bits, rows_per_scalar, num_scalars) = (128, 8, 100);
        let mut gates = vec![];
        for s in 0..num_scalars {
            for i in 0..rows_per_scalar {
                let row = rows_per_scalar * s + i;
                gates.push(CircuitGate::new(GateType::EndoMulScalar, Wire::for_row(row), vec![]));
            }
        }
        let (_, endo_scalar_coeff) = endos::<Vesta>();
        input("coeff", &[endo_scalar_coeff]);
        let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);
        let rng = &mut rng(70);
        let mut xs = vec![];
        let mut results = vec![];
        for _ in 0..num_scalars {
            let bits_lsb: Vec<_> = BitIteratorLE::new(Fp::rand(rng).into_bigint()).take(num_bits).collect();
            let x = Fp::from_bigint(<Fp as PrimeField>::BigInt::from_bits_le(&bits_lsb[..])).unwrap();
            let r = endomul_scalar::gen_witness(&mut w, x, endo_scalar_coeff, num_bits);
            assert_eq!(ScalarChallenge::new(x).to_field(&endo_scalar_coeff), r);
            xs.push(x);
            results.push(r);
        }
        input("x", &xs);
        input("result", &results);
        witness_digest(&w);
        verdict("prove", &prove_and_verify(gates, w, vec![]));
    }
    {
        test("test_degenerate_case");
        let gate = CircuitGate::new(GateType::EndoMul, Wire::for_row(0), vec![]);
        let zero = CircuitGate::new(GateType::Zero, Wire::for_row(1), vec![]);
        let cs = ConstraintSystem::<Fp>::create(vec![gate.clone(), zero]).build().unwrap();
        let (b1, b2, b3, b4) = (Fp::zero(), Fp::one(), Fp::one(), Fp::one());
        let base = <Other as ark_ec::AffineRepr>::generator();
        let (xt, yt) = (base.x, base.y);
        let (xp, yp) = (xt, yt);
        let (xr, yr) = (xp, -yp);
        let s1 = Fp::from(1337u64);
        let (xs, ys) = (xr, -yr);
        let endo = cs.endo;
        let (xq2, yq2) = (endo * xt, yt);
        let s3 = (yq2 - yr) * (xq2 - xr).inverse().unwrap();
        let (n, n_next) = (Fp::zero(), Fp::from(7u64));
        let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); 2]);
        let cells = [(0, xt), (1, yt), (4, xp), (5, yp), (6, n), (7, xr), (8, yr), (9, s1), (10, s3), (11, b1), (12, b2), (13, b3), (14, b4)];
        for (c, v) in cells {
            w[c][0] = v;
        }
        w[4][1] = xs;
        w[5][1] = ys;
        w[6][1] = n_next;
        witness(&w);
        let result = gate.verify_endomul::<55, Vesta>(0, &w, &cs);
        verdict("is_err", &result.is_err());
    }
}
