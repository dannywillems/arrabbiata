//! kimchi/src/tests/varbasemul.rs
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, BigInteger, BitIteratorLE, Field, One, PrimeField, UniformRand, Zero};
use core::ops::Mul;
use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    polynomials::varbasemul,
    wires::{Wire, COLUMNS},
};
use mina_curves::pasta::{Fp, Pallas as Other};

use crate::common::*;

pub fn run() {
    test("varbase_mul_test");
    let num_bits = Fp::MODULUS_BIT_SIZE as usize;
    let chunks = num_bits / 5;
    let num_scalars = 10;
    let rows_per_scalar = 2 * (255 / 5);
    let mut gates = vec![];
    for i in 0..(chunks * num_scalars) {
        let row = 2 * i;
        gates.push(CircuitGate::new(GateType::VarBaseMul, Wire::for_row(row), vec![]));
        gates.push(CircuitGate::new(GateType::Zero, Wire::for_row(row + 1), vec![]));
    }
    let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![Fp::zero(); rows_per_scalar * num_scalars]);
    let rng = &mut rng(73);
    let base = Other::generator();
    input("base", &[base.x, base.y]);
    let (mut xs, mut accs, mut ns) = (vec![], vec![], vec![]);
    for i in 0..num_scalars {
        let x = Fp::rand(rng);
        let bits_lsb: Vec<_> = BitIteratorLE::new(x.into_bigint()).take(num_bits).collect();
        let x_ = <Other as AffineRepr>::ScalarField::from_bigint(<Fp as PrimeField>::BigInt::from_bits_le(&bits_lsb[..])).unwrap();
        let g = Other::generator().into_group();
        let acc = (g + g).into_affine();
        let bits_msb: Vec<_> = bits_lsb.iter().take(num_bits).copied().rev().collect();
        let res = varbasemul::witness(&mut w, i * rows_per_scalar, (base.x, base.y), &bits_msb, (acc.x, acc.y));
        let shift = <Other as AffineRepr>::ScalarField::from(2).pow([bits_msb.len() as u64]);
        let expected = g.mul(&(<Other as AffineRepr>::ScalarField::one() + shift + x_.double())).into_affine();
        assert_eq!(x_.into_bigint(), res.n.into_bigint());
        assert_eq!((expected.x, expected.y), res.acc);
        xs.push(x_);
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
