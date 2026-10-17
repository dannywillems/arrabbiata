//! kimchi/src/tests/ec.rs
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, Field, One, UniformRand, Zero};
use core::ops::Mul;
use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    wires::{Wire, COLUMNS},
};
use mina_curves::pasta::{Fp as F, Pallas as Other};

use crate::common::*;

pub fn run() {
    test("ec_test");
    let rng = &mut rng(74);
    let (num_doubles, num_additions, num_infs) = (100, 100, 100);
    let mut gates = vec![];
    for row in 0..(num_doubles + num_additions + num_infs) {
        gates.push(CircuitGate::new(GateType::CompleteAdd, Wire::for_row(row), vec![]));
    }
    let mut witness: [Vec<F>; COLUMNS] = core::array::from_fn(|_| vec![]);
    let g = Other::generator();
    input("base", &[g.x, g.y]);
    let r1 = <Other as AffineRepr>::ScalarField::rand(rng);
    let ps: Vec<Other> = {
        let p = Other::generator().into_group().mul(r1);
        let mut res = vec![];
        let mut acc = p;
        for _ in 0..num_additions {
            res.push(acc);
            acc += p;
        }
        <Other as AffineRepr>::Group::normalize_batch(&res)
    };
    let r2 = <Other as AffineRepr>::ScalarField::rand(rng);
    let qs: Vec<Other> = {
        let q = Other::generator().into_group().mul(r2);
        let mut res = vec![];
        let mut acc = q;
        for _ in 0..num_additions {
            res.push(acc);
            acc += q;
        }
        <Other as AffineRepr>::Group::normalize_batch(&res)
    };
    input("r", &[r1, r2]);
    let mut push = |row: [F; 11]| {
        for (c, v) in row.into_iter().enumerate() {
            witness[c].push(v);
        }
        for column in &mut witness[11..] {
            column.push(F::zero());
        }
    };
    for &p in ps.iter().take(num_doubles) {
        let p2: Other = (p + p).into();
        let x1_squared = p.x.square();
        let s = (x1_squared.double() + x1_squared) / p.y.double();
        push([p.x, p.y, p.x, p.y, p2.x, p2.y, F::zero(), F::one(), s, F::zero(), F::zero()]);
    }
    for i in 0..num_additions {
        let (p, q) = (ps[i], qs[i]);
        let pq: Other = (p + q).into();
        let s = (q.y - p.y) / (q.x - p.x);
        push([p.x, p.y, q.x, q.y, pq.x, pq.y, F::zero(), F::zero(), s, F::zero(), (q.x - p.x).inverse().unwrap()]);
    }
    for &p in ps.iter().take(num_infs) {
        let q: Other = -p;
        let p2: Other = (p + p).into();
        let x1_squared = p.x.square();
        let s = (x1_squared.double() + x1_squared) / p.y.double();
        push([p.x, p.y, q.x, q.y, p2.x, p2.y, F::one(), F::one(), s, (q.y - p.y).inverse().unwrap(), F::zero()]);
    }
    witness_digest(&witness);
    verdict("prove", &prove_and_verify(gates, witness, vec![]));
}
