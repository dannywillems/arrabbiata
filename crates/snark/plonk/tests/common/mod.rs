//! Random straight-line programs, over any field, for the PlonK tests.

#![allow(dead_code)]

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::{Builder, Wire, build};
use arrabbiata_plonk::{
    Witness,
    vanilla::{self, VanillaCircuit},
};
use proptest::prelude::*;

/// One step of a program over the wires built so far.
#[derive(Clone, Debug)]
pub enum Op {
    Witness(u64),
    Constant(u64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    AddConstant(usize, u64),
    Scale(usize, u64),
    Output(usize),
}

pub fn arb_program(max_len: usize) -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        any::<u64>().prop_map(Op::Witness),
        any::<u64>().prop_map(Op::Constant),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Add(i, j)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Sub(i, j)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, j)| Op::Mul(i, j)),
        (any::<usize>(), any::<u64>()).prop_map(|(i, c)| Op::AddConstant(i, c)),
        (any::<usize>(), any::<u64>()).prop_map(|(i, c)| Op::Scale(i, c)),
        any::<usize>().prop_map(Op::Output),
    ];
    prop::collection::vec(op, 1..max_len)
}

fn run<'id, F>(b: &mut Builder<'id, F>, program: &[Op], lie: bool)
where
    F: Field,
{
    let mut wires: Vec<Wire<'id, F>> = vec![b.witness(F::ONE)];
    for op in program {
        let pick = |i: usize| wires[i % wires.len()];
        let w = match op {
            Op::Witness(v) => b.witness(F::from_u64(*v)),
            Op::Constant(c) => b.constant(F::from_u64(*c)),
            Op::Add(i, j) => b.add(&pick(*i), &pick(*j)),
            Op::Sub(i, j) => b.sub(&pick(*i), &pick(*j)),
            Op::Mul(i, j) => b.mul(&pick(*i), &pick(*j)),
            Op::AddConstant(i, c) => b.add_constant(&pick(*i), F::from_u64(*c)),
            Op::Scale(i, c) => b.scale(&pick(*i), F::from_u64(*c)),
            Op::Output(i) => {
                let w = pick(*i);
                let out = b.public_input(w.value());
                b.assert_equal(&w, &out);
                out
            }
        };
        wires.push(w);
    }
    let last = *wires.last().unwrap();
    let claimed = if lie {
        last.value() + F::ONE
    } else {
        last.value()
    };
    let output = b.public_input(claimed);
    b.assert_equal(&last, &output);
}

/// The frontend circuit of `program` and its witness, before compilation.
pub fn frontend<F>(
    program: &[Op],
    lie: bool,
) -> (
    arrabbiata_frontend::vanilla::Circuit<F>,
    arrabbiata_frontend::vanilla::Witness<F>,
)
where
    F: Field,
{
    let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, program, lie));
    (circuit, witness)
}

/// The circuit of `program`, its last wire exposed as a public output that
/// claims the right value, or a wrong one when `lie` is set.
pub fn circuit<F>(program: &[Op], lie: bool) -> (VanillaCircuit<F>, Witness<F, 3>)
where
    F: Field,
{
    let (circuit, witness, ()) = build::<F, _, _>(|b| run(b, program, lie));
    (
        vanilla::circuit(&circuit).unwrap(),
        vanilla::witness(&witness),
    )
}
