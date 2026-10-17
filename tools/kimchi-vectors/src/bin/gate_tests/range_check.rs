//! kimchi/src/tests/range_check.rs
use ark_ff::{Field, One, Zero};
use kimchi::circuits::{
    constraints::ConstraintSystem,
    gate::CircuitGate,
    polynomials::{
        foreign_field_common::{BigUintArrayFieldHelpers, BigUintForeignFieldHelpers, FieldArrayCompact, KimchiForeignElement},
        generic::GenericGateSpec,
        range_check,
    },
    wires::{Wire, COLUMNS},
};
use mina_curves::pasta::{Fp, Vesta};
use num_bigint::{BigUint, RandBigInt};
use o1_utils::{foreign_field::ForeignFieldHelpers, FieldHelpers};

use crate::common::*;

const R: usize = 55;
type W = [Vec<Fp>; COLUMNS];

fn cs(compact: bool) -> ConstraintSystem<Fp> {
    let (_, gates) = if compact {
        CircuitGate::<Fp>::create_compact_multi_range_check(0)
    } else {
        CircuitGate::<Fp>::create_multi_range_check(0)
    };
    ConstraintSystem::create(gates).build().unwrap()
}

fn hex_le(h: &str) -> Fp {
    Fp::from_hex(h).unwrap()
}

/// A test's verdicts, in order.
struct Steps<'a> {
    cs: &'a ConstraintSystem<Fp>,
    n: usize,
}

impl Steps<'_> {
    fn check(&mut self, w: &W, row: usize) {
        if self.n == 0 {
            gates_line(&self.cs.gates);
        }
        let result = self.cs.gates[row].verify_witness::<R, Vesta>(row, w, self.cs, &w[0][0..self.cs.public]);
        verdict(&format!("step{}", self.n), &result);
        self.n += 1;
    }
}

fn multi(v0: Fp, v1: Fp, v2: Fp) -> W {
    input("multi", &[v0, v1, v2]);
    let w = range_check::witness::create_multi::<Fp>(v0, v1, v2);
    witness(&w);
    w
}

fn two_pow(k: u64) -> Fp {
    Fp::from(2u64).pow([k])
}

fn prove(w: W, label: &str, gates: &[CircuitGate<Fp>]) {
    verdict(label, &prove_and_verify(gates.to_vec(), w, vec![]));
}

pub fn run() {
    let c = cs(false);
    {
        test("verify_range_check0_zero_valid_witness");
        let w: W = core::array::from_fn(|_| vec![Fp::zero(); 4]);
        let mut s = Steps { cs: &c, n: 0 };
        s.check(&w, 0);
        s.check(&w, 1);
    }
    {
        test("verify_range_check0_one_invalid_witness");
        let w: W = core::array::from_fn(|_| vec![Fp::one(); 4]);
        let mut s = Steps { cs: &c, n: 0 };
        s.check(&w, 0);
        s.check(&w, 1);
    }
    {
        test("verify_range_check0_valid_witness");
        let mut s = Steps { cs: &c, n: 0 };
        let w = multi(
            hex_le("115655443433221211ffef000000000000000000000000000000000000000000"),
            hex_le("eeddcdccbbabaa99898877000000000000000000000000000000000000000000"),
            hex_le("7766565544343322121100000000000000000000000000000000000000000000"),
        );
        s.check(&w, 0);
        s.check(&w, 1);
        let w = multi(
            hex_le("23d406ac800d1af73040dd000000000000000000000000000000000000000000"),
            hex_le("a8fe8555371eb021469863000000000000000000000000000000000000000000"),
            hex_le("3edff808d8f533be9af500000000000000000000000000000000000000000000"),
        );
        s.check(&w, 0);
        s.check(&w, 1);
    }
    {
        test("verify_range_check0_invalid_witness");
        let mut s = Steps { cs: &c, n: 0 };
        let mut w = multi(
            hex_le("22f6b4e7ecb4488433ade7000000000000000000000000000000000000000000"),
            hex_le("e20e9d80333f2fba463ffd000000000000000000000000000000000000000000"),
            hex_le("25d28bfd6cdff91ca9bc00000000000000000000000000000000000000000000"),
        );
        w[1][0] += Fp::one();
        s.check(&w, 0);
        w[2][1] += Fp::one();
        s.check(&w, 1);
        let mut w = multi(
            hex_le("22cab5e27101eeafd2cbe1000000000000000000000000000000000000000000"),
            hex_le("1ab61d31f4e27fe41a318c000000000000000000000000000000000000000000"),
            hex_le("449a45cd749f1e091a3000000000000000000000000000000000000000000000"),
        );
        w[8][0] = w[0][0] + Fp::one();
        s.check(&w, 0);
        w[9][1] = w[0][1] + Fp::one();
        s.check(&w, 1);
    }
    let z = Fp::zero();
    for (name, row, values) in [
        ("verify_range_check0_valid_v0_in_range", 0, vec![two_pow(88) - Fp::one(), two_pow(64), Fp::from(42u64), Fp::one()]),
        ("verify_range_check0_valid_v1_in_range", 1, vec![two_pow(88) - Fp::one(), two_pow(63), Fp::from(48u64), Fp::one() + Fp::one()]),
        ("verify_range_check0_invalid_v0_not_in_range", 0, vec![two_pow(88), two_pow(96)]),
        ("verify_range_check0_invalid_v1_not_in_range", 1, vec![two_pow(88), two_pow(96)]),
        ("verify_range_check1_valid_v2_in_range", 2, vec![two_pow(88) - Fp::one(), two_pow(64), Fp::from(42u64), Fp::one()]),
        ("verify_range_check1_invalid_v2_not_in_range", 2, vec![two_pow(88), two_pow(96)]),
    ] {
        test(name);
        let mut s = Steps { cs: &c, n: 0 };
        for v in values {
            let w = match row {
                0 => multi(v, z, z),
                1 => multi(z, v, z),
                _ => multi(z, z, v),
            };
            s.check(&w, row);
        }
    }
    {
        test("verify_range_check0_test_copy_constraints");
        let mut s = Steps { cs: &c, n: 0 };
        for row in 0..=1 {
            for col in 1..=2 {
                let mut w = multi(two_pow(88) - Fp::one(), two_pow(88) - Fp::one(), z);
                s.check(&w, row);
                assert_ne!(w[col][row], Fp::zero());
                w[col][row] = Fp::zero();
                s.check(&w, row);
            }
        }
    }
    {
        test("verify_range_check0_v0_test_lookups");
        let mut s = Steps { cs: &c, n: 0 };
        let w = multi(two_pow(88) - Fp::one(), z, z);
        s.check(&w, 0);
        for i in 3..=6 {
            let mut w = w.clone();
            w[i][0] += Fp::from(2u64.pow(12));
            w[i - 1][0] -= Fp::one();
            if i == 3 {
                w[4][3] -= Fp::one();
            }
            prove(w, &format!("prove{i}"), &c.gates);
        }
    }
    {
        test("verify_range_check0_v1_test_lookups");
        let mut s = Steps { cs: &c, n: 0 };
        let w = multi(z, two_pow(88) - Fp::one(), z);
        s.check(&w, 1);
        for i in 3..=6 {
            let mut w = w.clone();
            w[i][1] += Fp::from(2u64.pow(12));
            w[i - 1][1] -= Fp::one();
            if i == 3 {
                w[6][3] -= Fp::one();
            }
            prove(w, &format!("prove{i}"), &c.gates);
        }
    }
    {
        test("verify_range_check1_zero_valid_witness");
        let w: W = core::array::from_fn(|_| vec![Fp::zero(); 4]);
        Steps { cs: &c, n: 0 }.check(&w, 2);
        test("verify_range_check1_one_invalid_witness");
        let w: W = core::array::from_fn(|_| vec![Fp::one(); 4]);
        Steps { cs: &c, n: 0 }.check(&w, 2);
    }
    {
        test("verify_range_check1_valid_witness");
        let mut s = Steps { cs: &c, n: 0 };
        let w = multi(
            hex_le("22cab5e27101eeafd2cbe1000000000000000000000000000000000000000000"),
            hex_le("1ab61d31f4e27fe41a318c000000000000000000000000000000000000000000"),
            hex_le("449a45cd749f1e091a3000000000000000000000000000000000000000000000"),
        );
        s.check(&w, 2);
        let w = multi(
            hex_le("0d96f6fc210316c73bcc4d000000000000000000000000000000000000000000"),
            hex_le("59c8e7b0ffb3cab6ce8d48000000000000000000000000000000000000000000"),
            hex_le("686c10e73930b92f375800000000000000000000000000000000000000000000"),
        );
        s.check(&w, 2);
    }
    {
        test("verify_range_check1_invalid_witness");
        let mut s = Steps { cs: &c, n: 0 };
        let mut w = multi(
            hex_le("2ce2d3ac942f98d59e7e11000000000000000000000000000000000000000000"),
            hex_le("52dd43524b95399f5d458d000000000000000000000000000000000000000000"),
            hex_le("60ca087b427918fa0e2600000000000000000000000000000000000000000000"),
        );
        w[0][2] = w[7][2];
        s.check(&w, 2);
        let mut w = multi(
            hex_le("1bd50c94d2dc83d32f01c0000000000000000000000000000000000000000000"),
            hex_le("e983d7cd9e28e440930f86000000000000000000000000000000000000000000"),
            hex_le("ea226054772cd009d2af00000000000000000000000000000000000000000000"),
        );
        w[13][2] = w[3][2];
        s.check(&w, 2);
    }
    {
        test("verify_range_check1_test_copy_constraints");
        let mut s = Steps { cs: &c, n: 0 };
        for row in 0..=1 {
            for col in 1..=2 {
                let mut w = multi(two_pow(88) - Fp::one(), two_pow(88) - Fp::one(), z);
                s.check(&w, 2);
                assert_ne!(w[col][row], Fp::zero());
                w[col][row] = Fp::zero();
                s.check(&w, 2);
                s.check(&w, 3);
            }
        }
    }
    {
        test("verify_range_check1_test_curr_row_lookups");
        let mut s = Steps { cs: &c, n: 0 };
        let w = multi(z, z, two_pow(88) - Fp::one());
        s.check(&w, 2);
        for i in 3..=6 {
            let mut w = w.clone();
            w[i][2] += Fp::from(2u64.pow(12));
            w[i - 1][2] -= Fp::one();
            prove(w, &format!("prove{i}"), &c.gates);
        }
    }
    {
        test("verify_range_check1_test_next_row_lookups");
        let mut s = Steps { cs: &c, n: 0 };
        let w = multi(two_pow(88) - Fp::one(), two_pow(88) - Fp::one(), z);
        s.check(&w, 2);
        for row in 0..=1 {
            for col in 1..=2 {
                let mut w = w.clone();
                w[col][row] += Fp::from(2u64.pow(12));
                if col > 1 {
                    w[col - 1][row] -= Fp::one();
                    w[col - 1 + 2 * row + 2][3] -= Fp::one();
                } else {
                    w[col - 1][row] += KimchiForeignElement::<Fp>::two_to_limb();
                }
                w[col - 1 + 2 * row + 3][3] += Fp::from(2u64.pow(12));
                prove(w, &format!("prove{row}{col}"), &c.gates);
            }
        }
    }
    {
        test("verify_64_bit_range_check");
        let mut gates = vec![CircuitGate::<Fp>::create_generic_gadget(Wire::for_row(0), GenericGateSpec::Pub, None)];
        gates.append(&mut CircuitGate::<Fp>::create_range_check(1).1);
        gates[1].wires[1] = Wire { row: 1, col: 2 };
        gates[1].wires[2] = Wire { row: 0, col: 0 };
        gates[0].wires[0] = Wire { row: 1, col: 1 };
        let c64 = ConstraintSystem::<Fp>::create(gates).build().unwrap();
        let mut s = Steps { cs: &c64, n: 0 };
        for v in [two_pow(64) - Fp::one(), two_pow(64)] {
            input("v", &[v]);
            let mut w: W = core::array::from_fn(|_| vec![Fp::zero()]);
            range_check::witness::create::<Fp>(v)
                .iter_mut()
                .enumerate()
                .for_each(|(row, col)| w[row].append(col));
            witness(&w);
            s.check(&w, 1);
        }
    }
    {
        test("compact_multi_range_check");
        let rng = &mut rng(30);
        let cc = cs(true);
        let mut s = Steps { cs: &cc, n: 0 };
        for _ in 0..3 {
            let limbs: [Fp; 3] = core::array::from_fn(|_| rng.gen_biguint_below(&BigUint::two_to_limb())).to_fields();
            input("limbs", &limbs);
            let limbs = limbs.to_compact_limbs();
            input("compact", &limbs);
            let mut w = range_check::witness::create_multi_compact_limbs::<Fp>(&limbs);
            witness(&w);
            s.check(&w, 1);
            w[1][2] = Fp::one();
            s.check(&w, 1);
        }
    }
    {
        test("verify_range_check_valid_proof1");
        let w = multi(
            hex_le("2bc0afaa2f6f50b1d1424b000000000000000000000000000000000000000000"),
            hex_le("8b30889f3a39e297ac851a000000000000000000000000000000000000000000"),
            hex_le("c1c85ec47635e8edac5600000000000000000000000000000000000000000000"),
        );
        let (_, gates) = CircuitGate::<Fp>::create_multi_range_check(0);
        prove(w, "prove", &gates);
    }
    {
        test("verify_compact_multi_range_check_proof");
        let rng = &mut rng(31);
        let limbs: [Fp; 3] = core::array::from_fn(|_| rng.gen_biguint_below(&BigUint::two_to_limb())).to_fields();
        input("limbs", &limbs);
        let limbs = limbs.to_compact_limbs();
        input("compact", &limbs);
        let w = range_check::witness::create_multi_compact_limbs::<Fp>(&limbs);
        witness(&w);
        let (_, gates) = CircuitGate::<Fp>::create_compact_multi_range_check(0);
        prove(w, "prove", &gates);
    }
}
