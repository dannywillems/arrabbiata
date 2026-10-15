//! Generalized PlonK, as a specification and against PlonK:
//!
//! - the embedding of PlonKish circuits preserves and reflects
//!   satisfaction, on random vanilla programs and on gate sets reading
//!   other rows and fixed columns;
//! - completeness and soundness on those circuits, with IPA and KZG, and
//!   agreement with the PlonK of `arrabbiata-plonk` on which statements
//!   verify;
//! - gates reading public inputs as scalars, of every degree up to the
//!   bound, gates on every row, and overlapping selectors;
//! - every opening and every evaluation is checked.

#[path = "../../plonk/tests/common/mod.rs"]
mod common;

use arrabbiata_bn254::Fr;
use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::{
    Circuit, Error, Gate, Variable, max_coefficients, plonkish::from_plonkish, prove, setup, verify,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_plonk::ipa as plonk_ipa;
use arrabbiata_plonkish::{Builder, Circuit as PlonkishCircuit, Gadget, Witness};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};
use arrabbiata_poseidon::{bn254_scalar_sponge, pallas_sponge, vesta_sponge};
use common::{arb_program, circuit};
use proptest::prelude::*;

fn ipa(size: usize) -> Ipa<Pallas, arrabbiata_poseidon::PoseidonSponge<Fq>> {
    Ipa::<Pallas, _>::setup(
        &mut pallas_sponge(0),
        size.next_power_of_two(),
        vesta_sponge(1),
    )
}

/// Proving and verifying a generalized circuit with IPA: `Ok` exactly when
/// the proof verifies.
fn proves<const N: usize, const D: usize>(
    c: &Circuit<Fq, N, D>,
    columns: &[Vec<Fq>],
    public: &[Fq],
) -> Result<(), Error> {
    let pcs = ipa(max_coefficients(c));
    let (pk, vk) = setup(c, &pcs)?;
    let (proof, _) = prove(&pk, &pcs, vesta_sponge(2), public, columns)?;
    verify(&vk, &pcs, vesta_sponge(2), public, proof).map(|_| ())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Vanilla programs: the embedding agrees with the circuit on every
    /// witness, honest or lying, and the proof verifies exactly when the
    /// witness satisfies; PlonK agrees.
    #[test]
    fn vanilla_programs(program in arb_program(12), lie in any::<bool>()) {
        let (c, w) = circuit::<Fq>(&program, lie);
        let g = from_plonkish::<Fq, _, 3, 5, 1, 8, 3>(&c).unwrap();
        let satisfied = c.is_satisfied(&w);
        prop_assert_eq!(g.is_satisfied(w.columns(), w.public_inputs()), satisfied);
        let ours = proves(&g, w.columns(), w.public_inputs());
        prop_assert_eq!(ours.is_ok(), satisfied, "{:?}", ours);
        if !satisfied {
            prop_assert_eq!(ours, Err(Error::Unsatisfied));
        }
        let pcs = plonk_ipa::pallas(&c);
        let (pk, vk) = arrabbiata_plonk::setup(&c, &pcs).unwrap();
        let theirs = arrabbiata_plonk::prove(&pk, &pcs, plonk_ipa::pallas_transcript(), &w)
            .and_then(|p| arrabbiata_plonk::verify(&vk, &pcs, plonk_ipa::pallas_transcript(), w.public_inputs(), p));
        prop_assert_eq!(theirs.is_ok(), satisfied);
    }
}

const W: usize = 2;
const X: usize = 1;

fn a(k: i32) -> Expr<Fq, W, X> {
    Expr::witness(WitnessColumn::new::<0>(), Rotation::new(k))
}

fn b(k: i32) -> Expr<Fq, W, X> {
    Expr::witness(WitnessColumn::new::<1>(), Rotation::new(k))
}

/// `a = a(-1) + f(-1) a(-2)`.
struct Recurrence;
impl Gadget<Fq, W, X> for Recurrence {
    const NAME: &'static str = "recurrence";
    fn constraints() -> Vec<Expr<Fq, W, X>> {
        let f = Expr::fixed(FixedColumn::new::<0>(), Rotation::PREV);
        vec![a(0) - a(-1) - f * a(-2)]
    }
}

/// `b = a(+1)^5`.
struct Pow;
impl Gadget<Fq, W, X> for Pow {
    const NAME: &'static str = "pow";
    fn constraints() -> Vec<Expr<Fq, W, X>> {
        let x = a(1);
        let x2 = x.clone() * x.clone();
        vec![b(0) - x2.clone() * x2 * x]
    }
}

type Gates = (Recurrence, (Pow, ()));

/// Seeds, weights, and where to compute a fifth power of the next value.
fn rotated_circuit(
    x0: u64,
    x1: u64,
    steps: &[(u64, bool)],
    lie: bool,
) -> (PlonkishCircuit<Fq, Gates, W, X, 2>, Witness<Fq, W>) {
    let len = steps.len() + 3;
    let mut av = vec![Fq::from_u64(x0), Fq::from_u64(x1)];
    let weights: Vec<Fq> = (0..len)
        .map(|i| Fq::from_u64(steps.get(i).map_or(0, |s| s.0)))
        .collect();
    for i in 2..len {
        let next = av[i - 1] + weights[i - 1] * av[i - 2];
        av.push(next);
    }
    let mut builder = Builder::<Fq, Gates, W, X, 2>::new();
    for i in 0..len {
        let pow = i + 1 < len && steps.get(i).is_some_and(|s| s.1);
        let bv = if pow { av[i + 1].pow(5) } else { Fq::ZERO };
        let bv = if lie && i == 0 { bv + Fq::ONE } else { bv };
        let witness = [av[i], bv];
        let fixed = [weights[i]];
        if i >= 2 {
            builder.row::<Recurrence, _>(witness, fixed);
        } else if pow {
            builder.row::<Pow, _>(witness, fixed);
        } else {
            builder.plain_row(witness, fixed);
        }
    }
    let last = len - 1;
    builder.public(last);
    builder.build().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Gates reading the previous rows, the next row and a fixed column at
    /// a rotation: the embedding agrees, and proofs verify iff satisfied.
    #[test]
    fn rotated_gates(x0 in any::<u64>(), x1 in any::<u64>(), steps in prop::collection::vec((0u64..5, any::<bool>()), 1..10), lie in any::<bool>()) {
        let (c, w) = rotated_circuit(x0, x1, &steps, lie);
        // a(-2), a(-1), a(0), a(+1), b(0), f(-1): 6 variables; degree 5.
        let g = from_plonkish::<Fq, _, W, X, 2, 6, 5>(&c).unwrap();
        let satisfied = c.is_satisfied(&w);
        prop_assert_eq!(g.is_satisfied(w.columns(), w.public_inputs()), satisfied);
        prop_assert_eq!(proves(&g, w.columns(), w.public_inputs()).is_ok(), satisfied);
    }
}

/// `w(+1) = s w^d` on the first `n - 1` rows, `s` public input 0, the last
/// value bound to public input 1: a gate of degree `d + 1` in its
/// variables, `d` in the columns.
fn scaled_power<const D: usize>(d: u32, n: usize) -> Circuit<Fq, 3, D> {
    let v = |i| Sparse::<Fq, 3, D>::variable(Var::try_new(i).unwrap()).unwrap();
    let mut power = Sparse::<Fq, 3, D>::one();
    for _ in 0..d {
        power = power.checked_mul(&v(0)).unwrap();
    }
    let gate = Gate::new(Some(0), v(1) - v(2).checked_mul(&power).unwrap());
    let variables = vec![
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Witness {
            column: 0,
            rotation: Rotation::NEXT,
        },
        Variable::Public(0),
    ];
    let mut selector = vec![Fq::ONE; n];
    selector[n - 1] = Fq::ZERO;
    Circuit::new(
        n,
        1,
        vec![],
        vec![selector],
        variables,
        vec![gate],
        (0..n).collect(),
        2,
        vec![(1, n - 1)],
    )
    .unwrap()
}

fn scaled_column(x: Fq, s: Fq, d: u32, n: usize) -> Vec<Fq> {
    let mut column = vec![x];
    for i in 0..n - 1 {
        column.push(s * column[i].pow(u64::from(d)));
    }
    column
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Public inputs read by a gate, at every degree up to the bound: the
    /// statement `(s, y)` verifies, `(s + 1, y)` does not.
    #[test]
    fn public_inputs_are_read(x in any::<u64>(), s in 1u64..1000, d in 1u32..6) {
        let n = 8;
        let c = scaled_power::<7>(d, n);
        let (x, s) = (Fq::from_u64(x), Fq::from_u64(s));
        let column = scaled_column(x, s, d, n);
        let public = [s, column[n - 1]];
        prop_assert!(c.is_satisfied(std::slice::from_ref(&column), &public));
        prop_assert_eq!(proves(&c, std::slice::from_ref(&column), &public), Ok(()));
        let pcs = ipa(max_coefficients(&c));
        let (pk, vk) = setup(&c, &pcs).unwrap();
        let (proof, _) = prove(&pk, &pcs, vesta_sponge(2), &public, &[column]).unwrap();
        let wrong = [s + Fq::ONE, public[1]];
        prop_assert!(verify(&vk, &pcs, vesta_sponge(2), &wrong, proof).is_err());
    }

    /// Every opening and every evaluation of a proof is checked.
    #[test]
    fn every_value_is_checked(x in any::<u64>(), s in 1u64..1000) {
        let n = 4;
        let c = scaled_power::<4>(3, n);
        let (x, s) = (Fq::from_u64(x), Fq::from_u64(s));
        let column = scaled_column(x, s, 3, n);
        let public = [s, column[n - 1]];
        let pcs = ipa(max_coefficients(&c));
        let (pk, vk) = setup(&c, &pcs).unwrap();
        let (proof, _) = prove(&pk, &pcs, vesta_sponge(2), &public, &[column]).unwrap();
        let count = proof.tail.tail.tail.head.iter().count();
        for i in 0..count {
            let mut bad = proof.clone();
            let value = bad.tail.tail.tail.head.iter_mut().nth(i).unwrap();
            *value = *value + Fq::ONE;
            prop_assert!(verify(&vk, &pcs, vesta_sponge(2), &public, bad).is_err(), "evaluation {}", i);
        }
        // Swapping two opening proofs breaks both.
        let mut bad = proof.clone();
        let proofs = &mut bad.tail.tail.tail.tail.head.value;
        let first = proofs.witness[0].clone();
        proofs.witness[0] = proofs.z.clone();
        proofs.z = first;
        prop_assert!(matches!(verify(&vk, &pcs, vesta_sponge(2), &public, bad), Err(Error::Opening(_))));
    }
}

/// A gate on every row (no selector), and two gates selected on the same
/// rows (selectors need not be one-hot).
#[test]
fn ungated_and_overlapping_gates() {
    let n = 4;
    let v = |i| Sparse::<Fq, 2, 2>::variable(Var::try_new(i).unwrap()).unwrap();
    // On every row, b = a^2; on rows 0 and 1, also a = 3 and b = 9.
    let square = Gate::new(None, v(1) - v(0).checked_mul(&v(0)).unwrap());
    let three = Gate::new(Some(0), v(0) - Sparse::constant(Fq::from_u64(3)));
    let nine = Gate::new(Some(1), v(1) - Sparse::constant(Fq::from_u64(9)));
    let variables = vec![
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Witness {
            column: 1,
            rotation: Rotation::CURR,
        },
    ];
    let on = vec![Fq::ONE, Fq::ONE, Fq::ZERO, Fq::ZERO];
    let c = Circuit::new(
        n,
        2,
        vec![],
        vec![on.clone(), on],
        variables,
        vec![square, three, nine],
        (0..2 * n).collect(),
        0,
        vec![],
    )
    .unwrap();
    let a = [3u64, 3, 5, 7].map(Fq::from_u64).to_vec();
    let b: Vec<Fq> = a.iter().map(|x| *x * *x).collect();
    assert_eq!(proves(&c, &[a.clone(), b.clone()], &[]), Ok(()));
    let mut bad = b;
    bad[3] = bad[3] + Fq::ONE;
    assert_eq!(proves(&c, &[a, bad], &[]), Err(Error::Unsatisfied));
}

/// KZG on BN254: a vanilla program and the scaled power.
#[test]
fn kzg() {
    let n = 8;
    let v = |i| Sparse::<Fr, 3, 4>::variable(Var::try_new(i).unwrap()).unwrap();
    let cube = v(0).checked_mul(&v(0)).unwrap().checked_mul(&v(0)).unwrap();
    let gate = Gate::new(Some(0), v(1) - v(2).checked_mul(&cube).unwrap());
    let variables = vec![
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Witness {
            column: 0,
            rotation: Rotation::NEXT,
        },
        Variable::Public(0),
    ];
    let mut selector = vec![Fr::ONE; n];
    selector[n - 1] = Fr::ZERO;
    let c = Circuit::new(
        n,
        1,
        vec![],
        vec![selector],
        variables,
        vec![gate],
        (0..n).collect(),
        2,
        vec![(1, n - 1)],
    )
    .unwrap();
    let s = Fr::from_u64(3);
    let mut column = vec![Fr::from_u64(2)];
    for i in 0..n - 1 {
        column.push(s * column[i].pow(3));
    }
    let public = [s, column[n - 1]];
    let pcs = Kzg::<arrabbiata_bn254::Bn254>::insecure_setup(
        max_coefficients(&c),
        Fr::from_u64(987_654_321),
    );
    let (pk, vk) = setup(&c, &pcs).unwrap();
    let (proof, _) = prove(&pk, &pcs, bn254_scalar_sponge(2), &public, &[column]).unwrap();
    verify(&vk, &pcs, bn254_scalar_sponge(2), &public, proof.clone()).unwrap();
    assert!(verify(&vk, &pcs, bn254_scalar_sponge(2), &[s, Fr::ZERO], proof).is_err());
}

/// Malformed circuits are rejected.
#[test]
fn malformed_circuits() {
    let v = Sparse::<Fq, 2, 1>::variable(Var::new::<1>()).unwrap();
    let one_var = vec![Variable::Witness {
        column: 0,
        rotation: Rotation::CURR,
    }];
    let cases: Vec<Result<Circuit<Fq, 2, 1>, Error>> = vec![
        // Reads variable 1, but only one is declared.
        Circuit::new(
            2,
            1,
            vec![],
            vec![],
            one_var.clone(),
            vec![Gate::new(None, v.clone())],
            vec![0, 1],
            0,
            vec![],
        ),
        // Selector 0 does not exist.
        Circuit::new(
            2,
            1,
            vec![],
            vec![],
            one_var.clone(),
            vec![Gate::new(Some(0), Sparse::zero())],
            vec![0, 1],
            0,
            vec![],
        ),
        // Column 1 does not exist.
        Circuit::new(
            2,
            1,
            vec![],
            vec![],
            vec![Variable::Witness {
                column: 1,
                rotation: Rotation::CURR,
            }],
            vec![],
            vec![0, 1],
            0,
            vec![],
        ),
        // Public input 0 does not exist.
        Circuit::new(
            2,
            1,
            vec![],
            vec![],
            vec![Variable::Public(0)],
            vec![],
            vec![0, 1],
            0,
            vec![],
        ),
        // Not a permutation.
        Circuit::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 0], 0, vec![]),
        // A row bound twice.
        Circuit::new(
            2,
            1,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            2,
            vec![(0, 1), (1, 1)],
        ),
        // No witness column to bind to.
        Circuit::new(
            2,
            0,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            1,
            vec![(0, 0)],
        ),
        // A fixed column of the wrong length.
        Circuit::new(
            2,
            0,
            vec![vec![Fq::ONE]],
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            vec![],
        ),
        // Too many variables for N.
        Circuit::new(
            2,
            1,
            vec![],
            vec![],
            vec![one_var[0]; 3],
            vec![],
            vec![0, 1],
            0,
            vec![],
        ),
    ];
    for (i, c) in cases.into_iter().enumerate() {
        assert_eq!(c.unwrap_err(), Error::Shape, "case {i}");
    }
}
