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

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript (its opening proofs).
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' Fiat-Shamir transcripts, apart from the
/// two above.
const PROOF_DOMAIN: u64 = 2;

fn ipa(size: usize) -> Ipa<Pallas, arrabbiata_poseidon::PoseidonSponge<Fq>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript)
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
    let (proof, _) = prove(&pk, &pcs, vesta_sponge(PROOF_DOMAIN), public, columns)?;
    verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), public, proof).map(|_| ())
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
    let selector = Some(0);
    let polynomial = v(1) - v(2).checked_mul(&power).unwrap();
    let gate = Gate::new(selector, polynomial);
    let mut selector = vec![Fq::ONE; n];
    selector[n - 1] = Fq::ZERO;
    let num_rows = n;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![selector];
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
    let gates = vec![gate];
    let sigma = (0..n).collect();
    let public_inputs = 2;
    let bindings = vec![(1, n - 1)];
    Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
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
        let (proof, _) = prove(&pk, &pcs, vesta_sponge(PROOF_DOMAIN), &public, &[column]).unwrap();
        let wrong = [s + Fq::ONE, public[1]];
        prop_assert!(verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), &wrong, proof).is_err());
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
        let (proof, _) = prove(&pk, &pcs, vesta_sponge(PROOF_DOMAIN), &public, &[column]).unwrap();
        let count = proof.tail.tail.tail.tail.tail.head.iter().count();
        for i in 0..count {
            let mut bad = proof.clone();
            let value = bad.tail.tail.tail.tail.tail.head.iter_mut().nth(i).unwrap();
            *value = *value + Fq::ONE;
            prop_assert!(verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), &public, bad).is_err(), "evaluation {}", i);
        }
        // Swapping two opening proofs breaks both.
        let mut bad = proof.clone();
        let proofs = &mut bad.tail.tail.tail.tail.tail.tail.head.value;
        let first = proofs.witness[0].clone();
        proofs.witness[0] = proofs.z.clone();
        proofs.z = first;
        prop_assert!(matches!(verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), &public, bad), Err(Error::Opening(_))));
    }
}

/// A gate on every row (no selector), and two gates selected on the same
/// rows (selectors need not be one-hot).
#[test]
fn ungated_and_overlapping_gates() {
    let n = 4;
    let v = |i| Sparse::<Fq, 2, 2>::variable(Var::try_new(i).unwrap()).unwrap();
    // On every row, b = a^2; on rows 0 and 1, also a = 3 and b = 9.
    let selector = None;
    let polynomial = v(1) - v(0).checked_mul(&v(0)).unwrap();
    let square = Gate::new(selector, polynomial);
    let selector = Some(0);
    let polynomial = v(0) - Sparse::constant(Fq::from_u64(3));
    let three = Gate::new(selector, polynomial);
    let selector = Some(1);
    let polynomial = v(1) - Sparse::constant(Fq::from_u64(9));
    let nine = Gate::new(selector, polynomial);
    let on = vec![Fq::ONE, Fq::ONE, Fq::ZERO, Fq::ZERO];
    let num_rows = n;
    let witness_columns = 2;
    let fixed = vec![];
    let selectors = vec![on.clone(), on];
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
    let gates = vec![square, three, nine];
    let sigma = (0..2 * n).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
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
    let selector = Some(0);
    let polynomial = v(1) - v(2).checked_mul(&cube).unwrap();
    let gate = Gate::new(selector, polynomial);
    let mut selector = vec![Fr::ONE; n];
    selector[n - 1] = Fr::ZERO;
    let num_rows = n;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![selector];
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
    let gates = vec![gate];
    let sigma = (0..n).collect();
    let public_inputs = 2;
    let bindings = vec![(1, n - 1)];
    let c = Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    )
    .unwrap();
    let s = Fr::from_u64(3);
    let mut column = vec![Fr::from_u64(2)];
    for i in 0..n - 1 {
        column.push(s * column[i].pow(3));
    }
    let public = [s, column[n - 1]];
    // The KZG trapdoor: known, so the key is for tests only.
    let tau = Fr::from_u64(987_654_321);
    let pcs = Kzg::<arrabbiata_bn254::Bn254>::insecure_setup(max_coefficients(&c), tau);
    let (pk, vk) = setup(&c, &pcs).unwrap();
    let (proof, _) = prove(
        &pk,
        &pcs,
        bn254_scalar_sponge(PROOF_DOMAIN),
        &public,
        &[column],
    )
    .unwrap();
    verify(
        &vk,
        &pcs,
        bn254_scalar_sponge(PROOF_DOMAIN),
        &public,
        proof.clone(),
    )
    .unwrap();
    assert!(
        verify(
            &vk,
            &pcs,
            bn254_scalar_sponge(PROOF_DOMAIN),
            &[s, Fr::ZERO],
            proof
        )
        .is_err()
    );
}

/// The parts of a circuit, named, for the malformed cases: each case
/// changes one part of a well-formed circuit.
#[derive(Clone)]
struct Parts {
    num_rows: usize,
    witness_columns: usize,
    fixed: Vec<Vec<Fq>>,
    selectors: Vec<Vec<Fq>>,
    variables: Vec<Variable>,
    gates: Vec<Gate<Fq, 2, 1>>,
    sigma: Vec<usize>,
    public_inputs: usize,
    bindings: Vec<(usize, usize)>,
}

impl Parts {
    fn build(self) -> Result<Circuit<Fq, 2, 1>, Error> {
        let Parts {
            num_rows,
            witness_columns,
            fixed,
            selectors,
            variables,
            gates,
            sigma,
            public_inputs,
            bindings,
        } = self;
        Circuit::new(
            num_rows,
            witness_columns,
            fixed,
            selectors,
            variables,
            gates,
            sigma,
            public_inputs,
            bindings,
        )
    }
}

/// Malformed circuits are rejected.
#[test]
fn malformed_circuits() {
    let curr = Variable::Witness {
        column: 0,
        rotation: Rotation::CURR,
    };
    let good = Parts {
        num_rows: 2,
        witness_columns: 1,
        fixed: vec![],
        selectors: vec![],
        variables: vec![curr],
        gates: vec![],
        sigma: vec![0, 1],
        public_inputs: 0,
        bindings: vec![],
    };
    assert!(good.clone().build().is_ok());
    let reads_undeclared = Sparse::<Fq, 2, 1>::variable(Var::new::<1>()).unwrap();
    let cases = [
        (
            "reads an undeclared variable",
            Parts {
                gates: vec![Gate::new(None, reads_undeclared)],
                ..good.clone()
            },
        ),
        (
            "selector 0 does not exist",
            Parts {
                gates: vec![Gate::new(Some(0), Sparse::zero())],
                ..good.clone()
            },
        ),
        (
            "column 1 does not exist",
            Parts {
                variables: vec![Variable::Witness {
                    column: 1,
                    rotation: Rotation::CURR,
                }],
                ..good.clone()
            },
        ),
        (
            "public input 0 does not exist",
            Parts {
                variables: vec![Variable::Public(0)],
                ..good.clone()
            },
        ),
        (
            "not a permutation",
            Parts {
                sigma: vec![0, 0],
                ..good.clone()
            },
        ),
        (
            "a row bound twice",
            Parts {
                public_inputs: 2,
                bindings: vec![(0, 1), (1, 1)],
                ..good.clone()
            },
        ),
        (
            "no witness column to bind to",
            Parts {
                witness_columns: 0,
                variables: vec![],
                sigma: vec![],
                public_inputs: 1,
                bindings: vec![(0, 0)],
                ..good.clone()
            },
        ),
        (
            "a fixed column of the wrong length",
            Parts {
                fixed: vec![vec![Fq::ONE]],
                ..good.clone()
            },
        ),
        (
            "too many variables for N",
            Parts {
                variables: vec![curr; 3],
                ..good.clone()
            },
        ),
        (
            "three rows",
            Parts {
                num_rows: 3,
                sigma: vec![0, 1, 2],
                ..good.clone()
            },
        ),
    ];
    for (name, parts) in cases {
        assert_eq!(parts.build().unwrap_err(), Error::Shape, "{name}");
    }
}
