//! Lookups in generalized PlonK, both arguments on the same circuits:
//!
//! - range checks: values in the table are proved, values outside are not;
//! - vector lookups (a XOR table), which exercise the compression by
//!   `theta`, with a wrong tuple rejected even when each coordinate is in
//!   its own column of the table;
//! - inputs that are polynomials of degree 2 in the columns;
//! - rows with the selector off are free; repeated queries and a table
//!   with repeated rows (multiplicities above one);
//! - every lookup opening is checked, and a lookup commitment cannot be
//!   moved to another place;
//! - plookup's reserved last row, an ungated plookup and a non-boolean
//!   selector are rejected when the lookup is attached;
//! - KZG on BN254.

use arrabbiata_bn254::Fr;
use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::{
    Argument, Circuit, Error, Gate, Lookup, Variable, max_coefficients, prove, setup, verify,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poly::Rotation;
use arrabbiata_poseidon::{PoseidonSponge, bn254_scalar_sponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

type Pcs = Ipa<Pallas, PoseidonSponge<Fq>>;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript (its opening proofs).
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' Fiat-Shamir transcripts, apart from the
/// two above.
const PROOF_DOMAIN: u64 = 2;

fn ipa(size: usize) -> Pcs {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript)
}

const ARGUMENTS: [Argument; 2] = [Argument::Plookup, Argument::LogUp];

fn curr(column: usize) -> Variable {
    Variable::Witness {
        column,
        rotation: Rotation::CURR,
    }
}

fn variable<const N: usize, const D: usize>(i: usize) -> Sparse<Fq, N, D> {
    Sparse::variable(Var::try_new(i).unwrap()).unwrap()
}

/// A selector on every row but the last (plookup reserves it).
fn all_but_last(n: usize) -> Vec<Fq> {
    let mut on = vec![Fq::ONE; n];
    on[n - 1] = Fq::ZERO;
    on
}

/// Witness column 0 in the range `[0, n)`, the table being fixed column 0
/// holding `0, 1, ..., n - 1`, on the rows of `selector`.
fn range_check(argument: Argument, n: usize, selector: Vec<Fq>) -> Circuit<Fq, 1, 1> {
    let num_rows = n;
    let witness_columns = 1;
    let fixed = vec![(0..n as u64).map(Fq::from_u64).collect()];
    let selectors = vec![selector];
    let variables = vec![curr(0)];
    let gates = vec![];
    let sigma = (0..n).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let circuit = Circuit::new(
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
    let lookup_selector = Some(0);
    let inputs = vec![variable(0)];
    let table = vec![0];
    let selector = lookup_selector;
    let lookup = Lookup::new(argument, selector, inputs, table);
    circuit.with_lookups(vec![lookup]).unwrap()
}

/// Proving and verifying with IPA.
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

    /// Values in the range are proved by both arguments; a value outside it,
    /// on a selected row, makes the witness unsatisfied.
    #[test]
    fn range_checks(values in prop::collection::vec(0u64..16, 15), out in 16u64..1000, at in 0usize..15) {
        let n = 16;
        for argument in ARGUMENTS {
            let c = range_check(argument, n, all_but_last(n));
            let mut column: Vec<Fq> = values.iter().map(|v| Fq::from_u64(*v)).collect();
            // The last row is off: any value.
            column.push(Fq::from_u64(out));
            prop_assert!(c.is_satisfied(std::slice::from_ref(&column), &[]));
            prop_assert_eq!(proves(&c, std::slice::from_ref(&column), &[]), Ok(()), "{:?}", argument);
            column[at] = Fq::from_u64(out);
            prop_assert!(!c.is_satisfied(std::slice::from_ref(&column), &[]));
            prop_assert_eq!(proves(&c, &[column], &[]), Err(Error::Unsatisfied));
        }
    }

    /// A vector lookup into the 2-bit XOR table `(a, b, a ^ b)`: proved for
    /// true triples, rejected for a triple whose coordinates are each in
    /// their column but which is not a row (compression by `theta`).
    #[test]
    fn vector_lookups(rows in prop::collection::vec((0u64..4, 0u64..4), 15), at in 0usize..15) {
        let n = 16;
        let a_column: Vec<Fq> = (0..16u64).map(|i| Fq::from_u64(i / 4)).collect();
        let b_column: Vec<Fq> = (0..16u64).map(|i| Fq::from_u64(i % 4)).collect();
        let x_column: Vec<Fq> = (0..16u64).map(|i| Fq::from_u64((i / 4) ^ (i % 4))).collect();
        for argument in ARGUMENTS {
            let num_rows = n;
            let witness_columns = 3;
            let fixed = vec![a_column.clone(), b_column.clone(), x_column.clone()];
            let selectors = vec![all_but_last(n)];
            let variables = vec![curr(0), curr(1), curr(2)];
            let gates = vec![];
            let sigma = (0..3 * n).collect();
            let public_inputs = 0;
            let bindings = vec![];
            let circuit = Circuit::<Fq, 3, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
            let lookup_selector = Some(0);
            let inputs = vec![variable(0), variable(1), variable(2)];
            let table = vec![0, 1, 2];
            let selector = lookup_selector;
            let c = circuit.with_lookups(vec![Lookup::new(argument, selector, inputs, table)]).unwrap();
            let mut columns = vec![Vec::new(), Vec::new(), Vec::new()];
            for (a, b) in rows.iter().chain(std::iter::once(&(0, 0))) {
                columns[0].push(Fq::from_u64(*a));
                columns[1].push(Fq::from_u64(*b));
                columns[2].push(Fq::from_u64(a ^ b));
            }
            prop_assert_eq!(proves(&c, &columns, &[]), Ok(()), "{:?}", argument);
            // (a, b, a ^ b ^ 1): every coordinate is in its table column.
            columns[2][at] = Fq::from_u64(rows[at].0 ^ rows[at].1 ^ 1);
            prop_assert!(!c.is_satisfied(&columns, &[]));
            prop_assert_eq!(proves(&c, &columns, &[]), Err(Error::Unsatisfied));
        }
    }

    /// The input is a polynomial of degree 2: `a * b` in the table of
    /// products `0..n`, next to a gate on every row.
    #[test]
    fn polynomial_inputs(pairs in prop::collection::vec((0u64..4, 0u64..4), 15)) {
        let n = 16;
        for argument in ARGUMENTS {
            let num_rows = n;
            let witness_columns = 2;
            let fixed = vec![(0..n as u64).map(Fq::from_u64).collect()];
            let selectors = vec![all_but_last(n)];
            let variables = vec![curr(0), curr(1)];
            // Also a gate: b = b, on every row, to mix gates and lookups.
            let b: Sparse<Fq, 2, 2> = variable(1);
            let selector = None;
            let polynomial = b.clone() - b;
            let gates = vec![Gate::new(selector, polynomial)];
            let sigma = (0..2 * n).collect();
            let public_inputs = 0;
            let bindings = vec![];
            let circuit = Circuit::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
            let product = variable::<2, 2>(0).checked_mul(&variable(1)).unwrap();
            let lookup_selector = Some(0);
            let inputs = vec![product];
            let table = vec![0];
            let selector = lookup_selector;
            let c = circuit.with_lookups(vec![Lookup::new(argument, selector, inputs, table)]).unwrap();
            prop_assert_eq!(c.query_degree(&c.lookups()[0]), 3);
            let mut columns = vec![Vec::new(), Vec::new()];
            for (a, b) in pairs.iter().chain(std::iter::once(&(5, 5))) {
                columns[0].push(Fq::from_u64(*a));
                columns[1].push(Fq::from_u64(*b));
            }
            prop_assert_eq!(proves(&c, &columns, &[]), Ok(()), "{:?}", argument);
            columns[0][0] = Fq::from_u64(17);
            columns[1][0] = Fq::ONE;
            prop_assert_eq!(proves(&c, &columns, &[]), Err(Error::Unsatisfied));
        }
    }

    /// Rows with the selector off are free, repeated queries are counted,
    /// and a table with repeated rows works.
    #[test]
    fn off_rows_and_multiplicities(free in prop::collection::vec(any::<u64>(), 8), repeat in 0u64..4) {
        let n = 16;
        // Selected on the even rows.
        let selector: Vec<Fq> = (0..n).map(|i| if i % 2 == 0 && i < n - 1 { Fq::ONE } else { Fq::ZERO }).collect();
        for argument in ARGUMENTS {
            let num_rows = n;
            let witness_columns = 1;
            // The table 0, 1, 2, 3, 0, 1, 2, 3, ...: every row repeated.
            let fixed = vec![(0..n as u64).map(|i| Fq::from_u64(i % 4)).collect()];
            let selectors = vec![selector.clone()];
            let variables = vec![curr(0)];
            let gates = vec![];
            let sigma = (0..n).collect();
            let public_inputs = 0;
            let bindings = vec![];
            let circuit = Circuit::<Fq, 1, 1>::new(num_rows, witness_columns, fixed, selectors, variables, gates, sigma, public_inputs, bindings).unwrap();
            let lookup_selector = Some(0);
            let inputs = vec![variable(0)];
            let table = vec![0];
            let selector = lookup_selector;
            let c = circuit.with_lookups(vec![Lookup::new(argument, selector, inputs, table)]).unwrap();
            let column: Vec<Fq> = (0..n).map(|i| if i % 2 == 0 { Fq::from_u64(repeat) } else { Fq::from_u64(free[i / 2]) }).collect();
            prop_assert_eq!(proves(&c, &[column], &[]), Ok(()), "{:?}", argument);
        }
    }

    /// Every lookup opening and every lookup evaluation is checked, and a
    /// lookup commitment cannot be swapped with another.
    #[test]
    fn lookup_values_are_checked(values in prop::collection::vec(0u64..8, 7)) {
        let n = 8;
        for argument in ARGUMENTS {
            let c = range_check(argument, n, all_but_last(n));
            let mut column: Vec<Fq> = values.iter().map(|v| Fq::from_u64(*v)).collect();
            column.push(Fq::ZERO);
            let pcs = ipa(max_coefficients(&c));
            let (pk, vk) = setup(&c, &pcs).unwrap();
            let (proof, _) = prove(&pk, &pcs, vesta_sponge(PROOF_DOMAIN), &[], &[column]).unwrap();
            prop_assert!(verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), &[], proof.clone()).is_ok());
            // The evaluations are the sixth message.
            let lookups = proof.tail.tail.tail.tail.tail.head.lookups.len();
            prop_assert!(lookups > 0);
            for i in 0..lookups {
                let mut bad = proof.clone();
                let value = &mut bad.tail.tail.tail.tail.tail.head.lookups[i];
                *value = *value + Fq::ONE;
                prop_assert!(verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), &[], bad).is_err(), "{:?} opening {}", argument, i);
            }
            // The lookup columns are the second message.
            let mut bad = proof.clone();
            let before = &mut bad.tail.head.value.0;
            if before.len() > 1 {
                before.swap(0, 1);
            } else {
                before[0] = before[0] + before[0];
            }
            prop_assert!(verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), &[], bad).is_err());
        }
    }
}

/// Plookup reserves the last row and needs a selector; a selector must be
/// boolean; inputs and table columns must match.
#[test]
fn malformed_lookups() {
    let n = 4;
    let base = |selector: Vec<Fq>| range_check(Argument::LogUp, n, selector).lookups().len();
    assert_eq!(base(vec![Fq::ONE; n]), 1);
    let num_rows = n;
    let witness_columns = 1;
    let fixed = vec![vec![Fq::ZERO; n]];
    let selectors = vec![vec![Fq::ONE; n], vec![Fq::from_u64(2); n]];
    let variables = vec![curr(0)];
    let gates = vec![];
    let sigma = (0..n).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<Fq, 1, 1>::new(
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
    let input = || vec![variable::<1, 1>(0)];
    let cases = [
        (
            "plookup on the last row",
            Argument::Plookup,
            Some(0),
            input(),
            vec![0],
        ),
        ("ungated plookup", Argument::Plookup, None, input(), vec![0]),
        (
            "a selector of value 2",
            Argument::LogUp,
            Some(1),
            input(),
            vec![0],
        ),
        (
            "selector 2 does not exist",
            Argument::LogUp,
            Some(2),
            input(),
            vec![0],
        ),
        (
            "table column 1 does not exist",
            Argument::LogUp,
            None,
            input(),
            vec![1],
        ),
        (
            "two inputs, one column",
            Argument::LogUp,
            None,
            vec![variable(0); 2],
            vec![0],
        ),
        ("no input", Argument::LogUp, None, vec![], vec![]),
    ];
    for (name, argument, selector, inputs, table) in cases {
        let lookup = Lookup::new(argument, selector, inputs, table);
        assert_eq!(
            c.clone().with_lookups(vec![lookup]).unwrap_err(),
            Error::Shape,
            "{name}"
        );
    }
    // LogUp needs no reserved row, and may be ungated.
    let argument = Argument::LogUp;
    let selector = None;
    let inputs = input();
    let table = vec![0];
    let ungated = Lookup::new(argument, selector, inputs, table);
    assert!(c.with_lookups(vec![ungated]).is_ok());
}

/// Both arguments with KZG on BN254, in one circuit with two lookups.
#[test]
fn kzg() {
    let n = 8;
    let num_rows = n;
    let witness_columns = 2;
    let fixed = vec![(0..n as u64).map(Fr::from_u64).collect()];
    let mut on = vec![Fr::ONE; n];
    on[n - 1] = Fr::ZERO;
    let selectors = vec![on];
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
    let gates = vec![];
    let sigma = (0..2 * n).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let circuit = Circuit::<Fr, 2, 1>::new(
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
    let column = |i: usize| Sparse::<Fr, 2, 1>::variable(Var::try_new(i).unwrap()).unwrap();
    let argument = Argument::Plookup;
    let selector = Some(0);
    let inputs = vec![column(0)];
    let table = vec![0];
    let plookup = Lookup::new(argument, selector, inputs, table);
    let argument = Argument::LogUp;
    let selector = None;
    let inputs = vec![column(1)];
    let table = vec![0];
    let logup = Lookup::new(argument, selector, inputs, table);
    let c = circuit.with_lookups(vec![plookup, logup]).unwrap();
    let columns = vec![
        [7u64, 0, 3, 3, 5, 1, 6, 100].map(Fr::from_u64).to_vec(),
        [1u64, 1, 1, 2, 0, 7, 7, 4].map(Fr::from_u64).to_vec(),
    ];
    // The KZG trapdoor: known, so the key is for tests only.
    let tau = Fr::from_u64(424_242);
    let pcs = Kzg::<arrabbiata_bn254::Bn254>::insecure_setup(max_coefficients(&c), tau);
    let (pk, vk) = setup(&c, &pcs).unwrap();
    let (proof, _) = prove(&pk, &pcs, bn254_scalar_sponge(PROOF_DOMAIN), &[], &columns).unwrap();
    verify(&vk, &pcs, bn254_scalar_sponge(PROOF_DOMAIN), &[], proof).unwrap();
    let mut bad = columns;
    bad[1][7] = Fr::from_u64(8);
    assert_eq!(
        prove(&pk, &pcs, bn254_scalar_sponge(PROOF_DOMAIN), &[], &bad).unwrap_err(),
        Error::Unsatisfied
    );
}
