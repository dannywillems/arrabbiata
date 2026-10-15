//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::build;
use arrabbiata_generalized_plonk::{
    Argument, Circuit, Commitments, Error, Gate, GeneralizedPlonk, Lookup, Opened, PublicInputs,
    Tagged, Variable, max_coefficients,
    plonkish::{columns_read, from_plonkish, polynomial, variable},
    quotient_factor, rotated, setup,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_pasta::{Fq as F, Fq, Pallas};
use arrabbiata_plonkish::{vanilla, vanilla::Vanilla};
use arrabbiata_poly::{Expr, Rotation, Variable as Read, WitnessColumn};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use arrabbiata_transcript::{
    Absorb, Transcript,
    typed::{Pattern, Protocol},
};

#[test]
fn circuit_overview() {
    // One gate, w0 * w0 - x0 = 0 on the rows of selector 0, with x0 public.
    let var = |i| Sparse::<F, 2, 2>::variable(Var::try_new(i).unwrap()).unwrap();
    let selector = Some(0);
    let polynomial = var(0).checked_mul(&var(0)).unwrap() - var(1);
    let gate = Gate::new(selector, polynomial);
    let variables = vec![
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Public(0),
    ];
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![vec![F::ONE, F::ZERO]];
    let gates = vec![gate];
    let sigma = (0..2).collect();
    let public_inputs = 1;
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
    let columns = vec![vec![F::from_u64(3), F::from_u64(5)]];
    assert!(circuit.is_satisfied(&columns, &[F::from_u64(9)]));
    assert!(!circuit.is_satisfied(&columns, &[F::from_u64(10)]));
}

#[test]
fn circuit_variable() {
    let v = Variable::Fixed {
        column: 2,
        rotation: Rotation::NEXT,
    };
    assert!(v.is_column());
    assert!(!Variable::Public(0).is_column());
}

#[test]
fn circuit_variable_is_column() {
    assert!(
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR
        }
        .is_column()
    );
}

#[test]
fn circuit_gate() {
    let selector = None;
    let polynomial = Sparse::<F, 1, 1>::zero();
    let g = Gate::new(selector, polynomial);
    assert_eq!(g.selector(), None);
}

#[test]
fn circuit_gate_new() {
    let selector = Some(1);
    let polynomial = Sparse::<F, 1, 1>::one();
    let g = Gate::new(selector, polynomial);
    assert_eq!(g.selector(), Some(1));
}

#[test]
fn circuit_gate_selector() {
    let selector = Some(0);
    let polynomial = Sparse::<F, 1, 1>::zero();
    assert_eq!(Gate::new(selector, polynomial).selector(), Some(0));
}

#[test]
fn circuit_gate_polynomial() {
    let selector = None;
    let polynomial = Sparse::<F, 1, 1>::zero();
    assert!(Gate::new(selector, polynomial).polynomial().is_zero());
}

#[test]
fn circuit_circuit() {
    // Three rows is not a power of two.
    let num_rows = 3;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = (0..3).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    );
    assert_eq!(c.unwrap_err(), Error::Shape);
}

#[test]
fn circuit_circuit_new() {
    // The gate reads variable 0, but no variable is declared.
    let selector = None;
    let polynomial = Sparse::<F, 1, 1>::variable(Var::new::<0>()).unwrap();
    let gate = Gate::new(selector, polynomial);
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![gate];
    let sigma = vec![0, 1];
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
    );
    assert_eq!(c.unwrap_err(), Error::Shape);
}

#[test]
fn circuit_circuit_with_permutation() {
    let num_rows = 2;
    let witness_columns = 3;
    let sigma = (0..6).collect();
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let repeated = vec![2, 2];
    assert_eq!(
        c.clone()
            .with_permutation(repeated, vec![0, 1, 2, 3])
            .unwrap_err(),
        Error::Shape
    );
    let only_column_2 = vec![2];
    let swap_its_rows = vec![1, 0];
    assert_eq!(
        c.with_permutation(only_column_2, swap_its_rows)
            .unwrap()
            .permuted(),
        &[2]
    );
}

#[test]
fn circuit_circuit_permuted() {
    let num_rows = 2;
    let witness_columns = 2;
    let sigma = (0..4).collect();
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.permuted(), &[0, 1]);
}

#[test]
fn circuit_circuit_with_lookups() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![vec![F::ZERO; 2]];
    let selectors = vec![vec![F::ONE; 2]];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    // A plookup on the last row.
    let argument = Argument::Plookup;
    let selector = Some(0);
    let inputs = vec![Sparse::zero()];
    let table = vec![0];
    let l = Lookup::new(argument, selector, inputs, table);
    assert_eq!(c.clone().with_lookups(vec![l]).unwrap_err(), Error::Shape);
    // LogUp has no reserved row.
    let argument = Argument::LogUp;
    let selector = Some(0);
    let inputs = vec![Sparse::zero()];
    let table = vec![0];
    let l = Lookup::new(argument, selector, inputs, table);
    assert!(c.with_lookups(vec![l]).is_ok());
}

#[test]
fn circuit_circuit_lookups() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert!(c.lookups().is_empty());
}

#[test]
fn circuit_circuit_num_rows() {
    let num_rows = 4;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.num_rows(), 4);
}

#[test]
fn circuit_circuit_witness_columns() {
    let num_rows = 2;
    let witness_columns = 3;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = (0..6).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.witness_columns(), 3);
}

#[test]
fn circuit_circuit_fixed() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![vec![F::ONE; 2]];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.fixed(), &[vec![F::ONE; 2]]);
}

#[test]
fn circuit_circuit_selectors() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![vec![F::ONE, F::ZERO]];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.selectors().len(), 1);
}

#[test]
fn circuit_circuit_variables() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![Variable::Public(0)];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 1;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.variables(), &[Variable::Public(0)]);
}

#[test]
fn circuit_circuit_gates() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert!(c.gates().is_empty());
}

#[test]
fn circuit_circuit_sigma() {
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![1, 0];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.sigma(), &[1, 0]);
}

#[test]
fn circuit_circuit_public_inputs() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 5;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.public_inputs(), 5);
}

#[test]
fn circuit_circuit_bindings() {
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![0, 1];
    let public_inputs = 1;
    let bindings = vec![(0, 1)];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(c.bindings(), &[(0, 1)]);
}

#[test]
fn circuit_circuit_column_degree() {
    let var = |i| Sparse::<F, 2, 2>::variable(Var::try_new(i).unwrap()).unwrap();
    // w0 * x0: one column, one scalar; gated.
    let selector = Some(0);
    let polynomial = var(0).checked_mul(&var(1)).unwrap();
    let gate = Gate::new(selector, polynomial);
    let vars = vec![
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Public(0),
    ];
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![vec![F::ONE; 2]];
    let variables = vars;
    let gates = vec![gate];
    let sigma = vec![0, 1];
    let public_inputs = 1;
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
    assert_eq!(c.column_degree(&c.gates()[0]), 2);
}

#[test]
fn circuit_circuit_query_degree() {
    let w = Sparse::<F, 1, 2>::variable(Var::new::<0>()).unwrap();
    let vars = vec![Variable::Witness {
        column: 0,
        rotation: Rotation::CURR,
    }];
    let argument = Argument::LogUp;
    let selector = Some(0);
    let inputs = vec![w.checked_mul(&w).unwrap()];
    let table = vec![0];
    let l = Lookup::new(argument, selector, inputs, table);
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![vec![F::ZERO; 2]];
    let selectors = vec![vec![F::ONE; 2]];
    let variables = vars;
    let gates = vec![];
    let sigma = vec![0, 1];
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
    .unwrap()
    .with_lookups(vec![l])
    .unwrap();
    assert_eq!(c.query_degree(&c.lookups()[0]), 3);
}

#[test]
fn circuit_circuit_is_satisfied() {
    // Two cells copied to each other, and one bound to x0.
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![1, 0];
    let public_inputs = 1;
    let bindings = vec![(0, 0)];
    let c = Circuit::<F, 1, 1>::new(
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
    assert!(c.is_satisfied(&[vec![F::ONE, F::ONE]], &[F::ONE]));
    assert!(!c.is_satisfied(&[vec![F::ONE, F::ZERO]], &[F::ONE]));
}

#[test]
fn circuit_circuit_impl() {
    // Sponge domain tags: each transcript gets its own, and prover and
    // verifier start from the same one.
    let transcript_domain = 0;
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![0, 1];
    let public_inputs = 0;
    let bindings = vec![];
    let a = Circuit::<F, 1, 1>::new(
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
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![1, 0];
    let public_inputs = 0;
    let bindings = vec![];
    let b = Circuit::<F, 1, 1>::new(
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
    let (mut s, mut t) = (
        vesta_sponge(transcript_domain),
        vesta_sponge(transcript_domain),
    );
    a.absorb_into(&mut s);
    b.absorb_into(&mut t);
    assert_ne!(s.squeeze(), t.squeeze());
}

#[test]
fn error() {
    assert_eq!(
        Error::IdentityFailed.to_string(),
        "the identity does not hold at zeta"
    );
}

#[test]
fn lookup_argument() {
    assert_ne!(Argument::Plookup, Argument::LogUp);
}

#[test]
fn lookup_lookup() {
    let argument = Argument::LogUp;
    let selector = None;
    let inputs = vec![Sparse::<F, 1, 1>::zero()];
    let table = vec![0];
    let l = Lookup::new(argument, selector, inputs, table);
    assert_eq!(
        (l.argument(), l.selector(), l.table()),
        (Argument::LogUp, None, &[0][..])
    );
}

#[test]
fn lookup_lookup_new() {
    let argument = Argument::Plookup;
    let selector = Some(0);
    let inputs = vec![Sparse::<F, 1, 1>::zero(); 2];
    let table = vec![0, 1];
    let l = Lookup::new(argument, selector, inputs, table);
    assert_eq!(l.inputs().len(), 2);
}

#[test]
fn lookup_lookup_argument() {
    let argument = Argument::Plookup;
    let selector = Some(0);
    let inputs = vec![Sparse::<F, 1, 1>::zero()];
    let table = vec![0];
    assert_eq!(
        Lookup::new(argument, selector, inputs, table).argument(),
        Argument::Plookup
    );
}

#[test]
fn lookup_lookup_selector() {
    let argument = Argument::LogUp;
    let selector = Some(2);
    let inputs = vec![Sparse::<F, 1, 1>::zero()];
    let table = vec![0];
    assert_eq!(
        Lookup::new(argument, selector, inputs, table).selector(),
        Some(2)
    );
}

#[test]
fn lookup_lookup_inputs() {
    let argument = Argument::LogUp;
    let selector = None;
    let inputs = vec![Sparse::<F, 1, 1>::one()];
    let table = vec![0];
    assert!(Lookup::new(argument, selector, inputs, table).inputs()[0].is_constant());
}

#[test]
fn lookup_lookup_table() {
    let argument = Argument::LogUp;
    let selector = None;
    let inputs = vec![Sparse::<F, 1, 1>::zero()];
    let table = vec![3];
    assert_eq!(Lookup::new(argument, selector, inputs, table).table(), &[3]);
}

#[test]
fn lookup_lookup_default() {
    let argument = Argument::LogUp;
    let selector = None;
    let inputs = vec![Sparse::<F, 1, 1>::zero()];
    let table = vec![0];
    assert!(
        Lookup::new(argument, selector, inputs, table)
            .default()
            .is_empty()
    );
}

#[test]
fn plonkish_variable() {
    let v = variable::<2, 0>(&Read::Witness(WitnessColumn::new::<1>(), Rotation::NEXT));
    assert_eq!(
        v,
        Variable::Witness {
            column: 1,
            rotation: Rotation::NEXT
        }
    );
}

#[test]
fn plonkish_polynomial() {
    let a = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let p: Sparse<F, 1, 2> = polynomial(&(a.clone() * a), &|_: Variable| Some(0)).unwrap();
    assert_eq!(p.eval(&[F::from_u64(3)]), F::from_u64(9));
}

#[test]
fn plonkish_columns_read() {
    // a, b, c and the five fixed columns, on the current row.
    assert_eq!(columns_read::<F, Vanilla, 3, 5>().len(), 8);
}

#[test]
fn plonkish_from_plonkish() {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(3));
        let y = b.mul(&x, &x);
        let out = b.public_input(F::from_u64(9));
        b.assert_equal(&y, &out);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    // 8 columns read; q_M a b has degree 3 in all of them.
    let g = from_plonkish::<F, _, 3, 5, 1, 8, 3>(&circuit).unwrap();
    assert!(g.is_satisfied(witness.columns(), witness.public_inputs()));
    assert!(from_plonkish::<F, _, 3, 5, 1, 8, 2>(&circuit).is_err());
}

#[test]
fn protocol_tagged() {
    let t = Tagged::<Ipa<Pallas, PoseidonSponge<Fq>>, _, 1>::new(5u8);
    assert_eq!(t.value, 5);
}

#[test]
fn protocol_tagged_new() {
    assert_eq!(
        Tagged::<Ipa<Pallas, PoseidonSponge<Fq>>, _, 2>::new("x").value,
        "x"
    );
}

#[test]
fn protocol_commitments() {
    assert_eq!(Commitments(vec![1u8, 2]).0.len(), 2);
}

#[test]
fn protocol_public_inputs() {
    assert_eq!(PublicInputs(vec![F::ONE]).0, vec![F::ONE]);
}

#[test]
fn protocol_opened() {
    let o = Opened {
        witness: vec![0u8],
        fixed: vec![],
        selectors: vec![1],
        sigmas: vec![2],
        z: 3,
        t: 4,
        z_next: 5,
        rotated: vec![6],
        lookups: vec![7],
    };
    assert_eq!(
        o.iter().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn protocol_opened_iter() {
    let o = Opened {
        witness: vec![],
        fixed: vec![],
        selectors: vec![],
        sigmas: vec![],
        z: 1u8,
        t: 2,
        z_next: 3,
        rotated: vec![],
        lookups: vec![],
    };
    assert_eq!(o.iter().count(), 3);
}

#[test]
fn protocol_opened_iter_mut() {
    let mut o = Opened {
        witness: vec![1u8],
        fixed: vec![],
        selectors: vec![],
        sigmas: vec![],
        z: 0,
        t: 0,
        z_next: 0,
        rotated: vec![],
        lookups: vec![],
    };
    *o.iter_mut().next().unwrap() = 9;
    assert_eq!(o.witness, vec![9]);
}

#[test]
fn protocol_rotated() {
    let vars = [
        Variable::Witness {
            column: 0,
            rotation: Rotation::NEXT,
        },
        Variable::Witness {
            column: 0,
            rotation: Rotation::CURR,
        },
        Variable::Public(0),
    ];
    assert_eq!(
        rotated(&vars),
        vec![Variable::Witness {
            column: 0,
            rotation: Rotation::NEXT
        }]
    );
}

#[test]
fn protocol_preprocessed() {
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![0, 1];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    let (_, vk) = setup(&c, &pcs).unwrap();
    assert_eq!(vk.value.num_rows(), 2);
}

#[test]
fn protocol_preprocessed_num_rows() {
    let num_rows = 4;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    assert_eq!(setup(&c, &pcs).unwrap().1.value.num_rows(), 4);
}

#[test]
fn protocol_preprocessed_public_inputs() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 3;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    assert_eq!(setup(&c, &pcs).unwrap().1.value.public_inputs(), 3);
}

#[test]
fn protocol_prover_key() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    let (pk, vk) = setup(&c, &pcs).unwrap();
    assert_eq!(pk.verifier_key(), &vk);
}

#[test]
fn protocol_prover_key_verifier_key() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    let (pk, _) = setup(&c, &pcs).unwrap();
    assert_eq!(pk.verifier_key().value.num_rows(), 2);
}

#[test]
fn protocol_prover_key_circuit() {
    let num_rows = 2;
    let witness_columns = 0;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    assert_eq!(setup(&c, &pcs).unwrap().0.circuit(), &c);
}

#[test]
fn protocol_generalized_plonk() {
    type P = <GeneralizedPlonk<F, Ipa<Pallas, PoseidonSponge<F>>, 3, 2> as Protocol>::Transcript;
    // Wires, lookup columns, z, lookup products, t, evaluations, openings.
    assert_eq!(<P as Pattern>::MESSAGES, 7);
    // theta, beta, gamma, alpha, zeta.
    assert_eq!(<P as Pattern>::CHALLENGES, 5);
}

#[test]
fn protocol_quotient_factor() {
    // Three witness columns: the permutation term has degree 4.
    let num_rows = 2;
    let witness_columns = 3;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = (0..6).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(quotient_factor(&c), 4);
}

#[test]
fn protocol_max_coefficients() {
    let num_rows = 4;
    let witness_columns = 3;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = (0..12).collect();
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    assert_eq!(max_coefficients(&c), 12);
}

#[test]
fn protocol_setup() {
    let num_rows = 2;
    let witness_columns = 1;
    let fixed = vec![];
    let selectors = vec![];
    let variables = vec![];
    let gates = vec![];
    let sigma = vec![0, 1];
    let public_inputs = 0;
    let bindings = vec![];
    let c = Circuit::<F, 1, 1>::new(
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
    let mut generators = pallas_sponge(0);
    let size = 4;
    let ipa_transcript = vesta_sponge(1);
    let pcs = Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript);
    assert!(setup(&c, &pcs).is_ok());
}
