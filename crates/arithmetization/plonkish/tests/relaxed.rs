//! The relaxation of a circuit, as a specification:
//!
//! - the chart: a witness of the circuit satisfies the relaxation at
//!   `u = 1, E = 0` for every coefficient vector;
//! - homogeneity: scaling `(u, w, x)` by `t` and `E` by `t^D` preserves it;
//! - at `u = 1, E = 0`, random coefficients detect an unsatisfied witness;
//! - the degree: below the gate set's, nothing is satisfied.

use arrabbiata_field::Field;
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonkish::{
    Builder, Gadget, RelaxedWitness, Witness,
    gate::{combiner_length, degree},
};
use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
use proptest::prelude::*;

const W: usize = 2;

fn a(k: i32) -> Expr<F, W, 0> {
    Expr::witness(WitnessColumn::new::<0>(), Rotation::new(k))
}

fn b(k: i32) -> Expr<F, W, 0> {
    Expr::witness(WitnessColumn::new::<1>(), Rotation::new(k))
}

/// `b = a^3 + 1` and `a(+1) = b`: two constraints of degree 3 and 1.
struct Cube;
impl Gadget<F, W, 0> for Cube {
    const NAME: &'static str = "cube";
    fn constraints() -> Vec<Expr<F, W, 0>> {
        vec![
            b(0) - a(0) * a(0) * a(0) - Expr::constant(F::ONE),
            a(1) - b(0),
        ]
    }
}

type Gates = (Cube, ());

/// A chain `x -> x^3 + 1 -> ...` of `steps` rows, the last value public.
fn chain(
    x: F,
    steps: usize,
) -> (
    arrabbiata_plonkish::Circuit<F, Gates, W, 0, 1>,
    Witness<F, W>,
) {
    let mut builder = Builder::<F, Gates, W, 0, 1>::new();
    let mut v = x;
    for _ in 0..steps {
        let next = v * v * v + F::ONE;
        builder.row::<Cube, _>([v, next], []);
        v = next;
    }
    let out = builder.plain_row([v, F::ZERO], []);
    builder.public(out);
    builder.build().unwrap()
}

fn arb_f() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(F::from_u64)
}

proptest! {
    #[test]
    fn the_chart_holds_for_every_coefficient(x in arb_f(), steps in 1usize..6, c in prop::collection::vec(arb_f(), 2)) {
        let (circuit, w) = chain(x, steps);
        prop_assert!(circuit.is_satisfied(&w));
        prop_assert_eq!(combiner_length::<F, Gates, W, 0>(), 2);
        let relaxed = RelaxedWitness::embed(&w);
        for d in 3..6 {
            prop_assert!(circuit.is_relaxed_satisfied(&relaxed, w.public_inputs(), F::ONE, &c, d));
        }
    }

    #[test]
    fn scaling_preserves_the_relaxation(x in arb_f(), steps in 1usize..6, t in arb_f(), c in prop::collection::vec(arb_f(), 2), error in arb_f(), row in 0usize..6) {
        let (circuit, w) = chain(x, steps);
        let d = degree::<F, Gates, W, 0>();
        prop_assert_eq!(d, 3);
        // A relaxed witness with an arbitrary error on one gadget row.
        let row = row % steps;
        let n = circuit.num_rows();
        let u = F::ONE;
        let mut e = vec![F::ZERO; n];
        let mut columns = w.columns().clone();
        columns[1][row] = columns[1][row] + error;
        // Recompute the error the change implies, and keep the copy (the
        // next row reads b) consistent by moving a(row + 1) too.
        if row + 1 < n {
            columns[0][row + 1] = columns[1][row];
        }
        for (i, slot) in e.iter_mut().enumerate() {
            *slot = circuit.relaxed_row(&columns, i, u, &c, d).unwrap();
        }
        let relaxed = RelaxedWitness::new(columns.clone(), e.clone());
        let public = vec![columns[0][circuit.public_rows()[0]]];
        prop_assert!(circuit.is_relaxed_satisfied(&relaxed, &public, u, &c, d));
        // Scale (u, w, x) by t and E by t^D.
        let scaled = RelaxedWitness::new(
            columns.map(|col| col.into_iter().map(|v| t * v).collect()),
            e.iter().map(|v| t.pow(d as u64) * *v).collect(),
        );
        let public: Vec<F> = public.iter().map(|v| t * *v).collect();
        prop_assert!(circuit.is_relaxed_satisfied(&scaled, &public, t * u, &c, d));
    }

    #[test]
    fn random_coefficients_detect_an_unsatisfied_witness(x in arb_f(), steps in 1usize..6, row in 0usize..6, alpha in arb_f()) {
        let (circuit, w) = chain(x, steps);
        let mut columns = w.columns().clone();
        let row = row % steps;
        columns[1][row] = columns[1][row] + F::ONE;
        let bad = Witness::new(columns, w.public_inputs().to_vec());
        prop_assert!(!circuit.is_satisfied(&bad));
        let c = [F::ONE, alpha];
        // The broken row has f_0 = 1 and f_1 = -1 (the next row still holds
        // the old b), so 1 - alpha is zero only at the root alpha = 1.
        let relaxed = RelaxedWitness::embed(&bad);
        let caught = !circuit.is_relaxed_satisfied(&relaxed, bad.public_inputs(), F::ONE, &c, 3);
        prop_assert_eq!(caught, alpha != F::ONE);
        prop_assert!(circuit.is_relaxed_satisfied(&relaxed, bad.public_inputs(), F::ONE, &[F::ONE, F::ONE], 3));
    }

    #[test]
    fn a_degree_below_the_gate_set_is_never_satisfied(x in arb_f(), steps in 1usize..4) {
        let (circuit, w) = chain(x, steps);
        let relaxed = RelaxedWitness::embed(&w);
        prop_assert!(!circuit.is_relaxed_satisfied(&relaxed, w.public_inputs(), F::ONE, &[F::ONE, F::ONE], 2));
        prop_assert_eq!(circuit.relaxed_row(w.columns(), 0, F::ONE, &[F::ONE, F::ONE], 2), None);
        // Too few coefficients for the gadget's two constraints.
        prop_assert_eq!(circuit.relaxed_row(w.columns(), 0, F::ONE, &[F::ONE], 3), None);
    }
}
