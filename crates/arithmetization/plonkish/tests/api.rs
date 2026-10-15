//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::build;
use arrabbiata_pasta::Fq as F;
use arrabbiata_plonkish::{
    Builder as PlonkishBuilder, Cell as PlonkishCell, CircuitError, Gadget as PlonkishGadget,
    RelaxedWitness, Witness as PlonkishWitness,
    circuit::{Builder, Cell, Circuit, Witness},
    gate::{
        Gadget, GateSet, Here, Member, Selector, There, absorb_expr, absorb_gates, combiner_length,
        degree, reads_fixed, total_degree, variables, witness_rotations,
    },
    vanilla,
    vanilla::{Arithmetic, Vanilla, VanillaCircuit},
};
use arrabbiata_poly::{Expr, Rotation, Variable, WitnessColumn};
use arrabbiata_poseidon::vesta_sponge;
use arrabbiata_transcript::{Absorb, Transcript};

#[test]
fn circuit_overview() {
    /// `b = a^2`.
    struct Square;
    impl Gadget<F, 2, 0> for Square {
        const NAME: &'static str = "square";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            vec![b - a.clone() * a]
        }
    }

    // 3 -> 9 -> 81, the output of each row copied to the input of the next,
    // and 81 published (on a row of its own, in column 0).
    let mut b = Builder::<F, (Square, ()), 2, 0, 1>::new();
    let r0 = b.row::<Square, _>([F::from_u64(3), F::from_u64(9)], []);
    let r1 = b.row::<Square, _>([F::from_u64(9), F::from_u64(81)], []);
    let out = b.plain_row([F::from_u64(81), F::ZERO], []);
    let (a, c) = (WitnessColumn::new::<0>(), WitnessColumn::new::<1>());
    b.copy(Cell::new(c, r0), Cell::new(a, r1));
    b.copy(Cell::new(c, r1), Cell::new(a, out));
    b.public(out);
    let (circuit, witness) = b.build().unwrap();
    assert_eq!(circuit.num_rows(), 4);
    assert!(circuit.is_satisfied(&witness));
}

#[test]
fn circuit_cell() {
    let c = Cell::<3>::new(WitnessColumn::new::<2>(), 5);
    assert_eq!(c.index(8), 2 * 8 + 5);
}

#[test]
fn circuit_cell_new() {
    assert_eq!(Cell::<2>::new(WitnessColumn::new::<1>(), 0).row, 0);
}

#[test]
fn circuit_cell_index() {
    assert_eq!(Cell::<2>::new(WitnessColumn::new::<1>(), 3).index(4), 7);
}

#[test]
fn circuit_witness() {
    let w = Witness::new([vec![F::ONE, F::ZERO]], vec![F::ONE]);
    assert_eq!(w.public_inputs(), &[F::ONE]);
}

#[test]
fn circuit_witness_new() {
    let w = Witness::new([vec![F::ONE; 2], vec![F::ZERO; 2]], vec![]);
    assert_eq!(w.columns()[1], vec![F::ZERO; 2]);
}

#[test]
fn circuit_witness_columns() {
    assert_eq!(
        Witness::new([vec![F::ONE]], vec![]).columns()[0],
        vec![F::ONE]
    );
}

#[test]
fn circuit_witness_public_inputs() {
    assert!(
        Witness::<F, 1>::new([vec![]], vec![])
            .public_inputs()
            .is_empty()
    );
}

#[test]
fn circuit_circuit() {
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    assert_eq!(circuit.num_rows(), 2);
}

#[test]
fn circuit_circuit_new() {
    let fixed: [Vec<F>; 5] = std::array::from_fn(|_| vec![F::ZERO; 2]);
    let on = Some(Selector::new::<0>());
    let identity: Vec<usize> = (0..6).collect();
    let ok = Circuit::<F, Vanilla, 3, 5, 1>::new(
        2,
        fixed.clone(),
        vec![on, None],
        identity.clone(),
        vec![],
    );
    assert!(ok.is_ok());
    // A permutation of 5 cells, not 6.
    let bad =
        Circuit::<F, Vanilla, 3, 5, 1>::new(2, fixed, vec![on, None], (0..5).collect(), vec![]);
    assert_eq!(bad.unwrap_err(), CircuitError::Shape);
}

#[test]
fn circuit_circuit_num_rows() {
    let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    assert!(c.num_rows().is_power_of_two());
}

#[test]
fn circuit_circuit_fixed() {
    let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    assert_eq!(c.fixed().len(), 5);
}

#[test]
fn circuit_circuit_gadgets() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    b.row::<Arithmetic, _>([F::ZERO; 3], [F::ZERO; 5]);
    let (c, _) = b.build().unwrap();
    assert_eq!(c.gadgets(), &[Some(Selector::new::<0>()), None]);
}

#[test]
fn circuit_circuit_selector_columns() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    b.row::<Arithmetic, _>([F::ZERO; 3], [F::ZERO; 5]);
    let (c, _) = b.build().unwrap();
    assert_eq!(c.selector_columns(), [vec![F::ONE, F::ZERO]]);
}

#[test]
fn circuit_circuit_sigma() {
    let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    // No copy: the identity.
    assert!(c.sigma().iter().enumerate().all(|(i, s)| i == *s));
}

#[test]
fn circuit_circuit_public_rows() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    b.public(r);
    let (c, w) = b.build().unwrap();
    assert_eq!(c.public_rows(), &[r]);
    assert_eq!(w.public_inputs(), &[F::ONE]);
}

#[test]
fn circuit_circuit_is_satisfied() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    b.public(r);
    let (c, w) = b.build().unwrap();
    assert!(c.is_satisfied(&w));
    // Another public input for the same witness.
    let other = Witness::new(w.columns().clone(), vec![F::ZERO]);
    assert!(!c.is_satisfied(&other));
}

#[test]
fn circuit_builder() {
    // 2 * 3 = 6: q_M a b - c = 0, with q_M = 1 and q_O = -1.
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let (two, three, six) = (F::from_u64(2), F::from_u64(3), F::from_u64(6));
    b.row::<Arithmetic, _>(
        [two, three, six],
        [F::ZERO, F::ZERO, -F::ONE, F::ONE, F::ZERO],
    );
    let (circuit, witness) = b.build().unwrap();
    assert!(circuit.is_satisfied(&witness));
}

#[test]
fn circuit_builder_new() {
    assert_eq!(Builder::<F, Vanilla, 3, 5, 1>::new().num_rows(), 0);
}

#[test]
fn circuit_builder_cyclic() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::cyclic();
    for _ in 0..3 {
        b.plain_row([F::ZERO; 3], [F::ZERO; 5]);
    }
    // 3 rows do not form a cycle of a power-of-two length.
    assert_eq!(b.build().unwrap_err(), CircuitError::Shape);
}

#[test]
fn circuit_builder_num_rows() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    b.plain_row([F::ZERO; 3], [F::ZERO; 5]);
    assert_eq!(b.num_rows(), 1);
}

#[test]
fn circuit_builder_row() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    // a = 1: q_L a + q_C = 0 with q_L = 1, q_C = -1.
    let r = b.row::<Arithmetic, _>(
        [F::ONE, F::ZERO, F::ZERO],
        [F::ONE, F::ZERO, F::ZERO, F::ZERO, -F::ONE],
    );
    assert_eq!(r, 0);
}

#[test]
fn circuit_builder_plain_row() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    assert_eq!(b.plain_row([F::ONE; 3], [F::ZERO; 5]), 0);
    assert_eq!(b.plain_row([F::ONE; 3], [F::ZERO; 5]), 1);
}

#[test]
fn circuit_builder_copy() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    b.copy(
        Cell::new(WitnessColumn::new::<0>(), r),
        Cell::new(WitnessColumn::new::<1>(), r),
    );
    let (c, w) = b.build().unwrap();
    assert!(!c.is_satisfied(&w));
}

#[test]
fn circuit_builder_public() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let r = b.plain_row([F::from_u64(7), F::ZERO, F::ZERO], [F::ZERO; 5]);
    b.public(r);
    let (_, w) = b.build().unwrap();
    assert_eq!(w.public_inputs(), &[F::from_u64(7)]);
}

#[test]
fn circuit_builder_build() {
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    b.public(0);
    assert_eq!(b.build().unwrap_err(), CircuitError::Shape);
}

#[test]
fn circuit_circuit_impl() {
    let mut b = PlonkishBuilder::<F, Vanilla, 3, 5, 1>::new();
    b.row::<Arithmetic, _>([F::ZERO; 3], [F::ZERO; 5]);
    let (on, _) = b.build().unwrap();
    let mut b = PlonkishBuilder::<F, Vanilla, 3, 5, 1>::new();
    b.plain_row([F::ZERO; 3], [F::ZERO; 5]);
    let (off, _) = b.build().unwrap();
    let (mut s, mut t) = (vesta_sponge(0), vesta_sponge(0));
    on.absorb_into(&mut s);
    off.absorb_into(&mut t);
    // Only the selector of row 0 differs.
    assert_ne!(s.squeeze(), t.squeeze());
}

#[test]
fn gate_overview() {
    /// `b = a^2` on the current row.
    struct Square;
    impl Gadget<F, 2, 0> for Square {
        const NAME: &'static str = "square";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            vec![b - a.clone() * a]
        }
    }

    /// `a' = a + b` on the next row.
    struct Step;
    impl Gadget<F, 2, 0> for Step {
        const NAME: &'static str = "step";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            let next = Expr::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
            vec![next - a - b]
        }
    }

    type Gates = (Square, (Step, ()));
    assert_eq!(<Gates as GateSet<F, 2, 0>>::SELECTORS, 2);
    assert_eq!(<Gates as Member<Step, _>>::INDEX, 1);
}

#[test]
fn gate_selector() {
    let s = Selector::<3>::new::<2>();
    assert_eq!(s.index(), 2);
    assert_eq!(Selector::<3>::try_new(3), None);
}

#[test]
fn gate_selector_new() {
    assert_eq!(Selector::<2>::new::<0>().index(), 0);
}

#[test]
fn gate_selector_try_new() {
    assert!(Selector::<2>::try_new(1).is_some());
    assert!(Selector::<2>::try_new(2).is_none());
}

#[test]
fn gate_selector_index() {
    assert_eq!(Selector::<4>::new::<3>().index(), 3);
}

#[test]
fn gate_gadget() {
    /// The Poseidon S-box: `b = a^5`.
    struct SBox;
    impl Gadget<F, 2, 0> for SBox {
        const NAME: &'static str = "sbox";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            let a2 = a.clone() * a.clone();
            vec![b - a2.clone() * a2 * a]
        }
    }
    assert_eq!(SBox::constraints()[0].degree(), 5);
}

#[test]
fn gate_selector_constraints() {
    assert_eq!(<Arithmetic as Gadget<F, 3, 5>>::constraints().len(), 1);
}

#[test]
fn gate_gate_set() {
    assert_eq!(<() as GateSet<F, 3, 5>>::SELECTORS, 0);
    assert_eq!(<Vanilla as GateSet<F, 3, 5>>::SELECTORS, 1);
}

#[test]
fn gate_selector_constraints_2() {
    assert_eq!(<Vanilla as GateSet<F, 3, 5>>::constraints(0).len(), 1);
    assert!(<Vanilla as GateSet<F, 3, 5>>::constraints(1).is_empty());
}

#[test]
fn gate_selector_name() {
    assert_eq!(<Vanilla as GateSet<F, 3, 5>>::name(0), Some("arithmetic"));
    assert_eq!(<Vanilla as GateSet<F, 3, 5>>::name(1), None);
}

#[test]
fn gate_here() {
    assert_eq!(<Vanilla as Member<Arithmetic, Here>>::INDEX, 0);
}

#[test]
fn gate_there() {
    struct A;
    struct B;
    assert_eq!(<(A, (B, ())) as Member<B, There<Here>>>::INDEX, 1);
}

#[test]
fn gate_member() {
    struct A;
    struct B;
    assert_eq!(<(A, (B, ())) as Member<A, _>>::INDEX, 0);
}

#[test]
fn gate_total_degree() {
    let gate = &<Arithmetic as Gadget<F, 3, 5>>::constraints()[0];
    // q_M a b: degree 3 in all columns, 2 in the witness.
    assert_eq!(total_degree(gate), 3);
    assert_eq!(gate.degree(), 2);
}

#[test]
fn gate_variables() {
    let a = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let mut out = Vec::new();
    variables(&(a.clone() * a), &mut out);
    assert_eq!(out.len(), 2);
    assert!(matches!(out[0], Variable::Witness(_, Rotation::CURR)));
}

#[test]
fn gate_degree() {
    assert_eq!(degree::<F, Vanilla, 3, 5>(), 2);
    assert_eq!(degree::<F, (), 3, 5>(), 0);
}

#[test]
fn gate_combiner_length() {
    assert_eq!(combiner_length::<F, Vanilla, 3, 5>(), 1);
    assert_eq!(combiner_length::<F, (), 3, 5>(), 0);
}

#[test]
fn gate_witness_rotations() {
    /// `a' = a`.
    struct Repeat;
    impl Gadget<F, 1, 0> for Repeat {
        const NAME: &'static str = "repeat";
        fn constraints() -> Vec<Expr<F, 1, 0>> {
            let a = |r| Expr::witness(WitnessColumn::new::<0>(), r);
            vec![a(Rotation::NEXT) - a(Rotation::CURR)]
        }
    }
    assert_eq!(
        witness_rotations::<F, (Repeat, ()), 1, 0>(),
        [Rotation::CURR, Rotation::NEXT]
    );
    assert_eq!(witness_rotations::<F, Vanilla, 3, 5>(), [Rotation::CURR]);
}

#[test]
fn gate_reads_fixed() {
    assert!(reads_fixed::<F, Vanilla, 3, 5>(0));
    assert!(!reads_fixed::<F, Vanilla, 3, 5>(1));
}

#[test]
fn gate_absorb_expr() {
    let a = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let (mut s, mut t) = (vesta_sponge(0), vesta_sponge(0));
    absorb_expr(&(a.clone() * a.clone()), &mut s);
    absorb_expr(&(a.clone() + a), &mut t);
    assert_ne!(s.squeeze(), t.squeeze());
}

#[test]
fn gate_absorb_gates() {
    let (mut s, mut t) = (vesta_sponge(0), vesta_sponge(0));
    absorb_gates::<F, Vanilla, _, 3, 5>(&mut s);
    absorb_gates::<F, (), _, 3, 5>(&mut t);
    assert_ne!(s.squeeze(), t.squeeze());
}

#[test]
fn crate_overview() {
    /// The Poseidon S-box, `b = a^5`.
    struct SBox;
    impl PlonkishGadget<F, 2, 0> for SBox {
        const NAME: &'static str = "sbox";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            let a2 = a.clone() * a.clone();
            vec![b - a2.clone() * a2 * a]
        }
    }

    let mut b = PlonkishBuilder::<F, (SBox, ()), 2, 0, 1>::new();
    let x = F::from_u64(3);
    let r0 = b.row::<SBox, _>([x, x.pow(5)], []);
    let out = b.plain_row([x.pow(5), F::ZERO], []);
    b.copy(
        PlonkishCell::new(WitnessColumn::new::<1>(), r0),
        PlonkishCell::new(WitnessColumn::new::<0>(), out),
    );
    b.public(out);
    let (circuit, witness) = b.build().unwrap();
    assert!(circuit.is_satisfied(&witness));
}

#[test]
fn circuit_error() {
    assert_eq!(
        CircuitError::Shape.to_string(),
        "a circuit or a witness has the wrong shape"
    );
}

#[test]
fn relaxed_overview() {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(3));
        let y = b.mul(&x, &x);
        let out = b.public_input(F::from_u64(9));
        b.assert_equal(&y, &out);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    let relaxed = RelaxedWitness::embed(&witness);
    // The chart u = 1, E = 0: any coefficients.
    let coefficients = [F::from_u64(7)];
    assert!(circuit.is_relaxed_satisfied(
        &relaxed,
        witness.public_inputs(),
        F::ONE,
        &coefficients,
        2
    ));
    // Scaling the witness, the slack and the error together keeps it
    // satisfied (homogeneity), here with E = 0.
    let t = F::from_u64(5);
    let scaled = RelaxedWitness::new(
        relaxed
            .columns()
            .clone()
            .map(|c| c.into_iter().map(|v| t * v).collect()),
        relaxed.error().to_vec(),
    );
    let public: Vec<F> = witness.public_inputs().iter().map(|v| t * *v).collect();
    assert!(circuit.is_relaxed_satisfied(&scaled, &public, t, &coefficients, 2));
}

#[test]
fn relaxed_relaxed_witness() {
    let w = RelaxedWitness::new([vec![F::ONE, F::ZERO]], vec![F::ZERO, F::ONE]);
    assert_eq!(w.error(), &[F::ZERO, F::ONE]);
}

#[test]
fn relaxed_relaxed_witness_new() {
    let w = RelaxedWitness::new([vec![F::ONE]], vec![F::ZERO]);
    assert_eq!(w.columns()[0], vec![F::ONE]);
}

#[test]
fn relaxed_relaxed_witness_columns() {
    let w = RelaxedWitness::new([vec![F::ONE], vec![F::ZERO]], vec![F::ZERO]);
    assert_eq!(w.columns().len(), 2);
}

#[test]
fn relaxed_relaxed_witness_error() {
    assert_eq!(
        RelaxedWitness::new([vec![F::ONE]], vec![F::ONE]).error(),
        &[F::ONE]
    );
}

#[test]
fn relaxed_relaxed_witness_embed() {
    let w = PlonkishWitness::new([vec![F::ONE, F::from_u64(2)]], vec![]);
    let r = RelaxedWitness::embed(&w);
    assert_eq!(r.columns(), w.columns());
    assert_eq!(r.error(), &[F::ZERO, F::ZERO]);
}

#[test]
fn relaxed_circuit_relaxed_row() {
    // a * b - c, with q_M = 1, q_O = -1.
    let mut b = PlonkishBuilder::<F, Vanilla, 3, 5, 1>::new();
    let one = F::ONE;
    let row = b.row::<Arithmetic, _>(
        [F::from_u64(2), F::from_u64(3), F::from_u64(7)],
        [F::ZERO, F::ZERO, -one, one, F::ZERO],
    );
    let (circuit, w) = b.build().unwrap();
    // a b - u c at u = 2: 6 - 14.
    let lhs = circuit.relaxed_row(w.columns(), row, F::from_u64(2), &[one], 2);
    assert_eq!(lhs, Some(F::from_u64(6) - F::from_u64(14)));
    // Degree 1 cannot hold a b.
    assert_eq!(circuit.relaxed_row(w.columns(), row, one, &[one], 1), None);
}

#[test]
fn relaxed_circuit_is_relaxed_satisfied() {
    let mut b = PlonkishBuilder::<F, Vanilla, 3, 5, 1>::new();
    let one = F::ONE;
    b.row::<Arithmetic, _>(
        [F::from_u64(2), F::from_u64(3), F::from_u64(7)],
        [F::ZERO, F::ZERO, -one, one, F::ZERO],
    );
    let (circuit, w) = b.build().unwrap();
    // 2 * 3 != 7, but the error column can absorb the difference.
    assert!(!circuit.is_satisfied(&w));
    let error = vec![-one, F::ZERO];
    let relaxed = RelaxedWitness::new(w.columns().clone(), error);
    assert!(circuit.is_relaxed_satisfied(&relaxed, &[], one, &[one], 2));
}

#[test]
fn vanilla_overview() {
    // x^3 + x + 5 = 35, with x = 3 private and 35 public.
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(3));
        let x2 = b.mul(&x, &x);
        let x3 = b.mul(&x2, &x);
        let sum = b.add(&x3, &x);
        let lhs = b.add_constant(&sum, F::from_u64(5));
        let rhs = b.public_input(F::from_u64(35));
        b.assert_equal(&lhs, &rhs);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    assert!(circuit.is_satisfied(&witness));
    assert_eq!(witness.public_inputs(), &[F::from_u64(35)]);
}

#[test]
fn vanilla_arithmetic() {
    let gate = &<Arithmetic as Gadget<F, 3, 5>>::constraints()[0];
    // Degree 2 in the witness (q_M a b).
    assert_eq!(gate.degree(), 2);
}

#[test]
fn vanilla_vanilla() {
    assert_eq!(<Vanilla as GateSet<F, 3, 5>>::SELECTORS, 1);
}

#[test]
fn vanilla_vanilla_circuit() {
    let (c, _, ()) = build::<F, _, _>(|_| ());
    let circuit: VanillaCircuit<F> = vanilla::circuit(&c).unwrap();
    assert_eq!(circuit.num_rows(), c.num_rows());
}

#[test]
fn vanilla_circuit() {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(2));
        let y = b.mul(&x, &x);
        let out = b.public_input(F::from_u64(4));
        b.assert_equal(&y, &out);
    });
    let circuit = vanilla::circuit(&c).unwrap();
    assert!(circuit.is_satisfied(&vanilla::witness(&w)));
}

#[test]
fn vanilla_witness() {
    let (_, w, ()) = build::<F, _, _>(|b| {
        b.public_input(F::from_u64(4));
    });
    assert_eq!(vanilla::witness(&w).public_inputs(), &[F::from_u64(4)]);
}
