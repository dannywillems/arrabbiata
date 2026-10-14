//! Tests of PlonK over a custom gate set: four gadgets over four witness
//! columns and one fixed column, of degree up to 6 with the selector, two
//! of them reading the next row (one a witness column, one the fixed
//! column).
//!
//! The circuits are random straight-line programs; every property is
//! checked on IPA over Pallas, and completeness also on KZG over BN254.
//! Boundary cases: an empty gate set (copies and public inputs only), and a
//! single witness column.

use arrabbiata_bn254::Fr;
use arrabbiata_field::Field;
use arrabbiata_pasta::Fq;
use arrabbiata_plonk::{
    Builder, Cell, Circuit, Gadget, GateSet, Member, Opened, PlonkError, Witness, ipa, kzg, prove,
    setup, verify,
};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};
use proptest::prelude::*;

const W: usize = 4;
const X: usize = 1;
const Q: usize = 4;

type E<F> = Expr<F, W, X>;

fn w<F: Field>(k: usize, r: Rotation) -> E<F> {
    E::witness(WitnessColumn::try_new(k).unwrap(), r)
}

fn f<F: Field>(r: Rotation) -> E<F> {
    E::fixed(FixedColumn::new::<0>(), r)
}

/// `d = a b`.
struct Mul;
impl<F: Field> Gadget<F, W, X> for Mul {
    const NAME: &'static str = "mul";
    fn constraints() -> Vec<E<F>> {
        vec![w(3, Rotation::CURR) - w(0, Rotation::CURR) * w(1, Rotation::CURR)]
    }
}

/// `b = a^5`: degree 5, 6 with the selector.
struct Pow5;
impl<F: Field> Gadget<F, W, X> for Pow5 {
    const NAME: &'static str = "pow5";
    fn constraints() -> Vec<E<F>> {
        let a = w::<F>(0, Rotation::CURR);
        let a2 = a.clone() * a.clone();
        vec![w(1, Rotation::CURR) - a2.clone() * a2 * a]
    }
}

/// `a' = a + b f`: the next row's `a` from this row.
struct Step;
impl<F: Field> Gadget<F, W, X> for Step {
    const NAME: &'static str = "step";
    fn constraints() -> Vec<E<F>> {
        vec![w(0, Rotation::NEXT) - w(0, Rotation::CURR) - w(1, Rotation::CURR) * f(Rotation::CURR)]
    }
}

/// `d = f'`: the fixed value of the next row.
struct Shift;
impl<F: Field> Gadget<F, W, X> for Shift {
    const NAME: &'static str = "shift";
    fn constraints() -> Vec<E<F>> {
        vec![w(3, Rotation::CURR) - f(Rotation::NEXT)]
    }
}

type Gates = (Mul, (Pow5, (Step, (Shift, ()))));

/// One row of a random program.
#[derive(Clone, Debug)]
enum Op {
    Mul,
    Pow5,
    Step,
    Shift,
    Plain,
}

#[derive(Clone, Debug)]
struct Row {
    op: Op,
    /// Values of the free cells.
    fresh: [u64; W],
    fixed: u64,
    /// Copy column `a` from an earlier output (index into the outputs so
    /// far), if any.
    copy: Option<usize>,
    public: bool,
}

fn arb_row() -> impl Strategy<Value = Row> {
    let op = prop_oneof![
        Just(Op::Mul),
        Just(Op::Pow5),
        Just(Op::Step),
        Just(Op::Shift),
        Just(Op::Plain),
    ];
    (
        op,
        any::<[u64; W]>(),
        any::<u64>(),
        prop::option::of(any::<usize>()),
        prop::bool::weighted(0.2),
    )
        .prop_map(|(op, fresh, fixed, copy, public)| Row {
            op,
            fresh,
            fixed,
            copy,
            public,
        })
}

fn arb_program() -> impl Strategy<Value = Vec<Row>> {
    prop::collection::vec(arb_row(), 1..12)
}

/// The circuit and a satisfying witness of `program`. A final plain row
/// receives the pending `a` of a last `Step` and is always public.
fn compile<F>(program: &[Row]) -> (Circuit<F, Gates, W, X, Q>, Witness<F, W>)
where
    F: Field,
{
    let num = |v: u64| F::from_u64(v);
    let fixed: Vec<F> = program.iter().map(|r| num(r.fixed)).collect();
    let mut b = Builder::<F, Gates, W, X, Q>::new();
    let mut outputs: Vec<(Cell<W>, F)> = Vec::new();
    let mut pending: Option<F> = None;
    let col = |k: usize| WitnessColumn::try_new(k).unwrap();
    for (r, row) in program.iter().enumerate() {
        let mut cells: [F; W] = row.fresh.map(num);
        let mut copy_from = None;
        if let Some(a) = pending.take() {
            cells[0] = a;
        } else if let Some(i) = row.copy
            && !outputs.is_empty()
        {
            let (cell, value) = outputs[i % outputs.len()];
            cells[0] = value;
            copy_from = Some(cell);
        }
        // The fixed value of the next row: the next program row, or the
        // final plain row (fixed 0).
        let rotated_fixed = fixed.get(r + 1).copied().unwrap_or(F::ZERO);
        let index = match row.op {
            Op::Mul => {
                cells[3] = cells[0] * cells[1];
                let i = b.row::<Mul, _>(cells, [fixed[r]]);
                outputs.push((Cell::new(col(3), i), cells[3]));
                i
            }
            Op::Pow5 => {
                cells[1] = cells[0].pow(5);
                let i = b.row::<Pow5, _>(cells, [fixed[r]]);
                outputs.push((Cell::new(col(1), i), cells[1]));
                i
            }
            Op::Step => {
                pending = Some(cells[0] + cells[1] * fixed[r]);
                b.row::<Step, _>(cells, [fixed[r]])
            }
            Op::Shift => {
                cells[3] = rotated_fixed;
                b.row::<Shift, _>(cells, [fixed[r]])
            }
            Op::Plain => b.plain_row(cells, [fixed[r]]),
        };
        if let Some(cell) = copy_from {
            b.copy(cell, Cell::new(col(0), index));
        }
        if row.public {
            b.public(index);
        }
    }
    let last = b.plain_row(
        [pending.unwrap_or(F::ONE), F::ZERO, F::ZERO, F::ZERO],
        [F::ZERO],
    );
    b.public(last);
    b.build().unwrap()
}

fn openings() -> usize {
    let next = Opened::<Fq, W, X, Q>::rotated_witness::<Fq, Gates>().len()
        + Opened::<Fq, W, X, Q>::rotated_fixed::<Fq, Gates>().len();
    2 * W + X + Q + 3 + next
}

#[test]
fn gate_set_shape() {
    assert_eq!(<Gates as GateSet<Fq, W, X>>::SELECTORS, Q);
    assert_eq!(<Gates as Member<Step, _>>::INDEX, 2);
    assert_eq!(<Gates as GateSet<Fq, W, X>>::name(3), Some("shift"));
    // Pow5 is degree 5, 6 with the selector, above W + 1 = 5.
    let (circuit, _) = compile::<Fq>(&[]);
    assert_eq!(circuit.quotient_factor(), 6);
    assert_eq!(
        Opened::<Fq, W, X, Q>::rotated_witness::<Fq, Gates>(),
        vec![(0, Rotation::NEXT)]
    );
    assert_eq!(
        Opened::<Fq, W, X, Q>::rotated_fixed::<Fq, Gates>(),
        vec![(0, Rotation::NEXT)]
    );
    assert_eq!(openings(), 18);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// The builder produces satisfying witnesses, and they prove.
    #[test]
    fn completeness_ipa(program in arb_program()) {
        let (circuit, witness) = compile::<Fq>(&program);
        prop_assert!(circuit.is_satisfied(&witness));
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof).is_ok());
    }

    /// The prover is exactly as strong as the relation: with one cell
    /// changed, `prove` succeeds iff the witness still satisfies the
    /// circuit, and then the proof verifies.
    #[test]
    fn prover_succeeds_iff_satisfied(
        program in arb_program(),
        column in 0usize..W,
        row in any::<usize>(),
        value in any::<u64>(),
    ) {
        let (circuit, witness) = compile::<Fq>(&program);
        let mut columns = witness.columns().clone();
        let row = row % circuit.num_rows();
        columns[column][row] = Fq::from_u64(value);
        let mut public = witness.public_inputs().to_vec();
        for (i, r) in circuit.public_rows().iter().enumerate() {
            public[i] = columns[0][*r];
        }
        let witness = Witness::new(columns, public);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        match prove(&pk, &pcs, ipa::pallas_transcript(), &witness) {
            Ok(proof) => {
                prop_assert!(circuit.is_satisfied(&witness));
                prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof).is_ok());
            }
            Err(e) => {
                prop_assert!(!circuit.is_satisfied(&witness));
                prop_assert_eq!(e, PlonkError::Unsatisfied);
            }
        }
    }

    /// Every claimed evaluation, at `zeta` and at `zeta w` (the rotated
    /// witness and fixed columns included), is checked by its opening.
    #[test]
    fn tampered_evaluation_is_rejected(program in arb_program(), index in 0usize..18) {
        let (circuit, witness) = compile::<Fq>(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        let value = proof.tail.tail.tail.head.iter_mut().nth(index).unwrap();
        *value = *value + Fq::ONE;
        prop_assert_eq!(
            verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// A proof is bound to its statement.
    #[test]
    fn wrong_public_input_is_rejected(program in arb_program(), delta in 1u64..1000) {
        let (circuit, witness) = compile::<Fq>(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        let mut wrong = witness.public_inputs().to_vec();
        let last = wrong.len() - 1;
        wrong[last] = wrong[last] + Fq::from_u64(delta);
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &wrong, proof).is_err());
    }

    /// A proof with a column read at the next row dropped has the wrong
    /// shape.
    #[test]
    fn missing_rotated_opening_is_rejected(program in arb_program()) {
        let (circuit, witness) = compile::<Fq>(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        proof.tail.tail.tail.head.witness_rotated.clear();
        prop_assert_eq!(
            verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Shape)
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    /// The same gate set over BN254, with KZG.
    #[test]
    fn completeness_kzg(program in arb_program()) {
        let (circuit, witness) = compile::<Fr>(&program);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0x5eed));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof).is_ok());
    }
}

/// A gadget that `Pow5` would satisfy on its rows, turned on where it does
/// not hold, is caught: the selector matters.
#[test]
fn selectors_route_constraints() {
    let x = Fq::from_u64(3);
    let mut b = Builder::<Fq, Gates, W, X, Q>::new();
    // Satisfies Pow5 (b = a^5) but not Mul (d = a b).
    b.row::<Mul, _>([x, x.pow(5), Fq::ZERO, Fq::ONE], [Fq::ZERO]);
    let (circuit, witness) = b.build().unwrap();
    assert!(!circuit.is_satisfied(&witness));
    let pcs = ipa::pallas(&circuit);
    let (pk, _) = setup(&circuit, &pcs).unwrap();
    assert_eq!(
        prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap_err(),
        PlonkError::Unsatisfied
    );
}

/// No gadget at all: the relation is copies and public inputs.
#[test]
fn empty_gate_set() {
    let mut b = Builder::<Fq, (), 2, 0, 0>::new();
    let (a, c) = (WitnessColumn::new::<0>(), WitnessColumn::new::<1>());
    let r0 = b.plain_row([Fq::from_u64(5), Fq::from_u64(5)], []);
    let r1 = b.plain_row([Fq::from_u64(5), Fq::ZERO], []);
    b.copy(Cell::new(a, r0), Cell::new(c, r0));
    b.copy(Cell::new(c, r0), Cell::new(a, r1));
    b.public(r1);
    let (circuit, witness) = b.build().unwrap();
    assert_eq!(circuit.quotient_factor(), 3);
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            &[Fq::from_u64(5)],
            proof.clone()
        )
        .is_ok()
    );
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            &[Fq::from_u64(6)],
            proof
        )
        .is_err()
    );
}

/// One witness column: the permutation argument over a single coset.
#[test]
fn single_column() {
    struct Square;
    impl Gadget<Fq, 1, 0> for Square {
        const NAME: &'static str = "square";
        fn constraints() -> Vec<Expr<Fq, 1, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let next = Expr::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
            vec![next - a.clone() * a]
        }
    }
    let mut b = Builder::<Fq, (Square, ()), 1, 0, 1>::new();
    let mut x = Fq::from_u64(2);
    let first = b.row::<Square, _>([x], []);
    for _ in 0..5 {
        x = x.square();
        b.row::<Square, _>([x], []);
    }
    let last = b.plain_row([x.square()], []);
    b.public(first);
    b.public(last);
    let (circuit, witness) = b.build().unwrap();
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            witness.public_inputs(),
            proof
        )
        .is_ok()
    );
}

/// Each of the four lists of rotated openings (evaluations and proofs, of
/// witness and of fixed columns) is checked for its length on its own.
#[test]
fn every_rotated_list_is_shape_checked() {
    let program = vec![
        Row {
            op: Op::Step,
            fresh: [1, 2, 3, 4],
            fixed: 5,
            copy: None,
            public: false,
        },
        Row {
            op: Op::Shift,
            fresh: [6, 7, 8, 9],
            fixed: 1,
            copy: None,
            public: true,
        },
    ];
    let (circuit, witness) = compile::<Fq>(&program);
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    for list in 0..4 {
        let mut p = proof.clone();
        match list {
            0 => drop(p.tail.tail.tail.head.witness_rotated.pop()),
            1 => drop(p.tail.tail.tail.head.fixed_rotated.pop()),
            2 => drop(p.tail.tail.tail.tail.head.value.witness_rotated.pop()),
            _ => drop(p.tail.tail.tail.tail.head.value.fixed_rotated.pop()),
        }
        assert_eq!(
            verify(
                &vk,
                &pcs,
                ipa::pallas_transcript(),
                witness.public_inputs(),
                p
            ),
            Err(PlonkError::Shape),
            "list {list}"
        );
    }
}

/// No witness column at all: a circuit of fixed columns proves (the
/// public-input and permutation terms are empty).
#[test]
fn no_witness_column() {
    let mut b = Builder::<Fq, (), 0, 1, 0>::new();
    for i in 0..5 {
        b.plain_row([], [Fq::from_u64(i)]);
    }
    let (circuit, witness) = b.build().unwrap();
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &[], proof).is_ok());
}

/// Two circuits have equal verifier keys iff they are equal, and a key
/// reports its circuit's size and public rows.
#[test]
fn verifier_keys_are_equal_iff_the_circuits_are() {
    let variant = |change: u8| {
        let mut b = Builder::<Fq, Gates, W, X, Q>::new();
        let f = if change == 1 { 9 } else { 1 };
        if change == 2 {
            b.row::<Pow5, _>([Fq::ONE; W], [Fq::from_u64(f)]);
        } else {
            b.row::<Mul, _>([Fq::ONE; W], [Fq::from_u64(f)]);
        }
        for _ in 0..4 {
            b.plain_row([Fq::ONE; W], [Fq::ZERO]);
        }
        let col = |k: usize| WitnessColumn::try_new(k).unwrap();
        if change == 3 {
            b.copy(Cell::new(col(0), 0), Cell::new(col(2), 3));
        }
        b.public(2);
        b.public(if change == 4 { 3 } else { 4 });
        b.build().unwrap().0
    };
    let reference = variant(0);
    let pcs = ipa::pallas(&reference);
    let (_, vk) = setup(&reference, &pcs).unwrap();
    assert_eq!(vk.value.num_rows(), reference.num_rows());
    assert_eq!(vk.value.public_rows(), &[2, 4]);
    assert_eq!(setup(&variant(0), &pcs).unwrap().1, vk);
    for change in 1..=4 {
        let other = setup(&variant(change), &pcs).unwrap().1;
        assert_ne!(other, vk, "change {change}");
        assert_ne!(other.value, vk.value, "change {change}");
    }
}
