//! Kimchi's CompleteAdd gate against Kimchi.
//!
//! - Kimchi's test (`tests/ec.rs`) has three kinds of rows: doublings
//!   `P + P`, additions `P + Q`, and `P + (-P)`. On each, the gadget writes
//!   the row Kimchi's test writes, cell for cell, the sum is the curve's,
//!   and the witness satisfies the circuit;
//! - every cell the constraints determine is caught when changed, and the
//!   cells they leave free are exactly the ones Kimchi's `zero_check` says
//!   (`x21_inv` when `x1 = x2`, `inf_z` when `y1 = y2`);
//! - a proof that four private points sum to a public point.

use arrabbiata_curve::{Affine, Projective};
use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    complete_add::{CompleteAdd, column},
};
use arrabbiata_pasta::{Fp, Pallas, Vesta};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

/// The column variables of the gate: columns 0-10 of the row.
const VARIABLES: usize = 11;
/// `same_x (2 s y1 - 3 x1^2)`.
const DEGREE: usize = 3;

/// The cells Kimchi's gate uses.
const CELLS: usize = 11;

/// One CompleteAdd row on private points.
#[derive(Clone, Copy, Debug)]
struct AddRow;

impl TypedGadget<Fp> for AddRow {
    type Input = Array<Scalar, 4>;
    type Output = Array<Scalar, 3>;
    const NAME: &'static str = "add-row";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<Scalar, 4>, E::Var>,
    ) -> Of<Array<Scalar, 3>, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&input[i]], |[v]| v));
        env.call(&CompleteAdd, cells)
    }
}

fn point(k: u64) -> Affine<Pallas> {
    Projective::<Pallas>::generator()
        .mul_limbs(&[k])
        .to_affine()
}

fn coordinates(p: &Affine<Pallas>) -> [Fp; 2] {
    [p.x().unwrap(), p.y().unwrap()]
}

fn negate(p: &Affine<Pallas>) -> Affine<Pallas> {
    Affine::new(p.x().unwrap(), -p.y().unwrap()).unwrap()
}

fn sum(p: &Affine<Pallas>, q: &Affine<Pallas>) -> Affine<Pallas> {
    (p.to_projective() + q.to_projective()).to_affine()
}

/// The row Kimchi's test writes for `p + q`.
fn kimchi_row(p: &Affine<Pallas>, q: &Affine<Pallas>) -> [Fp; CELLS] {
    let [x1, y1] = coordinates(p);
    let [x2, y2] = coordinates(q);
    let doubled = sum(p, p);
    let tangent = (x1 * x1 + x1 * x1 + x1 * x1) * (y1 + y1).inverse().unwrap();
    if x1 == x2 && y1 == y2 {
        let [x3, y3] = coordinates(&doubled);
        [
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            Fp::ZERO,
            Fp::ONE,
            tangent,
            Fp::ZERO,
            Fp::ZERO,
        ]
    } else if x1 == x2 {
        // P + (-P): the sum cells hold the doubling of P.
        let [x3, y3] = coordinates(&doubled);
        let inf_z = (y2 - y1).inverse().unwrap();
        [
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            Fp::ONE,
            Fp::ONE,
            tangent,
            inf_z,
            Fp::ZERO,
        ]
    } else {
        let [x3, y3] = coordinates(&sum(p, q));
        let chord = (y2 - y1) * (x2 - x1).inverse().unwrap();
        let x21_inv = (x2 - x1).inverse().unwrap();
        [
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            Fp::ZERO,
            Fp::ZERO,
            chord,
            Fp::ZERO,
            x21_inv,
        ]
    }
}

/// The three kinds of rows of Kimchi's test.
fn cases(a: u64, b: u64) -> [(Affine<Pallas>, Affine<Pallas>); 3] {
    let p = point(a);
    let q = point(b);
    [(p, p), (p, q), (p, negate(&p))]
}

/// The cells the constraints leave free for `p + q`.
fn free(p: &Affine<Pallas>, q: &Affine<Pallas>) -> Vec<usize> {
    let mut free = Vec::new();
    if p.x() == q.x() {
        free.push(column::X21_INV);
    }
    if p.y() == q.y() {
        free.push(column::INF_Z);
    }
    free
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn kimchis_rows(a in 1u64..u64::MAX, b in 1u64..u64::MAX) {
        prop_assume!(a != b && a.wrapping_add(b) != 0);
        for (p, q) in cases(a, b) {
            let input = [coordinates(&p), coordinates(&q)].concat();
            let input: [Fp; 4] = input.try_into().unwrap();
            let (compiled, [x3, y3, inf]) = witness::<Fp, _, VARIABLES, DEGREE>(&AddRow, LAYOUT, &input).unwrap();
            prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
            let row: [Fp; CELLS] = core::array::from_fn(|c| compiled.columns[c][0]);
            prop_assert_eq!(row, kimchi_row(&p, &q));
            let unused: Vec<Fp> = (CELLS..LAYOUT.witness).map(|c| compiled.columns[c][0]).collect();
            prop_assert!(unused.iter().all(|v| *v == Fp::ZERO));
            let s = sum(&p, &q);
            if s.is_identity() {
                prop_assert_eq!(inf, Fp::ONE);
            } else {
                prop_assert_eq!(([x3, y3], inf), (coordinates(&s), Fp::ZERO));
            }
        }
    }

    #[test]
    fn exactly_the_free_cells_are_free(a in 1u64..u64::MAX, b in 1u64..u64::MAX, delta in 1u64..1000) {
        prop_assume!(a != b && a.wrapping_add(b) != 0);
        for (p, q) in cases(a, b) {
            let input: [Fp; 4] = [coordinates(&p), coordinates(&q)].concat().try_into().unwrap();
            let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&AddRow, LAYOUT, &input).unwrap();
            let free = free(&p, &q);
            for c in column::X3..CELLS {
                let mut columns = compiled.columns.clone();
                columns[c][0] = columns[c][0] + Fp::from_u64(delta);
                let satisfied = compiled.circuit.is_satisfied(&columns, &compiled.public);
                prop_assert_eq!(satisfied, free.contains(&c), "column {}", c);
            }
        }
    }
}

/// A CompleteAdd whose result is not the point at infinity.
#[derive(Clone, Copy, Debug)]
struct FiniteAdd;

impl TypedGadget<Fp> for FiniteAdd {
    type Input = Array<Scalar, 4>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "finite-add";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<Scalar, 4>, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let [x3, y3, inf] = env.call(&CompleteAdd, input);
        env.assert_zero(inf);
        [x3, y3]
    }
}

/// The sum of four private points, made public.
#[derive(Clone, Copy, Debug)]
struct SumOfFour;

impl TypedGadget<Fp> for SumOfFour {
    type Input = Array<Scalar, 8>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "sum-of-four";
    const ROWS: usize = 5;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        points: Of<Array<Scalar, 8>, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let mut x = env.advice(Cell::new(column::X1, 0), [&points[0]], |[v]| v);
        let mut y = env.advice(Cell::new(column::Y1, 0), [&points[1]], |[v]| v);
        for row in 0..3 {
            let px = env.advice(Cell::new(column::X2, row), [&points[2 * row + 2]], |[v]| v);
            let py = env.advice(Cell::new(column::Y2, row), [&points[2 * row + 3]], |[v]| v);
            // No partial sum is the point at infinity.
            [x, y] = env.call(&FiniteAdd, [x, y, px, py]);
        }
        let public_x = env.copy(Cell::new(0, 3), &x);
        let public_y = env.copy(Cell::new(0, 4), &y);
        env.public(&public_x);
        env.public(&public_y);
        [public_x, public_y]
    }
}

fn ipa(size: usize) -> Ipa<Vesta, PoseidonSponge<Fp>> {
    let mut generators = vesta_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = pallas_sponge(IPA_DOMAIN);
    Ipa::<Vesta, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_sum_is_proved() {
    let points = [point(3), point(5), point(5), point(11)];
    let input: [Fp; 8] = points
        .iter()
        .flat_map(coordinates)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let (compiled, total) =
        witness::<Fp, _, VARIABLES, DEGREE>(&SumOfFour, LAYOUT, &input).unwrap();
    let expected = point(3 + 5 + 5 + 11);
    assert_eq!(total, coordinates(&expected));

    let pcs = ipa(max_coefficients(&compiled.circuit));
    let (prover_key, verifier_key) = setup(&compiled.circuit, &pcs).unwrap();
    let (proof, _) = prove(
        &prover_key,
        &pcs,
        pallas_sponge(PROOF_DOMAIN),
        &compiled.public,
        &compiled.columns,
    )
    .unwrap();
    let claimed = coordinates(&expected);
    assert!(
        verify(
            &verifier_key,
            &pcs,
            pallas_sponge(PROOF_DOMAIN),
            &claimed,
            proof.clone()
        )
        .is_ok()
    );
    let another = coordinates(&point(25));
    assert!(
        verify(
            &verifier_key,
            &pcs,
            pallas_sponge(PROOF_DOMAIN),
            &another,
            proof
        )
        .is_err()
    );
}
