//! Elliptic-curve arithmetic as PlonK gadgets, checked against the native
//! curve implementation.
//!
//! The gadgets work over any curve `y^2 = x^3 + b` whose base field is the
//! circuit's field: Vesta in a circuit over `Fq` (proved with IPA on
//! Pallas, which commits to `Fq` polynomials: the Pasta cycle), and
//! Grumpkin in a circuit over the BN254 scalar field (proved with KZG on
//! BN254: the BN254/Grumpkin cycle).
//!
//! Columns (`W = 8`): `x1, y1, x2, y2, x3, y3, lambda, inv`.
//!
//! - `OnCurve`: `y1^2 = x1^3 + b`;
//! - `Add` (incomplete, `x1 != x2`): `lambda (x2 - x1) = y2 - y1`,
//!   `x3 = lambda^2 - x1 - x2`, `y3 = lambda (x1 - x3) - y1`, and
//!   `(x2 - x1) inv = 1`, which makes `x1 = x2` unsatisfiable;
//! - `Double` (`y1 != 0`): `2 y1 lambda = 3 x1^2`, `x3 = lambda^2 - 2 x1`,
//!   `y3 = lambda (x1 - x3) - y1`, and `2 y1 inv = 1`.
//!
//! A scalar multiplication `[k] P` is double-and-add from the most
//! significant bit, with every intermediate point copied from the output
//! of a row to the input of the next, and `P`, `[k] P` public.

use core::marker::PhantomData;

use arrabbiata_bn254::Fr;
use arrabbiata_curve::{Curve, Projective};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_grumpkin::Grumpkin;
use arrabbiata_pasta::{Fq, Vesta};
use arrabbiata_plonk::{
    Builder, Cell, Circuit, Gadget, PlonkError, Witness, ipa, kzg, prove, setup, verify,
};
use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
use proptest::prelude::*;

fn arb_scalar<S: PrimeField>() -> impl Strategy<Value = S> {
    any::<[u64; 4]>().prop_map(|l| {
        let mut l = l;
        l[3] >>= 3;
        S::from_limbs(&l).unwrap_or(S::ONE)
    })
}

fn arb_point<C: Curve>() -> impl Strategy<Value = Projective<C>> {
    arb_scalar::<C::Scalar>().prop_map(|s| Projective::<C>::generator() * s)
}

// ------------------------------------------------------------- gadgets

const W: usize = 8;

fn v<F: Field>(k: usize) -> Expr<F, W, 0> {
    Expr::witness(WitnessColumn::try_new(k).unwrap(), Rotation::CURR)
}

fn c<F: Field>(x: u64) -> Expr<F, W, 0> {
    Expr::constant(F::from_u64(x))
}

/// `y1^2 = x1^3 + b`.
struct OnCurve<C>(PhantomData<C>);
impl<C: Curve> Gadget<C::Base, W, 0> for OnCurve<C> {
    const NAME: &'static str = "on curve";
    fn constraints() -> Vec<Expr<C::Base, W, 0>> {
        let (x, y) = (v(0), v(1));
        vec![y.clone() * y - x.clone() * x.clone() * x - Expr::constant(C::b())]
    }
}

/// Incomplete affine addition, `x1 != x2`.
struct Add<C>(PhantomData<C>);
impl<C: Curve> Gadget<C::Base, W, 0> for Add<C> {
    const NAME: &'static str = "add";
    fn constraints() -> Vec<Expr<C::Base, W, 0>> {
        let (x1, y1, x2, y2, x3, y3, l, inv) = (v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7));
        vec![
            l.clone() * (x2.clone() - x1.clone()) - (y2 - y1.clone()),
            l.clone() * l.clone() - x1.clone() - x2.clone() - x3.clone(),
            l * (x1.clone() - x3) - y1 - y3,
            (x2 - x1) * inv - c(1),
        ]
    }
}

/// Affine doubling, `y1 != 0`.
struct Double<C>(PhantomData<C>);
impl<C: Curve> Gadget<C::Base, W, 0> for Double<C> {
    const NAME: &'static str = "double";
    fn constraints() -> Vec<Expr<C::Base, W, 0>> {
        let (x1, y1, x3, y3, l, inv) = (v(0), v(1), v(4), v(5), v(6), v(7));
        vec![
            c(2) * y1.clone() * l.clone() - c(3) * x1.clone() * x1.clone(),
            l.clone() * l.clone() - c(2) * x1.clone() - x3.clone(),
            l * (x1 - x3) - y1.clone() - y3,
            c(2) * y1 * inv - c(1),
        ]
    }
}

type Gates<C> = (OnCurve<C>, (Add<C>, (Double<C>, ())));
type Ec<C> = Circuit<<C as Curve>::Base, Gates<C>, W, 0, 3>;

/// A point in affine coordinates.
type Point<C> = (<C as Curve>::Base, <C as Curve>::Base);

/// A circuit, its witness, and the point it computes.
type Computed<C> = (Ec<C>, Witness<<C as Curve>::Base, W>, Point<C>);

// ---------------------------------------------------- native witness rows

/// The witness row of `p + q` (`x1 != x2`), and the sum.
fn add_row<F: Field>((x1, y1): (F, F), (x2, y2): (F, F)) -> ([F; W], (F, F)) {
    let inv = (x2 - x1).inverse().unwrap_or(F::ZERO);
    let l = (y2 - y1) * inv;
    let x3 = l * l - x1 - x2;
    let y3 = l * (x1 - x3) - y1;
    ([x1, y1, x2, y2, x3, y3, l, inv], (x3, y3))
}

/// The witness row of `2 p` (`y1 != 0`), and the double.
fn double_row<F: Field>((x1, y1): (F, F)) -> ([F; W], (F, F)) {
    let two = F::ONE + F::ONE;
    let inv = (two * y1).inverse().unwrap_or(F::ZERO);
    let l = (two + F::ONE) * x1 * x1 * inv;
    let x3 = l * l - two * x1;
    let y3 = l * (x1 - x3) - y1;
    ([x1, y1, F::ZERO, F::ZERO, x3, y3, l, inv], (x3, y3))
}

fn affine<C: Curve>(p: &Projective<C>) -> (C::Base, C::Base) {
    let a = p.to_affine();
    (a.x().unwrap(), a.y().unwrap())
}

/// The circuit of `[k] P` (`k > 0`), its witness, and the result. With
/// `lie`, the published result is off by one in `y`.
fn scalar_mul<C: Curve>(p: (C::Base, C::Base), k: u64, lie: bool) -> Computed<C> {
    assert!(k > 0);
    let zero = C::Base::ZERO;
    let col = |i: usize| WitnessColumn::try_new(i).unwrap();
    let mut b = Builder::<C::Base, Gates<C>, W, 0, 3>::new();
    let base = b.row::<OnCurve<C>, _>([p.0, p.1, zero, zero, zero, zero, zero, zero], []);
    let mut acc = p;
    let mut acc_cell = (base, 0);
    for i in (0..63 - k.leading_zeros()).rev() {
        let (cells, d) = double_row(acc);
        let row = b.row::<Double<C>, _>(cells, []);
        b.copy(
            Cell::new(col(acc_cell.1), acc_cell.0),
            Cell::new(col(0), row),
        );
        b.copy(
            Cell::new(col(acc_cell.1 + 1), acc_cell.0),
            Cell::new(col(1), row),
        );
        acc = d;
        acc_cell = (row, 4);
        if (k >> i) & 1 == 1 {
            let (cells, s) = add_row(acc, p);
            let row = b.row::<Add<C>, _>(cells, []);
            b.copy(
                Cell::new(col(acc_cell.1), acc_cell.0),
                Cell::new(col(0), row),
            );
            b.copy(
                Cell::new(col(acc_cell.1 + 1), acc_cell.0),
                Cell::new(col(1), row),
            );
            b.copy(Cell::new(col(0), base), Cell::new(col(2), row));
            b.copy(Cell::new(col(1), base), Cell::new(col(3), row));
            acc = s;
            acc_cell = (row, 4);
        }
    }
    let shown_y = if lie { acc.1 + C::Base::ONE } else { acc.1 };
    let publish = |b: &mut Builder<C::Base, Gates<C>, W, 0, 3>,
                   value: C::Base,
                   from: Option<(usize, usize)>| {
        let mut cells = [zero; W];
        cells[0] = value;
        let row = b.plain_row(cells, []);
        if let Some((r, k)) = from {
            b.copy(Cell::new(col(k), r), Cell::new(col(0), row));
        }
        b.public(row);
    };
    publish(&mut b, p.0, Some((base, 0)));
    publish(&mut b, p.1, Some((base, 1)));
    publish(&mut b, acc.0, Some((acc_cell.0, acc_cell.1)));
    // A lie must not be copied from the honest cell, or the copy would
    // carry the truth: it is published from its own, unconstrained row
    // and tied to the result by a copy that the lie breaks.
    if lie {
        let mut cells = [zero; W];
        cells[0] = shown_y;
        let row = b.plain_row(cells, []);
        b.copy(
            Cell::new(col(acc_cell.1 + 1), acc_cell.0),
            Cell::new(col(0), row),
        );
        b.public(row);
    } else {
        publish(&mut b, acc.1, Some((acc_cell.0, acc_cell.1 + 1)));
    }
    let (circuit, witness) = b.build().unwrap();
    (circuit, witness, acc)
}

fn public<F: Field>(p: (F, F), r: (F, F)) -> Vec<F> {
    vec![p.0, p.1, r.0, r.1]
}

// --------------------------------------------------------------- vectors

/// Scalars exercising every bit length, all-ones, a single high bit, and
/// alternating bits.
const SCALARS: [u64; 12] = [
    1,
    2,
    3,
    5,
    7,
    0xff,
    0x100,
    0xffff,
    0x5555_5555_5555_5555,
    0xaaaa_aaaa_aaaa_aaaa,
    0x8000_0000_0000_0001,
    u64::MAX,
];

fn known_answers<C: Curve>() {
    let g = Projective::<C>::generator();
    let p = affine(&(g * C::Scalar::from_u64(0x1234_5678)));
    for k in SCALARS {
        let (circuit, witness, r) = scalar_mul::<C>(p, k, false);
        let native = affine(
            &(Projective::<C>::generator() * C::Scalar::from_u64(0x1234_5678)).mul_limbs(&[k]),
        );
        assert_eq!(r, native, "[{k:#x}] P");
        assert!(circuit.is_satisfied(&witness), "[{k:#x}] P");
    }
}

#[test]
fn vesta_known_answers() {
    known_answers::<Vesta>();
}

#[test]
fn grumpkin_known_answers() {
    known_answers::<Grumpkin>();
}

/// `[1] G` is `G`, `[2] G` its double, `[3] G` its triple, natively and
/// in the circuit; `[r - 1] G` is `-G`, natively.
#[test]
fn small_multiples_of_the_generator() {
    let g = Projective::<Vesta>::generator();
    let ga = affine(&g);
    for (k, expected) in [(1u64, g), (2, g.double()), (3, g.double() + g)] {
        let (_, _, r) = scalar_mul::<Vesta>(ga, k, false);
        assert_eq!(r, affine(&expected));
    }
    assert_eq!(g * (-<Vesta as Curve>::Scalar::ONE), -g);
}

#[test]
fn proved_on_vesta_with_ipa() {
    let p = affine(&Projective::<Vesta>::generator());
    for k in [1u64, 0x5a5a, u64::MAX] {
        let (circuit, witness, r) = scalar_mul::<Vesta>(p, k, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        assert!(
            verify(
                &vk,
                &pcs,
                ipa::pallas_transcript(),
                &public(p, r),
                proof.clone()
            )
            .is_ok()
        );
        let wrong = (r.0, r.1 + Fq::ONE);
        assert!(
            verify(
                &vk,
                &pcs,
                ipa::pallas_transcript(),
                &public(p, wrong),
                proof
            )
            .is_err()
        );
    }
}

#[test]
fn proved_on_grumpkin_with_kzg() {
    let p = affine(&Projective::<Grumpkin>::generator());
    for k in [3u64, 0xbeef] {
        let (circuit, witness, r) = scalar_mul::<Grumpkin>(p, k, false);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0x6e7));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        assert!(verify(&vk, &pcs, kzg::transcript(), &public(p, r), proof).is_ok());
    }
}

// ------------------------------------------------------------- negatives

/// The addition gadget cannot double: `x1 = x2` leaves no inverse.
#[test]
fn add_rejects_equal_abscissas() {
    let p = affine(&Projective::<Vesta>::generator());
    let (cells, _) = add_row(p, p);
    let mut b = Builder::<Fq, Gates<Vesta>, W, 0, 3>::new();
    b.row::<Add<Vesta>, _>(cells, []);
    let (circuit, witness) = b.build().unwrap();
    assert!(!circuit.is_satisfied(&witness));
}

/// A point off the curve fails the curve check.
#[test]
fn off_curve_point_is_rejected() {
    let (x, y) = affine(&Projective::<Vesta>::generator());
    let (circuit, witness, _) = scalar_mul::<Vesta>((x, y + Fq::ONE), 5, false);
    assert!(!circuit.is_satisfied(&witness));
    let pcs = ipa::pallas(&circuit);
    let (pk, _) = setup(&circuit, &pcs).unwrap();
    assert_eq!(
        prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap_err(),
        PlonkError::Unsatisfied
    );
}

/// A wrong claimed result has no proof.
#[test]
fn wrong_result_is_unsatisfied() {
    let p = affine(&Projective::<Vesta>::generator());
    let (circuit, witness, _) = scalar_mul::<Vesta>(p, 0x1234, true);
    assert!(!circuit.is_satisfied(&witness));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// On random points and scalars, the circuit computes the native
    /// scalar multiplication, and any single changed cell of a gadget row
    /// breaks it.
    #[test]
    fn circuit_computes_the_scalar_multiplication(
        p in arb_point::<Vesta>(),
        k in 1u64..,
        row in any::<usize>(),
        column in 0usize..W,
        delta in 1u64..1000,
    ) {
        let pa = affine(&p);
        let (circuit, witness, r) = scalar_mul::<Vesta>(pa, k, false);
        prop_assert_eq!(r, affine(&p.mul_limbs(&[k])));
        prop_assert!(circuit.is_satisfied(&witness));
        // Change a cell of a gadget row that its gadget reads.
        let rows: Vec<usize> = circuit
            .gadgets()
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_some())
            .map(|(i, _)| i)
            .collect();
        let row = rows[row % rows.len()];
        let reads = match circuit.gadgets()[row].map(|s| s.index()) {
            Some(0) => column < 2,
            Some(1) => true,
            _ => !(2..4).contains(&column),
        };
        let mut columns = witness.columns().clone();
        columns[column][row] = columns[column][row] + Fq::from_u64(delta);
        let broken = Witness::new(columns, witness.public_inputs().to_vec());
        if reads {
            prop_assert!(!circuit.is_satisfied(&broken));
        }
    }

    /// The same on Grumpkin.
    #[test]
    fn grumpkin_circuit_computes_the_scalar_multiplication(p in arb_point::<Grumpkin>(), k in 1u64..) {
        let (circuit, witness, r) = scalar_mul::<Grumpkin>(affine(&p), k, false);
        prop_assert_eq!(r, affine(&p.mul_limbs(&[k])));
        prop_assert!(circuit.is_satisfied(&witness));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3))]

    /// Random scalar multiplications are proved and verified.
    #[test]
    fn random_scalar_multiplications_are_proved(p in arb_point::<Vesta>(), k in 1u64..0x1_0000) {
        let pa = affine(&p);
        let (circuit, witness, r) = scalar_mul::<Vesta>(pa, k, false);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &public(pa, r), proof).is_ok());
    }
}
