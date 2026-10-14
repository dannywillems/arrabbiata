//! PlonK with gadgets reading arbitrary rows: before (`-1`, `-2`), after
//! (`+1`, `+3`), and the fixed column at `-1`. Each such pair
//! `(column, k)` is opened at `zeta w^k`.
//!
//! - completeness on random recurrences, with IPA and KZG;
//! - every opening, the rotated ones included, is checked;
//! - on a cycle, a rotation by `k` and by `k + n` prove the same relation,
//!   and a proof for one verifies for the other only when they agree.

use arrabbiata_bn254::Fr;
use arrabbiata_field::Field;
use arrabbiata_pasta::Fq;
use arrabbiata_plonk::{
    Builder, Circuit, Gadget, Opened, PlonkError, Witness, ipa, kzg, prove, setup, verify,
};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};
use proptest::prelude::*;

const W: usize = 2;
const X: usize = 1;
const Q: usize = 2;

fn a<F: Field>(k: i32) -> Expr<F, W, X> {
    Expr::witness(WitnessColumn::new::<0>(), Rotation::new(k))
}

fn b<F: Field>(k: i32) -> Expr<F, W, X> {
    Expr::witness(WitnessColumn::new::<1>(), Rotation::new(k))
}

/// `a = a(-1) + a(-2) f(-1)`: a weighted Fibonacci recurrence.
struct Recurrence;
impl<F: Field> Gadget<F, W, X> for Recurrence {
    const NAME: &'static str = "recurrence";
    fn constraints() -> Vec<Expr<F, W, X>> {
        let f = Expr::fixed(FixedColumn::new::<0>(), Rotation::PREV);
        vec![a(0) - a(-1) - a(-2) * f]
    }
}

/// `b = a(+1) a(+3)`: a look-ahead product.
struct LookAhead;
impl<F: Field> Gadget<F, W, X> for LookAhead {
    const NAME: &'static str = "look-ahead";
    fn constraints() -> Vec<Expr<F, W, X>> {
        vec![b(0) - a(1) * a(3)]
    }
}

type Gates = (Recurrence, (LookAhead, ()));

/// A program: two seeds, a weight per row, and where to put look-aheads.
fn arb_program() -> impl Strategy<Value = (u64, u64, Vec<(u64, bool)>)> {
    (
        any::<u64>(),
        any::<u64>(),
        prop::collection::vec((0u64..5, any::<bool>()), 1..14),
    )
}

/// The circuit of a program: rows 0, 1 hold the seeds; each later row
/// applies the recurrence; a look-ahead sits on rows that have three rows
/// after them.
fn compile<F: Field>(
    (x0, x1, steps): &(u64, u64, Vec<(u64, bool)>),
) -> (Circuit<F, Gates, W, X, Q>, Witness<F, W>) {
    let len = 2 + steps.len();
    let mut av = vec![F::from_u64(*x0), F::from_u64(*x1)];
    let weights: Vec<F> = (0..len)
        .map(|i| {
            F::from_u64(if i == 0 {
                0
            } else {
                steps.get(i - 1).map_or(0, |s| s.0)
            })
        })
        .collect();
    for i in 2..len {
        let next = av[i - 1] + av[i - 2] * weights[i - 1];
        av.push(next);
    }
    let mut bld = Builder::<F, Gates, W, X, Q>::new();
    for i in 0..len {
        let look = i >= 2 && i + 3 < len && steps[i - 2].1;
        let bv = if look { av[i + 1] * av[i + 3] } else { F::ZERO };
        let cells = [av[i], bv];
        let fixed = [weights[i]];
        if i < 2 {
            bld.plain_row(cells, fixed);
        } else if look {
            // One gadget per row: a look-ahead row is not a recurrence row,
            // so its `a` must still follow the recurrence through the next
            // row's constraint, which reads only `a`.
            bld.row::<LookAhead, _>(cells, fixed);
        } else {
            bld.row::<Recurrence, _>(cells, fixed);
        }
    }
    bld.public(0);
    bld.public(len - 1);
    bld.build().unwrap()
}

#[test]
fn openings_of_the_gate_set() {
    type O = Opened<Fq, W, X, Q>;
    assert_eq!(
        O::rotated_witness::<Fq, Gates>(),
        vec![
            (0, Rotation::new(-2)),
            (0, Rotation::PREV),
            (0, Rotation::NEXT),
            (0, Rotation::new(3)),
        ]
    );
    assert_eq!(O::rotated_fixed::<Fq, Gates>(), vec![(0, Rotation::PREV)]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Completeness with IPA on Pallas.
    #[test]
    fn completeness_ipa(program in arb_program()) {
        let (circuit, witness) = compile::<Fq>(&program);
        prop_assert!(circuit.is_satisfied(&witness));
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof).is_ok());
    }

    /// Every evaluation, at `zeta` and at each `zeta w^k`, is checked by
    /// its opening (2 W + X + Q + 3 + 4 + 1 = 15 openings).
    #[test]
    fn tampered_evaluation_is_rejected(program in arb_program(), index in 0usize..15) {
        let (circuit, witness) = compile::<Fq>(&program);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        prop_assert_eq!(proof.tail.tail.tail.head.iter().count(), 15);
        let value = proof.tail.tail.tail.head.iter_mut().nth(index).unwrap();
        *value = *value + Fq::ONE;
        prop_assert_eq!(
            verify(&vk, &pcs, ipa::pallas_transcript(), witness.public_inputs(), proof),
            Err(PlonkError::Opening(index))
        );
    }

    /// With one value of `a` changed, `prove` succeeds iff the witness
    /// still satisfies the circuit.
    #[test]
    fn prover_succeeds_iff_satisfied(program in arb_program(), row in any::<usize>()) {
        let (circuit, witness) = compile::<Fq>(&program);
        let mut columns = witness.columns().clone();
        let len = 2 + program.2.len();
        let row = 2 + row % (len - 2);
        columns[0][row] = columns[0][row] + Fq::ONE;
        let mut public = witness.public_inputs().to_vec();
        public[1] = columns[0][len - 1];
        let broken = Witness::new(columns, public);
        // `prove` succeeds iff the changed witness still satisfies the
        // circuit (a look-ahead row's `a` may be read by no recurrence).
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        match prove(&pk, &pcs, ipa::pallas_transcript(), &broken) {
            Ok(proof) => {
                prop_assert!(circuit.is_satisfied(&broken));
                prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), broken.public_inputs(), proof).is_ok());
            }
            Err(e) => {
                prop_assert!(!circuit.is_satisfied(&broken));
                prop_assert_eq!(e, PlonkError::Unsatisfied);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    /// Completeness with KZG on BN254.
    #[test]
    fn completeness_kzg(program in arb_program()) {
        let (circuit, witness) = compile::<Fr>(&program);
        let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0x0123));
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof).is_ok());
    }
}

/// `b = a(K)` for an offset `K` fixed in the type.
struct Shifted<const K: i32>;
impl<const K: i32> Gadget<Fq, 2, 0> for Shifted<K> {
    const NAME: &'static str = "shifted";
    fn constraints() -> Vec<Expr<Fq, 2, 0>> {
        let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::new(K));
        let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
        vec![b - a]
    }
}

/// A cyclic circuit of 8 rows, every row reading `a` at offset `K`, with
/// `b_i = a_(i + shift mod 8)`; public `a_0` and `b_0`.
fn cyclic<const K: i32>(
    a: &[u64; 8],
    shift: i64,
) -> (Circuit<Fq, (Shifted<K>, ()), 2, 0, 1>, Witness<Fq, 2>) {
    let mut bld = Builder::<Fq, (Shifted<K>, ()), 2, 0, 1>::cyclic();
    for i in 0..8 {
        let j = (i as i64 + shift).rem_euclid(8) as usize;
        bld.row::<Shifted<K>, _>([Fq::from_u64(a[i]), Fq::from_u64(a[j])], []);
    }
    bld.public(0);
    bld.build().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    /// On a cycle of 8 rows, offsets congruent mod 8 define the same
    /// relation; both prove, and each verifies only its own proof type.
    #[test]
    fn congruent_rotations_prove_the_same_relation(a in any::<[u64; 8]>(), shift in -12i64..12) {
        let (c3, w3) = cyclic::<3>(&a, shift);
        let (c11, w11) = cyclic::<11>(&a, shift);
        let (cm5, wm5) = cyclic::<-5>(&a, shift);
        let s = c3.is_satisfied(&w3);
        prop_assert_eq!(s, c11.is_satisfied(&w11));
        prop_assert_eq!(s, cm5.is_satisfied(&wm5));
        if shift.rem_euclid(8) == 3 {
            prop_assert!(s);
        }
        if s {
            let pcs = ipa::pallas(&c11);
            let (pk, vk) = setup(&c11, &pcs).unwrap();
            let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &w11).unwrap();
            prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), w11.public_inputs(), proof).is_ok());
        }
    }
}

/// The verifier key absorbs the gate set's constraints, rotations included.
/// Offsets 3 and 11 define the same relation on 8 rows, with the same
/// columns and the same commitments, so the two proofs commit to the same
/// wires; but the absorbed gate sets differ, so the challenges differ, and
/// with them the evaluation point `zeta` and the evaluations. (The grand
/// product would not show it: with no copy, `sigma` is the identity and
/// `z = 1` whatever the challenges.)
#[test]
fn the_gate_structure_is_absorbed() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8];
    let (c3, w3) = cyclic::<3>(&a, 3);
    let (c11, w11) = cyclic::<11>(&a, 3);
    assert_eq!(w3, w11);
    let pcs = ipa::pallas(&c3);
    let (pk3, _) = setup(&c3, &pcs).unwrap();
    let (pk11, _) = setup(&c11, &pcs).unwrap();
    let p3 = prove(&pk3, &pcs, ipa::pallas_transcript(), &w3).unwrap();
    let p11 = prove(&pk11, &pcs, ipa::pallas_transcript(), &w11).unwrap();
    assert_eq!(p3.head.value, p11.head.value);
    assert_eq!(p3.tail.head.value, p11.tail.head.value);
    assert_ne!(
        p3.tail.tail.tail.head.witness,
        p11.tail.tail.tail.head.witness
    );
}
