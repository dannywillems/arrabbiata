//! Kimchi's EndoMulScalar gate against Kimchi.
//!
//! - Kimchi's test (`tests/endomul_scalar.rs`): for random 128-bit `x`,
//!   over `Fp` with Vesta's `endo_r` (our `Vesta::lambda`), the rows give
//!   `n = x` and `a lambda + b = ScalarChallenge(x).to_field(lambda)`; every
//!   cell of the 8 rows is the one Kimchi's `endomul_scalar::gen_witness`
//!   writes, and the witness satisfies the circuit;
//! - the scalar is the one EndoMul multiplies by: over `Fq` with Pallas'
//!   `lambda`, `a lambda + b` is the `k` of `tests/endomul.rs`;
//! - a row determines its cells, and a crumb outside `{0, 1, 2, 3}` is
//!   rejected;
//! - a proof that a public `(a, b)` comes from a private challenge.

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget, evaluate, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    endomul_scalar::{CRUMBS_PER_ROW, EndoMulScalar},
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

/// The column variables of the gate: columns 0-13 of the row.
const VARIABLES: usize = 14;
/// The range check of a crumb.
const DEGREE: usize = 4;

/// The bits of a scalar challenge, and its crumbs.
const CHALLENGE_BITS: usize = 128;
const CRUMBS: usize = CHALLENGE_BITS / 2;

/// The crumbs of `x`, most significant first: crumb `i` is
/// `2 x_(2j+1) + x_(2j)` for `j = 63 - i`.
fn crumbs<F: Field>(x: u128) -> [F; CRUMBS] {
    core::array::from_fn(|i| {
        let j = CRUMBS - 1 - i;
        F::from_u64(((x >> (2 * j)) & 3) as u64)
    })
}

/// Kimchi's `ScalarChallenge::to_field_with_length(128, lambda)`.
fn to_field<F: Field>(x: u128, lambda: F) -> F {
    let mut a = F::from_u64(2);
    let mut b = F::from_u64(2);
    for i in (0..CRUMBS).rev() {
        a = a + a;
        b = b + b;
        let s = if (x >> (2 * i)) & 1 == 0 {
            -F::ONE
        } else {
            F::ONE
        };
        if (x >> (2 * i + 1)) & 1 == 0 {
            b = b + s;
        } else {
            a = a + s;
        }
    }
    a * lambda + b
}

fn as_field<F: Field>(x: u128) -> F {
    let two_64 = F::from_u64(1 << 32) * F::from_u64(1 << 32);
    F::from_u64(x as u64) + F::from_u64((x >> 64) as u64) * two_64
}

/// The rows Kimchi's `gen_witness` writes.
fn kimchi_rows(x: u128) -> Vec<[Fp; 15]> {
    let mut rows = Vec::new();
    let one = Fp::ONE;
    let (mut n, mut a, mut b) = (Fp::ZERO, Fp::from_u64(2), Fp::from_u64(2));
    let crumbs: [Fp; CRUMBS] = crumbs(x);
    for chunk in crumbs.chunks(CRUMBS_PER_ROW) {
        let mut row = [Fp::ZERO; 15];
        row[0] = n;
        row[2] = a;
        row[3] = b;
        for (j, crumb) in chunk.iter().enumerate() {
            row[6 + j] = *crumb;
            let value = (0..4u64).find(|v| Fp::from_u64(*v) == *crumb).unwrap();
            let (b1, b0) = (value >> 1 == 1, value & 1 == 1);
            a = a + a;
            b = b + b;
            let s = if b0 { one } else { -one };
            if b1 {
                a = a + s;
            } else {
                b = b + s;
            }
            n = n + n + n + n + *crumb;
        }
        row[1] = n;
        row[4] = a;
        row[5] = b;
        rows.push(row);
    }
    rows
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn kimchis_test(x in any::<u128>()) {
        let input: [Fp; CRUMBS] = crumbs(x);
        let (compiled, [n, a, b]) = witness::<Fp, _, VARIABLES, DEGREE>(&EndoMulScalar::<CRUMBS>, LAYOUT, &input).unwrap();
        prop_assert_eq!(n, as_field::<Fp>(x));
        let lambda = Vesta::lambda();
        prop_assert_eq!(a * lambda + b, to_field(x, lambda));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (row, expected) in kimchi_rows(x).iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][row]);
            prop_assert_eq!(&found, expected, "row {}", row);
        }
    }

    /// Over `Fq`, with Pallas' `lambda`: the scalar EndoMul multiplies by.
    #[test]
    fn the_scalar_of_endomul(x in any::<u128>()) {
        let input: [Fq; CRUMBS] = crumbs(x);
        let [_, a, b] = evaluate(&EndoMulScalar::<CRUMBS>, LAYOUT, &input).unwrap();
        let lambda = Pallas::lambda();
        prop_assert_eq!(a * lambda + b, to_field(x, lambda));
    }

    #[test]
    fn a_row_determines_its_cells(x in 0u128..(1 << 16), column in 0usize..14, delta in 1u64..1000) {
        let input: [Fp; CRUMBS_PER_ROW] = core::array::from_fn(|i| Fp::from_u64(((x >> (2 * (7 - i))) & 3) as u64));
        let (compiled, _) =
            witness::<Fp, _, VARIABLES, DEGREE>(&EndoMulScalar::<CRUMBS_PER_ROW>, LAYOUT, &input).unwrap();
        let mut columns = compiled.columns.clone();
        columns[column][0] = columns[column][0] + Fp::from_u64(delta);
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }
}

#[test]
fn a_crumb_out_of_range_is_rejected() {
    let mut input = [Fp::ZERO; CRUMBS_PER_ROW];
    input[3] = Fp::from_u64(4);
    let found =
        witness::<Fp, _, VARIABLES, DEGREE>(&EndoMulScalar::<CRUMBS_PER_ROW>, LAYOUT, &input);
    assert!(matches!(
        found,
        Err(Error::Unsatisfied {
            gadget: "kimchi-endomul-scalar",
            ..
        })
    ));
}

/// The pair `(a, b)` of a private challenge, made public.
#[derive(Clone, Copy, Debug)]
struct Challenge;

impl TypedGadget<Fp> for Challenge {
    type Input = Array<Scalar, CRUMBS>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "challenge";
    const ROWS: usize = <EndoMulScalar<CRUMBS> as TypedGadget<Fp>>::ROWS + 2;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        crumbs: Of<Array<Scalar, CRUMBS>, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let [_, a, b] = env.call(&EndoMulScalar::<CRUMBS>, crumbs);
        let rows = <EndoMulScalar<CRUMBS> as TypedGadget<Fp>>::ROWS;
        let a = env.copy(Cell::new(0, rows), &a);
        let b = env.copy(Cell::new(0, rows + 1), &b);
        env.public(&a);
        env.public(&b);
        [a, b]
    }
}

fn ipa(size: usize) -> Ipa<Vesta, PoseidonSponge<Fp>> {
    let mut generators = vesta_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = pallas_sponge(IPA_DOMAIN);
    Ipa::<Vesta, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_challenge_is_proved() {
    let x: u128 = 0x0f1e_2d3c_4b5a_6978_8796_a5b4_c3d2_e1f0;
    let input: [Fp; CRUMBS] = crumbs(x);
    let (compiled, [a, b]) =
        witness::<Fp, _, VARIABLES, DEGREE>(&Challenge, LAYOUT, &input).unwrap();
    let lambda = Vesta::lambda();
    assert_eq!(a * lambda + b, to_field(x, lambda));
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
    let claimed = [a, b];
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
    let another = [a, b + Fp::ONE];
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
