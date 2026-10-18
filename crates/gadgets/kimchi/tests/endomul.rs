//! Kimchi's EndoMul gate against Kimchi.
//!
//! - Kimchi's test (`tests/endomul.rs`): 128 random bits `x`, from
//!   `P = 2 (phi(G) + G)` on Pallas' generator `G`, give `[k] G` for
//!   `k = ScalarChallenge(x).to_field(endo_r)`, and `n = x`; every cell of the
//!   32 rows is the one Kimchi's `endosclmul::gen_witness` writes, and the
//!   witness satisfies the circuit. `endo` and `endo_r` are Pallas' GLV
//!   `beta` and `lambda`, which are Kimchi's `endos`;
//! - one row determines its cells: changing any written cell breaks it;
//! - a proof of `Q = [k] T` for a private 128-bit challenge, with `T` and
//!   `Q` public.

use arrabbiata_curve::{Affine, Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    endomul::{BITS_PER_ROW, EndoMul, EndoMulGateOf, EndoMulOf, column},
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

/// The column variables of the gate: 14 cells of the row, 3 of the next.
const VARIABLES: usize = 17;
/// Kimchi's largest EndoMul degree.
const DEGREE: usize = 4;

/// The bits of Kimchi's test, and of a scalar challenge.
const CHALLENGE_BITS: usize = 128;

/// A multiplication on private inputs: `T` in columns 0-1 and `P` in
/// columns 4-5 of the first row, then the rows.
#[derive(Clone, Copy, Debug)]
struct Multiply<const B: usize>;

impl<const B: usize> TypedGadget<Fp> for Multiply<B> {
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Array<Scalar, B>);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "multiply";
    const ROWS: usize = <EndoMul<Fp, B> as TypedGadget<Fp>>::ROWS;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        (t, p, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let t =
            core::array::from_fn(|i| env.advice(Cell::new(column::XT + i, 0), [&t[i]], |[v]| v));
        let p =
            core::array::from_fn(|i| env.advice(Cell::new(column::XP + i, 0), [&p[i]], |[v]| v));
        let gadget = EndoMul::<Fp, B> {
            endo: Pallas::beta(),
        };
        env.call(&gadget, (t, p, bits))
    }
}

fn coordinates(p: &Affine<Pallas>) -> [Fp; 2] {
    [p.x().unwrap(), p.y().unwrap()]
}

/// `2 (phi(T) + T)`, Kimchi's starting accumulator.
fn start(t: &Projective<Pallas>) -> Projective<Pallas> {
    let sum = t.endomorphism() + *t;
    sum + sum
}

/// The low `B` bits of `x`, most significant first.
fn bits_msb<const B: usize>(x: u128) -> [Fp; B] {
    core::array::from_fn(|i| Fp::from_u64(((x >> (B - 1 - i)) & 1) as u64))
}

/// Kimchi's `ScalarChallenge::to_field_with_length(128, endo_r)`.
fn to_field(x: u128) -> Fq {
    let mut a = Fq::from_u64(2);
    let mut b = Fq::from_u64(2);
    for i in (0..CHALLENGE_BITS / 2).rev() {
        a = a + a;
        b = b + b;
        let s = if (x >> (2 * i)) & 1 == 0 {
            -Fq::ONE
        } else {
            Fq::ONE
        };
        if (x >> (2 * i + 1)) & 1 == 0 {
            b = b + s;
        } else {
            a = a + s;
        }
    }
    a * Pallas::lambda() + b
}

/// The rows Kimchi's `gen_witness` writes: 15 cells per row, then the
/// accumulator and `n` on the row after.
fn kimchi_rows(endo: Fp, t: [Fp; 2], p: [Fp; 2], bits: &[Fp]) -> Vec<[Fp; 15]> {
    let one = Fp::ONE;
    let mut rows = Vec::new();
    let mut acc = p;
    let mut n = Fp::ZERO;
    for chunk in bits.chunks(BITS_PER_ROW) {
        let [b1, b2, b3, b4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let [xt, yt] = t;
        let [xp, yp] = acc;
        let xq1 = (one + (endo - one) * b1) * xt;
        let yq1 = (b2 + b2 - one) * yt;
        let s1 = (yq1 - yp) * (xq1 - xp).inverse().unwrap();
        let s1_squared = s1 * s1;
        let s2 = (yp + yp) * (xp + xp + xq1 - s1_squared).inverse().unwrap() - s1;
        let xr = xq1 + s2 * s2 - s1_squared;
        let yr = (xp - xr) * s2 - yp;
        let xq2 = (one + (endo - one) * b3) * xt;
        let yq2 = (b4 + b4 - one) * yt;
        let s3 = (yq2 - yr) * (xq2 - xr).inverse().unwrap();
        let s3_squared = s3 * s3;
        let s4 = (yr + yr) * (xr + xr + xq2 - s3_squared).inverse().unwrap() - s3;
        let xs = xq2 + s4 * s4 - s3_squared;
        let ys = (xr - xs) * s4 - yr;
        let inv = ((xp - xr) * (xr - xs)).inverse().unwrap();
        let mut row = [Fp::ZERO; 15];
        row[0] = xt;
        row[1] = yt;
        row[2] = inv;
        row[4] = xp;
        row[5] = yp;
        row[6] = n;
        row[7] = xr;
        row[8] = yr;
        row[9] = s1;
        row[10] = s3;
        row[11..15].copy_from_slice(&[b1, b2, b3, b4]);
        rows.push(row);
        acc = [xs, ys];
        for b in [b1, b2, b3, b4] {
            n = n + n + b;
        }
    }
    let mut last = [Fp::ZERO; 15];
    last[4] = acc[0];
    last[5] = acc[1];
    last[6] = n;
    rows.push(last);
    rows
}

fn generator() -> Projective<Pallas> {
    Projective::<Pallas>::generator()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn kimchis_test(x in any::<u128>()) {
        let g = generator();
        let t = coordinates(&g.to_affine());
        let p = coordinates(&start(&g).to_affine());
        let bits = bits_msb::<CHALLENGE_BITS>(x);
        let input = (t, p, bits);
        let (compiled, (acc, n)) =
            witness::<Fp, _, VARIABLES, DEGREE>(&Multiply::<CHALLENGE_BITS>, LAYOUT, &input).unwrap();
        let low = Fp::from_u64(x as u64);
        let high = Fp::from_u64((x >> 64) as u64);
        let two_64 = Fp::from_u64(1 << 32) * Fp::from_u64(1 << 32);
        prop_assert_eq!(n, low + high * two_64);
        prop_assert_eq!(acc, coordinates(&(g * to_field(x)).to_affine()));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (row, expected) in kimchi_rows(Pallas::beta(), t, p, &bits).iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][row]);
            prop_assert_eq!(&found, expected, "row {}", row);
        }
    }

    #[test]
    fn a_row_determines_its_cells(k in 2u64..1_000_000, x in 0u128..16, row in 0usize..2, column in 0usize..15, delta in 1u64..1000) {
        let t = generator() * Fq::from_u64(k);
        let input = (coordinates(&t.to_affine()), coordinates(&start(&t).to_affine()), bits_msb::<BITS_PER_ROW>(x));
        let (compiled, _) =
            witness::<Fp, _, VARIABLES, DEGREE>(&Multiply::<BITS_PER_ROW>, LAYOUT, &input).unwrap();
        // The cells Kimchi leaves empty: column 3, and all but S, n' on the
        // row after.
        let written = if row == 0 { column != 3 } else { [column::XP, column::YP, column::N].contains(&column) };
        prop_assume!(written);
        let mut columns = compiled.columns.clone();
        columns[column][row] = columns[column][row] + Fp::from_u64(delta);
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }
}

/// `Q = [k] T` for a private 128-bit challenge, `T` and `Q` public.
#[derive(Clone, Copy, Debug)]
struct Scale;

impl TypedGadget<Fp> for Scale {
    type Input = (
        Array<Scalar, 2>,
        Array<Scalar, 2>,
        Array<Scalar, CHALLENGE_BITS>,
    );
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "scale";
    const ROWS: usize = <EndoMul<Fp, CHALLENGE_BITS> as TypedGadget<Fp>>::ROWS + 3;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        (t, p, bits): Of<Self::Input, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let t: [E::Var; 2] =
            core::array::from_fn(|i| env.advice(Cell::new(column::XT + i, 0), [&t[i]], |[v]| v));
        let p =
            core::array::from_fn(|i| env.advice(Cell::new(column::XP + i, 0), [&p[i]], |[v]| v));
        let gadget = EndoMul::<Fp, CHALLENGE_BITS> {
            endo: Pallas::beta(),
        };
        let ([x, y], _) = env.call(&gadget, (t.clone(), p, bits));
        // Public inputs are cells of column 0: xT already is; yT and Q go to
        // the rows after the result's.
        let rows = <EndoMul<Fp, CHALLENGE_BITS> as TypedGadget<Fp>>::ROWS;
        let yt = env.copy(Cell::new(0, rows + 1), &t[1]);
        let qx = env.copy(Cell::new(0, rows + 2), &x);
        let qy = env.copy(Cell::new(0, rows + 3), &y);
        for value in [&t[0], &yt, &qx, &qy] {
            env.public(value);
        }
        [qx, qy]
    }
}

fn ipa(size: usize) -> Ipa<Vesta, PoseidonSponge<Fp>> {
    let mut generators = vesta_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = pallas_sponge(IPA_DOMAIN);
    Ipa::<Vesta, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_scaling_is_proved() {
    let t = generator() * Fq::from_u64(11);
    let challenge: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
    let input = (
        coordinates(&t.to_affine()),
        coordinates(&start(&t).to_affine()),
        bits_msb::<CHALLENGE_BITS>(challenge),
    );
    let (compiled, q) = witness::<Fp, _, VARIABLES, DEGREE>(&Scale, LAYOUT, &input).unwrap();
    let expected = coordinates(&(t * to_field(challenge)).to_affine());
    assert_eq!(q, expected);
    let [xt, yt] = coordinates(&t.to_affine());
    let claimed = [xt, yt, expected[0], expected[1]];
    assert_eq!(compiled.public, claimed.to_vec());

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
    let [ax, ay] = coordinates(&(t * Fq::from_u64(3)).to_affine());
    let another = [xt, yt, ax, ay];
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

/// [`Multiply`] with the bits taken from the challenge's value.
#[derive(Clone, Copy, Debug)]
struct MultiplyOf<const B: usize>;

impl<const B: usize> TypedGadget<Fp> for MultiplyOf<B> {
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "multiply";
    const ROWS: usize = <EndoMulOf<Fp, B> as TypedGadget<Fp>>::ROWS;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        (t, p, x): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let t =
            core::array::from_fn(|i| env.advice(Cell::new(column::XT + i, 0), [&t[i]], |[v]| v));
        let p =
            core::array::from_fn(|i| env.advice(Cell::new(column::XP + i, 0), [&p[i]], |[v]| v));
        let gadget = EndoMulOf::<Fp, B> {
            endo: Pallas::beta(),
        };
        env.call(&gadget, (t, p, x))
    }
}

/// The element of `Fp` of the 128 bits `x`.
fn element(x: u128) -> Fp {
    Fp::from_limbs(&[x as u64, (x >> 64) as u64]).unwrap()
}

#[test]
fn endomul_of_is_endomul_of_the_bits() {
    // The bits taken from the challenge's value are the bits given: the
    // same cells, the same result, n the challenge, and [k] T.
    let x: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
    let t = generator() * Fq::from_u64(5);
    let input_t = coordinates(&t.to_affine());
    let input_p = coordinates(&start(&t).to_affine());
    let given = (input_t, input_p, bits_msb::<CHALLENGE_BITS>(x));
    let (bits, (q, n)) =
        witness::<Fp, _, VARIABLES, DEGREE>(&Multiply::<CHALLENGE_BITS>, LAYOUT, &given).unwrap();
    let (of, (q_of, n_of)) = witness::<Fp, _, VARIABLES, DEGREE>(
        &MultiplyOf::<CHALLENGE_BITS>,
        LAYOUT,
        &(input_t, input_p, element(x)),
    )
    .unwrap();
    assert_eq!(of.columns, bits.columns);
    assert_eq!((q_of, n_of), (q, n));
    assert_eq!(n_of, element(x));
    assert_eq!(q_of, coordinates(&(t * to_field(x)).to_affine()));
}

#[test]
fn endomul_gate_of() {
    let gate = EndoMulGateOf {
        endo: Pallas::beta(),
        top: 3,
    };
    assert_eq!(<EndoMulGateOf<Fp> as TypedGadget<Fp>>::ROWS, 1);
    assert_eq!(gate.top, 3);
}
