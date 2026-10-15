//! Kimchi's VarBaseMul gate against Kimchi.
//!
//! - Kimchi's test (`tests/varbasemul.rs`): the 255 bits of a random `x`,
//!   most significant first, from the accumulator `2 G` on the base `G`
//!   (Pallas' generator), give `(1 + 2^255 + 2 x) G` and `n = x`; every cell
//!   of the 102 rows is the one Kimchi's `varbasemul::witness` writes, and
//!   the witness satisfies the circuit;
//! - one gate determines its two rows: changing any written cell breaks
//!   it;
//! - a proof of `Q = (1 + 2^20 + 2 x) T` for a private 20-bit `x`, with
//!   `T` and `Q` public.

use arrabbiata_curve::{Affine, Projective};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    varbasemul::{ACCUMULATORS, BITS, BITS_PER_GATE, N, N_NEXT, SLOPES, VarBaseMul},
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

/// The column variables of the gate: the first row's 14 cells and the
/// second row's 12.
const VARIABLES: usize = 26;
/// `u^2`, with `u = 2 yI - t s` of degree 3.
const DEGREE: usize = 6;

/// The bits of Kimchi's test: those of an element of `Fp`.
const KIMCHI_BITS: usize = 255;

/// A scalar multiplication on private inputs: `T` and `A` in columns 0-3
/// of the first row, then the gates.
#[derive(Clone, Copy, Debug)]
struct Multiply<const B: usize>;

impl<const B: usize> TypedGadget<Fp> for Multiply<B> {
    type Input = (Array<2>, Array<2>, Array<B>);
    type Output = (Array<2>, Scalar);
    const NAME: &'static str = "multiply";
    const ROWS: usize = <VarBaseMul<B> as TypedGadget<Fp>>::ROWS;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        (t, a, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let t = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&t[i]], |[v]| v));
        let a = core::array::from_fn(|i| env.advice(Cell::new(2 + i, 0), [&a[i]], |[v]| v));
        env.call(&VarBaseMul::<B>, (t, a, bits))
    }
}

fn coordinates(p: &Affine<Pallas>) -> [Fp; 2] {
    [p.x().unwrap(), p.y().unwrap()]
}

/// The bits of `x`, most significant first.
fn bits_msb<const B: usize>(x: &Fp) -> [Fp; B] {
    core::array::from_fn(|i| {
        let k = B - 1 - i;
        let bit = (x.limb(k / 64) >> (k % 64)) & 1;
        Fp::from_u64(bit)
    })
}

/// `[k] P` for `k` given by little-endian limbs.
fn mul(p: &Affine<Pallas>, limbs: &[u64]) -> Affine<Pallas> {
    p.to_projective().mul_limbs(limbs).to_affine()
}

/// `1 + 2^B + 2 x` as little-endian limbs (of the scalar field, which is
/// larger than `x`'s field).
fn shifted<const B: usize>(x: &Fp) -> [u64; 5] {
    let mut limbs = [0u64; 5];
    let mut add = |position: usize, bits: u64| {
        let mut carry = bits;
        let mut i = position;
        while carry != 0 {
            let (sum, overflow) = limbs[i].overflowing_add(carry);
            limbs[i] = sum;
            carry = u64::from(overflow);
            i += 1;
        }
    };
    // 2 x, limb by limb.
    for i in 0..4 {
        let v = x.limb(i);
        add(i, v << 1);
        add(i + 1, v >> 63);
    }
    add(B / 64, 1 << (B % 64));
    add(0, 1);
    limbs
}

/// The rows Kimchi's `varbasemul::witness` writes, for the base `t`, the
/// accumulator `a` and the bits.
fn kimchi_rows(t: [Fp; 2], a: [Fp; 2], bits: &[Fp]) -> Vec<[[Fp; 15]; 2]> {
    let mut rows = Vec::new();
    let mut acc = a;
    let mut n = Fp::ZERO;
    for chunk in bits.chunks(BITS_PER_GATE) {
        let mut row = [[Fp::ZERO; 15]; 2];
        row[0][0] = t[0];
        row[0][1] = t[1];
        row[0][N] = n;
        for (k, b) in chunk.iter().enumerate() {
            let (r, c) = ACCUMULATORS[k];
            row[r][c] = acc[0];
            row[r][c + 1] = acc[1];
            row[1][BITS[k]] = *b;
            n = n + n + *b;
            let sign = *b + *b - Fp::ONE;
            let s1 = (acc[1] - sign * t[1]) * (acc[0] - t[0]).inverse().unwrap();
            let s1_squared = s1 * s1;
            let s2 =
                (acc[1] + acc[1]) * (acc[0] + acc[0] + t[0] - s1_squared).inverse().unwrap() - s1;
            let xo = t[0] + s2 * s2 - s1_squared;
            let yo = (acc[0] - xo) * s2 - acc[1];
            row[1][SLOPES[k]] = s1;
            acc = [xo, yo];
        }
        let (r, c) = ACCUMULATORS[BITS_PER_GATE];
        row[r][c] = acc[0];
        row[r][c + 1] = acc[1];
        row[0][N_NEXT] = n;
        rows.push(row);
    }
    rows
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn kimchis_test(limbs in prop::array::uniform4(any::<u64>())) {
        // A random element of Fp.
        let x = Fp::from_limbs(&limbs).unwrap_or(Fp::from_u64(limbs[0]));
        let g = Projective::<Pallas>::generator().to_affine();
        let base = coordinates(&g);
        let start = coordinates(&mul(&g, &[2]));
        let bits = bits_msb::<KIMCHI_BITS>(&x);
        let input = (base, start, bits);
        let (compiled, (acc, n)) =
            witness::<Fp, _, VARIABLES, DEGREE>(&Multiply::<KIMCHI_BITS>, LAYOUT, &input).unwrap();
        prop_assert_eq!(n, x);
        prop_assert_eq!(acc, coordinates(&mul(&g, &shifted::<KIMCHI_BITS>(&x))));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (i, row) in kimchi_rows(base, start, &bits).iter().enumerate() {
            for (r, expected) in row.iter().enumerate() {
                let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][2 * i + r]);
                prop_assert_eq!(&found, expected, "gate {}, row {}", i, r);
            }
        }
    }

    #[test]
    fn a_gate_determines_its_rows(k in 2u64..1_000_000, x in 0u64..32, row in 0usize..2, column in 0usize..15, delta in 1u64..1000) {
        let t = mul(&Projective::<Pallas>::generator().to_affine(), &[k]);
        let start = mul(&t, &[2]);
        let bits = bits_msb::<BITS_PER_GATE>(&Fp::from_u64(x));
        let input = (coordinates(&t), coordinates(&start), bits);
        let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&Multiply::<BITS_PER_GATE>, LAYOUT, &input).unwrap();
        // The cells Kimchi leaves empty.
        let empty = (row == 0 && column == 6) || (row == 1 && column > SLOPES[BITS_PER_GATE - 1]);
        prop_assume!(!empty);
        let mut columns = compiled.columns.clone();
        columns[column][row] = columns[column][row] + Fp::from_u64(delta);
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }
}

/// `Q = (1 + 2^20 + 2 x) T` for a private `x`, with `T` and `Q` public.
#[derive(Clone, Copy, Debug)]
struct Shifted;

const PROOF_BITS: usize = 20;

impl TypedGadget<Fp> for Shifted {
    type Input = (Array<2>, Array<2>, Array<PROOF_BITS>);
    type Output = Array<2>;
    const NAME: &'static str = "shifted";
    const ROWS: usize = <VarBaseMul<PROOF_BITS> as TypedGadget<Fp>>::ROWS + 3;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        (t, a, bits): Of<Self::Input, E::Var>,
    ) -> Of<Array<2>, E::Var> {
        let t: [E::Var; 2] =
            core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&t[i]], |[v]| v));
        let a = core::array::from_fn(|i| env.advice(Cell::new(2 + i, 0), [&a[i]], |[v]| v));
        let ([x, y], _) = env.call(&VarBaseMul::<PROOF_BITS>, (t.clone(), a, bits));
        // Public inputs are cells of column 0: xT already is; yT and Q are
        // copied to the rows after the gates.
        let rows = <VarBaseMul<PROOF_BITS> as TypedGadget<Fp>>::ROWS;
        let yt = env.copy(Cell::new(0, rows), &t[1]);
        let qx = env.copy(Cell::new(0, rows + 1), &x);
        let qy = env.copy(Cell::new(0, rows + 2), &y);
        for p in [&t[0], &yt, &qx, &qy] {
            env.public(p);
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
fn a_multiplication_is_proved() {
    let t = mul(&Projective::<Pallas>::generator().to_affine(), &[7]);
    let x = Fp::from_u64(0b1011_0110_1110_0101_1001);
    let input = (
        coordinates(&t),
        coordinates(&mul(&t, &[2])),
        bits_msb::<PROOF_BITS>(&x),
    );
    let (compiled, result) = witness::<Fp, _, VARIABLES, DEGREE>(&Shifted, LAYOUT, &input).unwrap();
    let expected = coordinates(&mul(&t, &shifted::<PROOF_BITS>(&x)));
    assert_eq!(result, expected);

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
    let [xt, yt] = coordinates(&t);
    let claimed = [xt, yt, expected[0], expected[1]];
    assert_eq!(compiled.public, claimed.to_vec());
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
    let [ax, ay] = coordinates(&mul(&t, &[3]));
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
