//! Kimchi's 64-bit XOR (four Xor16 rows and a zero row) against Kimchi.
//!
//! - for random 64-bit words, the five rows are Kimchi's `xor::init_xor`,
//!   cell for cell: row `i` holds the words shifted by `16 i` and their four
//!   nybbles, the last row is zero; and `out = in1 XOR in2`;
//! - a wrong nybble of `out`, with its recombination kept exact, is caught
//!   by the lookup alone;
//! - a word of more than 64 bits is rejected by the zero row;
//! - a proof that a public `out` is the XOR of two private words.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, Scalar, TypedGadget, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

/// The column variables: every cell of a row, the first three of the next,
/// and the Generic row's coefficients.
const VARIABLES: usize = 32;
/// The Generic row's `c3 w0 w1`.
const DEGREE: usize = 3;

const BITS: usize = 64;
const ROWS: usize = BITS / 16;

/// Two private words in columns 0 and 1 of the first row, then the XOR.
#[derive(Clone, Debug)]
struct Words;

impl TypedGadget<Fp> for Words {
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "words";
    const ROWS: usize = ROWS + 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        words: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&words[i]], |[v]| v));
        env.call(&Xor::<Fp, BITS> { table: xor_table() }, cells)
    }
}

fn fp(v: u128) -> Fp {
    let two_64 = Fp::from_u64(1 << 32) * Fp::from_u64(1 << 32);
    Fp::from_u64(v as u64) + Fp::from_u64((v >> 64) as u64) * two_64
}

/// Kimchi's `init_xor` rows for 64-bit words.
fn kimchi_rows(in1: u64, in2: u64) -> Vec<[u64; 15]> {
    let out = in1 ^ in2;
    let mut rows = Vec::new();
    for i in 0..ROWS {
        let words = [in1, in2, out].map(|w| w >> (16 * i));
        let mut row = [0u64; 15];
        row[..3].copy_from_slice(&words);
        for (w, word) in words.iter().enumerate() {
            for k in 0..4 {
                row[3 + 4 * w + k] = (word >> (4 * k)) & 0xf;
            }
        }
        rows.push(row);
    }
    rows.push([0u64; 15]);
    rows
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn kimchis_layout(in1 in any::<u64>(), in2 in any::<u64>()) {
        let input = [in1, in2].map(Fp::from_u64);
        let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&Words, LAYOUT, &input).unwrap();
        prop_assert_eq!(out, Fp::from_u64(in1 ^ in2));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (r, expected) in kimchi_rows(in1, in2).iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][r]);
            prop_assert_eq!(found, expected.map(Fp::from_u64), "row {}", r);
        }
    }

    #[test]
    fn a_wider_word_is_rejected(low in any::<u64>(), high in 1u64..1000) {
        let input = [fp(u128::from(low) | (u128::from(high) << 64)), Fp::ONE];
        let found = witness::<Fp, _, VARIABLES, DEGREE>(&Words, LAYOUT, &input);
        prop_assert!(matches!(found, Err(Error::Unsatisfied { .. })), "{:?}", found.err());
    }
}

/// The lowest nybble of `out` raised by one, and `out` with it: the row
/// still recombines, and only the lookup `(in1_0, in2_0, out_0)` fails.
#[test]
fn only_the_lookup_catches_a_wrong_nybble() {
    let input = [Fp::from_u64(0x1230), Fp::from_u64(0x0450)];
    let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&Words, LAYOUT, &input).unwrap();
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
    let mut columns = compiled.columns.clone();
    columns[11][0] = columns[11][0] + Fp::ONE;
    columns[2][0] = columns[2][0] + Fp::ONE;
    assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
}

/// The XOR of two private words, made public.
#[derive(Clone, Debug)]
struct PublicXor;

impl TypedGadget<Fp> for PublicXor {
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "public-xor";
    const ROWS: usize = ROWS + 2;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        words: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let out = env.call(&Words, words);
        let public = env.copy(Cell::new(0, ROWS + 1), &out);
        env.public(&public);
        public
    }
}

fn ipa(size: usize) -> Ipa<Vesta, PoseidonSponge<Fp>> {
    let mut generators = vesta_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = pallas_sponge(IPA_DOMAIN);
    Ipa::<Vesta, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_xor_is_proved() {
    let (in1, in2) = (0xdead_beef_0123_4567_u64, 0x0f0f_f0f0_aaaa_5555_u64);
    let input = [Fp::from_u64(in1), Fp::from_u64(in2)];
    let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&PublicXor, LAYOUT, &input).unwrap();
    assert_eq!(out, Fp::from_u64(in1 ^ in2));
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
            &[out],
            proof.clone()
        )
        .is_ok()
    );
    assert!(
        verify(
            &verifier_key,
            &pcs,
            pallas_sponge(PROOF_DOMAIN),
            &[out + Fp::ONE],
            proof
        )
        .is_err()
    );
}
