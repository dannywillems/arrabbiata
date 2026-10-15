//! Kimchi's Poseidon gate against Kimchi.
//!
//! - the gadget's permutation is `arrabbiata_poseidon::kimchi`'s, which
//!   reproduces proof-systems' `kimchi.json`; the hash of two elements (the
//!   sponge absorbs them into `[a, b, 0]` and squeezes element 0 after one
//!   permutation) is checked on that file's two-input vector;
//! - the witness has Kimchi's layout: on each row, the state after round
//!   `r` in the slot `STATE_ORDER[(r + 1) % 5]`, the last on the next row;
//! - the circuit has one selector for the 11 Poseidon rows, 15 gates of
//!   degree 7, and the round constants as its 15 coefficient columns;
//! - "I know a preimage of this hash" is proved and verified.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, circuit, evaluate, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    poseidon::{PoseidonPermutation, ROUNDS_PER_ROW, ROWS_PER_PERMUTATION, STATE_ORDER},
};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::{
    PoseidonSponge,
    kimchi::{KimchiPoseidon, pallas_base},
    pallas_sponge, vesta_sponge,
};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

/// The column variables of the gates: 15 witness cells of a row, the first
/// 3 of the next, and 15 coefficients.
const VARIABLES: usize = 33;
/// The S-box.
const DEGREE: usize = 7;

/// The two-input vector of `kimchi.json`.
const INPUTS: [[u64; 4]; 2] = [
    [
        0xdeee3c188f1c3fbd,
        0x0bd379beed8050a1,
        0x12baf26bb813d6d7,
        0x3793e30ac6917000,
    ],
    [
        0xba0677b84af0e465,
        0x80a7e0eec7136db0,
        0xe6ecb468e28c0b7d,
        0x2fc4c98e50e0b1aa,
    ],
];
const OUTPUT: [u64; 4] = [
    0xa1207602f23624fe,
    0x17a1d0558b313312,
    0xb7d1264867096f08,
    0x336c73d08ad408ce,
];

fn fp(limbs: &[u64; 4]) -> Fp {
    Fp::from_limbs(limbs).expect("a vector element is below the modulus")
}

/// The permutation of a private state.
#[derive(Clone, Copy, Debug)]
struct Permute<'a> {
    poseidon: &'a KimchiPoseidon<Fp>,
}

impl TypedGadget<Fp> for Permute<'_> {
    type Input = Array<3>;
    type Output = Array<3>;
    const NAME: &'static str = "permute";
    const ROWS: usize = ROWS_PER_PERMUTATION + 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<3>, E::Var>,
    ) -> Of<Array<3>, E::Var> {
        let state = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&input[i]], |[v]| v));
        let permutation = PoseidonPermutation {
            poseidon: self.poseidon,
        };
        env.call(&permutation, state)
    }
}

/// Kimchi's hash of two private elements, made public.
#[derive(Clone, Copy, Debug)]
struct Hash<'a> {
    poseidon: &'a KimchiPoseidon<Fp>,
}

impl TypedGadget<Fp> for Hash<'_> {
    type Input = Array<2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-hash";
    const ROWS: usize = ROWS_PER_PERMUTATION + 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let a = env.advice(Cell::new(0, 0), [&input[0]], |[v]| v);
        let b = env.advice(Cell::new(1, 0), [&input[1]], |[v]| v);
        let zero = env.constant(Fp::ZERO);
        let capacity = env.define(Cell::new(2, 0), zero);
        let permutation = PoseidonPermutation {
            poseidon: self.poseidon,
        };
        let [digest, _, _] = env.call(&permutation, [a, b, capacity]);
        env.public(&digest);
        digest
    }
}

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<u64>().prop_map(Fp::from_u64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// The gadget computes Kimchi's permutation.
    #[test]
    fn the_permutation_is_kimchis(state in prop::array::uniform3(arb_fp())) {
        let poseidon = pallas_base();
        let gadget = Permute { poseidon: &poseidon };
        let mut expected = state;
        poseidon.permute_state(&mut expected);
        prop_assert_eq!(evaluate(&gadget, LAYOUT, &state).unwrap(), expected);
    }
}

#[test]
fn the_kimchi_vector() {
    let poseidon = pallas_base();
    let gadget = Hash {
        poseidon: &poseidon,
    };
    let input = INPUTS.map(|l| fp(&l));
    let (compiled, digest) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input).unwrap();
    assert_eq!(digest, fp(&OUTPUT));
    assert_eq!(compiled.public, vec![digest]);
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
}

#[test]
fn the_layout_is_kimchis() {
    let poseidon = pallas_base();
    let gadget = Permute {
        poseidon: &poseidon,
    };
    let state = [Fp::from_u64(1), Fp::from_u64(2), Fp::from_u64(3)];
    let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &state).unwrap();
    let columns = &compiled.columns;
    let mut expected = state;
    for row in 0..ROWS_PER_PERMUTATION {
        for r in 0..ROUNDS_PER_ROW {
            poseidon.round(ROUNDS_PER_ROW * row + r, &mut expected);
            let slot = STATE_ORDER[(r + 1) % ROUNDS_PER_ROW];
            let at = row + usize::from(r + 1 == ROUNDS_PER_ROW);
            let found: [Fp; 3] = core::array::from_fn(|i| columns[3 * slot + i][at]);
            assert_eq!(found, expected, "row {row}, round {r}");
        }
    }
}

#[test]
fn one_selector_fifteen_gates() {
    let poseidon = pallas_base();
    let gadget = Hash {
        poseidon: &poseidon,
    };
    let c = circuit::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT).unwrap();
    // The Poseidon rows, and the hash's own row 0 (its capacity cell).
    assert_eq!(c.selectors().len(), 2);
    let poseidon_rows: Vec<usize> = (0..c.num_rows())
        .filter(|r| c.selectors()[1][*r] == Fp::ONE)
        .collect();
    let rows: Vec<usize> = (0..ROWS_PER_PERMUTATION).collect();
    assert_eq!(poseidon_rows, rows);
    let poseidon_gates: Vec<_> = c
        .gates()
        .iter()
        .filter(|g| g.selector() == Some(1))
        .collect();
    assert_eq!(poseidon_gates.len(), 15);
    // Each is a round's `x^7`, times the selector.
    for gate in &poseidon_gates {
        assert_eq!(c.column_degree(gate), DEGREE + 1);
    }
    // The coefficients are the round constants.
    for row in 0..ROWS_PER_PERMUTATION {
        for r in 0..ROUNDS_PER_ROW {
            let constants = poseidon.round_constants()[ROUNDS_PER_ROW * row + r];
            let found: [Fp; 3] = core::array::from_fn(|i| c.fixed()[3 * r + i][row]);
            assert_eq!(found, constants);
        }
    }
}

fn ipa(size: usize) -> Ipa<Vesta, PoseidonSponge<Fp>> {
    let mut generators = vesta_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = pallas_sponge(IPA_DOMAIN);
    Ipa::<Vesta, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_preimage_is_proved() {
    let poseidon = pallas_base();
    let gadget = Hash {
        poseidon: &poseidon,
    };
    let input = INPUTS.map(|l| fp(&l));
    let (compiled, digest) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input).unwrap();
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
    let claimed = [digest];
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
    let another_digest = [digest + Fp::ONE];
    assert!(
        verify(
            &verifier_key,
            &pcs,
            pallas_sponge(PROOF_DOMAIN),
            &another_digest,
            proof
        )
        .is_err()
    );
}
