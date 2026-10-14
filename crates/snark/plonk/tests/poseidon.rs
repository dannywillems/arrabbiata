//! The Poseidon permutation as a PlonK circuit, one row per round, checked
//! against the native permutation and against external test vectors:
//!
//! - over the Pasta field `Fq` (Poseidon of Halo 2 and Orchard over Vesta's
//!   base field), on the four Zcash vectors (`orchard_poseidon/permute`,
//!   as in `halo2_poseidon/src/test_vectors.rs`), proved with IPA on
//!   Pallas;
//! - over the BN254 scalar field, on the reference vector of the Poseidon
//!   authors (`hadeshash`, `poseidonperm_x5_254_3`), proved with KZG.
//!
//! Layout (`W = 3` state columns, `X = 3` fixed columns holding the round
//! constants): row `r` holds the state before round `r`, and its gadget
//! (a full or a partial round) constrains the next row to the state after
//! it: `s' = M S(s + c)`, with `S` the S-box on every element (full) or on
//! the first only (partial). The row after the last round holds the
//! output; six plain rows copy the input and the output into column 0 to
//! make them public.

use std::sync::OnceLock;

use arrabbiata_bn254::Fr;
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_pasta::Fq;
use arrabbiata_plonk::{
    Builder, Cell, Circuit, Gadget, PlonkError, Witness, ipa, kzg, prove, setup, verify,
};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};
use arrabbiata_poseidon::{Poseidon, bn254_scalar, vesta_base};
use proptest::prelude::*;

/// A field with a width-3 Poseidon instance.
trait Instance: PrimeField + 'static {
    fn poseidon() -> &'static Poseidon<Self, 3>;
}

impl Instance for Fq {
    fn poseidon() -> &'static Poseidon<Fq, 3> {
        static P: OnceLock<Poseidon<Fq, 3>> = OnceLock::new();
        P.get_or_init(vesta_base)
    }
}

impl Instance for Fr {
    fn poseidon() -> &'static Poseidon<Fr, 3> {
        static P: OnceLock<Poseidon<Fr, 3>> = OnceLock::new();
        P.get_or_init(bn254_scalar)
    }
}

type E<F> = Expr<F, 3, 3>;

fn state<F: Field>(i: usize, r: Rotation) -> E<F> {
    E::witness(WitnessColumn::try_new(i).unwrap(), r)
}

fn constant<F: Field>(i: usize) -> E<F> {
    E::fixed(FixedColumn::try_new(i).unwrap(), Rotation::CURR)
}

fn pow<F: Field>(x: E<F>, e: u64) -> E<F> {
    let mut acc = x.clone();
    for _ in 1..e {
        acc = acc * x.clone();
    }
    acc
}

/// `s' = M S(s + c)` with the S-box on the elements `sbox` selects.
fn round<F: Instance>(full: bool) -> Vec<E<F>> {
    let p = F::poseidon();
    let after: Vec<E<F>> = (0..3)
        .map(|j| {
            let x = state::<F>(j, Rotation::CURR) + constant(j);
            if full || j == 0 { pow(x, p.alpha()) } else { x }
        })
        .collect();
    (0..3)
        .map(|i| {
            let mut mixed = E::constant(F::ZERO);
            for (j, s) in after.iter().enumerate() {
                mixed = mixed + E::constant(p.mds()[i][j]) * s.clone();
            }
            state(i, Rotation::NEXT) - mixed
        })
        .collect()
}

/// A full round: the S-box on the whole state.
struct Full;
impl<F: Instance> Gadget<F, 3, 3> for Full {
    const NAME: &'static str = "full round";
    fn constraints() -> Vec<E<F>> {
        round(true)
    }
}

/// A partial round: the S-box on the first element.
struct Partial;
impl<F: Instance> Gadget<F, 3, 3> for Partial {
    const NAME: &'static str = "partial round";
    fn constraints() -> Vec<E<F>> {
        round(false)
    }
}

type Gates = (Full, (Partial, ()));

/// The circuit computing the permutation of `input`, its witness, and the
/// native output. When `claim` is given, the output row holds it instead
/// of the real output (a false statement).
fn permutation<F: Instance>(
    input: [F; 3],
    claim: Option<[F; 3]>,
) -> (Circuit<F, Gates, 3, 3, 2>, Witness<F, 3>, [F; 3]) {
    let p = F::poseidon();
    let rounds = p.full_rounds() + p.partial_rounds();
    let half = p.full_rounds() / 2;
    let mut b = Builder::<F, Gates, 3, 3, 2>::new();
    let mut s = input;
    for r in 0..rounds {
        let c = p.round_constants()[r];
        let full = r < half || r >= half + p.partial_rounds();
        if full {
            b.row::<Full, _>(s, c);
        } else {
            b.row::<Partial, _>(s, c);
        }
        let mut next = s;
        for i in 0..3 {
            next[i] = next[i] + c[i];
            if full || i == 0 {
                next[i] = next[i].pow(p.alpha());
            }
        }
        let mixed: [F; 3] =
            std::array::from_fn(|i| (0..3).fold(F::ZERO, |acc, j| acc + p.mds()[i][j] * next[j]));
        s = mixed;
    }
    let output = s;
    let shown = claim.unwrap_or(output);
    let last = b.plain_row(shown, [F::ZERO; 3]);
    let col = |k: usize| WitnessColumn::try_new(k).unwrap();
    for (row, values) in [(0, input), (last, shown)] {
        for (k, v) in values.iter().enumerate() {
            let public = b.plain_row([*v, F::ZERO, F::ZERO], [F::ZERO; 3]);
            b.copy(Cell::new(col(k), row), Cell::new(col(0), public));
            b.public(public);
        }
    }
    let (circuit, witness) = b.build().unwrap();
    (circuit, witness, output)
}

fn public<F: Field>(input: [F; 3], output: [F; 3]) -> Vec<F> {
    input.iter().chain(output.iter()).copied().collect()
}

/// A permutation test vector: initial and final states, as canonical limbs.
type Vector = ([[u64; 4]; 3], [[u64; 4]; 3]);

/// The Zcash vectors over `Fq`, copied from
/// `arrabbiata-poseidon/tests/vectors.rs`.
const FQ_VECTORS: [Vector; 4] = [
    (
        [
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
            [
                0x0000000000000001,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
            [
                0x0000000000000002,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        ],
        [
            [
                0x0eb08ea813bebe59,
                0x4d43d1973dd336c6,
                0xeddd74f22f8f2ff7,
                0x315a1f4cdb942f7c,
            ],
            [
                0xf9f126e61ea165f1,
                0x413ee0eb7bbd2198,
                0x642adee0dd13aa48,
                0x3be475f2d7642bde,
            ],
            [
                0x14d542372a7ba0d9,
                0x5019bfd4e0423fa0,
                0x117fdb2420d8ea60,
                0x25ab8aece9537168,
            ],
        ],
    ),
    (
        [
            [
                0x08574279738f7a5c,
                0x07576bac494c637e,
                0x9d9364b1666e4d4c,
                0x082169eef62efaaf,
            ],
            [
                0xc08a6d8ab386dd1a,
                0xe6772b980d9efda6,
                0xc78849f2a39cefb0,
                0x0dcdb1cf014253b3,
            ],
            [
                0x296141ca25b869bd,
                0x27c7a99b667ffa6e,
                0xd6368e9c7e1fe01f,
                0x1435a7304e9de2a5,
            ],
        ],
        [
            [
                0x8f72c7df92838fcd,
                0xa47060c44c856d5f,
                0x65dc2d33807aba0c,
                0x0e5e23dec34ae2cb,
            ],
            [
                0x8aaa833c95e553c2,
                0x7ec0bf7f58d5d423,
                0xfaf5d1467d1f3378,
                0x271b24d4bd6a4d54,
            ],
            [
                0xd12a47f99b33c8b8,
                0xa92c81995eb727c5,
                0x1a6f464d7fbd5c1c,
                0x3464766650675a13,
            ],
        ],
    ),
    (
        [
            [
                0xe562a7b9429850bc,
                0x08c05aed4751ea58,
                0xb6bcec2f7bfac262,
                0x0e81632a9168694b,
            ],
            [
                0xf5621d6ad566c13d,
                0x1393fdb51d55d7a8,
                0xd4f76a993d20c7e8,
                0x0df89ad556370877,
            ],
            [
                0xe8dcc529f445a705,
                0xbf3ef0f0fd200c4e,
                0xb097263633ab3081,
                0x36f6b8cc63c7e4e4,
            ],
        ],
        [
            [
                0x49784a6264565ba3,
                0xa2791805fae0ce9e,
                0xa4d25b9b53a41cad,
                0x08402d4e8deae267,
            ],
            [
                0xf8aa75e57a21c236,
                0xdc1c1060e0d654f2,
                0xf03b54093c39aa85,
                0x0227b95c7c7a9948,
            ],
            [
                0x56ae11afbe7e1238,
                0x1c603b2905144764,
                0x3a1178408e3ef043,
                0x1f1b9a9f11103763,
            ],
        ],
    ),
    (
        [
            [
                0x311eba7f2f225c49,
                0xe1c2ef575a3dfade,
                0xfbd58755031ab0e9,
                0x3c3d90941de0381a,
            ],
            [
                0x83f031eb36d30a3d,
                0x7d8d98420e7729ce,
                0x531a9bf438a1c925,
                0x11984ae34bf0dc7e,
            ],
            [
                0x99034d36b69dafa4,
                0xd75e39883d35503d,
                0xa88480ad52001ba4,
                0x16ad0d328d94dab9,
            ],
        ],
        [
            [
                0x7a431ac74f63765c,
                0xea1cb5b39d89c73c,
                0x80a696140bd09ae6,
                0x187908371783d332,
            ],
            [
                0xc4ac0a45ab4eccd3,
                0xb8509abb7e329b5f,
                0xc3546b100e08ca59,
                0x2adf791d1ec2091c,
            ],
            [
                0xa300bae91e8f1e09,
                0x253d95d6d585cfe3,
                0xb70fde43db8b1ee0,
                0x1c7e698f6a4e8230,
            ],
        ],
    ),
];

fn limbs<F: PrimeField>(v: [[u64; 4]; 3]) -> [F; 3] {
    v.map(|l| F::from_limbs(&l).unwrap())
}

/// A big-endian hex string as an element of `Fr`.
fn fr(hex: &str) -> Fr {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let end = hex.len() - 16 * i;
        *limb = u64::from_str_radix(&hex[end - 16..end], 16).unwrap();
    }
    Fr::from_limbs(&limbs).unwrap()
}

#[test]
fn layout() {
    let (circuit, witness, _) = permutation::<Fq>([Fq::ZERO; 3], None);
    // 64 rounds, the output, and six public rows: 71 rows on 128.
    assert_eq!(circuit.num_rows(), 128);
    assert_eq!(circuit.public_rows().len(), 6);
    // x^5 on s + c (both columns): degree 5, 6 with the selector.
    assert_eq!(circuit.quotient_factor(), 6);
    assert!(circuit.is_satisfied(&witness));
}

/// The circuit, the native permutation and the Zcash vectors agree, and
/// each vector is proved and verified with IPA on Pallas.
#[test]
fn zcash_vectors_are_proved() {
    let (circuit, _, _) = permutation::<Fq>([Fq::ZERO; 3], None);
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    for (input, output) in FQ_VECTORS {
        let (input, expected) = (limbs::<Fq>(input), limbs::<Fq>(output));
        let (c, witness, output) = permutation(input, None);
        // Every input gives the same circuit: only the witness changes.
        assert_eq!(c, circuit);
        assert_eq!(output, expected);
        let mut native = input;
        Fq::poseidon().permute_state(&mut native);
        assert_eq!(native, expected);
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        assert!(
            verify(
                &vk,
                &pcs,
                ipa::pallas_transcript(),
                &public(input, expected),
                proof.clone()
            )
            .is_ok()
        );
        // Another output is another statement.
        let mut other = expected;
        other[2] = other[2] + Fq::ONE;
        assert!(
            verify(
                &vk,
                &pcs,
                ipa::pallas_transcript(),
                &public(input, other),
                proof
            )
            .is_err()
        );
    }
}

/// The reference vector of the Poseidon authors over the BN254 scalar
/// field, proved with KZG.
#[test]
fn hadeshash_vector_is_proved_with_kzg() {
    let input = [Fr::ZERO, Fr::ONE, Fr::from_u64(2)];
    let expected = [
        fr("115cc0f5e7d690413df64c6b9662e9cf2a3617f2743245519e19607a4417189a"),
        fr("0fca49b798923ab0239de1c9e7a4a9a2210312b6a2f616d18b5a87f9b628ae29"),
        fr("0e7ae82e40091e63cbd4f16a6d16310b3729d4b6e138fcf54110e2867045a30c"),
    ];
    let (circuit, witness, output) = permutation(input, None);
    assert_eq!(output, expected);
    let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0x9e5e));
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
    assert!(
        verify(
            &vk,
            &pcs,
            kzg::transcript(),
            &public(input, expected),
            proof
        )
        .is_ok()
    );
}

/// A false output has no proof.
#[test]
fn false_output_is_unsatisfied() {
    let input = [Fq::ONE, Fq::from_u64(2), Fq::from_u64(3)];
    let (_, _, output) = permutation(input, None);
    let mut lie = output;
    lie[0] = lie[0] + Fq::ONE;
    let (circuit, witness, _) = permutation(input, Some(lie));
    assert!(!circuit.is_satisfied(&witness));
    let pcs = ipa::pallas(&circuit);
    let (pk, _) = setup(&circuit, &pcs).unwrap();
    assert_eq!(
        prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap_err(),
        PlonkError::Unsatisfied
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// On random inputs, the circuit is satisfied by the native
    /// permutation and by nothing else at any single cell.
    #[test]
    fn circuit_computes_the_permutation(
        input in any::<[u64; 3]>(),
        row in 0usize..65,
        column in 0usize..3,
        delta in 1u64..1000,
    ) {
        let input = input.map(Fq::from_u64);
        let (circuit, witness, output) = permutation(input, None);
        let mut native = input;
        Fq::poseidon().permute_state(&mut native);
        prop_assert_eq!(output, native);
        prop_assert!(circuit.is_satisfied(&witness));
        let mut columns = witness.columns().clone();
        columns[column][row] = columns[column][row] + Fq::from_u64(delta);
        let broken = Witness::new(columns, witness.public_inputs().to_vec());
        prop_assert!(!circuit.is_satisfied(&broken));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3))]

    /// Random inputs are proved and verified.
    #[test]
    fn random_inputs_are_proved(input in any::<[u64; 3]>()) {
        let input = input.map(Fq::from_u64);
        let (circuit, witness, output) = permutation(input, None);
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
        prop_assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &public(input, output), proof).is_ok());
    }
}
