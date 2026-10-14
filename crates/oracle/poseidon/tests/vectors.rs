//! The Poseidon instances over the Pasta fields reproduce the permutation
//! of Halo 2 and Orchard, on the test vectors of
//! zcash-hackworks/zcash-test-vectors (`orchard_poseidon/permute`), as
//! shipped in `halo2_poseidon/src/test_vectors.rs`.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_pasta::{Fp, Fq};
use arrabbiata_poseidon::{Poseidon, pallas_base, pallas_sponge, vesta_base};
use arrabbiata_transcript::Transcript;
use proptest::prelude::*;

/// A permutation test vector: initial and final states, as canonical limbs.
type Vector = ([[u64; 4]; 3], [[u64; 4]; 3]);

const FP_VECTORS: [Vector; 4] = [
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
                0xaeb1bc024aeca456,
                0xf7e69a71d0b642a0,
                0x94efb364f966240f,
                0x2a526acd0b64b453,
            ],
            [
                0x012a3e9628e5b82a,
                0xdcd42e7fbed9dafe,
                0x76ff7dae343d5512,
                0x13c5d1568b4aa430,
            ],
            [
                0x359029a1d34e9ddd,
                0xf7cfdfe1bda42c7b,
                0x256fcd597984561a,
                0x0a49c868c6976544,
            ],
        ],
    ),
    (
        [
            [
                0xfb70fcad738f7a5c,
                0x07576bac4994133f,
                0x9d9364b1666e4d4c,
                0x082169eef62efaaf,
            ],
            [
                0xa6bde1f2b386dd1a,
                0xe6772b980e2e5d2a,
                0xc78849f2a39cefb0,
                0x0dcdb1cf014253b3,
            ],
            [
                0x0f94b63225b869bd,
                0x27c7a99b670f59f2,
                0xd6368e9c7e1fe01f,
                0x1435a7304e9de2a5,
            ],
        ],
        [
            [
                0x7e8a9238832f6ed0,
                0xcd878092770c38e7,
                0x036952a16129fda2,
                0x21ddae20d1d6227a,
            ],
            [
                0xd6106f415fa45529,
                0xc969c0d094ac79bc,
                0x1f1e4809bdf4e549,
                0x0d1451eeb8b98c36,
            ],
            [
                0x2b5dd6e9bb76830d,
                0x87ab82d9b76f131e,
                0x795cbe4430401cc5,
                0x105bf9ac68bb569d,
            ],
        ],
    ),
    (
        [
            [
                0xbeafd655429850bc,
                0x08c05aed4828f99e,
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
                0xdbf67f5df445a705,
                0xbf3ef0f0fd67bc10,
                0xb097263633ab3081,
                0x36f6b8cc63c7e4e4,
            ],
        ],
        [
            [
                0x0c5a140753ec770b,
                0xc36af9d6a9c72d05,
                0x8ed532086472ae41,
                0x12178017a492eb51,
            ],
            [
                0x3f460ef0443f523b,
                0x0c280efcd4d70f8b,
                0x076881f127b9dedb,
                0x182e5a67f262b37b,
            ],
            [
                0x1535ccff06977a95,
                0x4c3111992a80ae64,
                0x4083cfaf223ee205,
                0x267605c1fa80df59,
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
                0x7709ec1f36d30a3d,
                0x7d8d98420ebed990,
                0x531a9bf438a1c925,
                0x11984ae34bf0dc7e,
            ],
            [
                0x72507bd2b69dafa4,
                0xd75e39883e0c5f83,
                0xa88480ad52001ba4,
                0x16ad0d328d94dab9,
            ],
        ],
        [
            [
                0x42cb827f3f088067,
                0x6a84834b6fb6e754,
                0xc96e9cc3b93f77c9,
                0x2a55092322068b81,
            ],
            [
                0x58b1402c7ea5f9a5,
                0x2c6502e6435316d8,
                0xe5eecadd640bfc3e,
                0x08509fd51f953dce,
            ],
            [
                0x7e477c123664a4dc,
                0x568ac67ca00f9583,
                0x5268c2ad5518546e,
                0x3b1e928824358797,
            ],
        ],
    ),
];
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

fn element<F>(limbs: [u64; 4]) -> F
where
    F: PrimeField,
{
    F::from_limbs(&limbs).expect("a test vector element is below the modulus")
}

fn check<F>(poseidon: &Poseidon<F, 3>, vectors: &[Vector])
where
    F: PrimeField,
{
    for (index, (initial, expected)) in vectors.iter().enumerate() {
        let mut state = initial.map(element::<F>);
        poseidon.permute_state(&mut state);
        assert_eq!(state, expected.map(element::<F>), "vector {index}");
    }
}

#[test]
fn pallas_base_matches_the_zcash_test_vectors() {
    check(&pallas_base(), &FP_VECTORS);
}

#[test]
fn vesta_base_matches_the_zcash_test_vectors() {
    check(&vesta_base(), &FQ_VECTORS);
}

#[test]
fn mds_is_invertible() {
    // A Cauchy matrix on distinct points is invertible; check det != 0.
    let m = *pallas_base().mds();
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(!det.is_zero());
}

#[test]
#[should_panic(expected = "alpha does not divide")]
fn rejects_an_sbox_that_is_not_a_permutation() {
    // 3 divides p - 1 for Pallas, so x^3 is not a permutation.
    let _ = Poseidon::<Fp, 3>::new(8, 56, 3);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// The permutation is injective on samples (a permutation is a
    /// bijection): distinct inputs give distinct outputs.
    #[test]
    fn distinct_inputs_give_distinct_outputs(a in any::<[u64; 3]>(), b in any::<[u64; 3]>()) {
        prop_assume!(a != b);
        let poseidon = pallas_base();
        let mut x = a.map(Fp::from_u64);
        let mut y = b.map(Fp::from_u64);
        poseidon.permute_state(&mut x);
        poseidon.permute_state(&mut y);
        prop_assert_ne!(x, y);
    }

    /// The sponge is deterministic and sensitive to every absorbed element.
    #[test]
    fn sponge_binds_its_input(a in any::<[u64; 4]>(), i in 0usize..4, delta in 1u64..1000) {
        let squeeze = |values: [u64; 4]| {
            let mut sponge = pallas_sponge(7);
            for v in values {
                sponge.absorb(Fp::from_u64(v));
            }
            sponge.squeeze()
        };
        let mut b = a;
        b[i] = b[i].wrapping_add(delta);
        prop_assert_eq!(squeeze(a), squeeze(a));
        prop_assert_ne!(squeeze(a), squeeze(b));
    }
}

#[test]
fn vesta_is_a_different_instance() {
    let mut x = [Fq::ZERO, Fq::ONE, Fq::from_u64(2)];
    vesta_base().permute_state(&mut x);
    assert_ne!(x, [Fq::ZERO, Fq::ONE, Fq::from_u64(2)]);
}
