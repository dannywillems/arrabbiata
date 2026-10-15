//! Kimchi's Poseidon against proof-systems.
//!
//! - every vector of `poseidon/tests/test_vectors/kimchi.json` (the hash over
//!   Pallas's base field `Fp`: absorb the inputs, squeeze once), copied
//!   below with each hex string, little-endian bytes, as canonical limbs;
//! - the regression challenges of `poseidon/tests/poseidon_tests.rs`: the
//!   Fq-sponge over Vesta squeezed on nothing gives a 128-bit challenge, the
//!   low two limbs of its first squeeze (the same test over Pallas is the
//!   first vector's low half);
//! - the sponge's squeeze schedule: two elements per permutation, so the
//!   state only changes every other squeeze, as
//!   `test_poseidon_challenge_multiple_times_without_absorption` checks.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_pasta::{Fp, Fq};
use arrabbiata_poseidon::kimchi::{KimchiSponge, RATE, hash, pallas_base, vesta_base};
use arrabbiata_transcript::Transcript;
use proptest::prelude::*;

/// A hash test vector: the inputs and the output, as canonical limbs.
type Vector = (&'static [[u64; 4]], [u64; 4]);

/// `kimchi.json`, in order.
const VECTORS: [Vector; 6] = [
    (
        &[],
        [
            0x304600f3e09eeba8,
            0xc873af205dfabb8a,
            0x8597455bc2abbd1b,
            0x2fadbe2852044d02,
        ],
    ),
    (
        &[[
            0x18fbe5f6d8e8eef2,
            0xfc93536cae0c612c,
            0x92b4e700d9c49de6,
            0x36fb00ad544e073b,
        ]],
        [
            0x33f9075cf69259fb,
            0x90d391d73ff49559,
            0x619e72176746ad12,
            0x3d4f050775295c04,
        ],
    ),
    (
        &[
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
        ],
        [
            0xa1207602f23624fe,
            0x17a1d0558b313312,
            0xb7d1264867096f08,
            0x336c73d08ad408ce,
        ],
    ),
    (
        &[
            [
                0x95347747ce61eaf5,
                0x2d3e1ca4f6c43d36,
                0xcf3a17ddd6134ba1,
                0x0024fb5773cac987,
            ],
            [
                0x4b8b659cf47385f2,
                0xaa6abdd02ee851a1,
                0x58df721a1d3145b0,
                0x18e0ed2b46ed1ec2,
            ],
            [
                0x7cf18592c345cf73,
                0x4347b5ae0d9ea9ce,
                0x2637fe181292f70c,
                0x1a842a688e600f01,
            ],
        ],
        [
            0x49af544a44941b9b,
            0x6472cae11f3d62a7,
            0x920457af8d090a9f,
            0x3f4b0eabb64e025f,
        ],
    ),
    (
        &[
            [
                0x2201e09871b8284c,
                0x5553a3dadb3cf907,
                0x620ea697fa1382ec,
                0x2059462d60621f70,
            ],
            [
                0xddc446c00fa48707,
                0x570640e5d0caf30f,
                0x747074c771e8ec7f,
                0x2747337d1c4f9894,
            ],
            [
                0xfc6d4e8ee4ff0415,
                0xcf86d3ed39d4dec4,
                0x30e8fa4ad9691b27,
                0x2dd5047c3eeef379,
            ],
            [
                0x58e76f5106650b29,
                0x14872edbf800518b,
                0x65ab607a4ed7c627,
                0x12c387c69bdd436f,
            ],
        ],
        [
            0xdc86ed2ebfd3ec47,
            0x047166d7f3ced2f8,
            0x0054cb9b4dba9d68,
            0x165a8cecf6660c6e,
        ],
    ),
    (
        &[
            [
                0x9fcdf2352b1899da,
                0x762526c45270138a,
                0xdb5236b8de167a37,
                0x3cf70c3a89749a45,
            ],
            [
                0x8202c49039570598,
                0x79b9d21c17dbc992,
                0xf7f6834949fcf902,
                0x1bdf55bc84c1a0e0,
            ],
            [
                0x13b2dab90110ff90,
                0x566a90b7f6d1aa58,
                0xc390152c5039c0d0,
                0x09441e95a82199ef,
            ],
            [
                0x45a3eda03862b588,
                0x79c3d1ec9f95db76,
                0x531723db1f31b50b,
                0x375b4a9785503c24,
            ],
            [
                0xc3a411e527776a89,
                0x232654f32b08990d,
                0xbb2eb6b0fa2a70fb,
                0x12688fe351ed01f3,
            ],
        ],
        [
            0x7c8b907752d5a209,
            0x0f5f60b345f71482,
            0xcd94b5c9d4dc5590,
            0x0ca2c3342c2959d7,
        ],
    ),
];

/// The low two limbs of the first squeeze of a fresh Fq-sponge over Vesta
/// (`test_regression_challenge_empty_vesta_kimchi`).
const EMPTY_VESTA_CHALLENGE: [u64; 2] = [0x70ce4c18c004e5c1, 0xc579d542f9d205a6];

fn fp(limbs: &[u64; 4]) -> Fp {
    Fp::from_limbs(limbs).expect("a vector element is below the modulus")
}

#[test]
fn the_kimchi_vectors() {
    for (inputs, output) in VECTORS {
        let inputs: Vec<Fp> = inputs.iter().map(fp).collect();
        let expected = fp(&output);
        assert_eq!(hash(pallas_base(), &inputs), expected);
    }
}

#[test]
fn the_empty_challenges() {
    let mut vesta = KimchiSponge::new(vesta_base());
    let squeezed: Fq = vesta.squeeze();
    let low = [squeezed.limb(0), squeezed.limb(1)];
    assert_eq!(low, EMPTY_VESTA_CHALLENGE);
    // Over Pallas, the challenge is the low half of the empty hash.
    let (_, empty_hash) = VECTORS[0];
    let mut pallas = KimchiSponge::new(pallas_base());
    let squeezed: Fp = pallas.squeeze();
    let low = [squeezed.limb(0), squeezed.limb(1)];
    assert_eq!(low, [empty_hash[0], empty_hash[1]]);
}

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u64; 4]>().prop_map(|limbs| Fp::from_limbs(&limbs).unwrap_or(Fp::from_u64(limbs[0])))
}

proptest! {
    /// Squeezing reads the rate before permuting again: the state changes
    /// every `RATE` squeezes, and the squeezes in between are its elements.
    #[test]
    fn squeezes_read_the_rate(inputs in prop::collection::vec(arb_fp(), 0..6), rounds in 1usize..4) {
        let mut sponge = KimchiSponge::new(pallas_base());
        for x in &inputs {
            sponge.absorb(*x);
        }
        for _ in 0..rounds {
            let first = sponge.squeeze();
            let state = *sponge.state();
            prop_assert_eq!(first, state[0]);
            for i in 1..RATE {
                prop_assert_eq!(sponge.squeeze(), state[i]);
                prop_assert_eq!(sponge.state(), &state);
            }
        }
    }

    /// Absorbing after a squeeze adds to the first element without
    /// permuting.
    #[test]
    fn absorbing_after_a_squeeze(inputs in prop::collection::vec(arb_fp(), 0..6), x in arb_fp()) {
        let mut sponge = KimchiSponge::new(pallas_base());
        for y in &inputs {
            sponge.absorb(*y);
        }
        sponge.squeeze();
        let before = *sponge.state();
        sponge.absorb(x);
        prop_assert_eq!(sponge.state(), &[before[0] + x, before[1], before[2]]);
    }
}
