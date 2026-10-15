//! The GLV constants of Pallas and Vesta: `beta`, `lambda` with `phi(P) = [lambda] P`
//! for `phi(x, y) = (beta x, y)`, and the short basis of the decomposition
//! (`arrabbiata_curve::glv`). Generated and checked by a script: `beta` is
//! the multiplicative generator to the power `(p - 1) / 3`, `lambda` the
//! cube root of unity of the scalars that `phi` multiplies the generator
//! by, and the basis is Algorithm 3.74 of the *Guide to Elliptic Curve
//! Cryptography* on `(r, lambda)`.
//!
//! `(beta, lambda)` are Kimchi's `endos` (o1-labs/proof-systems,
//! `poly_commitment::ipa::endos`), as printed by Kimchi.

use arrabbiata_curve::glv::{Basis, Glv, Signed, canonical};

use crate::{Fp, Fq, Pallas, Vesta};

impl Glv for Pallas {
    fn beta() -> Fp {
        canonical::<Fp>([
            0x7b7fd22f0201b547,
            0x05270d29d19fc7d2,
            0xd3552a23a8554e50,
            0x2d33357cb532458e,
        ])
    }

    fn lambda() -> Fq {
        canonical::<Fq>([
            0x619d1840af55f1b1,
            0x1259527ec1d4752e,
            0xaee24b27e308f0a6,
            0x397e65a7d7c1ad71,
        ])
    }

    const BASIS: Basis = Basis {
        a1: Signed {
            negative: false,
            magnitude: [
                0x8cb1279300000000,
                0x49e69d1640a89953,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b1: Signed {
            negative: true,
            magnitude: [
                0x7fcae1c700000001,
                0x49e69d1640f04915,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        a2: Signed {
            negative: false,
            magnitude: [
                0x0c7c095a00000001,
                0x93cd3a2c8198e269,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b2: Signed {
            negative: false,
            magnitude: [
                0x8cb1279300000000,
                0x49e69d1640a89953,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        g1: Signed {
            negative: false,
            magnitude: [
                0x32c49e4bffffffff,
                0x279a745902a2654e,
                0x0000000000000001,
                0x0000000000000000,
            ],
        },
        g2: Signed {
            negative: false,
            magnitude: [
                0xff2b871c00000003,
                0x279a745903c12455,
                0x0000000000000001,
                0x0000000000000000,
            ],
        },
    };
}

impl Glv for Vesta {
    fn beta() -> Fq {
        canonical::<Fq>([
            0x2aa9d2e050aa0e4f,
            0x0fed467d47c033af,
            0x511db4d81cf70f5a,
            0x06819a58283e528e,
        ])
    }

    fn lambda() -> Fp {
        canonical::<Fp>([
            0x1dad5ebdfdfe4ab9,
            0x1d1f8bd237ad3149,
            0x2caad5dc57aab1b0,
            0x12ccca834acdba71,
        ])
    }

    const BASIS: Basis = Basis {
        a1: Signed {
            negative: false,
            magnitude: [
                0x7fcae1c700000000,
                0x49e69d1640f04915,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b1: Signed {
            negative: true,
            magnitude: [
                0x8cb1279300000001,
                0x49e69d1640a89953,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        a2: Signed {
            negative: false,
            magnitude: [
                0x8cb1279300000001,
                0x49e69d1640a89953,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b2: Signed {
            negative: false,
            magnitude: [
                0x0c7c095a00000001,
                0x93cd3a2c8198e269,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        g1: Signed {
            negative: false,
            magnitude: [
                0x31f0256800000003,
                0x4f34e8b2066389a4,
                0x0000000000000002,
                0x0000000000000000,
            ],
        },
        g2: Signed {
            negative: false,
            magnitude: [
                0x32c49e4c00000003,
                0x279a745902a2654e,
                0x0000000000000001,
                0x0000000000000000,
            ],
        },
    };
}
