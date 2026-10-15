//! The GLV constants of BN254's G1 and G2: `beta`, `lambda` with `phi(P) = [lambda] P`
//! for `phi(x, y) = (beta x, y)`, and the short basis of the decomposition
//! (`arrabbiata_curve::glv`). Generated and checked by a script: `beta` is
//! the multiplicative generator to the power `(p - 1) / 3`, `lambda` the
//! cube root of unity of the scalars that `phi` multiplies the generator
//! by, and the basis is Algorithm 3.74 of the *Guide to Elliptic Curve
//! Cryptography* on `(r, lambda)`.
//!
//! On G2, the twist over `Fq2`, `phi` uses G1's `beta` (in `Fq`) and acts
//! as `lambda^2` of G1's `lambda`.

use arrabbiata_curve::glv::{Basis, Glv, Signed, canonical};

use arrabbiata_field::Field;

use crate::{Fq, Fq2, Fr, G1, G2};

impl Glv for G1 {
    fn beta() -> Fq {
        canonical::<Fq>([
            0xe4bd44e5607cfd48,
            0xc28f069fbb966e3d,
            0x5e6dd9e7e0acccb0,
            0x30644e72e131a029,
        ])
    }

    fn lambda() -> Fr {
        canonical::<Fr>([
            0xb8ca0b2d36636f23,
            0xcc37a73fec2bc5e9,
            0x048b6e193fd84104,
            0x30644e72e131a029,
        ])
    }

    const BASIS: Basis = Basis {
        a1: Signed {
            negative: false,
            magnitude: [
                0x8211bbeb7d4f1128,
                0x6f4d8248eeb859fc,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b1: Signed {
            negative: true,
            magnitude: [
                0x89d3256894d213e3,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        a2: Signed {
            negative: false,
            magnitude: [
                0x89d3256894d213e3,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b2: Signed {
            negative: false,
            magnitude: [
                0x0be4e1541221250b,
                0x6f4d8248eeb859fd,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        g1: Signed {
            negative: false,
            magnitude: [
                0x5398fd0300ff6565,
                0x4ccef014a773d2d2,
                0x0000000000000002,
                0x0000000000000000,
            ],
        },
        g2: Signed {
            negative: false,
            magnitude: [
                0xd91d232ec7e0b3d7,
                0x0000000000000002,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
    };
}

impl Glv for G2 {
    fn beta() -> Fq2 {
        Fq2::new(
            canonical::<Fq>([
                0xe4bd44e5607cfd48,
                0xc28f069fbb966e3d,
                0x5e6dd9e7e0acccb0,
                0x30644e72e131a029,
            ]),
            Fq::ZERO,
        )
    }

    fn lambda() -> Fr {
        canonical::<Fr>([
            0x8b17ea66b99c90dd,
            0x5bfc41088d8daaa7,
            0xb3c4d79d41a91758,
            0x0000000000000000,
        ])
    }

    const BASIS: Basis = Basis {
        a1: Signed {
            negative: false,
            magnitude: [
                0x89d3256894d213e3,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b1: Signed {
            negative: true,
            magnitude: [
                0x8211bbeb7d4f1128,
                0x6f4d8248eeb859fc,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        a2: Signed {
            negative: false,
            magnitude: [
                0x0be4e1541221250b,
                0x6f4d8248eeb859fd,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b2: Signed {
            negative: false,
            magnitude: [
                0x89d3256894d213e3,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        g1: Signed {
            negative: false,
            magnitude: [
                0xd91d232ec7e0b3d7,
                0x0000000000000002,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        g2: Signed {
            negative: false,
            magnitude: [
                0x7a7bd9d4391eb18e,
                0x4ccef014a773d2cf,
                0x0000000000000002,
                0x0000000000000000,
            ],
        },
    };
}
