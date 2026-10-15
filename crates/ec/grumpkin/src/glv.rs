//! The GLV constants of Grumpkin: `beta`, `lambda` with `phi(P) = [lambda] P`
//! for `phi(x, y) = (beta x, y)`, and the short basis of the decomposition
//! (`arrabbiata_curve::glv`). Generated and checked by a script: `beta` is
//! the multiplicative generator to the power `(p - 1) / 3`, `lambda` the
//! cube root of unity of the scalars that `phi` multiplies the generator
//! by, and the basis is Algorithm 3.74 of the *Guide to Elliptic Curve
//! Cryptography* on `(r, lambda)`.

use arrabbiata_curve::glv::{Basis, Glv, Signed, canonical};

use crate::{Fq, Fr, Grumpkin};

impl Glv for Grumpkin {
    fn beta() -> Fr {
        canonical::<Fr>([
            0x8b17ea66b99c90dd,
            0x5bfc41088d8daaa7,
            0xb3c4d79d41a91758,
            0x0000000000000000,
        ])
    }

    fn lambda() -> Fq {
        canonical::<Fq>([
            0x5763473177fffffe,
            0xd4f263f1acdb5c4f,
            0x59e26bcea0d48bac,
            0x0000000000000000,
        ])
    }

    const BASIS: Basis = Basis {
        a1: Signed {
            negative: false,
            magnitude: [
                0x89d3256894d213e2,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        b1: Signed {
            negative: true,
            magnitude: [
                0x8211bbeb7d4f1129,
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
                0x89d3256894d213e2,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        },
        g1: Signed {
            negative: false,
            magnitude: [
                0xd91d232ec7e0b3d2,
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
