//! The canonical encoding of points: arkworks' compressed form, which Kimchi
//! writes. A point is its `x` coordinate, little-endian, in
//! `ceil((MODULUS_BITS + 2) / 8)` bytes, with two flags in the top bits of
//! the last byte: bit 6 for the point at infinity (with `x = 0`), bit 7
//! when `y` is the larger of `y` and `-y` as integers. `y` is recovered as
//! the square root of `x^3 + b` the flag names.
//!
//! The decoder refuses both flags set, an infinity with a nonzero `x`, an
//! `x` from the modulus up and an `x` off the curve, so each point has
//! exactly one encoding.

use alloc::vec::Vec;

use arrabbiata_codec::{Decodable, Encodable, Error, Input, msgpack};
use arrabbiata_field::{
    Field, PrimeField, Sqrt,
    codec::{byte_len, from_le_bytes, to_le_bytes},
};

use crate::{Affine, Curve, Projective};

/// The flag of the point at infinity.
const INFINITY: u8 = 1 << 6;

/// The flag of a `y` larger than `-y`.
const Y_NEGATIVE: u8 = 1 << 7;

/// The bytes of a point over `C`.
const fn point_len<C: Curve<Base: PrimeField>>() -> usize {
    byte_len(C::Base::MODULUS_BITS + 2)
}

/// Whether `y <= -y` as canonical integers (arkworks' "positive" `y`).
fn is_positive<F: PrimeField>(y: F) -> bool {
    let minus = -y;
    let mut i = F::LIMBS;
    while i > 0 {
        i -= 1;
        let (a, b) = (y.limb(i), minus.limb(i));
        if a != b {
            return a < b;
        }
    }
    true
}

impl<C> Encodable for Affine<C>
where
    C: Curve<Base: PrimeField>,
{
    fn write(&self, out: &mut Vec<u8>) {
        let len = point_len::<C>();
        let mut bytes = match (self.x(), self.y()) {
            (Some(x), Some(y)) => {
                let mut bytes = to_le_bytes(x, len);
                if !is_positive(y) {
                    bytes[len - 1] |= Y_NEGATIVE;
                }
                bytes
            }
            _ => {
                let mut bytes = alloc::vec![0u8; len];
                bytes[len - 1] = INFINITY;
                bytes
            }
        };
        bytes.truncate(len);
        msgpack::write_bin(out, &bytes);
    }
}

impl<C> Decodable<()> for Affine<C>
where
    C: Curve<Base: PrimeField + Sqrt>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        let len = point_len::<C>();
        let bytes = msgpack::read_bin(input)?;
        if bytes.len() != len {
            return Err(Error::Invalid("a point of the right length"));
        }
        let flags = bytes[len - 1] & (INFINITY | Y_NEGATIVE);
        let mut x_bytes = bytes.to_vec();
        x_bytes[len - 1] &= !(INFINITY | Y_NEGATIVE);
        let Some(x) = from_le_bytes::<C::Base>(&x_bytes) else {
            return Err(Error::Invalid("a coordinate below the modulus"));
        };
        if flags == INFINITY {
            return if x == C::Base::ZERO {
                Ok(Self::identity())
            } else {
                Err(Error::Invalid("the point at infinity, with x = 0"))
            };
        }
        if flags == INFINITY | Y_NEGATIVE {
            return Err(Error::Invalid("a point with at most one flag"));
        }
        let Some(root) = (x * x * x + C::b()).sqrt() else {
            return Err(Error::Invalid("the x of a point of the curve"));
        };
        let positive = if is_positive(root) { root } else { -root };
        let y = if flags == Y_NEGATIVE {
            -positive
        } else {
            positive
        };
        // A y of zero is its own negative: only the positive flag names it.
        if flags == Y_NEGATIVE && is_positive(y) {
            return Err(Error::Invalid("a point whose flag names its y"));
        }
        match Self::new(x, y) {
            Some(p) => Ok(p),
            None => Err(Error::Invalid("a point of the curve")),
        }
    }
}

impl<C> Encodable for Projective<C>
where
    C: Curve<Base: PrimeField>,
{
    fn write(&self, out: &mut Vec<u8>) {
        self.to_affine().write(out);
    }
}

impl<C> Decodable<()> for Projective<C>
where
    C: Curve<Base: PrimeField + Sqrt>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        match Affine::<C>::read(input, ()) {
            Ok(a) => Ok(a.to_projective()),
            Err(e) => Err(e),
        }
    }
}
