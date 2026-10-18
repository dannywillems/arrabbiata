//! The codec of points (arkworks' compressed form): its two laws, sampled
//! on Pallas, Vesta and Grumpkin, and golden bytes from the encoding's
//! definition. On Pasta the coordinate has 255 bits and the two flags take
//! a 33rd byte; on Grumpkin it has 254 and they fit in the 32nd.

use arrabbiata_codec::{Error, decode, encode, laws};
use arrabbiata_curve::{Affine, Curve, Projective};
use arrabbiata_field::{Field, PrimeField, codec::to_le_bytes};
use arrabbiata_grumpkin::Grumpkin;
use arrabbiata_pasta::{Pallas, Vesta};
use proptest::prelude::*;

/// `[k] G` on `C`.
fn multiple<C: Curve>(k: u64) -> Projective<C> {
    let (x, y) = C::generator();
    Affine::<C>::new(x, y).unwrap().to_projective() * C::Scalar::from_u64(k)
}

/// Both laws at `[k] G`, and canonicity on `bytes`.
fn check<C>(k: u64, bytes: &[u8])
where
    C: Curve<Base: PrimeField + arrabbiata_field::Sqrt>,
{
    let p = multiple::<C>(k);
    assert_eq!(laws::round_trip(&p, ()), Ok(()));
    assert_eq!(laws::round_trip(&-p, ()), Ok(()));
    assert_eq!(laws::canonical::<Projective<C>, ()>(bytes, ()), Ok(()));
    // The near miss of a real encoding: one bit flipped.
    let mut near = encode(&p);
    let last = near.len() - 1;
    near[2 + (bytes.len() % (last - 1))] ^= 1 << (bytes.len() % 8);
    assert_eq!(laws::canonical::<Projective<C>, ()>(&near, ()), Ok(()));
}

proptest! {
    #[test]
    fn points_obey_both_laws(k in any::<u64>(), tail in prop::collection::vec(any::<u8>(), 33)) {
        let mut bytes = vec![0xc4, 33];
        bytes.extend(&tail);
        check::<Pallas>(k, &bytes);
        check::<Vesta>(k, &bytes);
        let mut short = vec![0xc4, 32];
        short.extend(&tail[..32]);
        check::<Grumpkin>(k, &short);
    }
}

#[test]
fn golden_bytes() {
    // The point at infinity: x = 0 and the flag 0x40, 33 bytes on Pasta.
    let mut infinity = vec![0xc4, 33];
    infinity.extend([0u8; 32]);
    infinity.push(0x40);
    assert_eq!(encode(&Projective::<Pallas>::identity()), infinity);
    // Pallas's generator (-1, 2): y = 2 is the smaller of y and -y.
    let (x, y) = Pallas::generator();
    let g = Affine::<Pallas>::new(x, y).unwrap();
    let mut expected = vec![0xc4, 33];
    expected.extend(to_le_bytes(x, 32));
    expected.push(0);
    assert_eq!(encode(&g), expected);
    // Its negative, (-1, -2): the flag 0x80.
    let minus = Affine::<Pallas>::new(x, -y).unwrap();
    *expected.last_mut().unwrap() = 0x80;
    assert_eq!(encode(&minus), expected);
}

#[test]
fn malformed_points_are_refused() {
    let mut both = vec![0xc4, 33];
    both.extend([0u8; 32]);
    both.push(0xc0);
    assert!(decode::<Projective<Pallas>>(&both).is_err());
    let mut infinity_with_x = vec![0xc4, 33, 1];
    infinity_with_x.extend([0u8; 31]);
    infinity_with_x.push(0x40);
    assert_eq!(
        decode::<Projective<Pallas>>(&infinity_with_x),
        Err(Error::Invalid("the point at infinity, with x = 0"))
    );
    assert_eq!(
        decode::<Projective<Pallas>>(&[0xc4, 32, 0]),
        Err(Error::Truncated)
    );
}
