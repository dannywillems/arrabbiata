//! The codec of prime field elements: its two laws, sampled on Goldilocks
//! (the Pasta fields are sampled with the points, in `arrabbiata-curve`),
//! and golden bytes from the encoding's definition: the canonical integer,
//! little-endian, in `ceil(MODULUS_BITS / 8)` bytes, in a `bin`.

use arrabbiata_codec::{Error, decode, encode, laws};
use arrabbiata_field::{Field, Goldilocks, goldilocks::MODULUS};
use proptest::prelude::*;

proptest! {
    #[test]
    fn goldilocks_round_trips(x in any::<u64>()) {
        let x = Goldilocks::from_u64(x);
        prop_assert_eq!(laws::round_trip(&x, ()), Ok(()));
    }

    #[test]
    fn arbitrary_bytes_are_canonical_when_accepted(bytes in prop::collection::vec(any::<u8>(), 0..12)) {
        prop_assert_eq!(laws::canonical::<Goldilocks, ()>(&bytes, ()), Ok(()));
        let mut bin = vec![0xc4, 8];
        bin.extend(bytes.iter().chain(&[0u8; 8]).take(8));
        prop_assert_eq!(laws::canonical::<Goldilocks, ()>(&bin, ()), Ok(()));
    }
}

#[test]
fn golden_bytes() {
    assert_eq!(encode(&Goldilocks::ONE), [0xc4, 8, 1, 0, 0, 0, 0, 0, 0, 0]);
    let minus_one = -Goldilocks::ONE;
    let mut expected = vec![0xc4, 8];
    expected.extend((MODULUS - 1).to_le_bytes());
    assert_eq!(encode(&minus_one), expected);
}

#[test]
fn the_modulus_and_other_lengths_are_refused() {
    let mut p = vec![0xc4, 8];
    p.extend(MODULUS.to_le_bytes());
    assert_eq!(
        decode::<Goldilocks>(&p),
        Err(Error::Invalid("a field element below the modulus"))
    );
    assert_eq!(
        decode::<Goldilocks>(&[0xc4, 1, 1]),
        Err(Error::Invalid("a field element of the right length"))
    );
}
