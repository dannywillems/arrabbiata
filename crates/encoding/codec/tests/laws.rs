//! The two laws of the codecs this crate defines, sampled: the round trip
//! over values, and canonicity over byte strings, arbitrary ones and the
//! longer forms MessagePack allows for the same value. Then golden bytes,
//! written out from MessagePack's definition, not from the code under test:
//! the laws are self-referential and would survive an encoding changed
//! consistently in both directions; a fixed byte string does not.

use arrabbiata_codec::{Error, decode, encode, laws};
use proptest::prelude::*;

proptest! {
    #[test]
    fn integers_round_trip(x in any::<u64>(), y in any::<u32>(), z in any::<u8>(), b in any::<bool>()) {
        prop_assert_eq!(laws::round_trip(&x, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&y, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&z, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&(x as usize), ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&b, ()), Ok(()));
    }

    #[test]
    fn containers_round_trip(
        v in prop::collection::vec(any::<u64>(), 0..40),
        o in prop::option::of(any::<u32>()),
        n in prop::collection::vec(prop::option::of(prop::collection::vec(any::<u8>(), 0..20)), 0..20),
        a in any::<[u16; 4]>(),
    ) {
        let a = a.map(u32::from);
        prop_assert_eq!(laws::round_trip(&v, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&o, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&n, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&a, ()), Ok(()));
        prop_assert_eq!(laws::round_trip(&(v.clone(), (o,)), ()), Ok(()));
    }

    #[test]
    fn arbitrary_bytes_are_canonical_when_accepted(bytes in prop::collection::vec(any::<u8>(), 0..64)) {
        prop_assert_eq!(laws::canonical::<u64, ()>(&bytes, ()), Ok(()));
        prop_assert_eq!(laws::canonical::<bool, ()>(&bytes, ()), Ok(()));
        prop_assert_eq!(laws::canonical::<Vec<u8>, ()>(&bytes, ()), Ok(()));
        prop_assert_eq!(laws::canonical::<Option<Vec<u32>>, ()>(&bytes, ()), Ok(()));
        prop_assert_eq!(laws::canonical::<[u8; 3], ()>(&bytes, ()), Ok(()));
        prop_assert_eq!(laws::canonical::<(u8, Vec<u64>), ()>(&bytes, ()), Ok(()));
    }

    #[test]
    fn a_longer_form_of_an_integer_is_refused(x in any::<u64>()) {
        // Every unsigned form wide enough for x, smallest first: only the
        // smallest is accepted.
        let forms: [(u8, usize); 4] = [(0xcc, 1), (0xcd, 2), (0xce, 4), (0xcf, 8)];
        let canonical = encode(&x);
        for (tag, width) in forms {
            if width < 8 && x >> (8 * width) != 0 {
                continue;
            }
            let mut bytes = vec![tag];
            bytes.extend(&x.to_be_bytes()[8 - width..]);
            if bytes == canonical {
                prop_assert_eq!(decode::<u64>(&bytes), Ok(x));
            } else {
                prop_assert_eq!(decode::<u64>(&bytes), Err(Error::NotCanonical));
            }
        }
    }

    #[test]
    fn a_longer_form_of_a_length_is_refused(n in 0usize..40) {
        let v = vec![0u8; n];
        let mut array16 = vec![0xdc, 0, n as u8];
        array16.extend(vec![0u8; n]);
        let expected = if n >= 16 { Ok(v.clone()) } else { Err(Error::NotCanonical) };
        prop_assert_eq!(decode::<Vec<u8>>(&array16), expected);
        let mut array32 = vec![0xdd, 0, 0, 0, n as u8];
        array32.extend(vec![0u8; n]);
        prop_assert_eq!(decode::<Vec<u8>>(&array32), Err(Error::NotCanonical));
    }
}

#[test]
fn golden_bytes() {
    // MessagePack's definition: positive fixint, uint8, uint16, uint32,
    // uint64; fixarray, array16; nil, false, true.
    assert_eq!(encode(&0x7fu64), [0x7f]);
    assert_eq!(encode(&0x80u64), [0xcc, 0x80]);
    assert_eq!(encode(&0x100u64), [0xcd, 0x01, 0x00]);
    assert_eq!(encode(&0x1_0000u64), [0xce, 0, 1, 0, 0]);
    assert_eq!(encode(&(1u64 << 32)), [0xcf, 0, 0, 0, 1, 0, 0, 0, 0]);
    assert_eq!(encode(&vec![1u8, 2]), [0x92, 1, 2]);
    let sixteen = encode(&vec![0u8; 16]);
    assert_eq!(&sixteen[..3], [0xdc, 0, 16]);
    assert_eq!(encode(&None::<u8>), [0xc0]);
    assert_eq!(encode(&Some(5u8)), [5]);
    assert_eq!(encode(&(false, true)), [0x92, 0xc2, 0xc3]);
    assert_eq!(encode(&[7u32; 2]), [0x92, 7, 7]);
    assert_eq!(encode(&(3u8,)), [0x91, 3]);
}

#[test]
fn malformed_bytes_are_refused() {
    assert_eq!(decode::<u64>(&[]), Err(Error::Truncated));
    assert_eq!(decode::<u64>(&[0xcd, 1]), Err(Error::Truncated));
    assert_eq!(decode::<u64>(&[0x92]), Err(Error::Unexpected("an integer")));
    assert_eq!(decode::<u8>(&[0xcd, 1, 0]), Err(Error::Invalid("a byte")));
    assert_eq!(
        decode::<[u8; 2]>(&[0x93, 1, 2, 3]),
        Err(Error::Length {
            expected: 2,
            found: 3
        })
    );
    assert_eq!(decode::<u8>(&[1, 2]), Err(Error::Trailing(1)));
    // A length past the bytes left is a truncation, found before
    // allocating for it.
    assert_eq!(
        decode::<Vec<u8>>(&[0xdd, 0xff, 0xff, 0xff, 0xff]),
        Err(Error::Truncated)
    );
}
