//! The behaviour of each public item, one test per item.

use arrabbiata_codec::{
    Codec, Decodable, Encodable, Error, Input, decode, decode_with, encode,
    laws::{self, Broken},
    msgpack,
};

/// A value decoded under a context: an integer offset by it.
#[derive(Debug, PartialEq)]
struct Offset(u64);

impl Encodable for Offset {
    fn write(&self, out: &mut Vec<u8>) {
        self.0.write(out);
    }
}

impl Decodable<u64> for Offset {
    fn read(input: &mut Input<'_>, context: u64) -> Result<Self, Error> {
        Ok(Self(u64::read(input, ())? + context))
    }
}

#[test]
fn error() {
    assert_eq!(Error::Truncated.to_string(), "the bytes end inside a value");
    assert_eq!(Error::Trailing(2).to_string(), "2 bytes after the value");
    assert_eq!(
        Error::Length {
            expected: 1,
            found: 2
        }
        .to_string(),
        "an array of 2 values, expected 1"
    );
}

#[test]
fn input() {
    let mut input = Input::new(&[1, 2, 3]);
    assert_eq!(input.peek(), Ok(1));
    assert_eq!(input.byte(), Ok(1));
    assert_eq!(input.take(2), Ok(&[2u8, 3][..]));
    assert_eq!(input.remaining(), 0);
    assert_eq!(input.byte(), Err(Error::Truncated));
}

#[test]
fn encodable() {
    assert_eq!(300u64.encoded_len(), 3);
    let mut out = Vec::new();
    true.write(&mut out);
    assert_eq!(out, [0xc3]);
}

#[test]
fn decodable() {
    // The context the bytes do not carry.
    let mut input = Input::new(&[5]);
    assert_eq!(Offset::read(&mut input, 10), Ok(Offset(15)));
}

#[test]
fn codec() {
    fn both<T: Codec>(x: &T) -> Vec<u8> {
        encode(x)
    }
    assert_eq!(both(&vec![1u8]), [0x91, 1]);
}

#[test]
fn encode_decode() {
    let bytes = encode(&Some(vec![1u32, 200]));
    assert_eq!(decode::<Option<Vec<u32>>>(&bytes), Ok(Some(vec![1, 200])));
}

#[test]
fn decode_with_context() {
    assert_eq!(decode_with::<Offset, u64>(&[5], 1), Ok(Offset(6)));
    assert_eq!(
        decode_with::<Offset, u64>(&[5, 0], 1),
        Err(Error::Trailing(1))
    );
}

#[test]
fn laws_round_trip_and_canonical() {
    assert_eq!(laws::round_trip(&7u8, ()), Ok(()));
    assert_eq!(laws::canonical::<u8, ()>(&[7], ()), Ok(()));
    // Refused bytes have nothing to check.
    assert_eq!(laws::canonical::<u8, ()>(&[0xcc, 7], ()), Ok(()));
    let _ = Broken::Canonical(vec![]);
}

#[test]
fn msgpack_primitives() {
    let mut out = Vec::new();
    msgpack::write_array(&mut out, 2);
    msgpack::write_nil(&mut out);
    msgpack::write_bin(&mut out, &[9, 9]);
    msgpack::write_uint(&mut out, 1000);
    assert_eq!(out, [0x92, 0xc0, 0xc4, 2, 9, 9, 0xcd, 0x03, 0xe8]);
    let mut input = Input::new(&out);
    assert_eq!(msgpack::read_array(&mut input), Ok(2));
    assert_eq!(msgpack::read_nil(&mut input), Ok(true));
    assert_eq!(msgpack::read_bin(&mut input), Ok(&[9u8, 9][..]));
    assert_eq!(msgpack::read_uint(&mut input), Ok(1000));
    let mut fields = Input::new(&[0x92]);
    assert_eq!(
        msgpack::read_fields(&mut fields, 3),
        Err(Error::Length {
            expected: 3,
            found: 2
        })
    );
}

#[test]
fn msgpack_skip() {
    // A map, a string and a negative integer, none of which a codec here
    // writes, skipped whole.
    let bytes = [0x81, 0xa1, b'k', 0xff, 0x07];
    let mut input = Input::new(&bytes);
    assert_eq!(msgpack::skip(&mut input), Ok(()));
    assert_eq!(input.remaining(), 1);
}
