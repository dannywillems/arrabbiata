//! The two laws of a codec, as checks for the tests of every codec.
//!
//! With `write : T -> B` and `read : B -> T` (see the crate's structure
//! section):
//!
//! - [`round_trip`]: `read(write(v)) = v`, the retraction law;
//! - [`canonical`]: `write(read(b)) = b` whenever `read` accepts `b`, i.e.
//!   the idempotent `write . read` is the identity on the domain of `read`.
//!
//! The first quantifies over values, the second over byte strings, so a
//! test feeds [`canonical`] bytes `write` did not produce: arbitrary ones,
//! and the near misses of a real encoding.

use alloc::vec::Vec;

use crate::{Decodable, Encodable, Error, decode_with, encode};

/// A law a codec breaks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Broken {
    /// `read(write(v))` is not `v`; the error, if `read` failed.
    RoundTrip(Option<Error>),
    /// `write(read(b))` is not `b`: these are the bytes `write` gave.
    Canonical(Vec<u8>),
}

/// `read(write(value)) = value`, in `context`.
///
/// # Errors
///
/// [`Broken::RoundTrip`] when it does not hold.
pub fn round_trip<T, A>(value: &T, context: A) -> Result<(), Broken>
where
    T: Encodable + Decodable<A> + PartialEq,
{
    match decode_with::<T, A>(&encode(value), context) {
        Ok(back) if back == *value => Ok(()),
        Ok(_) => Err(Broken::RoundTrip(None)),
        Err(e) => Err(Broken::RoundTrip(Some(e))),
    }
}

/// `write(read(bytes)) = bytes` when `read` accepts `bytes`, in `context`;
/// nothing to check when it refuses them.
///
/// # Errors
///
/// [`Broken::Canonical`] when `read` accepts `bytes` and `write` gives
/// others back.
pub fn canonical<T, A>(bytes: &[u8], context: A) -> Result<(), Broken>
where
    T: Encodable + Decodable<A>,
{
    match decode_with::<T, A>(bytes, context) {
        Ok(value) => {
            let again = encode(&value);
            if again == bytes {
                Ok(())
            } else {
                Err(Broken::Canonical(again))
            }
        }
        Err(_) => Ok(()),
    }
}
