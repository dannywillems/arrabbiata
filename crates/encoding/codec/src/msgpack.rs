//! The MessagePack the canonical format uses, written in its smallest form
//! and read only in it.
//!
//! MessagePack has several forms of one array length (`fixarray`,
//! `array16`, `array32`) and one integer (`positive fixint`, `uint8` ..
//! `uint64`); `rmp_serde` writes the smallest, and so do the writers here.
//! The readers refuse any other with [`Error::NotCanonical`], which is what
//! makes every codec built on them canonical (see the crate's structure
//! section).

use alloc::vec::Vec;

use crate::{Error, Input};

/// `nil`.
const NIL: u8 = 0xc0;

/// An array of `n` values follows.
pub fn write_array(out: &mut Vec<u8>, n: usize) {
    if n < 16 {
        out.push(0x90 | n as u8);
    } else if let Ok(n) = u16::try_from(n) {
        out.push(0xdc);
        out.extend(n.to_be_bytes());
    } else {
        out.push(0xdd);
        out.extend((n as u32).to_be_bytes());
    }
}

/// The length of the array that follows.
///
/// # Errors
///
/// [`Error::Unexpected`] when no array follows, [`Error::NotCanonical`]
/// when its length is not in its smallest form.
pub fn read_array(input: &mut Input<'_>) -> Result<usize, Error> {
    let (n, smallest) = match input.byte()? {
        b @ 0x90..=0x9f => return Ok(usize::from(b & 0x0f)),
        0xdc => (big_endian(input.take(2)?), 16),
        0xdd => (big_endian(input.take(4)?), 0x1_0000),
        _ => return Err(Error::Unexpected("an array")),
    };
    // `array16` only from 16 values, `array32` only past 65535.
    if n < smallest {
        Err(Error::NotCanonical)
    } else {
        Ok(n)
    }
}

/// An array of exactly `n` values follows: a struct of `n` fields, a
/// tuple, a fixed-size array.
///
/// # Errors
///
/// As [`read_array`], and [`Error::Length`] for another length.
pub fn read_fields(input: &mut Input<'_>, n: usize) -> Result<(), Error> {
    let found = read_array(input)?;
    if found == n {
        Ok(())
    } else {
        Err(Error::Length { expected: n, found })
    }
}

/// `nil`.
pub fn write_nil(out: &mut Vec<u8>) {
    out.push(NIL);
}

/// Whether `nil` follows, read if so.
///
/// # Errors
///
/// [`Error::Truncated`] at the end.
pub fn read_nil(input: &mut Input<'_>) -> Result<bool, Error> {
    if input.peek()? == NIL {
        input.byte()?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// `bytes`, as a MessagePack `bin`.
pub fn write_bin(out: &mut Vec<u8>, bytes: &[u8]) {
    let n = bytes.len();
    if let Ok(n) = u8::try_from(n) {
        out.push(0xc4);
        out.push(n);
    } else if let Ok(n) = u16::try_from(n) {
        out.push(0xc5);
        out.extend(n.to_be_bytes());
    } else {
        out.push(0xc6);
        out.extend((n as u32).to_be_bytes());
    }
    out.extend(bytes);
}

/// The bytes of the `bin` that follows.
///
/// # Errors
///
/// [`Error::Unexpected`] when no `bin` follows, [`Error::NotCanonical`]
/// when its length is not in its smallest form.
pub fn read_bin<'a>(input: &mut Input<'a>) -> Result<&'a [u8], Error> {
    let (n, width) = match input.byte()? {
        0xc4 => (big_endian(input.take(1)?), 1),
        0xc5 => (big_endian(input.take(2)?), 2),
        0xc6 => (big_endian(input.take(4)?), 4),
        _ => return Err(Error::Unexpected("bytes")),
    };
    let smallest = if n <= 0xff {
        1
    } else if n <= 0xffff {
        2
    } else {
        4
    };
    if width != smallest {
        return Err(Error::NotCanonical);
    }
    input.take(n)
}

/// `x`, in the smallest of MessagePack's unsigned forms.
pub fn write_uint(out: &mut Vec<u8>, x: u64) {
    if x < 0x80 {
        out.push(x as u8);
    } else if let Ok(x) = u8::try_from(x) {
        out.push(0xcc);
        out.push(x);
    } else if let Ok(x) = u16::try_from(x) {
        out.push(0xcd);
        out.extend(x.to_be_bytes());
    } else if let Ok(x) = u32::try_from(x) {
        out.push(0xce);
        out.extend(x.to_be_bytes());
    } else {
        out.push(0xcf);
        out.extend(x.to_be_bytes());
    }
}

/// The unsigned integer that follows.
///
/// # Errors
///
/// [`Error::Unexpected`] when none follows, [`Error::NotCanonical`] when it
/// is not in its smallest form.
pub fn read_uint(input: &mut Input<'_>) -> Result<u64, Error> {
    let (x, width) = match input.byte()? {
        b @ 0x00..=0x7f => return Ok(u64::from(b)),
        0xcc => (big_endian(input.take(1)?), 1),
        0xcd => (big_endian(input.take(2)?), 2),
        0xce => (big_endian(input.take(4)?), 4),
        0xcf => (big_endian(input.take(8)?), 8),
        _ => return Err(Error::Unexpected("an integer")),
    };
    let x = x as u64;
    let mut smallest = Vec::new();
    write_uint(&mut smallest, x);
    if smallest.len() == 1 + width {
        Ok(x)
    } else {
        Err(Error::NotCanonical)
    }
}

/// Skips the value that follows, whatever it is: for a struct's fields a
/// decoder does not need.
///
/// # Errors
///
/// [`Error::Truncated`] when the value is cut, [`Error::Unexpected`] on a
/// byte that starts no MessagePack value.
pub fn skip(input: &mut Input<'_>) -> Result<(), Error> {
    let b = input.byte()?;
    let (inline, elements) = match b {
        0x00..=0x7f | 0xe0..=0xff | 0xc0 | 0xc2 | 0xc3 => (0, 0),
        0x80..=0x8f => (0, 2 * usize::from(b & 0x0f)),
        0x90..=0x9f => (0, usize::from(b & 0x0f)),
        0xa0..=0xbf => (usize::from(b & 0x1f), 0),
        0xc4 | 0xd9 => (big_endian(input.take(1)?), 0),
        0xc5 | 0xda => (big_endian(input.take(2)?), 0),
        0xc6 | 0xdb => (big_endian(input.take(4)?), 0),
        0xcc | 0xd0 => (1, 0),
        0xcd | 0xd1 => (2, 0),
        0xca | 0xce | 0xd2 => (4, 0),
        0xcb | 0xcf | 0xd3 => (8, 0),
        0xd4 => (2, 0),
        0xd5 => (3, 0),
        0xd6 => (5, 0),
        0xd7 => (9, 0),
        0xd8 => (17, 0),
        0xc7 => (big_endian(input.take(1)?) + 1, 0),
        0xc8 => (big_endian(input.take(2)?) + 1, 0),
        0xc9 => (big_endian(input.take(4)?) + 1, 0),
        0xdc => (0, big_endian(input.take(2)?)),
        0xdd => (0, big_endian(input.take(4)?)),
        0xde => (0, 2 * big_endian(input.take(2)?)),
        0xdf => (0, 2 * big_endian(input.take(4)?)),
        _ => return Err(Error::Unexpected("a MessagePack value")),
    };
    input.take(inline)?;
    for _ in 0..elements {
        skip(input)?;
    }
    Ok(())
}

/// The big-endian integer of `bytes` (at most 8 of them).
fn big_endian(bytes: &[u8]) -> usize {
    let mut x = 0usize;
    for b in bytes {
        x = (x << 8) | usize::from(*b);
    }
    x
}
