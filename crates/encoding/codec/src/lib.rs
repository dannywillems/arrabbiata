//! One codec for every canonical binary encoding of Arrabbiata.
//!
//! A type with a canonical encoding implements [`Encodable`] (write it) and
//! [`Decodable`] (read it back); [`Codec`] is the alias over the pair for
//! context-free encodings. Everything that goes to bytes in the workspace
//! goes through these traits: the field elements, the curve points, and the
//! structures built from them (Kimchi's proofs).
//!
//! The split follows the two operations' shapes: writing takes `&self` and
//! cannot fail, reading is an associated function that needs `Self: Sized`
//! and can. [`Decodable`]'s type parameter is the context a decoder needs
//! and the bytes do not carry; it is `()` for self-describing encodings,
//! and a type parameter rather than an associated type so that one type can
//! decode under a family of contexts (a relation, not a function, from
//! types to contexts).
//!
//! # The byte format
//!
//! The canonical format is MessagePack with every struct written as the
//! array of its fields in declaration order, the form `rmp_serde::to_vec`
//! produces, which is how Kimchi serializes its proofs. [`msgpack`] has its
//! primitives; this crate implements the containers:
//!
//! | type | bytes |
//! | --- | --- |
//! | `Vec<T>`, `[T]` | an array of the elements |
//! | `[T; N]` | an array of `N` elements |
//! | `(A, B, ..)` | an array of the members |
//! | `Option<T>` | `nil`, or the value |
//! | `u8`, `u32`, `u64`, `usize` | a MessagePack integer, smallest form |
//! | `bool` | `false` or `true` |
//!
//! A field element or a point is a `bin` of arkworks' compressed bytes; its
//! own crate implements the traits for it.
//!
//! # The structure
//!
//! Write `B` for the byte strings. A codec for `T` is a total `write : T ->
//! B` and a partial `read : B -> T`, with the law `read . write = id_T`
//! (composition right to left): `write` is a section, `read` a retraction,
//! and `T` a retract of `B`. Then `e = write . read`, partial on `B`, is
//! idempotent,
//!
//! ```text
//!     e . e = write . (read . write) . read = write . read = e
//! ```
//!
//! and `T` is the splitting of `e`. Two properties follow, and they are not
//! the same one:
//!
//! - round trip, `read(write(v)) = v` for every value: the retraction law;
//!   re-encoding the decoded value adds nothing, since it follows from it;
//! - canonicity, `write(read(b)) = b` for every `b` read accepts, i.e. `e`
//!   is the identity on the domain of `read`: no value has two encodings. It
//!   quantifies over byte strings, not values, so it is checked by feeding
//!   `read` bytes `write` did not produce. MessagePack has several forms of
//!   the same array length or integer; the decoder here accepts only the
//!   smallest, which is the form `write` produces, and so is canonical.
//!
//! [`laws`] states both as checks.
//!
//! `T` occurs in `write` as an argument and in `read` as a result, so a
//! codec is an invariant functor in `T`: a codec for `T` transports to `U`
//! only along a (partial) isomorphism `T <-> U`, never along a function. The
//! container impls are the action of the list, option, power and product
//! functors on codecs, which is why one blanket impl each composes to any
//! depth. Rendel and Ostermann's invertible syntax descriptions (Haskell
//! Symposium 2010) build parser-printers on exactly these pieces (`IsoFunctor`,
//! `ProductFunctor`, `Alternative`); without higher kinds they appear here as
//! impls. Choice is only prefix-determined (`nil` or a value), as MessagePack
//! is read once, from the front.
//!
//! Coherence allows one impl per type, so a type has one encoding; a second
//! encoding of the same carrier would be a second idempotent on `B`, and
//! lives on a newtype.
//!
//! The design is that of zcash/librustzcash#2809 (`zcash_encoding`'s
//! `Encodable`/`Decodable`), on a different byte format.

#![no_std]

extern crate alloc;

pub mod laws;
pub mod msgpack;

use alloc::vec::Vec;

/// Why bytes are not the canonical encoding of a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The bytes end inside a value.
    Truncated,
    /// A value of another kind than the one expected: an array, bytes, an
    /// integer, `nil`.
    Unexpected(&'static str),
    /// An array of another length than a fixed-size value's.
    Length {
        /// The length expected.
        expected: usize,
        /// The length read.
        found: usize,
    },
    /// A value in a longer form than its smallest, which `write` never
    /// produces: accepting it would give one value two encodings.
    NotCanonical,
    /// Bytes that are no value of the type (a field element not below the
    /// modulus, a point off the curve).
    Invalid(&'static str),
    /// Bytes left after the value.
    Trailing(usize),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "the bytes end inside a value"),
            Self::Unexpected(what) => write!(f, "expected {what}"),
            Self::Length { expected, found } => {
                write!(f, "an array of {found} values, expected {expected}")
            }
            Self::NotCanonical => write!(f, "a value in a longer form than its smallest"),
            Self::Invalid(what) => write!(f, "not {what}"),
            Self::Trailing(n) => write!(f, "{n} bytes after the value"),
        }
    }
}

/// The bytes a decoder reads, front to back.
#[derive(Clone, Debug)]
pub struct Input<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Input<'a> {
    /// The input `bytes`, from their first byte.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// The next `n` bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Truncated`] when fewer are left.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let Some(end) = self.at.checked_add(n) else {
            return Err(Error::Truncated);
        };
        match self.bytes.get(self.at..end) {
            Some(slice) => {
                self.at = end;
                Ok(slice)
            }
            None => Err(Error::Truncated),
        }
    }

    /// The next byte.
    ///
    /// # Errors
    ///
    /// [`Error::Truncated`] at the end.
    pub fn byte(&mut self) -> Result<u8, Error> {
        match self.take(1)? {
            [b] => Ok(*b),
            _ => Err(Error::Truncated),
        }
    }

    /// The next byte, left unread.
    ///
    /// # Errors
    ///
    /// [`Error::Truncated`] at the end.
    pub fn peek(&self) -> Result<u8, Error> {
        match self.bytes.get(self.at) {
            Some(b) => Ok(*b),
            None => Err(Error::Truncated),
        }
    }

    /// The bytes left.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }
}

/// A type with a canonical binary encoding.
pub trait Encodable {
    /// Appends the encoding of `self` to `out`.
    fn write(&self, out: &mut Vec<u8>);

    /// The number of bytes [`Encodable::write`] appends.
    fn encoded_len(&self) -> usize {
        let mut out = Vec::new();
        self.write(&mut out);
        out.len()
    }
}

/// A type read back from its canonical encoding, given the context `A` the
/// bytes do not carry (`()` when they carry everything).
pub trait Decodable<A>: Sized {
    /// Reads a value from the front of `input`.
    ///
    /// # Errors
    ///
    /// An [`Error`] when the bytes are not the canonical encoding of a
    /// value.
    fn read(input: &mut Input<'_>, context: A) -> Result<Self, Error>;
}

/// A type with a complete, context-free codec.
pub trait Codec: Encodable + Decodable<()> {}

impl<T> Codec for T where T: Encodable + Decodable<()> {}

/// The encoding of `value`.
#[must_use]
pub fn encode<T>(value: &T) -> Vec<u8>
where
    T: Encodable + ?Sized,
{
    let mut out = Vec::new();
    value.write(&mut out);
    out
}

/// The value `bytes` encode, in `context`, all the bytes read.
///
/// # Errors
///
/// An [`Error`] when they are not one value's canonical encoding, or bytes
/// are left after it.
pub fn decode_with<T, A>(bytes: &[u8], context: A) -> Result<T, Error>
where
    T: Decodable<A>,
{
    let mut input = Input::new(bytes);
    let value = T::read(&mut input, context)?;
    match input.remaining() {
        0 => Ok(value),
        n => Err(Error::Trailing(n)),
    }
}

/// [`decode_with`] a context-free codec.
///
/// # Errors
///
/// As [`decode_with`].
pub fn decode<T>(bytes: &[u8]) -> Result<T, Error>
where
    T: Decodable<()>,
{
    decode_with(bytes, ())
}

// The containers: the action of the list, power, product and option
// functors on codecs.

impl<T> Encodable for [T]
where
    T: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, self.len());
        for x in self {
            x.write(out);
        }
    }
}

impl<T> Encodable for Vec<T>
where
    T: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        self.as_slice().write(out);
    }
}

impl<A, T> Decodable<A> for Vec<T>
where
    A: Copy,
    T: Decodable<A>,
{
    fn read(input: &mut Input<'_>, context: A) -> Result<Self, Error> {
        let n = msgpack::read_array(input)?;
        // At least one byte per element: a length past the input is a
        // truncation, found before allocating for it.
        if n > input.remaining() {
            return Err(Error::Truncated);
        }
        let mut all = Vec::with_capacity(n);
        for _ in 0..n {
            all.push(T::read(input, context)?);
        }
        Ok(all)
    }
}

impl<T, const N: usize> Encodable for [T; N]
where
    T: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        self.as_slice().write(out);
    }
}

impl<A, T, const N: usize> Decodable<A> for [T; N]
where
    A: Copy,
    T: Decodable<A>,
{
    fn read(input: &mut Input<'_>, context: A) -> Result<Self, Error> {
        msgpack::read_fields(input, N)?;
        let mut all = Vec::with_capacity(N);
        for _ in 0..N {
            all.push(T::read(input, context)?);
        }
        match all.try_into() {
            Ok(array) => Ok(array),
            Err(_) => Err(Error::Truncated),
        }
    }
}

impl<T> Encodable for Option<T>
where
    T: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Some(x) => x.write(out),
            None => msgpack::write_nil(out),
        }
    }
}

impl<A, T> Decodable<A> for Option<T>
where
    T: Decodable<A>,
{
    fn read(input: &mut Input<'_>, context: A) -> Result<Self, Error> {
        if msgpack::read_nil(input)? {
            Ok(None)
        } else {
            T::read(input, context).map(Some)
        }
    }
}

impl<T> Encodable for &T
where
    T: Encodable + ?Sized,
{
    fn write(&self, out: &mut Vec<u8>) {
        (**self).write(out);
    }
}

impl<T0> Encodable for (T0,)
where
    T0: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, 1);
        self.0.write(out);
    }
}

impl<A, T0> Decodable<A> for (T0,)
where
    T0: Decodable<A>,
{
    fn read(input: &mut Input<'_>, context: A) -> Result<Self, Error> {
        msgpack::read_fields(input, 1)?;
        Ok((T0::read(input, context)?,))
    }
}

impl<T0, T1> Encodable for (T0, T1)
where
    T0: Encodable,
    T1: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, 2);
        self.0.write(out);
        self.1.write(out);
    }
}

impl<A, T0, T1> Decodable<A> for (T0, T1)
where
    A: Copy,
    T0: Decodable<A>,
    T1: Decodable<A>,
{
    fn read(input: &mut Input<'_>, context: A) -> Result<Self, Error> {
        msgpack::read_fields(input, 2)?;
        Ok((T0::read(input, context)?, T1::read(input, context)?))
    }
}

impl Encodable for bool {
    fn write(&self, out: &mut Vec<u8>) {
        out.push(if *self { 0xc3 } else { 0xc2 });
    }
}

impl Decodable<()> for bool {
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        match input.byte()? {
            0xc2 => Ok(false),
            0xc3 => Ok(true),
            _ => Err(Error::Unexpected("a boolean")),
        }
    }
}

impl Encodable for u64 {
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_uint(out, *self);
    }
}

impl Decodable<()> for u64 {
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        msgpack::read_uint(input)
    }
}

impl Encodable for u32 {
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_uint(out, u64::from(*self));
    }
}

impl Decodable<()> for u32 {
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        match Self::try_from(msgpack::read_uint(input)?) {
            Ok(x) => Ok(x),
            Err(_) => Err(Error::Invalid("a 32-bit integer")),
        }
    }
}

impl Encodable for u8 {
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_uint(out, u64::from(*self));
    }
}

impl Decodable<()> for u8 {
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        match Self::try_from(msgpack::read_uint(input)?) {
            Ok(x) => Ok(x),
            Err(_) => Err(Error::Invalid("a byte")),
        }
    }
}

impl Encodable for usize {
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_uint(out, *self as u64);
    }
}

impl Decodable<()> for usize {
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        match Self::try_from(msgpack::read_uint(input)?) {
            Ok(x) => Ok(x),
            Err(_) => Err(Error::Invalid("a size")),
        }
    }
}
