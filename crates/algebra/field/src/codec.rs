//! The canonical encoding of prime field elements: the canonical integer in
//! `[0, p)`, little-endian, in `ceil(MODULUS_BITS / 8)` bytes, as a `bin`
//! (arkworks' compressed form, which Kimchi writes).
//!
//! A field is a retract of these byte strings, and the encoding is
//! canonical: a decoder refuses any length but that one and any integer
//! from `p` up, so each element has exactly one encoding.

use alloc::vec::Vec;

use arrabbiata_codec::{Decodable, Encodable, Error, Input, msgpack};

use crate::{Goldilocks, MontField, MontgomeryConfig, PrimeField};

/// The bytes holding `bits` bits.
#[must_use]
pub const fn byte_len(bits: u32) -> usize {
    bits.div_ceil(8) as usize
}

/// The canonical integer of `x`, little-endian, in `len` bytes (those past
/// the integer zero).
#[must_use]
pub fn to_le_bytes<F: PrimeField>(x: F, len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    for i in 0..len {
        bytes.push((x.limb(i / 8) >> (8 * (i % 8))) as u8);
    }
    bytes
}

/// The element whose canonical integer is `bytes`, little-endian, or `None`
/// from the modulus up.
#[must_use]
pub fn from_le_bytes<F: PrimeField>(bytes: &[u8]) -> Option<F> {
    let mut limbs: Vec<u64> = Vec::new();
    for (i, b) in bytes.iter().enumerate() {
        if i % 8 == 0 {
            limbs.push(0);
        }
        let last = limbs.len() - 1;
        limbs[last] |= u64::from(*b) << (8 * (i % 8));
    }
    // Limbs past the field's are zero, or the integer is too large.
    for limb in limbs.iter().skip(F::LIMBS) {
        if *limb != 0 {
            return None;
        }
    }
    limbs.truncate(F::LIMBS);
    F::from_limbs(&limbs)
}

/// `x` as a `bin` of its canonical bytes.
fn write<F: PrimeField>(x: F, out: &mut Vec<u8>) {
    let bytes = to_le_bytes(x, byte_len(F::MODULUS_BITS));
    msgpack::write_bin(out, &bytes);
}

/// The element of the `bin` that follows.
fn read<F: PrimeField>(input: &mut Input<'_>) -> Result<F, Error> {
    let bytes = msgpack::read_bin(input)?;
    if bytes.len() != byte_len(F::MODULUS_BITS) {
        return Err(Error::Invalid("a field element of the right length"));
    }
    match from_le_bytes(bytes) {
        Some(x) => Ok(x),
        None => Err(Error::Invalid("a field element below the modulus")),
    }
}

impl<C, const N: usize> Encodable for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    fn write(&self, out: &mut Vec<u8>) {
        write(*self, out);
    }
}

impl<C, const N: usize> Decodable<()> for MontField<C, N>
where
    C: MontgomeryConfig<N>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        read(input)
    }
}

impl Encodable for Goldilocks {
    fn write(&self, out: &mut Vec<u8>) {
        write(*self, out);
    }
}

impl Decodable<()> for Goldilocks {
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        read(input)
    }
}
