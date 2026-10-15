//! A Fiat-Shamir transcript from any 256-bit hash function of bytes: the
//! oracle a post-quantum protocol uses, where an arithmetic sponge over the
//! field has no reason to exist.
//!
//! The hash is a parameter ([`ByteHash`]): SHA-256, SHA3-256, BLAKE3 or any
//! other implementation plugs in, and this crate depends on none of them.
//! The transcript is a hash chain over a 32-byte state:
//!
//! ```text
//! start:    s = H("arrabbiata/hash-transcript" || domain)
//! absorb x: s = H(s || 0x01 || x)                 (x little-endian limbs)
//! squeeze:  c = (H(s || 0x02 || 0) || H(s || 0x02 || 1) || ...) mod p,
//!           2 |p| bits at least, then s = H(s || 0x03)
//! ```
//!
//! In the random oracle model each challenge is uniform given everything
//! absorbed before it, up to a statistical distance below `2^-|p|` (the
//! reduction of at least `2 |p|` uniform bits). Each step's input is
//! prefixed with its tag and has a fixed length per field, so distinct
//! histories hash distinct strings. Field elements are encoded by their
//! canonical limbs, so the transcript is defined for [`PrimeField`]s.

use core::marker::PhantomData;

use arrabbiata_field::{Field, PrimeField};

use crate::Transcript;

/// A hash function from bytes to 32 bytes, modelled as a random oracle.
pub trait ByteHash {
    /// The hash of the concatenation of `parts`.
    fn hash(&self, parts: &[&[u8]]) -> [u8; 32];
}

/// The tag of an absorbed element.
const ABSORB: u8 = 0x01;
/// The tag of a squeezed block.
const SQUEEZE: u8 = 0x02;
/// The tag of the state update after a squeeze.
const RATCHET: u8 = 0x03;
/// The prefix of the initial state.
const NAME: &[u8] = b"arrabbiata/hash-transcript";
/// The largest number of 64-bit limbs of a field this transcript encodes.
const MAX_LIMBS: usize = 8;

/// A Fiat-Shamir transcript over `F` from the hash `H`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashTranscript<F, H> {
    hash: H,
    state: [u8; 32],
    field: PhantomData<fn() -> F>,
}

impl<F, H> HashTranscript<F, H>
where
    F: PrimeField,
    H: ByteHash,
{
    /// A fresh transcript, separated by `domain`.
    ///
    /// # Panics
    ///
    /// If the field has more than 512 bits.
    #[must_use]
    pub fn new(hash: H, domain: u64) -> Self {
        assert!(F::LIMBS <= MAX_LIMBS, "the field has more than 512 bits");
        let state = hash.hash(&[NAME, &domain.to_le_bytes()]);
        Self {
            hash,
            state,
            field: PhantomData,
        }
    }
}

/// `2^64` in `F`.
fn two_64<F>() -> F
where
    F: Field,
{
    let two_32 = F::from_u64(1 << 32);
    two_32 * two_32
}

impl<F, H> Transcript<F> for HashTranscript<F, H>
where
    F: PrimeField,
    H: ByteHash,
{
    fn absorb(&mut self, value: F) {
        let mut bytes = [0u8; 8 * MAX_LIMBS];
        let mut i = 0;
        while i < F::LIMBS {
            let limb = value.limb(i).to_le_bytes();
            let mut j = 0;
            while j < 8 {
                bytes[8 * i + j] = limb[j];
                j += 1;
            }
            i += 1;
        }
        let encoded = &bytes[..8 * F::LIMBS];
        self.state = self.hash.hash(&[&self.state, &[ABSORB], encoded]);
    }

    fn squeeze(&mut self) -> F {
        // At least twice the field's bits: 16 LIMBS bytes, 32 per block.
        let blocks = (16 * F::LIMBS).div_ceil(32);
        let mut value = F::ZERO;
        let base = two_64::<F>();
        let mut k = 0;
        while k < blocks {
            let block = self
                .hash
                .hash(&[&self.state, &[SQUEEZE], &(k as u64).to_le_bytes()]);
            let mut w = 0;
            while w < 4 {
                let mut word = [0u8; 8];
                let mut j = 0;
                while j < 8 {
                    word[j] = block[8 * w + j];
                    j += 1;
                }
                value = value * base + F::from_u64(u64::from_le_bytes(word));
                w += 1;
            }
            k += 1;
        }
        self.state = self.hash.hash(&[&self.state, &[RATCHET]]);
        value
    }
}
