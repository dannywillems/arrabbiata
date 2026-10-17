//! Randomness as a parameter, so that a proof is a function of its inputs
//! and a run of Kimchi can be reproduced.
//!
//! A [`RandomSource`] is a stream of `u64`; a [`Sampler`] turns a stream
//! into field elements. [`Arkworks`] samples as `ark-ff` 0.5 does
//! (`UniformRand` for `Fp`), which is what Kimchi's prover calls: it reads
//! `LIMBS` words as the element's MONTGOMERY representation, so the value
//! is `limbs 2^(-64 LIMBS)`, after clearing the bits above the modulus and
//! rejecting a representation at least the modulus. Given the same stream,
//! it gives the same elements as Kimchi, in the same order.
//!
//! [`SplitMix64`] is a small, well-known stream for tests (Steele, Lea and
//! Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014);
//! it is not a cryptographic generator. A prover that must hide its witness
//! needs a [`RandomSource`] over a cryptographic generator.

use arrabbiata_field::PrimeField;

/// A stream of 64-bit words.
pub trait RandomSource {
    /// The next word.
    fn next_u64(&mut self) -> u64;
}

/// A source of field elements.
pub trait Sampler<F> {
    /// The next element.
    fn sample(&mut self) -> F;
}

/// SplitMix64 from a seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64(pub u64);

impl RandomSource for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Field elements sampled from a stream as `ark-ff` samples them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arkworks<R>(pub R);

impl<F, R> Sampler<F> for Arkworks<R>
where
    F: PrimeField,
    R: RandomSource,
{
    fn sample(&mut self) -> F {
        let limbs = F::LIMBS;
        let shave = 64 * limbs as u32 - F::MODULUS_BITS;
        let mask = if shave >= 64 { 0 } else { u64::MAX >> shave };
        // R^(-1) with R = 2^(64 limbs): the Montgomery representation `m`
        // stands for `m R^(-1)`.
        let two_to_the_64 = F::from_u64(1 << 32) * F::from_u64(1 << 32);
        let r = two_to_the_64.pow(limbs as u64);
        let r_inverse = match r.inverse() {
            Some(x) => x,
            None => unreachable!("a power of two is invertible in an odd field"),
        };
        loop {
            let mut words: alloc::vec::Vec<u64> = (0..limbs).map(|_| self.0.next_u64()).collect();
            if let Some(top) = words.last_mut() {
                *top &= mask;
            }
            if let Some(m) = F::from_limbs(&words) {
                return m * r_inverse;
            }
        }
    }
}
