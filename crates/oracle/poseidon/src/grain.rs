//! The Grain LFSR that generates Poseidon's parameters.
//!
//! Poseidon is a family: its round constants and MDS matrix are not chosen,
//! they are the output of a fixed pseudorandom generator seeded with the
//! parameters (field, S-box, width, round numbers). Two implementations
//! agree on the permutation only if they run this generator identically.
//! This follows the reference script of Grassi, Khovratovich, Rechberger,
//! Roy and Schofnegger (`generate_parameters_grain.sage`), as implemented in
//! `halo2_poseidon`, and is checked against its test vectors.
//!
//! The generator is an 80-bit LFSR with feedback
//! `b[i+80] = b[i+62] + b[i+51] + b[i+38] + b[i+23] + b[i+13] + b[i]`, whose
//! output is thinned by the self-shrinking rule (from each pair of bits, emit
//! the second if the first is 1).
//!
//! ```
//! use arrabbiata_pasta::Fp;
//! use arrabbiata_poseidon::grain::Grain;
//! let mut a = Grain::<Fp>::new(3, 8, 56);
//! let mut b = Grain::<Fp>::new(3, 8, 56);
//! assert_eq!(a.next_field_element(), b.next_field_element());
//! ```

use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use arrabbiata_field::PrimeField;

const STATE: usize = 80;

/// The Grain LFSR, producing elements of `F`.
///
/// ```
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_poseidon::grain::Grain;
/// let mut g = Grain::<Fq>::new(3, 8, 56);
/// assert_ne!(g.next_field_element(), g.next_field_element());
/// ```
#[derive(Clone, Debug)]
pub struct Grain<F> {
    state: [bool; STATE],
    next_bit: usize,
    field: PhantomData<F>,
}

impl<F> Grain<F>
where
    F: PrimeField,
{
    /// The generator for a prime field, the S-box `x^alpha`, width `t`, and
    /// `r_f` full and `r_p` partial rounds.
    ///
    /// ```
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::grain::Grain;
    /// let _ = Grain::<Fp>::new(3, 8, 56);
    /// ```
    #[must_use]
    pub fn new(t: u16, r_f: u16, r_p: u16) -> Self {
        let mut state = [true; STATE];
        // Field type: 1 for a prime field. S-box type: 0 for x^alpha.
        set_bits(&mut state, 0, 2, 1);
        set_bits(&mut state, 2, 4, 0);
        set_bits(&mut state, 6, 12, F::MODULUS_BITS as u16);
        set_bits(&mut state, 18, 12, t);
        set_bits(&mut state, 30, 10, r_f);
        set_bits(&mut state, 40, 10, r_p);
        let mut grain = Self {
            state,
            next_bit: STATE,
            field: PhantomData,
        };
        // Discard the first 160 bits.
        let mut i = 0;
        while i < 20 {
            grain.load_next_8_bits();
            grain.next_bit = STATE;
            i += 1;
        }
        grain
    }

    fn load_next_8_bits(&mut self) {
        let mut new_bits = [false; 8];
        let mut i = 0;
        while i < 8 {
            new_bits[i] = self.state[i + 62]
                ^ self.state[i + 51]
                ^ self.state[i + 38]
                ^ self.state[i + 23]
                ^ self.state[i + 13]
                ^ self.state[i];
            i += 1;
        }
        self.state.rotate_left(8);
        self.next_bit -= 8;
        let mut j = 0;
        while j < 8 {
            self.state[self.next_bit + j] = new_bits[j];
            j += 1;
        }
    }

    fn raw_bit(&mut self) -> bool {
        if self.next_bit == STATE {
            self.load_next_8_bits();
        }
        let bit = self.state[self.next_bit];
        self.next_bit += 1;
        bit
    }

    /// The next output bit, after self-shrinking.
    fn next_bit(&mut self) -> bool {
        while !self.raw_bit() {
            self.raw_bit();
        }
        self.raw_bit()
    }

    /// The next `MODULUS_BITS` output bits, most significant first, as limbs.
    fn next_limbs(&mut self) -> Vec<u64> {
        let bits = F::MODULUS_BITS as usize;
        let mut limbs = vec![0u64; F::LIMBS];
        let mut k = 0;
        while k < bits {
            let position = bits - 1 - k;
            if self.next_bit() {
                limbs[position / 64] |= 1 << (position % 64);
            }
            k += 1;
        }
        limbs
    }

    /// The next field element, by rejection sampling: integers not below the
    /// modulus are discarded. Used for the round constants.
    ///
    /// ```
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::grain::Grain;
    /// let _ = Grain::<Fp>::new(3, 8, 56).next_field_element();
    /// ```
    pub fn next_field_element(&mut self) -> F {
        loop {
            if let Some(element) = F::from_limbs(&self.next_limbs()) {
                return element;
            }
        }
    }

    /// The next field element, reduced modulo `p` without rejection. Used for
    /// the MDS matrix.
    ///
    /// ```
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::grain::Grain;
    /// let _ = Grain::<Fp>::new(3, 8, 56).next_field_element_without_rejection();
    /// ```
    pub fn next_field_element_without_rejection(&mut self) -> F {
        let limbs = self.next_limbs();
        let two_32 = F::from_u64(1 << 32);
        let two_64 = two_32 * two_32;
        let mut acc = F::ZERO;
        let mut i = limbs.len();
        while i > 0 {
            i -= 1;
            acc = acc * two_64 + F::from_u64(limbs[i]);
        }
        acc
    }
}

/// Write `len` bits of `value`, most significant first, at `offset`.
fn set_bits(state: &mut [bool; STATE], offset: usize, len: usize, value: u16) {
    let mut i = 0;
    while i < len {
        state[offset + len - 1 - i] = (value >> i) & 1 != 0;
        i += 1;
    }
}
