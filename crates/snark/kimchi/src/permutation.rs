//! Kimchi's permutation shifts (`kimchi/src/circuits/polynomials/
//! permutation.rs`, `Shifts::new`).
//!
//! The copy argument labels cell `(column, row)` with `k_column w^row`;
//! the labels are distinct when the shifts `k_i` lie in distinct cosets of
//! the domain `H`. Kimchi takes `k_0 = 1` and samples the others by
//! hashing a counter: from `7`, increment, take the first 31 bytes of
//! `Blake2b512(counter as 4 big-endian bytes)` as a little-endian integer
//! (below the modulus, being 248 bits), and accept it when it is a
//! quadratic non-residue, not in `H` (`k^n != 1`), and not already taken.
//!
//! A non-residue is not in `H` when `n` divides `(p - 1) / 2`, but Kimchi
//! checks both, and so does this.

use blake2::{Blake2b512, Digest};

use arrabbiata_field::{Field, PrimeField};

/// The number of permuted columns.
pub const PERMUTS: usize = 7;

/// `x^e` for `e` given by its 64-bit limbs, least significant first.
fn pow_limbs<F: Field>(x: F, e: &[u64]) -> F {
    let mut acc = F::ONE;
    let mut i = e.len();
    while i > 0 {
        i -= 1;
        let mut bit = 64;
        while bit > 0 {
            bit -= 1;
            acc = acc * acc;
            if (e[i] >> bit) & 1 == 1 {
                acc = acc * x;
            }
        }
    }
    acc
}

/// Whether `x` is a quadratic non-residue: `x^((p - 1) / 2) = -1`.
fn is_non_residue<F: PrimeField>(x: F) -> bool {
    let p_minus_1 = -F::ONE;
    let limbs: alloc::vec::Vec<u64> = (0..F::LIMBS).map(|i| p_minus_1.limb(i)).collect();
    let mut half = alloc::vec![0u64; limbs.len()];
    for i in 0..limbs.len() {
        half[i] = (limbs[i] >> 1) | limbs.get(i + 1).map_or(0, |w| w << 63);
    }
    pow_limbs(x, &half) == -F::ONE
}

/// The candidate of counter value `counter`.
fn candidate<F: PrimeField>(counter: u32) -> F {
    let digest = Blake2b512::digest(counter.to_be_bytes());
    let mut limbs = [0u64; 4];
    for (i, byte) in digest[..31].iter().enumerate() {
        limbs[i / 8] |= u64::from(*byte) << (8 * (i % 8));
    }
    match F::from_limbs(&limbs[..F::LIMBS.min(4)]) {
        Some(x) => x,
        None => unreachable!("248 bits fit in a field of more than 248 bits"),
    }
}

/// Kimchi's shifts for a domain of `2^log_size` rows.
///
/// # Panics
///
/// If the field has fewer than 249 bits (Kimchi's sampling reads 31 bytes).
#[must_use]
pub fn shifts<F: PrimeField>(log_size: u32) -> [F; PERMUTS] {
    assert!(F::MODULUS_BITS > 248, "Kimchi's shifts read 31 bytes");
    let n = 1u64 << log_size;
    let mut shifts = [F::ZERO; PERMUTS];
    shifts[0] = F::ONE;
    let mut counter: u32 = 7;
    let sample = |counter: &mut u32| loop {
        *counter += 1;
        let k = candidate::<F>(*counter);
        if is_non_residue(k) && k.pow(n) != F::ONE {
            return k;
        }
    };
    for i in 1..PERMUTS {
        let mut k = sample(&mut counter);
        while shifts[..i].contains(&k) {
            k = sample(&mut counter);
        }
        shifts[i] = k;
    }
    shifts
}
