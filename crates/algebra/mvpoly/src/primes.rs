//! Monomials as integers, through the primes.
//!
//! The first `N` primes `p_0 = 2, p_1 = 3, ...` give the map
//! `x^e |-> p_0^(e_0) ... p_(N-1)^(e_(N-1))`. By the fundamental theorem of
//! arithmetic it is injective, and it turns the product of monomials into
//! the product of integers: an isomorphism of the free commutative monoid
//! on `N` generators onto the submonoid of the positive integers generated
//! by the first `N` primes.

use alloc::vec::Vec;

use crate::monomial::Monomial;

/// The first `n` primes, by trial division.
///
/// ```
/// use arrabbiata_mvpoly::primes::first;
/// assert_eq!(first(6), vec![2, 3, 5, 7, 11, 13]);
/// assert!(first(0).is_empty());
/// ```
#[must_use]
pub fn first(n: usize) -> Vec<u64> {
    let mut primes: Vec<u64> = Vec::with_capacity(n);
    let mut candidate = 2u64;
    while primes.len() < n {
        if primes
            .iter()
            .take_while(|p| **p * **p <= candidate)
            .all(|p| !candidate.is_multiple_of(*p))
        {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

/// The integer of `m`, `prod_i p_i^(e_i)`, or `None` if it overflows a
/// `u128`.
///
/// ```
/// use arrabbiata_mvpoly::{Monomial, primes};
/// let p = primes::first(3);
/// // x_0^2 x_2 = 2^2 * 5.
/// assert_eq!(primes::code(&Monomial::new([2, 0, 1]), &p), Some(20));
/// assert_eq!(primes::code(&Monomial::<3>::one(), &p), Some(1));
/// ```
#[must_use]
pub fn code<const N: usize>(m: &Monomial<N>, primes: &[u64]) -> Option<u128> {
    let mut acc: u128 = 1;
    for (e, p) in m.exponents().iter().zip(primes) {
        for _ in 0..*e {
            acc = acc.checked_mul(*p as u128)?;
        }
    }
    Some(acc)
}

/// The monomial of the integer `n`: its exponents on the first `N`
/// primes, or `None` if `n` is zero or has another prime factor.
///
/// ```
/// use arrabbiata_mvpoly::{Monomial, primes};
/// let p = primes::first(3);
/// assert_eq!(primes::decode::<3>(20, &p), Some(Monomial::new([2, 0, 1])));
/// // 7 is not among the first three primes.
/// assert_eq!(primes::decode::<3>(14, &p), None);
/// ```
#[must_use]
pub fn decode<const N: usize>(mut n: u128, primes: &[u64]) -> Option<Monomial<N>> {
    if n == 0 {
        return None;
    }
    let mut e = [0u32; N];
    for (i, p) in primes.iter().take(N).enumerate() {
        let p = *p as u128;
        while n.is_multiple_of(p) {
            n /= p;
            e[i] += 1;
        }
    }
    if n == 1 { Some(Monomial::new(e)) } else { None }
}
