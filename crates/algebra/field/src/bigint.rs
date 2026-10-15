//! Fixed-width unsigned integers as little-endian arrays of 64-bit limbs.
//!
//! These are the `const fn` helpers the Montgomery fields use to derive
//! their constants (`R`, `R^2`, `-p^-1 mod 2^64`, `(p - 1) / 2^s`) from the
//! modulus at compile time, so that the modulus is the only number written
//! by hand.

/// `a + b` and the carry out.
#[must_use]
pub const fn add<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut out = [0u64; N];
    let mut carry = 0u64;
    let mut i = 0;
    while i < N {
        let s = a[i] as u128 + b[i] as u128 + carry as u128;
        out[i] = s as u64;
        carry = (s >> 64) as u64;
        i += 1;
    }
    (out, carry != 0)
}

/// `a - b` and the borrow out.
#[must_use]
pub const fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut out = [0u64; N];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < N {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        out[i] = d2;
        borrow = (b1 | b2) as u64;
        i += 1;
    }
    (out, borrow != 0)
}

/// Whether `a >= b`.
#[must_use]
pub const fn geq<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
    let mut i = N;
    while i > 0 {
        i -= 1;
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true
}

/// Whether every limb is zero.
#[must_use]
pub const fn is_zero<const N: usize>(a: &[u64; N]) -> bool {
    let mut i = 0;
    while i < N {
        if a[i] != 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// `a >> 1`.
#[must_use]
pub const fn shr1<const N: usize>(a: &[u64; N]) -> [u64; N] {
    let mut out = [0u64; N];
    let mut i = 0;
    while i < N {
        out[i] = a[i] >> 1;
        if i + 1 < N {
            out[i] |= a[i + 1] << 63;
        }
        i += 1;
    }
    out
}

/// Bit `i` of `a`.
#[must_use]
pub const fn bit<const N: usize>(a: &[u64; N], i: usize) -> bool {
    if i >= 64 * N {
        return false;
    }
    (a[i / 64] >> (i % 64)) & 1 == 1
}

/// The number of significant bits of `a`.
#[must_use]
pub const fn num_bits<const N: usize>(a: &[u64; N]) -> u32 {
    let mut i = N;
    while i > 0 {
        i -= 1;
        if a[i] != 0 {
            return 64 * i as u32 + (64 - a[i].leading_zeros());
        }
    }
    0
}

/// `(a + b) mod m` for `a, b < m`.
#[must_use]
pub const fn add_mod<const N: usize>(a: &[u64; N], b: &[u64; N], m: &[u64; N]) -> [u64; N] {
    let (s, carry) = add(a, b);
    if carry || geq(&s, m) { sub(&s, m).0 } else { s }
}

/// `2^(64 N) mod m`, by `64 N` modular doublings of 1.
#[must_use]
pub const fn pow2_mod<const N: usize>(m: &[u64; N]) -> [u64; N] {
    let mut r = [0u64; N];
    r[0] = 1;
    let mut i = 0;
    while i < 64 * N {
        r = add_mod(&r, &r, m);
        i += 1;
    }
    r
}

/// `(a * 2^(64 N)) mod m` for `a < m`, by `64 N` modular doublings.
#[must_use]
pub const fn shift_mod<const N: usize>(a: &[u64; N], m: &[u64; N]) -> [u64; N] {
    let mut r = *a;
    let mut i = 0;
    while i < 64 * N {
        r = add_mod(&r, &r, m);
        i += 1;
    }
    r
}

/// `-m^-1 mod 2^64` for odd `m0`, by Newton iteration.
#[must_use]
pub const fn neg_inv_mod_2_64(m0: u64) -> u64 {
    // x <- x (2 - m0 x) doubles the number of correct low bits; m0 is its own
    // inverse mod 8 for odd m0, so five iterations reach 64 bits.
    let mut x = m0;
    let mut i = 0;
    while i < 5 {
        x = x.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(x)));
        i += 1;
    }
    x.wrapping_neg()
}

/// The 2-adic valuation of `a - 1` for odd `a > 1`.
#[must_use]
pub const fn two_adicity<const N: usize>(a: &[u64; N]) -> u32 {
    let mut m = sub(a, &one()).0;
    let mut s = 0;
    while !is_zero(&m) && m[0] & 1 == 0 {
        m = shr1(&m);
        s += 1;
    }
    s
}

/// The integer 1.
#[must_use]
pub const fn one<const N: usize>() -> [u64; N] {
    let mut r = [0u64; N];
    r[0] = 1;
    r
}
