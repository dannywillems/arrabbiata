//! Kimchi's types on ours, for the ports of Kimchi's foreign field tests:
//! num-bigint's `BigUint` over our `Nat`, o1-utils' `ForeignElement` of our
//! field, and the traits Kimchi's test text calls (`Zero`, `One`, `Two`,
//! `modulus_biguint`), so that the text reads as Kimchi writes it.

#![allow(dead_code)]

use std::{
    cmp::Ordering,
    ops::{Add, Deref, Div, Index, Mul, Rem, Sub},
};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi_gadgets::foreign_field::Nat;

/// The bits of the largest foreign modulus Kimchi accepts, `2^259 - 1`
/// (`max_foreign_field_modulus`, the bound of its multiplication RFC, used
/// for addition too).
pub const MAX_FOREIGN_FIELD_MODULUS_BITS: u32 = 259;

/// num-bigint's `BigUint`, on our `Nat`.
#[derive(Clone, Debug)]
pub struct BigUint(pub Nat);

impl PartialEq for BigUint {
    fn eq(&self, other: &Self) -> bool {
        self.0.compare(&other.0) == Ordering::Equal
    }
}

impl Eq for BigUint {}

impl Deref for BigUint {
    type Target = Nat;
    fn deref(&self) -> &Nat {
        &self.0
    }
}

impl BigUint {
    pub fn new(digits: Vec<u32>) -> Self {
        let mut x = Nat::default();
        for d in digits.iter().rev() {
            x = x.shl(32).add(&Nat::from_u128(u128::from(*d)));
        }
        Self(x)
    }

    pub fn from_bytes_be(bytes: &[u8]) -> Self {
        let mut x = Nat::default();
        for b in bytes {
            x = x.shl(8).add(&Nat::from_u128(u128::from(*b)));
        }
        Self(x)
    }

    pub fn from_hex(hex: &str) -> Self {
        let padded = if hex.len() % 2 == 1 {
            format!("0{hex}")
        } else {
            hex.to_string()
        };
        let bytes: Vec<u8> = (0..padded.len() / 2)
            .map(|i| u8::from_str_radix(&padded[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        Self::from_bytes_be(&bytes)
    }

    /// The bytes, big-endian, without leading zeros (`[0]` for zero).
    pub fn to_bytes_be(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = self
            .0
            .0
            .iter()
            .rev()
            .flat_map(|w| w.to_be_bytes())
            .collect();
        while bytes.len() > 1 && bytes[0] == 0 {
            bytes.remove(0);
        }
        if bytes.is_empty() {
            bytes.push(0);
        }
        bytes
    }

    /// As num-bigint's `{:x}`.
    pub fn hex(&self) -> String {
        let s: String = self
            .to_bytes_be()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let t = s.trim_start_matches('0');
        if t.is_empty() {
            "0".to_string()
        } else {
            t.to_string()
        }
    }

    pub fn pow(&self, k: u32) -> Self {
        (0..k).fold(Self::one(), |acc, _| acc * self.clone())
    }

    // Kimchi's signature: the bound is 2^259 - 1 whatever the native field,
    // and its tests name one (`max_foreign_field_modulus::<PallasField>()`).
    #[allow(clippy::extra_unused_type_parameters)]
    pub fn max_foreign_field_modulus<F>() -> Self {
        let one = Nat::from_u128(1);
        Self(one.shl(MAX_FOREIGN_FIELD_MODULUS_BITS).sub(&one))
    }
}

impl From<u32> for BigUint {
    fn from(x: u32) -> Self {
        Self(Nat::from_u128(u128::from(x)))
    }
}

impl Add for BigUint {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0.add(&other.0))
    }
}

impl Sub for BigUint {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0.sub(&other.0))
    }
}

impl Sub<u32> for BigUint {
    type Output = Self;
    fn sub(self, other: u32) -> Self {
        self - Self::from(other)
    }
}

impl Mul for BigUint {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self(self.0.mul(&other.0))
    }
}

impl Rem for BigUint {
    type Output = Self;
    fn rem(self, m: Self) -> Self {
        Self(self.0.divrem(&m.0).1)
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.compare(&other.0))
    }
}

/// num-traits' `Zero`, `One`, and o1-utils' `Two`.
pub trait Zero {
    fn zero() -> Self;
}
pub trait One {
    fn one() -> Self;
}
pub trait Two {
    fn two() -> Self;
}

impl Zero for BigUint {
    fn zero() -> Self {
        Self(Nat::default())
    }
}
impl One for BigUint {
    fn one() -> Self {
        Self(Nat::from_u128(1))
    }
}
impl<F: Field> Zero for F {
    fn zero() -> Self {
        F::ZERO
    }
}
impl<F: Field> One for F {
    fn one() -> Self {
        F::ONE
    }
}
impl<F: Field> Two for F {
    fn two() -> Self {
        F::ONE + F::ONE
    }
}

/// o1-utils' `FieldHelpers::modulus_biguint`, and `From` as Kimchi's
/// fields have it.
pub trait KimchiField: PrimeField {
    fn modulus_biguint() -> BigUint {
        let minus_one = -Self::ONE;
        let words: Vec<u64> = (0..Self::LIMBS).map(|i| minus_one.limb(i)).collect();
        BigUint(Nat(words)) + BigUint::one()
    }
    fn from_kimchi<T: IntoField>(x: T) -> Self {
        x.into_field()
    }
    /// The canonical integer of the element.
    fn to_biguint(&self) -> BigUint {
        let words: Vec<u64> = (0..Self::LIMBS).map(|i| self.limb(i)).collect();
        BigUint(Nat(words))
    }
    /// `2^k`.
    fn two_pow(k: u64) -> Self {
        Self::from_u64(2).pow(k)
    }
    /// The element of little-endian bytes, `None` from the modulus up.
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let x = BigUint::from_bytes_be(&bytes.iter().rev().copied().collect::<Vec<u8>>());
        (x < Self::modulus_biguint()).then(|| x.0.to_field())
    }
}
impl<F: PrimeField> KimchiField for F {}

pub trait IntoField {
    fn into_field<F: PrimeField>(self) -> F;
}
impl IntoField for u128 {
    fn into_field<F: PrimeField>(self) -> F {
        arrabbiata_kimchi_gadgets::foreign_field::field_of(self)
    }
}
impl IntoField for BigUint {
    fn into_field<F: PrimeField>(self) -> F {
        self.0.to_field()
    }
}

/// o1-utils' `ForeignElement`: a number in `N` limbs of `B` bits, as
/// field elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignElement<F, const B: usize, const N: usize> {
    pub limbs: [F; N],
}

impl<F: PrimeField, const B: usize, const N: usize> ForeignElement<F, B, N> {
    pub fn from_biguint(x: &BigUint) -> Self {
        let base = BigUint::from(2).pow(B as u32);
        let mut rest = x.clone();
        let limbs = core::array::from_fn(|_| {
            let limb = rest.clone() % base.clone();
            rest = BigUint(rest.0.shr(B as u32));
            limb.0.to_field()
        });
        Self { limbs }
    }

    pub fn from_be(bytes: &[u8]) -> Self {
        Self::from_biguint(&BigUint::from_bytes_be(bytes))
    }

    pub fn zero() -> Self {
        Self {
            limbs: [F::ZERO; N],
        }
    }

    pub fn to_biguint(&self) -> BigUint {
        let limbs: Vec<u128> = self
            .limbs
            .iter()
            .map(|l| u128::from(l.limb(0)) | (u128::from(l.limb(1)) << 64))
            .collect();
        BigUint(Nat::from_limbs(&limbs))
    }

    /// `modulus - self`.
    pub fn neg(&self, modulus: &BigUint) -> Self {
        Self::from_biguint(&(modulus.clone() - self.to_biguint()))
    }
}

impl<F, const B: usize, const N: usize> Index<usize> for ForeignElement<F, B, N> {
    type Output = F;
    fn index(&self, i: usize) -> &F {
        &self.limbs[i]
    }
}

// The operators num-bigint has on references, and the helpers of
// o1-utils' `BigUintForeignFieldHelpers` and `*ArrayCompose` that Kimchi's
// foreign field multiplication tests call.

impl Mul<&BigUint> for &BigUint {
    type Output = BigUint;
    fn mul(self, other: &BigUint) -> BigUint {
        BigUint(self.0.mul(&other.0))
    }
}

impl Mul<BigUint> for &BigUint {
    type Output = BigUint;
    fn mul(self, other: BigUint) -> BigUint {
        self * &other
    }
}

impl Rem<&BigUint> for BigUint {
    type Output = BigUint;
    fn rem(self, m: &BigUint) -> BigUint {
        BigUint(self.0.divrem(&m.0).1)
    }
}

impl Sub<BigUint> for &BigUint {
    type Output = BigUint;
    fn sub(self, other: BigUint) -> BigUint {
        BigUint(self.0.sub(&other.0))
    }
}

impl Div for BigUint {
    type Output = BigUint;
    fn div(self, d: BigUint) -> BigUint {
        BigUint(self.0.divrem(&d.0).0)
    }
}

impl Div<&BigUint> for BigUint {
    type Output = BigUint;
    fn div(self, d: &BigUint) -> BigUint {
        BigUint(self.0.divrem(&d.0).0)
    }
}

impl Div<BigUint> for &BigUint {
    type Output = BigUint;
    fn div(self, d: BigUint) -> BigUint {
        BigUint(self.0.divrem(&d.0).0)
    }
}

impl Two for BigUint {
    fn two() -> Self {
        Self::from(2)
    }
}

impl BigUint {
    /// The integer square root, bit by bit from the top.
    pub fn sqrt(&self) -> Self {
        let mut root = Self::zero();
        for bit in (0..160u32).rev() {
            let candidate = root.clone() + Self::one().pow_two(bit);
            if candidate.clone() * candidate.clone() <= *self {
                root = candidate;
            }
        }
        root
    }

    /// `self 2^k`.
    fn pow_two(&self, k: u32) -> Self {
        Self(self.0.shl(k))
    }

    /// `2^88`.
    pub fn two_to_limb() -> Self {
        Self::one().pow_two(88)
    }

    /// The three 88-bit limbs, least significant first.
    pub fn to_limbs(&self) -> [Self; 3] {
        self.0.limbs().map(|l| Self(Nat::from_u128(l)))
    }

    /// The three limbs in `F`.
    pub fn to_field_limbs<F: PrimeField>(&self) -> [F; 3] {
        self.0.limbs().map(|l| Nat::from_u128(l).to_field())
    }

    /// The compact limbs `(lo + 2^88 mi, hi)` in `F`.
    pub fn to_compact_field_limbs<F: PrimeField>(&self) -> [F; 2] {
        let [lo, mi, hi] = self.to_field_limbs::<F>();
        [lo + mi * F::two_pow(88), hi]
    }
}

impl PartialOrd<BigUint> for &BigUint {
    fn partial_cmp(&self, other: &BigUint) -> Option<Ordering> {
        Some(self.0.compare(&other.0))
    }
}

impl PartialEq<BigUint> for &BigUint {
    fn eq(&self, other: &BigUint) -> bool {
        **self == *other
    }
}

/// o1-utils' `compose`, of limbs into the number.
pub trait Compose {
    fn compose(&self) -> BigUint;
}

impl Compose for [BigUint; 3] {
    fn compose(&self) -> BigUint {
        let limbs: Vec<u128> = self.iter().map(|l| l.0.limbs()[0]).collect();
        BigUint(Nat::from_limbs(&limbs))
    }
}

impl<F: PrimeField> Compose for [F; 2] {
    /// Of the compact limbs `(lo + 2^88 mi, hi)`.
    fn compose(&self) -> BigUint {
        self[0].to_biguint() + self[1].to_biguint().pow_two(176)
    }
}

impl<F: PrimeField> Compose for [F; 3] {
    fn compose(&self) -> BigUint {
        let limbs: Vec<u128> = self.iter().map(|l| l.to_biguint().0.limbs()[0]).collect();
        BigUint(Nat::from_limbs(&limbs))
    }
}

impl IntoField for u32 {
    fn into_field<F: PrimeField>(self) -> F {
        F::from_u64(u64::from(self))
    }
}

impl IntoField for u64 {
    fn into_field<F: PrimeField>(self) -> F {
        F::from_u64(self)
    }
}

impl IntoField for i32 {
    fn into_field<F: PrimeField>(self) -> F {
        let magnitude = F::from_u64(u64::from(self.unsigned_abs()));
        if self < 0 { -magnitude } else { magnitude }
    }
}
