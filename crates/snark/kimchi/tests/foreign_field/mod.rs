//! Kimchi's types on ours, for the ports of Kimchi's foreign field tests:
//! num-bigint's `BigUint` over our `Nat`, o1-utils' `ForeignElement` of our
//! field, and the traits Kimchi's test text calls (`Zero`, `One`, `Two`,
//! `modulus_biguint`), so that the text reads as Kimchi writes it.

#![allow(dead_code)]

use std::{
    cmp::Ordering,
    ops::{Add, Deref, Index, Mul, Rem, Sub},
};

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi_gadgets::foreign_field::Nat;

/// The bits of the largest foreign modulus Kimchi accepts, `2^259 - 1`
/// (`max_foreign_field_modulus`, the bound of its multiplication RFC, used
/// for addition too).
pub const MAX_FOREIGN_FIELD_MODULUS_BITS: u32 = 259;

/// num-bigint's `BigUint`, on our `Nat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigUint(pub Nat);

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
