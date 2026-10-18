//! What a statement is made of when it crosses the cycle: words, and the
//! sorts that are written in them.
//!
//! A proof over one field has its public input in that field; its verifier
//! runs as a circuit over the other. No ring homomorphism joins `Z/p` and
//! `Z/q`, so no arithmetic crosses, only values: each crossing lifts a field
//! element to an integer (a [`Word`]), which the other field reduces. A word
//! below `2^254` is the same element of both Pasta fields (both moduli exceed
//! `2^254`); a longer word is not, and its lift is where soundness is decided.
//!
//! A [`Sort`] is a kind of value and its words: `encode` a section, `decode`
//! a retraction, `decode . encode = id`. Each word has a bound of at most
//! 254 bits, which both sides check, so its lift is canonical: one cell, one
//! integer.
//!
//! No sort can do without it. A cell of a Pasta field is the residue of up
//! to two integers below `2^255`, `x` and `x + q`, and a word's meaning is
//! read modulo the other prime `p`, by a map injective modulo `p` (the
//! action `L -> [L] P` on a group of order `p`, or the decoding
//! `L -> 2 L + 2^255 + 1`). Injective modulo `p` and `q` not a multiple of
//! `p`, it tells `x` from `x + q`: the two integers mean different things,
//! and a prover choosing the lift chooses the meaning. Below `2^254`, under
//! both moduli, there is one integer. The laws are in `tests/statement.rs`; the
//! derivation, and Mina's sorts checked against it, in the design note on
//! cross-field statements.
//!
//! Curve points are not a sort: they cross only as digests of the messages
//! that carry them, hashed in their home field.

use alloc::{vec, vec::Vec};
use core::{fmt::Debug, marker::PhantomData};

use arrabbiata_curve::Curve;
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_ipa::generic::shift_scalar;

/// The bits of the window every word of a statement is below, unless its
/// sort says otherwise: both Pasta moduli exceed `2^254`, so below it a word
/// is the same element of both fields.
pub const WINDOW_BITS: u32 = 254;

/// The bits of a shift: `2^255`, the shifts of Pickles' shifted values.
pub const SHIFT_BITS: u32 = 255;

/// An integer below `2^256`, little-endian limbs: the unit a statement
/// crosses the cycle in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Word(pub [u64; 4]);

/// The integers' order: from the most significant limb (a derived order on
/// the little-endian limbs would compare the least significant first).
impl Ord for Word {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for Word {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Word {
    /// The canonical integer of `x`, below the modulus of its field.
    #[must_use]
    pub fn of<F: PrimeField>(x: F) -> Self {
        Self(core::array::from_fn(|i| {
            if i < F::LIMBS { x.limb(i) } else { 0 }
        }))
    }

    /// The word modulo the modulus of `F`: the element of `F` it is read as.
    #[must_use]
    pub fn reduce<F: PrimeField>(self) -> F {
        let base = F::from_u64(1 << 32).square();
        self.0
            .iter()
            .rev()
            .fold(F::ZERO, |acc, limb| acc * base + F::from_u64(*limb))
    }

    /// The word of `x`, an integer below `2^128`.
    #[must_use]
    pub const fn of_u128(x: u128) -> Self {
        Self([x as u64, (x >> 64) as u64, 0, 0])
    }

    /// The number of bits of the word: the least `n` with the word below
    /// `2^n`.
    #[must_use]
    pub fn bits(self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return 64 * i as u32 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    /// Whether the word is below `2^bits`.
    #[must_use]
    pub fn below(self, bits: u32) -> bool {
        self.bits() <= bits
    }

    /// `self + other`, or `None` past `2^256`.
    #[must_use]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for (i, slot) in out.iter_mut().enumerate() {
            let (s, c1) = self.0[i].overflowing_add(other.0[i]);
            let (s, c2) = s.overflowing_add(carry);
            *slot = s;
            carry = u64::from(c1) + u64::from(c2);
        }
        (carry == 0).then_some(Self(out))
    }

    /// The modulus of `F` as a word.
    #[must_use]
    pub fn modulus<F: PrimeField>() -> Self {
        let minus_one = Self::of(-F::ONE);
        match minus_one.checked_add(Self([1, 0, 0, 0])) {
            Some(m) => m,
            None => unreachable!("the moduli are below 2^256"),
        }
    }

    /// The high bits and the low bit: `(w >> 1, w & 1)`.
    #[must_use]
    pub const fn halve(self) -> (Self, bool) {
        let w = self.0;
        let high = [
            (w[0] >> 1) | (w[1] << 63),
            (w[1] >> 1) | (w[2] << 63),
            (w[2] >> 1) | (w[3] << 63),
            w[3] >> 1,
        ];
        (Self(high), w[0] & 1 == 1)
    }
}

/// A kind of value a statement holds, and the words it is written in.
///
/// Laws: `decode(encode(v)) = v`; every word of `encode(v)` is within its
/// bound; `decode` refuses words beyond the bounds, so each accepted word is
/// the canonical integer of the cell holding it, in either field.
pub trait Sort {
    /// The value, in its home field.
    type Value: Clone + Debug + PartialEq;

    /// The bound of each word, in order, in bits: the word is below `2^n`,
    /// `1 <= n <= 254`, and both sides check it.
    const BOUNDS: &'static [u32];

    /// The words of `v`.
    fn encode(v: &Self::Value) -> Vec<Word>;

    /// The value of `words`, or `None` for words the sort does not accept.
    fn decode(words: &[Word]) -> Option<Self::Value>;

    /// The number of words of a value.
    #[must_use]
    fn words() -> usize {
        Self::BOUNDS.len()
    }
}

/// Whether `words` are as many as the sort's and each within its bound.
fn bounded<S: Sort + ?Sized>(words: &[Word]) -> bool {
    words.len() == S::BOUNDS.len() && words.iter().zip(S::BOUNDS).all(|(w, n)| w.below(*n))
}

/// `2^255` in `F`.
fn shift<F: PrimeField>() -> F {
    (F::ONE + F::ONE).pow(u64::from(SHIFT_BITS))
}

/// A scalar of `C` whose field is the smaller of the cycle, held as its
/// Type1 shifted value: one word `t` with `s = 2 t + 2^255 + 1` (Pickles'
/// `Shifted_value.Type1`), below `2^254`. A `t` at or past `2^254` (about
/// `2^-128` of the scalars) has no encoding: the circuit's VarBaseMul
/// reads `t`'s canonical integer only below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Type1<C>(PhantomData<C>);

impl<C> Sort for Type1<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    type Value = C::Scalar;
    const BOUNDS: &'static [u32] = &[WINDOW_BITS];

    fn encode(s: &C::Scalar) -> Vec<Word> {
        vec![Word::of(shift_scalar::<C>(*s))]
    }

    fn decode(words: &[Word]) -> Option<C::Scalar> {
        if !bounded::<Self>(words) {
            return None;
        }
        let t: C::Scalar = words[0].reduce();
        Some(t + t + shift::<C::Scalar>() + C::Scalar::ONE)
    }
}

/// A scalar of `C` whose field is the larger of the cycle, held as its
/// Type2 shifted value `t = s - 2^255`, in two words: its high bits and its
/// low bit (Pickles' `Shifted_value.Type2`). Both words are bounded, so the
/// lift is canonical: `2 high + low` is `t` itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Type2<C>(PhantomData<C>);

impl<C> Sort for Type2<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    type Value = C::Scalar;
    const BOUNDS: &'static [u32] = &[WINDOW_BITS, 1];

    fn encode(s: &C::Scalar) -> Vec<Word> {
        let (high, low) = Word::of(shift_scalar::<C>(*s)).halve();
        vec![high, Word([u64::from(low), 0, 0, 0])]
    }

    fn decode(words: &[Word]) -> Option<C::Scalar> {
        if !bounded::<Self>(words) {
            return None;
        }
        let high: C::Scalar = words[0].reduce();
        let low: C::Scalar = words[1].reduce();
        Some(high + high + low + shift::<C::Scalar>())
    }
}

/// A 128-bit challenge, a scalar challenge or a bulletproof prechallenge:
/// one word below `2^128`, the same in both fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Challenge;

impl Sort for Challenge {
    type Value = u128;
    const BOUNDS: &'static [u32] = &[128];

    fn encode(c: &u128) -> Vec<Word> {
        vec![Word::of_u128(*c)]
    }

    fn decode(words: &[Word]) -> Option<u128> {
        if !bounded::<Self>(words) {
            return None;
        }
        let w = words[0].0;
        Some(u128::from(w[0]) | (u128::from(w[1]) << 64))
    }
}

/// A digest: a sponge's output in its home field, crossing as its integer,
/// bounded below `2^254` so that it is one element of both fields. Its
/// producer checks the bound, and a digest past it (probability about
/// `2^-128` for a Pasta sponge) has no proof: a completeness loss, where an
/// unbounded 255-bit lift would give the prover a choice among two readings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Digest;

impl Sort for Digest {
    type Value = Word;
    const BOUNDS: &'static [u32] = &[WINDOW_BITS];

    fn encode(d: &Word) -> Vec<Word> {
        vec![*d]
    }

    fn decode(words: &[Word]) -> Option<Word> {
        bounded::<Self>(words).then(|| words[0])
    }
}

/// A boolean: one word, `0` or `1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bool;

impl Sort for Bool {
    type Value = bool;
    const BOUNDS: &'static [u32] = &[1];

    fn encode(b: &bool) -> Vec<Word> {
        vec![Word([u64::from(*b), 0, 0, 0])]
    }

    fn decode(words: &[Word]) -> Option<bool> {
        bounded::<Self>(words).then(|| words[0].0[0] == 1)
    }
}

/// How many proofs a step rule verifies (Pickles' `Proofs_verified`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProofsVerified {
    /// None: a base case.
    N0,
    /// One.
    N1,
    /// Two.
    N2,
}

impl ProofsVerified {
    /// Its two bits, least significant first (`Proofs_verified.to_bool_vec`:
    /// `N1` is `[false; true]`).
    #[must_use]
    pub const fn mask(self) -> [bool; 2] {
        match self {
            Self::N0 => [false, false],
            Self::N1 => [false, true],
            Self::N2 => [true, true],
        }
    }
}

/// Which branch a proof is of: the proofs its rule verifies and the log2 of
/// its domain, one word `4 domain_log2 + b0 + 2 b1` of 10 bits (Pickles'
/// `Branch_data.pack`). The mask `[true; false]` is no branch's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BranchData;

impl Sort for BranchData {
    type Value = (ProofsVerified, u8);
    const BOUNDS: &'static [u32] = &[10];

    fn encode((proofs, log2): &(ProofsVerified, u8)) -> Vec<Word> {
        let [b0, b1] = proofs.mask();
        let w = 4 * u64::from(*log2) + u64::from(b0) + 2 * u64::from(b1);
        vec![Word([w, 0, 0, 0])]
    }

    fn decode(words: &[Word]) -> Option<(ProofsVerified, u8)> {
        if !bounded::<Self>(words) {
            return None;
        }
        let w = words[0].0[0];
        let proofs = match w & 3 {
            0 => ProofsVerified::N0,
            2 => ProofsVerified::N1,
            3 => ProofsVerified::N2,
            _ => return None,
        };
        u8::try_from(w >> 2).ok().map(|log2| (proofs, log2))
    }
}
