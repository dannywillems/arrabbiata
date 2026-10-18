//! The laws of the sorts a statement crosses the cycle in: the retraction,
//! the bounds, descent for the unbounded word, canonicity for the bounded
//! ones, and the window: a word below `2^254` is the same element of both
//! Pasta fields.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_pickles::statement::{
    Bool, Challenge, Digest, Sort, Type1, Type2, WINDOW_BITS, Word,
};
use proptest::prelude::*;

/// An element of `F` from four arbitrary limbs, reduced.
fn element<F: PrimeField>(limbs: [u64; 4]) -> F {
    Word(limbs).reduce()
}

/// The retraction law and the bounds, for one value.
fn retract<S: Sort>(v: &S::Value) -> Result<(), TestCaseError> {
    let words = S::encode(v);
    prop_assert_eq!(words.len(), S::words());
    for (w, b) in words.iter().zip(S::BOUNDS) {
        if let Some(n) = b {
            prop_assert!(w.below(*n), "a word past its bound");
        }
    }
    prop_assert_eq!(S::decode(&words), Some(v.clone()));
    Ok(())
}

proptest! {
    #[test]
    fn type1_retracts_and_descends(limbs in any::<[u64; 4]>()) {
        // Vesta's scalars, the smaller field, in Fq: t and t + p decode
        // alike, the consumer's reduction cannot tell them apart and neither
        // does the decoding.
        let s: Fp = element(limbs);
        retract::<Type1<Vesta>>(&s)?;
        let [t] = Type1::<Vesta>::encode(&s)[..] else { unreachable!() };
        if let Some(alias) = t.checked_add(Word::modulus::<Fp>()) {
            prop_assert_eq!(Type1::<Vesta>::decode(&[alias]), Some(s));
        }
    }

    #[test]
    fn type2_retracts_and_is_canonical(limbs in any::<[u64; 4]>()) {
        // Pallas's scalars, the larger field, in Fp: two bounded words, and a
        // high part past 254 bits is refused, so no word has two readings.
        let s: Fq = element(limbs);
        retract::<Type2<Pallas>>(&s)?;
        let words = Type2::<Pallas>::encode(&s);
        let past = Word([0, 0, 0, 1 << 62]);
        prop_assert_eq!(Type2::<Pallas>::decode(&[past, words[1]]), None);
        prop_assert_eq!(Type2::<Pallas>::decode(&[words[0], Word([2, 0, 0, 0])]), None);
    }

    #[test]
    fn challenges_retract_and_are_canonical(c in any::<u128>()) {
        retract::<Challenge>(&c)?;
        let aliased = Word::of_u128(c).checked_add(Word([0, 0, 1, 0])).unwrap();
        prop_assert_eq!(Challenge::decode(&[aliased]), None);
    }

    #[test]
    fn digests_retract_within_the_window(limbs in any::<[u64; 4]>()) {
        let mut d = Word(limbs);
        d.0[3] &= (1 << 62) - 1;
        retract::<Digest>(&d)?;
        prop_assert_eq!(Digest::decode(&[Word([0, 0, 0, 1 << 62])]), None);
    }

    #[test]
    fn the_window_is_common_to_both_fields(limbs in any::<[u64; 4]>()) {
        // Below 2^254 a word reduces to an element of each field whose
        // canonical integer is the word: the same value on both sides.
        let mut w = Word(limbs);
        w.0[3] &= (1 << 62) - 1;
        prop_assert!(w.below(WINDOW_BITS));
        prop_assert_eq!(Word::of(w.reduce::<Fp>()), w);
        prop_assert_eq!(Word::of(w.reduce::<Fq>()), w);
    }
}

#[test]
fn booleans_retract() {
    for b in [false, true] {
        assert_eq!(Bool::decode(&Bool::encode(&b)), Some(b));
    }
    assert_eq!(Bool::decode(&[Word([2, 0, 0, 0])]), None);
}

#[test]
fn a_decoding_of_the_integer_is_not_sound() {
    // The non-standard model the laws must exclude: a decoding that reads
    // the integer rather than its residue modulo the consumer passes the
    // retraction, but tells t and t + p apart.
    let integer_decode = |w: Word| -> Fq { w.reduce() };
    let s = Fp::from_u64(5);
    let [t] = Type1::<Vesta>::encode(&s)[..] else {
        unreachable!()
    };
    let alias = t.checked_add(Word::modulus::<Fp>()).unwrap();
    assert_ne!(integer_decode(t), integer_decode(alias));
    assert_eq!(
        Type1::<Vesta>::decode(&[t]),
        Type1::<Vesta>::decode(&[alias])
    );
}

#[test]
fn words() {
    // The arithmetic of words: bits, halving, the moduli.
    assert_eq!(Word([0, 0, 0, 0]).bits(), 0);
    assert_eq!(Word([1, 0, 0, 0]).bits(), 1);
    assert_eq!(Word([0, 0, 0, 1 << 62]).bits(), 255);
    assert_eq!(Word([3, 0, 0, 0]).halve(), (Word([1, 0, 0, 0]), true));
    assert!(Word::modulus::<Fp>() < Word::modulus::<Fq>());
    assert!(Word([0, 1, 0, 0]) > Word([u64::MAX, 0, 0, 0]));
    assert_eq!(Word::modulus::<Fp>().bits(), 255);
    assert_eq!(Word::modulus::<Fp>().reduce::<Fp>(), Fp::ZERO);
}
