//! The laws of the sorts a statement crosses the cycle in: the retraction,
//! the bounds, canonicity, and the window: a word below `2^254` is the same
//! element of both Pasta fields. And the obstruction that makes canonicity
//! necessary: one cell, two integers, two meanings.

use arrabbiata_curve::Projective;
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_pickles::statement::{
    Bool, BranchData, Challenge, Digest, ProofsVerified, Sort, Type1, Type2, WINDOW_BITS, Word,
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
    for (w, n) in words.iter().zip(S::BOUNDS) {
        prop_assert!(w.below(*n), "a word past its bound");
    }
    prop_assert_eq!(S::decode(&words), Some(v.clone()));
    Ok(())
}

proptest! {
    #[test]
    fn type1_retracts_and_is_canonical(limbs in any::<[u64; 4]>()) {
        // Vesta's scalars, the smaller field, in Fq: one word below 2^254,
        // and t + p, past it, is refused. A scalar whose t is past 2^254 has
        // no encoding (about 2^-128 of them, none drawn here).
        let s: Fp = element(limbs);
        let [t] = Type1::<Vesta>::encode(&s)[..] else { unreachable!() };
        prop_assume!(t.below(WINDOW_BITS));
        retract::<Type1<Vesta>>(&s)?;
        let alias = t.checked_add(Word::modulus::<Fp>()).unwrap();
        prop_assert_eq!(Type1::<Vesta>::decode(&[alias]), None);
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
fn an_unbounded_lift_gives_one_cell_two_meanings() {
    // The non-standard model the bounds exclude: a Type1 sort without its
    // bound passes the retraction, but the cell t of Fq is also the integer
    // t + q, below 2^255, and the two act differently on Vesta (order p) and
    // decode to different scalars of Fp: both readings are injective
    // modulo p, and q is not a multiple of p. Only a canonical lift is sound.
    let unbounded_decode = |w: Word| -> Fp {
        let t: Fp = w.reduce();
        t + t + (Fp::ONE + Fp::ONE).pow(255) + Fp::ONE
    };
    let t = Word::of_u128(5);
    let alias = t.checked_add(Word::modulus::<Fq>()).unwrap();
    assert!(!alias.below(WINDOW_BITS) && alias.below(255));
    assert_eq!(t.reduce::<Fq>(), alias.reduce::<Fq>(), "one cell");
    assert_ne!(unbounded_decode(t), unbounded_decode(alias), "two scalars");
    let g = Projective::<Vesta>::generator();
    assert_ne!(g * t.reduce::<Fp>(), g * alias.reduce::<Fp>(), "two points");
    assert_eq!(Type1::<Vesta>::decode(&[alias]), None);
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

#[test]
fn branch_data_retracts() {
    // Every mask and domain round-trips; Mina's packing of N1 is 2; the mask
    // 1 is no branch's.
    for proofs in [ProofsVerified::N0, ProofsVerified::N1, ProofsVerified::N2] {
        for log2 in [0u8, 1, 16, 255] {
            let words = BranchData::encode(&(proofs, log2));
            assert!(words[0].below(10));
            assert_eq!(BranchData::decode(&words), Some((proofs, log2)));
        }
    }
    assert_eq!(
        BranchData::encode(&(ProofsVerified::N1, 3)),
        vec![Word([14, 0, 0, 0])]
    );
    assert_eq!(BranchData::decode(&[Word([13, 0, 0, 0])]), None);
    assert_eq!(BranchData::decode(&[Word([1 << 10, 0, 0, 0])]), None);
}
