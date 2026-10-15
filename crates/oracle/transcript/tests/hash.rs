//! The hash transcript, on SHA-256 (any `ByteHash` would do):
//!
//! - determinism: two transcripts with the same history agree;
//! - the challenge depends on the domain, on every absorbed value, and on
//!   their order; consecutive challenges differ;
//! - it is a `Transcript`, so the typed tape runs on it.

use arrabbiata_field::{Field, Goldilocks};
use arrabbiata_transcript::{
    Transcript,
    hash::{ByteHash, HashTranscript},
};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

/// SHA-256 as a `ByteHash`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Sha;

impl ByteHash for Sha {
    fn hash(&self, parts: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part);
        }
        hasher.finalize().into()
    }
}

type F = arrabbiata_pasta::Fq;

fn transcript(domain: u64) -> HashTranscript<F, Sha> {
    HashTranscript::new(Sha, domain)
}

fn arb_f() -> impl Strategy<Value = F> {
    any::<[u64; 4]>().prop_map(|l| {
        l.iter().fold(F::ZERO, |acc, x| {
            acc * F::from_u64(u64::MAX) + F::from_u64(*x)
        })
    })
}

fn challenge_after(domain: u64, values: &[F]) -> F {
    let mut t = transcript(domain);
    for v in values {
        t.absorb(*v);
    }
    t.squeeze()
}

proptest! {
    #[test]
    fn determinism(values in prop::collection::vec(arb_f(), 0..6), domain in any::<u64>()) {
        prop_assert_eq!(challenge_after(domain, &values), challenge_after(domain, &values));
    }

    #[test]
    fn every_value_matters(values in prop::collection::vec(arb_f(), 1..6), which in any::<usize>(), domain in any::<u64>()) {
        let mut changed = values.clone();
        let which = which % changed.len();
        changed[which] = changed[which] + F::ONE;
        prop_assert_ne!(challenge_after(domain, &values), challenge_after(domain, &changed));
    }

    #[test]
    fn order_and_length_matter(a in arb_f(), b in arb_f(), domain in any::<u64>()) {
        prop_assume!(a != b);
        prop_assert_ne!(challenge_after(domain, &[a, b]), challenge_after(domain, &[b, a]));
        prop_assert_ne!(challenge_after(domain, &[a]), challenge_after(domain, &[a, F::ZERO]));
    }

    #[test]
    fn the_domain_separates(values in prop::collection::vec(arb_f(), 0..4), domain in any::<u64>()) {
        prop_assert_ne!(challenge_after(domain, &values), challenge_after(domain.wrapping_add(1), &values));
    }

    #[test]
    fn consecutive_challenges_differ(domain in any::<u64>()) {
        let mut t = transcript(domain);
        let first = t.squeeze();
        let second = t.squeeze();
        prop_assert_ne!(first, second);
    }
}

/// A 64-bit field (Goldilocks) and a 255-bit one (Pasta) both work: the
/// number of squeezed blocks follows the field's size.
#[test]
fn small_and_large_fields() {
    let mut small: HashTranscript<Goldilocks, Sha> = HashTranscript::new(Sha, 0);
    small.absorb(Goldilocks::ONE);
    let _ = small.squeeze();
    let mut large = transcript(0);
    large.absorb(F::ONE);
    assert_ne!(large.squeeze(), F::ZERO);
}
