//! The sponge and the absorption helpers, on Poseidon over Goldilocks.

use arrabbiata_field::Goldilocks;
use arrabbiata_poseidon::goldilocks_sponge;

use arrabbiata_transcript::{Transcript, absorb_slice};

fn challenge(domain: u64, values: &[u64]) -> Goldilocks {
    let mut t = goldilocks_sponge(domain);
    for v in values {
        t.absorb(Goldilocks::new(*v));
    }
    t.squeeze()
}

#[test]
fn deterministic() {
    assert_eq!(challenge(1, &[5]), challenge(1, &[5]));
}

#[test]
fn sensitive_to_input_domain_and_order() {
    let base = challenge(1, &[1, 2, 3]);
    assert_ne!(base, challenge(2, &[1, 2, 3]));
    assert_ne!(base, challenge(1, &[1, 2, 4]));
    assert_ne!(base, challenge(1, &[2, 1, 3]));
    assert_ne!(base, challenge(1, &[1, 2]));
}

#[test]
fn slices_absorb_with_length_prefix() {
    let mut a = goldilocks_sponge(0);
    absorb_slice(&mut a, &[Goldilocks::new(1)]);
    absorb_slice(&mut a, &[Goldilocks::new(2)]);
    let mut b = goldilocks_sponge(0);
    absorb_slice(&mut b, &[Goldilocks::new(1), Goldilocks::new(2)]);
    assert_ne!(a.squeeze(), b.squeeze());
}
