//! The behaviour of each public item, one test per item.

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::oracle::{CHALLENGE_LIMBS, FqSponge, FrSponge, ScalarChallenge};
use arrabbiata_pasta::{Fp, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

#[test]
fn challenge_limbs() {
    assert_eq!(CHALLENGE_LIMBS, 2);
}

#[test]
fn scalar_challenge() {
    let lambda = Vesta::lambda();
    let zero = ScalarChallenge::<Fp>::from_limbs([0, 0]);
    assert_eq!(
        zero.to_field(lambda),
        zero.to_field_with_length(128, lambda)
    );
    // Two zero bits: from a = b = 2, double both, then -1 on b.
    let two = zero.to_field_with_length(2, lambda);
    assert_eq!(two, Fp::from_u64(4) * lambda + Fp::from_u64(3));
}

#[test]
fn fq_sponge() {
    // A challenge is 128 bits: its limbs past the second are zero.
    let mut sponge = FqSponge::<Vesta, _>::new(KimchiSponge::new(vesta_base()));
    sponge.absorb_g(&[Projective::<Vesta>::generator()]);
    let c = sponge.challenge();
    assert_eq!((c.limb(2), c.limb(3)), (0, 0));
    assert_ne!(sponge.digest(), Fp::ZERO);
}

#[test]
fn fr_sponge() {
    let mut sponge = FrSponge::<Fp, _>::new(KimchiSponge::new(pallas_base()));
    sponge.absorb(Fp::ONE);
    let c = sponge.challenge();
    assert_eq!((c.0.limb(2), c.0.limb(3)), (0, 0));
    assert_ne!(sponge.digest(), Fp::ZERO);
}
