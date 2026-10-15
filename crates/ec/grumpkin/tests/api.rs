//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_bn254::Fq;
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_grumpkin::Grumpkin;

#[test]
fn crate_overview() {
    // Grumpkin scalars are BN254 base field elements.
    let p = Projective::<Grumpkin>::generator() * Fq::from_u64(7);
    assert!(p.is_on_curve());
}

#[test]
fn grumpkin() {
    assert!(Projective::<Grumpkin>::generator().is_on_curve());
}
