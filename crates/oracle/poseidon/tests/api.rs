//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::Field;
use arrabbiata_pasta::{Fp, Fq};
use arrabbiata_poseidon::{
    Poseidon, PoseidonSponge, bn254_base, bn254_base_sponge, bn254_scalar, bn254_scalar_sponge,
    goldilocks, goldilocks_sponge,
    grain::Grain,
    kimchi::{
        KimchiPoseidon, KimchiSponge, ROUNDS, hash, pallas_base as kimchi_pallas_base,
        vesta_base as kimchi_vesta_base,
    },
    pallas_base, pallas_sponge, vesta_base, vesta_sponge,
};
use arrabbiata_transcript::{Duplex, Transcript};

#[test]
fn grain_overview() {
    let mut a = Grain::<Fp>::new(3, 8, 56);
    let mut b = Grain::<Fp>::new(3, 8, 56);
    assert_eq!(a.next_field_element(), b.next_field_element());
}

#[test]
fn grain_grain() {
    let mut g = Grain::<Fq>::new(3, 8, 56);
    assert_ne!(g.next_field_element(), g.next_field_element());
}

#[test]
fn grain_grain_new() {
    let _ = Grain::<Fp>::new(3, 8, 56);
}

#[test]
fn grain_grain_next_field_element() {
    let _ = Grain::<Fp>::new(3, 8, 56).next_field_element();
}

#[test]
fn grain_grain_next_field_element_without_rejection() {
    let _ = Grain::<Fp>::new(3, 8, 56).next_field_element_without_rejection();
}

#[test]
fn kimchi_kimchi_poseidon() {
    assert_eq!(kimchi_pallas_base().round_constants().len(), ROUNDS);
}

#[test]
fn kimchi_kimchi_poseidon_new() {
    let identity = [
        [Fp::ONE, Fp::ZERO, Fp::ZERO],
        [Fp::ZERO, Fp::ONE, Fp::ZERO],
        [Fp::ZERO, Fp::ZERO, Fp::ONE],
    ];
    let no_constants = [[Fp::ZERO; 3]; ROUNDS];
    let power = KimchiPoseidon::new(identity, no_constants);
    let mut state = [Fp::ONE; 3];
    power.permute_state(&mut state);
    assert_eq!(state, [Fp::ONE; 3]);
}

#[test]
fn kimchi_kimchi_poseidon_mds() {
    assert_eq!(kimchi_vesta_base().mds().len(), 3);
}

#[test]
fn kimchi_kimchi_poseidon_round_constants() {
    assert_eq!(kimchi_vesta_base().round_constants().len(), ROUNDS);
}

#[test]
fn kimchi_kimchi_poseidon_round() {
    let poseidon = kimchi_pallas_base();
    // The zero state goes to the first round's constants.
    let mut state = [Fp::ZERO; 3];
    poseidon.round(0, &mut state);
    assert_eq!(&state, &poseidon.round_constants()[0]);
}

#[test]
fn kimchi_kimchi_poseidon_permute_state() {
    let mut state = [Fp::ZERO; 3];
    kimchi_pallas_base().permute_state(&mut state);
    assert_ne!(state, [Fp::ZERO; 3]);
}

#[test]
fn kimchi_pallas_base_works() {
    let _ = kimchi_pallas_base();
}

#[test]
fn kimchi_vesta_base_works() {
    let _ = kimchi_vesta_base();
}

#[test]
fn kimchi_kimchi_sponge() {
    let mut a = KimchiSponge::new(kimchi_pallas_base());
    let mut b = KimchiSponge::new(kimchi_pallas_base());
    a.absorb(Fp::ONE);
    b.absorb(Fp::ONE);
    b.absorb(Fp::ZERO);
    assert_eq!(a.squeeze(), b.squeeze());
}

#[test]
fn kimchi_kimchi_sponge_new() {
    assert_eq!(
        KimchiSponge::new(kimchi_vesta_base()).state(),
        &[Fq::ZERO; 3]
    );
}

#[test]
fn kimchi_kimchi_sponge_state() {
    let mut sponge = KimchiSponge::new(kimchi_pallas_base());
    sponge.absorb(Fp::ONE);
    assert_eq!(sponge.state(), &[Fp::ONE, Fp::ZERO, Fp::ZERO]);
}

#[test]
fn kimchi_hash() {
    assert_ne!(
        hash(kimchi_pallas_base(), &[Fp::ONE]),
        hash(kimchi_pallas_base(), &[Fp::ZERO])
    );
}

#[test]
fn crate_overview() {
    let mut sponge: PoseidonSponge<Fp> = Duplex::new(pallas_base(), 0);
    sponge.absorb(Fp::from_u64(42));
    let challenge = sponge.squeeze();
    let _ = challenge;
}

#[test]
fn poseidon() {
    let poseidon = Poseidon::<Fp, 3>::new(8, 56, 5);
    let mut state = [Fp::ZERO, Fp::ONE, Fp::from_u64(2)];
    poseidon.permute_state(&mut state);
    assert_ne!(state, [Fp::ZERO, Fp::ONE, Fp::from_u64(2)]);
}

#[test]
fn poseidon_new() {
    let p = Poseidon::<Fq, 3>::new(8, 56, 5);
    assert_eq!(p.round_constants().len(), 64);
}

#[test]
fn poseidon_round_constants() {
    assert_eq!(pallas_base().round_constants()[0].len(), 3);
}

#[test]
fn poseidon_mds() {
    assert!(!pallas_base().mds()[0][0].is_zero());
}

#[test]
fn poseidon_full_rounds() {
    assert_eq!(pallas_base().full_rounds(), 8);
}

#[test]
fn poseidon_partial_rounds() {
    assert_eq!(pallas_base().partial_rounds(), 56);
    assert_eq!(bn254_scalar().partial_rounds(), 57);
}

#[test]
fn poseidon_alpha() {
    assert_eq!(pallas_base().alpha(), 5);
}

#[test]
fn poseidon_permute_state() {
    let mut a = [Fp::ZERO; 3];
    let mut b = [Fp::ZERO; 3];
    pallas_base().permute_state(&mut a);
    pallas_base().permute_state(&mut b);
    assert_eq!(a, b);
}

#[test]
fn pallas_base_works() {
    assert_eq!(pallas_base().round_constants().len(), 64);
}

#[test]
fn vesta_base_works() {
    assert_eq!(vesta_base().round_constants().len(), 64);
}

#[test]
fn pallas_sponge_works() {
    let mut t = pallas_sponge(1);
    let _ = t.squeeze();
}

#[test]
fn vesta_sponge_works() {
    let mut t = vesta_sponge(1);
    let _ = t.squeeze();
}

#[test]
fn bn254_scalar_works() {
    assert_eq!(bn254_scalar().round_constants().len(), 65);
}

#[test]
fn bn254_base_works() {
    assert_eq!(bn254_base().round_constants().len(), 65);
}

#[test]
fn bn254_scalar_sponge_works() {
    let _ = bn254_scalar_sponge(0).squeeze();
}

#[test]
fn bn254_base_sponge_works() {
    let _ = bn254_base_sponge(0).squeeze();
}

#[test]
fn goldilocks_works() {
    assert_eq!(goldilocks().round_constants().len(), 30);
}

#[test]
fn goldilocks_sponge_works() {
    assert_eq!(
        goldilocks_sponge(1).squeeze(),
        goldilocks_sponge(1).squeeze()
    );
    assert_ne!(
        goldilocks_sponge(1).squeeze(),
        goldilocks_sponge(2).squeeze()
    );
}
