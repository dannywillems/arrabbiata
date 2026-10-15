//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn grain_overview() {
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::grain::Grain;
    let mut a = Grain::<Fp>::new(3, 8, 56);
    let mut b = Grain::<Fp>::new(3, 8, 56);
    assert_eq!(a.next_field_element(), b.next_field_element());
}

#[test]
fn grain_grain() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_poseidon::grain::Grain;
    let mut g = Grain::<Fq>::new(3, 8, 56);
    assert_ne!(g.next_field_element(), g.next_field_element());
}

#[test]
fn grain_grain_new() {
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::grain::Grain;
    let _ = Grain::<Fp>::new(3, 8, 56);
}

#[test]
fn grain_grain_next_field_element() {
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::grain::Grain;
    let _ = Grain::<Fp>::new(3, 8, 56).next_field_element();
}

#[test]
fn grain_grain_next_field_element_without_rejection() {
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::grain::Grain;
    let _ = Grain::<Fp>::new(3, 8, 56).next_field_element_without_rejection();
}

#[test]
fn kimchi_kimchi_poseidon() {
    use arrabbiata_poseidon::kimchi::{ROUNDS, pallas_base};
    assert_eq!(pallas_base().round_constants().len(), ROUNDS);
}

#[test]
fn kimchi_kimchi_poseidon_new() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::kimchi::{KimchiPoseidon, ROUNDS};
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
    use arrabbiata_poseidon::kimchi::vesta_base;
    assert_eq!(vesta_base().mds().len(), 3);
}

#[test]
fn kimchi_kimchi_poseidon_round_constants() {
    use arrabbiata_poseidon::kimchi::{ROUNDS, vesta_base};
    assert_eq!(vesta_base().round_constants().len(), ROUNDS);
}

#[test]
fn kimchi_kimchi_poseidon_round() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::kimchi::pallas_base;
    let poseidon = pallas_base();
    // The zero state goes to the first round's constants.
    let mut state = [Fp::ZERO; 3];
    poseidon.round(0, &mut state);
    assert_eq!(&state, &poseidon.round_constants()[0]);
}

#[test]
fn kimchi_kimchi_poseidon_permute_state() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::kimchi::pallas_base;
    let mut state = [Fp::ZERO; 3];
    pallas_base().permute_state(&mut state);
    assert_ne!(state, [Fp::ZERO; 3]);
}

#[test]
fn kimchi_pallas_base() {
    use arrabbiata_poseidon::kimchi::pallas_base;
    let _ = pallas_base();
}

#[test]
fn kimchi_vesta_base() {
    use arrabbiata_poseidon::kimchi::vesta_base;
    let _ = vesta_base();
}

#[test]
fn kimchi_kimchi_sponge() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base};
    use arrabbiata_transcript::Transcript;
    let mut a = KimchiSponge::new(pallas_base());
    let mut b = KimchiSponge::new(pallas_base());
    a.absorb(Fp::ONE);
    b.absorb(Fp::ONE);
    b.absorb(Fp::ZERO);
    assert_eq!(a.squeeze(), b.squeeze());
}

#[test]
fn kimchi_kimchi_sponge_new() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq;
    use arrabbiata_poseidon::kimchi::{KimchiSponge, vesta_base};
    assert_eq!(KimchiSponge::new(vesta_base()).state(), &[Fq::ZERO; 3]);
}

#[test]
fn kimchi_kimchi_sponge_state() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base};
    use arrabbiata_transcript::Transcript;
    let mut sponge = KimchiSponge::new(pallas_base());
    sponge.absorb(Fp::ONE);
    assert_eq!(sponge.state(), &[Fp::ONE, Fp::ZERO, Fp::ZERO]);
}

#[test]
fn kimchi_hash() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::kimchi::{hash, pallas_base};
    assert_ne!(
        hash(pallas_base(), &[Fp::ONE]),
        hash(pallas_base(), &[Fp::ZERO])
    );
}

#[test]
fn crate_overview() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::{PoseidonSponge, pallas_base};
    use arrabbiata_transcript::{Duplex, Transcript};

    let mut sponge: PoseidonSponge<Fp> = Duplex::new(pallas_base(), 0);
    sponge.absorb(Fp::from_u64(42));
    let challenge = sponge.squeeze();
    let _ = challenge;
}

#[test]
fn poseidon() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::Poseidon;
    let poseidon = Poseidon::<Fp, 3>::new(8, 56, 5);
    let mut state = [Fp::ZERO, Fp::ONE, Fp::from_u64(2)];
    poseidon.permute_state(&mut state);
    assert_ne!(state, [Fp::ZERO, Fp::ONE, Fp::from_u64(2)]);
}

#[test]
fn poseidon_new() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_poseidon::Poseidon;
    let p = Poseidon::<Fq, 3>::new(8, 56, 5);
    assert_eq!(p.round_constants().len(), 64);
}

#[test]
fn poseidon_round_constants() {
    use arrabbiata_poseidon::pallas_base;
    assert_eq!(pallas_base().round_constants()[0].len(), 3);
}

#[test]
fn poseidon_mds() {
    use arrabbiata_field::Field;
    use arrabbiata_poseidon::pallas_base;
    assert!(!pallas_base().mds()[0][0].is_zero());
}

#[test]
fn poseidon_full_rounds() {
    use arrabbiata_poseidon::pallas_base;
    assert_eq!(pallas_base().full_rounds(), 8);
}

#[test]
fn poseidon_partial_rounds() {
    use arrabbiata_poseidon::{bn254_scalar, pallas_base};
    assert_eq!(pallas_base().partial_rounds(), 56);
    assert_eq!(bn254_scalar().partial_rounds(), 57);
}

#[test]
fn poseidon_alpha() {
    use arrabbiata_poseidon::pallas_base;
    assert_eq!(pallas_base().alpha(), 5);
}

#[test]
fn poseidon_permute_state() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fp;
    use arrabbiata_poseidon::pallas_base;
    let mut a = [Fp::ZERO; 3];
    let mut b = [Fp::ZERO; 3];
    pallas_base().permute_state(&mut a);
    pallas_base().permute_state(&mut b);
    assert_eq!(a, b);
}

#[test]
fn pallas_base() {
    use arrabbiata_poseidon::pallas_base;
    assert_eq!(pallas_base().round_constants().len(), 64);
}

#[test]
fn vesta_base() {
    use arrabbiata_poseidon::vesta_base;
    assert_eq!(vesta_base().round_constants().len(), 64);
}

#[test]
fn pallas_sponge() {
    use arrabbiata_poseidon::pallas_sponge;
    use arrabbiata_transcript::Transcript;
    let mut t = pallas_sponge(1);
    let _ = t.squeeze();
}

#[test]
fn vesta_sponge() {
    use arrabbiata_poseidon::vesta_sponge;
    use arrabbiata_transcript::Transcript;
    let mut t = vesta_sponge(1);
    let _ = t.squeeze();
}

#[test]
fn bn254_scalar() {
    use arrabbiata_poseidon::bn254_scalar;
    assert_eq!(bn254_scalar().round_constants().len(), 65);
}

#[test]
fn bn254_base() {
    use arrabbiata_poseidon::bn254_base;
    assert_eq!(bn254_base().round_constants().len(), 65);
}

#[test]
fn bn254_scalar_sponge() {
    use arrabbiata_poseidon::bn254_scalar_sponge;
    use arrabbiata_transcript::Transcript;
    let _ = bn254_scalar_sponge(0).squeeze();
}

#[test]
fn bn254_base_sponge() {
    use arrabbiata_poseidon::bn254_base_sponge;
    use arrabbiata_transcript::Transcript;
    let _ = bn254_base_sponge(0).squeeze();
}

#[test]
fn goldilocks() {
    use arrabbiata_poseidon::goldilocks;
    assert_eq!(goldilocks().round_constants().len(), 30);
}

#[test]
fn goldilocks_sponge() {
    use arrabbiata_poseidon::goldilocks_sponge;
    use arrabbiata_transcript::Transcript;
    assert_eq!(
        goldilocks_sponge(1).squeeze(),
        goldilocks_sponge(1).squeeze()
    );
    assert_ne!(
        goldilocks_sponge(1).squeeze(),
        goldilocks_sponge(2).squeeze()
    );
}
