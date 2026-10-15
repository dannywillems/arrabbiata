//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn laws_overview() {
    use arrabbiata_curve::{Projective, laws};
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Pallas};

    let g = Projective::<Pallas>::generator();
    let (a, b) = (g * Fq::from_u64(3), g * Fq::from_u64(5));
    assert!(laws::abelian_group(a, b, g));
    assert!(laws::module_action(a, b, Fq::from_u64(2), Fq::from_u64(7)));
}

#[test]
fn laws_item() {
    use arrabbiata_curve::{Projective, laws};
    use arrabbiata_pasta::Vesta;
    let g = Projective::<Vesta>::generator();
    assert!(laws::abelian_group(g, g + g, Projective::identity()));
}

#[test]
fn laws_affine_add() {
    use arrabbiata_curve::{Projective, laws};
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator().to_affine();
    assert_eq!(
        laws::affine_add(g, g),
        (Projective::<Pallas>::generator().double()).to_affine()
    );
}

#[test]
fn laws_complete_formula_is_affine_addition() {
    use arrabbiata_bn254::G1;
    use arrabbiata_curve::{Projective, laws};
    let g = Projective::<G1>::generator();
    assert!(laws::complete_formula_is_affine_addition(g, g + g));
}

#[test]
fn laws_module_action() {
    use arrabbiata_bn254::{Fr, G2};
    use arrabbiata_curve::{Projective, laws};
    use arrabbiata_field::Field;
    let g = Projective::<G2>::generator();
    assert!(laws::module_action(
        g,
        g + g,
        Fr::from_u64(3),
        Fr::from_u64(4)
    ));
}

#[test]
fn laws_generator_has_order_r() {
    use arrabbiata_curve::laws;
    use arrabbiata_pasta::Pallas;
    assert!(laws::generator_has_order_r::<Pallas>());
}

#[test]
fn laws_affine_round_trip() {
    use arrabbiata_curve::{Projective, laws};
    use arrabbiata_pasta::Vesta;
    assert!(laws::affine_round_trip(
        Projective::<Vesta>::generator().double()
    ));
}

#[test]
fn laws_msm_is_the_sum() {
    use arrabbiata_curve::{Projective, laws};
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Pallas};
    let g = Projective::<Pallas>::generator();
    assert!(laws::msm_is_the_sum(
        &[g, g.double()],
        &[Fq::from_u64(2), Fq::from_u64(3)]
    ));
}

#[test]
fn crate_overview() {
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Pallas};

    let g = Projective::<Pallas>::generator();
    let five_g = g * Fq::from_u64(5);
    assert_eq!(five_g, g + g + g + g + g);
    // The group has order q: (q - 1) G = -G.
    assert_eq!(g * (-Fq::ONE), -g);
}

#[test]
fn curve() {
    use arrabbiata_curve::Curve;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fp, Pallas};
    assert_eq!(Pallas::b(), Fp::from_u64(5));
}

#[test]
fn b() {
    use arrabbiata_curve::Curve;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Vesta};
    assert_eq!(Vesta::b(), Fq::from_u64(5));
}

#[test]
fn generator() {
    use arrabbiata_curve::{Affine, Curve};
    use arrabbiata_pasta::Pallas;
    let (x, y) = Pallas::generator();
    assert!(Affine::<Pallas>::new(x, y).is_some());
}

#[test]
fn affine() {
    use arrabbiata_curve::{Affine, Projective};
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator().to_affine();
    assert_eq!(Affine::new(g.x().unwrap(), g.y().unwrap()), Some(g));
}

#[test]
fn affine_identity() {
    use arrabbiata_curve::Affine;
    use arrabbiata_pasta::Pallas;
    assert!(Affine::<Pallas>::identity().is_identity());
}

#[test]
fn affine_new() {
    use arrabbiata_curve::Affine;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fp, Pallas};
    // (-1, 2): 2^2 = (-1)^3 + 5.
    assert!(Affine::<Pallas>::new(-Fp::ONE, Fp::from_u64(2)).is_some());
    assert!(Affine::<Pallas>::new(Fp::ONE, Fp::ONE).is_none());
}

#[test]
fn affine_is_identity() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    assert!(!Projective::<Pallas>::generator().to_affine().is_identity());
}

#[test]
fn affine_x() {
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fp, Pallas};
    assert_eq!(
        Projective::<Pallas>::generator().to_affine().x(),
        Some(-Fp::ONE)
    );
}

#[test]
fn affine_y() {
    use arrabbiata_curve::Projective;
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fp, Pallas};
    assert_eq!(
        Projective::<Pallas>::generator().to_affine().y(),
        Some(Fp::from_u64(2))
    );
}

#[test]
fn affine_to_projective() {
    use arrabbiata_curve::Affine;
    use arrabbiata_pasta::Pallas;
    assert!(Affine::<Pallas>::identity().to_projective().is_identity());
}

#[test]
fn projective() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator();
    assert_eq!(g + Projective::identity(), g);
    assert!((g - g).is_identity());
}

#[test]
fn projective_identity() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Vesta;
    assert!(Projective::<Vesta>::identity().is_identity());
}

#[test]
fn projective_generator() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Vesta;
    assert!(!Projective::<Vesta>::generator().is_identity());
}

#[test]
fn projective_is_identity() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator();
    assert!((g + (-g)).is_identity());
}

#[test]
fn projective_is_on_curve() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator();
    assert!((g + g + g).is_on_curve());
}

#[test]
fn projective_to_affine() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator();
    assert_eq!((g + g).to_affine().to_projective(), g + g);
}

#[test]
fn projective_double() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator();
    assert_eq!(g.double(), g + g);
}

#[test]
fn projective_mul_limbs() {
    use arrabbiata_curve::Projective;
    use arrabbiata_pasta::Pallas;
    let g = Projective::<Pallas>::generator();
    assert_eq!(g.mul_limbs(&[3]), g + g + g);
}

#[test]
fn msm() {
    use arrabbiata_curve::{Projective, msm};
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Pallas};
    let g = Projective::<Pallas>::generator();
    let h = g + g;
    assert_eq!(
        msm(&[g, h], &[Fq::from_u64(3), Fq::from_u64(1)]),
        g * Fq::from_u64(5)
    );
}

#[test]
fn absorb_point() {
    use arrabbiata_curve::{Projective, absorb_point};
    use arrabbiata_pasta::Pallas;
    use arrabbiata_poseidon::vesta_sponge;
    use arrabbiata_transcript::Transcript;
    let mut t = vesta_sponge(0);
    absorb_point(&Projective::<Pallas>::generator(), &mut t);
    let _ = t.squeeze();
}

#[test]
fn points_from_transcript() {
    use arrabbiata_curve::points_from_transcript;
    use arrabbiata_pasta::Pallas;
    use arrabbiata_poseidon::pallas_sponge;
    let points = points_from_transcript::<Pallas, _>(&mut pallas_sponge(0), 4);
    assert_eq!(points.len(), 4);
    assert!(points.iter().all(|p| p.is_on_curve() && !p.is_identity()));
}

#[test]
fn pairing() {
    use arrabbiata_bn254::{Bn254, Fr};
    use arrabbiata_curve::{Pairing, Projective};
    use arrabbiata_field::Field;

    fn check<E>(a: <E::G1 as arrabbiata_curve::Curve>::Scalar) -> bool
    where
        E: Pairing,
    {
        let (p, q) = (
            Projective::<E::G1>::generator(),
            Projective::<E::G2>::generator(),
        );
        // e(aP, Q) e(-P, aQ) = 1.
        E::pairing_product_is_one(&[(p * a, q), (-p, q * a)])
    }
    assert!(check::<Bn254>(Fr::from_u64(7)));
}

#[test]
fn projective_pairing_product_is_one() {
    use arrabbiata_bn254::Bn254;
    use arrabbiata_curve::{Pairing, Projective};
    let (p, q) = (Projective::generator(), Projective::generator());
    assert!(!Bn254::pairing_product_is_one(&[(p, q)]));
}
