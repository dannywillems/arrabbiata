//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn crate_overview() {
    use arrabbiata_folding::Fold;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
    use arrabbiata_relation::{aggregation::Powers, examples};

    // Fibonacci: degree 2 constraints, two of them, combined with Powers,
    // so the relation has degree 3 and a fold has 2 cross-terms.
    type Scheme = Fold<F, Ipa<Pallas, PoseidonSponge<F>>, Powers, 2, 1, 2, 2, 2>;

    let cs = examples::fibonacci::<F>(8);
    let ck = Ipa::<Pallas, PoseidonSponge<F>>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let fresh = |x, y| {
        let trace = examples::fibonacci_trace(&cs, x, y);
        let instance = cs.commit(&ck, &trace, &mut vesta_sponge(3));
        cs.relax(&ck, instance, trace)
    };
    let (acc, acc_w) = fresh(1, 2);
    let (new, new_w) = fresh(3, 4);

    let (folded, folded_w, proof, _) =
        Scheme::prove(&cs, &ck, vesta_sponge(2), (&acc, &acc_w), (&new, &new_w)).unwrap();
    let (verified, _) = Scheme::verify(&cs, &ck, vesta_sponge(2), &acc, &new, proof);

    assert_eq!(folded, verified);
    cs.check_relaxed(&ck, &verified, &folded_w).unwrap();
}

#[test]
fn cross_terms() {
    use arrabbiata_folding::CrossTerms;
    let t = CrossTerms::new([1u8, 2]);
    assert_eq!(t.commitments(), &[1, 2]);
}

#[test]
fn cross_terms_new() {
    use arrabbiata_folding::CrossTerms;
    assert_eq!(CrossTerms::new(["T1"]).commitments()[0], "T1");
}

#[test]
fn cross_terms_commitments() {
    use arrabbiata_folding::CrossTerms;
    assert_eq!(CrossTerms::new([0u8; 3]).commitments().len(), 3);
}

#[test]
fn folding_transcript() {
    use arrabbiata_folding::FoldingTranscript;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_poseidon::PoseidonSponge;
    use arrabbiata_relation::aggregation::Single;
    use arrabbiata_transcript::typed::Pattern;

    type P = FoldingTranscript<F, Ipa<Pallas, PoseidonSponge<F>>, Single, 1, 1, 1, 2, 1>;
    assert_eq!(<P as Pattern>::MESSAGES, 1);
    assert_eq!(<P as Pattern>::CHALLENGES, 1);
}

#[test]
fn fold() {
    use arrabbiata_folding::Fold;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
    use arrabbiata_relation::{aggregation::Powers, examples};

    type Scheme = Fold<F, Ipa<Pallas, PoseidonSponge<F>>, Powers, 2, 1, 2, 2, 2>;
    let cs = examples::fibonacci::<F>(4);
    let ck = Ipa::<Pallas, PoseidonSponge<F>>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let trace = examples::fibonacci_trace(&cs, 1, 1);
    let (i, w) = cs.relax(&ck, cs.commit(&ck, &trace, &mut vesta_sponge(2)), trace);
    let (folded, _, proof, _) =
        Scheme::prove(&cs, &ck, vesta_sponge(2), (&i, &w), (&i, &w)).unwrap();
    assert_eq!(
        Scheme::verify(&cs, &ck, vesta_sponge(2), &i, &i, proof).0,
        folded
    );
}

#[test]
fn nova() {
    use arrabbiata_field::Field;
    use arrabbiata_folding::Nova;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;

    let cs = examples::squaring::<F>(4);
    let ck = Ipa::<Pallas, PoseidonSponge<F>>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let relaxed = |x| {
        let trace = examples::squaring_trace(&cs, x);
        cs.relax(&ck, cs.commit(&ck, &trace, &mut vesta_sponge(2)), trace)
    };
    let (a, aw) = relaxed(2);
    let (b, bw) = relaxed(3);
    let (_, w, _, _) = Nova::<F, Ipa<Pallas, PoseidonSponge<F>>, 1, 1>::prove(
        &cs,
        &ck,
        vesta_sponge(2),
        (&a, &aw),
        (&b, &bw),
    )
    .unwrap();
    // The folded error is no longer zero: the accumulator is relaxed.
    assert!(w.error().iter().any(|e| *e != F::from_u64(0)));
}

#[test]
fn assignment_prove() {
    use arrabbiata_folding::Nova;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;

    let cs = examples::squaring::<F>(4);
    let ck = Ipa::<Pallas, PoseidonSponge<F>>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let trace = examples::squaring_trace(&cs, 5);
    let (inst, wit) = cs.relax(&ck, cs.commit(&ck, &trace, &mut vesta_sponge(2)), trace);
    let (folded, folded_w, _proof, _sponge) =
        Nova::<F, _, 1, 1>::prove(&cs, &ck, vesta_sponge(2), (&inst, &wit), (&inst, &wit)).unwrap();
    cs.check_relaxed(&ck, &folded, &folded_w).unwrap();
}

#[test]
fn assignment_verify() {
    use arrabbiata_folding::Nova;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;

    let cs = examples::squaring::<F>(4);
    let ck = Ipa::<Pallas, PoseidonSponge<F>>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));
    let trace = examples::squaring_trace(&cs, 5);
    let (inst, wit) = cs.relax(&ck, cs.commit(&ck, &trace, &mut vesta_sponge(2)), trace);
    let (p_inst, _, proof, _) =
        Nova::<F, _, 1, 1>::prove(&cs, &ck, vesta_sponge(2), (&inst, &wit), (&inst, &wit)).unwrap();
    let (v_inst, _) = Nova::<F, _, 1, 1>::verify(&cs, &ck, vesta_sponge(2), &inst, &inst, proof);
    assert_eq!(p_inst, v_inst);
}
