//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn fold_cross_terms() {
    use arrabbiata_plonkish_folding::CrossTerms;
    let t = CrossTerms([1u8, 2]);
    assert_eq!(t.0[0], 1);
}

#[test]
fn instance_relaxed_instance() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    let i = RelaxedInstance::new(F::ONE, vec![F::ONE], vec![F::from_u64(9)], ["w0"], "E");
    assert_eq!((i.u(), i.public()), (F::ONE, &[F::from_u64(9)][..]));
    assert_eq!(i.error(), &"E");
}

#[test]
fn instance_relaxed_instance_new() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    let i = RelaxedInstance::new(F::from_u64(2), vec![], vec![], [0u8; 3], 1u8);
    assert_eq!(i.columns(), &[0, 0, 0]);
}

#[test]
fn instance_relaxed_instance_coefficients() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    let i = RelaxedInstance::new(F::ONE, vec![F::ONE, F::from_u64(3)], vec![], [(); 0], ());
    assert_eq!(i.coefficients()[1], F::from_u64(3));
}

#[test]
fn instance_relaxed_instance_public() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    let i = RelaxedInstance::new(F::ONE, vec![], vec![F::ZERO], [(); 0], ());
    assert_eq!(i.public(), &[F::ZERO]);
}

#[test]
fn instance_relaxed_instance_columns() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    let i = RelaxedInstance::new(F::ONE, vec![], vec![], [7u8], 0u8);
    assert_eq!(i.columns()[0], 7);
}

#[test]
fn instance_relaxed_instance_error() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    assert_eq!(
        RelaxedInstance::new(F::ONE, vec![], vec![], [0u8; 0], 5u8).error(),
        &5
    );
}

#[test]
fn instance_relaxed_instance_u() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::RelaxedInstance;
    assert_eq!(
        RelaxedInstance::new(F::from_u64(4), vec![], vec![], [(); 0], ()).u(),
        F::from_u64(4)
    );
}

#[test]
fn instance_digest() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::Digest;
    assert_eq!(Digest::new(F::ONE).value(), F::ONE);
}

#[test]
fn instance_digest_new() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::Digest;
    assert_eq!(Digest::new(F::ZERO), Digest::new(F::ZERO));
}

#[test]
fn instance_digest_value() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::Digest;
    assert_eq!(Digest::new(F::from_u64(3)).value(), F::from_u64(3));
}

#[test]
fn instance_public_inputs() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_plonkish_folding::PublicInputs;
    assert_eq!(PublicInputs(vec![F::ONE]).0, vec![F::ONE]);
}

#[test]
fn instance_columns() {
    use arrabbiata_plonkish_folding::Columns;
    assert_eq!(Columns([1u8, 2]).0[1], 2);
}

#[test]
fn error() {
    use arrabbiata_plonkish_folding::Error;
    assert_eq!(
        Error::Commitment(0).to_string(),
        "commitment 0 does not open to the witness"
    );
}

#[test]
fn params_params() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::{Error, Params};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    // The arithmetic gate has degree 2: D = 1 is too small.
    let low =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 1>::new(circuit.clone(), ck.clone(), vesta_sponge(2));
    assert_eq!(low.unwrap_err(), Error::Degree);
    // Three columns need three variables.
    let narrow =
        Params::<F, _, Vanilla, 3, 5, 1, 2, 2>::new(circuit.clone(), ck.clone(), vesta_sponge(2));
    assert_eq!(narrow.unwrap_err(), Error::Layout);
    assert!(Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).is_ok());
}

#[test]
fn params_new() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::Params;
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
    assert_eq!(p.circuit().num_rows(), 2);
}

#[test]
fn params_circuit() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::Params;
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit.clone(), ck, vesta_sponge(2)).unwrap();
    assert_eq!(p.circuit(), &circuit);
}

#[test]
fn params_ck() {
    use arrabbiata_commit::HomomorphicCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::Params;
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck.clone(), vesta_sponge(2)).unwrap();
    assert_eq!(p.ck().commit(&[F::ONE]), ck.commit(&[F::ONE]));
}

#[test]
fn params_digest() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::Params;
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let a =
        Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit.clone(), ck.clone(), vesta_sponge(2))
            .unwrap();
    let b = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(3)).unwrap();
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn params_combiner_length() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::Params;
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
    assert_eq!(p.combiner_length(), 1);
}

#[test]
fn params_satisfiability() {
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::Params;
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_reduction::Relation;
    let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
    let r = b.plain_row([F::ONE, F::ZERO, F::ZERO], [F::ZERO; 5]);
    b.public(r);
    let (circuit, w) = b.build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
    assert!(p.satisfiability().check(&vec![F::ONE], &w).is_ok());
    assert!(p.satisfiability().check(&vec![F::ZERO], &w).is_err());
}

#[test]
fn params_relaxed() {
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, RelaxedWitness, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::{Error, Params, RelaxedInstance};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_reduction::Relation;
    let (circuit, w) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
    let rw = RelaxedWitness::embed(&w);
    let columns = std::array::from_fn(|k| p.commit_column(&rw.columns()[k]).unwrap());
    let error = p.commit_column(rw.error()).unwrap();
    let instance = RelaxedInstance::new(F::ONE, vec![F::ONE], vec![], columns, error);
    assert!(p.relaxed().check(&instance, &rw).is_ok());
    // One coefficient per constraint of the largest gadget, not two.
    let wide = RelaxedInstance::new(
        F::ONE,
        vec![F::ONE; 2],
        vec![],
        *instance.columns(),
        *instance.error(),
    );
    assert_eq!(p.relaxed().check(&wide, &rw), Err(Error::Shape));
}

#[test]
fn params_commit_column() {
    use arrabbiata_commit::HomomorphicCommitment;
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::{Error, Params};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    let (circuit, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
    let (a, b, s) = (
        [F::ONE, F::from_u64(2)],
        [F::from_u64(3), F::ZERO],
        F::from_u64(5),
    );
    let folded = [a[0] + s * b[0], a[1] + s * b[1]];
    let ck = p.ck();
    assert_eq!(
        p.commit_column(&folded).unwrap(),
        ck.add(
            &p.commit_column(&a).unwrap(),
            &ck.scale(&p.commit_column(&b).unwrap(), s)
        ),
    );
    assert_eq!(p.commit_column(&[F::ONE]), Err(Error::Shape));
}

#[test]
fn params_satisfiability_2() {
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq as F, Pallas};
    use arrabbiata_plonkish::{Builder, vanilla::Vanilla};
    use arrabbiata_plonkish_folding::{Error, Params};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_reduction::Relation;
    let (circuit, w) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 2, vesta_sponge(1));
    let p = Params::<F, _, Vanilla, 3, 5, 1, 3, 2>::new(circuit, ck, vesta_sponge(2)).unwrap();
    assert_eq!(p.satisfiability().check(&vec![], &w), Ok(()));
    assert_eq!(
        p.satisfiability().check(&vec![F::ONE], &w),
        Err(Error::PublicInputs)
    );
}
