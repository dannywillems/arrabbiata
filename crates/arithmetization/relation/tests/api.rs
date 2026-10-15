//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn aggregation_overview() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, Powers, Single};

    assert_eq!(<Single as Aggregation<F, 1>>::EXTRA_DEGREE, 0);
    assert_eq!(<Powers as Aggregation<F, 3>>::EXTRA_DEGREE, 1);
}

#[test]
fn aggregation_aggregation() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, Combiner, Powers};

    let a = Combiner::<F, 2>::powers(F::new(3));
    let b = Combiner::<F, 2>::powers(F::new(5));
    let folded = <Powers as Aggregation<F, 2>>::fold(&a, &b, F::new(2));
    // 3 + 2 * 5: the coefficient of constraint 1 is folded linearly.
    assert_eq!(
        <Powers as Aggregation<F, 2>>::coefficient(&folded, 1),
        Some(F::new(13))
    );
}

#[test]
fn aggregation_extra_degree() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, Powers};
    assert_eq!(<Powers as Aggregation<F, 4>>::EXTRA_DEGREE, 1);
}

#[test]
fn aggregation_coefficient() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, NoCoefficients, Single};
    assert_eq!(
        <Single as Aggregation<F, 1>>::coefficient(&NoCoefficients, 0),
        None
    );
}

#[test]
fn aggregation_fold() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, Combiner, Powers};
    let one = Combiner::<F, 1>::powers(F::new(9));
    let folded = <Powers as Aggregation<F, 1>>::fold(&one, &one, F::new(4));
    // c_0 = 1 on both, so the folded c_0 is 1 + 4.
    assert_eq!(folded.values(), &[F::new(5)]);
}

#[test]
fn aggregation_sample() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_poseidon::vesta_sponge;
    use arrabbiata_relation::aggregation::{Aggregation, Powers};
    let c = <Powers as Aggregation<F, 3>>::sample(&mut vesta_sponge(2));
    let alpha = c.values()[1];
    assert_eq!(c.values(), &[F::ONE, alpha, alpha * alpha]);
}

#[test]
fn aggregation_single() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, Single};
    assert_eq!(<Single as Aggregation<F, 1>>::EXTRA_DEGREE, 0);
}

#[test]
fn aggregation_no_coefficients() {
    use arrabbiata_relation::aggregation::NoCoefficients;
    assert_eq!(NoCoefficients, NoCoefficients);
}

#[test]
fn aggregation_powers() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::{Aggregation, Powers};
    assert_eq!(<Powers as Aggregation<F, 2>>::EXTRA_DEGREE, 1);
}

#[test]
fn aggregation_combiner() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_relation::aggregation::Combiner;
    let c = Combiner::<F, 3>::powers(F::new(2));
    assert_eq!(c.values(), &[F::ONE, F::new(2), F::new(4)]);
}

#[test]
fn aggregation_combiner_powers() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_relation::aggregation::Combiner;
    assert_eq!(
        Combiner::<F, 2>::powers(F::new(7)).values(),
        &[F::ONE, F::new(7)]
    );
}

#[test]
fn aggregation_combiner_values() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::aggregation::Combiner;
    assert_eq!(Combiner::<F, 1>::powers(F::new(7)).values().len(), 1);
}

#[test]
fn error_shape_error() {
    use arrabbiata_relation::error::ShapeError;
    let e = ShapeError::ConstraintDegree {
        constraint: 0,
        degree: 3,
        maximum: 2,
    };
    assert_eq!(
        e.to_string(),
        "constraint 0 has degree 3, above the maximum 2"
    );
}

#[test]
fn error_slot() {
    use arrabbiata_relation::error::Slot;
    assert_ne!(Slot::Column(0), Slot::Error);
}

#[test]
fn error_relation_error() {
    use arrabbiata_relation::error::RelationError;
    let e = RelationError::Unsatisfied {
        row: 3,
        constraint: Some(1),
    };
    assert_eq!(e.to_string(), "constraint 1 is not satisfied on row 3");
}

#[test]
fn examples_overview() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;

    let cs = examples::fibonacci::<Fq>(6);
    let trace = examples::fibonacci_trace(&cs, 2, 3);
    assert!(cs.check_trace(&trace).is_ok());
}

#[test]
fn examples_squaring() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples::{Squaring, squaring};
    let cs: Squaring<Fq> = squaring(4);
    assert_eq!(Squaring::<Fq>::CROSS_TERMS, 1);
    let _ = cs;
}

#[test]
fn examples_fibonacci() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples::{Fibonacci, fibonacci};
    let cs: Fibonacci<Fq> = fibonacci(4);
    assert_eq!(Fibonacci::<Fq>::DEGREE, 3);
    let _ = cs;
}

#[test]
fn examples_squaring_2() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples::squaring;
    assert_eq!(squaring::<Fq>(16).num_rows(), 16);
}

#[test]
fn examples_squaring_trace() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples::{squaring, squaring_trace};
    let cs = squaring::<Fq>(3);
    let expected = vec![Fq::from_u64(3), Fq::from_u64(9), Fq::from_u64(81)];
    assert_eq!(squaring_trace(&cs, 3).columns()[0], expected);
}

#[test]
fn examples_fibonacci_2() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples::fibonacci;
    assert_eq!(fibonacci::<Fq>(5).constraints().len(), 2);
}

#[test]
fn examples_fibonacci_trace() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples::{fibonacci, fibonacci_trace};
    let cs = fibonacci::<Fq>(3);
    let t = fibonacci_trace(&cs, 2, 3);
    assert_eq!(
        t.columns()[1],
        vec![Fq::from_u64(3), Fq::from_u64(6), Fq::from_u64(18)]
    );
}

#[test]
fn instance_instance() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    // The column commitment is the Pedersen commitment to the column.
    use arrabbiata_commit::HomomorphicCommitment;
    assert_eq!(instance.columns()[0], ck.commit(&trace.columns()[0]));
}

#[test]
fn instance_instance_new() {
    use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    let i = Instance::new([7u8], NoCoefficients);
    assert_eq!(i.columns(), &[7]);
}

#[test]
fn instance_instance_columns() {
    use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    assert_eq!(Instance::new([1u8, 2], NoCoefficients).columns().len(), 2);
}

#[test]
fn instance_instance_coefficients() {
    use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    assert_eq!(
        Instance::new([1u8], NoCoefficients).coefficients(),
        &NoCoefficients
    );
}

#[test]
fn instance_instance_into_parts() {
    use arrabbiata_relation::{Instance, aggregation::NoCoefficients};
    let (columns, _) = Instance::new([1u8], NoCoefficients).into_parts();
    assert_eq!(columns, [1]);
}

#[test]
fn instance_relaxed_instance() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    let i = RelaxedInstance::new(F::ONE, NoCoefficients, [10u8], 0u8);
    assert_eq!(i.u(), F::ONE);
}

#[test]
fn instance_relaxed_instance_new() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    let i = RelaxedInstance::new(F::new(2), NoCoefficients, [1u8, 2], 3u8);
    assert_eq!(i.error(), &3);
}

#[test]
fn instance_relaxed_instance_u() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    assert_eq!(
        RelaxedInstance::new(F::new(4), NoCoefficients, [0u8], 0u8).u(),
        F::new(4)
    );
}

#[test]
fn instance_relaxed_instance_coefficients() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    let i = RelaxedInstance::new(F::new(1), NoCoefficients, [0u8], 0u8);
    assert_eq!(i.coefficients(), &NoCoefficients);
}

#[test]
fn instance_relaxed_instance_columns() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    let i = RelaxedInstance::new(F::new(1), NoCoefficients, [5u8], 0u8);
    assert_eq!(i.columns(), &[5]);
}

#[test]
fn instance_relaxed_instance_error() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::{RelaxedInstance, aggregation::NoCoefficients};
    let i = RelaxedInstance::new(F::new(1), NoCoefficients, [5u8], 6u8);
    assert_eq!(i.error(), &6);
}

#[test]
fn instance_relaxed_witness() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_relation::RelaxedWitness;
    let w = RelaxedWitness::new([vec![F::ONE]], vec![F::ZERO]);
    assert_eq!(w.error(), &[F::ZERO]);
}

#[test]
fn instance_relaxed_witness_new() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_relation::RelaxedWitness;
    let w = RelaxedWitness::new([vec![F::ONE, F::ONE]], vec![F::ZERO; 2]);
    assert_eq!(w.columns()[0].len(), 2);
}

#[test]
fn instance_relaxed_witness_columns() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_relation::RelaxedWitness;
    assert_eq!(
        RelaxedWitness::new([vec![F::ONE]], vec![]).columns()[0],
        vec![F::ONE]
    );
}

#[test]
fn instance_relaxed_witness_error() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_relation::RelaxedWitness;
    assert!(
        RelaxedWitness::<F, 1>::new([vec![]], vec![])
            .error()
            .is_empty()
    );
}

#[test]
fn crate_overview() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;

    // Pedersen commitments on Pallas, Poseidon for Fiat-Shamir.
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 8, vesta_sponge(1));

    // Arithmetize, commit, relax, check.
    let cs = examples::fibonacci::<Fq>(8);
    let trace = examples::fibonacci_trace(&cs, 1, 2);
    cs.check_trace(&trace).unwrap();
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    let (instance, witness) = cs.relax(&ck, instance, trace);
    cs.check_relaxed(&ck, &instance, &witness).unwrap();
}

#[test]
fn system_constraint_system() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    assert!(cs.check_trace(&trace).is_ok());
}

#[test]
fn system_trace() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 2);
    assert_eq!(trace.columns()[0][..3], [2, 4, 16].map(Fq::from_u64));
}

#[test]
fn system_trace_columns() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;
    let cs = examples::squaring::<Fq>(2);
    assert_eq!(examples::squaring_trace(&cs, 5).columns()[0].len(), 2);
}

#[test]
fn system_row_view() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Fq as F};
    use arrabbiata_poly::{Assignment, Rotation, Scalars, WitnessColumn};
    use arrabbiata_relation::{RowView, examples};

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let row = RowView::new(trace.columns(), cs.fixed_columns(), 1, cs.num_rows());
    let next = row.witness(&Scalars::new(), WitnessColumn::new::<0>(), Rotation::NEXT);
    assert_eq!(next, F::from_u64(81));
}

#[test]
fn system_row_view_new() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::{RowView, examples};
    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let _row = RowView::new(trace.columns(), cs.fixed_columns(), 0, cs.num_rows());
}

#[test]
fn system_row_view_degree() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;
    assert_eq!(examples::Squaring::<Fq>::DEGREE, 2);
    assert_eq!(examples::Fibonacci::<Fq>::DEGREE, 3);
}

#[test]
fn system_row_view_cross_terms() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;
    assert_eq!(examples::Squaring::<Fq>::CROSS_TERMS, 1);
    assert_eq!(examples::Fibonacci::<Fq>::CROSS_TERMS, 2);
}

#[test]
fn system_row_view_new_2() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
    use arrabbiata_relation::{ConstraintSystem, aggregation::Single, error::ShapeError};
    use core::num::NonZeroUsize;

    let x = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
    let cubic = x.clone() * x.clone() * x;
    let rows = NonZeroUsize::new(4).unwrap();
    // A cubic constraint does not fit a degree-2 system.
    let cs = ConstraintSystem::<F, Single, 1, 0, 1, 2>::new(rows, [], [cubic]);
    assert_eq!(
        cs.unwrap_err(),
        ShapeError::ConstraintDegree {
            constraint: 0,
            degree: 3,
            maximum: 2
        }
    );
}

#[test]
fn system_row_view_num_rows() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;
    assert_eq!(examples::squaring::<Fq>(8).num_rows(), 8);
}

#[test]
fn system_row_view_fixed_columns() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;
    assert_eq!(examples::squaring::<Fq>(3).fixed_columns()[0].len(), 3);
}

#[test]
fn system_row_view_constraints() {
    use arrabbiata_pasta::Fq;
    use arrabbiata_relation::examples;
    assert_eq!(examples::fibonacci::<Fq>(4).constraints().len(), 2);
}

#[test]
fn system_row_view_trace() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Fq as F};
    use arrabbiata_relation::examples;
    let cs = examples::squaring::<Fq>(2);
    assert!(cs.trace([vec![F::from_u64(1), F::from_u64(1)]]).is_ok());
    assert!(cs.trace([vec![F::from_u64(1)]]).is_err());
}

#[test]
fn system_row_view_evaluate_row() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Fq as F};
    use arrabbiata_poly::Scalars;
    use arrabbiata_relation::{RowView, examples};

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let row = RowView::new(trace.columns(), cs.fixed_columns(), 0, cs.num_rows());
    assert_eq!(
        cs.evaluate_row(&Scalars::new(), &row, &F::ONE, None),
        F::ZERO
    );
}

#[test]
fn system_row_view_check_trace() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::{Fq, Fq as F};
    use arrabbiata_relation::{error::RelationError, examples};

    let cs = examples::squaring::<Fq>(3);
    let bad = cs
        .trace([vec![F::from_u64(2), F::from_u64(5), F::from_u64(25)]])
        .unwrap();
    assert_eq!(
        cs.check_trace(&bad),
        Err(RelationError::Unsatisfied {
            row: 0,
            constraint: Some(0)
        })
    );
}

#[test]
fn system_row_view_check_relaxed_witness() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    let (instance, witness) = cs.relax(&ck, instance, trace);
    assert!(
        cs.check_relaxed_witness(instance.u(), instance.coefficients(), &witness)
            .is_ok()
    );
}

#[test]
fn system_row_view_check_relaxed() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::fibonacci::<Fq>(8);
    let trace = examples::fibonacci_trace(&cs, 1, 1);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    let (instance, witness) = cs.relax(&ck, instance, trace);
    assert!(cs.check_relaxed(&ck, &instance, &witness).is_ok());
}

#[test]
fn system_row_view_commit() {
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    assert_eq!(instance.columns().len(), 1);
}

#[test]
fn system_row_view_relax() {
    use arrabbiata_field::Field;
    use arrabbiata_ipa::Ipa;
    use arrabbiata_pasta::{Fq, Fq as F, Pallas};
    use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    use arrabbiata_relation::examples;
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    let (instance, witness) = cs.relax(&ck, instance, trace);
    assert_eq!(instance.u(), F::ONE);
    assert!(witness.error().iter().all(|e| *e == F::ZERO));
}
