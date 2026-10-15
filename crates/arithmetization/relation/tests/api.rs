//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::{Field, Goldilocks as F};
use arrabbiata_ipa::Ipa;
use arrabbiata_pasta::{Fq, Fq as PastaFq, Pallas};
use arrabbiata_poly::{Assignment, Expr, Rotation, Scalars, WitnessColumn};
use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
use arrabbiata_relation::{
    ConstraintSystem, Instance, RelaxedInstance, RelaxedWitness, RowView,
    aggregation::{Aggregation, Combiner, NoCoefficients, Powers, Single},
    error::{RelationError, ShapeError, Slot},
    examples,
    examples::{Fibonacci, Squaring, fibonacci, fibonacci_trace, squaring, squaring_trace},
};
use core::num::NonZeroUsize;

#[test]
fn aggregation_overview() {
    assert_eq!(<Single as Aggregation<F, 1>>::EXTRA_DEGREE, 0);
    assert_eq!(<Powers as Aggregation<F, 3>>::EXTRA_DEGREE, 1);
}

#[test]
fn aggregation_aggregation() {
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
    assert_eq!(<Powers as Aggregation<F, 4>>::EXTRA_DEGREE, 1);
}

#[test]
fn aggregation_coefficient() {
    assert_eq!(
        <Single as Aggregation<F, 1>>::coefficient(&NoCoefficients, 0),
        None
    );
}

#[test]
fn aggregation_fold() {
    let one = Combiner::<F, 1>::powers(F::new(9));
    let folded = <Powers as Aggregation<F, 1>>::fold(&one, &one, F::new(4));
    // c_0 = 1 on both, so the folded c_0 is 1 + 4.
    assert_eq!(folded.values(), &[F::new(5)]);
}

#[test]
fn aggregation_sample() {
    let c = <Powers as Aggregation<PastaFq, 3>>::sample(&mut vesta_sponge(2));
    let alpha = c.values()[1];
    assert_eq!(c.values(), &[PastaFq::ONE, alpha, alpha * alpha]);
}

#[test]
fn aggregation_single() {
    assert_eq!(<Single as Aggregation<F, 1>>::EXTRA_DEGREE, 0);
}

#[test]
fn aggregation_no_coefficients() {
    assert_eq!(NoCoefficients, NoCoefficients);
}

#[test]
fn aggregation_powers() {
    assert_eq!(<Powers as Aggregation<F, 2>>::EXTRA_DEGREE, 1);
}

#[test]
fn aggregation_combiner() {
    let c = Combiner::<F, 3>::powers(F::new(2));
    assert_eq!(c.values(), &[F::ONE, F::new(2), F::new(4)]);
}

#[test]
fn aggregation_combiner_powers() {
    assert_eq!(
        Combiner::<F, 2>::powers(F::new(7)).values(),
        &[F::ONE, F::new(7)]
    );
}

#[test]
fn aggregation_combiner_values() {
    assert_eq!(Combiner::<F, 1>::powers(F::new(7)).values().len(), 1);
}

#[test]
fn error_shape_error() {
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
    assert_ne!(Slot::Column(0), Slot::Error);
}

#[test]
fn error_relation_error() {
    let e = RelationError::Unsatisfied {
        row: 3,
        constraint: Some(1),
    };
    assert_eq!(e.to_string(), "constraint 1 is not satisfied on row 3");
}

#[test]
fn examples_overview() {
    let cs = examples::fibonacci::<Fq>(6);
    let trace = examples::fibonacci_trace(&cs, 2, 3);
    assert!(cs.check_trace(&trace).is_ok());
}

#[test]
fn examples_squaring() {
    let cs: Squaring<Fq> = squaring(4);
    assert_eq!(Squaring::<Fq>::CROSS_TERMS, 1);
    let _ = cs;
}

#[test]
fn examples_fibonacci() {
    let cs: Fibonacci<Fq> = fibonacci(4);
    assert_eq!(Fibonacci::<Fq>::DEGREE, 3);
    let _ = cs;
}

#[test]
fn examples_squaring_2() {
    assert_eq!(squaring::<Fq>(16).num_rows(), 16);
}

#[test]
fn examples_squaring_trace() {
    let cs = squaring::<Fq>(3);
    let expected = vec![Fq::from_u64(3), Fq::from_u64(9), Fq::from_u64(81)];
    assert_eq!(squaring_trace(&cs, 3).columns()[0], expected);
}

#[test]
fn examples_fibonacci_2() {
    assert_eq!(fibonacci::<Fq>(5).constraints().len(), 2);
}

#[test]
fn examples_fibonacci_trace() {
    let cs = fibonacci::<Fq>(3);
    let t = fibonacci_trace(&cs, 2, 3);
    assert_eq!(
        t.columns()[1],
        vec![Fq::from_u64(3), Fq::from_u64(6), Fq::from_u64(18)]
    );
}

#[test]
fn instance_instance() {
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    // The column commitment is the Pedersen commitment to the column.
    assert_eq!(instance.columns()[0], ck.commit(&trace.columns()[0]));
}

#[test]
fn instance_instance_new() {
    let i = Instance::new([7u8], NoCoefficients);
    assert_eq!(i.columns(), &[7]);
}

#[test]
fn instance_instance_columns() {
    assert_eq!(Instance::new([1u8, 2], NoCoefficients).columns().len(), 2);
}

#[test]
fn instance_instance_coefficients() {
    assert_eq!(
        Instance::new([1u8], NoCoefficients).coefficients(),
        &NoCoefficients
    );
}

#[test]
fn instance_instance_into_parts() {
    let (columns, _) = Instance::new([1u8], NoCoefficients).into_parts();
    assert_eq!(columns, [1]);
}

#[test]
fn instance_relaxed_instance() {
    let i = RelaxedInstance::new(F::ONE, NoCoefficients, [10u8], 0u8);
    assert_eq!(i.u(), F::ONE);
}

#[test]
fn instance_relaxed_instance_new() {
    let i = RelaxedInstance::new(F::new(2), NoCoefficients, [1u8, 2], 3u8);
    assert_eq!(i.error(), &3);
}

#[test]
fn instance_relaxed_instance_u() {
    assert_eq!(
        RelaxedInstance::new(F::new(4), NoCoefficients, [0u8], 0u8).u(),
        F::new(4)
    );
}

#[test]
fn instance_relaxed_instance_coefficients() {
    let i = RelaxedInstance::new(F::new(1), NoCoefficients, [0u8], 0u8);
    assert_eq!(i.coefficients(), &NoCoefficients);
}

#[test]
fn instance_relaxed_instance_columns() {
    let i = RelaxedInstance::new(F::new(1), NoCoefficients, [5u8], 0u8);
    assert_eq!(i.columns(), &[5]);
}

#[test]
fn instance_relaxed_instance_error() {
    let i = RelaxedInstance::new(F::new(1), NoCoefficients, [5u8], 6u8);
    assert_eq!(i.error(), &6);
}

#[test]
fn instance_relaxed_witness() {
    let w = RelaxedWitness::new([vec![F::ONE]], vec![F::ZERO]);
    assert_eq!(w.error(), &[F::ZERO]);
}

#[test]
fn instance_relaxed_witness_new() {
    let w = RelaxedWitness::new([vec![F::ONE, F::ONE]], vec![F::ZERO; 2]);
    assert_eq!(w.columns()[0].len(), 2);
}

#[test]
fn instance_relaxed_witness_columns() {
    assert_eq!(
        RelaxedWitness::new([vec![F::ONE]], vec![]).columns()[0],
        vec![F::ONE]
    );
}

#[test]
fn instance_relaxed_witness_error() {
    assert!(
        RelaxedWitness::<F, 1>::new([vec![]], vec![])
            .error()
            .is_empty()
    );
}

#[test]
fn crate_overview() {
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
    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    assert!(cs.check_trace(&trace).is_ok());
}

#[test]
fn system_trace() {
    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 2);
    assert_eq!(trace.columns()[0][..3], [2, 4, 16].map(Fq::from_u64));
}

#[test]
fn system_trace_columns() {
    let cs = examples::squaring::<Fq>(2);
    assert_eq!(examples::squaring_trace(&cs, 5).columns()[0].len(), 2);
}

#[test]
fn system_row_view() {
    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let row = RowView::new(trace.columns(), cs.fixed_columns(), 1, cs.num_rows());
    let next = row.witness(&Scalars::new(), WitnessColumn::new::<0>(), Rotation::NEXT);
    assert_eq!(next, PastaFq::from_u64(81));
}

#[test]
fn system_row_view_new() {
    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let _row = RowView::new(trace.columns(), cs.fixed_columns(), 0, cs.num_rows());
}

#[test]
fn system_row_view_degree() {
    assert_eq!(examples::Squaring::<Fq>::DEGREE, 2);
    assert_eq!(examples::Fibonacci::<Fq>::DEGREE, 3);
}

#[test]
fn system_row_view_cross_terms() {
    assert_eq!(examples::Squaring::<Fq>::CROSS_TERMS, 1);
    assert_eq!(examples::Fibonacci::<Fq>::CROSS_TERMS, 2);
}

#[test]
fn system_row_view_new_2() {
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
    assert_eq!(examples::squaring::<Fq>(8).num_rows(), 8);
}

#[test]
fn system_row_view_fixed_columns() {
    assert_eq!(examples::squaring::<Fq>(3).fixed_columns()[0].len(), 3);
}

#[test]
fn system_row_view_constraints() {
    assert_eq!(examples::fibonacci::<Fq>(4).constraints().len(), 2);
}

#[test]
fn system_row_view_trace() {
    let cs = examples::squaring::<Fq>(2);
    assert!(
        cs.trace([vec![PastaFq::from_u64(1), PastaFq::from_u64(1)]])
            .is_ok()
    );
    assert!(cs.trace([vec![PastaFq::from_u64(1)]]).is_err());
}

#[test]
fn system_row_view_evaluate_row() {
    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let row = RowView::new(trace.columns(), cs.fixed_columns(), 0, cs.num_rows());
    assert_eq!(
        cs.evaluate_row(&Scalars::new(), &row, &PastaFq::ONE, None),
        PastaFq::ZERO
    );
}

#[test]
fn system_row_view_check_trace() {
    let cs = examples::squaring::<Fq>(3);
    let bad = cs
        .trace([vec![
            PastaFq::from_u64(2),
            PastaFq::from_u64(5),
            PastaFq::from_u64(25),
        ]])
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
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::fibonacci::<Fq>(8);
    let trace = examples::fibonacci_trace(&cs, 1, 1);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    let (instance, witness) = cs.relax(&ck, instance, trace);
    assert!(cs.check_relaxed(&ck, &instance, &witness).is_ok());
}

#[test]
fn system_row_view_commit() {
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    assert_eq!(instance.columns().len(), 1);
}

#[test]
fn system_row_view_relax() {
    let ck = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 16, vesta_sponge(1));

    let cs = examples::squaring::<Fq>(4);
    let trace = examples::squaring_trace(&cs, 3);
    let instance = cs.commit(&ck, &trace, &mut vesta_sponge(2));
    let (instance, witness) = cs.relax(&ck, instance, trace);
    assert_eq!(instance.u(), PastaFq::ONE);
    assert!(witness.error().iter().all(|e| *e == PastaFq::ZERO));
}
