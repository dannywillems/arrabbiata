//! Property tests of the folding scheme.
//!
//! The generators produce *arbitrary* satisfying relaxed pairs, not only
//! fresh ones: random witness columns, random `u` and coefficients, and the
//! error vector `E` set to the value of the relation, so that `P(u, c, w) =
//! E` holds by construction. This exercises the scheme on the full relaxed
//! relation rather than on the special case `u = 1, E = 0`.

use arrabbiata_commit::{HomomorphicCommitment, IdentityCommitment};
use arrabbiata_field::{Field, Goldilocks as F};
use arrabbiata_folding::{Fold, Nova};
use arrabbiata_poly::Scalars;
use arrabbiata_relation::{
    ConstraintSystem, RelaxedInstance, RelaxedWitness, RowView,
    aggregation::{Aggregation, Combiner, NoCoefficients, Powers, Single},
    examples,
};
use arrabbiata_transcript::toy_sponge;
use proptest::prelude::*;

type Ck = IdentityCommitment;
type C = <Ck as HomomorphicCommitment<F>>::Commitment;
type FibScheme = Fold<F, Ck, Powers, 2, 1, 2, 2, 2>;
type SqScheme = Nova<F, Ck, 1, 1>;

const ROWS: usize = 6;

fn arb_f() -> impl Strategy<Value = F> {
    prop_oneof![
        1 => Just(F::ZERO),
        1 => Just(F::ONE),
        8 => any::<u64>().prop_map(F::new),
    ]
}

fn arb_columns<const W: usize>() -> impl Strategy<Value = [Vec<F>; W]> {
    prop::collection::vec(prop::collection::vec(arb_f(), ROWS), W)
        .prop_map(|v| <[Vec<F>; W]>::try_from(v).expect("W columns"))
}

/// The relaxed pair `(u, c, w, E := P(u, c, w))`, which satisfies the
/// relaxed relation by construction.
fn satisfying<A, const W: usize, const X: usize, const M: usize, const D: usize>(
    cs: &ConstraintSystem<F, A, W, X, M, D>,
    u: F,
    coefficients: A::Coefficients,
    columns: [Vec<F>; W],
) -> (
    RelaxedInstance<F, C, A::Coefficients, W>,
    RelaxedWitness<F, W>,
)
where
    A: Aggregation<F, M>,
{
    let ring = Scalars::new();
    let lifted: Option<Vec<F>> = A::coefficient(&coefficients, 0).map(|_| {
        (0..M)
            .map(|j| A::coefficient(&coefficients, j).unwrap())
            .collect()
    });
    let error: Vec<F> = (0..ROWS)
        .map(|row| {
            let view = RowView::new(&columns, cs.fixed_columns(), row, ROWS);
            cs.evaluate_row(&ring, &view, &u, lifted.as_deref())
        })
        .collect();
    let ck = IdentityCommitment;
    let commitments = columns.clone().map(|c| ck.commit(&c));
    (
        RelaxedInstance::new(u, coefficients, commitments, ck.commit(&error)),
        RelaxedWitness::new(columns, error),
    )
}

fn arb_fib_pair() -> impl Strategy<
    Value = (
        RelaxedInstance<F, C, Combiner<F, 2>, 2>,
        RelaxedWitness<F, 2>,
    ),
> {
    (arb_f(), arb_f(), arb_f(), arb_columns::<2>()).prop_map(|(u, c0, c1, columns)| {
        // Arbitrary coefficients: folded ones are not powers of anything.
        let fresh = Combiner::<F, 2>::powers(c1);
        let coefficients = <Powers as Aggregation<F, 2>>::fold(&fresh, &fresh, c0);
        satisfying(&examples::fibonacci(ROWS), u, coefficients, columns)
    })
}

fn arb_sq_pair() -> impl Strategy<
    Value = (
        RelaxedInstance<F, C, NoCoefficients, 1>,
        RelaxedWitness<F, 1>,
    ),
> {
    (arb_f(), arb_columns::<1>())
        .prop_map(|(u, columns)| satisfying(&examples::squaring(ROWS), u, NoCoefficients, columns))
}

proptest! {
    /// Completeness: folding two satisfying relaxed pairs gives a satisfying
    /// pair, and the verifier computes the prover's instance.
    #[test]
    fn completeness_powers(a in arb_fib_pair(), b in arb_fib_pair(), seed in any::<u64>()) {
        let cs = examples::fibonacci(ROWS);
        let ck = IdentityCommitment;
        cs.check_relaxed(&ck, &a.0, &a.1).unwrap();
        cs.check_relaxed(&ck, &b.0, &b.1).unwrap();
        let (inst, wit, proof, _) =
            FibScheme::prove(&cs, &ck, toy_sponge(seed), (&a.0, &a.1), (&b.0, &b.1)).unwrap();
        prop_assert!(cs.check_relaxed(&ck, &inst, &wit).is_ok());
        let (verified, _) = FibScheme::verify(&cs, &ck, toy_sponge(seed), &a.0, &b.0, proof);
        prop_assert_eq!(verified, inst);
    }

    /// Completeness for Nova.
    #[test]
    fn completeness_nova(a in arb_sq_pair(), b in arb_sq_pair(), seed in any::<u64>()) {
        let cs = examples::squaring(ROWS);
        let ck = IdentityCommitment;
        let (inst, wit, proof, _) =
            SqScheme::prove(&cs, &ck, toy_sponge(seed), (&a.0, &a.1), (&b.0, &b.1)).unwrap();
        prop_assert!(cs.check_relaxed(&ck, &inst, &wit).is_ok());
        let (verified, _) = SqScheme::verify(&cs, &ck, toy_sponge(seed), &a.0, &b.0, proof);
        prop_assert_eq!(verified, inst);
    }

    /// Nova's closed form. For the squaring constraint q (next - curr^2),
    /// homogenized to q (u next - curr^2), the cross-term is
    /// q (u1 next2 + u2 next1 - 2 curr1 curr2), and the folded error is
    /// E1 + r T + r^2 E2. Recover T from the folded witness at two seeds.
    #[test]
    fn nova_cross_term_closed_form(a in arb_sq_pair(), b in arb_sq_pair()) {
        let cs = examples::squaring(ROWS);
        let ck = IdentityCommitment;
        let (inst, wit, _, _) =
            SqScheme::prove(&cs, &ck, toy_sponge(0), (&a.0, &a.1), (&b.0, &b.1)).unwrap();
        // r is u - u1 over u2, when u2 is invertible.
        prop_assume!(!b.0.u().is_zero());
        let r = (inst.u() - a.0.u()) * b.0.u().inverse().unwrap();
        let q = &cs.fixed_columns()[0];
        let (w1, w2) = (&a.1.columns()[0], &b.1.columns()[0]);
        for row in 0..ROWS {
            let next = (row + 1) % ROWS;
            let t = q[row]
                * (a.0.u() * w2[next] + b.0.u() * w1[next] - F::new(2) * w1[row] * w2[row]);
            let expected = a.1.error()[row] + r * t + r * r * b.1.error()[row];
            prop_assert_eq!(wit.error()[row], expected);
        }
    }

    /// Accumulating many fresh instances, as an IVC prover does, keeps the
    /// accumulator satisfying.
    #[test]
    fn repeated_folding(starts in prop::collection::vec((1u64..1000, 1u64..1000), 1..6)) {
        let cs = examples::fibonacci(ROWS);
        let ck = IdentityCommitment;
        let fresh = |(x, y): (u64, u64)| {
            let trace = examples::fibonacci_trace(&cs, x, y);
            let instance = cs.commit(&ck, &trace, &mut toy_sponge(x ^ y));
            cs.relax(&ck, instance, trace)
        };
        let (mut acc, mut acc_w) = fresh((7, 11));
        for (i, start) in starts.into_iter().enumerate() {
            let (new, new_w) = fresh(start);
            let (inst, wit, proof, _) =
                FibScheme::prove(&cs, &ck, toy_sponge(i as u64), (&acc, &acc_w), (&new, &new_w))
                    .unwrap();
            let (verified, _) =
                FibScheme::verify(&cs, &ck, toy_sponge(i as u64), &acc, &new, proof);
            prop_assert_eq!(&verified, &inst);
            acc = verified;
            acc_w = wit;
            prop_assert!(cs.check_relaxed(&ck, &acc, &acc_w).is_ok());
        }
    }

    /// Soundness, informally: folding a fresh instance whose witness does
    /// not satisfy the circuit yields an accumulator that does not satisfy
    /// the relaxed relation (except with probability about 3 / |F| over r).
    #[test]
    fn unsatisfied_input_stays_unsatisfied(a in arb_fib_pair(), columns in arb_columns::<2>()) {
        let cs = examples::fibonacci(ROWS);
        let ck = IdentityCommitment;
        let bad = cs.trace(columns).unwrap();
        prop_assume!(cs.check_trace(&bad).is_err());
        let instance = cs.commit(&ck, &bad, &mut toy_sponge(0));
        let (b, bw) = cs.relax(&ck, instance, bad);
        let (inst, wit, _, _) =
            FibScheme::prove(&cs, &ck, toy_sponge(0), (&a.0, &a.1), (&b, &bw)).unwrap();
        prop_assert!(cs.check_relaxed(&ck, &inst, &wit).is_err());
    }

    /// A tampered proof changes the challenge and the verifier's instance,
    /// which then does not open to the honest folded witness.
    #[test]
    fn tampered_proof_is_rejected(a in arb_fib_pair(), b in arb_fib_pair(), delta in 1u64..1000) {
        let cs = examples::fibonacci(ROWS);
        let ck = IdentityCommitment;
        let (_, wit, proof, _) =
            FibScheme::prove(&cs, &ck, toy_sponge(0), (&a.0, &a.1), (&b.0, &b.1)).unwrap();
        let mut terms = proof.head.commitments().clone();
        terms[0] = ck.add(&terms[0], &ck.commit(&[F::new(delta)]));
        let tampered = arrabbiata_transcript::hlist::HCons::new(
            arrabbiata_folding::CrossTerms::new(terms),
            arrabbiata_transcript::hlist::HNil,
        );
        let (verified, _) = FibScheme::verify(&cs, &ck, toy_sponge(0), &a.0, &b.0, tampered);
        prop_assert!(cs.check_relaxed(&ck, &verified, &wit).is_err());
    }
}

#[test]
fn prover_and_verifier_transcripts_end_in_the_same_state() {
    use arrabbiata_transcript::Transcript;
    let cs = examples::squaring(ROWS);
    let ck = IdentityCommitment;
    let trace = examples::squaring_trace(&cs, 3);
    let (inst, wit) = cs.relax(&ck, cs.commit(&ck, &trace, &mut toy_sponge(0)), trace);
    let (_, _, proof, mut p) =
        SqScheme::prove(&cs, &ck, toy_sponge(9), (&inst, &wit), (&inst, &wit)).unwrap();
    let (_, mut v) = SqScheme::verify(&cs, &ck, toy_sponge(9), &inst, &inst, proof);
    assert_eq!(p.squeeze(), v.squeeze());
}

#[test]
fn single_is_nova() {
    // The type alias is the instantiation, nothing more.
    let _: Option<Nova<F, Ck, 1, 1>> = None::<Fold<F, Ck, Single, 1, 1, 1, 2, 1>>;
}
