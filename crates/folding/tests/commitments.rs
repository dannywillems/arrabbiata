//! Folding with real commitments: the Pedersen vector commitment of IPA on
//! Pallas, and KZG on BN254. Folding only needs a module morphism, which
//! both are.

use std::num::NonZeroUsize;

use arrabbiata_bn254::{Bn254, Fr};
use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_folding::Fold;
use arrabbiata_ipa::Ipa;
use arrabbiata_kzg::Kzg;
use arrabbiata_pasta::{Fq, Pallas};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};
use arrabbiata_poseidon::{PoseidonSponge, bn254_scalar_sponge, pallas_sponge, vesta_sponge};
use arrabbiata_relation::{ConstraintSystem, Trace, aggregation::Powers};
use proptest::prelude::*;

const ROWS: usize = 8;

/// The multiplicative Fibonacci circuit over any field: two constraints,
/// aggregated with Powers, so the relation has degree 3.
type Fibonacci<F> = ConstraintSystem<F, Powers, 2, 1, 2, 2>;

fn fibonacci<F>() -> Fibonacci<F>
where
    F: PrimeField,
{
    type E<F> = Expr<F, 2, 1>;
    let q = E::<F>::fixed(FixedColumn::new::<0>(), Rotation::Curr);
    let x = E::<F>::witness(WitnessColumn::new::<0>(), Rotation::Curr);
    let y = E::<F>::witness(WitnessColumn::new::<1>(), Rotation::Curr);
    let xn = E::<F>::witness(WitnessColumn::new::<0>(), Rotation::Next);
    let yn = E::<F>::witness(WitnessColumn::new::<1>(), Rotation::Next);
    let mut selector = vec![F::ONE; ROWS];
    selector[ROWS - 1] = F::ZERO;
    Fibonacci::new(
        NonZeroUsize::new(ROWS).unwrap(),
        [selector],
        [q.clone() * (xn - y.clone()), q * (yn - x * y)],
    )
    .unwrap()
}

fn trace<F>(cs: &Fibonacci<F>, x0: u64, y0: u64) -> Trace<F, 2>
where
    F: PrimeField,
{
    let (mut x, mut y) = (vec![F::from_u64(x0)], vec![F::from_u64(y0)]);
    for i in 1..ROWS {
        let (px, py) = (x[i - 1], y[i - 1]);
        x.push(py);
        y.push(px * py);
    }
    cs.trace([x, y]).unwrap()
}

/// Fold `steps` fresh instances into an accumulator and check it, with the
/// commitment `ck` and the transcripts `transcript()`.
fn accumulate<F, PC, T>(ck: &PC, transcript: impl Fn() -> T, steps: &[(u64, u64)])
where
    F: PrimeField,
    PC: HomomorphicCommitment<F>,
    T: arrabbiata_transcript::Transcript<F>,
{
    type Scheme<F, PC> = Fold<F, PC, Powers, 2, 1, 2, 2, 2>;
    let cs = fibonacci::<F>();
    let fresh = |(x, y): (u64, u64)| {
        let t = trace(&cs, x, y);
        let instance = cs.commit(ck, &t, &mut transcript());
        cs.relax(ck, instance, t)
    };
    let (mut acc, mut acc_w) = fresh((1, 2));
    for step in steps {
        let (new, new_w) = fresh(*step);
        let (folded, folded_w, proof, _) =
            Scheme::<F, PC>::prove(&cs, ck, transcript(), (&acc, &acc_w), (&new, &new_w)).unwrap();
        let (verified, _) = Scheme::<F, PC>::verify(&cs, ck, transcript(), &acc, &new, proof);
        assert_eq!(verified, folded);
        cs.check_relaxed(ck, &verified, &folded_w).unwrap();
        acc = verified;
        acc_w = folded_w;
    }
}

fn pedersen() -> Ipa<Pallas, PoseidonSponge<Fq>> {
    Ipa::setup(&mut pallas_sponge(0), ROWS, vesta_sponge(1))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    /// IVC-style accumulation with Pedersen commitments on Pallas and a
    /// Poseidon transcript over Fq.
    #[test]
    fn folding_with_pedersen_on_pallas(steps in prop::collection::vec((1u64..1000, 1u64..1000), 1..4)) {
        accumulate::<Fq, _, _>(&pedersen(), || vesta_sponge(2), &steps);
    }

    /// The same with KZG commitments on BN254.
    #[test]
    fn folding_with_kzg_on_bn254(steps in prop::collection::vec((1u64..1000, 1u64..1000), 1..4)) {
        let kzg = Kzg::<Bn254>::insecure_setup(ROWS, Fr::from_u64(0xfeed));
        accumulate::<Fr, _, _>(&kzg, || bn254_scalar_sponge(2), &steps);
    }
}

/// An unsatisfied fresh instance makes the accumulator unsatisfied, with
/// binding commitments.
#[test]
fn unsatisfied_instance_is_caught_with_pedersen() {
    type Scheme = Fold<Fq, Ipa<Pallas, PoseidonSponge<Fq>>, Powers, 2, 1, 2, 2, 2>;
    let ck = pedersen();
    let cs = fibonacci::<Fq>();
    let good = trace(&cs, 1, 2);
    let mut columns = trace(&cs, 3, 4).columns().clone();
    columns[1][2] = columns[1][2] + Fq::ONE;
    let bad = cs.trace(columns).unwrap();
    assert!(cs.check_trace(&bad).is_err());
    let (a, aw) = cs.relax(&ck, cs.commit(&ck, &good, &mut vesta_sponge(2)), good);
    let (b, bw) = cs.relax(&ck, cs.commit(&ck, &bad, &mut vesta_sponge(2)), bad);
    let (folded, folded_w, _, _) =
        Scheme::prove(&cs, &ck, vesta_sponge(2), (&a, &aw), (&b, &bw)).unwrap();
    assert!(cs.check_relaxed(&ck, &folded, &folded_w).is_err());
}
