//! The laws of the permutation argument.
//!
//! - `from_copies` is a permutation that depends only on the partition the
//!   copies generate (their order and redundancy do not matter), and its
//!   cycles are the classes;
//! - `sigma` is respected exactly by assignments constant on the classes;
//! - completeness: respected columns close the grand product, for every
//!   `beta, gamma`; soundness (sampled): an assignment that breaks a class
//!   does not;
//! - the identity terms vanish on every row of `H` exactly when the columns
//!   are respected;
//! - the labels of distinct cells are distinct (the shifts are in distinct
//!   cosets).

use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_pasta::Fq as F;
use arrabbiata_permutation::{At, Permutation, closes, grand_product, labels, shifts, terms};
use arrabbiata_poly::Domain;
use proptest::prelude::*;

const LOG_ROWS: u32 = 3;
const ROWS: usize = 1 << LOG_ROWS;

/// A number of columns and copies between their cells.
fn arb_copies() -> impl Strategy<Value = (usize, Vec<(usize, usize)>)> {
    (1usize..5).prop_flat_map(|columns| {
        let cells = columns * ROWS;
        (
            Just(columns),
            prop::collection::vec((0..cells, 0..cells), 0..12),
        )
    })
}

fn arb_f() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(F::from_u64)
}

/// The class of each cell: the smallest cell of its class.
fn classes(p: &Permutation) -> Vec<usize> {
    let mut class = vec![usize::MAX; p.sigma().len()];
    for start in 0..p.sigma().len() {
        if class[start] != usize::MAX {
            continue;
        }
        let mut members = vec![start];
        let mut x = p.sigma()[start];
        while x != start {
            members.push(x);
            x = p.sigma()[x];
        }
        let min = *members.iter().min().unwrap();
        for m in members {
            class[m] = min;
        }
    }
    class
}

/// An assignment constant on the classes, from a value per class.
fn constant_on_classes(p: &Permutation, value: &[F]) -> Vec<Vec<F>> {
    let class = classes(p);
    (0..p.columns())
        .map(|c| {
            (0..ROWS)
                .map(|i| value[class[c * ROWS + i] % value.len()])
                .collect()
        })
        .collect()
}

/// The identity terms on every row of `H`, with the columns and labels as
/// the values of their polynomials there.
fn terms_on_rows(
    columns: &[Vec<F>],
    k: &[F],
    l: &[Vec<F>],
    domain: &Domain<F>,
    z: &[F],
    beta: F,
    gamma: F,
) -> Vec<[F; 2]> {
    (0..ROWS)
        .map(|i| {
            let witness: Vec<F> = columns.iter().map(|c| c[i]).collect();
            let sigmas: Vec<F> = l.iter().map(|s| s[i]).collect();
            let l0 = if i == 0 { F::ONE } else { F::ZERO };
            let at = At {
                x: domain.element(i),
                witness: &witness,
                sigmas: &sigmas,
                z: z[i],
                z_next: z[(i + 1) % ROWS],
                l0,
            };
            terms(&at, k, beta, gamma)
        })
        .collect()
}

proptest! {
    #[test]
    fn copies_give_the_partition((columns, copies) in arb_copies(), seed in any::<u64>()) {
        let p = Permutation::from_copies(columns, ROWS, &copies).unwrap();
        // A permutation.
        prop_assert!(Permutation::new(columns, ROWS, p.sigma().to_vec()).is_ok());
        // Every copy is within one cycle.
        let class = classes(&p);
        for (x, y) in &copies {
            prop_assert_eq!(class[*x], class[*y]);
        }
        // Reversed, rotated and duplicated copies give the same permutation.
        let mut other: Vec<(usize, usize)> = copies.iter().rev().map(|(x, y)| (*y, *x)).collect();
        if !other.is_empty() {
            let shift = (seed as usize) % other.len();
            other.rotate_left(shift);
            other.push(other[0]);
        }
        prop_assert_eq!(Permutation::from_copies(columns, ROWS, &other).unwrap(), p);
    }

    #[test]
    fn respected_iff_constant_on_classes((columns, copies) in arb_copies(), values in prop::collection::vec(arb_f(), 1..8), cell in any::<usize>(), delta in 1u64..1000) {
        let p = Permutation::from_copies(columns, ROWS, &copies).unwrap();
        let good = constant_on_classes(&p, &values);
        prop_assert!(p.is_respected_by(&good));
        // Change one cell of a class with more than one member.
        let class = classes(&p);
        let cells = columns * ROWS;
        let broken = (0..cells).map(|i| (cell + i) % cells).find(|x| p.sigma()[*x] != *x);
        if let Some(x) = broken {
            let mut bad = good.clone();
            bad[x / ROWS][x % ROWS] = bad[x / ROWS][x % ROWS] + F::from_u64(delta);
            prop_assert!(!p.is_respected_by(&bad));
            prop_assert_ne!(class[x], usize::MAX);
        }
    }

    #[test]
    fn the_product_closes_iff_respected((columns, copies) in arb_copies(), values in prop::collection::vec(arb_f(), 1..8), beta in arb_f(), gamma in arb_f(), cell in any::<usize>(), delta in 1u64..1000) {
        let p = Permutation::from_copies(columns, ROWS, &copies).unwrap();
        let domain = Domain::<F>::new(LOG_ROWS).unwrap();
        let k = shifts::<F>(columns, ROWS).unwrap();
        let l = labels(&p, &k, &domain);
        let good = constant_on_classes(&p, &values);
        if let Ok(z) = grand_product(&good, &k, &l, &domain, beta, gamma) {
            prop_assert!(closes(&good, &k, &l, &domain, &z, beta, gamma));
            // On every row of H, both terms vanish.
            for [step, boundary] in terms_on_rows(&good, &k, &l, &domain, &z, beta, gamma) {
                prop_assert!(step.is_zero() && boundary.is_zero());
            }
        }
        let cells = columns * ROWS;
        if let Some(x) = (0..cells).map(|i| (cell + i) % cells).find(|x| p.sigma()[*x] != *x) {
            let mut bad = good;
            bad[x / ROWS][x % ROWS] = bad[x / ROWS][x % ROWS] + F::from_u64(delta);
            if let Ok(z) = grand_product(&bad, &k, &l, &domain, beta, gamma) {
                prop_assert!(!closes(&bad, &k, &l, &domain, &z, beta, gamma));
                // Some row of H breaks an identity: the last step's.
                let broken = terms_on_rows(&bad, &k, &l, &domain, &z, beta, gamma)
                    .iter()
                    .any(|[step, boundary]| !step.is_zero() || !boundary.is_zero());
                prop_assert!(broken);
            }
        }
    }

    #[test]
    fn labels_are_distinct(columns in 1usize..8) {
        let domain = Domain::<F>::new(LOG_ROWS).unwrap();
        let k = shifts::<F>(columns, ROWS).unwrap();
        let identity = labels(&Permutation::identity(columns, ROWS), &k, &domain);
        let all: Vec<F> = identity.into_iter().flatten().collect();
        for i in 0..all.len() {
            for j in 0..i {
                prop_assert_ne!(all[i], all[j]);
            }
        }
        prop_assert_eq!(k[0], F::ONE);
        prop_assert_eq!(k.get(1).copied().unwrap_or(F::MULTIPLICATIVE_GENERATOR), F::MULTIPLICATIVE_GENERATOR);
    }
}

/// Shapes the functions reject.
#[test]
fn shapes() {
    let domain = Domain::<F>::new(LOG_ROWS).unwrap();
    let k = shifts::<F>(2, ROWS).unwrap();
    let p = Permutation::identity(2, ROWS);
    let l = labels(&p, &k, &domain);
    let short = [vec![F::ONE; ROWS - 1], vec![F::ONE; ROWS]];
    let beta = F::from_u64(2);
    let gamma = F::from_u64(3);
    assert!(grand_product(&short, &k, &l, &domain, beta, gamma).is_err());
    assert!(Permutation::from_copies(1, ROWS, &[(0, ROWS)]).is_err());
    assert!(!p.is_respected_by(&short));
}
