//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn crate_overview() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::{Permutation, grand_product, labels, shifts};
    use arrabbiata_poly::Domain;

    // Two columns of four rows; copy (0, 1) to (1, 2).
    let columns = 2;
    let rows = 4;
    let copies = [(1, 4 + 2)];
    let sigma = Permutation::from_copies(columns, rows, &copies).unwrap();
    let domain = Domain::<F>::new(2).unwrap();
    let k = shifts::<F>(columns, rows).unwrap();
    let sigma_labels = labels(&sigma, &k, &domain);
    let beta = F::from_u64(3);
    let gamma = F::from_u64(5);
    let good = [
        vec![F::ONE, F::from_u64(7), F::ZERO, F::ZERO],
        vec![F::ZERO, F::ZERO, F::from_u64(7), F::ZERO],
    ];
    let z = grand_product(&good, &k, &sigma_labels, &domain, beta, gamma).unwrap();
    assert_eq!(z[0], F::ONE);
    assert!(sigma.is_respected_by(&good));
    // The product closes: the last step returns to 1 exactly when respected.
    assert!(arrabbiata_permutation::closes(
        &good,
        &k,
        &sigma_labels,
        &domain,
        &z,
        beta,
        gamma
    ));
    let mut bad = good.clone();
    bad[1][2] = F::from_u64(8);
    let z = grand_product(&bad, &k, &sigma_labels, &domain, beta, gamma).unwrap();
    assert!(!sigma.is_respected_by(&bad));
    assert!(!arrabbiata_permutation::closes(
        &bad,
        &k,
        &sigma_labels,
        &domain,
        &z,
        beta,
        gamma
    ));
}

#[test]
fn error() {
    use arrabbiata_permutation::Error;
    assert_eq!(
        Error::Shape.to_string(),
        "not a permutation of the cells, or the wrong shape"
    );
}

#[test]
fn permutation() {
    use arrabbiata_permutation::{Error, Permutation};
    let p = Permutation::new(1, 2, vec![1, 0]).unwrap();
    assert_eq!(p.sigma(), &[1, 0]);
    assert_eq!(
        Permutation::new(1, 2, vec![0, 0]).unwrap_err(),
        Error::Shape
    );
}

#[test]
fn permutation_new() {
    use arrabbiata_permutation::Permutation;
    assert!(Permutation::new(2, 2, vec![3, 1, 2, 0]).is_ok());
}

#[test]
fn permutation_identity() {
    use arrabbiata_permutation::Permutation;
    assert_eq!(Permutation::identity(2, 2).sigma(), &[0, 1, 2, 3]);
}

#[test]
fn permutation_from_copies() {
    use arrabbiata_permutation::Permutation;
    // 0 ~ 2 ~ 3, and 1 alone: the cycle 0 -> 2 -> 3 -> 0.
    let p = Permutation::from_copies(1, 4, &[(3, 0), (2, 3)]).unwrap();
    assert_eq!(p.sigma(), &[2, 1, 3, 0]);
    assert_eq!(
        p,
        Permutation::from_copies(1, 4, &[(0, 2), (2, 3), (0, 3)]).unwrap()
    );
}

#[test]
fn permutation_sigma() {
    use arrabbiata_permutation::Permutation;
    assert_eq!(Permutation::identity(1, 2).sigma(), &[0, 1]);
}

#[test]
fn permutation_columns() {
    use arrabbiata_permutation::Permutation;
    assert_eq!(Permutation::identity(3, 2).columns(), 3);
}

#[test]
fn permutation_rows() {
    use arrabbiata_permutation::Permutation;
    assert_eq!(Permutation::identity(3, 2).rows(), 2);
}

#[test]
fn permutation_is_respected_by() {
    use arrabbiata_permutation::Permutation;
    let p = Permutation::new(1, 2, vec![1, 0]).unwrap();
    assert!(p.is_respected_by(&[vec![5, 5]]));
    assert!(!p.is_respected_by(&[vec![5, 6]]));
}

#[test]
fn shifts() {
    use arrabbiata_field::{Field, TwoAdicField};
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::shifts;
    let k = shifts::<F>(3, 8).unwrap();
    assert_eq!(k[0], F::ONE);
    assert_eq!(k[2], F::MULTIPLICATIVE_GENERATOR.pow(2));
}

#[test]
fn labels() {
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::{Permutation, labels, shifts};
    use arrabbiata_poly::Domain;
    let p = Permutation::identity(2, 2);
    let domain = Domain::<F>::new(1).unwrap();
    let k = shifts::<F>(2, 2).unwrap();
    // The identity labels each cell by itself.
    assert_eq!(
        labels(&p, &k, &domain)[1],
        vec![k[1], k[1] * domain.element(1)]
    );
}

#[test]
fn grand_product() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::{Permutation, grand_product, labels, shifts};
    use arrabbiata_poly::Domain;
    // With the identity, every factor is 1.
    let p = Permutation::identity(1, 4);
    let domain = Domain::<F>::new(2).unwrap();
    let k = shifts::<F>(1, 4).unwrap();
    let column = [vec![F::from_u64(9); 4]];
    let z = grand_product(
        &column,
        &k,
        &labels(&p, &k, &domain),
        &domain,
        F::from_u64(2),
        F::from_u64(3),
    )
    .unwrap();
    assert_eq!(z, vec![F::ONE; 4]);
}

#[test]
fn closes() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::{Permutation, closes, grand_product, labels, shifts};
    use arrabbiata_poly::Domain;
    let p = Permutation::new(1, 2, vec![1, 0]).unwrap();
    let domain = Domain::<F>::new(1).unwrap();
    let k = shifts::<F>(1, 2).unwrap();
    let l = labels(&p, &k, &domain);
    let (beta, gamma) = (F::from_u64(2), F::from_u64(3));
    let column = [vec![F::ONE, F::ONE]];
    let z = grand_product(&column, &k, &l, &domain, beta, gamma).unwrap();
    assert!(closes(&column, &k, &l, &domain, &z, beta, gamma));
}

#[test]
fn at() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::At;
    let at = At {
        x: F::ONE,
        witness: &[F::ONE],
        sigmas: &[F::ONE],
        z: F::ONE,
        z_next: F::ONE,
        l0: F::ZERO,
    };
    assert_eq!(at.witness.len(), 1);
}

#[test]
fn terms() {
    use arrabbiata_field::Field;
    use arrabbiata_pasta::Fq as F;
    use arrabbiata_permutation::{At, terms};
    // One column, the identity permutation at the point x = sigma = k x:
    // the step is z - z(w x).
    let at = At {
        x: F::ONE,
        witness: &[F::from_u64(4)],
        sigmas: &[F::ONE],
        z: F::from_u64(7),
        z_next: F::from_u64(7),
        l0: F::ONE,
    };
    let [step, boundary] = terms(&at, &[F::ONE], F::from_u64(2), F::from_u64(3));
    assert!(step.is_zero());
    assert_eq!(boundary, F::from_u64(6));
}

#[test]
fn degree() {
    use arrabbiata_permutation::degree;
    assert_eq!(degree(7), 8);
}
