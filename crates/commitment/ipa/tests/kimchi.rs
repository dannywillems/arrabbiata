//! The challenge polynomial and the combined inner product against Kimchi's
//! own `b_poly`, `b_poly_coefficients` and `combined_inner_product`
//! (`poly-commitment/src/commitment.rs`), run on proof-systems
//! `9227444890` by the `ipa_vectors` tool of `tools/kimchi-vectors`, over
//! both Pasta scalar fields. The inputs are small integers, and so are the
//! outputs, which are the same in the two fields.

use arrabbiata_field::Field;
use arrabbiata_ipa::{b_poly, b_poly_coefficients, combined_inner_product};
use arrabbiata_pasta::{Fp, Fq};

/// Kimchi's `b_poly(&[2, 3, 5, 7], 11)`.
const B_POLY: u64 = 0x0c5a_4890_7d38_dc70;
/// Kimchi's `b_poly_coefficients(&[2, 3, 5, 7])`.
const B_POLY_COEFFICIENTS: [u64; 16] = [
    0x01, 0x07, 0x05, 0x23, 0x03, 0x15, 0x0f, 0x69, 0x02, 0x0e, 0x0a, 0x46, 0x06, 0x2a, 0x1e, 0xd2,
];
/// Kimchi's `combined_inner_product(13, 17, values)` for three polynomials
/// at two points, `values[k][j] = 10 k + j + 1`.
const COMBINED_INNER_PRODUCT: u64 = 0x0001_0fd1;

fn chals<F: Field>() -> Vec<F> {
    [2u64, 3, 5, 7].map(F::from_u64).to_vec()
}

fn values<F: Field>() -> Vec<Vec<F>> {
    (0..3u64)
        .map(|k| (0..2u64).map(|j| F::from_u64(10 * k + j + 1)).collect())
        .collect()
}

fn agrees<F: Field>() {
    let x = F::from_u64(11);
    assert_eq!(b_poly(&chals::<F>(), x), F::from_u64(B_POLY));
    let expected: Vec<F> = B_POLY_COEFFICIENTS.map(F::from_u64).to_vec();
    assert_eq!(b_poly_coefficients(&chals::<F>()), expected);
    let polyscale = F::from_u64(13);
    let evalscale = F::from_u64(17);
    assert_eq!(
        combined_inner_product(polyscale, evalscale, &values::<F>()),
        F::from_u64(COMBINED_INNER_PRODUCT)
    );
}

#[test]
fn kimchis_challenge_polynomial_over_fp() {
    agrees::<Fp>();
}

#[test]
fn kimchis_challenge_polynomial_over_fq() {
    agrees::<Fq>();
}
