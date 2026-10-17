//! Kimchi's IPA helpers on fixed inputs, over both Pasta scalar fields:
//! `b_poly`, `b_poly_coefficients` and `combined_inner_product`. Each value
//! is printed as its four canonical little-endian limbs.
use ark_ff::{BigInteger, PrimeField};
use mina_curves::pasta::{Fp, Fq};
use poly_commitment::commitment::{b_poly, b_poly_coefficients, combined_inner_product};

fn limbs<F: PrimeField>(x: F) -> String {
    let bytes = x.into_bigint().to_bytes_le();
    let words: Vec<String> = bytes
        .chunks(8)
        .map(|c| format!("0x{:016x}", u64::from_le_bytes(c.try_into().unwrap())))
        .collect();
    format!("[{}]", words.join(", "))
}

fn run<F: PrimeField>(name: &str) {
    let chals: Vec<F> = [2u64, 3, 5, 7].iter().map(|c| F::from(*c)).collect();
    let x = F::from(11u64);
    println!("{name} b_poly(chals=[2,3,5,7], x=11) = {}", limbs(b_poly(&chals, x)));
    let s = b_poly_coefficients(&chals);
    for (i, c) in s.iter().enumerate() {
        println!("{name} b_poly_coefficients[{i}] = {}", limbs(*c));
    }
    // Three polynomials, two points: values[k][j] = 10 k + j + 1.
    let polys: Vec<Vec<Vec<F>>> = (0..3u64)
        .map(|k| (0..2u64).map(|j| vec![F::from(10 * k + j + 1)]).collect())
        .collect();
    let polyscale = F::from(13u64);
    let evalscale = F::from(17u64);
    println!(
        "{name} combined_inner_product(polyscale=13, evalscale=17) = {}",
        limbs(combined_inner_product(&polyscale, &evalscale, &polys))
    );
}

fn main() {
    run::<Fp>("Fp");
    run::<Fq>("Fq");
}
