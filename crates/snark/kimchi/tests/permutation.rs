//! Kimchi's permutation shifts against Kimchi: `ConstraintSystem::shift`
//! for domains of 16, 128 and 1024 rows (`vectors/kimchi_shifts.txt`, from
//! the `shifts` tool of the Kimchi figures notes), and their properties.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_kimchi::permutation::{PERMUTS, shifts};
use arrabbiata_pasta::Fp;

/// Kimchi's output.
const VECTORS: &str = include_str!("vectors/kimchi_shifts.txt");

fn element(hex: &str) -> Fp {
    let limbs: Vec<u64> = (0..4)
        .rev()
        .map(|i| u64::from_str_radix(&hex[16 * i..16 * (i + 1)], 16).unwrap())
        .collect();
    Fp::from_limbs(&limbs).unwrap()
}

#[test]
fn the_shifts_are_kimchis() {
    let mut cases = 0;
    for line in VECTORS.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let size: u64 = words[1].parse().unwrap();
        let expected: Vec<Fp> = words[2..].iter().map(|h| element(h)).collect();
        let found = shifts::<Fp>(size.trailing_zeros());
        assert_eq!(found.to_vec(), expected, "{size} rows");
        cases += 1;
    }
    assert_eq!(cases, 3);
}

/// The shifts are distinct, the first is 1, and no two lie in the same
/// coset of the domain: `(k_i / k_j)^n != 1`.
#[test]
fn the_shifts_label_distinct_cosets() {
    let log_size = 10;
    let k = shifts::<Fp>(log_size);
    assert_eq!(k[0], Fp::ONE);
    for i in 0..PERMUTS {
        for j in 0..i {
            let ratio = k[i] * k[j].inverse().unwrap();
            assert_ne!(ratio.pow(1 << log_size), Fp::ONE, "k_{i} and k_{j}");
        }
    }
}
