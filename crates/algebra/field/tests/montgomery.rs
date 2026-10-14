//! Differential tests of the Montgomery fields against an independent
//! reference: schoolbook multiplication into 2N limbs and reduction by
//! binary long division. Nothing in the reference uses Montgomery form.

use arrabbiata_field::{Field, MontField, MontgomeryConfig, PrimeField, Sqrt, TwoAdicField, laws};
use proptest::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pallas;
impl MontgomeryConfig<4> for Pallas {
    const MODULUS: [u64; 4] = [
        0x992d30ed00000001,
        0x224698fc094cf91b,
        0,
        0x4000000000000000,
    ];
    const GENERATOR: u64 = 5;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Bn254Q;
impl MontgomeryConfig<4> for Bn254Q {
    const MODULUS: [u64; 4] = [
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ];
    const GENERATOR: u64 = 3;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct M127;
impl MontgomeryConfig<2> for M127 {
    const MODULUS: [u64; 2] = [u64::MAX, 0x7fffffffffffffff];
    const GENERATOR: u64 = 3;
}

/// `a mod m` for arbitrary-length `a`, by binary long division.
fn reduce(a: &[u64], m: &[u64]) -> Vec<u64> {
    let n = m.len();
    let mut r = vec![0u64; n + 1];
    for i in (0..a.len() * 64).rev() {
        let mut carry = (a[i / 64] >> (i % 64)) & 1;
        for limb in r.iter_mut() {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        let mut ge = true;
        for j in (0..=n).rev() {
            let mj = if j < n { m[j] } else { 0 };
            if r[j] != mj {
                ge = r[j] > mj;
                break;
            }
        }
        if ge {
            let mut borrow = 0u64;
            for j in 0..=n {
                let mj = if j < n { m[j] } else { 0 };
                let (d1, b1) = r[j].overflowing_sub(mj);
                let (d2, b2) = d1.overflowing_sub(borrow);
                r[j] = d2;
                borrow = u64::from(b1 | b2);
            }
        }
    }
    r.truncate(n);
    r
}

fn mul_ref(a: &[u64], b: &[u64], m: &[u64]) -> Vec<u64> {
    let mut product = vec![0u64; a.len() + b.len()];
    for i in 0..a.len() {
        let mut carry = 0u128;
        for j in 0..b.len() {
            let t = product[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry;
            product[i + j] = t as u64;
            carry = t >> 64;
        }
        product[i + b.len()] = carry as u64;
    }
    reduce(&product, m)
}

fn add_ref(a: &[u64], b: &[u64], m: &[u64]) -> Vec<u64> {
    let mut sum = vec![0u64; a.len() + 1];
    let mut carry = 0u128;
    for i in 0..a.len() {
        let t = a[i] as u128 + b[i] as u128 + carry;
        sum[i] = t as u64;
        carry = t >> 64;
    }
    sum[a.len()] = carry as u64;
    reduce(&sum, m)
}

macro_rules! differential {
    ($name:ident, $config:ty, $n:literal) => {
        mod $name {
            use super::*;

            type F = MontField<$config, $n>;
            const P: [u64; $n] = <$config as MontgomeryConfig<$n>>::MODULUS;

            fn arb_limbs() -> impl Strategy<Value = [u64; $n]> {
                prop::array::uniform::<_, $n>(any::<u64>())
                    .prop_map(|l| <[u64; $n]>::try_from(reduce(&l, &P)).unwrap())
            }

            fn arb() -> impl Strategy<Value = F> {
                prop_oneof![
                    Just(F::ZERO),
                    Just(F::ONE),
                    Just(-F::ONE),
                    arb_limbs().prop_map(|l| F::from_canonical_limbs(l).unwrap()),
                ]
            }

            proptest! {
                #[test]
                fn round_trip(l in arb_limbs()) {
                    prop_assert_eq!(F::from_canonical_limbs(l).unwrap().to_canonical_limbs(), l);
                }

                #[test]
                fn mul_matches_reference(a in arb_limbs(), b in arb_limbs()) {
                    let fa = F::from_canonical_limbs(a).unwrap();
                    let fb = F::from_canonical_limbs(b).unwrap();
                    prop_assert_eq!((fa * fb).to_canonical_limbs().to_vec(), mul_ref(&a, &b, &P));
                }

                #[test]
                fn add_matches_reference(a in arb_limbs(), b in arb_limbs()) {
                    let fa = F::from_canonical_limbs(a).unwrap();
                    let fb = F::from_canonical_limbs(b).unwrap();
                    prop_assert_eq!((fa + fb).to_canonical_limbs().to_vec(), add_ref(&a, &b, &P));
                }

                #[test]
                fn axioms(a in arb(), b in arb(), c in arb()) {
                    prop_assert!(laws::additive_group(a, b, c));
                    prop_assert!(laws::multiplicative_monoid(a, b, c));
                    prop_assert!(laws::distributive(a, b, c));
                    prop_assert!(laws::inverse(a));
                }

                #[test]
                fn sqrt_of_square(a in arb()) {
                    let r = (a * a).sqrt().unwrap();
                    prop_assert!(r == a || r == -a);
                }

                #[test]
                fn from_u64_matches_reference(n in any::<u64>()) {
                    let mut l = [0u64; $n];
                    l[0] = n;
                    prop_assert_eq!(F::from_u64(n).to_canonical_limbs().to_vec(), reduce(&l, &P));
                }
            }

            #[test]
            fn generator_is_a_non_residue() {
                assert_eq!(F::MULTIPLICATIVE_GENERATOR.legendre(), -F::ONE);
                assert_eq!(F::MULTIPLICATIVE_GENERATOR.sqrt(), None);
            }

            #[test]
            fn two_adic_generators_have_exact_order() {
                for bits in 0..=F::TWO_ADICITY {
                    let w = F::two_adic_generator(bits).unwrap();
                    let mut x = w;
                    for _ in 0..bits {
                        x = x.square();
                    }
                    assert_eq!(x, F::ONE, "w^(2^{bits}) = 1");
                    if bits > 0 {
                        let mut y = w;
                        for _ in 0..bits - 1 {
                            y = y.square();
                        }
                        assert_eq!(y, -F::ONE, "w^(2^({bits} - 1)) = -1");
                    }
                }
                assert!(F::two_adic_generator(F::TWO_ADICITY + 1).is_none());
            }

            #[test]
            fn modulus_bits() {
                assert_eq!(F::LIMBS, $n);
                assert!(F::MODULUS_BITS > 64 * ($n - 1));
            }
        }
    };
}

differential!(pallas, Pallas, 4);
differential!(bn254_q, Bn254Q, 4);
differential!(m127, M127, 2);

#[test]
fn known_two_adicities() {
    assert_eq!(MontField::<Pallas, 4>::TWO_ADICITY, 32);
    assert_eq!(MontField::<Bn254Q, 4>::TWO_ADICITY, 1);
    assert_eq!(MontField::<M127, 2>::TWO_ADICITY, 1);
}
