//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_field::{
    Field, FieldModel, Goldilocks as F, Goldilocks, Native, PrimeField, Sqrt, TwoAdicField, bigint,
    extension::{Quadratic, QuadraticConfig},
    goldilocks::MODULUS,
    inner_product, laws,
    montgomery::{MontField, MontgomeryConfig},
    powers,
    rng::{Arkworks, RandomSource, Sampler, SplitMix64},
};

#[test]
fn bigint_overview() {
    let a = [u64::MAX, 0];
    let (sum, carry) = bigint::add(&a, &[1, 0]);
    assert_eq!((sum, carry), ([0, 1], false));
}

#[test]
fn bigint_add() {
    assert_eq!(bigint::add(&[u64::MAX], &[1]), ([0], true));
}

#[test]
fn bigint_sub() {
    assert_eq!(bigint::sub(&[0, 1], &[1, 0]), ([u64::MAX, 0], false));
    assert_eq!(bigint::sub(&[0], &[1]), ([u64::MAX], true));
}

#[test]
fn bigint_geq() {
    assert!(bigint::geq(&[0, 1], &[u64::MAX, 0]));
    assert!(!bigint::geq(&[1, 0], &[0, 1]));
}

#[test]
fn bigint_is_zero() {
    assert!(bigint::is_zero(&[0, 0]));
}

#[test]
fn bigint_shr1() {
    assert_eq!(bigint::shr1(&[0, 1]), [1 << 63, 0]);
}

#[test]
fn bigint_bit() {
    assert!(bigint::bit(&[0, 2], 65));
    assert!(!bigint::bit(&[0, 2], 64));
}

#[test]
fn bigint_num_bits() {
    assert_eq!(bigint::num_bits(&[0, 1]), 65);
    assert_eq!(bigint::num_bits(&[0, 0]), 0);
}

#[test]
fn bigint_add_mod() {
    assert_eq!(bigint::add_mod(&[5], &[6], &[7]), [4]);
}

#[test]
fn bigint_pow2_mod() {
    // 2^64 mod 7 = 2.
    assert_eq!(bigint::pow2_mod(&[7]), [2]);
}

#[test]
fn bigint_shift_mod() {
    // 3 * 2^64 mod 7 = 6.
    assert_eq!(bigint::shift_mod(&[3], &[7]), [6]);
}

#[test]
fn bigint_neg_inv_mod_2_64() {
    let m0 = 0x992d30ed00000001u64;
    assert_eq!(m0.wrapping_mul(bigint::neg_inv_mod_2_64(m0)), u64::MAX);
}

#[test]
fn bigint_two_adicity() {
    assert_eq!(bigint::two_adicity(&[0x41]), 6);
}

#[test]
fn bigint_one() {
    assert_eq!(bigint::one::<2>(), [1, 0]);
}

#[test]
fn extension_overview() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F7;
    impl MontgomeryConfig<1> for F7 {
        const MODULUS: [u64; 1] = [7];
        const GENERATOR: u64 = 3;
    }
    /// F_49 = F_7[u] / (u^2 + 1), since -1 is not a square modulo 7.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F49;
    impl QuadraticConfig for F49 {
        type Base = MontField<F7, 1>;
        fn non_residue() -> Self::Base {
            -MontField::ONE
        }
    }
    type E = Quadratic<F49>;
    let u = E::new(MontField::ZERO, MontField::ONE);
    assert_eq!(u * u, -E::ONE);
    assert_eq!(u * u.inverse().unwrap(), E::ONE);
}

#[test]
fn extension_quadratic_config() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        // 7 is not a square modulo the Goldilocks prime.
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    assert_eq!(G2::non_residue(), Goldilocks::new(7));
}

#[test]
fn extension_non_residue() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    let _ = G2::non_residue();
}

#[test]
fn extension_quadratic() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    let a = Quadratic::<G2>::new(Goldilocks::new(1), Goldilocks::new(2));
    assert_eq!(a.c1(), Goldilocks::new(2));
}

#[test]
fn extension_quadratic_new() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    let one = Quadratic::<G2>::new(Goldilocks::ONE, Goldilocks::ZERO);
    assert_eq!(one, Quadratic::ONE);
}

#[test]
fn extension_quadratic_c0() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    assert_eq!(Quadratic::<G2>::ONE.c0(), Goldilocks::ONE);
}

#[test]
fn extension_quadratic_c1() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    assert_eq!(Quadratic::<G2>::ONE.c1(), Goldilocks::ZERO);
}

#[test]
fn extension_quadratic_conjugate() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    let a = Quadratic::<G2>::new(Goldilocks::new(3), Goldilocks::new(5));
    // a times its conjugate is the norm, in the base field.
    assert_eq!((a * a.conjugate()).c1(), Goldilocks::ZERO);
}

#[test]
fn extension_quadratic_scale() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct G2;
    impl QuadraticConfig for G2 {
        type Base = Goldilocks;
        fn non_residue() -> Goldilocks {
            Goldilocks::new(7)
        }
    }
    let a = Quadratic::<G2>::new(Goldilocks::new(1), Goldilocks::new(2));
    assert_eq!(a.scale(Goldilocks::new(3)), a + a + a);
}

#[test]
fn goldilocks_overview() {
    // Since p = 2^64 - 2^32 + 1, we have 2^64 = 2^32 - 1 in the field.
    let two = Goldilocks::new(2);
    assert_eq!(two.pow(64), Goldilocks::new((1u64 << 32) - 1));
    assert_eq!((-Goldilocks::ONE).value(), MODULUS - 1);
}

#[test]
fn goldilocks_modulus() {
    assert_eq!(u128::from(MODULUS), (1u128 << 64) - (1u128 << 32) + 1);
}

#[test]
fn goldilocks_goldilocks() {
    assert_eq!(Goldilocks::new(MODULUS + 3), Goldilocks::new(3));
}

#[test]
fn goldilocks_goldilocks_new() {
    assert_eq!(Goldilocks::new(MODULUS), Goldilocks::ZERO);
}

#[test]
fn goldilocks_goldilocks_value() {
    assert_eq!(Goldilocks::new(u64::MAX).value(), u64::MAX - MODULUS);
}

#[test]
fn laws_overview() {
    let (a, b, c) = (Goldilocks::new(2), Goldilocks::new(3), Goldilocks::new(4));
    assert!(laws::additive_group(a, b, c) && laws::distributive(a, b, c));
}

#[test]
fn laws_zero_ne_one() {
    assert!(laws::zero_ne_one::<Goldilocks>());
}

#[test]
fn laws_item() {
    assert!(laws::additive_group(
        Goldilocks::new(1),
        Goldilocks::new(2),
        Goldilocks::new(3)
    ));
}

#[test]
fn laws_multiplicative_monoid() {
    assert!(laws::multiplicative_monoid(
        Goldilocks::new(1),
        Goldilocks::new(2),
        Goldilocks::new(3)
    ));
}

#[test]
fn laws_distributive() {
    assert!(laws::distributive(
        Goldilocks::new(4),
        Goldilocks::new(5),
        Goldilocks::new(6)
    ));
}

#[test]
fn laws_inverse() {
    assert!(laws::inverse(Goldilocks::new(9)));
    assert!(laws::inverse(Goldilocks::new(0)));
}

#[test]
fn crate_overview() {
    let a = Goldilocks::new(3);
    let b = Goldilocks::new(5);
    assert_eq!(a * b + Goldilocks::ONE, Goldilocks::new(16));
    assert_eq!(a * a.inverse().unwrap(), Goldilocks::ONE);
}

#[test]
fn field() {
    fn horner<F>(coefficients: &[F], x: F) -> F
    where
        F: Field,
    {
        coefficients
            .iter()
            .rev()
            .fold(F::ZERO, |acc, c| acc * x + *c)
    }

    // 1 + 2x + 3x^2 at x = 2 is 17.
    let p = [1, 2, 3].map(Goldilocks::new);
    assert_eq!(horner(&p, Goldilocks::new(2)), Goldilocks::new(17));
}

#[test]
fn zero() {
    let x = Goldilocks::new(7);
    assert_eq!(x + Goldilocks::ZERO, x);
}

#[test]
fn one() {
    let x = Goldilocks::new(7);
    assert_eq!(x * Goldilocks::ONE, x);
}

#[test]
fn inverse() {
    let x = Goldilocks::new(7);
    assert_eq!(x * x.inverse().unwrap(), Goldilocks::ONE);
    assert_eq!(Goldilocks::ZERO.inverse(), None);
}

#[test]
fn from_u64() {
    assert_eq!(
        Goldilocks::from_u64(2) + Goldilocks::from_u64(3),
        Goldilocks::from_u64(5)
    );
}

#[test]
fn is_zero() {
    assert!(Goldilocks::ZERO.is_zero());
    assert!(!Goldilocks::ONE.is_zero());
}

#[test]
fn square() {
    assert_eq!(Goldilocks::new(9).square(), Goldilocks::new(81));
}

#[test]
fn pow() {
    assert_eq!(Goldilocks::new(2).pow(10), Goldilocks::new(1024));
    assert_eq!(Goldilocks::new(5).pow(0), Goldilocks::ONE);
}

#[test]
fn prime_field() {
    let x = Goldilocks::new(42);
    assert_eq!(x.limb(0), 42);
    assert_eq!(x.limb(1), 0);
    assert_eq!(Goldilocks::LIMBS, 1);
}

#[test]
fn limbs() {
    assert_eq!(Goldilocks::LIMBS, 1);
}

#[test]
fn modulus_bits() {
    assert_eq!(Goldilocks::MODULUS_BITS, 64);
}

#[test]
fn limb() {
    assert_eq!(
        Goldilocks::new(u64::MAX).limb(0),
        u64::MAX - arrabbiata_field::goldilocks::MODULUS
    );
}

#[test]
fn from_limbs() {
    assert_eq!(Goldilocks::from_limbs(&[5]), Some(Goldilocks::new(5)));
    assert_eq!(Goldilocks::from_limbs(&[MODULUS]), None);
    assert_eq!(Goldilocks::from_limbs(&[1, 1]), None);
}

#[test]
fn two_adic_field() {
    let w = Goldilocks::two_adic_generator(3).unwrap(); // order 8
    assert_eq!(w.pow(8), Goldilocks::ONE);
    assert_eq!(w.pow(4), -Goldilocks::ONE);
}

#[test]
fn two_adicity() {
    assert_eq!(Goldilocks::TWO_ADICITY, 32);
}

#[test]
fn multiplicative_generator() {
    assert_ne!(Goldilocks::MULTIPLICATIVE_GENERATOR, Goldilocks::ONE);
}

#[test]
fn two_adic_generator() {
    let w2 = Goldilocks::two_adic_generator(2).unwrap();
    let w3 = Goldilocks::two_adic_generator(3).unwrap();
    assert_eq!(w3.square(), w2);
    assert!(Goldilocks::two_adic_generator(33).is_none());
}

#[test]
fn sqrt() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    type F = MontField<F13, 1>;
    let r = F::from_u64(10).sqrt().unwrap(); // 6^2 = 36 = 10 mod 13
    assert_eq!(r * r, F::from_u64(10));
    assert_eq!(F::from_u64(2).sqrt(), None);
}

#[test]
fn sqrt_2() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    assert_eq!(MontField::<F13, 1>::ZERO.sqrt(), Some(MontField::ZERO));
}

#[test]
fn montgomery_overview() {
    /// The field of order 2^61 - 1 (a Mersenne prime), on one limb.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct M61;
    impl MontgomeryConfig<1> for M61 {
        const MODULUS: [u64; 1] = [(1 << 61) - 1];
        const GENERATOR: u64 = 37;
    }
    type F = MontField<M61, 1>;

    let a = F::from_u64(1 << 40);
    assert_eq!(a * a, F::from_u64(1 << 19)); // 2^80 = 2^19 mod 2^61 - 1
    assert_eq!(a * a.inverse().unwrap(), F::ONE);
}

#[test]
fn montgomery_montgomery_config() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F7;
    impl MontgomeryConfig<1> for F7 {
        const MODULUS: [u64; 1] = [7];
        const GENERATOR: u64 = 3;
    }
    assert_eq!(F7::MODULUS, [7]);
}

#[test]
fn montgomery_mont_field() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    let x = MontField::<F13, 1>::from_u64(20);
    assert_eq!(x.to_canonical_limbs(), [7]);
}

#[test]
fn montgomery_mont_field_from_canonical_limbs() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    type F = MontField<F13, 1>;
    assert_eq!(F::from_canonical_limbs([5]), Some(F::from_u64(5)));
    assert_eq!(F::from_canonical_limbs([13]), None);
}

#[test]
fn montgomery_mont_field_to_canonical_limbs() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    assert_eq!((-MontField::<F13, 1>::ONE).to_canonical_limbs(), [12]);
}

#[test]
fn montgomery_mont_field_from_u64_const() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    const FIVE: MontField<F13, 1> = MontField::from_u64_const(18);
    assert_eq!(FIVE, MontField::from_u64(5));
}

#[test]
fn montgomery_mont_field_pow_limbs() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    let two = MontField::<F13, 1>::from_u64(2);
    // Fermat: 2^12 = 1 mod 13.
    assert_eq!(two.pow_limbs(&[12]), MontField::ONE);
}

#[test]
fn montgomery_mont_field_legendre() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F13;
    impl MontgomeryConfig<1> for F13 {
        const MODULUS: [u64; 1] = [13];
        const GENERATOR: u64 = 2;
    }
    type F = MontField<F13, 1>;
    assert_eq!(F::from_u64(4).legendre(), F::ONE);
    assert_eq!(F::from_u64(2).legendre(), -F::ONE);
}

#[test]
fn vector_overview() {
    // <(1, z, z^2), (a0, a1, a2)> is the polynomial a0 + a1 X + a2 X^2 at z.
    let a = [F::new(1), F::new(2), F::new(3)];
    assert_eq!(inner_product(&a, &powers(F::new(2), 3)), F::new(17));
}

#[test]
fn vector_powers() {
    assert_eq!(
        powers(F::new(3), 4),
        vec![F::ONE, F::new(3), F::new(9), F::new(27)]
    );
    assert!(powers(F::new(3), 0).is_empty());
}

#[test]
fn vector_inner_product() {
    let a = [F::new(1), F::new(2)];
    let b = [F::new(3), F::new(4), F::new(5)];
    assert_eq!(inner_product(&a, &b), F::new(11));
}

/// The zero stream.
struct Zeros;

impl RandomSource for Zeros {
    fn next_u64(&mut self) -> u64 {
        0
    }
}

#[test]
fn rng_random_source() {
    assert_eq!(Zeros.next_u64(), 0);
}

#[test]
fn rng_sampler() {
    // A sampler is a stream of field elements.
    struct Ones;
    impl Sampler<F> for Ones {
        fn sample(&mut self) -> F {
            F::ONE
        }
    }
    assert_eq!(Ones.sample(), F::ONE);
}

#[test]
fn rng_split_mix_64() {
    // The first output of SplitMix64(0) (Steele, Lea, Flood).
    assert_eq!(SplitMix64(0).next_u64(), 0xe220_a839_7b1d_cdaf);
}

#[test]
fn rng_arkworks() {
    // A zero stream is the zero element; two samples differ.
    let zero: F = Arkworks(Zeros).sample();
    assert_eq!(zero, F::ZERO);
    let mut rng = Arkworks(SplitMix64(1));
    let a: F = rng.sample();
    let b: F = rng.sample();
    assert_ne!(a, b);
}

#[test]
fn codec_byte_len() {
    use arrabbiata_field::codec::byte_len;
    assert_eq!(byte_len(255), 32);
    assert_eq!(byte_len(257), 33);
    assert_eq!(byte_len(64), 8);
}

#[test]
fn codec_to_and_from_le_bytes() {
    use arrabbiata_field::codec::{from_le_bytes, to_le_bytes};
    let x = Goldilocks::from_u64(0x0102);
    assert_eq!(to_le_bytes(x, 3), [2, 1, 0]);
    assert_eq!(from_le_bytes::<Goldilocks>(&[2, 1, 0]), Some(x));
    // From the modulus up, no element.
    assert_eq!(from_le_bytes::<Goldilocks>(&MODULUS.to_le_bytes()), None);
    assert_eq!(
        from_le_bytes::<Goldilocks>(&[0; 8].iter().chain(&[1]).copied().collect::<Vec<u8>>()),
        None
    );
}

/// A model that counts its products and defines only the signature's
/// required operations, so the provided ones are its defaults.
#[derive(Default)]
struct Counting {
    products: usize,
}

impl FieldModel<F> for Counting {
    type Value = F;

    fn constant(&mut self, c: F) -> F {
        c
    }

    fn add(&mut self, a: &F, b: &F) -> F {
        *a + *b
    }

    fn mul(&mut self, a: &F, b: &F) -> F {
        self.products += 1;
        *a * *b
    }

    fn neg(&mut self, a: &F) -> F {
        -*a
    }

    fn inverse(&mut self, a: &F) -> F {
        a.inverse().unwrap_or(F::ZERO)
    }
}

#[test]
fn field_model() {
    // The provided operations, from the required ones: the same values as
    // the field's operators, pow with one squaring per bit.
    let (a, b) = (F::from_u64(7), F::from_u64(3));
    let mut m = Counting::default();
    assert_eq!(m.sub(&a, &b), F::from_u64(4));
    assert_eq!(m.scale(&a, b), F::from_u64(21));
    assert_eq!(m.shift(&a, b), F::from_u64(10));
    assert_eq!(m.square(&b), F::from_u64(9));
    let before = m.products;
    assert_eq!(m.pow(&a, 13), a.pow(13));
    // 13 = 0b1101: four squarings, three multiplications.
    assert_eq!(m.products - before, 7);
    assert_eq!(m.pow(&a, 0), F::ONE);
}

#[test]
fn native() {
    // The field itself: each operation its operator, and zero's inverse
    // zero.
    let (a, b) = (F::from_u64(7), F::from_u64(3));
    let mut m = Native;
    assert_eq!(FieldModel::<F>::add(&mut m, &a, &b), a + b);
    assert_eq!(FieldModel::<F>::mul(&mut m, &a, &b), a * b);
    assert_eq!(FieldModel::<F>::neg(&mut m, &a), -a);
    assert_eq!(FieldModel::<F>::constant(&mut m, a), a);
    assert_eq!(FieldModel::<F>::inverse(&mut m, &a) * a, F::ONE);
    assert_eq!(FieldModel::<F>::inverse(&mut m, &F::ZERO), F::ZERO);
    assert_eq!(FieldModel::<F>::pow(&mut m, &a, 5), a.pow(5));
    assert_eq!(FieldModel::<F>::square(&mut m, &a), a * a);
}
