//! Kimchi's ForeignFieldMul (the gate, its Zero row, and the external
//! checks) against Kimchi, for secp256k1's base field, BN254's base field
//! and `2^64 - 59`:
//!
//! - the remainder is `a b mod m`, computed independently by double-and-add
//!   on 64-bit words;
//! - rows 0 and 1 are Kimchi's `foreign_field_mul::witness::create`, cell
//!   for cell (the inputs, `p10`, the carry `v1` in pieces, the compact
//!   remainder, the quotient, `q'2`, `p110`, `p111`, `v0`);
//! - a wrong quotient limb, remainder, or carry piece is rejected;
//! - it composes with ForeignFieldAdd: `a b + c mod m`.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, witness};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    foreign_field::{Limbs, Nat},
    foreign_field_add::{FFOps, ForeignFieldAdd},
    foreign_field_mul::ForeignFieldMul,
    range_check::range_check_table,
};
use arrabbiata_pasta::Fp;
use proptest::prelude::*;

/// The column variables: every cell of a row, those of the next, and the
/// coefficients.
const VARIABLES: usize = 40;
/// A crumb's check.
const DEGREE: usize = 4;

/// secp256k1's base field, as 64-bit words.
const SECP256K1: [u64; 4] = [
    0xffff_fffe_ffff_fc2f,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];
/// BN254's base field, as 64-bit words.
const BN254: [u64; 4] = [
    0x3c20_8c16_d87c_fd47,
    0x9781_6a91_6871_ca8d,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];
/// `2^64 - 59`, a prime.
const SMALL: [u64; 4] = [0xffff_ffff_ffff_ffc5, 0, 0, 0];

fn limbs_of(words: [u64; 4]) -> Limbs {
    let bit = |i: usize| (words[i / 64] >> (i % 64)) & 1;
    let mut limbs = [0u128; 4];
    for i in 0..256 {
        limbs[i / 88] |= u128::from(bit(i)) << (i % 88);
    }
    limbs
}

fn less(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

fn add_words(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut out = [0u64; 4];
    let mut carry = false;
    for i in 0..4 {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(u64::from(carry));
        out[i] = s;
        carry = c1 || c2;
    }
    (out, carry)
}

fn sub_words(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        out[i] = d;
        borrow = b1 || b2;
    }
    out
}

/// `(a + b) mod m` for `a, b < m`.
fn add_mod(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (sum, carry) = add_words(a, b);
    if carry || !less(&sum, m) {
        sub_words(&sum, m)
    } else {
        sum
    }
}

/// `a b mod m` by double-and-add, independently of the gadget.
fn mul_mod(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let mut acc = [0u64; 4];
    for i in (0..256).rev() {
        acc = add_mod(&acc, &acc, m);
        if (b[i / 64] >> (i % 64)) & 1 == 1 {
            acc = add_mod(&acc, a, m);
        }
    }
    acc
}

fn field(x: u128) -> Fp {
    let two_32 = Fp::from_u64(1 << 32);
    Fp::from_u64(x as u64) + Fp::from_u64((x >> 64) as u64) * two_32 * two_32
}

fn field_limbs(words: &[u64; 4]) -> [Fp; 3] {
    let l = limbs_of(*words);
    core::array::from_fn(|i| field(l[i]))
}

/// `a`, `b` (and `c`) in the rows after the gadgets, then `a b` (`+ c`).
#[derive(Clone, Debug)]
struct Mul {
    modulus: [u64; 4],
    add: bool,
}

/// The rows of `ForeignFieldMul`.
const MUL_ROWS: usize = 19;
/// The rows of `ForeignFieldAdd`.
const ADD_ROWS: usize = 13;

impl TypedGadget<Fp> for Mul {
    type Input = Array<Scalar, 9>;
    type Output = Array<Scalar, 3>;
    const NAME: &'static str = "mul";
    const ROWS: usize = MUL_ROWS + ADD_ROWS + 2;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<Scalar, 9>, E::Var>,
    ) -> Of<Array<Scalar, 3>, E::Var> {
        // a and b on one row, c on the next: copies need columns 0-6.
        let row = Self::ROWS - 2;
        let cells: [E::Var; 9] = core::array::from_fn(|c| {
            let at = if c < 6 {
                Cell::new(c, row)
            } else {
                Cell::new(c - 6, row + 1)
            };
            env.advice(at, [&input[c]], |[v]| v)
        });
        let [a0, a1, a2, b0, b1, b2, c0, c1, c2] = cells;
        let modulus = limbs_of(self.modulus);
        let mul = ForeignFieldMul {
            modulus,
            table: range_check_table(),
        };
        let product = env.call(&mul, ([a0, a1, a2], [b0, b1, b2]));
        if self.add {
            let add = ForeignFieldAdd {
                modulus,
                op: FFOps::Add,
                table: range_check_table(),
            };
            env.call(&add, (product, [c0, c1, c2]))
        } else {
            product
        }
    }
}

fn arb_below(m: [u64; 4]) -> impl Strategy<Value = [u64; 4]> {
    any::<[u64; 4]>().prop_map(move |mut w| {
        let top = (0..256)
            .rev()
            .find(|i| (m[i / 64] >> (i % 64)) & 1 == 1)
            .unwrap_or(0);
        for i in top + 1..256 {
            w[i / 64] &= !(1u64 << (i % 64));
        }
        if less(&w, &m) { w } else { sub_words(&w, &m) }
    })
}

fn arb_case() -> impl Strategy<Value = ([u64; 4], [u64; 4], [u64; 4], [u64; 4])> {
    let modulus = prop_oneof![Just(SECP256K1), Just(BN254), Just(SMALL)];
    modulus.prop_flat_map(|m| (arb_below(m), arb_below(m), arb_below(m), Just(m)))
}

fn input(a: &[u64; 4], b: &[u64; 4], c: &[u64; 4]) -> [Fp; 9] {
    let (fa, fb, fc) = (field_limbs(a), field_limbs(b), field_limbs(c));
    [
        fa[0], fa[1], fa[2], fb[0], fb[1], fb[2], fc[0], fc[1], fc[2],
    ]
}

/// The 64-bit words of the low 128 bits of `x`.
fn word(x: &Nat) -> u128 {
    u128::from(*x.0.first().unwrap_or(&0)) | (u128::from(*x.0.get(1).unwrap_or(&0)) << 64)
}

/// Kimchi's rows 0 and 1, from its `create`: products with the negated
/// modulus, `p1` split at 88 and 176 bits, the carries.
fn kimchi_rows(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4]) -> [[Fp; 15]; 2] {
    let (la, lb, lm) = (limbs_of(*a), limbs_of(*b), limbs_of(*m));
    let n = |x: u128| Nat::from_u128(x);
    let (q, r) = Nat::from_limbs(&la[..3])
        .mul(&Nat::from_limbs(&lb[..3]))
        .divrem(&Nat::from_limbs(&lm[..3]));
    let (q, r) = (q.limbs(), r.limbs());
    let f = Nat::from_u128(1)
        .shl(264)
        .sub(&Nat::from_limbs(&lm[..3]))
        .limbs();
    let p = |x: u128, y: u128| n(x).mul(&n(y));
    let p0 = p(la[0], lb[0]).add(&p(q[0], f[0]));
    let p1 = p(la[0], lb[1])
        .add(&p(la[1], lb[0]))
        .add(&p(q[0], f[1]))
        .add(&p(q[1], f[0]));
    let p2 = p(la[0], lb[2])
        .add(&p(la[2], lb[0]))
        .add(&p(la[1], lb[1]))
        .add(&p(q[0], f[2]))
        .add(&p(q[2], f[0]))
        .add(&p(q[1], f[1]));
    let p10 = p1.low(88);
    let p11 = p1.shr(88);
    let (p110, p111) = (p11.low(88), p11.shr(88));
    let v0 = p0
        .add(&p10.shl(88))
        .sub(&n(r[0]))
        .sub(&n(r[1]).shl(88))
        .shr(176);
    let v1 = word(&p2.add(&p11).add(&v0).sub(&n(r[2])).shr(88));
    let piece = |s: u32, e: u32| field((v1 >> s) & ((1u128 << (e - s)) - 1));
    let mut row0 = [Fp::ZERO; 15];
    row0[..3].copy_from_slice(&field_limbs(a));
    row0[3..6].copy_from_slice(&field_limbs(b));
    row0[6] = field(word(&p10));
    for (k, (s, e)) in [
        (0, 12),
        (12, 24),
        (24, 36),
        (36, 48),
        (84, 86),
        (86, 88),
        (88, 90),
        (90, 91),
    ]
    .iter()
    .enumerate()
    {
        row0[7 + k] = piece(*s, *e);
    }
    let mut row1 = [Fp::ZERO; 15];
    row1[0] = field(r[0]) + field(r[1]) * field(1 << 88);
    row1[1] = field(r[2]);
    row1[2] = field(q[0]);
    row1[3] = field(q[1]);
    row1[4] = field(q[2]);
    row1[5] = field(q[2] + (1u128 << 88) - lm[2] - 1);
    row1[6] = field(word(&p110));
    row1[7] = field(word(&p111));
    row1[8] = piece(48, 60);
    row1[9] = piece(60, 72);
    row1[10] = piece(72, 84);
    row1[11] = field(word(&v0));
    [row0, row1]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn kimchis_layout((a, b, _c, m) in arb_case()) {
        let gadget = Mul { modulus: m, add: false };
        let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input(&a, &b, &[0; 4])).unwrap();
        prop_assert_eq!(out, field_limbs(&mul_mod(&a, &b, &m)));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (k, row) in kimchi_rows(&a, &b, &m).iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][k]);
            prop_assert_eq!(&found, row, "row {}", k);
        }
    }

    #[test]
    fn tampering_is_caught((a, b, _c, m) in arb_case(), which in 0usize..4) {
        let gadget = Mul { modulus: m, add: false };
        let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input(&a, &b, &[0; 4])).unwrap();
        // A quotient limb, the compact remainder, the high remainder, a
        // 12-bit piece of v1.
        let (column, row) = [(2, 1), (0, 1), (1, 1), (8, 0)][which];
        let mut columns = compiled.columns.clone();
        columns[column][row] = columns[column][row] + Fp::ONE;
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }

    #[test]
    fn composes_with_addition((a, b, c, m) in arb_case()) {
        let gadget = Mul { modulus: m, add: true };
        let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input(&a, &b, &c)).unwrap();
        let expected = add_mod(&mul_mod(&a, &b, &m), &c, &m);
        prop_assert_eq!(out, field_limbs(&expected));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
    }
}

/// `Nat`'s division is a division: `n = q d + r`, `r < d`.
#[test]
fn nat_division() {
    let n = Nat::from_limbs(&[123, 456, 789]).mul(&Nat::from_limbs(&[1, 2, 3]));
    let d = Nat::from_limbs(&[7, 11, 13]);
    let (q, r) = n.divrem(&d);
    assert_eq!(q.mul(&d).add(&r), n);
    assert_eq!(r.compare(&d), core::cmp::Ordering::Less);
}
