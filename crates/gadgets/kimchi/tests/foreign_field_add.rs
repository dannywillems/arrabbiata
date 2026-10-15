//! Kimchi's ForeignFieldAdd (the operation, the bound, the Zero row)
//! against Kimchi, for three foreign moduli: secp256k1's base field,
//! BN254's base field, and the 64-bit prime `2^64 - 59`.
//!
//! - the result is `(a + b) mod m`, or `(a - b) mod m`, computed with
//!   `u128` arithmetic independently of the gadget;
//! - rows 2-4 are Kimchi's `foreign_field_add::witness::create_chain` for
//!   one operation, cell for cell: the operation row (`a`, `b`, `q`,
//!   `carry`), the bound row (the result, `0, 0, 2^88`, `1`, the bound
//!   carry), and the bound's limbs;
//! - a wrong limb of the result, a flipped overflow, or a bound overflow
//!   other than 1 is rejected;
//! - the bound alone: for `x < 2^264`, the bound gate and the range check
//!   of its output accept exactly the `x < m` (for `x >= m` the bound's high
//!   limb is `2^88` or more).

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Error, Of, TypedGadget, witness};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    foreign_field::Limbs,
    foreign_field_add::{FFOps, ForeignFieldAdd, ForeignFieldAddGate},
    generic::{GenericRow, GenericSpec},
    range_check::{MultiRangeCheck, range_check_table},
    zero::Zero,
};
use arrabbiata_pasta::Fp;
use proptest::prelude::*;

/// The column variables: every cell of a row, those of the next, and the
/// coefficients.
const VARIABLES: usize = 40;
/// A crumb's check.
const DEGREE: usize = 4;
/// The row of the inputs, after the gadget's 13.
const INPUT_ROW: usize = 13;

/// secp256k1's base field, `2^256 - 2^32 - 977`, as 64-bit words.
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

/// The 88-bit limbs of a number given by 64-bit words.
fn limbs_of(words: [u64; 4]) -> Limbs {
    let bit = |i: usize| (words[i / 64] >> (i % 64)) & 1;
    let mut limbs = [0u128; 4];
    for i in 0..256 {
        limbs[i / 88] |= u128::from(bit(i)) << (i % 88);
    }
    limbs
}

/// `a < b` on words.
fn less(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// `a + b` and `a - b` on words, with the carry or the borrow.
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

fn sub_words(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut out = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        out[i] = d;
        borrow = b1 || b2;
    }
    (out, borrow)
}

/// `(a op b) mod m`, for `a, b < m`, independently of the gadget.
fn expected(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4], op: FFOps) -> [u64; 4] {
    match op {
        FFOps::Add => {
            let (sum, carry) = add_words(a, b);
            if carry || !less(&sum, m) {
                sub_words(&sum, m).0
            } else {
                sum
            }
        }
        FFOps::Sub => {
            let (difference, borrow) = sub_words(a, b);
            if borrow {
                add_words(&difference, m).0
            } else {
                difference
            }
        }
    }
}

fn field(x: u128) -> Fp {
    let two_32 = Fp::from_u64(1 << 32);
    Fp::from_u64(x as u64) + Fp::from_u64((x >> 64) as u64) * two_32 * two_32
}

fn field_limbs(words: &[u64; 4]) -> [Fp; 3] {
    let l = limbs_of(*words);
    core::array::from_fn(|i| field(l[i]))
}

/// `a` and `b` in row 13, then the addition.
#[derive(Clone, Debug)]
struct Add {
    modulus: [u64; 4],
    op: FFOps,
}

impl TypedGadget<Fp> for Add {
    type Input = Array<6>;
    type Output = Array<3>;
    const NAME: &'static str = "add";
    const ROWS: usize = INPUT_ROW + 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Array<6>, E::Var>,
    ) -> Of<Array<3>, E::Var> {
        let cells: [E::Var; 6] =
            core::array::from_fn(|c| env.advice(Cell::new(c, INPUT_ROW), [&input[c]], |[v]| v));
        let [a0, a1, a2, b0, b1, b2] = cells;
        let gadget = ForeignFieldAdd {
            modulus: limbs_of(self.modulus),
            op: self.op,
            table: range_check_table(),
        };
        env.call(&gadget, ([a0, a1, a2], [b0, b1, b2]))
    }
}

fn arb_below(m: [u64; 4]) -> impl Strategy<Value = [u64; 4]> {
    any::<[u64; 4]>().prop_map(move |mut w| {
        // Below the modulus: clear the bits above its top bit, then reduce
        // once if still too large.
        let top = (0..256)
            .rev()
            .find(|i| (m[i / 64] >> (i % 64)) & 1 == 1)
            .unwrap_or(0);
        for i in top + 1..256 {
            w[i / 64] &= !(1u64 << (i % 64));
        }
        if less(&w, &m) { w } else { sub_words(&w, &m).0 }
    })
}

fn arb_case() -> impl Strategy<Value = ([u64; 4], [u64; 4], [u64; 4], FFOps)> {
    let modulus = prop_oneof![Just(SECP256K1), Just(BN254), Just(SMALL)];
    let op = prop_oneof![Just(FFOps::Add), Just(FFOps::Sub)];
    (modulus, op).prop_flat_map(|(m, op)| (arb_below(m), arb_below(m), Just(m), Just(op)))
}

/// Kimchi's rows for one operation: the operation, the bound, the bound's
/// limbs (`create_chain` with one opcode).
fn kimchi_rows(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4], op: FFOps) -> [[Fp; 15]; 3] {
    let r = expected(a, b, m, op);
    let sign = op.sign::<Fp>();
    let overflow = match op {
        FFOps::Add => !less(&add_words(a, b).0, m) || add_words(a, b).1,
        FFOps::Sub => less(a, b),
    };
    let q = if overflow { sign } else { Fp::ZERO };
    let two_88 = field(1 << 88);
    let bot = |x: [Fp; 3]| x[0] + x[1] * two_88;
    let inverse = (two_88 * two_88).inverse().unwrap();
    let (fa, fb, fm, fr) = (
        field_limbs(a),
        field_limbs(b),
        field_limbs(m),
        field_limbs(&r),
    );
    let carry = (bot(fa) + sign * bot(fb) - q * bot(fm) - bot(fr)) * inverse;
    // The bound: r + 2^264 - m, below 2^264 since r < m.
    let mut bound = limbs_of(r);
    bound[3] = 1;
    let m_limbs = limbs_of(*m);
    let mut borrow = 0u128;
    for i in 0..4 {
        let need = m_limbs[i] + borrow;
        if bound[i] >= need {
            bound[i] -= need;
            borrow = 0;
        } else {
            bound[i] = bound[i] + (1u128 << 88) - need;
            borrow = 1;
        }
    }
    let fu: [Fp; 3] = core::array::from_fn(|i| field(bound[i]));
    // The bound row: s = 1, q = 1, right = (0, 0, 2^88), so only m enters
    // the low limbs.
    let bound_carry = (bot(fr) - bot(fm) - bot(fu)) * inverse;
    let mut row0 = [Fp::ZERO; 15];
    row0[..3].copy_from_slice(&fa);
    row0[3..6].copy_from_slice(&fb);
    row0[6] = q;
    row0[7] = carry;
    let mut row1 = [Fp::ZERO; 15];
    row1[..3].copy_from_slice(&fr);
    row1[5] = two_88;
    row1[6] = Fp::ONE;
    row1[7] = bound_carry;
    let mut row2 = [Fp::ZERO; 15];
    row2[..3].copy_from_slice(&fu);
    [row0, row1, row2]
}

fn input(a: &[u64; 4], b: &[u64; 4]) -> [Fp; 6] {
    let (fa, fb) = (field_limbs(a), field_limbs(b));
    [fa[0], fa[1], fa[2], fb[0], fb[1], fb[2]]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn kimchis_layout((a, b, m, op) in arb_case()) {
        let gadget = Add { modulus: m, op };
        let (compiled, out) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input(&a, &b)).unwrap();
        prop_assert_eq!(out, field_limbs(&expected(&a, &b, &m, op)));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        for (k, row) in kimchi_rows(&a, &b, &m, op).iter().enumerate() {
            let found: [Fp; 15] = core::array::from_fn(|c| compiled.columns[c][2 + k]);
            prop_assert_eq!(&found, row, "row {}", 2 + k);
        }
    }

    #[test]
    fn tampering_is_caught((a, b, m, op) in arb_case(), which in 0usize..3) {
        let gadget = Add { modulus: m, op };
        let (compiled, _) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &input(&a, &b)).unwrap();
        // A limb of the result (row 3), the overflow (row 2, column 6), or
        // the bound's overflow (row 3, column 6).
        let (column, row) = [(which, 3), (6, 2), (6, 3)][which];
        let mut columns = compiled.columns.clone();
        columns[column][row] = columns[column][row] + Fp::ONE;
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &compiled.public));
    }
}

/// The bound check alone on `x`, in row 9: constants, the bound gate, the
/// Zero row, the range check of the bound.
#[derive(Clone, Debug)]
struct Bound {
    modulus: [u64; 4],
}

impl TypedGadget<Fp> for Bound {
    type Input = Array<3>;
    type Output = arrabbiata_gadget::Unit;
    const NAME: &'static str = "bound";
    const ROWS: usize = 10;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        x: Of<Array<3>, E::Var>,
    ) -> Of<arrabbiata_gadget::Unit, E::Var> {
        let two_88 = field(1 << 88);
        let cells: [E::Var; 3] =
            core::array::from_fn(|c| env.advice(Cell::new(c, 9), [&x[c]], |[v]| v));
        let constants: [E::Var; 6] = core::array::from_fn(|c| {
            let value = match c {
                0 => Fp::ONE,
                3 => two_88,
                _ => Fp::ZERO,
            };
            env.advice(Cell::new(c, 0), [], move |[]| value)
        });
        let (one, base, zero) = (
            constants[0].clone(),
            constants[3].clone(),
            constants[1].clone(),
        );
        let row = GenericRow::new(
            GenericSpec::Const(Fp::ONE),
            Some(GenericSpec::Const(two_88)),
        );
        env.call(&row, constants);
        let gate = ForeignFieldAddGate {
            modulus: limbs_of(self.modulus),
            op: FFOps::Add,
        };
        let [u0, u1, u2, q] = env.call(&gate, (cells, [zero.clone(), zero, base]));
        env.connect(&one, &q);
        env.call(&Zero, ());
        let check = MultiRangeCheck {
            table: range_check_table(),
        };
        env.call(&check, [u0, u1, u2]);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn the_bound_accepts_exactly_the_reduced(m in prop_oneof![Just(SECP256K1), Just(BN254), Just(SMALL)], x in any::<[u64; 4]>()) {
        let gadget = Bound { modulus: m };
        let found = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &field_limbs(&x));
        if less(&x, &m) {
            prop_assert!(found.is_ok(), "{:?}", found.err());
        } else {
            // The bound gate holds (its high limb is 2^88 or more); the
            // range check of the bound is what fails.
            let failed = matches!(
                found,
                Err(Error::Unsatisfied { gadget: "kimchi-range-check-0" | "kimchi-range-check-1", .. })
            );
            prop_assert!(failed, "{:?}", found.err());
        }
    }
}
