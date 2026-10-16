//! Kimchi's ForeignFieldMul gate: `left right = quotient m + remainder` for
//! a foreign modulus `m`, on three 88-bit limbs (Kimchi's
//! `foreign_field_mul/circuitgates.rs`, the RFC "ffmul-revised").
//!
//! The product is checked twice: modulo the native field (`a_n b_n + q_n
//! f'_n - q_n 2^264 = r_n`, with `f' = 2^264 - m`), and modulo `2^264`
//! limb by limb through the intermediate products `p0, p1, p2` and the
//! carries `v0` (2 bits) and `v1` (91 bits). Together they give the
//! equation over the integers, by the Chinese remainder theorem, as long as
//! `m^2 < 2^264 n` for the native modulus `n`.
//!
//! The rows, as Kimchi lays them out:
//!
//! | col | row 0 (ForeignFieldMul) | row 1 (Zero)       |
//! | --- | ----------------------- | ------------------ |
//! | 0-2 | `a0 a1 a2` (copy)       | `r01 r2 q0`        |
//! | 3-5 | `b0 b1 b2` (copy)       | `q1 q2 q'2`        |
//! | 6   | `p10`                   | `p110`             |
//! | 7   | `v1` bits 0-12          | `p111`             |
//! | 8   | `v1` bits 12-24         | `v1` bits 48-60    |
//! | 9   | `v1` bits 24-36         | `v1` bits 60-72    |
//! | 10  | `v1` bits 36-48         | `v1` bits 72-84    |
//! | 11  | `v1` bits 84-86         | `v0`               |
//! | 12  | `v1` bits 86-88         |                    |
//! | 13  | `v1` bits 88-90         |                    |
//! | 14  | `v1` bit 90             |                    |
//!
//! Columns 7-10 of both rows are looked up in the 12-bit table.
//! Coefficient 0 is `m2`, coefficients 1-3 the limbs of `f'`.
//!
//! [`ForeignFieldMul`] is one multiplication, sound on its own: the gate
//! and its Zero row, then the checks Kimchi leaves to the circuit designer
//! (`foreign_field_mul::witness::ExternalChecks`): the range checks of the
//! quotient, of `(q'2, p10, p110)`, and of the remainder in compact form,
//! the high bound `r2 + 2^88 - m2 - 1` of the remainder (a Generic `Plus`
//! row) and its range check. The inputs are the caller's to range-check.

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, TypedGadget};

use crate::{
    foreign_field::{LIMB_BITS, Limbs, Nat, field_of, two_pow},
    generic::{GenericRow, GenericSpec},
    range_check::{CompactMultiRangeCheck, MultiRangeCheck, crumb},
    zero::Zero,
};

/// The witness of one multiplication, as integers.
struct Witness {
    quotient: [u128; 3],
    remainder: [u128; 3],
    product1_lo: u128,
    product1_hi_0: u128,
    product1_hi_1: u128,
    carry0: u128,
    carry1: u128,
    quotient_hi_bound: u128,
}

/// `2^264 - m`, the negated modulus, as limbs.
fn negated(modulus: &Limbs) -> [u128; 3] {
    let m = Nat::from_limbs(&modulus[..3]);
    Nat::from_u128(1).shl(3 * LIMB_BITS).sub(&m).limbs()
}

/// The witness of `left right` modulo `modulus` (Kimchi's
/// `foreign_field_mul::witness::create`).
fn witness_of(left: [u128; 3], right: [u128; 3], modulus: &Limbs) -> Witness {
    let (a, b) = (Nat::from_limbs(&left), Nat::from_limbs(&right));
    let m = Nat::from_limbs(&modulus[..3]);
    let (q, r) = a.mul(&b).divrem(&m);
    let (q, r) = (q.limbs(), r.limbs());
    let f = negated(modulus);
    let n = Nat::from_u128;
    let product = |x: u128, y: u128| n(x).mul(&n(y));
    let p0 = product(left[0], right[0]).add(&product(q[0], f[0]));
    let p1 = product(left[0], right[1])
        .add(&product(left[1], right[0]))
        .add(&product(q[0], f[1]))
        .add(&product(q[1], f[0]));
    let p2 = product(left[0], right[2])
        .add(&product(left[2], right[0]))
        .add(&product(left[1], right[1]))
        .add(&product(q[0], f[2]))
        .add(&product(q[2], f[0]))
        .add(&product(q[1], f[1]));
    let word = |x: &Nat| -> u128 {
        u128::from(*x.0.first().unwrap_or(&0)) | (u128::from(*x.0.get(1).unwrap_or(&0)) << 64)
    };
    let p1_lo = p1.low(LIMB_BITS);
    let p1_hi = p1.shr(LIMB_BITS);
    let p1_hi_0 = p1_hi.low(LIMB_BITS);
    let p1_hi_1 = p1_hi.shr(LIMB_BITS);
    let carry0 = p0
        .add(&p1_lo.shl(LIMB_BITS))
        .sub(&n(r[0]))
        .sub(&n(r[1]).shl(LIMB_BITS))
        .shr(2 * LIMB_BITS);
    let carry1 = p2.add(&p1_hi).add(&carry0).sub(&n(r[2])).shr(LIMB_BITS);
    Witness {
        quotient: q,
        remainder: r,
        product1_lo: word(&p1_lo),
        product1_hi_0: word(&p1_hi_0),
        product1_hi_1: word(&p1_hi_1),
        carry0: word(&carry0),
        carry1: word(&carry1),
        quotient_hi_bound: q[2] + (1u128 << LIMB_BITS) - modulus[2] - 1,
    }
}

/// The limbs of field elements below `2^88`, as integers.
fn limbs<F: PrimeField>(x: [F; 3]) -> [u128; 3] {
    x.map(|v| u128::from(v.limb(0)) | (u128::from(v.limb(1)) << 64))
}

/// The bits `start .. end` of `x` below `2^128`, in `F`.
fn bits_of<F: Field>(x: u128, start: u32, end: u32) -> F {
    field_of((x >> start) & ((1u128 << (end - start)) - 1))
}

/// The ForeignFieldMul row and its next: input `(left, right)`, wired by
/// copy to columns 0-2 and 3-5; output `([r01, r2], [q0, q1, q2], [q'2,
/// p10, p110])`, the cells the external checks read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignFieldMulGate<F> {
    /// The foreign modulus.
    pub modulus: Limbs,
    /// The 12-bit table.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for ForeignFieldMulGate<F>
where
    F: PrimeField,
{
    type Input = (Array<3>, Array<3>);
    type Output = (Array<2>, Array<3>, Array<3>);
    const NAME: &'static str = "kimchi-foreign-field-mul";
    const ROWS: usize = 1;

    #[allow(clippy::too_many_lines)]
    fn synthesize<E>(
        &self,
        env: &mut E,
        (left, right): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        let modulus = self.modulus;
        let a: [E::Var; 3] = core::array::from_fn(|i| env.copy(Cell::new(i, 0), &left[i]));
        let b: [E::Var; 3] = core::array::from_fn(|i| env.copy(Cell::new(3 + i, 0), &right[i]));
        let inputs = [&a[0], &a[1], &a[2], &b[0], &b[1], &b[2]];
        let of = |[a0, a1, a2, b0, b1, b2]: [F; 6]| {
            witness_of(limbs([a0, a1, a2]), limbs([b0, b1, b2]), &modulus)
        };
        let cell = |env: &mut E, column: usize, row: usize, value: fn(&Witness) -> F| {
            env.advice(Cell::new(column, row), inputs, move |x| value(&of(x)))
        };
        // Row 0.
        let p1_lo = cell(env, 6, 0, |w| field_of(w.product1_lo));
        let carry1_limbs: [E::Var; 4] = [
            cell(env, 7, 0, |w| bits_of(w.carry1, 0, 12)),
            cell(env, 8, 0, |w| bits_of(w.carry1, 12, 24)),
            cell(env, 9, 0, |w| bits_of(w.carry1, 24, 36)),
            cell(env, 10, 0, |w| bits_of(w.carry1, 36, 48)),
        ];
        let crumb0 = cell(env, 11, 0, |w| bits_of(w.carry1, 84, 86));
        let crumb1 = cell(env, 12, 0, |w| bits_of(w.carry1, 86, 88));
        let crumb2 = cell(env, 13, 0, |w| bits_of(w.carry1, 88, 90));
        let bit = cell(env, 14, 0, |w| bits_of(w.carry1, 90, 91));
        // Row 1.
        let r01 = cell(env, 0, 1, |w| {
            field_of::<F>(w.remainder[0]) + field_of::<F>(w.remainder[1]) * two_pow::<F>(LIMB_BITS)
        });
        let r2 = cell(env, 1, 1, |w| field_of(w.remainder[2]));
        let q = [
            cell(env, 2, 1, |w| field_of(w.quotient[0])),
            cell(env, 3, 1, |w| field_of(w.quotient[1])),
            cell(env, 4, 1, |w| field_of(w.quotient[2])),
        ];
        let q_bound = cell(env, 5, 1, |w| field_of(w.quotient_hi_bound));
        let p1_hi_0 = cell(env, 6, 1, |w| field_of(w.product1_hi_0));
        let p1_hi_1 = cell(env, 7, 1, |w| field_of(w.product1_hi_1));
        let carry1_next: [E::Var; 3] = [
            cell(env, 8, 1, |w| bits_of(w.carry1, 48, 60)),
            cell(env, 9, 1, |w| bits_of(w.carry1, 60, 72)),
            cell(env, 10, 1, |w| bits_of(w.carry1, 72, 84)),
        ];
        let carry0 = cell(env, 11, 1, |w| field_of(w.carry0));
        // Coefficients: m2, then f'.
        let f_limbs = negated(&modulus);
        let m2 = env.fixed(Cell::new(0, 0), field_of(modulus[2]));
        let f: [E::Var; 3] =
            core::array::from_fn(|i| env.fixed(Cell::new(1 + i, 0), field_of(f_limbs[i])));
        let k = |env: &mut E, e: u32| env.constant(two_pow::<F>(e));
        // v1 from its parts (Kimchi's compose_carry).
        let parts = [
            (carry1_limbs[0].clone(), 0),
            (carry1_limbs[1].clone(), 12),
            (carry1_limbs[2].clone(), 24),
            (carry1_limbs[3].clone(), 36),
            (carry1_next[0].clone(), 48),
            (carry1_next[1].clone(), 60),
            (carry1_next[2].clone(), 72),
            (crumb0.clone(), 84),
            (crumb1.clone(), 86),
            (crumb2.clone(), 88),
            (bit.clone(), 90),
        ];
        let mut carry1 = parts[0].0.clone();
        for (part, e) in &parts[1..] {
            let power = k(env, *e);
            carry1 = carry1 + power * part.clone();
        }
        // The intermediate products.
        let p0 = a[0].clone() * b[0].clone() + q[0].clone() * f[0].clone();
        let p1 = a[0].clone() * b[1].clone()
            + a[1].clone() * b[0].clone()
            + q[0].clone() * f[1].clone()
            + q[1].clone() * f[0].clone();
        let p2 = a[0].clone() * b[2].clone()
            + a[2].clone() * b[0].clone()
            + a[1].clone() * b[1].clone()
            + q[0].clone() * f[2].clone()
            + q[2].clone() * f[0].clone()
            + q[1].clone() * f[1].clone();
        // The native values.
        let native = |env: &mut E, x: &[E::Var; 3]| {
            let (l2, l1) = (k(env, 2 * LIMB_BITS), k(env, LIMB_BITS));
            l2 * x[2].clone() + l1 * x[1].clone() + x[0].clone()
        };
        let a_n = native(env, &a);
        let b_n = native(env, &b);
        let q_n = native(env, &q);
        let two_176 = k(env, 2 * LIMB_BITS);
        let r_n = two_176.clone() * r2.clone() + r01.clone();
        let f_n = native(env, &f);
        let two_88 = k(env, LIMB_BITS);
        let one = env.constant(F::ONE);
        let bound = q[2].clone() + two_88.clone() - m2 - one.clone();
        // C1: p111 is a crumb. C2: v0 is a crumb.
        let c1 = crumb(env, &p1_hi_1);
        env.assert_zero(c1);
        let c2 = crumb(env, &carry0);
        env.assert_zero(c2);
        // C3: p1 = 2^88 p11 + p10, p11 = 2^88 p111 + p110.
        let p1_hi = two_88.clone() * p1_hi_1 + p1_hi_0.clone();
        let product1 = two_88.clone() * p1_hi.clone() + p1_lo.clone();
        env.assert_zero(p1 - product1);
        // C4: 2^176 v0 = p0 + 2^88 p10 - r01.
        env.assert_zero(
            two_176 * carry0.clone() - (p0 + two_88.clone() * p1_lo.clone() - r01.clone()),
        );
        // C5: a_n b_n + q_n f'_n - r_n - q_n 2^264 = 0.
        let two_264 = k(env, 3 * LIMB_BITS);
        env.assert_zero(a_n * b_n + q_n.clone() * f_n - r_n - q_n * two_264);
        // C6-C8: crumbs of v1; C9: its top bit.
        for c in [&crumb0, &crumb1, &crumb2] {
            let check = crumb(env, c);
            env.assert_zero(check);
        }
        env.assert_zero(bit.clone() * (bit - one));
        // C10: 2^88 v1 = p2 + p11 + v0 - r2.
        env.assert_zero(two_88 * carry1 - (p2 + p1_hi + carry0 - r2.clone()));
        // C11: q'2 = q2 + 2^88 - m2 - 1.
        env.assert_zero(q_bound.clone() - bound);
        // The 12-bit lookups: columns 7-10 of both rows.
        for c in carry1_limbs
            .iter()
            .chain([&carry1_next[0], &carry1_next[1], &carry1_next[2]])
        {
            env.read(&self.table, core::slice::from_ref(c));
        }
        let dummy = env.cell(Cell::new(7, 1));
        env.read(&self.table, &[dummy]);
        ([r01, r2], q, [q_bound, p1_lo, p1_hi_0])
    }
}

/// One foreign field multiplication, sound on its own, on 19 rows: the gate
/// and its Zero row, the range checks of the quotient (rows 2-5) and of
/// `(q'2, p10, p110)` (rows 6-9), the compact range check of the remainder
/// (rows 10-13), the high bound of the remainder (row 14) and its range
/// check (rows 15-18). Input `(left, right)`, cells of copied columns
/// holding range-checked limbs; output the remainder's limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignFieldMul<F> {
    /// The foreign modulus.
    pub modulus: Limbs,
    /// The 12-bit table.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for ForeignFieldMul<F>
where
    F: PrimeField,
{
    type Input = (Array<3>, Array<3>);
    type Output = Array<3>;
    const NAME: &'static str = "kimchi-ffmul";
    const ROWS: usize = 19;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (left, right): Of<Self::Input, E::Var>,
    ) -> Of<Array<3>, E::Var>
    where
        E: Env<F>,
    {
        let gate = ForeignFieldMulGate {
            modulus: self.modulus,
            table: self.table.clone(),
        };
        let ([r01, r2], q, bounds) = env.call(&gate, (left, right));
        env.call(&Zero, ());
        let check = MultiRangeCheck {
            table: self.table.clone(),
        };
        env.call(&check, q);
        env.call(&check, bounds);
        let compact = CompactMultiRangeCheck {
            table: self.table.clone(),
        };
        let [r0, r1] = env.call(&compact, [r01, r2.clone()]);
        // Row 14: r2 + (2^88 - m2 - 1), then its range check.
        let hi = two_pow::<F>(LIMB_BITS) - field_of::<F>(self.modulus[2]) - F::ONE;
        let left_cell = env.copy(Cell::new(0, 14), &r2);
        let zero = env.advice(Cell::new(1, 14), [], |[]| F::ZERO);
        let r2_bound = env.advice(Cell::new(2, 14), [&left_cell], move |[x]| x + hi);
        let rest: [E::Var; 3] =
            core::array::from_fn(|c| env.advice(Cell::new(3 + c, 14), [], |[]| F::ZERO));
        let [x3, x4, x5] = rest;
        let row = GenericRow::new(GenericSpec::Plus(hi), None);
        env.call(&row, [left_cell, zero, r2_bound.clone(), x3, x4, x5]);
        let pad: [E::Var; 2] =
            core::array::from_fn(|i| env.advice(Cell::new(0, 16 + i), [], |[]| F::ZERO));
        let [p1, p2] = pad;
        env.call(&check, [r2_bound, p1, p2]);
        [r0, r1, r2]
    }
}
