//! Kimchi's ForeignFieldAdd gate: `left +/- right = q m + result` for a
//! foreign modulus `m`, on elements of three 88-bit limbs, with the low two
//! limbs compacted into one of 176 bits (Kimchi's
//! `foreign_field_add/circuitgates.rs`).
//!
//! The row, as Kimchi lays it out (the result on the next row, where the
//! next gate of a chain reads it as its left input):
//!
//! | 0-2         | 3-5          | 6              | 7       |
//! | ----------- | ------------ | -------------- | ------- |
//! | `left` limbs| `right` limbs| `q` (overflow) | `carry` |
//!
//! Coefficients 0-2 are the limbs of `m`, coefficient 3 the sign `s = +1`
//! (addition) or `-1` (subtraction). The constraints: `q (q - s) = 0`,
//! `carry` in `{-1, 0, 1}`, and the two limb equations
//!
//! ```text
//! a_bot + s b_bot - q m_bot - c 2^176 = r_bot     (bot = lo + 2^88 mi)
//! a_hi  + s b_hi  - q m_hi  + c       = r_hi
//! ```
//!
//! [`ForeignFieldAdd`] is one addition, made sound on its own: the gate,
//! then the bound addition `result + 2^264 - m` (the same gate, with right
//! input `(0, 0, 2^88)` and overflow `1`, both copied from constant
//! Generic rows), Kimchi's Zero row holding the bound, and the range checks
//! of the result and the bound. `bound < 2^264` is `result < m`. The inputs
//! are the caller's to range-check, as in Kimchi.

use core::cmp::Ordering;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Rom, TypedGadget};

use crate::{
    foreign_field::{self, LIMB_BITS, Limbs, two_pow},
    generic::{GenericRow, GenericSpec},
    range_check::MultiRangeCheck,
    zero::Zero,
};

/// The operation: addition or subtraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FFOps {
    /// `left + right`.
    Add,
    /// `left - right`.
    Sub,
}

impl FFOps {
    /// `+1` or `-1`, Kimchi's `FFOps::sign`.
    #[must_use]
    pub fn sign<F: Field>(self) -> F {
        match self {
            Self::Add => F::ONE,
            Self::Sub => -F::ONE,
        }
    }
}

/// `lo + 2^88 mi`, Kimchi's `compact_limb`.
fn compact<F, E>(env: &mut E, lo: &E::Var, mi: &E::Var) -> E::Var
where
    F: Field,
    E: Env<F>,
{
    let base = env.constant(two_pow::<F>(LIMB_BITS));
    lo.clone() + mi.clone() * base
}

/// The witness of one addition: the result, the overflow `q` and the
/// carry, from the limbs of `left` and `right` (Kimchi's
/// `compute_ffadd_values`).
fn values<F: PrimeField>(left: [F; 3], right: [F; 3], op: FFOps, modulus: &Limbs) -> (Limbs, F, F) {
    let a = foreign_field::from_field(left);
    let b = foreign_field::from_field(right);
    let (overflow, result) = match op {
        FFOps::Add => {
            let sum = foreign_field::add(&a, &b);
            if foreign_field::compare(&sum, modulus) == Ordering::Less {
                (false, sum)
            } else {
                (true, foreign_field::sub(&sum, modulus))
            }
        }
        FFOps::Sub => {
            if foreign_field::compare(&a, &b) == Ordering::Less {
                (
                    true,
                    foreign_field::sub(&foreign_field::add(modulus, &a), &b),
                )
            } else {
                (false, foreign_field::sub(&a, &b))
            }
        }
    };
    let sign = op.sign::<F>();
    let q = if overflow { sign } else { F::ZERO };
    let m: [F; 3] = foreign_field::to_field(modulus);
    let r: [F; 3] = foreign_field::to_field(&result);
    let base = two_pow::<F>(LIMB_BITS);
    let bot = |x: [F; 3]| x[0] + x[1] * base;
    let carry = (bot(left) + sign * bot(right) - q * bot(m) - bot(r))
        * match two_pow::<F>(2 * LIMB_BITS).inverse() {
            Some(inverse) => inverse,
            None => unreachable!("2^176 is invertible in a field of odd characteristic"),
        };
    (result, q, carry)
}

/// The ForeignFieldAdd row: input `(left, right)`, wired by copy to columns
/// 0-2 and 3-5; output `[r_lo, r_mi, r_hi, q]`, the result on the next row
/// and the overflow in column 6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForeignFieldAddGate {
    /// The foreign modulus.
    pub modulus: Limbs,
    /// The operation.
    pub op: FFOps,
}

impl<F> TypedGadget<F> for ForeignFieldAddGate
where
    F: PrimeField,
{
    type Input = (Array<3>, Array<3>);
    type Output = Array<4>;
    const NAME: &'static str = "kimchi-foreign-field-add";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (left, right): Of<Self::Input, E::Var>,
    ) -> Of<Array<4>, E::Var>
    where
        E: Env<F>,
    {
        let (op, modulus) = (self.op, self.modulus);
        let a: [E::Var; 3] = core::array::from_fn(|i| env.copy(Cell::new(i, 0), &left[i]));
        let b: [E::Var; 3] = core::array::from_fn(|i| env.copy(Cell::new(3 + i, 0), &right[i]));
        let m_limbs: [F; 3] = foreign_field::to_field(&modulus);
        let m: [E::Var; 3] = core::array::from_fn(|i| env.fixed(Cell::new(i, 0), m_limbs[i]));
        let s = env.fixed(Cell::new(3, 0), op.sign::<F>());
        let inputs = [&a[0], &a[1], &a[2], &b[0], &b[1], &b[2]];
        let split =
            |[a0, a1, a2, b0, b1, b2]: [F; 6]| values([a0, a1, a2], [b0, b1, b2], op, &modulus);
        let q = env.advice(Cell::new(6, 0), inputs, |x| split(x).1);
        let carry = env.advice(Cell::new(7, 0), inputs, |x| split(x).2);
        let r: [E::Var; 3] = core::array::from_fn(|i| {
            env.advice(Cell::new(i, 1), inputs, |x| {
                let r: [F; 3] = foreign_field::to_field(&split(x).0);
                r[i]
            })
        });
        // C1: q (q - s) = 0.
        env.assert_zero(q.clone() * (q.clone() - s.clone()));
        // C2: carry in {-1, 0, 1}.
        let one = env.constant(F::ONE);
        env.assert_zero(carry.clone() * (carry.clone() - one.clone()) * (carry.clone() + one));
        // C3 and C4: the limb equations.
        let a_bot = compact(env, &a[0], &a[1]);
        let b_bot = compact(env, &b[0], &b[1]);
        let m_bot = compact(env, &m[0], &m[1]);
        let r_bot = compact(env, &r[0], &r[1]);
        let two_176 = env.constant(two_pow::<F>(2 * LIMB_BITS));
        let bot = a_bot + s.clone() * b_bot - q.clone() * m_bot - carry.clone() * two_176;
        env.assert_zero(bot - r_bot);
        let top = a[2].clone() + s * b[2].clone() - q.clone() * m[2].clone() + carry;
        env.assert_zero(top - r[2].clone());
        let [r0, r1, r2] = r;
        [r0, r1, r2, q]
    }
}

/// One foreign field addition or subtraction, sound on its own, on 13
/// rows: two Generic rows of constants (`1`, `2^88`, `0`), the gate, the
/// bound gate, Kimchi's Zero row with the bound, and the range checks of
/// the result and of the bound. Input `(left, right)`, cells of copied
/// columns holding range-checked limbs; output the result's limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignFieldAdd<F> {
    /// The foreign modulus, below `2^259` (Kimchi's maximum for a native
    /// field of 255 bits).
    pub modulus: Limbs,
    /// The operation.
    pub op: FFOps,
    /// The 12-bit table of the range checks.
    pub table: Rom<F>,
}

impl<F> TypedGadget<F> for ForeignFieldAdd<F>
where
    F: PrimeField,
{
    type Input = (Array<3>, Array<3>);
    type Output = Array<3>;
    const NAME: &'static str = "kimchi-ffadd";
    const ROWS: usize = 13;

    fn synthesize<E>(
        &self,
        env: &mut E,
        (left, right): Of<Self::Input, E::Var>,
    ) -> Of<Array<3>, E::Var>
    where
        E: Env<F>,
    {
        // Rows 0 and 1: w0 = 1 and w3 = 2^88, then w0 = 0.
        let two_88 = two_pow::<F>(LIMB_BITS);
        let first: [E::Var; 6] = core::array::from_fn(|c| {
            let value = match c {
                0 => F::ONE,
                3 => two_88,
                _ => F::ZERO,
            };
            env.advice(Cell::new(c, 0), [], move |[]| value)
        });
        let second: [E::Var; 6] =
            core::array::from_fn(|c| env.advice(Cell::new(c, 1), [], |[]| F::ZERO));
        let (one, base, zero) = (first[0].clone(), first[3].clone(), second[0].clone());
        let constants =
            GenericRow::new(GenericSpec::Const(F::ONE), Some(GenericSpec::Const(two_88)));
        env.call(&constants, first);
        let zeros = GenericRow::new(GenericSpec::Const(F::ZERO), None);
        env.call(&zeros, second);
        // Row 2: the operation; row 3: the bound; row 4: the bound's limbs.
        let gate = ForeignFieldAddGate {
            modulus: self.modulus,
            op: self.op,
        };
        let [r0, r1, r2, _] = env.call(&gate, (left, right));
        let bound = ForeignFieldAddGate {
            modulus: self.modulus,
            op: FFOps::Add,
        };
        let result = [r0, r1, r2];
        let [u0, u1, u2, q] = env.call(&bound, (result.clone(), [zero.clone(), zero, base]));
        env.connect(&one, &q);
        env.call(&Zero, ());
        // Rows 5-12: the result and the bound are three 88-bit limbs each.
        let check = MultiRangeCheck {
            table: self.table.clone(),
        };
        env.call(&check, result.clone());
        env.call(&check, [u0, u1, u2]);
        result
    }
}
