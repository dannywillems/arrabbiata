//! Kimchi's AND gadget (`polynomials/and.rs`): `a AND b` from the identity
//! `a + b = (a XOR b) + 2 (a AND b)`, as the XOR of the words and one
//! double Generic row, `sum = a + b` then `sum - xor - 2 and = 0`:
//!
//! | row        | 0   | 1   | 2     | 3     | 4     | 5     |
//! | ---------- | --- | --- | ----- | ----- | ----- | ----- |
//! | XOR rows   | `a` | `b` | `xor` |       |       |       |
//! | last       | `a` | `b` | `sum` | `sum` | `xor` | `and` |
//!
//! `a`, `b` and `xor` copied from the XOR's first row, `sum` from itself
//! across the halves (Kimchi's `extend_and` wiring).

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget};

use crate::{
    generic::{GenericRow, GenericSpec},
    xor::Xor,
};

/// The first half-gate: `sum = a + b`.
fn sum<F: Field>() -> GenericSpec<F> {
    GenericSpec::Add {
        left: None,
        right: None,
        output: None,
    }
}

/// The second: `sum - xor - 2 and = 0`.
fn and<F: Field>() -> GenericSpec<F> {
    GenericSpec::Add {
        left: None,
        right: Some(-F::ONE),
        output: Some(-(F::ONE + F::ONE)),
    }
}

/// The AND of two words of `BITS` bits: input `[a, b]`, cells of copied
/// columns; output `a AND b`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct And<F, const BITS: usize> {
    /// The XOR of the words.
    pub xor: Xor<F, BITS>,
}

impl<F, const B: usize> TypedGadget<F> for And<F, B>
where
    F: PrimeField,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "kimchi-and";
    const ROWS: usize = Xor::<F, B>::ROWS + 1;

    fn synthesize<E>(&self, env: &mut E, [a, b]: Of<Array<Scalar, 2>, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let xor = env.call(&self.xor, [a.clone(), b.clone()]);
        let row = Xor::<F, B>::ROWS;
        let at = |column| Cell::new(column, row);
        let output =
            |spec: GenericSpec<F>| move |[l, r]: [F; 2]| spec.output(l, r).unwrap_or(F::ZERO);
        let left = env.copy(at(0), &a);
        let right = env.copy(at(1), &b);
        let total = env.advice(at(2), [&left, &right], output(sum()));
        let again = env.copy(at(3), &total);
        let xored = env.copy(at(4), &xor);
        let conjunction = env.advice(at(5), [&again, &xored], output(and()));
        let generic = GenericRow::new(sum(), Some(and()));
        env.call(
            &generic,
            [left, right, total, again, xored, conjunction.clone()],
        );
        conjunction
    }
}
