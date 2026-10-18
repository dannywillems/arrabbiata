//! Kimchi's scalar-field transcript in the circuit model: what a circuit
//! needs so that `arrabbiata_kimchi::verifier::scalar_transcript_in` runs
//! there, on the same typed pattern (`arrabbiata_kimchi::transcript::
//! Evaluation`) as the native verifier.
//!
//! - [`Permute`] and the `PermutationIn` of Kimchi's Poseidon in
//!   [`Circuit`]: the permutation on the rows of its gate;
//! - the `ChallengeModel` of [`Circuit`]: a squeezed element `x` written `lo
//!   + 2^128 hi` ([`Split128`]), `hi` checked to 128 bits and `lo` left to be
//!   checked where it is used (Pickles' `lowest_128_bits`), and the scalar
//!   `a lambda + b` of a challenge on the EndoMulScalar rows, which also check
//!   that it has 128 bits (Pickles' `to_field_checked`).

use alloc::vec::Vec;

use arrabbiata_field::{Field, FieldModel, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget};
use arrabbiata_kimchi::oracle::ChallengeModel;
use arrabbiata_kimchi_gadgets::{
    endomul_scalar::EndoMulScalarOf,
    generic::GenericRow,
    poseidon::{PoseidonPermutation, ROWS_PER_PERMUTATION},
};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, WIDTH};
use arrabbiata_transcript::PermutationIn;

use crate::model::{Affine, Circuit};

/// The crumbs (two bits each) of a 128-bit challenge.
const CRUMBS: usize = 64;

/// Kimchi's permutation on the rows of its gate, the output on the row
/// after its rounds, which it reserves.
#[derive(Clone, Copy, Debug)]
pub struct Permute<'a, F> {
    /// The permutation.
    pub poseidon: &'a KimchiPoseidon<F>,
}

impl<F> TypedGadget<F> for Permute<'_, F>
where
    F: Field,
{
    type Input = Array<Scalar, WIDTH>;
    type Output = Array<Scalar, WIDTH>;
    const NAME: &'static str = "pickles-permute";
    const ROWS: usize = ROWS_PER_PERMUTATION + 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        env.call(
            &PoseidonPermutation {
                poseidon: self.poseidon,
            },
            input,
        )
    }
}

impl<'e, F, E> PermutationIn<Circuit<'e, F, E>, F, WIDTH> for KimchiPoseidon<F>
where
    F: Field,
    E: Env<F>,
{
    fn permute_in(
        &self,
        model: &mut Circuit<'e, F, E>,
        state: [Affine<E::Var, F>; WIDTH],
    ) -> [Affine<E::Var, F>; WIDTH] {
        let cells = state.map(|v| model.cell(&v));
        let out = model.env().call(&Permute { poseidon: self }, cells);
        out.map(Affine::of)
    }
}

/// `x = lo + 2^128 hi` on one Generic row: `x` copied into column 0, `lo`
/// and `hi` witnessed in columns 1 and 2, the canonical split of `x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Split128;

impl<F> TypedGadget<F> for Split128
where
    F: PrimeField,
{
    type Input = Scalar;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "pickles-split-128";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Array<Scalar, 2>, E::Var>
    where
        E: Env<F>,
    {
        let x = env.copy(Cell::new(0, 0), &input);
        let lo = env.advice(Cell::new(1, 0), [&x], |[v]| {
            F::from_limbs(&[v.limb(0), v.limb(1)]).unwrap_or(F::ZERO)
        });
        let hi = env.advice(Cell::new(2, 0), [&x], |[v]| {
            let rest: Vec<u64> = (2..F::LIMBS).map(|i| v.limb(i)).collect();
            F::from_limbs(&rest).unwrap_or(F::ZERO)
        });
        let unused: [E::Var; 3] =
            core::array::from_fn(|i| env.advice(Cell::new(3 + i, 0), [], |[]| F::ZERO));
        let two_128 = F::from_u64(1 << 32).pow(4);
        let mut coefficients = [F::ZERO; 10];
        coefficients[0] = F::ONE;
        coefficients[1] = -F::ONE;
        coefficients[2] = -two_128;
        let [a, b, c] = unused;
        env.call(
            &GenericRow { coefficients },
            [x, lo.clone(), hi.clone(), a, b, c],
        );
        [lo, hi]
    }
}

impl<F, E> Circuit<'_, F, E>
where
    F: PrimeField,
    E: Env<F>,
{
    /// The EndoMulScalar rows on the cell `c`, `n` wired to `c`: `[a, b]`.
    fn endo_rows(&mut self, c: &E::Var) -> [E::Var; 2] {
        let zero = self.cell(&Affine::constant(F::ZERO));
        let two = self.cell(&Affine::constant(F::from_u64(2)));
        let start = [zero, two.clone(), two];
        let [n, a, b] = self
            .env()
            .call(&EndoMulScalarOf::<CRUMBS>, (start, c.clone()));
        self.env().connect(&n, c);
        [a, b]
    }
}

impl<F, E> ChallengeModel<F> for Circuit<'_, F, E>
where
    F: PrimeField,
    E: Env<F>,
{
    fn lowest_128_bits(&mut self, x: &Affine<E::Var, F>) -> Affine<E::Var, F> {
        let cell = self.cell(x);
        let [lo, hi] = self.env().call(&Split128, cell);
        // hi has 128 bits; lo is checked where it is used.
        let _ = self.endo_rows(&hi);
        Affine::of(lo)
    }

    fn endo(&mut self, chal: &Affine<E::Var, F>, lambda: F) -> Affine<E::Var, F> {
        let cell = self.cell(chal);
        let [a, b] = self.endo_rows(&cell);
        let scaled = self.scale(&Affine::of(a), lambda);
        self.add(&scaled, &Affine::of(b))
    }
}
