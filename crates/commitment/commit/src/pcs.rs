//! Polynomial commitment schemes.
//!
//! A polynomial commitment scheme (PCS) commits to a polynomial from some
//! *family* and later proves its value at a point. The family is a type
//! ([`PolynomialFamily`]), because schemes differ in what they commit to:
//!
//! - [`Univariate`]: a polynomial in one variable, by its coefficients,
//!   evaluated at a point of `F` (KZG, IPA; what PlonK needs).
//! - [`Multilinear`]: a multilinear polynomial in `k` variables, by its
//!   evaluations on the boolean hypercube `{0,1}^k`, evaluated at a point of
//!   `F^k` (Hyrax, Brakedown, and lattice schemes such as Greyhound, Hachi
//!   and Akita, which are post-quantum).
//!
//! A protocol states the family it needs as a bound
//! (`PC: PolynomialCommitment<F, Family = Univariate>`), so compiling a
//! univariate IOP with a multilinear scheme is a type error rather than a
//! silent misuse.
//!
//! Nothing here assumes that commitments form an `F`-module. Lattice
//! commitments are homomorphic only for combinations of small norm, so they
//! must not be folded with arbitrary field challenges; folding asks for the
//! separate, stronger [`crate::HomomorphicCommitment`].

use alloc::vec::Vec;
use core::{borrow::Borrow, fmt::Debug};

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, Transcript, absorb_slice, typed::Label};

/// A family of polynomials over `F`: how a polynomial is represented and
/// where it is evaluated.
pub trait PolynomialFamily<F>
where
    F: Field,
{
    /// A point of evaluation.
    type Point: ?Sized;

    /// A point of evaluation as an owned value, which claims and
    /// accumulators hold, and which absorbs into a transcript.
    type OwnedPoint: Clone + Debug + PartialEq + Absorb<F> + Borrow<Self::Point>;

    /// The value of the polynomial with representation `polynomial` at
    /// `point`. This is the reference semantics every scheme of the family
    /// must agree with.
    fn evaluate(polynomial: &[F], point: &Self::Point) -> F;
}

/// Univariate polynomials, by their coefficients (constant first),
/// evaluated at a point of `F`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Univariate;

impl Label for Univariate {
    const LABEL: u64 = 0x756e_6976;
}

impl<F> PolynomialFamily<F> for Univariate
where
    F: Field,
{
    type Point = F;
    type OwnedPoint = F;

    fn evaluate(polynomial: &[F], point: &F) -> F {
        let mut acc = F::ZERO;
        let mut i = polynomial.len();
        while i > 0 {
            i -= 1;
            acc = acc * *point + polynomial[i];
        }
        acc
    }
}

/// Multilinear polynomials in `k` variables, by their `2^k` evaluations on
/// the boolean hypercube, evaluated at a point of `F^k`.
///
/// Convention: the table is indexed by `b_0 b_1 ... b_(k-1)` read as a
/// binary number with `b_0` the most significant bit, and `point[i]` is the
/// value of variable `b_i`. A table whose length is not `2^point.len()`
/// evaluates to zero (no polynomial of the family has that shape).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Multilinear;

impl Label for Multilinear {
    const LABEL: u64 = 0x6d6c_696e;
}

/// A point of `F^k`, owned: the [`PolynomialFamily::OwnedPoint`] of
/// [`Multilinear`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coordinates<F>(pub Vec<F>);

impl<F> Borrow<[F]> for Coordinates<F> {
    fn borrow(&self) -> &[F] {
        &self.0
    }
}

impl<F> Absorb<F> for Coordinates<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.0);
    }
}

impl<F> PolynomialFamily<F> for Multilinear
where
    F: Field,
{
    type Point = [F];
    type OwnedPoint = Coordinates<F>;

    fn evaluate(polynomial: &[F], point: &[F]) -> F {
        let k = point.len();
        if k >= usize::BITS as usize || polynomial.len() != 1usize << k {
            return F::ZERO;
        }
        // Fix the variables one at a time, most significant first: the table
        // of f(r, b_1, ...) is (1 - r) * low half + r * high half.
        let mut table = polynomial.to_vec();
        let mut half = polynomial.len() / 2;
        let mut i = 0;
        while i < k {
            let r = point[i];
            let mut j = 0;
            while j < half {
                table[j] = table[j] + r * (table[j + half] - table[j]);
                j += 1;
            }
            half /= 2;
            i += 1;
        }
        table[0]
    }
}

/// A polynomial commitment scheme over `F` for a family of polynomials.
///
/// Laws: `verify(commit(p), x, v, proof)` holds for `(v, proof) = open(p, x)`
/// and `v = Family::evaluate(p, x)` (completeness); for a binding scheme,
/// `verify` fails except with negligible probability when `v` is not the
/// value of the committed polynomial (evaluation binding).
///
/// Every scheme carries a [`Label`] identifying it. Protocols compiled with a
/// scheme put the label in their transcript pattern, so that two schemes with
/// the same commitment type (IPA and KZG both commit to curve points) still
/// produce incompatible proofs, at the type level and in the transcript.
pub trait PolynomialCommitment<F>
where
    F: Field,
    Self: Label,
{
    /// The family of polynomials the scheme commits to.
    type Family: PolynomialFamily<F>;

    /// A commitment. It need not live in `F`, only absorb into a transcript
    /// over `F`.
    type Commitment: Clone + Debug + PartialEq + Absorb<F>;

    /// A proof of an evaluation.
    type Proof: Clone + Debug + PartialEq + Absorb<F>;

    /// Commit to `polynomial`.
    fn commit(&self, polynomial: &[F]) -> Self::Commitment;

    /// The value of `polynomial` at `point`, with a proof.
    fn open(
        &self,
        polynomial: &[F],
        point: &<Self::Family as PolynomialFamily<F>>::Point,
    ) -> (F, Self::Proof);

    /// Whether `proof` shows that the committed polynomial takes `value` at
    /// `point`.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &<Self::Family as PolynomialFamily<F>>::Point,
        value: F,
        proof: &Self::Proof,
    ) -> bool;
}

#[cfg(test)]
mod tests {
    extern crate std;

    use arrabbiata_field::{Field, Goldilocks as F};
    use proptest::prelude::*;
    use std::vec::Vec;

    use super::*;

    fn arb_f() -> impl Strategy<Value = F> {
        any::<u64>().prop_map(F::new)
    }

    proptest! {
        /// The multilinear extension agrees with its table on the hypercube.
        #[test]
        fn mle_interpolates_its_table(
            k in 0usize..5,
            seed in prop::collection::vec(arb_f(), 16),
            index in 0usize..16,
        ) {
            let table: Vec<F> = seed[..1 << k].to_vec();
            let index = index % (1 << k);
            let point: Vec<F> = (0..k)
                .map(|i| F::from_u64(((index >> (k - 1 - i)) & 1) as u64))
                .collect();
            prop_assert_eq!(<Multilinear as PolynomialFamily<F>>::evaluate(&table, &point), table[index]);
        }

        /// The multilinear extension is affine in each variable.
        #[test]
        fn mle_is_multilinear(
            table in prop::collection::vec(arb_f(), 8),
            point in prop::collection::vec(arb_f(), 3),
            i in 0usize..3,
            s in arb_f(), t in arb_f(),
        ) {
            let eval = |x: F| {
                let mut p = point.clone();
                p[i] = x;
                <Multilinear as PolynomialFamily<F>>::evaluate(&table, &p)
            };
            // f(x) affine in x: f(s + t) - f(s) = f(t) - f(0).
            prop_assert_eq!(eval(s + t) - eval(s), eval(t) - eval(F::ZERO));
        }
    }
}
