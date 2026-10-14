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
//!
//! ```
//! use arrabbiata_commit::{Multilinear, PolynomialFamily, Univariate};
//! use arrabbiata_field::{Field, Goldilocks as F};
//!
//! // The reference semantics of the two families.
//! let p = [F::new(1), F::new(2)]; // 1 + 2X
//! assert_eq!(<Univariate as PolynomialFamily<F>>::evaluate(&p, &F::new(5)), F::new(11));
//! let table = [F::new(3), F::new(7)]; // f(0) = 3, f(1) = 7
//! let v = <Multilinear as PolynomialFamily<F>>::evaluate(&table, &[F::new(2)][..]);
//! assert_eq!(v, F::new(11)); // 3 + 2 * (7 - 3)
//! ```

use core::fmt::Debug;

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, typed::Label};

/// A family of polynomials over `F`: how a polynomial is represented and
/// where it is evaluated.
///
/// ```
/// use arrabbiata_commit::{PolynomialFamily, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// // 1 + X + X^2 at 2.
/// let v = <Univariate as PolynomialFamily<F>>::evaluate(&[F::new(1); 3], &F::new(2));
/// assert_eq!(v, F::new(7));
/// ```
pub trait PolynomialFamily<F>
where
    F: Field,
{
    /// A point of evaluation.
    type Point: ?Sized;

    /// The value of the polynomial with representation `polynomial` at
    /// `point`. This is the reference semantics every scheme of the family
    /// must agree with.
    ///
    /// ```
    /// use arrabbiata_commit::{Multilinear, PolynomialFamily};
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// // On the hypercube, the multilinear extension is the table.
    /// let table = [F::new(1), F::new(2), F::new(3), F::new(4)];
    /// let at = |b0: u64, b1: u64| <Multilinear as PolynomialFamily<F>>::evaluate(&table, &[F::new(b0), F::new(b1)][..]);
    /// assert_eq!(at(1, 0), F::new(3));
    /// ```
    fn evaluate(polynomial: &[F], point: &Self::Point) -> F;
}

/// Univariate polynomials, by their coefficients (constant first),
/// evaluated at a point of `F`.
///
/// ```
/// use arrabbiata_commit::{PolynomialFamily, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// assert_eq!(<Univariate as PolynomialFamily<F>>::evaluate(&[F::new(4)], &F::new(9)), F::new(4));
/// ```
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
///
/// ```
/// use arrabbiata_commit::{Multilinear, PolynomialFamily};
/// use arrabbiata_field::{Field, Goldilocks as F};
/// // f(b0, b1) with table [f(0,0), f(0,1), f(1,0), f(1,1)] = [0, 1, 1, 0]
/// // (XOR), at (1/2, 1/2) is the average, 1/2.
/// let half = F::new(2).inverse().unwrap();
/// let table = [F::ZERO, F::ONE, F::ONE, F::ZERO];
/// assert_eq!(<Multilinear as PolynomialFamily<F>>::evaluate(&table, &[half, half][..]), half);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Multilinear;

impl Label for Multilinear {
    const LABEL: u64 = 0x6d6c_696e;
}

impl<F> PolynomialFamily<F> for Multilinear
where
    F: Field,
{
    type Point = [F];

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
///
/// ```
/// use arrabbiata_bn254::{Bn254, Fr};
/// use arrabbiata_commit::{PolynomialCommitment, Univariate};
/// use arrabbiata_field::Field;
/// use arrabbiata_kzg::Kzg;
///
/// fn check<PC>(pc: &PC, p: &[Fr], x: Fr) -> bool
/// where
///     PC: PolynomialCommitment<Fr, Family = Univariate>,
/// {
///     let (v, proof) = pc.open(p, &x);
///     pc.verify(&pc.commit(p), &x, v, &proof)
/// }
/// let kzg = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
/// assert!(check(&kzg, &[Fr::ONE, Fr::ONE], Fr::from_u64(3)));
/// ```
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
    ///
    /// ```
    /// use arrabbiata_bn254::{Bn254, Fr};
    /// use arrabbiata_commit::PolynomialCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_kzg::Kzg;
    /// // A test setup: its secret is known.
    /// let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    /// // A constant polynomial commits to a multiple of the generator.
    /// let c = pc.commit(&[Fr::from_u64(4)]);
    /// assert_eq!(c, pc.commit(&[Fr::from_u64(4), Fr::ZERO]));
    /// ```
    fn commit(&self, polynomial: &[F]) -> Self::Commitment;

    /// The value of `polynomial` at `point`, with a proof.
    ///
    /// ```
    /// use arrabbiata_bn254::{Bn254, Fr};
    /// use arrabbiata_commit::PolynomialCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_kzg::Kzg;
    /// // A test setup: its secret is known.
    /// let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    /// assert_eq!(pc.open(&[Fr::ONE, Fr::ONE], &Fr::from_u64(9)).0, Fr::from_u64(10));
    /// ```
    fn open(
        &self,
        polynomial: &[F],
        point: &<Self::Family as PolynomialFamily<F>>::Point,
    ) -> (F, Self::Proof);

    /// Whether `proof` shows that the committed polynomial takes `value` at
    /// `point`.
    ///
    /// ```
    /// use arrabbiata_bn254::{Bn254, Fr};
    /// use arrabbiata_commit::PolynomialCommitment;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_kzg::Kzg;
    /// // A test setup: its secret is known.
    /// let pc = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(1234));
    /// let p = [Fr::from_u64(5)];
    /// let (v, proof) = pc.open(&p, &Fr::ZERO);
    /// assert!(pc.verify(&pc.commit(&p), &Fr::ZERO, v, &proof));
    /// assert!(!pc.verify(&pc.commit(&p), &Fr::ZERO, v + Fr::ONE, &proof));
    /// ```
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
