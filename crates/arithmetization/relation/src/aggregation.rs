//! How the `M` constraints of a circuit are combined into one equation per
//! row.
//!
//! A relaxed instance claims, for every row `i`,
//!
//! ```text
//! sum_j c_j * f_j^D(u, w_i) = E_i
//! ```
//!
//! where `f_j^D` is constraint `j` homogenized to degree `D`. What the
//! coefficients `c_j` are is the aggregation strategy, chosen by a type:
//!
//! - [`Single`]: one constraint, no coefficient (`c_0 = 1`, degree zero).
//!   The relation has degree `D`. Only implemented for `M = 1`, so a
//!   multi-constraint circuit with this strategy does not compile. Nova is
//!   this strategy with `D = 2`.
//! - [`Powers`]: the coefficients are instance variables, `c_j = alpha^j` on
//!   a fresh instance for a Fiat-Shamir challenge `alpha`, and folded
//!   linearly like every other instance variable. Each `c_j` has degree one,
//!   so the relation has degree `D + 1`. This is the construction of the
//!   Arrabbiata note (section "combining constraints"), following
//!   ProtoStar: the challenge cannot be folded as a power, so its powers are
//!   folded independently, and the folded coefficients are in general no
//!   longer powers of anything.
//!
//! ```
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_relation::aggregation::{Aggregation, Powers, Single};
//!
//! assert_eq!(<Single as Aggregation<F, 1>>::EXTRA_DEGREE, 0);
//! assert_eq!(<Powers as Aggregation<F, 3>>::EXTRA_DEGREE, 1);
//! ```
//!
//! ```compile_fail
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_relation::aggregation::{Aggregation, Single};
//! // Single only aggregates one constraint.
//! let _ = <Single as Aggregation<F, 2>>::EXTRA_DEGREE;
//! ```

use core::fmt::Debug;

use arrabbiata_field::Field;
use arrabbiata_transcript::{Absorb, Transcript, absorb_length, tag};

/// An aggregation strategy for `M` constraints over `F`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_relation::aggregation::{Aggregation, Combiner, Powers};
///
/// let a = Combiner::<F, 2>::powers(F::new(3));
/// let b = Combiner::<F, 2>::powers(F::new(5));
/// let folded = <Powers as Aggregation<F, 2>>::fold(&a, &b, F::new(2));
/// // 3 + 2 * 5: the coefficient of constraint 1 is folded linearly.
/// assert_eq!(<Powers as Aggregation<F, 2>>::coefficient(&folded, 1), Some(F::new(13)));
/// ```
pub trait Aggregation<F, const M: usize>
where
    F: Field,
{
    /// The degree the coefficients add to the relation: 0 or 1.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_relation::aggregation::{Aggregation, Powers};
    /// assert_eq!(<Powers as Aggregation<F, 4>>::EXTRA_DEGREE, 1);
    /// ```
    const EXTRA_DEGREE: usize;

    /// The coefficients carried by an instance.
    type Coefficients: Clone + Debug + PartialEq + Absorb<F>;

    /// The coefficient of constraint `j`, or `None` when the strategy has no
    /// coefficient variables (the coefficient is then the constant 1).
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_relation::aggregation::{Aggregation, NoCoefficients, Single};
    /// assert_eq!(<Single as Aggregation<F, 1>>::coefficient(&NoCoefficients, 0), None);
    /// ```
    fn coefficient(coefficients: &Self::Coefficients, j: usize) -> Option<F>;

    /// `a + r * b`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::aggregation::{Aggregation, Combiner, Powers};
    /// let one = Combiner::<F, 1>::powers(F::new(9));
    /// let folded = <Powers as Aggregation<F, 1>>::fold(&one, &one, F::new(4));
    /// // c_0 = 1 on both, so the folded c_0 is 1 + 4.
    /// assert_eq!(folded.values(), &[F::new(5)]);
    /// ```
    fn fold(a: &Self::Coefficients, b: &Self::Coefficients, r: F) -> Self::Coefficients;

    /// The coefficients of a fresh instance, drawn from the transcript.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::aggregation::{Aggregation, Powers};
    /// use arrabbiata_transcript::toy_sponge;
    /// let c = <Powers as Aggregation<F, 3>>::sample(&mut toy_sponge(0));
    /// let alpha = c.values()[1];
    /// assert_eq!(c.values(), &[F::ONE, alpha, alpha * alpha]);
    /// ```
    fn sample<T>(transcript: &mut T) -> Self::Coefficients
    where
        T: Transcript<F>;
}

/// One constraint, no combiner. See the module documentation.
///
/// ```
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_relation::aggregation::{Aggregation, Single};
/// assert_eq!(<Single as Aggregation<F, 1>>::EXTRA_DEGREE, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Single;

/// The (absent) coefficients of the [`Single`] strategy.
///
/// ```
/// use arrabbiata_relation::aggregation::NoCoefficients;
/// assert_eq!(NoCoefficients, NoCoefficients::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoCoefficients;

impl<F> Absorb<F> for NoCoefficients
where
    F: Field,
{
    fn absorb_into<T>(&self, _transcript: &mut T)
    where
        T: Transcript<F>,
    {
    }
}

impl<F> Aggregation<F, 1> for Single
where
    F: Field,
{
    const EXTRA_DEGREE: usize = 0;

    type Coefficients = NoCoefficients;

    fn coefficient(_: &NoCoefficients, _: usize) -> Option<F> {
        None
    }

    fn fold(_: &NoCoefficients, _: &NoCoefficients, _: F) -> NoCoefficients {
        NoCoefficients
    }

    fn sample<T>(_: &mut T) -> NoCoefficients
    where
        T: Transcript<F>,
    {
        NoCoefficients
    }
}

/// Folded powers of a challenge. See the module documentation.
///
/// ```
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_relation::aggregation::{Aggregation, Powers};
/// assert_eq!(<Powers as Aggregation<F, 2>>::EXTRA_DEGREE, 1);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Powers;

/// The coefficients `c_0, ..., c_{M-1}` of the [`Powers`] strategy.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_relation::aggregation::Combiner;
/// let c = Combiner::<F, 3>::powers(F::new(2));
/// assert_eq!(c.values(), &[F::ONE, F::new(2), F::new(4)]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Combiner<F, const M: usize>([F; M]);

impl<F, const M: usize> Combiner<F, M>
where
    F: Field,
{
    /// The coefficients `1, alpha, ..., alpha^(M-1)` of a fresh instance.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::aggregation::Combiner;
    /// assert_eq!(Combiner::<F, 2>::powers(F::new(7)).values(), &[F::ONE, F::new(7)]);
    /// ```
    #[must_use]
    pub fn powers(alpha: F) -> Self {
        let mut coefficients = [F::ONE; M];
        let mut j = 1;
        while j < M {
            coefficients[j] = coefficients[j - 1] * alpha;
            j += 1;
        }
        Self(coefficients)
    }

    /// The coefficients.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_relation::aggregation::Combiner;
    /// assert_eq!(Combiner::<F, 1>::powers(F::new(7)).values().len(), 1);
    /// ```
    #[must_use]
    pub const fn values(&self) -> &[F; M] {
        &self.0
    }
}

impl<F, const M: usize> Absorb<F> for Combiner<F, M>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_length(transcript, M);
        let mut j = 0;
        while j < M {
            transcript.absorb(self.0[j]);
            j += 1;
        }
    }
}

impl<F, const M: usize> Aggregation<F, M> for Powers
where
    F: Field,
{
    const EXTRA_DEGREE: usize = 1;

    type Coefficients = Combiner<F, M>;

    fn coefficient(coefficients: &Combiner<F, M>, j: usize) -> Option<F> {
        coefficients.0.get(j).copied()
    }

    fn fold(a: &Combiner<F, M>, b: &Combiner<F, M>, r: F) -> Combiner<F, M> {
        let mut out = a.0;
        let mut j = 0;
        while j < M {
            out[j] = a.0[j] + r * b.0[j];
            j += 1;
        }
        Combiner(out)
    }

    fn sample<T>(transcript: &mut T) -> Combiner<F, M>
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::COMBINER));
        Combiner::powers(transcript.squeeze())
    }
}
