//! The challenge polynomial of an IPA, and the accumulator it defines.
//!
//! After `k` rounds with challenges `u_1, ..., u_k`, the folded evaluation
//! vector is the value at the point of
//!
//! ```text
//! b(X) = prod_{i=1..k} (1 + u_i X^(2^(k-i))),
//! ```
//!
//! and the folded generator `sg` is the commitment to `b`: its coefficients
//! `s` are the weights of the generators. `b(x)` costs `k` operations;
//! `sg = <s, G>` costs an MSM of the size of the key. Halo defers the
//! second: the pair `(u, sg)` is a [`RecursionChallenge`], a claim that `sg`
//! commits to `b`, checked once for many proofs.
//!
//! The conventions (which challenge goes with which power, the order of the
//! coefficients, the combination of evaluations) are Kimchi's
//! (`poly-commitment/src/commitment.rs`), checked on Kimchi's own output in
//! `tests/kimchi.rs`.

use alloc::vec::Vec;

use arrabbiata_curve::{Curve, Projective, absorb_point};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_transcript::{Absorb, Transcript, absorb_slice, typed::Label};

/// `b(x) = prod_{i=0..k-1} (1 + chals[i] x^(2^(k-1-i)))`, Kimchi's
/// `b_poly`.
#[must_use]
pub fn b_poly<F>(chals: &[F], x: F) -> F
where
    F: Field,
{
    let k = chals.len();
    // pow_twos[i] = x^(2^i).
    let mut pow_twos = Vec::with_capacity(k);
    let mut power = x;
    let mut i = 0;
    while i < k {
        pow_twos.push(power);
        power = power * power;
        i += 1;
    }
    let mut product = F::ONE;
    let mut i = 0;
    while i < k {
        product = product * (F::ONE + chals[i] * pow_twos[k - 1 - i]);
        i += 1;
    }
    product
}

/// `sum_j evalscale^j b(points[j])`: the folded evaluation vector `b0` of
/// an opening at `points` combined by `evalscale`, a function of the round
/// challenges alone (Kimchi's `b0`, Pickles' `b`).
#[must_use]
pub fn folded_evaluation<F>(chals: &[F], points: &[F], evalscale: F) -> F
where
    F: Field,
{
    let mut b0 = F::ZERO;
    let mut scale = F::ONE;
    for point in points {
        b0 = b0 + scale * b_poly(chals, *point);
        scale = scale * evalscale;
    }
    b0
}

/// The `2^k` coefficients of `b`, Kimchi's `b_poly_coefficients`: `s_i` is
/// the product of `chals[k - 1 - j]` over the bits `j` set in `i`.
#[must_use]
pub fn b_poly_coefficients<F>(chals: &[F]) -> Vec<F>
where
    F: Field,
{
    let k = chals.len();
    let length = 1usize << k;
    let mut s = alloc::vec![F::ONE; length];
    // Each index is the previous one with its highest bit set: s_i =
    // s_(i - 2^j) chals[k - 1 - j] for 2^j <= i < 2^(j + 1).
    let mut j = 0;
    while j < k {
        let low = 1usize << j;
        let mut i = low;
        while i < 2 * low {
            s[i] = s[i - low] * chals[k - 1 - j];
            i += 1;
        }
        j += 1;
    }
    s
}

/// `sum_i polyscale^i sum_j evalscale^j values[i][j]`, for `values[i][j]`
/// the value of polynomial `i` at point `j`: the value of the combined
/// polynomial on the combined evaluation vector (Kimchi's
/// `combined_inner_product`, without chunks).
#[must_use]
pub fn combined_inner_product<F>(polyscale: F, evalscale: F, values: &[Vec<F>]) -> F
where
    F: Field,
{
    let mut result = F::ZERO;
    let mut polyscale_i = F::ONE;
    let mut i = 0;
    while i < values.len() {
        let mut row = F::ZERO;
        let mut j = values[i].len();
        // Horner in evalscale.
        while j > 0 {
            j -= 1;
            row = row * evalscale + values[i][j];
        }
        result = result + polyscale_i * row;
        polyscale_i = polyscale_i * polyscale;
        i += 1;
    }
    result
}

/// `sum_j evalscale^j (1, x_j, x_j^2, ..., x_j^(n-1))`: the vector an
/// opening at `points` takes the inner product with.
#[must_use]
pub fn evaluation_vector<F>(points: &[F], evalscale: F, n: usize) -> Vec<F>
where
    F: Field,
{
    let mut b = alloc::vec![F::ZERO; n];
    let mut scale = F::ONE;
    let mut j = 0;
    while j < points.len() {
        let mut power = scale;
        let mut i = 0;
        while i < n {
            b[i] = b[i] + power;
            power = power * points[j];
            i += 1;
        }
        scale = scale * evalscale;
        j += 1;
    }
    b
}

/// The accumulator of an IPA opening: its round challenges and `sg`, the
/// claim that `sg` commits to their challenge polynomial `b`. Kimchi's
/// `RecursionChallenge`, with `comm` one point (no chunks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursionChallenge<C>
where
    C: Curve,
{
    /// The round challenges, first round first.
    pub chals: Vec<C::Scalar>,
    /// The claimed commitment to `b`.
    pub comm: Projective<C>,
}

impl<C> RecursionChallenge<C>
where
    C: Curve,
{
    /// `b(x)`, in `k` operations: what lets a verifier use the accumulator
    /// as an evaluation claim without the MSM.
    #[must_use]
    pub fn evaluate(&self, x: C::Scalar) -> C::Scalar {
        b_poly(&self.chals, x)
    }

    /// The coefficients of `b`: the witness behind the claim.
    #[must_use]
    pub fn coefficients(&self) -> Vec<C::Scalar> {
        b_poly_coefficients(&self.chals)
    }
}

impl<C> Label for RecursionChallenge<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x7265_6368;
}

impl<C> Absorb<C::Scalar> for RecursionChallenge<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<C::Scalar>,
    {
        absorb_slice(transcript, &self.chals);
        absorb_point(&self.comm, transcript);
    }
}
