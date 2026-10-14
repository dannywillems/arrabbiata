//! The reduced Tate pairing `e: G1 x G2 -> Gt`.
//!
//! `e(P, Q) = f_{r,P}(psi(Q))^((p^12 - 1) / r)`, where `f_{r,P}` is the
//! Miller function with divisor `r (P) - r (O)` and `psi` maps the twist
//! into the curve over `Fq12`: `psi(x, y) = (x w^2, y w^3)`.
//!
//! Choices made for clarity over speed, each a known optimization left out:
//!
//! - the loop runs over the bits of `r` (the Tate pairing), not over the
//!   shorter `6 x + 2` of the optimal ate pairing;
//! - the point `P` is in `G1`, so the Miller loop's point arithmetic is in
//!   `Fq`, in affine coordinates, and only the line values live in `Fq12`;
//! - vertical lines are skipped: their values `x(psi(Q)) - x_T` lie in the
//!   subfield `Fq6 = Fq2(w^2)`, which the factor `p^6 - 1` of the final
//!   exponent sends to 1 (denominator elimination);
//! - the final exponentiation is a plain square-and-multiply by
//!   `(p^12 - 1) / r`, with no Frobenius shortcuts.
//!
//! ```
//! use arrabbiata_bn254::{G1, G2, pairing::pairing};
//! use arrabbiata_curve::Projective;
//! let e = pairing(&Projective::<G1>::generator(), &Projective::<G2>::generator());
//! assert!(!e.is_one());
//! ```

use arrabbiata_curve::Projective;
use arrabbiata_field::{Field, MontgomeryConfig, bigint};

use crate::{
    Fq, Fq2, FrConfig, G1, G2,
    fq12::{FINAL_EXPONENT, Fq12},
};

/// An element of the target group `Gt`, the subgroup of order `r` of
/// `Fq12*`, written multiplicatively.
///
/// ```
/// use arrabbiata_bn254::{G1, G2, pairing::{Gt, pairing}};
/// use arrabbiata_curve::Projective;
/// let e = pairing(&Projective::<G1>::generator(), &Projective::<G2>::generator());
/// assert_eq!(e * Gt::one(), e);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gt(Fq12);

impl Gt {
    /// The identity.
    ///
    /// ```
    /// use arrabbiata_bn254::pairing::Gt;
    /// assert!(Gt::one().is_one());
    /// ```
    #[must_use]
    pub const fn one() -> Self {
        Self(Fq12::one())
    }

    /// Whether this is the identity.
    ///
    /// ```
    /// use arrabbiata_bn254::pairing::Gt;
    /// assert!((Gt::one() * Gt::one()).is_one());
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.0.is_one()
    }

    /// `self^e` for an exponent given as little-endian limbs.
    ///
    /// ```
    /// use arrabbiata_bn254::{G1, G2, pairing::pairing};
    /// use arrabbiata_curve::Projective;
    /// let e = pairing(&Projective::<G1>::generator(), &Projective::<G2>::generator());
    /// assert_eq!(e.pow_limbs(&[2]), e * e);
    /// ```
    #[must_use]
    pub fn pow_limbs(&self, exponent: &[u64]) -> Self {
        Self(self.0.pow_limbs(exponent))
    }

    /// The underlying element of `Fq12`.
    ///
    /// ```
    /// use arrabbiata_bn254::pairing::Gt;
    /// assert!(Gt::one().as_fq12().is_one());
    /// ```
    #[must_use]
    pub const fn as_fq12(&self) -> &Fq12 {
        &self.0
    }
}

impl core::ops::Mul for Gt {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

/// The value at `psi(Q) = (xq w^2, yq w^3)` of the line through `T` of
/// slope `lambda`: `y - y_T - lambda (x - x_T)`.
fn line(lambda: Fq, xt: Fq, yt: Fq, xq: Fq2, yq: Fq2) -> Fq12 {
    let mut c = [Fq2::ZERO; 6];
    c[0] = Fq2::new(lambda * xt - yt, Fq::ZERO);
    c[2] = -xq.scale(lambda);
    c[3] = yq;
    Fq12::new(c)
}

/// The Miller function `f_{r,P}` evaluated at `psi(Q)`, up to factors in
/// proper subfields (removed by the final exponentiation).
///
/// ```
/// use arrabbiata_bn254::{G1, G2, pairing::{final_exponentiation, miller_loop, pairing}};
/// use arrabbiata_curve::Projective;
/// let (p, q) = (Projective::<G1>::generator(), Projective::<G2>::generator());
/// assert_eq!(final_exponentiation(&miller_loop(&p, &q)), pairing(&p, &q));
/// ```
#[must_use]
pub fn miller_loop(p: &Projective<G1>, q: &Projective<G2>) -> Fq12 {
    let p = p.to_affine();
    let q = q.to_affine();
    let (Some(xp), Some(yp), Some(xq), Some(yq)) = (p.x(), p.y(), q.x(), q.y()) else {
        return Fq12::one();
    };
    let r = &<FrConfig as MontgomeryConfig<4>>::MODULUS;
    let bits = bigint::num_bits(r) as usize;
    let two = Fq::from_u64(2);
    let three = Fq::from_u64(3);
    let (mut xt, mut yt) = (xp, yp);
    let mut at_infinity = false;
    let mut f = Fq12::one();
    let mut i = bits - 1;
    while i > 0 {
        i -= 1;
        if at_infinity {
            // Only reachable after the last addition, (r - 1) P + P = O.
            f = f * f;
            continue;
        }
        // Doubling: tangent at T. y_T != 0 since the group has odd order.
        let Some(inverse) = (two * yt).inverse() else {
            unreachable!("a point of order 2 in a group of odd order");
        };
        let lambda = three * xt * xt * inverse;
        f = f * f * line(lambda, xt, yt, xq, yq);
        let x2 = lambda * lambda - xt - xt;
        yt = lambda * (xt - x2) - yt;
        xt = x2;
        if bigint::bit(r, i) {
            if xt == xp {
                // T = -P (T = P cannot happen before the end): the line is
                // vertical, skipped, and T becomes the point at infinity.
                at_infinity = true;
            } else {
                let Some(inverse) = (xp - xt).inverse() else {
                    unreachable!("xp != xt");
                };
                let lambda = (yp - yt) * inverse;
                f = f * line(lambda, xt, yt, xq, yq);
                let x3 = lambda * lambda - xt - xp;
                yt = lambda * (xt - x3) - yt;
                xt = x3;
            }
        }
    }
    f
}

/// Raise to `(p^12 - 1) / r`, mapping `Fq12*` onto `Gt`.
///
/// ```
/// use arrabbiata_bn254::{fq12::Fq12, pairing::final_exponentiation};
/// assert!(final_exponentiation(&Fq12::one()).is_one());
/// ```
#[must_use]
pub fn final_exponentiation(f: &Fq12) -> Gt {
    Gt(f.pow_limbs(&FINAL_EXPONENT))
}

/// The reduced Tate pairing.
///
/// ```
/// use arrabbiata_bn254::{Fr, G1, G2, pairing::pairing};
/// use arrabbiata_curve::Projective;
/// use arrabbiata_field::Field;
/// let (p, q) = (Projective::<G1>::generator(), Projective::<G2>::generator());
/// let two = Fr::from_u64(2);
/// assert_eq!(pairing(&(p * two), &q), pairing(&p, &(q * two)));
/// ```
#[must_use]
pub fn pairing(p: &Projective<G1>, q: &Projective<G2>) -> Gt {
    final_exponentiation(&miller_loop(p, q))
}

/// `prod_i e(P_i, Q_i)`, with one final exponentiation for all the pairs.
///
/// ```
/// use arrabbiata_bn254::{G1, G2, pairing::{multi_pairing, pairing}};
/// use arrabbiata_curve::Projective;
/// let (p, q) = (Projective::<G1>::generator(), Projective::<G2>::generator());
/// // e(P, Q) e(-P, Q) = 1.
/// assert!(multi_pairing(&[(p, q), (-p, q)]).is_one());
/// ```
#[must_use]
pub fn multi_pairing(pairs: &[(Projective<G1>, Projective<G2>)]) -> Gt {
    let mut f = Fq12::one();
    let mut i = 0;
    while i < pairs.len() {
        f = f * miller_loop(&pairs[i].0, &pairs[i].1);
        i += 1;
    }
    final_exponentiation(&f)
}
