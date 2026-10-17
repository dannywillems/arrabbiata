//! Kimchi's structured reference string (`poly-commitment/src/ipa.rs`,
//! `SRS::create`) and its commitments.
//!
//! The SRS is `depth` generators `g_i` and a blinding point `h`, each the
//! image of 64 hashed bytes under Kimchi's group map: `g_i` from
//! `Blake2b512(i as 4 big-endian bytes)`, `h` from
//! `Blake2b512("srs_misc" || 0u32)`. Nobody knows a discrete-log relation
//! between them, which is all the IPA needs (no trusted setup).
//!
//! Bytes become a base-field element by Kimchi's packing: the first 31
//! bytes, bit `j` of byte `i` being bit `247 - (8 i + j)` of the integer
//! (`BigInt::from_bits_be` on the bits read least significant first in
//! each byte).
//!
//! The group map ([`GroupMap`], Kimchi's `groupmap::BWParameters`) sends a
//! field element `t` to a point of `y^2 = x^3 + b`: it computes three
//! candidate abscissas from `t` and the precomputed point `u` (the least
//! positive integer with `f(u) = u^3 + b != 0`), and takes the first whose
//! `f(x)` is a square, with `y = sqrt(f(x))`. Which root `sqrt` returns
//! is part of the definition; the field crate's Tonelli-Shanks is
//! arkworks' step for step, with the same non-residue, so it returns the
//! same one (the tests check the points against Kimchi's).
//!
//! A commitment ([`PolyComm`]) to a polynomial of `n` coefficients is
//! `ceil(n / depth)` chunks, chunk `k` committing coefficients
//! `k depth .. (k + 1) depth` against `g_0, g_1, ...` (Kimchi commits the
//! evaluations against chunked Lagrange bases, the same points).

use alloc::vec::Vec;

use arrabbiata_curve::{Affine, Curve, Projective, msm};
use arrabbiata_field::{Field, PrimeField, Sqrt};
use blake2::{Blake2b512, Digest};

/// The bytes read from each hash.
const PACKED_BYTES: usize = 31;

/// Kimchi's group map for the curve `C` (`groupmap::BWParameters`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupMap<C>
where
    C: Curve,
{
    u: C::Base,
    fu: C::Base,
    sqrt_neg_three_u_squared_minus_u_over_2: C::Base,
    sqrt_neg_three_u_squared: C::Base,
    inv_three_u_squared: C::Base,
}

/// `x^3 + b`.
fn curve_eqn<C: Curve>(x: C::Base) -> C::Base {
    x.square() * x + C::b()
}

impl<C> GroupMap<C>
where
    C: Curve,
    C::Base: Sqrt,
{
    /// The parameters: `u` the least of `1, 2, ...` with `f(u) != 0`, and
    /// the constants of the candidate abscissas.
    ///
    /// # Panics
    ///
    /// If `-3 u^2` is not a square, or 2 or 3 is not invertible: then the
    /// field has no such map.
    #[must_use]
    pub fn setup() -> Self {
        let mut u = C::Base::ONE;
        while curve_eqn::<C>(u).is_zero() {
            u = u + C::Base::ONE;
        }
        let fu = curve_eqn::<C>(u);
        let two = C::Base::ONE + C::Base::ONE;
        let three = two + C::Base::ONE;
        let three_u_squared = u.square() * three;
        let (Some(inv_three_u_squared), Some(sqrt_neg_three_u_squared), Some(two_inv)) = (
            three_u_squared.inverse(),
            (-three_u_squared).sqrt(),
            two.inverse(),
        ) else {
            panic!("the field has no group map for this curve")
        };
        Self {
            u,
            fu,
            sqrt_neg_three_u_squared_minus_u_over_2: (sqrt_neg_three_u_squared - u) * two_inv,
            sqrt_neg_three_u_squared,
            inv_three_u_squared,
        }
    }

    /// The three candidate abscissas of `t`.
    #[must_use]
    pub fn potential_xs(&self, t: C::Base) -> [C::Base; 3] {
        let t2 = t.square();
        let alpha = ((t2 + self.fu) * t2).inverse().unwrap_or(C::Base::ZERO);
        let x1 = self.sqrt_neg_three_u_squared_minus_u_over_2
            - t2.square() * alpha * self.sqrt_neg_three_u_squared;
        let x2 = -self.u - x1;
        let t2_plus_fu = t2 + self.fu;
        let x3 = self.u - t2_plus_fu.square() * (alpha * t2_plus_fu) * self.inv_three_u_squared;
        [x1, x2, x3]
    }

    /// The point of `t`: the first candidate on the curve.
    ///
    /// # Panics
    ///
    /// Never for a valid map: one of the three `f(x_i)` is always a square.
    #[must_use]
    pub fn to_group(&self, t: C::Base) -> Affine<C> {
        for x in self.potential_xs(t) {
            if let Some(y) = curve_eqn::<C>(x).sqrt()
                && let Some(p) = Affine::new(x, y)
            {
                return p;
            }
        }
        panic!("no candidate abscissa is on the curve")
    }
}

/// The base-field element of `bytes` (at least 31 of them) by Kimchi's
/// packing.
///
/// # Panics
///
/// If fewer than 31 bytes are given, or the field has at most 248 bits.
#[must_use]
pub fn packed<F: PrimeField>(bytes: &[u8]) -> F {
    assert!(bytes.len() >= PACKED_BYTES, "Kimchi packs 31 bytes");
    let mut limbs = [0u64; 4];
    let top = 8 * PACKED_BYTES - 1;
    for (i, byte) in bytes[..PACKED_BYTES].iter().enumerate() {
        for j in 0..8 {
            if (byte >> j) & 1 == 1 {
                let k = top - (8 * i + j);
                limbs[k / 64] |= 1 << (k % 64);
            }
        }
    }
    match F::from_limbs(&limbs[..F::LIMBS.min(4)]) {
        Some(x) => x,
        None => panic!("248 bits do not fit in the field"),
    }
}

/// The point of 64 hashed bytes: the group map of their packing.
#[must_use]
pub fn point_of_random_bytes<C>(map: &GroupMap<C>, bytes: &[u8]) -> Projective<C>
where
    C: Curve,
    C::Base: PrimeField + Sqrt,
{
    // The Pasta curves have cofactor 1: Kimchi's `mul_by_cofactor` is the
    // identity on them.
    map.to_group(packed(bytes)).to_projective()
}

/// A chunked commitment: one point per chunk of `depth` coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyComm<C>
where
    C: Curve,
{
    /// The chunks, lowest coefficients first.
    pub chunks: Vec<Projective<C>>,
}

/// Kimchi's SRS over `C`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Srs<C>
where
    C: Curve,
{
    /// The commitment generators.
    pub g: Vec<Projective<C>>,
    /// The blinding generator.
    pub h: Projective<C>,
}

impl<C> Srs<C>
where
    C: Curve,
    C::Base: PrimeField + Sqrt,
{
    /// Kimchi's `SRS::create(depth)`.
    ///
    /// # Panics
    ///
    /// If `depth` exceeds `2^32`.
    #[must_use]
    pub fn create(depth: usize) -> Self {
        let map = GroupMap::setup();
        let g = (0..depth)
            .map(|i| {
                let index = match u32::try_from(i) {
                    Ok(index) => index,
                    Err(_) => panic!("Kimchi's SRS indexes points by u32"),
                };
                point_of_random_bytes(&map, &Blake2b512::digest(index.to_be_bytes()))
            })
            .collect();
        let mut misc = Blake2b512::new();
        misc.update(b"srs_misc");
        misc.update(0u32.to_be_bytes());
        let h = point_of_random_bytes(&map, &misc.finalize());
        Self { g, h }
    }
}

impl<C> Srs<C>
where
    C: Curve,
{
    /// The chunk size: the number of generators.
    #[must_use]
    pub fn max_poly_size(&self) -> usize {
        self.g.len()
    }

    /// The commitment to the polynomial with `coefficients`, padded or not,
    /// in `ceil(rows / depth)` chunks (at least one): Kimchi sizes the
    /// chunks by the domain, not by the degree, so a polynomial of `rows`
    /// evaluations always has as many chunks.
    ///
    /// # Panics
    ///
    /// If there are more coefficients than `rows`, or the SRS is empty.
    #[must_use]
    pub fn commit_non_hiding(&self, coefficients: &[C::Scalar], rows: usize) -> PolyComm<C> {
        assert!(coefficients.len() <= rows, "more coefficients than rows");
        let depth = self.max_poly_size();
        assert!(depth > 0, "an empty SRS commits to nothing");
        let chunks = rows.div_ceil(depth).max(1);
        PolyComm {
            chunks: (0..chunks)
                .map(|k| {
                    let start = (k * depth).min(coefficients.len());
                    let end = ((k + 1) * depth).min(coefficients.len());
                    msm(&self.g, &coefficients[start..end])
                })
                .collect(),
        }
    }

    /// Kimchi's `mask_fixed`: every chunk blinded by `h` with blinder 1.
    #[must_use]
    pub fn mask_fixed(&self, commitment: PolyComm<C>) -> PolyComm<C> {
        PolyComm {
            chunks: commitment.chunks.into_iter().map(|c| c + self.h).collect(),
        }
    }
}
