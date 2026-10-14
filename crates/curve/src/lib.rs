//! Elliptic curves `y^2 = x^3 + b` in short Weierstrass form.
//!
//! A curve is a type implementing [`Curve`]: its base field, its scalar
//! field (the field of order the group order `r`, assumed prime), the
//! coefficient `b`, and a generator. Points are [`Projective`] (for
//! arithmetic) or [`Affine`] (for input, output and absorption).
//!
//! Addition uses the complete formulas of Renes, Costello and Batina
//! (*Complete addition formulas for prime order elliptic curves*,
//! EUROCRYPT 2016, ePrint 2015/1060, Algorithm 7 for `a = 0`). They are
//! complete on curves whose group has odd order: one formula, with no
//! special case for doubling, for the identity, or for inverse points. The
//! code has no branches on point values, which keeps it short, uniform, and
//! easy to extract and reason about.
//!
//! ```
//! use arrabbiata_curve::{Curve, Projective};
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::{Fq, Pallas};
//!
//! let g = Projective::<Pallas>::generator();
//! let five_g = g * Fq::from_u64(5);
//! assert_eq!(five_g, g + g + g + g + g);
//! // The group has order q: (q - 1) G = -G.
//! assert_eq!(g * (-Fq::ONE), -g);
//! ```

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use arrabbiata_field::{Field, PrimeField, Sqrt};
use arrabbiata_transcript::{Absorb, Transcript, typed::Label};

/// A curve `y^2 = x^3 + b` over `Base` whose group of points has prime
/// order `r`, the characteristic of `Scalar`.
///
/// ```
/// use arrabbiata_curve::Curve;
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::{Fp, Pallas};
/// assert_eq!(Pallas::b(), Fp::from_u64(5));
/// ```
pub trait Curve
where
    Self: 'static + Copy + Debug + Eq + Send + Sync,
{
    /// The field of coordinates.
    type Base: Field;

    /// The field of scalars, of order the group order.
    type Scalar: PrimeField;

    /// The coefficient `b`.
    ///
    /// ```
    /// use arrabbiata_curve::Curve;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::{Fq, Vesta};
    /// assert_eq!(Vesta::b(), Fq::from_u64(5));
    /// ```
    fn b() -> Self::Base;

    /// A generator of the group, in affine coordinates `(x, y)`.
    ///
    /// ```
    /// use arrabbiata_curve::{Affine, Curve};
    /// use arrabbiata_pasta::Pallas;
    /// let (x, y) = Pallas::generator();
    /// assert!(Affine::<Pallas>::new(x, y).is_some());
    /// ```
    fn generator() -> (Self::Base, Self::Base);
}

/// A point in affine coordinates, or the point at infinity.
///
/// Points are only constructed on the curve: [`Affine::new`] checks the
/// equation.
///
/// ```
/// use arrabbiata_curve::{Affine, Projective};
/// use arrabbiata_pasta::Pallas;
/// let g = Projective::<Pallas>::generator().to_affine();
/// assert_eq!(Affine::new(g.x().unwrap(), g.y().unwrap()), Some(g));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine<C>
where
    C: Curve,
{
    coordinates: Option<(C::Base, C::Base)>,
}

impl<C> Affine<C>
where
    C: Curve,
{
    /// The point at infinity.
    ///
    /// ```
    /// use arrabbiata_curve::Affine;
    /// use arrabbiata_pasta::Pallas;
    /// assert!(Affine::<Pallas>::identity().is_identity());
    /// ```
    #[must_use]
    pub const fn identity() -> Self {
        Self { coordinates: None }
    }

    /// The point `(x, y)`, or `None` if it is not on the curve.
    ///
    /// ```
    /// use arrabbiata_curve::Affine;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::{Fp, Pallas};
    /// // (-1, 2): 2^2 = (-1)^3 + 5.
    /// assert!(Affine::<Pallas>::new(-Fp::ONE, Fp::from_u64(2)).is_some());
    /// assert!(Affine::<Pallas>::new(Fp::ONE, Fp::ONE).is_none());
    /// ```
    #[must_use]
    pub fn new(x: C::Base, y: C::Base) -> Option<Self> {
        if y * y == x * x * x + C::b() {
            Some(Self {
                coordinates: Some((x, y)),
            })
        } else {
            None
        }
    }

    /// Whether this is the point at infinity.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Pallas;
    /// assert!(!Projective::<Pallas>::generator().to_affine().is_identity());
    /// ```
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        self.coordinates.is_none()
    }

    /// The `x` coordinate, or `None` at infinity.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::{Fp, Pallas};
    /// assert_eq!(Projective::<Pallas>::generator().to_affine().x(), Some(-Fp::ONE));
    /// ```
    #[must_use]
    pub fn x(&self) -> Option<C::Base> {
        match self.coordinates {
            Some((x, _)) => Some(x),
            None => None,
        }
    }

    /// The `y` coordinate, or `None` at infinity.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::{Fp, Pallas};
    /// assert_eq!(Projective::<Pallas>::generator().to_affine().y(), Some(Fp::from_u64(2)));
    /// ```
    #[must_use]
    pub fn y(&self) -> Option<C::Base> {
        match self.coordinates {
            Some((_, y)) => Some(y),
            None => None,
        }
    }

    /// The projective point.
    ///
    /// ```
    /// use arrabbiata_curve::{Affine, Projective};
    /// use arrabbiata_pasta::Pallas;
    /// assert!(Affine::<Pallas>::identity().to_projective().is_identity());
    /// ```
    #[must_use]
    pub fn to_projective(&self) -> Projective<C> {
        match self.coordinates {
            Some((x, y)) => Projective {
                x,
                y,
                z: C::Base::ONE,
            },
            None => Projective::identity(),
        }
    }
}

/// A point in homogeneous projective coordinates `(X : Y : Z)`, standing
/// for `(X / Z, Y / Z)`; the identity is `(0 : 1 : 0)`.
///
/// Equality compares the points, not the representatives.
///
/// ```
/// use arrabbiata_curve::Projective;
/// use arrabbiata_pasta::Pallas;
/// let g = Projective::<Pallas>::generator();
/// assert_eq!(g + Projective::identity(), g);
/// assert!((g - g).is_identity());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Projective<C>
where
    C: Curve,
{
    x: C::Base,
    y: C::Base,
    z: C::Base,
}

impl<C> PartialEq for Projective<C>
where
    C: Curve,
{
    fn eq(&self, other: &Self) -> bool {
        // (X1 : Y1 : Z1) = (X2 : Y2 : Z2) iff the cross products agree.
        self.x * other.z == other.x * self.z && self.y * other.z == other.y * self.z
    }
}

impl<C> Eq for Projective<C> where C: Curve {}

impl<C> Projective<C>
where
    C: Curve,
{
    /// The identity `(0 : 1 : 0)`.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Vesta;
    /// assert!(Projective::<Vesta>::identity().is_identity());
    /// ```
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            x: C::Base::ZERO,
            y: C::Base::ONE,
            z: C::Base::ZERO,
        }
    }

    /// The generator of the curve.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Vesta;
    /// assert!(!Projective::<Vesta>::generator().is_identity());
    /// ```
    ///
    /// # Panics
    ///
    /// If the curve's generator is not on the curve, which is a bug in the
    /// curve definition and is tested for every curve.
    #[must_use]
    pub fn generator() -> Self {
        let (x, y) = C::generator();
        match Affine::<C>::new(x, y) {
            Some(g) => g.to_projective(),
            None => panic!("the generator of the curve is not on the curve"),
        }
    }

    /// Whether this is the identity.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Pallas;
    /// let g = Projective::<Pallas>::generator();
    /// assert!((g + (-g)).is_identity());
    /// ```
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    /// Whether the coordinates satisfy the projective equation
    /// `Y^2 Z = X^3 + b Z^3`.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Pallas;
    /// let g = Projective::<Pallas>::generator();
    /// assert!((g + g + g).is_on_curve());
    /// ```
    #[must_use]
    pub fn is_on_curve(&self) -> bool {
        let z2 = self.z * self.z;
        self.y * self.y * self.z == self.x * self.x * self.x + C::b() * z2 * self.z
    }

    /// The affine point.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Pallas;
    /// let g = Projective::<Pallas>::generator();
    /// assert_eq!((g + g).to_affine().to_projective(), g + g);
    /// ```
    #[must_use]
    pub fn to_affine(&self) -> Affine<C> {
        match self.z.inverse() {
            None => Affine::identity(),
            Some(z_inv) => Affine {
                coordinates: Some((self.x * z_inv, self.y * z_inv)),
            },
        }
    }

    /// `2 self`.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Pallas;
    /// let g = Projective::<Pallas>::generator();
    /// assert_eq!(g.double(), g + g);
    /// ```
    #[must_use]
    pub fn double(&self) -> Self {
        *self + *self
    }

    /// `[k] self` for a scalar given by its canonical limbs, by
    /// double-and-add from the most significant bit. Not constant time.
    ///
    /// ```
    /// use arrabbiata_curve::Projective;
    /// use arrabbiata_pasta::Pallas;
    /// let g = Projective::<Pallas>::generator();
    /// assert_eq!(g.mul_limbs(&[3]), g + g + g);
    /// ```
    #[must_use]
    pub fn mul_limbs(&self, scalar: &[u64]) -> Self {
        let mut acc = Self::identity();
        let mut i = scalar.len();
        while i > 0 {
            i -= 1;
            let mut bit: u32 = 64;
            while bit > 0 {
                bit -= 1;
                acc = acc.double();
                if (scalar[i] >> bit) & 1 == 1 {
                    acc = acc + *self;
                }
            }
        }
        acc
    }
}

impl<C> Add for Projective<C>
where
    C: Curve,
{
    type Output = Self;

    /// Renes-Costello-Batina 2016, Algorithm 7 (complete addition, `a = 0`).
    fn add(self, rhs: Self) -> Self {
        let b3 = C::b() + C::b() + C::b();
        let (x1, y1, z1) = (self.x, self.y, self.z);
        let (x2, y2, z2) = (rhs.x, rhs.y, rhs.z);
        let mut t0 = x1 * x2;
        let mut t1 = y1 * y2;
        let mut t2 = z1 * z2;
        let mut t3 = x1 + y1;
        let mut t4 = x2 + y2;
        t3 = t3 * t4;
        t4 = t0 + t1;
        t3 = t3 - t4;
        t4 = y1 + z1;
        let mut x3 = y2 + z2;
        t4 = t4 * x3;
        x3 = t1 + t2;
        t4 = t4 - x3;
        x3 = x1 + z1;
        let mut y3 = x2 + z2;
        x3 = x3 * y3;
        y3 = t0 + t2;
        y3 = x3 - y3;
        x3 = t0 + t0;
        t0 = x3 + t0;
        t2 = b3 * t2;
        let mut z3 = t1 + t2;
        t1 = t1 - t2;
        y3 = b3 * y3;
        x3 = t4 * y3;
        t2 = t3 * t1;
        x3 = t2 - x3;
        y3 = y3 * t0;
        t1 = t1 * z3;
        y3 = t1 + y3;
        t0 = t0 * t3;
        z3 = z3 * t4;
        z3 = z3 + t0;
        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }
}

impl<C> Neg for Projective<C>
where
    C: Curve,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
            z: self.z,
        }
    }
}

impl<C> Sub for Projective<C>
where
    C: Curve,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<C> Mul<C::Scalar> for Projective<C>
where
    C: Curve,
{
    type Output = Self;

    fn mul(self, scalar: C::Scalar) -> Self {
        let mut limbs = Vec::with_capacity(<C::Scalar as PrimeField>::LIMBS);
        let mut i = 0;
        while i < <C::Scalar as PrimeField>::LIMBS {
            limbs.push(scalar.limb(i));
            i += 1;
        }
        self.mul_limbs(&limbs)
    }
}

/// `sum_i scalars[i] points[i]`, the multi-scalar multiplication, computed
/// naively. Extra entries of the longer slice are ignored.
///
/// ```
/// use arrabbiata_curve::{Projective, msm};
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::{Fq, Pallas};
/// let g = Projective::<Pallas>::generator();
/// let h = g + g;
/// assert_eq!(msm(&[g, h], &[Fq::from_u64(3), Fq::from_u64(1)]), g * Fq::from_u64(5));
/// ```
#[must_use]
pub fn msm<C>(points: &[Projective<C>], scalars: &[C::Scalar]) -> Projective<C>
where
    C: Curve,
{
    let mut acc = Projective::identity();
    let mut i = 0;
    while i < points.len() && i < scalars.len() {
        acc = acc + points[i] * scalars[i];
        i += 1;
    }
    acc
}

/// Absorb a point into a transcript over a prime field `F`, through the
/// canonical limbs of its affine coordinates (each limb is below `2^64`,
/// hence below any field of more than 64 bits, so the encoding is
/// injective), preceded by a flag for the identity.
///
/// ```
/// use arrabbiata_curve::{Projective, absorb_point};
/// use arrabbiata_pasta::{Fq, Pallas};
/// use arrabbiata_transcript::{Transcript, toy_sponge};
/// let mut t = toy_sponge::<Fq>(0);
/// absorb_point(&Projective::<Pallas>::generator(), &mut t);
/// let _ = t.squeeze();
/// ```
pub fn absorb_point<C, F, T>(point: &Projective<C>, transcript: &mut T)
where
    C: Curve,
    C::Base: PrimeField,
    F: Field,
    T: Transcript<F>,
{
    let affine = point.to_affine();
    match affine.coordinates {
        None => transcript.absorb(F::ZERO),
        Some((x, y)) => {
            transcript.absorb(F::ONE);
            let mut i = 0;
            while i < <C::Base as PrimeField>::LIMBS {
                transcript.absorb(F::from_u64(x.limb(i)));
                transcript.absorb(F::from_u64(y.limb(i)));
                i += 1;
            }
        }
    }
}

impl<C> Label for Projective<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x7074;
}

impl<C> Absorb<C::Scalar> for Projective<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<C::Scalar>,
    {
        absorb_point(self, transcript);
    }
}

/// Deterministic points with unknown discrete logarithms relative to each
/// other: `x` coordinates are squeezed from `transcript`, and each is kept
/// if `x^3 + b` is a square (try-and-increment), with the root of even
/// canonical value as `y`.
///
/// Nobody knows a discrete-log relation between the points provided the
/// transcript behaves as a random oracle; with the toy sponge this is a
/// test fixture, not a setup.
///
/// ```
/// use arrabbiata_curve::points_from_transcript;
/// use arrabbiata_pasta::{Fp, Pallas};
/// use arrabbiata_transcript::toy_sponge;
/// let points = points_from_transcript::<Pallas, _>(&mut toy_sponge::<Fp>(0), 4);
/// assert_eq!(points.len(), 4);
/// assert!(points.iter().all(|p| p.is_on_curve() && !p.is_identity()));
/// ```
#[must_use]
pub fn points_from_transcript<C, T>(transcript: &mut T, count: usize) -> Vec<Projective<C>>
where
    C: Curve,
    C::Base: PrimeField + Sqrt,
    T: Transcript<C::Base>,
{
    let mut points = Vec::with_capacity(count);
    while points.len() < count {
        let x = transcript.squeeze();
        if let Some(y) = (x * x * x + C::b()).sqrt() {
            let y = if y.limb(0) & 1 == 0 { y } else { -y };
            if let Some(point) = Affine::<C>::new(x, y) {
                points.push(point.to_projective());
            }
        }
    }
    points
}
