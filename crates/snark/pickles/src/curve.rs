//! The curve in the circuit model: [`GroupModel`] and [`GroupMapModel`] for
//! [`Circuit`], on Kimchi's gates, so the group half written once over a
//! model of points ([`crate::group`]) is a circuit over the base field.
//!
//! - A point is its two coordinates, in cells.
//! - An addition is Kimchi's CompleteAdd row.
//! - A scaling by a scalar of the other field is Pickles' `scale_fast` of a
//!   shifted value `t` (`Shifted_value.Type1`, the scalar `2 t + 2^255 + 1`):
//!   a doubling, then VarBaseMul over the 255 bits of `t`, taken from its
//!   value and pinned to it by the gates' running sum.
//! - A scaling by the endomorphism of a 128-bit challenge is Pickles'
//!   `endo`: `2 (T + phi(T))`, then EndoMul over the challenge's bits, the
//!   running sum wired to the challenge.
//! - A scaling by the inverse of a challenge's scalar is witnessed, and the
//!   witness scaled back by the challenge is wired to the point.
//! - The group map's choice is two witnessed booleans picking a candidate
//!   and a witnessed ordinate on the curve.
//!
//! A scalar of the other field is held as Kimchi absorbs it ([`Shifted`]):
//! Type1, one cell `t` with `s = 2 t + 2^255 + 1`, when the scalar field is
//! the smaller (a Vesta proof in a circuit over `Fq`); Type2, the high bits
//! and the low bit of `t` with `s = t + 2^255`, when it is the larger (a
//! Pallas proof in a circuit over `Fp`), scaled as Pickles' `scale_fast2`.
//!
//! Deviation: Type2's high part is pinned to its bits but not to 254 of
//! them (Mina constrains the top bit to zero), so `2 high + low` may be a
//! non-canonical representative of the scalar, the same scalar modulo the
//! group order.

use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use arrabbiata_curve::{Affine as CurvePoint, Projective, glv::Glv};
use arrabbiata_field::{Field, FieldModel, PrimeField, Sqrt};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, measure};
use arrabbiata_kimchi::{
    constraints::COLUMNS,
    oracle::{ScalarChallenge, coordinates, scalar_elements},
    srs::GroupMapModel,
    transcript::Point,
};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    complete_add::CompleteAdd,
    endomul::{BITS_PER_ROW, EndoMulOf},
    varbasemul::VarBaseMulBelowOf,
};

use arrabbiata_poseidon::kimchi::KimchiPoseidon;

use crate::{
    group::{Constants, DeferredIn, GroupModel, ProofIn, PublicWord, group_half_in},
    model::{Affine, Circuit, load},
    statement::{WINDOW_BITS, Word},
};

/// The bits of a 128-bit challenge.
pub const CHALLENGE_BITS: usize = 128;

/// The bits of a shifted scalar of the Pasta curves.
pub const SCALAR_BITS: usize = 255;

/// [`EndoMulOf`] over a 128-bit challenge, with the row after its rows,
/// where its result lands: input `(T, P, x, n0)`, output `(P', n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoScale<F> {
    /// The curve's GLV `beta`.
    pub endo: F,
}

impl<F> TypedGadget<F> for EndoScale<F>
where
    F: PrimeField,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar, Scalar);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "pickles-endo-scale";
    const ROWS: usize = CHALLENGE_BITS / BITS_PER_ROW + 1;

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<F>,
    {
        env.call(&EndoMulOf::<F, CHALLENGE_BITS> { endo: self.endo }, input)
    }
}

/// The point of `(x, y)`, or `None` off the curve.
pub(crate) fn point_at<C: Glv>(x: C::Base, y: C::Base) -> Option<Projective<C>> {
    CurvePoint::<C>::new(x, y).map(|p| p.to_projective())
}

/// The scalar of the 128-bit challenge `chal` through the endomorphism.
fn endo_of<C>(chal: C::Base) -> C::Scalar
where
    C: Glv,
    C::Base: PrimeField,
{
    let low = C::Scalar::from_limbs(&[chal.limb(0), chal.limb(1)]).unwrap_or(C::Scalar::ZERO);
    ScalarChallenge(low).to_field(C::lambda())
}

impl<F, E> GroupMapModel<F> for Circuit<'_, F, E>
where
    F: PrimeField + Sqrt,
    E: Env<F>,
{
    /// Two booleans pick the first candidate whose `x^3 + b` is a square
    /// (the witness picks the native choice), and the ordinate is a witness
    /// on the curve, the native square root.
    fn first_on_curve(&mut self, xs: &[Self::Value; 3], b: F) -> (Self::Value, Self::Value) {
        let curve = |m: &mut Self, x: &Self::Value| {
            let square = m.square(x);
            let cube = m.mul(&square, x);
            m.add(&cube, &Affine::constant(b))
        };
        let f: Vec<Self::Value> = xs.iter().map(|x| curve(self, x)).collect();
        let is_square = |[v]: [F; 1]| {
            if v.sqrt().is_some() { F::ONE } else { F::ZERO }
        };
        let b1 = self.witness([&f[0]], is_square);
        let b2 = self.witness([&f[1]], is_square);
        for bit in [&b1, &b2] {
            let squared = self.mul(bit, bit);
            self.assert_equal(&squared, bit);
        }
        // x = b1 x1 + (1 - b1) (b2 x2 + (1 - b2) x3).
        let x2_minus_x3 = self.sub(&xs[1], &xs[2]);
        let b2_part = self.mul(&b2, &x2_minus_x3);
        let rest = self.add(&b2_part, &xs[2]);
        let x1_minus_rest = self.sub(&xs[0], &rest);
        let b1_part = self.mul(&b1, &x1_minus_rest);
        let x = self.add(&b1_part, &rest);
        let y = self.witness([&x], |[x]| (x.square() * x + b).sqrt().unwrap_or(F::ZERO));
        let y_squared = self.square(&y);
        let fx = curve(self, &x);
        self.assert_equal(&y_squared, &fx);
        (x, y)
    }
}

impl<C, E> GroupModel<C> for Circuit<'_, C::Base, E>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    E: Env<C::Base>,
{
    type Point = [E::Var; 2];
    type Scalar = Shifted<Affine<E::Var, C::Base>>;

    /// The coordinates of `p` as constants (the identity as `(0, 0)`, which
    /// the gates cannot add: the group half never embeds it).
    fn constant_point(&mut self, p: &Projective<C>) -> [E::Var; 2] {
        let (x, y) = coordinates(p);
        [
            self.cell(&Affine::constant(x)),
            self.cell(&Affine::constant(y)),
        ]
    }

    fn point_of(&mut self, x: &Self::Value, y: &Self::Value) -> [E::Var; 2] {
        [self.cell(x), self.cell(y)]
    }

    fn coordinates_of(&mut self, p: &[E::Var; 2]) -> Point<Self::Value> {
        Point {
            x: Affine::of(p[0].clone()),
            y: Affine::of(p[1].clone()),
        }
    }

    fn add_points(&mut self, a: &[E::Var; 2], b: &[E::Var; 2]) -> [E::Var; 2] {
        let [x, y, _] = self.env().call(
            &CompleteAdd,
            [a[0].clone(), a[1].clone(), b[0].clone(), b[1].clone()],
        );
        [x, y]
    }

    fn neg_point(&mut self, a: &[E::Var; 2]) -> [E::Var; 2] {
        let y = Affine::of(a[1].clone());
        let minus = self.neg(&y);
        [a[0].clone(), self.cell(&minus)]
    }

    /// Type1: `[2 t + 2^255 + 1] p`, VarBaseMul over the bits of `t` from
    /// `2 p`. Type2: the same over the high part `h`, which is
    /// `[2 h + 1 + 2^255] p`, then `p` taken off when the low bit is zero, so
    /// `[t + 2^255] p` (Pickles' `scale_fast2`).
    ///
    /// The top bit of the 255 is zero (a copy of the zero cell), so the bits
    /// are the canonical integer of the cell: below `2^255` a cell is the
    /// residue of `x` and possibly `x + m`, `m` the circuit's modulus, and the
    /// two act differently on a curve of another order. The cost is
    /// completeness for a value at or past `2^254`, about `2^-128` of them.
    ///
    /// # Panics
    ///
    /// If the scalars of `C` do not have 255 bits.
    fn scale_point(&mut self, p: &[E::Var; 2], s: &Self::Scalar) -> [E::Var; 2] {
        assert_eq!(
            <C::Scalar as PrimeField>::MODULUS_BITS as usize,
            SCALAR_BITS,
            "the shifted form is over 255 bits"
        );
        let double = GroupModel::<C>::add_points(self, p, p);
        let t = self.cell(&s.high);
        let n0 = self.cell(&Affine::constant(C::Base::ZERO));
        let (acc, n) = self.env().call(
            &VarBaseMulBelowOf::<SCALAR_BITS>,
            (p.clone(), double, t.clone(), n0),
        );
        self.env().connect(&n, &t);
        let Some(low) = &s.low else {
            return acc;
        };
        let squared = self.mul(low, low);
        self.assert_equal(&squared, low);
        let minus = GroupModel::<C>::neg_point(self, p);
        let less = GroupModel::<C>::add_points(self, &acc, &minus);
        // low h + (1 - low) (h - p), coordinate by coordinate.
        let pick = |m: &mut Self, a: &E::Var, b: &E::Var| {
            let (a, b) = (Affine::of(a.clone()), Affine::of(b.clone()));
            let difference = m.sub(&a, &b);
            let scaled = m.mul(low, &difference);
            let chosen = m.add(&scaled, &b);
            m.cell(&chosen)
        };
        [pick(self, &acc[0], &less[0]), pick(self, &acc[1], &less[1])]
    }

    fn endo_scale(&mut self, p: &[E::Var; 2], chal: &Self::Value) -> [E::Var; 2] {
        let beta = C::beta();
        let x = Affine::of(p[0].clone());
        let phi_x = self.scale(&x, beta);
        let phi = [self.cell(&phi_x), p[1].clone()];
        let sum = GroupModel::<C>::add_points(self, p, &phi);
        let start = GroupModel::<C>::add_points(self, &sum, &sum);
        let c = self.cell(chal);
        let n0 = self.cell(&Affine::constant(C::Base::ZERO));
        let (acc, n) = self
            .env()
            .call(&EndoScale { endo: beta }, (p.clone(), start, c.clone(), n0));
        self.env().connect(&n, &c);
        acc
    }

    fn endo_inv_scale(&mut self, p: &[E::Var; 2], chal: &Self::Value) -> [E::Var; 2] {
        let (x, y) = (Affine::of(p[0].clone()), Affine::of(p[1].clone()));
        let coordinate = |k: usize| {
            move |[px, py, c]: [C::Base; 3]| {
                let q = point_at::<C>(px, py)
                    .zip(endo_of::<C>(c).inverse())
                    .map(|(p, inv)| coordinates(&(p * inv)));
                q.map_or(C::Base::ZERO, |(qx, qy)| if k == 0 { qx } else { qy })
            }
        };
        let qx = self.witness([&x, &y, chal], coordinate(0));
        let qy = self.witness([&x, &y, chal], coordinate(1));
        let q = [self.cell(&qx), self.cell(&qy)];
        let back = GroupModel::<C>::endo_scale(self, &q, chal);
        self.env().connect(&back[0], &p[0]);
        self.env().connect(&back[1], &p[1]);
        q
    }

    /// The cells of the shifted value: what Kimchi absorbs for it.
    fn shifted_elements(&mut self, s: &Self::Scalar) -> Vec<Self::Value> {
        s.elements()
    }

    /// The word split as `w = 2 high + low` (asserted), `high` bounded below
    /// `2^(n - 1)` for a word of bound `n`, so that `2 high + low` is the
    /// integer of `w` and not `w` plus the modulus; then Type2's action,
    /// `[2 high + low + 2^255] p`.
    fn act_shifted(&mut self, p: &Projective<C>, w: &PublicWord<Self::Value>) -> [E::Var; 2] {
        let high = self.witness([&w.value], |[v]| Word::of(v).halve().0.reduce());
        let low = self.witness([&w.value], |[v]| {
            if Word::of(v).halve().1 {
                C::Base::ONE
            } else {
                C::Base::ZERO
            }
        });
        let two = C::Base::ONE + C::Base::ONE;
        let twice = self.scale(&high, two);
        let sum = self.add(&twice, &low);
        self.assert_equal(&sum, &w.value);
        if w.bound == 1 {
            self.assert_equal(&high, &Affine::constant(C::Base::ZERO));
        } else {
            self.assert_below(&high, w.bound - 1);
        }
        let base = GroupModel::<C>::constant_point(self, p);
        let scalar = Shifted {
            high,
            low: Some(low),
        };
        GroupModel::<C>::scale_point(self, &base, &scalar)
    }
}

/// A scalar of the other field as a circuit holds it, in the cells Kimchi
/// absorbs for it: Type1, `high` the shifted value `t` and no `low`;
/// Type2, `high` and `low` the high bits and the low bit of `t`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shifted<V> {
    /// `t`, or its high bits.
    pub high: V,
    /// The low bit of `t`, for Type2.
    pub low: Option<V>,
}

impl<V: Clone> Shifted<V> {
    /// The scalar of the cells `values` (one for Type1, two for Type2).
    ///
    /// # Panics
    ///
    /// If `values` has neither one nor two cells.
    #[must_use]
    pub fn of(values: &[V]) -> Self {
        match values {
            [t] => Self {
                high: t.clone(),
                low: None,
            },
            [high, low] => Self {
                high: high.clone(),
                low: Some(low.clone()),
            },
            _ => panic!("a shifted scalar is one or two cells"),
        }
    }

    /// Its cells, in order.
    #[must_use]
    pub fn elements(&self) -> Vec<V> {
        let mut out = vec![self.high.clone()];
        out.extend(self.low.clone());
        out
    }
}

/// The cells a scalar `s` of `C` is held in by a circuit over its base
/// field: those Kimchi absorbs for its shifted value, one (Type1) or two
/// (Type2).
#[must_use]
pub fn held<C>(s: C::Scalar) -> Vec<C::Base>
where
    C: Glv,
    C::Base: PrimeField,
{
    scalar_elements::<C>(arrabbiata_ipa::generic::shift_scalar::<C>(s))
}

/// The cells of one scalar of `C`: one or two.
#[must_use]
pub fn scalar_width<C>() -> usize
where
    C: Glv,
    C::Base: PrimeField,
{
    scalar_elements::<C>(C::Scalar::ZERO).len()
}

/// The shape of a proof the group half circuit reads: how many of each
/// thing its flat input holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The previous accumulators.
    pub previous: usize,
    /// The chunks of a commitment.
    pub chunks: usize,
    /// The chunks of the quotient's commitment.
    pub quotient_chunks: usize,
    /// The IPA's rounds.
    pub rounds: usize,
    /// The public input's length.
    pub public: usize,
    /// The cells of a scalar ([`scalar_width`]).
    pub width: usize,
}

impl Shape {
    /// The points of the flat input.
    const fn points(&self) -> usize {
        self.previous
            + COLUMNS * self.chunks
            + self.chunks
            + self.quotient_chunks
            + 2 * self.rounds
            + 2
    }

    /// The size of the flat input ([`flatten`]): points as two
    /// coordinates, scalars as their shifted value.
    #[must_use]
    pub const fn inputs(&self) -> usize {
        2 * self.points() + self.width * (2 + 5) + self.public + 1
    }

    /// The size of the output: the five draws, the prechallenges and `sg`.
    #[must_use]
    pub const fn outputs(&self) -> usize {
        5 + self.rounds + 2
    }
}

/// The flat input of the group half circuit, in its order: the previous
/// `sg`s, the witness columns', `z`'s and `t`'s chunks, the rounds' `L`
/// and `R`, `delta`, `sg` (points as `x`, `y`), then `z1`, `z2`, the public
/// input, `perm`, `zeta^m`, `zeta^n`, `xi`, the combined inner product and
/// `b`, each value as `value` reads it.
pub fn flatten<P, S, V>(
    proof: &ProofIn<P, S>,
    deferred: &DeferredIn<V, S>,
    point: impl Fn(&P) -> [V; 2],
    scalar: impl Fn(&S) -> Vec<V>,
) -> Vec<V>
where
    V: Clone,
{
    let mut points: Vec<&P> = proof.previous.iter().collect();
    for w in &proof.witness {
        points.extend(w);
    }
    points.extend(&proof.z);
    points.extend(&proof.t);
    for (l, r) in &proof.rounds {
        points.push(l);
        points.push(r);
    }
    points.push(&proof.delta);
    points.push(&proof.sg);
    let mut out: Vec<V> = points.into_iter().flat_map(&point).collect();
    out.extend(scalar(&proof.z1));
    out.extend(scalar(&proof.z2));
    out.extend(deferred.public.iter().map(|w| w.value.clone()));
    out.extend(scalar(&deferred.perm));
    out.extend(scalar(&deferred.zeta_to_srs_length));
    out.extend(scalar(&deferred.zeta_to_domain_size));
    out.push(deferred.xi.clone());
    out.extend(scalar(&deferred.combined_inner_product));
    out.extend(scalar(&deferred.b));
    out
}

/// The values of a flat input, read in order.
struct Reader<'a, V> {
    flat: &'a [V],
    at: usize,
}

impl<V: Clone> Reader<'_, V> {
    /// The next value.
    fn value(&mut self) -> V {
        let v = self.flat[self.at].clone();
        self.at += 1;
        v
    }

    /// The next point.
    fn point(&mut self) -> [V; 2] {
        [self.value(), self.value()]
    }

    /// The next `n` points.
    fn points(&mut self, n: usize) -> Vec<[V; 2]> {
        (0..n).map(|_| self.point()).collect()
    }

    /// The next scalar, of `width` cells.
    fn scalar(&mut self, width: usize) -> Shifted<V> {
        let cells: Vec<V> = (0..width).map(|_| self.value()).collect();
        Shifted::of(&cells)
    }
}

/// The proof and deferred values of the flat input `flat` of `shape`, the
/// inverse of [`flatten`]: points of two cells, scalars of `shape.width`.
#[allow(clippy::type_complexity)]
fn unflatten<V: Clone>(
    shape: &Shape,
    flat: &[V],
) -> (ProofIn<[V; 2], Shifted<V>>, DeferredIn<V, Shifted<V>>) {
    let mut r = Reader { flat, at: 0 };
    let w = shape.width;
    let previous = r.points(shape.previous);
    let witness: [Vec<[V; 2]>; COLUMNS] = core::array::from_fn(|_| r.points(shape.chunks));
    let z = r.points(shape.chunks);
    let t = r.points(shape.quotient_chunks);
    let rounds = (0..shape.rounds).map(|_| (r.point(), r.point())).collect();
    let delta = r.point();
    let sg = r.point();
    let z1 = r.scalar(w);
    let z2 = r.scalar(w);
    let proof = ProofIn {
        previous,
        witness,
        z,
        t,
        rounds,
        delta,
        z1,
        z2,
        sg,
    };
    let deferred = DeferredIn {
        public: (0..shape.public)
            .map(|_| PublicWord {
                value: r.value(),
                bound: WINDOW_BITS,
            })
            .collect(),
        perm: r.scalar(w),
        zeta_to_srs_length: r.scalar(w),
        zeta_to_domain_size: r.scalar(w),
        xi: r.value(),
        combined_inner_product: r.scalar(w),
        b: r.scalar(w),
    };
    (proof, deferred)
}

/// The group half as a circuit over the base field of `C`: its input the
/// [`flatten`] of a proof of `shape`, `N` values; it asserts the IPA's two
/// sides equal, and outputs the draws (`alpha`, `beta`, `gamma`, `zeta`,
/// the digest), the prechallenges and `sg`, `M` values. Its rows depend on
/// the index ([`TypedGadget::rows`]).
#[derive(Clone, Debug)]
pub struct GroupHalf<'a, C, const N: usize, const M: usize>
where
    C: Glv,
{
    constants: Constants<C>,
    poseidon: &'a KimchiPoseidon<C::Base>,
    shape: Shape,
    bounds: Vec<u32>,
}

/// The input or output size does not match the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeError {
    /// The input size the shape needs.
    pub inputs: usize,
    /// The output size the shape needs.
    pub outputs: usize,
}

impl<'a, C, const N: usize, const M: usize> GroupHalf<'a, C, N, M>
where
    C: Glv,
{
    /// The circuit for `constants` and proofs of `shape`, with Kimchi's
    /// `poseidon` for the base sponge, the public input's words of the
    /// bounds `bounds` (see [`PublicWord`]).
    ///
    /// # Errors
    ///
    /// [`SizeError`] when `N` or `M` is not the shape's, or the bounds are
    /// not one per public word, each of 1 to 254 bits.
    pub fn new(
        constants: Constants<C>,
        poseidon: &'a KimchiPoseidon<C::Base>,
        shape: Shape,
        bounds: Vec<u32>,
    ) -> Result<Self, SizeError> {
        let in_window = bounds.iter().all(|b| (1..=WINDOW_BITS).contains(b));
        if shape.inputs() != N || shape.outputs() != M || bounds.len() != shape.public || !in_window
        {
            return Err(SizeError {
                inputs: shape.inputs(),
                outputs: shape.outputs(),
            });
        }
        Ok(Self {
            constants,
            poseidon,
            shape,
            bounds,
        })
    }
}

impl<C, const N: usize, const M: usize> TypedGadget<C::Base> for GroupHalf<'_, C, N, M>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
{
    type Input = Array<Scalar, N>;
    type Output = Array<Scalar, M>;
    const NAME: &'static str = "pickles-group-half";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> Of<Self::Output, E::Var>
    where
        E: Env<C::Base>,
    {
        let cells = load(env, &input);
        let (cell_proof, cell_deferred) = unflatten(&self.shape, &cells);
        let value = |v: &E::Var| Affine::of(v.clone());
        let scalar = |s: &Shifted<E::Var>| Shifted {
            high: value(&s.high),
            low: s.low.as_ref().map(value),
        };
        let proof = ProofIn {
            previous: cell_proof.previous,
            witness: cell_proof.witness,
            z: cell_proof.z,
            t: cell_proof.t,
            rounds: cell_proof.rounds,
            delta: cell_proof.delta,
            z1: scalar(&cell_proof.z1),
            z2: scalar(&cell_proof.z2),
            sg: cell_proof.sg,
        };
        let deferred = DeferredIn {
            public: cell_deferred
                .public
                .iter()
                .zip(&self.bounds)
                .map(|(w, bound)| PublicWord {
                    value: value(&w.value),
                    bound: *bound,
                })
                .collect(),
            perm: scalar(&cell_deferred.perm),
            zeta_to_srs_length: scalar(&cell_deferred.zeta_to_srs_length),
            zeta_to_domain_size: scalar(&cell_deferred.zeta_to_domain_size),
            xi: value(&cell_deferred.xi),
            combined_inner_product: scalar(&cell_deferred.combined_inner_product),
            b: scalar(&cell_deferred.b),
        };
        let model = RefCell::new(Circuit::new(env));
        let out = group_half_in::<C, _>(&model, self.poseidon, &self.constants, &proof, &deferred);
        let mut m = model.into_inner();
        m.env().connect(&out.left[0], &out.right[0]);
        m.env().connect(&out.left[1], &out.right[1]);
        let d = &out.drawn;
        let mut outputs: Vec<E::Var> = [&d.alpha, &d.beta, &d.gamma, &d.zeta, &d.digest]
            .into_iter()
            .map(|v| m.cell(v))
            .collect();
        outputs.extend(out.prechallenges.iter().map(|v| m.cell(v)));
        outputs.extend(out.sg);
        let mut next = outputs.into_iter();
        core::array::from_fn(|_| match next.next() {
            Some(v) => v,
            None => unreachable!("the output has the shape's size"),
        })
    }
}
