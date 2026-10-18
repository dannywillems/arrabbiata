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
//! Deviation: only the shifted form of a scalar field smaller than the base
//! field (Type1, a Vesta proof in a circuit over Vesta's base field) is
//! held; Type2, for the other side of the cycle, is not written yet.

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
    varbasemul::VarBaseMulOf,
};

use arrabbiata_poseidon::kimchi::KimchiPoseidon;

use crate::{
    group::{Constants, DeferredIn, GroupModel, ProofIn, group_half_in},
    model::{Affine, Circuit, load},
};

/// The bits of a 128-bit challenge.
pub const CHALLENGE_BITS: usize = 128;

/// The bits of a shifted scalar of the Pasta curves.
pub const SCALAR_BITS: usize = 255;

/// [`EndoMulOf`] over a 128-bit challenge, with the row after its rows,
/// where its result lands: input `(T, P, x)`, output `(P', n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndoScale<F> {
    /// The curve's GLV `beta`.
    pub endo: F,
}

impl<F> TypedGadget<F> for EndoScale<F>
where
    F: PrimeField,
{
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Scalar);
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
fn point_at<C: Glv>(x: C::Base, y: C::Base) -> Option<Projective<C>> {
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
    type Scalar = Affine<E::Var, C::Base>;

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
        let t = self.cell(s);
        let (acc, n) = self
            .env()
            .call(&VarBaseMulOf::<SCALAR_BITS>, (p.clone(), double, t.clone()));
        self.env().connect(&n, &t);
        acc
    }

    fn endo_scale(&mut self, p: &[E::Var; 2], chal: &Self::Value) -> [E::Var; 2] {
        let beta = C::beta();
        let x = Affine::of(p[0].clone());
        let phi_x = self.scale(&x, beta);
        let phi = [self.cell(&phi_x), p[1].clone()];
        let sum = GroupModel::<C>::add_points(self, p, &phi);
        let start = GroupModel::<C>::add_points(self, &sum, &sum);
        let c = self.cell(chal);
        let (acc, n) = self
            .env()
            .call(&EndoScale { endo: beta }, (p.clone(), start, c.clone()));
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

    /// The shifted value itself, as a scalar smaller than the base field
    /// is absorbed.
    ///
    /// # Panics
    ///
    /// If the scalar field of `C` is not the smaller (Type2, not written).
    fn shifted_elements(&mut self, s: &Self::Scalar) -> Vec<Self::Value> {
        assert_eq!(
            scalar_elements::<C>(C::Scalar::ZERO).len(),
            1,
            "only the shifted form of a smaller scalar field is held"
        );
        vec![s.clone()]
    }
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
        2 * self.points() + 2 + self.public + 6
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
    scalar: impl Fn(&S) -> V,
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
    out.push(scalar(&proof.z1));
    out.push(scalar(&proof.z2));
    out.extend(deferred.public.iter().map(&scalar));
    out.push(scalar(&deferred.perm));
    out.push(scalar(&deferred.zeta_to_srs_length));
    out.push(scalar(&deferred.zeta_to_domain_size));
    out.push(deferred.xi.clone());
    out.push(scalar(&deferred.combined_inner_product));
    out.push(scalar(&deferred.b));
    out
}

/// The proof and deferred values of the flat input `flat` of `shape`, the
/// inverse of [`flatten`]: points of two cells, scalars and values of one.
fn unflatten<V: Clone>(shape: &Shape, flat: &[V]) -> (ProofIn<[V; 2], V>, DeferredIn<V, V>) {
    let mut next = flat.iter().cloned();
    let mut take = || match next.next() {
        Some(v) => v,
        None => unreachable!("the input has the shape's size"),
    };
    let mut point = || [take(), take()];
    let mut points = |n: usize| (0..n).map(|_| point()).collect::<Vec<_>>();
    let previous = points(shape.previous);
    let witness: [Vec<[V; 2]>; COLUMNS] = core::array::from_fn(|_| points(shape.chunks));
    let z = points(shape.chunks);
    let t = points(shape.quotient_chunks);
    let rounds = (0..shape.rounds)
        .map(|_| {
            let mut two = points(2);
            let r = two.pop();
            let l = two.pop();
            match (l, r) {
                (Some(l), Some(r)) => (l, r),
                _ => unreachable!("two points"),
            }
        })
        .collect();
    let mut last = points(2);
    let (sg, delta) = match (last.pop(), last.pop()) {
        (Some(sg), Some(delta)) => (sg, delta),
        _ => unreachable!("two points"),
    };
    let z1 = take();
    let z2 = take();
    let public = (0..shape.public).map(|_| take()).collect();
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
        public,
        perm: take(),
        zeta_to_srs_length: take(),
        zeta_to_domain_size: take(),
        xi: take(),
        combined_inner_product: take(),
        b: take(),
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
    /// `poseidon` for the base sponge.
    ///
    /// # Errors
    ///
    /// [`SizeError`] when `N` or `M` is not the shape's.
    pub fn new(
        constants: Constants<C>,
        poseidon: &'a KimchiPoseidon<C::Base>,
        shape: Shape,
    ) -> Result<Self, SizeError> {
        if shape.inputs() != N || shape.outputs() != M {
            return Err(SizeError {
                inputs: shape.inputs(),
                outputs: shape.outputs(),
            });
        }
        Ok(Self {
            constants,
            poseidon,
            shape,
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
        let (points, values) = {
            let p: Vec<E::Var> = cells.clone();
            let v: Vec<Affine<E::Var, C::Base>> = cells.into_iter().map(Affine::of).collect();
            (p, v)
        };
        let (cell_proof, _) = unflatten(&self.shape, &points);
        let (value_proof, deferred) = unflatten(&self.shape, &values);
        let proof = ProofIn {
            previous: cell_proof.previous,
            witness: cell_proof.witness,
            z: cell_proof.z,
            t: cell_proof.t,
            rounds: cell_proof.rounds,
            delta: cell_proof.delta,
            z1: value_proof.z1,
            z2: value_proof.z2,
            sg: cell_proof.sg,
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
