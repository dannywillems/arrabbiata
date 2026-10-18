//! The group half of Kimchi's verifier, written once over a model of the
//! base field and of the curve's points: Pickles' `incrementally_verify_proof`.
//!
//! The group half absorbs the proof's commitments into the base sponge,
//! draws `beta`, `gamma`, `alpha`, `zeta`, rebuilds the public commitment
//! `x_hat` and `ft`'s commitment from deferred scalars, combines every
//! commitment by `xi`, and checks the IPA opening with the deferred
//! combined inner product and folded evaluation `b`. It takes those scalars
//! on trust; the scalar half checks them in the other field.
//!
//! It runs in any [`GroupModel`]: natively (points of the curve, scalars of
//! its field) it is the native group half, which the tests compare with
//! `arrabbiata_kimchi::deferred::group_half`; in a circuit over the base
//! field, a point is its coordinates and a scalar of the other field is
//! held in Pickles' shifted form. It does not assert anything: it returns
//! what must be equal ([`Outputs`]), and the native caller compares while a
//! circuit constrains.

use alloc::vec::Vec;
use core::cell::RefCell;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, Native, PrimeField, Sqrt, TwoAdicField};
use arrabbiata_ipa::generic::shift_scalar;
use arrabbiata_kimchi::{
    constraints::COLUMNS,
    oracle::{ChallengeModel, ScalarChallenge, coordinates, scalar_elements},
    srs::{GroupMap, GroupMapModel, PolyComm, Srs, num_chunks},
    transcript::Point,
    verifier::{PiopChallenges, PiopInputs, base_transcript_in, fixed_commitments},
    verifier_index::VerifierIndex,
};
use arrabbiata_poly::Domain;
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, WIDTH};
use arrabbiata_transcript::PermutationIn;

/// Point arithmetic in a model of the base field of `C`: what the group
/// half needs beyond field arithmetic, the group map's choice and the
/// 128-bit challenges.
///
/// - [`GroupModel::Point`]: natively a point; in a circuit its coordinates.
/// - [`GroupModel::Scalar`]: a scalar of `C`, foreign to the base field:
///   natively the scalar; in a circuit Pickles' shifted value.
/// - a 128-bit challenge is a value of the model (it fits either field),
///   scaled by through the endomorphism ([`GroupModel::endo_scale`]), as
///   EndoMul does.
pub trait GroupModel<C>: GroupMapModel<C::Base> + ChallengeModel<C::Base>
where
    C: Glv,
{
    /// A point.
    type Point: Clone;

    /// A scalar of `C`.
    type Scalar: Clone;

    /// The constant point `p`.
    fn constant_point(&mut self, p: &Projective<C>) -> Self::Point;

    /// The point of coordinates `(x, y)`, which must be on the curve.
    fn point_of(&mut self, x: &Self::Value, y: &Self::Value) -> Self::Point;

    /// The coordinates of `p` as the base sponge absorbs them (the identity
    /// as `(0, 0)`).
    fn coordinates_of(&mut self, p: &Self::Point) -> Point<Self::Value>;

    /// `a + b`.
    fn add_points(&mut self, a: &Self::Point, b: &Self::Point) -> Self::Point;

    /// `-a`.
    fn neg_point(&mut self, a: &Self::Point) -> Self::Point;

    /// `[s] p`.
    fn scale_point(&mut self, p: &Self::Point, s: &Self::Scalar) -> Self::Point;

    /// `[x] p`, for `x` the scalar of the 128-bit challenge `chal` through
    /// the endomorphism (`ScalarChallenge::to_field`).
    fn endo_scale(&mut self, p: &Self::Point, chal: &Self::Value) -> Self::Point;

    /// `[x^(-1)] p`, for `x` as in [`GroupModel::endo_scale`]: a circuit
    /// witnesses the result and checks its endo-scaling is `p`.
    fn endo_inv_scale(&mut self, p: &Self::Point, chal: &Self::Value) -> Self::Point;

    /// The base-field elements the base sponge absorbs for the IPA's bound
    /// value `s`: those of its shifted form (Kimchi's `ShiftedValue`
    /// through `absorb_fr`), which is how a circuit holds it.
    fn shifted_elements(&mut self, s: &Self::Scalar) -> Vec<Self::Value>;

    /// `a - b`.
    fn sub_points(&mut self, a: &Self::Point, b: &Self::Point) -> Self::Point {
        let minus = self.neg_point(b);
        self.add_points(a, &minus)
    }
}

/// A 128-bit value of the base field, as the scalar it is.
fn as_scalar<C>(x: &C::Base) -> C::Scalar
where
    C: Glv,
    C::Base: PrimeField,
{
    match C::Scalar::from_limbs(&[x.limb(0), x.limb(1)]) {
        Some(s) => s,
        None => unreachable!("128 bits fit in the scalar field"),
    }
}

/// The scalar of the 128-bit challenge `chal` through the endomorphism.
fn endo_of<C>(chal: &C::Base) -> C::Scalar
where
    C: Glv,
    C::Base: PrimeField,
{
    ScalarChallenge(as_scalar::<C>(chal)).to_field(C::lambda())
}

impl<C> GroupModel<C> for Native
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
{
    type Point = Projective<C>;
    type Scalar = C::Scalar;

    fn constant_point(&mut self, p: &Projective<C>) -> Projective<C> {
        *p
    }

    /// # Panics
    ///
    /// If `(x, y)` is not on the curve.
    fn point_of(&mut self, x: &C::Base, y: &C::Base) -> Projective<C> {
        match arrabbiata_curve::Affine::<C>::new(*x, *y) {
            Some(p) => p.to_projective(),
            None => panic!("the point is not on the curve"),
        }
    }

    fn coordinates_of(&mut self, p: &Projective<C>) -> Point<C::Base> {
        let (x, y) = coordinates(p);
        Point { x, y }
    }

    fn add_points(&mut self, a: &Projective<C>, b: &Projective<C>) -> Projective<C> {
        *a + *b
    }

    fn neg_point(&mut self, a: &Projective<C>) -> Projective<C> {
        -*a
    }

    fn scale_point(&mut self, p: &Projective<C>, s: &C::Scalar) -> Projective<C> {
        *p * *s
    }

    fn endo_scale(&mut self, p: &Projective<C>, chal: &C::Base) -> Projective<C> {
        *p * endo_of::<C>(chal)
    }

    /// # Panics
    ///
    /// If the challenge's scalar is zero.
    fn endo_inv_scale(&mut self, p: &Projective<C>, chal: &C::Base) -> Projective<C> {
        match endo_of::<C>(chal).inverse() {
            Some(inv) => *p * inv,
            None => panic!("a zero round challenge"),
        }
    }

    fn shifted_elements(&mut self, s: &C::Scalar) -> Vec<C::Base> {
        scalar_elements::<C>(shift_scalar::<C>(*s))
    }
}

/// What the group half takes from the verifier index and the SRS: native
/// constants, embedded into the model as constants.
#[derive(Clone, Debug)]
pub struct Constants<C>
where
    C: Glv,
{
    /// The verifier index's digest.
    pub index_digest: C::Base,
    /// The commitments of the public rows' Lagrange polynomials.
    pub lagrange: Vec<PolyComm<C>>,
    /// The chunks of a commitment.
    pub chunks: usize,
    /// The blinding point `h`.
    pub h: Projective<C>,
    /// The last sigma's commitment, which `ft`'s scales by `perm`.
    pub sigma_last: PolyComm<C>,
    /// The index's commitments before the witness columns' in the opening.
    pub before_witness: Vec<PolyComm<C>>,
    /// The index's commitments after them.
    pub after_witness: Vec<PolyComm<C>>,
    /// The group map of the IPA's value base.
    pub group_map: GroupMap<C>,
}

impl<C> Constants<C>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    C::Scalar: TwoAdicField,
{
    /// The constants of `index` on `srs`, its digest by `base_sponge`; none
    /// for an index with lookups, which the group half does not cover yet.
    ///
    /// # Panics
    ///
    /// If the index's domain is not one of the field's.
    pub fn new<S>(index: &VerifierIndex<C>, srs: &Srs<C>, base_sponge: &S) -> Option<Self>
    where
        S: arrabbiata_transcript::Transcript<C::Base> + Clone,
    {
        if index.lookup_index.is_some() {
            return None;
        }
        let n = index.domain_size;
        let chunks = num_chunks(n, index.max_poly_size);
        let Some(domain) = Domain::<C::Scalar>::containing(n) else {
            panic!("the index's domain is not one of the field's")
        };
        let lagrange = (0..index.public)
            .map(|row| {
                let mut e = alloc::vec![C::Scalar::ZERO; n];
                e[row] = C::Scalar::ONE;
                match domain.interpolate(&e) {
                    Some(p) => srs.commit_non_hiding(p.coefficients(), chunks),
                    None => unreachable!("one value per row"),
                }
            })
            .collect();
        let fixed = fixed_commitments(index);
        Some(Self {
            index_digest: index.digest(base_sponge.clone()),
            lagrange,
            chunks,
            h: srs.h,
            sigma_last: index.sigma_comm[arrabbiata_kimchi::permutation::PERMUTS - 1].clone(),
            before_witness: fixed.before_witness,
            after_witness: fixed.after_witness,
            group_map: GroupMap::setup(),
        })
    }
}

/// The proof's part the group half reads, in the model: commitments as
/// lists of chunks, the IPA's rounds and final step.
#[derive(Clone, Debug)]
pub struct ProofIn<P, S> {
    /// The previous accumulators' `sg`.
    pub previous: Vec<P>,
    /// The witness columns' commitments.
    pub witness: [Vec<P>; COLUMNS],
    /// The permutation's commitment.
    pub z: Vec<P>,
    /// The quotient's commitment.
    pub t: Vec<P>,
    /// The IPA's rounds, `(L, R)`.
    pub rounds: Vec<(P, P)>,
    /// The IPA's `delta`.
    pub delta: P,
    /// The IPA's `z1`.
    pub z1: S,
    /// The IPA's `z2`.
    pub z2: S,
    /// The IPA's `sg`.
    pub sg: P,
}

/// The deferred values the group half takes on trust, in the model.
#[derive(Clone, Debug)]
pub struct DeferredIn<V, S> {
    /// The public input, its scalars.
    pub public: Vec<S>,
    /// `perm`.
    pub perm: S,
    /// `zeta^m`.
    pub zeta_to_srs_length: S,
    /// `zeta^n`.
    pub zeta_to_domain_size: S,
    /// The 128-bit challenge `xi` comes from.
    pub xi: V,
    /// The combined inner product.
    pub combined_inner_product: S,
    /// The folded evaluation vector, `b`.
    pub b: S,
}

/// What the group half computes: the draws to compare with the deferred
/// ones, and the two sides of the IPA's final equation, equal exactly when
/// the opening holds (Pickles asserts them equal rather than subtracting,
/// which would need the point at infinity).
#[derive(Clone, Debug)]
pub struct Outputs<V, P> {
    /// The base transcript's draws (128-bit challenges unmapped).
    pub drawn: PiopChallenges<V>,
    /// The IPA's round challenges, unmapped (Pickles' prechallenges).
    pub prechallenges: Vec<V>,
    /// `[c] Q + delta`.
    pub left: P,
    /// `[z1] (sg + [b] U) + [z2] H`.
    pub right: P,
    /// The IPA's `sg`, the new accumulator's commitment.
    pub sg: P,
}

/// `sum_i y^i c_i` by Horner, `y` a scalar of `C`.
fn chunked<C, M>(m: &mut M, chunks: &[M::Point], y: &M::Scalar) -> M::Point
where
    C: Glv,
    M: GroupModel<C>,
{
    let Some((last, rest)) = chunks.split_last() else {
        return m.constant_point(&Projective::identity());
    };
    let mut acc = last.clone();
    for c in rest.iter().rev() {
        let scaled = m.scale_point(&acc, y);
        acc = m.add_points(&scaled, c);
    }
    acc
}

/// A constant commitment's chunks in the model.
fn constant<C, M>(m: &mut M, c: &PolyComm<C>) -> Vec<M::Point>
where
    C: Glv,
    M: GroupModel<C>,
{
    c.chunks.iter().map(|p| m.constant_point(p)).collect()
}

/// The coordinates of points, for the base sponge.
fn held<C, M>(m: &mut M, points: &[M::Point]) -> Vec<Point<M::Value>>
where
    C: Glv,
    M: GroupModel<C>,
{
    points.iter().map(|p| m.coordinates_of(p)).collect()
}

/// The commitments in the opening's order, combined by `xi` (Horner, by
/// the endomorphism). A constant commitment that is the identity (a column
/// of the index the circuit leaves zero) is the unit of the sum: its step
/// scales without adding, so no gate sees the identity, which has no
/// coordinates.
fn combined<C, M>(
    m: &mut M,
    constants: &Constants<C>,
    proof: &ProofIn<M::Point, M::Scalar>,
    public: Vec<M::Point>,
    ft: M::Point,
    xi: &M::Value,
) -> M::Point
where
    C: Glv,
    M: GroupModel<C>,
{
    let mut comms: Vec<Option<M::Point>> = proof.previous.iter().cloned().map(Some).collect();
    comms.extend(public.into_iter().map(Some));
    comms.push(Some(ft));
    comms.extend(proof.z.iter().cloned().map(Some));
    let fixed = |m: &mut M, c: &PolyComm<C>, comms: &mut Vec<Option<M::Point>>| {
        for p in &c.chunks {
            comms.push((!p.is_identity()).then(|| m.constant_point(p)));
        }
    };
    for c in &constants.before_witness {
        fixed(m, c, &mut comms);
    }
    for w in &proof.witness {
        comms.extend(w.iter().cloned().map(Some));
    }
    for c in &constants.after_witness {
        fixed(m, c, &mut comms);
    }
    let mut acc: Option<M::Point> = None;
    for c in comms.iter().rev() {
        acc = match (acc, c) {
            (None, c) => c.clone(),
            (Some(a), c) => {
                let scaled = m.endo_scale(&a, xi);
                Some(match c {
                    Some(c) => m.add_points(&scaled, c),
                    None => scaled,
                })
            }
        };
    }
    match acc {
        Some(a) => a,
        None => unreachable!("ft's commitment is never the identity"),
    }
}

/// `[perm] sigma_last - [zeta^n - 1] t`, chunks combined by `zeta^m`.
fn ft_comm<C, M>(
    m: &mut M,
    constants: &Constants<C>,
    t: &[M::Point],
    deferred: &DeferredIn<M::Value, M::Scalar>,
) -> M::Point
where
    C: Glv,
    M: GroupModel<C>,
{
    let sigma = constant::<C, M>(m, &constants.sigma_last);
    let sigma = chunked::<C, M>(m, &sigma, &deferred.zeta_to_srs_length);
    let sigma = m.scale_point(&sigma, &deferred.perm);
    let t = chunked::<C, M>(m, t, &deferred.zeta_to_srs_length);
    let t_n = m.scale_point(&t, &deferred.zeta_to_domain_size);
    let t_n_minus_t = m.sub_points(&t_n, &t);
    m.sub_points(&sigma, &t_n_minus_t)
}

/// `x_hat`: `h` plus the Lagrange commitments scaled by minus the public
/// input, chunk by chunk (`h` alone without public input).
fn x_hat<C, M>(m: &mut M, constants: &Constants<C>, public: &[M::Scalar]) -> Vec<M::Point>
where
    C: Glv,
    M: GroupModel<C>,
{
    (0..constants.chunks)
        .map(|k| {
            let mut acc = m.constant_point(&constants.h);
            for (l, p) in constants.lagrange.iter().zip(public) {
                let chunk = m.constant_point(&l.chunks[k]);
                let scaled = m.scale_point(&chunk, p);
                acc = m.sub_points(&acc, &scaled);
            }
            acc
        })
        .collect()
}

/// The group half in `model`, on Kimchi's base sponge over `permutation`
/// in that model (see the module documentation). The model is shared with
/// the sponge, so each step borrows it in turn.
pub fn group_half_in<C, M>(
    model: &RefCell<M>,
    permutation: &KimchiPoseidon<C::Base>,
    constants: &Constants<C>,
    proof: &ProofIn<M::Point, M::Scalar>,
    deferred: &DeferredIn<M::Value, M::Scalar>,
) -> Outputs<M::Value, M::Point>
where
    C: Glv,
    C::Base: PrimeField + Sqrt,
    M: GroupModel<C>,
    KimchiPoseidon<C::Base>: PermutationIn<M, C::Base, WIDTH>,
{
    let (public, inputs) = {
        let mut m = model.borrow_mut();
        let public = x_hat::<C, M>(&mut m, constants, &deferred.public);
        let index_digest = m.constant(constants.index_digest);
        let inputs = PiopInputs {
            index_digest,
            previous: held::<C, M>(&mut m, &proof.previous),
            public: held::<C, M>(&mut m, &public),
            witness: core::array::from_fn(|i| held::<C, M>(&mut m, &proof.witness[i])),
            lookup: None,
            z: held::<C, M>(&mut m, &proof.z),
            t: held::<C, M>(&mut m, &proof.t),
        };
        (public, inputs)
    };
    let (drawn, mut sponge) = base_transcript_in::<C, M, _>(model, permutation, &inputs);
    let (combined, elements) = {
        let mut m = model.borrow_mut();
        let ft = ft_comm::<C, M>(&mut m, constants, &proof.t, deferred);
        let combined = combined::<C, M>(&mut m, constants, proof, public, ft, &deferred.xi);
        let elements = m.shifted_elements(&deferred.combined_inner_product);
        (combined, elements)
    };
    // The IPA: the shifted value bound, u hashed to the curve, the rounds,
    // then the final step.
    for e in &elements {
        sponge.absorb_value(e);
    }
    let t_u = sponge.squeeze_value();
    let (u, mut q) = {
        let mut m = model.borrow_mut();
        let (ux, uy) = constants.group_map.to_group_in(&mut *m, &t_u);
        let u = m.point_of(&ux, &uy);
        let value = m.scale_point(&u, &deferred.combined_inner_product);
        let q = m.add_points(&combined, &value);
        (u, q)
    };
    let mut prechallenges = Vec::with_capacity(proof.rounds.len());
    for (l, r) in &proof.rounds {
        let (lc, rc) = {
            let mut m = model.borrow_mut();
            (m.coordinates_of(l), m.coordinates_of(r))
        };
        for c in [lc, rc] {
            sponge.absorb_value(&c.x);
            sponge.absorb_value(&c.y);
        }
        let chal = sponge.challenge_read();
        let mut m = model.borrow_mut();
        let l_inv = m.endo_inv_scale(l, &chal);
        let r_x = m.endo_scale(r, &chal);
        let q_l = m.add_points(&q, &l_inv);
        q = m.add_points(&q_l, &r_x);
        prechallenges.push(chal);
    }
    let delta = model.borrow_mut().coordinates_of(&proof.delta);
    sponge.absorb_value(&delta.x);
    sponge.absorb_value(&delta.y);
    let c = sponge.challenge_read();
    let mut m = model.borrow_mut();
    // [c] Q + delta = [z1] (sg + [b] U) + [z2] H.
    let cq = m.endo_scale(&q, &c);
    let left = m.add_points(&cq, &proof.delta);
    let b_u = m.scale_point(&u, &deferred.b);
    let sg_b_u = m.add_points(&proof.sg, &b_u);
    let z1_term = m.scale_point(&sg_b_u, &proof.z1);
    let h = m.constant_point(&constants.h);
    let z2_term = m.scale_point(&h, &proof.z2);
    let right = m.add_points(&z1_term, &z2_term);
    Outputs {
        drawn,
        prechallenges,
        left,
        right,
        sg: proof.sg.clone(),
    }
}
