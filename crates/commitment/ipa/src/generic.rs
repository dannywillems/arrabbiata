//! The IPA with its deviations as parameters: one opening, [`GenericIpa`], and a trait per choice an
//! implementation makes, so that this crate's IPA and Kimchi's are two
//! instances of one construction rather than two code paths.
//!
//! | parameter | trait | this crate ([`crate::Ipa`]) | Kimchi |
//! | --- | --- | --- | --- |
//! | the oracle | [`IpaOracle`] | a scalar-field transcript ([`ScalarOracle`]) | Kimchi's base-field sponge |
//! | what binds the statement | [`Binding`] | the whole statement ([`BindStatement`]) | the shifted value only ([`BindValue`]), the protocol having absorbed the rest |
//! | the challenges | [`ChallengeMap`] | full scalars ([`Full`]) | 128-bit endomorphism challenges ([`Endo`]) |
//! | the base of the value | [`ValueBase`] | `xi U`, `xi` squeezed ([`ScaledGenerator`]) | hashed to the curve ([`HashedToCurve`]) |
//! | hiding | [`Hiding`] | none ([`NotHiding`]) | blinded rounds and `(delta, z1, z2)` ([`Blinded`]) |
//!
//! The rounds are common to all: with `U` the value base, `L = <G_lo,
//! a_hi> + r_l H + <a_hi, b_lo> U`, `R = <G_hi, a_lo> + r_r H + <a_lo, b_hi>
//! U` (`r_l = r_r = 0` without hiding), absorbed; the round challenge `x`
//! (mapped); `a <- a_lo + x^(-1) a_hi`, `b <- b_lo + x b_hi`, `G <- G_lo +
//! x G_hi`. With `Q = C + v U + sum_j (x_j^(-1) L_j + x_j R_j)` the
//! verifier's side, the final step is the hiding's:
//!
//! - [`NotHiding`]: the proof sends `a`, the check is `Q = a sg + a b0 U`;
//! - [`Blinded`]: with random `d, r_delta`, `delta = d (sg + b0 U) +
//!   r_delta H`, the challenge `c` after it, `z1 = a c + d`, `z2 = r' c +
//!   r_delta` (`r'` the commitment's blinder plus the rounds' `r_l x^(-1) +
//!   r_r x`); the check is `c Q + delta = z1 (sg + b0 U) + z2 H`.
//!
//! Either way the verifier trusts `sg`, and its output is the accumulator
//! `(x, sg)`, decided by an MSM of the size of the key: the hiding changes
//! the proof, not the accumulation.

use alloc::vec::Vec;
use core::fmt::Debug;

use arrabbiata_curve::{Curve, Projective, absorb_point, msm};
use arrabbiata_field::{Field, PrimeField, inner_product, rng::Sampler};
use arrabbiata_transcript::{Transcript, absorb_slice};

use crate::{RecursionChallenge, Statement, b_poly, evaluation_vector};

/// The oracle an opening talks to: it absorbs scalars and points and
/// squeezes scalars.
pub trait IpaOracle<C>
where
    C: Curve,
{
    /// Absorb a scalar.
    fn absorb_scalar(&mut self, x: C::Scalar);

    /// Absorb a point.
    fn absorb_point(&mut self, p: &Projective<C>);

    /// A scalar, before the [`ChallengeMap`].
    fn squeeze_scalar(&mut self) -> C::Scalar;
}

/// An oracle that can also squeeze a base-field element, the input of a
/// hash to the curve.
pub trait BaseOracle<C>: IpaOracle<C>
where
    C: Curve,
{
    /// A base-field element.
    fn squeeze_base(&mut self) -> C::Base;
}

/// A scalar-field transcript as an oracle: points absorbed by their
/// coordinates as `arrabbiata-curve`'s `absorb_point` does.
#[derive(Debug)]
pub struct ScalarOracle<'a, T>(pub &'a mut T);

impl<C, T> IpaOracle<C> for ScalarOracle<'_, T>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar>,
{
    fn absorb_scalar(&mut self, x: C::Scalar) {
        self.0.absorb(x);
    }

    fn absorb_point(&mut self, p: &Projective<C>) {
        absorb_point(p, self.0);
    }

    fn squeeze_scalar(&mut self) -> C::Scalar {
        self.0.squeeze()
    }
}

/// What the opening absorbs of its statement before the value base.
pub trait Binding<C, O>
where
    C: Curve,
{
    /// Absorb `statement` into `oracle`.
    fn bind(&self, oracle: &mut O, statement: &Statement<'_, C>);
}

/// Domain separator absorbed at the start of every [`BindStatement`]
/// opening.
pub const DOMAIN: u64 = 0x6970_615f_6f70_656e;

/// The whole statement: a domain separator, the commitment, the points,
/// `evalscale` and the value. For an opening on its own transcript.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindStatement;

impl<C, T> Binding<C, ScalarOracle<'_, T>> for BindStatement
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar>,
{
    fn bind(&self, oracle: &mut ScalarOracle<'_, T>, statement: &Statement<'_, C>) {
        let transcript = &mut *oracle.0;
        transcript.absorb(C::Scalar::from_u64(DOMAIN));
        absorb_point(statement.commitment, transcript);
        absorb_slice(transcript, statement.points);
        transcript.absorb(statement.evalscale);
        transcript.absorb(statement.value);
    }
}

/// The value only, shifted as Kimchi's `shift_scalar` (so that a scalar
/// the base-field sponge reads back is the one meant): for an opening
/// inside a protocol whose transcript has the rest of the statement
/// (Kimchi absorbs the commitments and draws the points and the scales
/// itself).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindValue;

/// Kimchi's `shift_scalar`: `(x - 2^b - 1) / 2` when the scalar modulus is
/// the smaller and `x - 2^b` otherwise, `b` the scalar modulus's bit
/// length.
#[must_use]
pub fn shift_scalar<C>(x: C::Scalar) -> C::Scalar
where
    C: Curve,
    C::Base: PrimeField,
{
    let two = C::Scalar::ONE + C::Scalar::ONE;
    let two_pow = two.pow(u64::from(C::Scalar::MODULUS_BITS));
    if scalar_modulus_is_smaller::<C>() {
        let half = match two.inverse() {
            Some(h) => h,
            None => unreachable!("the scalar field is odd"),
        };
        (x - (two_pow + C::Scalar::ONE)) * half
    } else {
        x - two_pow
    }
}

/// Whether the scalar modulus is below the base modulus (compared through
/// `p - 1`).
fn scalar_modulus_is_smaller<C>() -> bool
where
    C: Curve,
    C::Base: PrimeField,
{
    let (a, b) = (-C::Scalar::ONE, -C::Base::ONE);
    let n = C::Scalar::LIMBS.max(C::Base::LIMBS);
    let at_a = |i: usize| if i < C::Scalar::LIMBS { a.limb(i) } else { 0 };
    let at_b = |i: usize| if i < C::Base::LIMBS { b.limb(i) } else { 0 };
    let mut i = n;
    while i > 0 {
        i -= 1;
        if at_a(i) != at_b(i) {
            return at_a(i) < at_b(i);
        }
    }
    false
}

impl<C, O> Binding<C, O> for BindValue
where
    C: Curve,
    C::Base: PrimeField,
    O: IpaOracle<C>,
{
    fn bind(&self, oracle: &mut O, statement: &Statement<'_, C>) {
        oracle.absorb_scalar(shift_scalar::<C>(statement.value));
    }
}

/// How a squeezed scalar becomes a challenge.
pub trait ChallengeMap<F> {
    /// The challenge of `raw`.
    fn map(&self, raw: F) -> F;
}

/// Full scalars: the identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Full;

impl<F> ChallengeMap<F> for Full {
    fn map(&self, raw: F) -> F {
        raw
    }
}

/// 128-bit endomorphism challenges (Kimchi's `ScalarChallenge::to_field`):
/// the low 128 bits `r` of the squeeze become `a lambda + b`, with `a, b`
/// built two bits at a time from `a = b = 2`, so that the in-circuit
/// verifier computes the scalar multiplication with EndoMul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endo<F> {
    /// The scalar of the curve's endomorphism, `phi(P) = [lambda] P`.
    pub lambda: F,
}

/// The bits of a challenge.
pub const CHALLENGE_BITS: usize = 128;

impl<F> ChallengeMap<F> for Endo<F>
where
    F: PrimeField,
{
    fn map(&self, raw: F) -> F {
        let two = F::ONE + F::ONE;
        let (mut a, mut b) = (two, two);
        let bit = |i: usize| (raw.limb(i / 64) >> (i % 64)) & 1 == 1;
        let mut i = CHALLENGE_BITS / 2;
        while i > 0 {
            i -= 1;
            a = a + a;
            b = b + b;
            let s = if bit(2 * i) { F::ONE } else { -F::ONE };
            if bit(2 * i + 1) {
                a = a + s;
            } else {
                b = b + s;
            }
        }
        a * self.lambda + b
    }
}

/// The base `U` that binds the value.
pub trait ValueBase<C, O>
where
    C: Curve,
{
    /// The base, from `oracle` after the statement, and the scheme's
    /// extra generator `u`.
    fn base(&self, oracle: &mut O, u: &Projective<C>) -> Projective<C>;
}

/// `xi U`, `xi` squeezed after the statement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScaledGenerator;

impl<C, O> ValueBase<C, O> for ScaledGenerator
where
    C: Curve,
    O: IpaOracle<C>,
{
    fn base(&self, oracle: &mut O, u: &Projective<C>) -> Projective<C> {
        *u * oracle.squeeze_scalar()
    }
}

/// A map from the base field to the curve.
pub trait HashToCurve<C>
where
    C: Curve,
{
    /// The point of `t`.
    fn to_group(&self, t: C::Base) -> Projective<C>;
}

/// `U` the hash to the curve of a squeezed base-field element (Kimchi's
/// group map), independent of every generator by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashedToCurve<M>(pub M);

impl<C, O, M> ValueBase<C, O> for HashedToCurve<M>
where
    C: Curve,
    O: BaseOracle<C>,
    M: HashToCurve<C>,
{
    fn base(&self, oracle: &mut O, _u: &Projective<C>) -> Projective<C> {
        self.0.to_group(oracle.squeeze_base())
    }
}

/// Everything the final step sees, prover side.
#[derive(Clone, Copy, Debug)]
pub struct Folded<C>
where
    C: Curve,
{
    /// The folded coefficient.
    pub a: C::Scalar,
    /// The folded evaluation vector, `b0`.
    pub b: C::Scalar,
    /// The folded generator.
    pub sg: Projective<C>,
    /// The value base.
    pub u: Projective<C>,
    /// The blinding generator.
    pub h: Projective<C>,
    /// The total blinder, the commitment's plus the rounds'.
    pub blinder: C::Scalar,
}

/// Whether the opening hides the polynomial.
pub trait Hiding<C>
where
    C: Curve,
{
    /// What the final step sends.
    type Final: Clone + Debug + PartialEq;

    /// The blinders of a round's `L` and `R`.
    fn round_blinders<R>(&self, rng: &mut R) -> (C::Scalar, C::Scalar)
    where
        R: Sampler<C::Scalar>;

    /// The final step, prover side.
    fn prove_final<O, M, R>(
        &self,
        oracle: &mut O,
        map: &M,
        folded: &Folded<C>,
        rng: &mut R,
    ) -> Self::Final
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
        R: Sampler<C::Scalar>;

    /// The final check, verifier side: an expression that is the identity
    /// exactly when it holds, given `Q`, `sg`, `b0`, `U` and `H`.
    #[allow(clippy::too_many_arguments)]
    fn check_final<O, M>(
        &self,
        oracle: &mut O,
        map: &M,
        last: &Self::Final,
        q: Projective<C>,
        sg: Projective<C>,
        b0: C::Scalar,
        u: Projective<C>,
        h: Projective<C>,
    ) -> Projective<C>
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>;
}

/// No hiding: the proof sends the folded coefficient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NotHiding;

impl<C> Hiding<C> for NotHiding
where
    C: Curve,
{
    type Final = C::Scalar;

    fn round_blinders<R>(&self, _rng: &mut R) -> (C::Scalar, C::Scalar)
    where
        R: Sampler<C::Scalar>,
    {
        (C::Scalar::ZERO, C::Scalar::ZERO)
    }

    fn prove_final<O, M, R>(
        &self,
        _oracle: &mut O,
        _map: &M,
        folded: &Folded<C>,
        _rng: &mut R,
    ) -> C::Scalar
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
        R: Sampler<C::Scalar>,
    {
        folded.a
    }

    fn check_final<O, M>(
        &self,
        _oracle: &mut O,
        _map: &M,
        a: &C::Scalar,
        q: Projective<C>,
        sg: Projective<C>,
        b0: C::Scalar,
        u: Projective<C>,
        _h: Projective<C>,
    ) -> Projective<C>
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
    {
        q - (sg * *a + u * (*a * b0))
    }
}

/// Kimchi's hiding (`SRS::open`): blinded rounds and a Schnorr-like final
/// step `(delta, z1, z2)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blinded;

/// The final step of a [`Blinded`] opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schnorr<C>
where
    C: Curve,
{
    /// `d (sg + b0 U) + r_delta H`.
    pub delta: Projective<C>,
    /// `a c + d`.
    pub z1: C::Scalar,
    /// `r' c + r_delta`.
    pub z2: C::Scalar,
}

impl<C> Hiding<C> for Blinded
where
    C: Curve,
{
    type Final = Schnorr<C>;

    fn round_blinders<R>(&self, rng: &mut R) -> (C::Scalar, C::Scalar)
    where
        R: Sampler<C::Scalar>,
    {
        let l = rng.sample();
        let r = rng.sample();
        (l, r)
    }

    fn prove_final<O, M, R>(
        &self,
        oracle: &mut O,
        map: &M,
        folded: &Folded<C>,
        rng: &mut R,
    ) -> Schnorr<C>
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
        R: Sampler<C::Scalar>,
    {
        let d: C::Scalar = rng.sample();
        let r_delta: C::Scalar = rng.sample();
        let delta = (folded.sg + folded.u * folded.b) * d + folded.h * r_delta;
        oracle.absorb_point(&delta);
        let c = map.map(oracle.squeeze_scalar());
        Schnorr {
            delta,
            z1: folded.a * c + d,
            z2: folded.blinder * c + r_delta,
        }
    }

    fn check_final<O, M>(
        &self,
        oracle: &mut O,
        map: &M,
        last: &Schnorr<C>,
        q: Projective<C>,
        sg: Projective<C>,
        b0: C::Scalar,
        u: Projective<C>,
        h: Projective<C>,
    ) -> Projective<C>
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
    {
        oracle.absorb_point(&last.delta);
        let c = map.map(oracle.squeeze_scalar());
        q * c + last.delta - (sg + u * b0) * last.z1 - h * last.z2
    }
}

/// An opening proof, whatever the hiding: the rounds' `(L, R)`, the final
/// step, and the folded generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening<C, Fin>
where
    C: Curve,
{
    /// The `(L, R)` of each round.
    pub rounds: Vec<(Projective<C>, Projective<C>)>,
    /// The final step.
    pub last: Fin,
    /// The folded generator.
    pub sg: Projective<C>,
}

/// The IPA on `generators`, the extra generator `u` (for
/// [`ScaledGenerator`]) and the blinding generator `h`, with its choices as
/// parameters.
#[derive(Clone, Debug)]
pub struct GenericIpa<C, B, Ch, V, H>
where
    C: Curve,
{
    /// The commitment generators, a power of two of them.
    pub generators: Vec<Projective<C>>,
    /// The extra generator.
    pub u: Projective<C>,
    /// The blinding generator.
    pub h: Projective<C>,
    /// What binds the statement.
    pub binding: B,
    /// The challenge map.
    pub challenges: Ch,
    /// The base of the value.
    pub value_base: V,
    /// The hiding.
    pub hiding: H,
}

/// A sampler for openings that draw nothing ([`NotHiding`]): it is never
/// called.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoRandomness;

impl<F> Sampler<F> for NoRandomness {
    fn sample(&mut self) -> F {
        unreachable!("an opening without hiding draws no randomness")
    }
}

/// What a prover opens: the coefficients and the commitment's blinder
/// (zero without hiding).
#[derive(Clone, Copy, Debug)]
pub struct Witness<'a, F> {
    /// The coefficients.
    pub coefficients: &'a [F],
    /// The blinder of the commitment, `C = <a, G> + r H`.
    pub blinder: F,
}

impl<C, B, Ch, V, H> GenericIpa<C, B, Ch, V, H>
where
    C: Curve,
    Ch: ChallengeMap<C::Scalar>,
    H: Hiding<C>,
{
    /// The number of rounds, `log2` of the number of generators.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.generators.len().trailing_zeros() as usize
    }

    /// Prove that `witness` opens `statement`, on `oracle`, drawing the
    /// blinders from `rng`: the proof and its accumulator `(x, sg)`.
    ///
    /// # Panics
    ///
    /// If there are more coefficients than generators.
    pub fn prove<O, R>(
        &self,
        oracle: &mut O,
        statement: &Statement<'_, C>,
        witness: &Witness<'_, C::Scalar>,
        rng: &mut R,
    ) -> (Opening<C, H::Final>, RecursionChallenge<C>)
    where
        O: IpaOracle<C>,
        B: Binding<C, O>,
        V: ValueBase<C, O>,
        R: Sampler<C::Scalar>,
    {
        let n = self.generators.len();
        assert!(
            witness.coefficients.len() <= n,
            "more coefficients than generators"
        );
        let mut a = witness.coefficients.to_vec();
        a.resize(n, C::Scalar::ZERO);
        let mut b = evaluation_vector(statement.points, statement.evalscale, n);
        self.binding.bind(oracle, statement);
        let u = self.value_base.base(oracle, &self.u);
        let mut g = self.generators.clone();
        let mut rounds = Vec::with_capacity(self.rounds());
        let mut chals = Vec::with_capacity(self.rounds());
        let mut blinder = witness.blinder;
        while a.len() > 1 {
            let half = a.len() / 2;
            let (a_lo, a_hi) = a.split_at(half);
            let (b_lo, b_hi) = b.split_at(half);
            let (g_lo, g_hi) = g.split_at(half);
            let (r_l, r_r) = self.hiding.round_blinders(rng);
            let l = msm(g_lo, a_hi) + self.h * r_l + u * inner_product(a_hi, b_lo);
            let r = msm(g_hi, a_lo) + self.h * r_r + u * inner_product(a_lo, b_hi);
            oracle.absorb_point(&l);
            oracle.absorb_point(&r);
            let x = self.challenges.map(oracle.squeeze_scalar());
            let x_inv = match x.inverse() {
                Some(inverse) => inverse,
                // Probability 1 / |F|; the verifier rejects such a proof.
                None => C::Scalar::ONE,
            };
            blinder = blinder + r_l * x_inv + r_r * x;
            let next_a = a_lo
                .iter()
                .zip(a_hi)
                .map(|(lo, hi)| *lo + *hi * x_inv)
                .collect();
            let next_b = b_lo
                .iter()
                .zip(b_hi)
                .map(|(lo, hi)| *lo + *hi * x)
                .collect();
            let next_g = g_lo
                .iter()
                .zip(g_hi)
                .map(|(lo, hi)| *lo + *hi * x)
                .collect();
            rounds.push((l, r));
            chals.push(x);
            a = next_a;
            b = next_b;
            g = next_g;
        }
        let folded = Folded {
            a: a[0],
            b: b[0],
            sg: g[0],
            u,
            h: self.h,
            blinder,
        };
        let last = self
            .hiding
            .prove_final(oracle, &self.challenges, &folded, rng);
        let accumulator = RecursionChallenge {
            chals,
            comm: folded.sg,
        };
        let opening = Opening {
            rounds,
            last,
            sg: folded.sg,
        };
        (opening, accumulator)
    }

    /// The succinct verifier, on `oracle`: the accumulator `(x, sg)` and
    /// the final expression, the identity exactly when the opening holds up
    /// to `sg` (which the accumulator claims commits to `b`); `None` for a
    /// proof of the wrong shape or a zero challenge.
    pub fn check<O>(
        &self,
        oracle: &mut O,
        statement: &Statement<'_, C>,
        opening: &Opening<C, H::Final>,
    ) -> Option<(RecursionChallenge<C>, Projective<C>)>
    where
        O: IpaOracle<C>,
        B: Binding<C, O>,
        V: ValueBase<C, O>,
    {
        let k = self.rounds();
        if opening.rounds.len() != k {
            return None;
        }
        self.binding.bind(oracle, statement);
        let u = self.value_base.base(oracle, &self.u);
        let mut chals = Vec::with_capacity(k);
        let mut q = *statement.commitment + u * statement.value;
        for (l, r) in &opening.rounds {
            oracle.absorb_point(l);
            oracle.absorb_point(r);
            let x = self.challenges.map(oracle.squeeze_scalar());
            let x_inv = x.inverse()?;
            q = q + *l * x_inv + *r * x;
            chals.push(x);
        }
        let mut b0 = C::Scalar::ZERO;
        let mut scale = C::Scalar::ONE;
        for point in statement.points {
            b0 = b0 + scale * b_poly(&chals, *point);
            scale = scale * statement.evalscale;
        }
        let expression = self.hiding.check_final(
            oracle,
            &self.challenges,
            &opening.last,
            q,
            opening.sg,
            b0,
            u,
            self.h,
        );
        let accumulator = RecursionChallenge {
            chals,
            comm: opening.sg,
        };
        Some((accumulator, expression))
    }

    /// [`GenericIpa::check`] with the expression required to vanish: the
    /// accumulator, or `None`.
    pub fn succinct<O>(
        &self,
        oracle: &mut O,
        statement: &Statement<'_, C>,
        opening: &Opening<C, H::Final>,
    ) -> Option<RecursionChallenge<C>>
    where
        O: IpaOracle<C>,
        B: Binding<C, O>,
        V: ValueBase<C, O>,
    {
        let (accumulator, expression) = self.check(oracle, statement, opening)?;
        expression.is_identity().then_some(accumulator)
    }

    /// The decider: whether `accumulator.comm` is `<s, G>`.
    #[must_use]
    pub fn decide(&self, accumulator: &RecursionChallenge<C>) -> bool {
        accumulator.chals.len() == self.rounds()
            && accumulator.comm == msm(&self.generators, &accumulator.coefficients())
    }
}
