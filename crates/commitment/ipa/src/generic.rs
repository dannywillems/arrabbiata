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
//!
//! The transcript is a type, [`OpeningTranscript`]: each parameter says
//! what it writes and draws as a step (`Binding::Step`, `ValueBase::Step`,
//! `Hiding::Step`), and the opening is the binding's step, the value base's,
//! the rounds (`(L, R)` sent, `x` drawn, as many as `log2` of the key), then
//! the hiding's. The prover and the verifier both follow it, on a tape that
//! borrows the oracle and absorbs nothing else.

use alloc::vec::Vec;
use core::{fmt::Debug, marker::PhantomData};

use arrabbiata_curve::{Curve, Projective, absorb_point, msm};
use arrabbiata_field::{Field, PrimeField, inner_product, rng::Sampler};
use arrabbiata_transcript::{
    Transcript, absorb_slice,
    typed::{Draw, End, Label, Rounds, Send, Squeeze, Statement as Absorbed, Tape, Write},
};

use crate::{RecursionChallenge, Statement, evaluation_vector, folded_evaluation};

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
    /// The step, then `Next`.
    type Step<Next>;

    /// Write `statement` to the tape.
    fn bind<'a, F, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        statement: &Statement<'_, C>,
    ) -> Tape<F, &'a mut O, Next>;
}

/// The step names of an opening's transcript, with their labels.
macro_rules! step {
    ($(#[$doc:meta])* $name:ident = $label:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;

        impl Label for $name {
            const LABEL: u64 = $label;
        }
    };
}

step!(
    /// The value base's scale `xi`.
    ValueChallenge = 0x6901
);
step!(
    /// A round's challenge.
    RoundChallenge = 0x6902
);
step!(
    /// The final step's challenge `c`.
    FinalChallenge = 0x6903
);

/// The whole statement, as [`BindStatement`] writes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundStatement<C>
where
    C: Curve,
{
    /// The commitment.
    pub commitment: Projective<C>,
    /// The points.
    pub points: Vec<C::Scalar>,
    /// The points' scale.
    pub evalscale: C::Scalar,
    /// The value.
    pub value: C::Scalar,
}

impl<C> Label for BoundStatement<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x6904;
}

impl<C, T> Write<&mut ScalarOracle<'_, T>> for BoundStatement<C>
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar>,
{
    fn write(&self, oracle: &mut &mut ScalarOracle<'_, T>) {
        let transcript = &mut *oracle.0;
        transcript.absorb(C::Scalar::from_u64(DOMAIN));
        absorb_point(&self.commitment, transcript);
        absorb_slice(transcript, &self.points);
        transcript.absorb(self.evalscale);
        transcript.absorb(self.value);
    }
}

/// The value, shifted, as [`BindValue`] writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShiftedValue<C>(pub C::Scalar)
where
    C: Curve;

impl<C> Label for ShiftedValue<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x6905;
}

impl<C, O> Write<&mut O> for ShiftedValue<C>
where
    C: Curve,
    O: IpaOracle<C>,
{
    fn write(&self, oracle: &mut &mut O) {
        oracle.absorb_scalar(self.0);
    }
}

/// A round's `(L, R)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundPoints<C>(pub Projective<C>, pub Projective<C>)
where
    C: Curve;

impl<C> Label for RoundPoints<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x6906;
}

impl<C, O> Write<&mut O> for RoundPoints<C>
where
    C: Curve,
    O: IpaOracle<C>,
{
    fn write(&self, oracle: &mut &mut O) {
        oracle.absorb_point(&self.0);
        oracle.absorb_point(&self.1);
    }
}

/// The final step's `delta` ([`Blinded`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delta<C>(pub Projective<C>)
where
    C: Curve;

impl<C> Label for Delta<C>
where
    C: Curve,
{
    const LABEL: u64 = 0x6907;
}

impl<C, O> Write<&mut O> for Delta<C>
where
    C: Curve,
    O: IpaOracle<C>,
{
    fn write(&self, oracle: &mut &mut O) {
        oracle.absorb_point(&self.0);
    }
}

/// A scalar squeezed for the step `Name`, before the [`ChallengeMap`].
#[derive(Debug)]
pub struct ScalarDraw<C, Name>(PhantomData<(C, Name)>);

impl<C, Name> Label for ScalarDraw<C, Name>
where
    Name: Label,
{
    const LABEL: u64 = Name::LABEL;
}

impl<C, O, Name> Draw<&mut O> for ScalarDraw<C, Name>
where
    C: Curve,
    O: IpaOracle<C>,
{
    type Value = C::Scalar;

    fn draw(oracle: &mut &mut O) -> C::Scalar {
        oracle.squeeze_scalar()
    }
}

/// A base-field element squeezed for the hash to the curve.
#[derive(Debug)]
pub struct BaseDraw<C>(PhantomData<C>);

impl<C> Label for BaseDraw<C> {
    const LABEL: u64 = 0x6908;
}

impl<C, O> Draw<&mut O> for BaseDraw<C>
where
    C: Curve,
    O: BaseOracle<C>,
{
    type Value = C::Base;

    fn draw(oracle: &mut &mut O) -> C::Base {
        oracle.squeeze_base()
    }
}

/// One round: `(L, R)` sent, the challenge drawn.
pub type Round<C> = Send<RoundPoints<C>, Squeeze<ScalarDraw<C, RoundChallenge>, End>>;

/// The transcript of an opening with the binding `B`, the value base `V`
/// and the hiding `H` on the oracle `O`.
pub type OpeningTranscript<C, B, V, H, O> = <B as Binding<C, O>>::Step<
    <V as ValueBase<C, O>>::Step<Rounds<Round<C>, <H as Hiding<C>>::Step<End>>>,
>;

/// A tape borrowing the oracle `O` at the start of [`OpeningTranscript`].
pub type OpeningTape<'a, C, B, V, H, O> =
    Tape<<C as Curve>::Scalar, &'a mut O, OpeningTranscript<C, B, V, H, O>>;

/// Domain separator absorbed at the start of every [`BindStatement`]
/// opening.
pub const DOMAIN: u64 = 0x6970_615f_6f70_656e;

/// The whole statement: a domain separator, the commitment, the points,
/// `evalscale` and the value. For an opening on its own transcript.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindStatement;

impl<'b, C, T> Binding<C, ScalarOracle<'b, T>> for BindStatement
where
    C: Curve,
    C::Base: PrimeField,
    T: Transcript<C::Scalar>,
{
    type Step<Next> = Absorbed<BoundStatement<C>, Next>;

    fn bind<'a, F, Next>(
        &self,
        tape: Tape<F, &'a mut ScalarOracle<'b, T>, Self::Step<Next>>,
        statement: &Statement<'_, C>,
    ) -> Tape<F, &'a mut ScalarOracle<'b, T>, Next> {
        tape.statement(&BoundStatement {
            commitment: *statement.commitment,
            points: statement.points.to_vec(),
            evalscale: statement.evalscale,
            value: statement.value,
        })
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
    type Step<Next> = Absorbed<ShiftedValue<C>, Next>;

    fn bind<'a, F, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        statement: &Statement<'_, C>,
    ) -> Tape<F, &'a mut O, Next> {
        tape.statement(&ShiftedValue(shift_scalar::<C>(statement.value)))
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
    /// The step, then `Next`.
    type Step<Next>;

    /// The base, drawn from the tape after the statement, and the scheme's
    /// extra generator `u`.
    fn base<'a, F, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        u: &Projective<C>,
    ) -> (Projective<C>, Tape<F, &'a mut O, Next>);
}

/// `xi U`, `xi` squeezed after the statement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScaledGenerator;

impl<C, O> ValueBase<C, O> for ScaledGenerator
where
    C: Curve,
    O: IpaOracle<C>,
{
    type Step<Next> = Squeeze<ScalarDraw<C, ValueChallenge>, Next>;

    fn base<'a, F, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        u: &Projective<C>,
    ) -> (Projective<C>, Tape<F, &'a mut O, Next>) {
        let (xi, tape) = tape.squeeze();
        (*u * xi, tape)
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
    type Step<Next> = Squeeze<BaseDraw<C>, Next>;

    fn base<'a, F, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        _u: &Projective<C>,
    ) -> (Projective<C>, Tape<F, &'a mut O, Next>) {
        let (t, tape) = tape.squeeze();
        (self.0.to_group(t), tape)
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

    /// The final step's messages and challenges, then `Next`.
    type Step<Next>;

    /// The blinders of a round's `L` and `R`.
    fn round_blinders<R>(&self, rng: &mut R) -> (C::Scalar, C::Scalar)
    where
        R: Sampler<C::Scalar>;

    /// The final step, prover side.
    fn prove_final<'a, F, O, M, R, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        map: &M,
        folded: &Folded<C>,
        rng: &mut R,
    ) -> (Self::Final, Tape<F, &'a mut O, Next>)
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
        R: Sampler<C::Scalar>;

    /// The final check, verifier side: an expression that is the identity
    /// exactly when it holds, given `Q`, `sg`, `b0`, `U` and `H`.
    #[allow(clippy::too_many_arguments)]
    fn check_final<'a, F, O, M, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        map: &M,
        last: &Self::Final,
        q: Projective<C>,
        sg: Projective<C>,
        b0: C::Scalar,
        u: Projective<C>,
        h: Projective<C>,
    ) -> (Projective<C>, Tape<F, &'a mut O, Next>)
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
    type Step<Next> = Next;

    fn round_blinders<R>(&self, _rng: &mut R) -> (C::Scalar, C::Scalar)
    where
        R: Sampler<C::Scalar>,
    {
        (C::Scalar::ZERO, C::Scalar::ZERO)
    }

    fn prove_final<'a, F, O, M, R, Next>(
        &self,
        tape: Tape<F, &'a mut O, Next>,
        _map: &M,
        folded: &Folded<C>,
        _rng: &mut R,
    ) -> (C::Scalar, Tape<F, &'a mut O, Next>)
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
        R: Sampler<C::Scalar>,
    {
        (folded.a, tape)
    }

    fn check_final<'a, F, O, M, Next>(
        &self,
        tape: Tape<F, &'a mut O, Next>,
        _map: &M,
        a: &C::Scalar,
        q: Projective<C>,
        sg: Projective<C>,
        b0: C::Scalar,
        u: Projective<C>,
        _h: Projective<C>,
    ) -> (Projective<C>, Tape<F, &'a mut O, Next>)
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
    {
        (q - (sg * *a + u * (*a * b0)), tape)
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
    type Step<Next> = Send<Delta<C>, Squeeze<ScalarDraw<C, FinalChallenge>, Next>>;

    fn round_blinders<R>(&self, rng: &mut R) -> (C::Scalar, C::Scalar)
    where
        R: Sampler<C::Scalar>,
    {
        let l = rng.sample();
        let r = rng.sample();
        (l, r)
    }

    fn prove_final<'a, F, O, M, R, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        map: &M,
        folded: &Folded<C>,
        rng: &mut R,
    ) -> (Schnorr<C>, Tape<F, &'a mut O, Next>)
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
        R: Sampler<C::Scalar>,
    {
        let d: C::Scalar = rng.sample();
        let r_delta: C::Scalar = rng.sample();
        let delta = (folded.sg + folded.u * folded.b) * d + folded.h * r_delta;
        let (raw, tape) = tape.send(&Delta(delta)).squeeze();
        let c = map.map(raw);
        let last = Schnorr {
            delta,
            z1: folded.a * c + d,
            z2: folded.blinder * c + r_delta,
        };
        (last, tape)
    }

    fn check_final<'a, F, O, M, Next>(
        &self,
        tape: Tape<F, &'a mut O, Self::Step<Next>>,
        map: &M,
        last: &Schnorr<C>,
        q: Projective<C>,
        sg: Projective<C>,
        b0: C::Scalar,
        u: Projective<C>,
        h: Projective<C>,
    ) -> (Projective<C>, Tape<F, &'a mut O, Next>)
    where
        O: IpaOracle<C>,
        M: ChallengeMap<C::Scalar>,
    {
        let (raw, tape) = tape.receive(&Delta(last.delta)).squeeze();
        let c = map.map(raw);
        let expression = q * c + last.delta - (sg + u * b0) * last.z1 - h * last.z2;
        (expression, tape)
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
        let tape: OpeningTape<'_, C, B, V, H, O> = Tape::undescribed(oracle);
        let tape = self.binding.bind(tape, statement);
        let (u, mut tape) = self.value_base.base(tape, &self.u);
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
            let (raw, next) = tape.round().send(&RoundPoints(l, r)).squeeze();
            tape = next;
            let x = self.challenges.map(raw);
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
        let (last, tape) = self
            .hiding
            .prove_final(tape.done(), &self.challenges, &folded, rng);
        let _ = tape.into_inner();
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
        let b0 =
            |chals: &[C::Scalar]| folded_evaluation(chals, statement.points, statement.evalscale);
        self.check_given(oracle, statement, opening, b0)
    }

    /// [`GenericIpa::check`] with the folded evaluation vector `b0` given
    /// by `b0` from the round challenges, instead of computed from the
    /// statement's points. With a `b0` that ignores the challenges and
    /// returns a claimed value, this is the group part of the check alone:
    /// it holds up to `sg` and up to the claim that `b0` is
    /// [`folded_evaluation`] of the challenges, which a verifier in the
    /// other field checks (Pickles' deferred `b`).
    pub fn check_given<O, F0>(
        &self,
        oracle: &mut O,
        statement: &Statement<'_, C>,
        opening: &Opening<C, H::Final>,
        b0: F0,
    ) -> Option<(RecursionChallenge<C>, Projective<C>)>
    where
        O: IpaOracle<C>,
        B: Binding<C, O>,
        V: ValueBase<C, O>,
        F0: FnOnce(&[C::Scalar]) -> C::Scalar,
    {
        let k = self.rounds();
        if opening.rounds.len() != k {
            return None;
        }
        let tape: OpeningTape<'_, C, B, V, H, O> = Tape::undescribed(oracle);
        let tape = self.binding.bind(tape, statement);
        let (u, mut tape) = self.value_base.base(tape, &self.u);
        let mut chals = Vec::with_capacity(k);
        let mut q = *statement.commitment + u * statement.value;
        for (l, r) in &opening.rounds {
            let (raw, next) = tape.round().receive(&RoundPoints(*l, *r)).squeeze();
            tape = next;
            let x = self.challenges.map(raw);
            let x_inv = x.inverse()?;
            q = q + *l * x_inv + *r * x;
            chals.push(x);
        }
        let b0 = b0(&chals);
        let (expression, tape) = self.hiding.check_final(
            tape.done(),
            &self.challenges,
            &opening.last,
            q,
            opening.sg,
            b0,
            u,
            self.h,
        );
        let _ = tape.into_inner();
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
