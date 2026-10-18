//! Kimchi's Poseidon (o1-labs/proof-systems, crate `mina-poseidon`,
//! `PlonkSpongeConstantsKimchi`): width 3, rate 2, 55 full rounds and no
//! partial round, the S-box `x^7`, a full MDS matrix, and no round key
//! before the first round.
//!
//! It differs from the Hades instances of the crate root in three ways, each
//! of which changes every output:
//!
//! - every round is full, and a round is S-box, then MDS, then round
//!   constants ([`KimchiPoseidon::round`]), where Hades adds the constants
//!   first;
//! - the constants and the matrix are Kimchi's own ([`pallas_base`],
//!   [`vesta_base`]), not the Grain LFSR's for these parameters;
//! - the sponge ([`KimchiSponge`]) starts from the zero state, with no
//!   domain in the capacity, and squeezes the rate before permuting again.
//!
//! The instances are checked against `poseidon/tests/test_vectors/kimchi.json`
//! and the sponge's regression challenges, in `tests/kimchi.rs`.

mod fp;
mod fq;

use arrabbiata_field::{Field, FieldModel, Native};
use arrabbiata_pasta::{Fp, Fq};
use arrabbiata_transcript::{Permutation, PermutationIn, Transcript};

/// The number of rounds, all full.
pub const ROUNDS: usize = 55;

/// The width of the state.
pub const WIDTH: usize = 3;

/// The rate of the sponge: the capacity is one element.
pub const RATE: usize = 2;

/// Kimchi's Poseidon permutation of `F^3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KimchiPoseidon<F> {
    mds: [[F; WIDTH]; WIDTH],
    round_constants: [[F; WIDTH]; ROUNDS],
}

impl<F> KimchiPoseidon<F>
where
    F: Field,
{
    /// The permutation with the matrix `mds` and the constants
    /// `round_constants[r]` of round `r`.
    #[must_use]
    pub const fn new(mds: [[F; WIDTH]; WIDTH], round_constants: [[F; WIDTH]; ROUNDS]) -> Self {
        Self {
            mds,
            round_constants,
        }
    }

    /// The MDS matrix, row by row.
    #[must_use]
    pub const fn mds(&self) -> &[[F; WIDTH]; WIDTH] {
        &self.mds
    }

    /// The round constants, one triple per round.
    #[must_use]
    pub const fn round_constants(&self) -> &[[F; WIDTH]; ROUNDS] {
        &self.round_constants
    }

    /// Round `r`: `x -> x^7` on every element, then the MDS matrix, then the
    /// constants of round `r`.
    ///
    /// # Panics
    ///
    /// If `r >= ROUNDS`.
    pub fn round(&self, r: usize, state: &mut [F; WIDTH]) {
        let powered = state.map(sbox);
        let constants = &self.round_constants[r];
        for (i, row) in self.mds.iter().enumerate() {
            let mut sum = constants[i];
            for (m, x) in row.iter().zip(&powered) {
                sum = sum + *m * *x;
            }
            state[i] = sum;
        }
    }

    /// The permutation: the [`ROUNDS`] rounds in order.
    pub fn permute_state(&self, state: &mut [F; WIDTH]) {
        for r in 0..ROUNDS {
            self.round(r, state);
        }
    }
}

/// `x^7`, by two squarings and two multiplications.
fn sbox<F>(x: F) -> F
where
    F: Field,
{
    let square = x * x;
    let fourth = square * square;
    x * square * fourth
}

impl<F> Permutation<F, WIDTH> for KimchiPoseidon<F>
where
    F: Field,
{
    fn permute(&self, state: &mut [F; WIDTH]) {
        self.permute_state(state);
    }
}

/// Kimchi's permutation over the Pallas base field `Fp` (`fp_kimchi`), the
/// sponge of Kimchi proofs over Vesta and of the Fq-sponge over Pallas.
#[must_use]
pub const fn pallas_base() -> KimchiPoseidon<Fp> {
    KimchiPoseidon::new(fp::MDS, fp::ROUND_CONSTANTS)
}

/// Kimchi's permutation over the Vesta base field `Fq` (`fq_kimchi`).
#[must_use]
pub const fn vesta_base() -> KimchiPoseidon<Fq> {
    KimchiPoseidon::new(fq::MDS, fq::ROUND_CONSTANTS)
}

/// Where the sponge is: absorbing into, or squeezing out of, the `n`-th
/// element of the rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Absorbed(usize),
    Squeezed(usize),
}

/// The state and the mode of Kimchi's sponge, in any model of the field
/// signature: the duplex alone, without its permutation. [`KimchiSponge`]
/// is it in the field itself; a circuit runs the same machine on its
/// values, so the two cannot drift apart.
///
/// - absorbing adds to the next rate element, permuting first when the rate
///   is full; right after a squeeze it adds to element 0 without permuting;
/// - squeezing after absorbing permutes and returns element 0; squeezing
///   again returns element 1, and permutes only once the rate is used up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KimchiDuplex<V> {
    state: [V; WIDTH],
    mode: Mode,
}

impl<V> KimchiDuplex<V>
where
    V: Clone,
{
    /// The zero state, nothing absorbed; `zero` is the model's zero.
    #[must_use]
    pub fn new(zero: V) -> Self {
        Self {
            state: core::array::from_fn(|_| zero.clone()),
            mode: Mode::Absorbed(0),
        }
    }

    /// The state.
    #[must_use]
    pub const fn state(&self) -> &[V; WIDTH] {
        &self.state
    }

    /// The state permuted by `permutation`, in `model`.
    fn permute<F, M, P>(&mut self, model: &mut M, permutation: &P)
    where
        F: Field,
        M: FieldModel<F, Value = V>,
        P: PermutationIn<M, F, WIDTH>,
    {
        self.state = permutation.permute_in(model, self.state.clone());
    }

    /// Absorb `value`.
    pub fn absorb<F, M, P>(&mut self, model: &mut M, permutation: &P, value: &V)
    where
        F: Field,
        M: FieldModel<F, Value = V>,
        P: PermutationIn<M, F, WIDTH>,
    {
        let position = match self.mode {
            Mode::Absorbed(RATE) => {
                self.permute(model, permutation);
                0
            }
            Mode::Absorbed(n) => n,
            Mode::Squeezed(_) => 0,
        };
        self.state[position] = model.add(&self.state[position], value);
        self.mode = Mode::Absorbed(position + 1);
    }

    /// Squeeze an element.
    pub fn squeeze<F, M, P>(&mut self, model: &mut M, permutation: &P) -> V
    where
        F: Field,
        M: FieldModel<F, Value = V>,
        P: PermutationIn<M, F, WIDTH>,
    {
        let position = match self.mode {
            Mode::Squeezed(n) if n < RATE => n,
            _ => {
                self.permute(model, permutation);
                0
            }
        };
        self.mode = Mode::Squeezed(position + 1);
        self.state[position].clone()
    }
}

/// Kimchi's sponge (`ArithmeticSponge`): [`KimchiDuplex`] in the field
/// itself, over Kimchi's permutation.
///
/// `[a]` and `[a, 0]` collide, as Kimchi documents: the sponge is meant for
/// inputs of a fixed length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KimchiSponge<F> {
    permutation: KimchiPoseidon<F>,
    duplex: KimchiDuplex<F>,
}

impl<F> KimchiSponge<F>
where
    F: Field,
{
    /// A fresh sponge: the zero state, nothing absorbed.
    #[must_use]
    pub fn new(permutation: KimchiPoseidon<F>) -> Self {
        Self {
            permutation,
            duplex: KimchiDuplex::new(F::ZERO),
        }
    }

    /// The state.
    #[must_use]
    pub const fn state(&self) -> &[F; WIDTH] {
        self.duplex.state()
    }
}

impl<F> Transcript<F> for KimchiSponge<F>
where
    F: Field,
{
    fn absorb(&mut self, value: F) {
        self.duplex.absorb(&mut Native, &self.permutation, &value);
    }

    fn squeeze(&mut self) -> F {
        self.duplex.squeeze(&mut Native, &self.permutation)
    }
}

/// Kimchi's hash of `inputs`: absorb them all, squeeze one element.
#[must_use]
pub fn hash<F>(permutation: KimchiPoseidon<F>, inputs: &[F]) -> F
where
    F: Field,
{
    let mut sponge = KimchiSponge::new(permutation);
    for x in inputs {
        sponge.absorb(*x);
    }
    sponge.squeeze()
}
