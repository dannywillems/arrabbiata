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

use arrabbiata_field::Field;
use arrabbiata_pasta::{Fp, Fq};
use arrabbiata_transcript::{Permutation, Transcript};

/// The number of rounds, all full.
pub const ROUNDS: usize = 55;

/// The width of the state.
pub const WIDTH: usize = 3;

/// The rate of the sponge: the capacity is one element.
pub const RATE: usize = 2;

/// Kimchi's Poseidon permutation of `F^3`.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fp;
/// use arrabbiata_poseidon::kimchi::{ROUNDS, pallas_base};
/// assert_eq!(pallas_base().round_constants().len(), ROUNDS);
/// ```
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
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::kimchi::{KimchiPoseidon, ROUNDS};
    /// let identity = [[Fp::ONE, Fp::ZERO, Fp::ZERO], [Fp::ZERO, Fp::ONE, Fp::ZERO], [Fp::ZERO, Fp::ZERO, Fp::ONE]];
    /// let no_constants = [[Fp::ZERO; 3]; ROUNDS];
    /// let power = KimchiPoseidon::new(identity, no_constants);
    /// let mut state = [Fp::ONE; 3];
    /// power.permute_state(&mut state);
    /// assert_eq!(state, [Fp::ONE; 3]);
    /// ```
    #[must_use]
    pub const fn new(mds: [[F; WIDTH]; WIDTH], round_constants: [[F; WIDTH]; ROUNDS]) -> Self {
        Self {
            mds,
            round_constants,
        }
    }

    /// The MDS matrix, row by row.
    ///
    /// ```
    /// use arrabbiata_poseidon::kimchi::vesta_base;
    /// assert_eq!(vesta_base().mds().len(), 3);
    /// ```
    #[must_use]
    pub const fn mds(&self) -> &[[F; WIDTH]; WIDTH] {
        &self.mds
    }

    /// The round constants, one triple per round.
    ///
    /// ```
    /// use arrabbiata_poseidon::kimchi::{ROUNDS, vesta_base};
    /// assert_eq!(vesta_base().round_constants().len(), ROUNDS);
    /// ```
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
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::kimchi::pallas_base;
    /// let poseidon = pallas_base();
    /// // The zero state goes to the first round's constants.
    /// let mut state = [Fp::ZERO; 3];
    /// poseidon.round(0, &mut state);
    /// assert_eq!(&state, &poseidon.round_constants()[0]);
    /// ```
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
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::kimchi::pallas_base;
    /// let mut state = [Fp::ZERO; 3];
    /// pallas_base().permute_state(&mut state);
    /// assert_ne!(state, [Fp::ZERO; 3]);
    /// ```
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
///
/// ```
/// use arrabbiata_poseidon::kimchi::pallas_base;
/// let _ = pallas_base();
/// ```
#[must_use]
pub const fn pallas_base() -> KimchiPoseidon<Fp> {
    KimchiPoseidon::new(fp::MDS, fp::ROUND_CONSTANTS)
}

/// Kimchi's permutation over the Vesta base field `Fq` (`fq_kimchi`).
///
/// ```
/// use arrabbiata_poseidon::kimchi::vesta_base;
/// let _ = vesta_base();
/// ```
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

/// Kimchi's sponge (`ArithmeticSponge`): the zero state, the rate the first
/// two elements.
///
/// - absorbing adds to the next rate element, permuting first when the rate
///   is full; right after a squeeze it adds to element 0 without permuting;
/// - squeezing after absorbing permutes and returns element 0; squeezing
///   again returns element 1, and permutes only once the rate is used up.
///
/// So `[a]` and `[a, 0]` collide, as Kimchi documents: the sponge is meant
/// for inputs of a fixed length.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fp;
/// use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base};
/// use arrabbiata_transcript::Transcript;
/// let mut a = KimchiSponge::new(pallas_base());
/// let mut b = KimchiSponge::new(pallas_base());
/// a.absorb(Fp::ONE);
/// b.absorb(Fp::ONE);
/// b.absorb(Fp::ZERO);
/// assert_eq!(a.squeeze(), b.squeeze());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KimchiSponge<F> {
    permutation: KimchiPoseidon<F>,
    state: [F; WIDTH],
    mode: Mode,
}

impl<F> KimchiSponge<F>
where
    F: Field,
{
    /// A fresh sponge: the zero state, nothing absorbed.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq;
    /// use arrabbiata_poseidon::kimchi::{KimchiSponge, vesta_base};
    /// assert_eq!(KimchiSponge::new(vesta_base()).state(), &[Fq::ZERO; 3]);
    /// ```
    #[must_use]
    pub fn new(permutation: KimchiPoseidon<F>) -> Self {
        Self {
            permutation,
            state: [F::ZERO; WIDTH],
            mode: Mode::Absorbed(0),
        }
    }

    /// The state.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fp;
    /// use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base};
    /// use arrabbiata_transcript::Transcript;
    /// let mut sponge = KimchiSponge::new(pallas_base());
    /// sponge.absorb(Fp::ONE);
    /// assert_eq!(sponge.state(), &[Fp::ONE, Fp::ZERO, Fp::ZERO]);
    /// ```
    #[must_use]
    pub const fn state(&self) -> &[F; WIDTH] {
        &self.state
    }
}

impl<F> Transcript<F> for KimchiSponge<F>
where
    F: Field,
{
    fn absorb(&mut self, value: F) {
        let position = match self.mode {
            Mode::Absorbed(RATE) => {
                self.permutation.permute_state(&mut self.state);
                0
            }
            Mode::Absorbed(n) => n,
            Mode::Squeezed(_) => 0,
        };
        self.state[position] = self.state[position] + value;
        self.mode = Mode::Absorbed(position + 1);
    }

    fn squeeze(&mut self) -> F {
        let position = match self.mode {
            Mode::Squeezed(n) if n < RATE => n,
            _ => {
                self.permutation.permute_state(&mut self.state);
                0
            }
        };
        self.mode = Mode::Squeezed(position + 1);
        self.state[position]
    }
}

/// Kimchi's hash of `inputs`: absorb them all, squeeze one element.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fp;
/// use arrabbiata_poseidon::kimchi::{hash, pallas_base};
/// assert_ne!(hash(pallas_base(), &[Fp::ONE]), hash(pallas_base(), &[Fp::ZERO]));
/// ```
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
