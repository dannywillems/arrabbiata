//! PlonK over any number of witness columns, fixed columns and selectors.
//!
//! The arithmetization is the PlonKish one of the original Arrabbiata
//! circuits: a table of `W` witness columns and `X` fixed columns, a set of
//! [gadgets](gate::Gadget) each turned on by its own selector, at most one
//! gadget per row (one-hot by construction), constraints that read any row
//! at a fixed offset from the constrained one ([`Rotation`]), copy
//! constraints between any two witness cells, and public inputs in witness
//! column 0. The gate set is a type ([`gate::GateSet`]), so the index of a
//! selector is the position of its gadget in the type, and the protocol
//! type [`Plonk`] fixes the gate set, the scheme and the sponge.
//!
//! The protocol compiles this polynomial IOP (Gabizon, Williamson,
//! Ciobotaru, ePrint 2019/953, generalized to custom gates) with any
//! univariate [`PolynomialCommitment`], over any [`TwoAdicField`], on the
//! typed transcript of [`arrabbiata_transcript::typed`]. Vanilla PlonK is
//! the instance with three witness columns, five fixed columns and one
//! gadget ([`vanilla`]).
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::{
//!     circuit::{Builder, Cell},
//!     gate::Gadget,
//!     ipa, prove, setup, verify,
//! };
//! use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
//!
//! /// The Poseidon S-box, `b = a^5`.
//! struct SBox;
//! impl Gadget<F, 2, 0> for SBox {
//!     const NAME: &'static str = "sbox";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
//!         let a2 = a.clone() * a.clone();
//!         vec![b - a2.clone() * a2 * a]
//!     }
//! }
//!
//! /// `a' = b`: the output of a row is the input of the next.
//! struct Chain;
//! impl Gadget<F, 2, 0> for Chain {
//!     const NAME: &'static str = "chain";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
//!         let next = Expr::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
//!         vec![next - b]
//!     }
//! }
//!
//! // 2 -> 2^5 -> 2^25, with 2 and 2^25 public.
//! type Gates = (SBox, (Chain, ()));
//! let mut b = Builder::<F, Gates, 2, 0, 2>::new();
//! let x = F::from_u64(2);
//! let y = x.pow(5);
//! let z = y.pow(5);
//! let input = b.plain_row([x, F::ZERO], []);
//! let r0 = b.row::<SBox, _>([x, y], []);
//! let r1 = b.row::<SBox, _>([y, z], []);
//! let output = b.plain_row([z, F::ZERO], []);
//! let (a, c) = (WitnessColumn::new::<0>(), WitnessColumn::new::<1>());
//! b.copy(Cell::new(a, input), Cell::new(a, r0));
//! b.copy(Cell::new(c, r0), Cell::new(a, r1));
//! b.copy(Cell::new(c, r1), Cell::new(a, output));
//! b.public(input);
//! b.public(output);
//! let (circuit, witness) = b.build().unwrap();
//!
//! let pcs = ipa::pallas(&circuit);
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
//! verify(&vk, &pcs, ipa::pallas_transcript(), &[x, z], proof).unwrap();
//! ```
//!
//! # The protocol
//!
//! With `n` rows, `H` the subgroup of order `n`, `w_k` the witness, `f_k`
//! the fixed and `q_g` the selector polynomials, the rounds follow
//! [`PlonkTranscript`]:
//!
//! 1. the prover commits to `w_0, ..., w_(W-1)`;
//! 2. the verifier sends `beta, gamma`; the prover commits to the grand
//!    product `z` of the copy-constraint argument over the `W` columns,
//!    with the cosets `g^k H` labelling column `k`;
//! 3. the verifier sends `alpha`; the prover commits to the quotient `t`
//!    of
//!    `sum_j alpha^j q_g(j) C_j + alpha^J (L_P w_0 - PI)
//!     + alpha^(J+1) perm + alpha^(J+2) L_1 (z - 1)` by `Z_H`, where `C_j`
//!    ranges over the `J` constraints of all gadgets, `L_P` is 1 on the
//!    public rows and 0 elsewhere, and `PI` interpolates the public inputs;
//! 4. the verifier sends `zeta`; the prover sends every polynomial at
//!    `zeta`, `z` at `zeta w`, and each column read at a rotation `k` at
//!    `zeta w^k`, with their opening proofs ([`Opened`]);
//! 5. the verifier checks the openings and the identity at `zeta`.
//!
//! The quotient is computed on a coset of a domain of size at least
//! `quotient_factor() * n`, pointwise, so a gadget of any degree costs one
//! NTT per column rather than polynomial products.
//!
//! # Deviations from the paper (why this is a toy)
//!
//! - No blinding factors: the proof is not zero-knowledge.
//! - No linearization and no batched openings: every polynomial is opened
//!   separately. This changes the cost, not the soundness argument.
//! - The quotient is committed as one polynomial rather than in chunks,
//!   which matters only for schemes with a degree bound (KZG).

#![no_std]

extern crate alloc;

pub mod ipa;
pub mod kzg;
pub mod relaxed;
pub mod vanilla;

use alloc::{collections::BTreeSet, vec, vec::Vec};
use core::{array, fmt, iter, marker::PhantomData};

use arrabbiata_commit::{PolynomialCommitment, Univariate};
use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_poly::{
    Assignment, Domain, Expr, FixedColumn, Rotation, Scalars, UniPoly, Variable, WitnessColumn,
    evaluate,
};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, absorb_slice,
    typed::{
        Challenge, End, Label, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape,
    },
};

pub use arrabbiata_plonkish::{
    CircuitError, circuit,
    circuit::{Builder, Cell, Circuit, Witness},
    gate,
    gate::{Gadget, GateSet, Member},
};

/// The commitment type of a scheme.
type Com<F, PC> = <PC as PolynomialCommitment<F>>::Commitment;

/// The opening proof type of a scheme.
type Prf<F, PC> = <PC as PolynomialCommitment<F>>::Proof;

/// The keys produced by [`setup`].
pub type Keys<F, PC, G, const W: usize, const X: usize, const Q: usize> =
    (ProverKey<F, PC, G, W, X, Q>, VerifierKey<F, PC, G, W, X, Q>);

/// A value of type `T` produced by the polynomial commitment scheme `PC`.
///
/// Every message of the PlonK transcript, and the verifier key, carries the
/// scheme in its type and in its label: a proof made with one scheme is not
/// a proof for another, even when the two use the same commitment type (IPA
/// and KZG both commit to curve points). The type checker rejects the mix,
/// and the transcripts differ from the first challenge.
///
/// ```
/// use arrabbiata_plonk::{Scheme, ipa::PallasIpa};
/// let m: Scheme<PallasIpa, u8, 1> = Scheme::new(7);
/// assert_eq!(m.value, 7);
/// ```
///
/// A proof made with one scheme does not type-check against the verifier of
/// another, even when their commitment and proof types coincide. Here `Twin`
/// is IPA on Pallas under another label:
///
/// ```compile_fail
/// use arrabbiata_commit::{PolynomialCommitment, Univariate};
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa, ipa::PallasIpa, prove, setup, vanilla, verify};
/// use arrabbiata_transcript::typed::Label;
///
/// type Point = <PallasIpa as PolynomialCommitment<F>>::Commitment;
/// type IpaProof = <PallasIpa as PolynomialCommitment<F>>::Proof;
/// struct Twin(PallasIpa);
/// impl Label for Twin {
///     const LABEL: u64 = 0x7477;
/// }
/// impl PolynomialCommitment<F> for Twin {
///     type Family = Univariate;
///     type Commitment = Point;
///     type Proof = IpaProof;
///     fn commit(&self, p: &[F]) -> Point { self.0.commit(p) }
///     fn open(&self, p: &[F], x: &F) -> (F, IpaProof) { self.0.open(p, x) }
///     fn verify(&self, c: &Point, x: &F, v: F, pr: &IpaProof) -> bool { self.0.verify(c, x, v, pr) }
/// }
///
/// let (c, w, ()) = build::<F, _, _>(|b| {
///     b.witness(F::from_u64(1));
/// });
/// let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
/// let pcs = ipa::pallas(&circuit);
/// let twin = Twin(ipa::pallas(&circuit));
/// let (pk, _) = setup(&circuit, &pcs).unwrap();
/// let (_, twin_vk) = setup(&circuit, &twin).unwrap();
/// let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
/// // A proof under `pcs` handed to the verifier of `twin`.
/// let _ = verify(&twin_vk, &twin, ipa::pallas_transcript(), &[], proof);
/// ```
pub struct Scheme<PC, T, const TAG: u64> {
    /// The value.
    pub value: T,
    scheme: PhantomData<fn() -> PC>,
}

impl<PC, T, const TAG: u64> Scheme<PC, T, TAG> {
    /// Tag `value` with the scheme `PC`.
    ///
    /// ```
    /// use arrabbiata_plonk::{Scheme, ipa::PallasIpa};
    /// let m = Scheme::<PallasIpa, _, 1>::new("x");
    /// assert_eq!(m.value, "x");
    /// ```
    pub const fn new(value: T) -> Self {
        Self {
            value,
            scheme: PhantomData,
        }
    }
}

impl<PC, T, const TAG: u64> Clone for Scheme<PC, T, TAG>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl<PC, T, const TAG: u64> fmt::Debug for Scheme<PC, T, TAG>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Scheme").field(&self.value).finish()
    }
}

impl<PC, T, const TAG: u64> PartialEq for Scheme<PC, T, TAG>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<PC, T, const TAG: u64> Eq for Scheme<PC, T, TAG> where T: Eq {}

impl<PC, T, const TAG: u64> Label for Scheme<PC, T, TAG>
where
    PC: Label,
{
    const LABEL: u64 = message_label(PC::LABEL, TAG);
}

/// The label of a message tagged `tag` from the scheme labelled `scheme`:
/// the scheme label rotated by 16 bits, combined with the tag. For scheme
/// labels below `2^48` the two occupy disjoint bits, so the map is
/// injective in the pair.
const fn message_label(scheme: u64, tag: u64) -> u64 {
    scheme.rotate_left(16) ^ tag
}

impl<F, PC, T, const TAG: u64> Absorb<F> for Scheme<PC, T, TAG>
where
    F: Field,
    PC: Label,
    T: Absorb<F>,
{
    fn absorb_into<Tr>(&self, transcript: &mut Tr)
    where
        Tr: Transcript<F>,
    {
        transcript.absorb(F::from_u64(<Self as Label>::LABEL));
        self.value.absorb_into(transcript);
    }
}

/// The commitment scheme `PC` and the sponge `S` a proof is made with.
///
/// Every message of a PlonK proof is tagged with both, so a proof is a
/// proof for one scheme and one Fiat-Shamir sponge: a verifier running
/// another sponge (or another scheme) does not type-check against it. The
/// label is the scheme's; the sponge is separated by its own type and state.
///
/// ```
/// use arrabbiata_pasta::Fq;
/// use arrabbiata_plonk::{With, ipa::PallasIpa};
/// use arrabbiata_poseidon::PoseidonSponge;
/// use arrabbiata_transcript::typed::Label;
/// type Setup = With<PallasIpa, PoseidonSponge<Fq>>;
/// assert_eq!(Setup::LABEL, <PallasIpa as Label>::LABEL);
/// ```
///
/// A proof made on the Poseidon sponge is not accepted by a verifier running
/// another sponge, even over the same field and scheme. Here the verifier
/// runs Poseidon of width 3 at rate 1 instead of rate 2:
///
/// ```compile_fail
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa, prove, setup, vanilla, verify};
/// use arrabbiata_poseidon::{Poseidon, vesta_base};
/// use arrabbiata_transcript::Duplex;
///
/// let (c, w, ()) = build::<F, _, _>(|b| {
///     b.witness(F::from_u64(1));
/// });
/// let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
/// let pcs = ipa::pallas(&circuit);
/// let (pk, vk) = setup(&circuit, &pcs).unwrap();
/// let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
/// let rate_one = Duplex::<F, Poseidon<F, 3>, 3, 1>::new(vesta_base(), 0);
/// let _ = verify(&vk, &pcs, rate_one, &[], proof);
/// ```
#[derive(Debug)]
pub struct With<PC, S>(PhantomData<fn() -> (PC, S)>);

impl<PC, S> Label for With<PC, S>
where
    PC: Label,
{
    const LABEL: u64 = PC::LABEL;
}

/// Commitments to the `W` witness polynomials.
pub type Wires<F, PC, S, const W: usize> = Scheme<With<PC, S>, Columns<Com<F, PC>, W>, 0x7769>;
/// Commitment to the grand product `z`.
pub type GrandProduct<F, PC, S> = Scheme<With<PC, S>, Com<F, PC>, 0x7a7a>;
/// Commitment to the quotient `t`.
pub type Quotient<F, PC, S> = Scheme<With<PC, S>, Com<F, PC>, 0x7474>;
/// The claimed evaluations.
pub type Evaluations<F, const W: usize, const X: usize, const Q: usize> = Opened<F, W, X, Q>;
/// The opening proofs.
pub type Openings<F, PC, S, const W: usize, const X: usize, const Q: usize> =
    Scheme<With<PC, S>, Opened<Prf<F, PC>, W, X, Q>, 0x6f70>;
/// The verifier key: the preprocessed circuit, committed with `PC`.
pub type VerifierKey<F, PC, G, const W: usize, const X: usize, const Q: usize> =
    Scheme<PC, Preprocessed<F, Com<F, PC>, G, W, X, Q>, 0x766b>;

/// `N` values of the same type, absorbed in order.
///
/// ```
/// use arrabbiata_plonk::Columns;
/// assert_eq!(Columns([1, 2, 3]).0[2], 3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Columns<C, const N: usize>(pub [C; N]);

impl<F, C, const N: usize> Absorb<F> for Columns<C, N>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        for c in &self.0 {
            c.absorb_into(transcript);
        }
    }
}

/// The public inputs of a statement.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_plonk::PublicInputs;
/// assert_eq!(PublicInputs(vec![F::from_u64(1)]).0.len(), 1);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInputs<F>(pub Vec<F>);

impl<F> Label for PublicInputs<F> {
    const LABEL: u64 = 0x7069;
}

impl<F> Absorb<F> for PublicInputs<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.0);
    }
}

/// One value per opening: an evaluation (`T = F`) or an opening proof.
///
/// The order, used by [`Opened::iter`] and [`PlonkError::Opening`], is:
/// the witness, fixed, selector and permutation polynomials, `z` and `t`
/// at `zeta`; `z` at `zeta w`; then each pair `(column, k)` of
/// [`Opened::rotated_witness`] and of [`Opened::rotated_fixed`] at
/// `zeta w^k`.
///
/// ```
/// use arrabbiata_plonk::Opened;
/// let o = Opened::<u8, 1, 1, 1> {
///     witness: [0], fixed: [1], selectors: [2], sigmas: [3], z: 4, t: 5,
///     z_next: 6, witness_rotated: vec![], fixed_rotated: vec![],
/// };
/// assert_eq!(o.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5, 6]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opened<T, const W: usize, const X: usize, const Q: usize> {
    /// The witness polynomials at `zeta`.
    pub witness: [T; W],
    /// The fixed polynomials at `zeta`.
    pub fixed: [T; X],
    /// The selector polynomials at `zeta`.
    pub selectors: [T; Q],
    /// The permutation polynomials at `zeta`.
    pub sigmas: [T; W],
    /// The grand product at `zeta`.
    pub z: T,
    /// The quotient at `zeta`.
    pub t: T,
    /// The grand product at `zeta w`.
    pub z_next: T,
    /// The witness polynomials read at a rotation `k`, at `zeta w^k`, in
    /// the order of [`Opened::rotated_witness`].
    pub witness_rotated: Vec<T>,
    /// The fixed polynomials read at a rotation `k`, at `zeta w^k`, in the
    /// order of [`Opened::rotated_fixed`].
    pub fixed_rotated: Vec<T>,
}

impl<T, const W: usize, const X: usize, const Q: usize> Opened<T, W, X, Q> {
    /// Every value, in the order of the openings.
    ///
    /// ```
    /// use arrabbiata_plonk::Opened;
    /// let o = Opened::<u8, 2, 0, 0> {
    ///     witness: [1, 2], fixed: [], selectors: [], sigmas: [3, 4], z: 5, t: 6,
    ///     z_next: 7, witness_rotated: vec![8], fixed_rotated: vec![],
    /// };
    /// assert_eq!(o.iter().count(), 8);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.witness
            .iter()
            .chain(&self.fixed)
            .chain(&self.selectors)
            .chain(&self.sigmas)
            .chain(iter::once(&self.z))
            .chain(iter::once(&self.t))
            .chain(iter::once(&self.z_next))
            .chain(&self.witness_rotated)
            .chain(&self.fixed_rotated)
    }

    /// Every value, mutably, in the order of the openings.
    ///
    /// ```
    /// use arrabbiata_plonk::Opened;
    /// let mut o = Opened::<u8, 1, 0, 0> {
    ///     witness: [1], fixed: [], selectors: [], sigmas: [2], z: 3, t: 4,
    ///     z_next: 5, witness_rotated: vec![], fixed_rotated: vec![],
    /// };
    /// *o.iter_mut().nth(4).unwrap() = 9;
    /// assert_eq!(o.z_next, 9);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.witness
            .iter_mut()
            .chain(&mut self.fixed)
            .chain(&mut self.selectors)
            .chain(&mut self.sigmas)
            .chain(iter::once(&mut self.z))
            .chain(iter::once(&mut self.t))
            .chain(iter::once(&mut self.z_next))
            .chain(&mut self.witness_rotated)
            .chain(&mut self.fixed_rotated)
    }

    /// The witness columns some constraint of `G` reads at a rotation
    /// other than the current row, with the rotation, in increasing order.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{Opened, vanilla::Vanilla};
    /// // Vanilla PlonK reads only the current row.
    /// assert!(Opened::<F, 3, 5, 1>::rotated_witness::<F, Vanilla>().is_empty());
    /// ```
    #[must_use]
    pub fn rotated_witness<F, G>() -> Vec<(usize, Rotation)>
    where
        F: Field,
        G: GateSet<F, W, X>,
    {
        rotations::<F, G, W, X>().0
    }

    /// The fixed columns some constraint of `G` reads at a rotation other
    /// than the current row, with the rotation, in increasing order.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{Opened, vanilla::Vanilla};
    /// assert!(Opened::<F, 3, 5, 1>::rotated_fixed::<F, Vanilla>().is_empty());
    /// ```
    #[must_use]
    pub fn rotated_fixed<F, G>() -> Vec<(usize, Rotation)>
    where
        F: Field,
        G: GateSet<F, W, X>,
    {
        rotations::<F, G, W, X>().1
    }

    fn map<U>(self, mut f: impl FnMut(T) -> U) -> Opened<U, W, X, Q> {
        Opened {
            witness: self.witness.map(&mut f),
            fixed: self.fixed.map(&mut f),
            selectors: self.selectors.map(&mut f),
            sigmas: self.sigmas.map(&mut f),
            z: f(self.z),
            t: f(self.t),
            z_next: f(self.z_next),
            witness_rotated: self.witness_rotated.into_iter().map(&mut f).collect(),
            fixed_rotated: self.fixed_rotated.into_iter().map(&mut f).collect(),
        }
    }
}

impl<T, const W: usize, const X: usize, const Q: usize> Label for Opened<T, W, X, Q> {
    const LABEL: u64 = 0x6576;
}

impl<F, T, const W: usize, const X: usize, const Q: usize> Absorb<F> for Opened<T, W, X, Q>
where
    F: Field,
    T: Absorb<F>,
{
    fn absorb_into<Tr>(&self, transcript: &mut Tr)
    where
        Tr: Transcript<F>,
    {
        absorb_length(transcript, self.witness_rotated.len());
        absorb_length(transcript, self.fixed_rotated.len());
        for value in self.iter() {
            value.absorb_into(transcript);
        }
    }
}

/// The pairs `(column, rotation)` with a rotation other than the current
/// row that some constraint of `G` reads, for the witness and the fixed
/// columns, each sorted and without repetition.
type Rotated = Vec<(usize, Rotation)>;

fn rotations<F, G, const W: usize, const X: usize>() -> (Rotated, Rotated)
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut witness = BTreeSet::new();
    let mut fixed = BTreeSet::new();
    for gadget in 0..G::SELECTORS {
        for c in G::constraints(gadget) {
            let mut vars = Vec::new();
            gate::variables(&c, &mut vars);
            for v in vars {
                match v {
                    Variable::Witness(col, r) if r != Rotation::CURR => {
                        witness.insert((col.index(), r));
                    }
                    Variable::Fixed(col, r) if r != Rotation::CURR => {
                        fixed.insert((col.index(), r));
                    }
                    _ => {}
                }
            }
        }
    }
    (witness.into_iter().collect(), fixed.into_iter().collect())
}

/// `w^k` for the generator `w` of `domain`, for any `k` in `Z`.
fn root_power<F>(domain: &Domain<F>, rotation: Rotation) -> F
where
    F: TwoAdicField,
{
    let n = domain.size() as i64;
    domain
        .generator()
        .pow((rotation.offset() as i64).rem_euclid(n) as u64)
}

/// The verifier's view of a circuit: the domain, the coset shifts of the
/// permutation argument, and the commitments to the fixed, selector and
/// permutation polynomials. Wrapped with its scheme as a [`VerifierKey`].
///
/// Its absorption also absorbs the constraints of the gate set `G`, so the
/// transcript is bound to the relation, not only to the columns.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa, vanilla};
/// let (c, _, ()) = build::<F, _, _>(|b| {
///     b.witness(F::from_u64(1));
/// });
/// let circuit = vanilla::circuit(&c).unwrap();
/// let (_, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
/// assert_eq!(vk.value.num_rows(), 2);
/// ```
pub struct Preprocessed<F, C, G, const W: usize, const X: usize, const Q: usize> {
    log_size: u32,
    shifts: [F; W],
    fixed: [C; X],
    selectors: [C; Q],
    sigmas: [C; W],
    public_rows: Vec<usize>,
    gates: PhantomData<fn() -> G>,
}

impl<F, C, G, const W: usize, const X: usize, const Q: usize> Preprocessed<F, C, G, W, X, Q> {
    /// The number of rows of the circuit.
    ///
    /// ```
    /// use arrabbiata_frontend::vanilla::build;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{ipa, vanilla};
    /// let (c, _, ()) = build::<F, _, _>(|_| ());
    /// let circuit = vanilla::circuit(&c).unwrap();
    /// let (_, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    /// assert_eq!(vk.value.num_rows(), circuit.num_rows());
    /// ```
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        1 << self.log_size
    }

    /// The rows whose witness column 0 is a public input.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_frontend::vanilla::build;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{ipa, vanilla};
    /// let (c, _, ()) = build::<F, _, _>(|b| {
    ///     b.public_input(F::ONE);
    /// });
    /// let circuit = vanilla::circuit(&c).unwrap();
    /// let (_, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    /// assert_eq!(vk.value.public_rows(), circuit.public_rows());
    /// ```
    #[must_use]
    pub fn public_rows(&self) -> &[usize] {
        &self.public_rows
    }
}

impl<F, C, G, const W: usize, const X: usize, const Q: usize> Clone
    for Preprocessed<F, C, G, W, X, Q>
where
    F: Clone,
    C: Clone,
{
    fn clone(&self) -> Self {
        Self {
            log_size: self.log_size,
            shifts: self.shifts.clone(),
            fixed: self.fixed.clone(),
            selectors: self.selectors.clone(),
            sigmas: self.sigmas.clone(),
            public_rows: self.public_rows.clone(),
            gates: PhantomData,
        }
    }
}

impl<F, C, G, const W: usize, const X: usize, const Q: usize> fmt::Debug
    for Preprocessed<F, C, G, W, X, Q>
where
    F: fmt::Debug,
    C: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Preprocessed")
            .field("log_size", &self.log_size)
            .field("shifts", &self.shifts)
            .field("fixed", &self.fixed)
            .field("selectors", &self.selectors)
            .field("sigmas", &self.sigmas)
            .field("public_rows", &self.public_rows)
            .finish()
    }
}

impl<F, C, G, const W: usize, const X: usize, const Q: usize> PartialEq
    for Preprocessed<F, C, G, W, X, Q>
where
    F: PartialEq,
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.log_size == other.log_size
            && self.shifts == other.shifts
            && self.fixed == other.fixed
            && self.selectors == other.selectors
            && self.sigmas == other.sigmas
            && self.public_rows == other.public_rows
    }
}

impl<F, C, G, const W: usize, const X: usize, const Q: usize> Eq for Preprocessed<F, C, G, W, X, Q>
where
    F: Eq,
    C: Eq,
{
}

impl<F, C, G, const W: usize, const X: usize, const Q: usize> Absorb<F>
    for Preprocessed<F, C, G, W, X, Q>
where
    F: Field,
    C: Absorb<F>,
    G: GateSet<F, W, X>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_length(transcript, self.log_size as usize);
        absorb_length(transcript, W);
        absorb_length(transcript, X);
        absorb_length(transcript, Q);
        gate::absorb_gates::<F, G, T, W, X>(transcript);
        for shift in &self.shifts {
            transcript.absorb(*shift);
        }
        for c in self.fixed.iter().chain(&self.selectors).chain(&self.sigmas) {
            c.absorb_into(transcript);
        }
        absorb_length(transcript, self.public_rows.len());
        for row in &self.public_rows {
            absorb_length(transcript, *row);
        }
    }
}

/// The prover's view of a circuit: the verifier key, the circuit, and the
/// polynomials it commits to.
///
/// ```
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa, vanilla};
/// let (c, _, ()) = build::<F, _, _>(|_| ());
/// let circuit = vanilla::circuit(&c).unwrap();
/// let (pk, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
/// assert_eq!(pk.verifier_key(), &vk);
/// ```
pub struct ProverKey<F, PC, G, const W: usize, const X: usize, const Q: usize>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    vk: VerifierKey<F, PC, G, W, X, Q>,
    circuit: Circuit<F, G, W, X, Q>,
    domain: Domain<F>,
    fixed: [UniPoly<F>; X],
    selectors: [UniPoly<F>; Q],
    sigmas: [UniPoly<F>; W],
    sigma_values: [Vec<F>; W],
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize> ProverKey<F, PC, G, W, X, Q>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    /// The verifier key.
    ///
    /// ```
    /// use arrabbiata_frontend::vanilla::build;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{ipa, vanilla};
    /// let (c, _, ()) = build::<F, _, _>(|_| ());
    /// let circuit = vanilla::circuit(&c).unwrap();
    /// let (pk, _) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    /// assert_eq!(pk.verifier_key().value.num_rows(), 2);
    /// ```
    pub const fn verifier_key(&self) -> &VerifierKey<F, PC, G, W, X, Q> {
        &self.vk
    }
}

/// The transcript of PlonK over the gate set `G`, compiled with the scheme
/// `PC`, on the sponge `S`. The prover and the verifier both follow it,
/// through [`Plonk`].
pub type PlonkTranscript<F, PC, S, G, const W: usize, const X: usize, const Q: usize> = Public<
    VerifierKey<F, PC, G, W, X, Q>,
    Public<
        PublicInputs<F>,
        Message<
            Wires<F, PC, S, W>,
            Challenge<
                Challenge<
                    Message<
                        GrandProduct<F, PC, S>,
                        Challenge<
                            Message<
                                Quotient<F, PC, S>,
                                Challenge<
                                    Message<
                                        Evaluations<F, W, X, Q>,
                                        Message<Openings<F, PC, S, W, X, Q>, End>,
                                    >,
                                >,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >,
>;

/// PlonK over `F` for the gate set `G` with `W` witness columns, `X` fixed
/// columns and `Q` selectors, compiled with the univariate scheme `PC`, on
/// the sponge `S`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{Plonk, ipa::PallasIpa, vanilla::Vanilla};
/// use arrabbiata_poseidon::PoseidonSponge;
/// use arrabbiata_transcript::typed::{Pattern, Protocol};
/// type P = <Plonk<F, PallasIpa, PoseidonSponge<F>, Vanilla, 3, 5, 1> as Protocol>::Transcript;
/// assert_eq!(<P as Pattern>::MESSAGES, 5);
/// assert_eq!(<P as Pattern>::CHALLENGES, 4);
/// ```
#[derive(Debug)]
pub struct Plonk<F, PC, S, G, const W: usize, const X: usize, const Q: usize>(
    PhantomData<Parameters<F, PC, S, G>>,
);

/// The type parameters of [`Plonk`], as a marker that owns none of them.
type Parameters<F, PC, S, G> = fn() -> (F, PC, S, G);

impl<F, PC, S, G, const W: usize, const X: usize, const Q: usize> Protocol
    for Plonk<F, PC, S, G, W, X, Q>
where
    F: Field,
    PC: PolynomialCommitment<F, Family = Univariate>,
    S: Transcript<F>,
    G: GateSet<F, W, X>,
{
    type Transcript = PlonkTranscript<F, PC, S, G, W, X, Q>;
}

/// Why building a circuit, setup, proving or verification failed.
///
/// ```
/// use arrabbiata_plonk::PlonkError;
/// assert_eq!(PlonkError::IdentityFailed.to_string(), "the PlonK identity does not hold at zeta");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlonkError {
    /// The field has no domain of the size the circuit needs.
    DomainTooLarge,
    /// The witness does not satisfy the circuit.
    Unsatisfied,
    /// The number of public inputs does not match the circuit.
    PublicInputs,
    /// An opening proof does not verify; the index is in the order of
    /// [`Opened::iter`].
    Opening(usize),
    /// The PlonK identity does not hold at `zeta`.
    IdentityFailed,
    /// The coset shifts `g^k` of the permutation argument do not give `W`
    /// disjoint cosets of the domain.
    CosetShifts,
    /// A circuit, a witness or a proof has the wrong shape.
    Shape,
    /// A gadget reads a row outside the rows of a path (see
    /// [`circuit::Builder::new`]).
    Rotation,
}

impl fmt::Display for PlonkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainTooLarge => write!(f, "the field has no domain of the circuit size"),
            Self::Unsatisfied => write!(f, "the witness does not satisfy the circuit"),
            Self::PublicInputs => write!(f, "wrong number of public inputs"),
            Self::Opening(i) => write!(f, "opening proof {i} does not verify"),
            Self::IdentityFailed => write!(f, "the PlonK identity does not hold at zeta"),
            Self::CosetShifts => write!(f, "the coset shifts are not in distinct cosets"),
            Self::Shape => write!(f, "a circuit, witness or proof has the wrong shape"),
            Self::Rotation => write!(f, "a gadget reads a row outside the circuit"),
        }
    }
}

impl core::error::Error for PlonkError {}

/// A circuit that cannot be built is a [`PlonkError::Shape`] or a
/// [`PlonkError::Rotation`].
///
/// ```
/// use arrabbiata_plonk::{CircuitError, PlonkError};
/// assert_eq!(PlonkError::from(CircuitError::Rotation), PlonkError::Rotation);
/// ```
impl From<CircuitError> for PlonkError {
    fn from(e: CircuitError) -> Self {
        match e {
            CircuitError::Shape => Self::Shape,
            CircuitError::Rotation => Self::Rotation,
        }
    }
}

/// The sizes PlonK needs for a circuit: how much larger than the domain
/// the quotient is, and how many coefficients a commitment must accept.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{QuotientSizing, circuit::Builder, vanilla::Vanilla};
/// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
/// // q * q_M a b has degree 4, and so has z (a + ..) (b + ..) (c + ..).
/// assert_eq!(c.quotient_factor(), 4);
/// assert_eq!(c.max_coefficients(), 3 * c.num_rows());
/// ```
pub trait QuotientSizing {
    /// The largest total degree of a gated constraint (selector included),
    /// and of the copy and public-input identities: the quotient has fewer
    /// than `(quotient_factor() - 1) n` coefficients.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{QuotientSizing, circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// assert_eq!(c.quotient_factor(), 4);
    /// ```
    fn quotient_factor(&self) -> usize;

    /// The number of coefficients a polynomial commitment for this circuit
    /// must accept: the size of the quotient, or of a column if larger.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{QuotientSizing, circuit::Builder, vanilla::Vanilla};
    /// let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    /// assert_eq!(c.max_coefficients(), 3 * c.num_rows());
    /// ```
    fn max_coefficients(&self) -> usize;
}

impl<F, G, const W: usize, const X: usize, const Q: usize> QuotientSizing for Circuit<F, G, W, X, Q>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    fn quotient_factor(&self) -> usize {
        // The permutation term has degree W + 1, the boundary and
        // public-input terms 2, and a gated constraint its total degree plus
        // the selector.
        let mut factor = (W + 1).max(2);
        for gadget in 0..G::SELECTORS {
            for c in G::constraints(gadget) {
                factor = factor.max(gate::total_degree(&c) + 1);
            }
        }
        factor
    }

    fn max_coefficients(&self) -> usize {
        ((self.quotient_factor() - 1) * self.num_rows()).max(self.num_rows())
    }
}

fn interpolate<F>(domain: &Domain<F>, values: &[F]) -> Result<UniPoly<F>, PlonkError>
where
    F: TwoAdicField,
{
    domain.interpolate(values).ok_or(PlonkError::DomainTooLarge)
}

/// A vector of exactly `N` elements, as an array.
fn into_array<T, const N: usize>(v: Vec<T>) -> [T; N] {
    match v.try_into() {
        Ok(array) => array,
        Err(_) => unreachable!("built with exactly N elements"),
    }
}

/// Interpolate every column on `domain`.
fn interpolate_all<F, const N: usize>(
    domain: &Domain<F>,
    columns: &[Vec<F>; N],
) -> Result<[UniPoly<F>; N], PlonkError>
where
    F: TwoAdicField,
{
    let mut polys = Vec::with_capacity(N);
    for column in columns {
        polys.push(interpolate(domain, column)?);
    }
    Ok(into_array(polys))
}

/// The shifts `g^k`, `k < W`, of the cosets `g^k H` labelling the witness
/// columns in the permutation argument, for `g` the multiplicative
/// generator and `H` the subgroup of order `n`.
///
/// `g^i H` and `g^j H` are disjoint iff `g^(i - j)` is not in `H`, that is
/// `g^((i - j) n) != 1`; for `0 <= j < i < W` it suffices to check
/// `g^(m n) != 1` for `0 < m < W`.
fn coset_shifts<F, const W: usize>(n: usize) -> Result<[F; W], PlonkError>
where
    F: TwoAdicField,
{
    let g = F::MULTIPLICATIVE_GENERATOR;
    let gn = g.pow(n as u64);
    let mut power = F::ONE;
    for _ in 1..W {
        power = power * gn;
        if power == F::ONE {
            return Err(PlonkError::CosetShifts);
        }
    }
    Ok(array::from_fn(|k| g.pow(k as u64)))
}

/// Preprocess a circuit: interpolate and commit to the fixed, selector and
/// permutation polynomials.
///
/// # Errors
///
/// [`PlonkError::DomainTooLarge`] if the field has no domain of the circuit
/// size, and [`PlonkError::CosetShifts`] if the field has no `W` disjoint
/// cosets of it generated by powers of its multiplicative generator.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa, vanilla};
/// let (c, _, ()) = build::<F, _, _>(|b| {
///     b.constant(F::from_u64(2));
/// });
/// let circuit = vanilla::circuit(&c).unwrap();
/// assert!(arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).is_ok());
/// ```
pub fn setup<F, PC, G, const W: usize, const X: usize, const Q: usize>(
    circuit: &Circuit<F, G, W, X, Q>,
    pcs: &PC,
) -> Result<Keys<F, PC, G, W, X, Q>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    G: GateSet<F, W, X>,
{
    let n = circuit.num_rows();
    let domain = Domain::<F>::containing(n).ok_or(PlonkError::DomainTooLarge)?;
    let shifts = coset_shifts::<F, W>(n)?;

    let fixed = interpolate_all(&domain, circuit.fixed())?;
    let selectors = interpolate_all(&domain, &circuit.selector_columns())?;
    // sigma_k(i) is the identity label of the cell sigma maps (k, i) to:
    // shift_(column) * w^(row).
    let sigma = circuit.sigma();
    let sigma_values: [Vec<F>; W] = array::from_fn(|k| {
        (0..n)
            .map(|i| {
                let target = sigma[k * n + i];
                shifts[target / n] * domain.element(target % n)
            })
            .collect()
    });
    let sigmas = interpolate_all(&domain, &sigma_values)?;

    let commit = |p: &UniPoly<F>| pcs.commit(p.coefficients());
    let vk = VerifierKey::new(Preprocessed {
        log_size: domain.log_size(),
        shifts,
        fixed: array::from_fn(|i| commit(&fixed[i])),
        selectors: array::from_fn(|i| commit(&selectors[i])),
        sigmas: array::from_fn(|i| commit(&sigmas[i])),
        public_rows: circuit.public_rows().to_vec(),
        gates: PhantomData,
    });
    let pk = ProverKey {
        vk: vk.clone(),
        circuit: circuit.clone(),
        domain,
        fixed,
        selectors,
        sigmas,
        sigma_values,
    };
    Ok((pk, vk))
}

/// The evaluations of `p` on the coset `shift H_E` of the domain `ext`.
fn coset_evaluations<F>(p: &UniPoly<F>, ext: &Domain<F>, shift: F) -> Result<Vec<F>, PlonkError>
where
    F: TwoAdicField,
{
    let mut scaled = p.coefficients().to_vec();
    let mut power = F::ONE;
    for c in &mut scaled {
        *c = *c * power;
        power = power * shift;
    }
    ext.evaluate(&UniPoly::new(scaled))
        .ok_or(PlonkError::DomainTooLarge)
}

/// The values of the columns at one point of the extended coset, and at the
/// point one row later.
struct CosetView<'a, F, const W: usize, const X: usize> {
    witness: &'a [Vec<F>; W],
    fixed: &'a [Vec<F>; X],
    point: usize,
    step: usize,
}

impl<F, const W: usize, const X: usize> CosetView<'_, F, W, X>
where
    F: Copy,
{
    fn at(&self, values: &[F], rotation: Rotation) -> F {
        let size = values.len() as i64;
        let shift = (rotation.offset() as i64 * self.step as i64).rem_euclid(size) as usize;
        values[(self.point + shift) % values.len()]
    }
}

impl<F, const W: usize, const X: usize> Assignment<F, Scalars<F>, W, X> for CosetView<'_, F, W, X>
where
    F: Field,
{
    fn witness(&self, _: &Scalars<F>, column: WitnessColumn<W>, rotation: Rotation) -> F {
        self.at(&self.witness[column.index()], rotation)
    }

    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F {
        self.at(&self.fixed[column.index()], rotation)
    }
}

/// Prove that `witness` satisfies the circuit of `pk`.
///
/// # Errors
///
/// [`PlonkError::PublicInputs`] if the number of public inputs does not
/// match the circuit, [`PlonkError::Shape`] if a witness column does not
/// have one value per row, and [`PlonkError::Unsatisfied`] if the witness
/// does not satisfy the circuit.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{PlonkError, ipa, prove, setup, vanilla};
///
/// // A false statement: 2 * 2 = 5.
/// let (c, w, ()) = build::<F, _, _>(|b| {
///     let x = b.witness(F::from_u64(2));
///     let y = b.mul(&x, &x);
///     let five = b.public_input(F::from_u64(5));
///     b.assert_equal(&y, &five);
/// });
/// let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
/// let pcs = ipa::pallas(&circuit);
/// let (pk, _) = setup(&circuit, &pcs).unwrap();
/// assert_eq!(prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap_err(), PlonkError::Unsatisfied);
/// ```
pub fn prove<F, PC, S, G, const W: usize, const X: usize, const Q: usize>(
    pk: &ProverKey<F, PC, G, W, X, Q>,
    pcs: &PC,
    transcript: S,
    witness: &Witness<F, W>,
) -> Result<ProofOf<Plonk<F, PC, S, G, W, X, Q>>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    S: Transcript<F>,
    G: GateSet<F, W, X>,
{
    prove_with(pk, pcs, transcript, witness, Prover::Honest)
}

/// Whether the prover refuses an unsatisfied witness.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prover {
    /// Check the witness, and the degree of the quotient.
    Honest,
    /// Run the rounds on any witness, truncating the quotient to its
    /// degree bound: the strongest generic cheating prover, against which
    /// the tests show that the verifier's identity check is load-bearing.
    #[cfg(test)]
    Cheating,
}

fn prove_with<F, PC, S, G, const W: usize, const X: usize, const Q: usize>(
    pk: &ProverKey<F, PC, G, W, X, Q>,
    pcs: &PC,
    transcript: S,
    witness: &Witness<F, W>,
    prover: Prover,
) -> Result<ProofOf<Plonk<F, PC, S, G, W, X, Q>>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    S: Transcript<F>,
    G: GateSet<F, W, X>,
{
    let key = &pk.vk.value;
    if witness.public_inputs().len() != key.public_rows.len() {
        return Err(PlonkError::PublicInputs);
    }
    if witness
        .columns()
        .iter()
        .any(|c| c.len() != pk.domain.size())
    {
        return Err(PlonkError::Shape);
    }
    let honest = prover == Prover::Honest;
    if honest && !pk.circuit.is_satisfied(witness) {
        return Err(PlonkError::Unsatisfied);
    }
    let commit = |p: &UniPoly<F>| pcs.commit(p.coefficients());

    // Round 1: the witness columns.
    let wires = interpolate_all(&pk.domain, witness.columns())?;
    let tape = prover_tape::<F, S, Plonk<F, PC, S, G, W, X, Q>>(transcript)
        .public(&pk.vk)
        .public(&PublicInputs(witness.public_inputs().to_vec()))
        .send(Scheme::new(Columns(array::from_fn(|k| commit(&wires[k])))));
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();

    // Round 2: the grand product of the copy constraints.
    let z = grand_product(pk, witness.columns(), beta, gamma)?;
    let tape = tape.send(Scheme::new(commit(&z)));
    let (alpha, tape) = tape.challenge();

    // Round 3: the quotient. The gate term is
    // sum_j alpha^j q_g(j) C_j, over the J constraints of all gadgets.
    let constraints: Vec<Vec<Expr<F, W, X>>> = (0..Q).map(|g| G::constraints(g)).collect();
    let ring = Scalars::new();
    let gates = |view: &CosetView<'_, F, W, X>, selectors: &[F], _: &[F]| -> (F, F) {
        let mut acc = F::ZERO;
        let mut power = F::ONE;
        for (selector, gadget) in selectors.iter().zip(&constraints) {
            for c in gadget {
                acc = acc + power * *selector * evaluate(&ring, c, view);
                power = power * alpha;
            }
        }
        (acc, power)
    };
    let t = quotient(
        pk,
        &wires,
        &z,
        witness.public_inputs(),
        &[],
        (beta, gamma, alpha),
        honest,
        gates,
    )?;
    let tape = tape.send(Scheme::new(commit(&t)));
    let (zeta, tape) = tape.challenge();

    // Round 4: evaluations and openings.
    let opened = open_all(pk, pcs, &wires, &z, &t, zeta);
    let evaluations = opened.clone().map(|(v, _)| v);
    let proofs = opened.map(|(_, p)| p);
    let (_, proof) = tape.send(evaluations).send(Scheme::new(proofs)).finish();
    Ok(proof)
}

/// The grand product of the copy-constraint argument:
/// `z(w^0) = 1`, `z(w^(i+1)) = z(w^i) * num_i / den_i`.
fn grand_product<F, PC, G, const W: usize, const X: usize, const Q: usize>(
    pk: &ProverKey<F, PC, G, W, X, Q>,
    columns: &[Vec<F>; W],
    beta: F,
    gamma: F,
) -> Result<UniPoly<F>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    G: GateSet<F, W, X>,
{
    let key = &pk.vk.value;
    let domain = &pk.domain;
    let n = domain.size();
    let mut z_values = vec![F::ONE; n];
    let mut i = 0;
    while i + 1 < n {
        let x = domain.element(i);
        let mut num = F::ONE;
        let mut den = F::ONE;
        for ((column, shift), sigma) in columns.iter().zip(&key.shifts).zip(&pk.sigma_values) {
            num = num * (column[i] + beta * *shift * x + gamma);
            den = den * (column[i] + beta * sigma[i] + gamma);
        }
        // den is zero only with probability about W n / |F| over gamma.
        z_values[i + 1] = z_values[i] * num * den.inverse().ok_or(PlonkError::Unsatisfied)?;
        i += 1;
    }
    interpolate(domain, &z_values)
}

/// The quotient `t`, pointwise on the coset `g H_E`: the gate term given by
/// `gates`, then `alpha`-weighted the public-input, permutation and
/// boundary terms, divided by `Z_H`.
///
/// `gates` receives the view of the columns at a point, the selectors at
/// that point, and the values at that point of the polynomials `extra`;
/// it returns the gate term and the power of `alpha` the next term starts
/// at. `honest` refuses a numerator that `Z_H` does not divide.
#[allow(clippy::too_many_arguments)]
fn quotient<F, PC, G, Gt, const W: usize, const X: usize, const Q: usize>(
    pk: &ProverKey<F, PC, G, W, X, Q>,
    wires: &[UniPoly<F>; W],
    z: &UniPoly<F>,
    public_inputs: &[F],
    extra: &[UniPoly<F>],
    (beta, gamma, alpha): (F, F, F),
    honest: bool,
    mut gates: Gt,
) -> Result<UniPoly<F>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    G: GateSet<F, W, X>,
    Gt: FnMut(&CosetView<'_, F, W, X>, &[F], &[F]) -> (F, F),
{
    let key = &pk.vk.value;
    let domain = &pk.domain;
    let n = domain.size();
    let factor = pk.circuit.quotient_factor();
    let ext = Domain::<F>::containing(factor * n).ok_or(PlonkError::DomainTooLarge)?;
    let size = ext.size();
    let step = size / n;
    let shift = F::MULTIPLICATIVE_GENERATOR;
    let on_coset = |p: &UniPoly<F>| coset_evaluations(p, &ext, shift);
    let on_coset_all = |polys: &[UniPoly<F>]| -> Result<Vec<Vec<F>>, PlonkError> {
        let mut values = Vec::with_capacity(polys.len());
        for p in polys {
            values.push(on_coset(p)?);
        }
        Ok(values)
    };
    let wire_values: [Vec<F>; W] = into_array(on_coset_all(wires)?);
    let fixed_values: [Vec<F>; X] = into_array(on_coset_all(&pk.fixed)?);
    let selector_values = on_coset_all(&pk.selectors)?;
    let sigma_values = on_coset_all(&pk.sigmas)?;
    let extra_values = on_coset_all(extra)?;
    let z_coset = on_coset(z)?;
    let mut first = vec![F::ZERO; n];
    first[0] = F::ONE;
    let l1 = on_coset(&interpolate(domain, &first)?)?;
    let mut public_indicator = vec![F::ZERO; n];
    let mut public_values = vec![F::ZERO; n];
    for (row, value) in key.public_rows.iter().zip(public_inputs) {
        public_indicator[*row] = F::ONE;
        public_values[*row] = *value;
    }
    let lp = on_coset(&interpolate(domain, &public_indicator)?)?;
    let pi = on_coset(&interpolate(domain, &public_values)?)?;

    // x_i = shift w_E^i, and Z_H(x_i) = x_i^n - 1 has period E / n in i.
    let shift_n = shift.pow(n as u64);
    let root_n = ext.generator().pow(n as u64);
    let mut vanishing_inverse = Vec::with_capacity(step);
    let mut power = F::ONE;
    for _ in 0..step {
        let zh = shift_n * power - F::ONE;
        vanishing_inverse.push(zh.inverse().ok_or(PlonkError::CosetShifts)?);
        power = power * root_n;
    }
    let mut t_values = vec![F::ZERO; size];
    let mut selectors_here = vec![F::ZERO; Q];
    let mut extra_here = vec![F::ZERO; extra.len()];
    let mut x = shift;
    let mut point = 0;
    while point < size {
        let view = CosetView {
            witness: &wire_values,
            fixed: &fixed_values,
            point,
            step,
        };
        for (s, values) in selectors_here.iter_mut().zip(&selector_values) {
            *s = values[point];
        }
        for (e, values) in extra_here.iter_mut().zip(&extra_values) {
            *e = values[point];
        }
        let (mut acc, mut power) = gates(&view, &selectors_here, &extra_here);
        if W > 0 {
            acc = acc + power * (lp[point] * wire_values[0][point] - pi[point]);
        }
        power = power * alpha;
        let next = (point + step) % size;
        let mut left = z_coset[point];
        let mut right = z_coset[next];
        for k in 0..W {
            let w = wire_values[k][point];
            left = left * (w + beta * key.shifts[k] * x + gamma);
            right = right * (w + beta * sigma_values[k][point] + gamma);
        }
        acc = acc + power * (left - right);
        power = power * alpha;
        acc = acc + power * l1[point] * (z_coset[point] - F::ONE);
        t_values[point] = acc * vanishing_inverse[point % step];
        x = x * ext.generator();
        point += 1;
    }
    let t_shifted = ext
        .interpolate(&t_values)
        .ok_or(PlonkError::DomainTooLarge)?;
    let shift_inverse = shift.inverse().ok_or(PlonkError::CosetShifts)?;
    let degree_bound = (factor - 1) * n;
    let mut t_coefficients = Vec::with_capacity(degree_bound);
    let mut power = F::ONE;
    for (k, c) in t_shifted.coefficients().iter().enumerate() {
        let value = *c * power;
        if k < degree_bound {
            t_coefficients.push(value);
        } else if honest && !value.is_zero() {
            // Not reached for a satisfied circuit: the numerator vanishes
            // on H and has degree below factor * n.
            return Err(PlonkError::Unsatisfied);
        }
        power = power * shift_inverse;
    }
    Ok(UniPoly::new(t_coefficients))
}

/// Every opening of the protocol at `zeta` (and `zeta w^k`), with its
/// value.
fn open_all<F, PC, G, const W: usize, const X: usize, const Q: usize>(
    pk: &ProverKey<F, PC, G, W, X, Q>,
    pcs: &PC,
    wires: &[UniPoly<F>; W],
    z: &UniPoly<F>,
    t: &UniPoly<F>,
    zeta: F,
) -> Opened<(F, Prf<F, PC>), W, X, Q>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    G: GateSet<F, W, X>,
{
    let domain = &pk.domain;
    let zeta_w = zeta * domain.generator();
    let (rotated_witness, rotated_fixed) = rotations::<F, G, W, X>();
    let at = |p: &UniPoly<F>, point: &F| pcs.open(p.coefficients(), point);
    Opened::<(F, Prf<F, PC>), W, X, Q> {
        witness: array::from_fn(|k| at(&wires[k], &zeta)),
        fixed: array::from_fn(|k| at(&pk.fixed[k], &zeta)),
        selectors: array::from_fn(|k| at(&pk.selectors[k], &zeta)),
        sigmas: array::from_fn(|k| at(&pk.sigmas[k], &zeta)),
        z: at(z, &zeta),
        t: at(t, &zeta),
        z_next: at(z, &zeta_w),
        witness_rotated: rotated_witness
            .iter()
            .map(|(k, r)| at(&wires[*k], &(zeta * root_power(domain, *r))))
            .collect(),
        fixed_rotated: rotated_fixed
            .iter()
            .map(|(k, r)| at(&pk.fixed[*k], &(zeta * root_power(domain, *r))))
            .collect(),
    }
}

/// The claimed evaluations, as an assignment of the variables of a
/// constraint at `zeta`.
struct OpenedView<'a, F, const W: usize, const X: usize, const Q: usize> {
    opened: &'a Opened<F, W, X, Q>,
    rotated_witness: &'a [(usize, Rotation)],
    rotated_fixed: &'a [(usize, Rotation)],
}

/// The value of `(column, rotation)` among the `opened` pairs.
fn lookup<F>(opened: &[(usize, Rotation)], values: &[F], column: usize, rotation: Rotation) -> F
where
    F: Field,
{
    match opened.binary_search(&(column, rotation)) {
        Ok(i) => values[i],
        // Not reached: the rotated columns are computed from the gate set
        // that produced the constraint.
        Err(_) => F::ZERO,
    }
}

impl<F, const W: usize, const X: usize, const Q: usize> Assignment<F, Scalars<F>, W, X>
    for OpenedView<'_, F, W, X, Q>
where
    F: Field,
{
    fn witness(&self, _: &Scalars<F>, column: WitnessColumn<W>, rotation: Rotation) -> F {
        if rotation == Rotation::CURR {
            self.opened.witness[column.index()]
        } else {
            lookup(
                self.rotated_witness,
                &self.opened.witness_rotated,
                column.index(),
                rotation,
            )
        }
    }

    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F {
        if rotation == Rotation::CURR {
            self.opened.fixed[column.index()]
        } else {
            lookup(
                self.rotated_fixed,
                &self.opened.fixed_rotated,
                column.index(),
                rotation,
            )
        }
    }
}

/// Verify `proof` for the statement `public_inputs` against `vk`.
///
/// # Errors
///
/// [`PlonkError::PublicInputs`] if the number of public inputs is wrong,
/// [`PlonkError::Shape`] if the proof opens the wrong number of rotated
/// columns, [`PlonkError::Opening`] if an opening proof fails, and
/// [`PlonkError::IdentityFailed`] if the PlonK identity does not hold at
/// `zeta`.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa, prove, setup, vanilla, verify};
///
/// let (c, w, ()) = build::<F, _, _>(|b| {
///     let x = b.witness(F::from_u64(4));
///     let y = b.mul(&x, &x);
///     let out = b.public_input(F::from_u64(16));
///     b.assert_equal(&y, &out);
/// });
/// let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
/// let pcs = ipa::pallas(&circuit);
/// let (pk, vk) = setup(&circuit, &pcs).unwrap();
/// let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
/// // The same proof does not verify for another statement.
/// assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &[F::from_u64(17)], proof.clone()).is_err());
/// assert!(verify(&vk, &pcs, ipa::pallas_transcript(), &[F::from_u64(16)], proof).is_ok());
/// ```
pub fn verify<F, PC, S, G, const W: usize, const X: usize, const Q: usize>(
    vk: &VerifierKey<F, PC, G, W, X, Q>,
    pcs: &PC,
    transcript: S,
    public_inputs: &[F],
    proof: ProofOf<Plonk<F, PC, S, G, W, X, Q>>,
) -> Result<(), PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    S: Transcript<F>,
    G: GateSet<F, W, X>,
{
    let key = &vk.value;
    if public_inputs.len() != key.public_rows.len() {
        return Err(PlonkError::PublicInputs);
    }
    let domain = Domain::<F>::new(key.log_size).ok_or(PlonkError::DomainTooLarge)?;
    let (wires, tape) = verifier_tape::<F, S, Plonk<F, PC, S, G, W, X, Q>>(transcript, proof)
        .public(vk)
        .public(&PublicInputs(public_inputs.to_vec()))
        .read();
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();
    let (z_commitment, tape) = tape.read();
    let (alpha, tape) = tape.challenge();
    let (t_commitment, tape) = tape.read();
    let (zeta, tape) = tape.challenge();
    let (evaluations, tape) = tape.read();
    let (proofs, tape) = tape.read();
    let _ = tape.finish();
    let (wires, z_commitment, t_commitment, proofs) = (
        wires.value.0,
        z_commitment.value,
        t_commitment.value,
        proofs.value,
    );

    check_openings::<F, PC, G, W, X, Q>(
        key,
        pcs,
        &domain,
        (&wires, &z_commitment, &t_commitment),
        zeta,
        &evaluations,
        &proofs,
    )?;

    // The gate term at zeta: sum_j alpha^j q_g(j) C_j.
    let (rotated_witness, rotated_fixed) = rotations::<F, G, W, X>();
    let view = OpenedView {
        opened: &evaluations,
        rotated_witness: &rotated_witness,
        rotated_fixed: &rotated_fixed,
    };
    let ring = Scalars::new();
    let mut lhs = F::ZERO;
    let mut power = F::ONE;
    for (gadget, selector) in evaluations.selectors.iter().enumerate() {
        for c in G::constraints(gadget) {
            lhs = lhs + power * *selector * evaluate(&ring, &c, &view);
            power = power * alpha;
        }
    }
    lhs = lhs
        + identity_rest(
            key,
            &domain,
            &evaluations,
            public_inputs,
            (beta, gamma, alpha),
            power,
            zeta,
        );
    if lhs == evaluations.t * domain.vanishing_at(zeta) {
        Ok(())
    } else {
        Err(PlonkError::IdentityFailed)
    }
}

/// Check that the proof opens every rotated column the gate set reads, and
/// every opening against its commitment and point.
#[allow(clippy::type_complexity)]
fn check_openings<F, PC, G, const W: usize, const X: usize, const Q: usize>(
    key: &Preprocessed<F, Com<F, PC>, G, W, X, Q>,
    pcs: &PC,
    domain: &Domain<F>,
    (wires, z_commitment, t_commitment): (&[Com<F, PC>; W], &Com<F, PC>, &Com<F, PC>),
    zeta: F,
    evaluations: &Opened<F, W, X, Q>,
    proofs: &Opened<Prf<F, PC>, W, X, Q>,
) -> Result<(), PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    G: GateSet<F, W, X>,
{
    let (rotated_witness, rotated_fixed) = rotations::<F, G, W, X>();
    if evaluations.witness_rotated.len() != rotated_witness.len()
        || evaluations.fixed_rotated.len() != rotated_fixed.len()
        || proofs.witness_rotated.len() != rotated_witness.len()
        || proofs.fixed_rotated.len() != rotated_fixed.len()
    {
        return Err(PlonkError::Shape);
    }

    // The commitment and the point of every opening, in the same shape as
    // the evaluations and the proofs, so the three are zipped by one
    // traversal.
    let at = |r: Rotation| zeta * root_power(domain, r);
    let claims = Opened::<(&Com<F, PC>, F), W, X, Q> {
        witness: array::from_fn(|k| (&wires[k], zeta)),
        fixed: array::from_fn(|k| (&key.fixed[k], zeta)),
        selectors: array::from_fn(|k| (&key.selectors[k], zeta)),
        sigmas: array::from_fn(|k| (&key.sigmas[k], zeta)),
        z: (z_commitment, zeta),
        t: (t_commitment, zeta),
        z_next: (z_commitment, at(Rotation::NEXT)),
        witness_rotated: rotated_witness
            .iter()
            .map(|(k, r)| (&wires[*k], at(*r)))
            .collect(),
        fixed_rotated: rotated_fixed
            .iter()
            .map(|(k, r)| (&key.fixed[*k], at(*r)))
            .collect(),
    };
    for (i, ((commitment, point), (value, proof))) in claims
        .iter()
        .zip(evaluations.iter().zip(proofs.iter()))
        .enumerate()
    {
        if !pcs.verify(commitment, point, *value, proof) {
            return Err(PlonkError::Opening(i));
        }
    }
    Ok(())
}

/// The public-input, permutation and boundary terms of the identity at
/// `zeta`, weighted by `power`, `power alpha` and `power alpha^2`.
fn identity_rest<F, C, G, const W: usize, const X: usize, const Q: usize>(
    key: &Preprocessed<F, C, G, W, X, Q>,
    domain: &Domain<F>,
    evaluations: &Opened<F, W, X, Q>,
    public_inputs: &[F],
    (beta, gamma, alpha): (F, F, F),
    mut power: F,
    zeta: F,
) -> F
where
    F: TwoAdicField,
{
    let mut lhs = F::ZERO;
    if W > 0 {
        let mut lp = F::ZERO;
        let mut pi = F::ZERO;
        for (row, value) in key.public_rows.iter().zip(public_inputs) {
            let l = domain.lagrange_at(*row, zeta);
            lp = lp + l;
            pi = pi + *value * l;
        }
        lhs = lhs + power * (lp * evaluations.witness[0] - pi);
    }
    power = power * alpha;
    let mut left = evaluations.z;
    let mut right = evaluations.z_next;
    for k in 0..W {
        let w = evaluations.witness[k];
        left = left * (w + beta * key.shifts[k] * zeta + gamma);
        right = right * (w + beta * evaluations.sigmas[k] + gamma);
    }
    lhs = lhs + power * (left - right);
    power = power * alpha;
    lhs + power * domain.first_lagrange_at(zeta) * (evaluations.z - F::ONE)
}

#[cfg(test)]
mod tests {
    //! Soundness of the verifier against a cheating prover: one that runs
    //! the rounds on an unsatisfied witness. Every opening then verifies (the
    //! prover commits to and opens real polynomials), so only the identity
    //! at `zeta` can reject; these tests show that it does, for each part
    //! of the relation.

    use arrabbiata_field::Field;
    use arrabbiata_frontend::vanilla::build;
    use arrabbiata_pasta::Fq as F;

    use super::*;

    /// A satisfied vanilla circuit: 3 * 4 + 5 = 17, with 17 public, and the
    /// product copied to the addition.
    fn statement() -> (vanilla::VanillaCircuit<F>, Witness<F, 3>) {
        let (c, w, ()) = build::<F, _, _>(|b| {
            let x = b.witness(F::from_u64(3));
            let y = b.witness(F::from_u64(4));
            let p = b.mul(&x, &y);
            let s = b.add_constant(&p, F::from_u64(5));
            let out = b.public_input(F::from_u64(17));
            b.assert_equal(&s, &out);
        });
        (vanilla::circuit(&c).unwrap(), vanilla::witness(&w))
    }

    fn cheat(
        circuit: &vanilla::VanillaCircuit<F>,
        witness: &Witness<F, 3>,
        public: &[F],
    ) -> Result<(), PlonkError> {
        let pcs = ipa::pallas(circuit);
        let (pk, vk) = setup(circuit, &pcs).unwrap();
        let proof = prove_with(
            &pk,
            &pcs,
            ipa::pallas_transcript(),
            witness,
            Prover::Cheating,
        )
        .unwrap();
        verify(&vk, &pcs, ipa::pallas_transcript(), public, proof)
    }

    #[test]
    fn honest_run_of_the_cheating_prover_verifies() {
        let (circuit, witness) = statement();
        assert_eq!(cheat(&circuit, &witness, &[F::from_u64(17)]), Ok(()));
    }

    /// Every cell changed alone breaks a gate, a copy or the public input:
    /// the identity rejects each one.
    #[test]
    fn every_broken_cell_fails_the_identity() {
        let (circuit, witness) = statement();
        let n = circuit.num_rows();
        let mut checked = 0;
        for k in 0..3 {
            for row in 0..n {
                let mut columns = witness.columns().clone();
                columns[k][row] = columns[k][row] + F::ONE;
                let broken = Witness::new(columns, witness.public_inputs().to_vec());
                if circuit.is_satisfied(&broken) {
                    continue;
                }
                checked += 1;
                assert_eq!(
                    cheat(&circuit, &broken, &[F::from_u64(17)]),
                    Err(PlonkError::IdentityFailed),
                    "cell ({k}, {row})"
                );
            }
        }
        assert!(checked > 0);
    }

    /// `b = a(-1) + a(+1)`.
    struct Neighbours;
    impl Gadget<F, 2, 1> for Neighbours {
        const NAME: &'static str = "neighbours";
        fn constraints() -> Vec<Expr<F, 2, 1>> {
            let a = |r| Expr::witness(WitnessColumn::new::<0>(), r);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            vec![b - a(Rotation::PREV) - a(Rotation::NEXT)]
        }
    }

    /// `b = f(+2) a(+2)`.
    struct Ahead;
    impl Gadget<F, 2, 1> for Ahead {
        const NAME: &'static str = "ahead";
        fn constraints() -> Vec<Expr<F, 2, 1>> {
            let two = Rotation::new(2);
            let a = Expr::witness(WitnessColumn::new::<0>(), two);
            let f = Expr::fixed(FixedColumn::new::<0>(), two);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            vec![b - f * a]
        }
    }

    /// Every cell changed alone in a circuit whose gadgets read other rows
    /// (witness at -1, +1, +2, fixed at +2) is rejected by the identity: the
    /// rotated openings carry the constraint, not only the current row.
    #[test]
    fn every_broken_cell_fails_the_identity_with_rotations() {
        type G = (Neighbours, (Ahead, ()));
        let a: Vec<F> = (0..8).map(|i| F::from_u64(3 * i + 1)).collect();
        let f: Vec<F> = (0..8).map(|i| F::from_u64(i + 2)).collect();
        let mut b = Builder::<F, G, 2, 1, 2>::new();
        for i in 0..8 {
            if (1..7).contains(&i) && i % 2 == 1 {
                b.row::<Neighbours, _>([a[i], a[i - 1] + a[i + 1]], [f[i]]);
            } else if i < 6 {
                b.row::<Ahead, _>([a[i], f[i + 2] * a[i + 2]], [f[i]]);
            } else {
                b.plain_row([a[i], F::ZERO], [f[i]]);
            }
        }
        b.public(7);
        let (circuit, witness) = b.build().unwrap();
        assert!(circuit.is_satisfied(&witness));
        let pcs = ipa::pallas(&circuit);
        let (pk, vk) = setup(&circuit, &pcs).unwrap();
        let mut checked = 0;
        for k in 0..2 {
            for row in 0..circuit.num_rows() {
                let mut columns = witness.columns().clone();
                columns[k][row] = columns[k][row] + F::ONE;
                let broken = Witness::new(columns, witness.public_inputs().to_vec());
                if circuit.is_satisfied(&broken) {
                    continue;
                }
                checked += 1;
                let proof = prove_with(
                    &pk,
                    &pcs,
                    ipa::pallas_transcript(),
                    &broken,
                    Prover::Cheating,
                )
                .unwrap();
                assert_eq!(
                    verify(
                        &vk,
                        &pcs,
                        ipa::pallas_transcript(),
                        witness.public_inputs(),
                        proof
                    ),
                    Err(PlonkError::IdentityFailed),
                    "cell ({k}, {row})"
                );
            }
        }
        // The cells some gadget reads; changing any other one changes
        // nothing the circuit constrains.
        assert_eq!(checked, 11);
    }

    /// The field of order 17: `p - 1 = 16 = 2^4`, generator 3.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct F17;
    impl arrabbiata_field::MontgomeryConfig<1> for F17 {
        const MODULUS: [u64; 1] = [17];
        const GENERATOR: u64 = 3;
    }
    type Small = arrabbiata_field::MontField<F17, 1>;

    /// The cosets `g^k H` of a subgroup `H` of order `n`, as sets of
    /// canonical integers.
    fn cosets<const W: usize>(shifts: &[Small; W], n: usize) -> Vec<Vec<u64>> {
        use arrabbiata_field::PrimeField;
        let h = Small::two_adic_generator(n.trailing_zeros()).unwrap();
        shifts
            .iter()
            .map(|s| {
                let mut c: Vec<u64> = (0..n).map(|i| (*s * h.pow(i as u64)).limb(0)).collect();
                c.sort_unstable();
                c
            })
            .collect()
    }

    /// `coset_shifts` succeeds exactly when the `W` cosets are pairwise
    /// disjoint, checked by enumerating them, for every subgroup of the
    /// field of order 17 and `W` up to 5.
    #[test]
    fn coset_shifts_are_disjoint_exactly_when_accepted() {
        fn check<const W: usize>() {
            for log in 0..=4 {
                let n = 1usize << log;
                let disjoint = |c: &Vec<Vec<u64>>| {
                    (0..W).all(|i| (0..i).all(|j| c[i].iter().all(|x| !c[j].contains(x))))
                };
                match coset_shifts::<Small, W>(n) {
                    Ok(shifts) => {
                        assert!(disjoint(&cosets(&shifts, n)), "W = {W}, n = {n}");
                        assert_eq!(shifts[0], Small::ONE);
                    }
                    Err(e) => {
                        assert_eq!(e, PlonkError::CosetShifts);
                        let g = Small::MULTIPLICATIVE_GENERATOR;
                        let all: [Small; W] = core::array::from_fn(|k| g.pow(k as u64));
                        assert!(!disjoint(&cosets(&all, n)), "W = {W}, n = {n}");
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        // 16 / n cosets of H fit in the 16 nonzero elements.
        assert!(coset_shifts::<Small, 2>(8).is_ok());
        assert_eq!(
            coset_shifts::<Small, 3>(8).unwrap_err(),
            PlonkError::CosetShifts
        );
        assert_eq!(
            coset_shifts::<Small, 2>(16).unwrap_err(),
            PlonkError::CosetShifts
        );
    }

    /// A true witness for one statement does not prove another.
    #[test]
    fn wrong_public_input_fails_the_identity() {
        let (circuit, witness) = statement();
        let lie = Witness::new(witness.columns().clone(), vec![F::from_u64(18)]);
        assert_eq!(
            cheat(&circuit, &lie, &[F::from_u64(18)]),
            Err(PlonkError::IdentityFailed)
        );
    }
}
