//! The protocol: PlonK's polynomial IOP with the gate identity of a
//! generalized circuit, compiled with a univariate polynomial commitment
//! on the typed transcript.
//!
//! With `n` rows, `H` the subgroup of order `n`, the rounds follow
//! [`GeneralizedTranscript`]:
//!
//! 1. the prover commits to the witness columns `w_0, ..., w_(W-1)`;
//! 2. the verifier sends `beta, gamma`; the prover commits to the grand
//!    product `z` of the copy permutation, the cosets `g^k H` labelling
//!    column `k`;
//! 3. the verifier sends `alpha`; the prover commits to the quotient `t` of
//!    `sum_j alpha^j q_j P_j + alpha^J (L_B w_0 - PI) + alpha^(J+1) perm
//!     + alpha^(J+2) L_1 (z - 1)` by `Z_H`, where `P_j` is gate `j` with
//!    its variables replaced by the column polynomials (rotated) and the
//!    public inputs, `q_j` its selector (or 1), `L_B` is 1 on the bound
//!    rows and `PI` interpolates their public inputs;
//! 4. the verifier sends `zeta`; the prover opens every column at `zeta`,
//!    each column a gate reads at a rotation `k` at `zeta w^k`, and `z` at
//!    `zeta w`;
//! 5. the verifier checks the openings and the identity at `zeta`.
//!
//! The quotient is computed pointwise on a coset of a domain large enough
//! for the largest [column degree](crate::Circuit::column_degree) of a
//! gate: public inputs are scalars, so a gate's degree in them costs
//! nothing.
//!
//! The messages carry the commitment scheme in their type and label, not
//! the sponge, so one protocol type runs on any transcript.

use alloc::{collections::BTreeSet, vec, vec::Vec};
use core::{fmt, iter, marker::PhantomData};

use arrabbiata_commit::{PolynomialCommitment, Univariate};
use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_mvpoly::MVPoly;
use arrabbiata_poly::{Domain, Rotation, UniPoly};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, absorb_slice,
    typed::{
        Challenge, End, Label, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape,
    },
};

use crate::{Circuit, Error, Gate, Variable, circuit::absorb_parts};

/// The commitment type of a scheme.
type Com<F, PC> = <PC as PolynomialCommitment<F>>::Commitment;

/// The opening proof type of a scheme.
type Prf<F, PC> = <PC as PolynomialCommitment<F>>::Proof;

/// A value of type `T` produced with the commitment scheme `PC`: its
/// label combines the scheme's with `TAG`, so a proof made with one
/// scheme is not a proof for another.
///
/// ```
/// use arrabbiata_generalized_plonk::Tagged;
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq, Pallas};
/// use arrabbiata_poseidon::PoseidonSponge;
/// let t = Tagged::<Ipa<Pallas, PoseidonSponge<Fq>>, _, 1>::new(5u8);
/// assert_eq!(t.value, 5);
/// ```
pub struct Tagged<PC, T, const TAG: u64> {
    /// The value.
    pub value: T,
    scheme: PhantomData<fn() -> PC>,
}

impl<PC, T, const TAG: u64> Tagged<PC, T, TAG> {
    /// Tag `value` with the scheme `PC`.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Tagged;
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq, Pallas};
    /// use arrabbiata_poseidon::PoseidonSponge;
    /// assert_eq!(Tagged::<Ipa<Pallas, PoseidonSponge<Fq>>, _, 2>::new("x").value, "x");
    /// ```
    pub const fn new(value: T) -> Self {
        Self {
            value,
            scheme: PhantomData,
        }
    }
}

impl<PC, T, const TAG: u64> Clone for Tagged<PC, T, TAG>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl<PC, T, const TAG: u64> fmt::Debug for Tagged<PC, T, TAG>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Tagged").field(&self.value).finish()
    }
}

impl<PC, T, const TAG: u64> PartialEq for Tagged<PC, T, TAG>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<PC, T, const TAG: u64> Eq for Tagged<PC, T, TAG> where T: Eq {}

impl<PC, T, const TAG: u64> Label for Tagged<PC, T, TAG>
where
    PC: Label,
{
    const LABEL: u64 = PC::LABEL.rotate_left(16) ^ TAG;
}

impl<F, PC, T, const TAG: u64> Absorb<F> for Tagged<PC, T, TAG>
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

/// Commitments, absorbed with their number.
///
/// ```
/// use arrabbiata_generalized_plonk::Commitments;
/// assert_eq!(Commitments(vec![1u8, 2]).0.len(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitments<C>(pub Vec<C>);

impl<F, C> Absorb<F> for Commitments<C>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_slice(transcript, &self.0);
    }
}

/// The public inputs of a statement.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_generalized_plonk::PublicInputs;
/// use arrabbiata_pasta::Fq as F;
/// assert_eq!(PublicInputs(vec![F::ONE]).0, vec![F::ONE]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInputs<F>(pub Vec<F>);

impl<F> Label for PublicInputs<F> {
    const LABEL: u64 = 0x6770_6969;
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
/// The order, used by [`Opened::iter`] and [`Error::Opening`], is: the
/// witness, fixed, selector and permutation polynomials, `z` and `t` at
/// `zeta`; `z` at `zeta w`; then each column some gate reads at a rotation
/// other than the current row, in the order of [`rotated`], at
/// `zeta w^k`.
///
/// ```
/// use arrabbiata_generalized_plonk::Opened;
/// let o = Opened { witness: vec![0u8], fixed: vec![], selectors: vec![1], sigmas: vec![2], z: 3, t: 4, z_next: 5, rotated: vec![6] };
/// assert_eq!(o.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5, 6]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opened<T> {
    /// The witness polynomials at `zeta`.
    pub witness: Vec<T>,
    /// The fixed polynomials at `zeta`.
    pub fixed: Vec<T>,
    /// The selector polynomials at `zeta`.
    pub selectors: Vec<T>,
    /// The permutation polynomials at `zeta`.
    pub sigmas: Vec<T>,
    /// The grand product at `zeta`.
    pub z: T,
    /// The quotient at `zeta`.
    pub t: T,
    /// The grand product at `zeta w`.
    pub z_next: T,
    /// The rotated columns, in the order of [`rotated`].
    pub rotated: Vec<T>,
}

impl<T> Opened<T> {
    /// Every value, in the order of the openings.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Opened;
    /// let o = Opened { witness: vec![], fixed: vec![], selectors: vec![], sigmas: vec![], z: 1u8, t: 2, z_next: 3, rotated: vec![] };
    /// assert_eq!(o.iter().count(), 3);
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
            .chain(&self.rotated)
    }

    /// Every value, mutably, in the order of the openings.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::Opened;
    /// let mut o = Opened { witness: vec![1u8], fixed: vec![], selectors: vec![], sigmas: vec![], z: 0, t: 0, z_next: 0, rotated: vec![] };
    /// *o.iter_mut().next().unwrap() = 9;
    /// assert_eq!(o.witness, vec![9]);
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
            .chain(&mut self.rotated)
    }

    /// Whether there is one value per witness, fixed, selector,
    /// permutation and rotated polynomial, for the counts
    /// `(witness, fixed, selectors, rotated)`.
    fn has_shape(&self, (w, x, q, r): (usize, usize, usize, usize)) -> bool {
        self.witness.len() == w
            && self.fixed.len() == x
            && self.selectors.len() == q
            && self.sigmas.len() == w
            && self.rotated.len() == r
    }

    fn map<U>(self, mut f: impl FnMut(T) -> U) -> Opened<U> {
        Opened {
            witness: self.witness.into_iter().map(&mut f).collect(),
            fixed: self.fixed.into_iter().map(&mut f).collect(),
            selectors: self.selectors.into_iter().map(&mut f).collect(),
            sigmas: self.sigmas.into_iter().map(&mut f).collect(),
            z: f(self.z),
            t: f(self.t),
            z_next: f(self.z_next),
            rotated: self.rotated.into_iter().map(&mut f).collect(),
        }
    }
}

impl<T> Label for Opened<T> {
    const LABEL: u64 = 0x6770_6576;
}

impl<F, T> Absorb<F> for Opened<T>
where
    F: Field,
    T: Absorb<F>,
{
    fn absorb_into<Tr>(&self, transcript: &mut Tr)
    where
        Tr: Transcript<F>,
    {
        absorb_length(transcript, self.witness.len());
        absorb_length(transcript, self.fixed.len());
        absorb_length(transcript, self.selectors.len());
        absorb_length(transcript, self.rotated.len());
        for value in self.iter() {
            value.absorb_into(transcript);
        }
    }
}

/// The column variables read at a rotation other than the current row,
/// with the rotation set to the one read, sorted and without repetition:
/// the openings at `zeta w^k` a proof makes.
///
/// ```
/// use arrabbiata_generalized_plonk::{Variable, rotated};
/// use arrabbiata_poly::Rotation;
/// let vars = [
///     Variable::Witness { column: 0, rotation: Rotation::NEXT },
///     Variable::Witness { column: 0, rotation: Rotation::CURR },
///     Variable::Public(0),
/// ];
/// assert_eq!(rotated(&vars), vec![Variable::Witness { column: 0, rotation: Rotation::NEXT }]);
/// ```
#[must_use]
pub fn rotated(variables: &[Variable]) -> Vec<Variable> {
    let mut set = BTreeSet::new();
    for v in variables {
        match v {
            Variable::Witness { rotation, .. } | Variable::Fixed { rotation, .. }
                if *rotation != Rotation::CURR =>
            {
                set.insert(*v);
            }
            _ => {}
        }
    }
    set.into_iter().collect()
}

/// The verifier's view of a circuit: its shape and gates, the coset
/// shifts of the permutation argument, and the commitments to the fixed,
/// selector and permutation polynomials.
///
/// ```
/// use arrabbiata_generalized_plonk::{Circuit, setup};
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 1], 0, vec![]).unwrap();
/// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// let (_, vk) = setup(&c, &pcs).unwrap();
/// assert_eq!(vk.value.num_rows(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preprocessed<F, C, const N: usize, const D: usize>
where
    F: Field,
{
    log_size: u32,
    witness_columns: usize,
    public_inputs: usize,
    variables: Vec<Variable>,
    gates: Vec<Gate<F, N, D>>,
    bindings: Vec<(usize, usize)>,
    shifts: Vec<F>,
    fixed: Vec<C>,
    selectors: Vec<C>,
    sigmas: Vec<C>,
}

impl<F, C, const N: usize, const D: usize> Preprocessed<F, C, N, D>
where
    F: Field,
{
    /// The number of rows of the circuit.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::{Circuit, setup};
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq as F, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// let c = Circuit::<F, 1, 1>::new(4, 0, vec![], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// assert_eq!(setup(&c, &pcs).unwrap().1.value.num_rows(), 4);
    /// ```
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        1 << self.log_size
    }

    /// The number of public inputs of a statement.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::{Circuit, setup};
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq as F, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![], vec![], vec![], 3, vec![]).unwrap();
    /// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// assert_eq!(setup(&c, &pcs).unwrap().1.value.public_inputs(), 3);
    /// ```
    #[must_use]
    pub const fn public_inputs(&self) -> usize {
        self.public_inputs
    }
}

impl<F, C, const N: usize, const D: usize> Absorb<F> for Preprocessed<F, C, N, D>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_parts::<F, T, N, D>(
            transcript,
            (
                self.num_rows(),
                self.witness_columns,
                self.fixed.len(),
                self.selectors.len(),
                self.public_inputs,
            ),
            &self.variables,
            &self.gates,
            &self.bindings,
        );
        for s in &self.shifts {
            transcript.absorb(*s);
        }
        for c in self.fixed.iter().chain(&self.selectors).chain(&self.sigmas) {
            c.absorb_into(transcript);
        }
    }
}

/// The verifier key: the preprocessed circuit, committed with `PC`.
pub type VerifierKey<F, PC, const N: usize, const D: usize> =
    Tagged<PC, Preprocessed<F, Com<F, PC>, N, D>, 0x766b>;
/// The commitments to the witness polynomials.
pub type Wires<F, PC> = Tagged<PC, Commitments<Com<F, PC>>, 0x7769>;
/// The commitment to the grand product `z`.
pub type GrandProduct<F, PC> = Tagged<PC, Com<F, PC>, 0x7a7a>;
/// The commitment to the quotient `t`.
pub type Quotient<F, PC> = Tagged<PC, Com<F, PC>, 0x7474>;
/// The opening proofs.
pub type Openings<F, PC> = Tagged<PC, Opened<Prf<F, PC>>, 0x6f70>;

/// The keys produced by [`setup`].
pub type Keys<F, PC, const N: usize, const D: usize> =
    (ProverKey<F, PC, N, D>, VerifierKey<F, PC, N, D>);

/// The prover's view of a circuit: the verifier key, the circuit, and the
/// polynomials it commits to.
///
/// ```
/// use arrabbiata_generalized_plonk::{Circuit, setup};
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
/// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// let (pk, vk) = setup(&c, &pcs).unwrap();
/// assert_eq!(pk.verifier_key(), &vk);
/// ```
pub struct ProverKey<F, PC, const N: usize, const D: usize>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    vk: VerifierKey<F, PC, N, D>,
    circuit: Circuit<F, N, D>,
    domain: Domain<F>,
    fixed: Vec<UniPoly<F>>,
    selectors: Vec<UniPoly<F>>,
    sigmas: Vec<UniPoly<F>>,
    sigma_values: Vec<Vec<F>>,
}

impl<F, PC, const N: usize, const D: usize> ProverKey<F, PC, N, D>
where
    F: Field,
    PC: PolynomialCommitment<F>,
{
    /// The verifier key.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::{Circuit, setup};
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq as F, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// let (pk, _) = setup(&c, &pcs).unwrap();
    /// assert_eq!(pk.verifier_key().value.num_rows(), 2);
    /// ```
    pub const fn verifier_key(&self) -> &VerifierKey<F, PC, N, D> {
        &self.vk
    }

    /// The circuit.
    ///
    /// ```
    /// use arrabbiata_generalized_plonk::{Circuit, setup};
    /// use arrabbiata_ipa::Ipa;
    /// use arrabbiata_pasta::{Fq as F, Pallas};
    /// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
    /// let c = Circuit::<F, 1, 1>::new(2, 0, vec![], vec![], vec![], vec![], vec![], 0, vec![]).unwrap();
    /// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
    /// assert_eq!(setup(&c, &pcs).unwrap().0.circuit(), &c);
    /// ```
    pub const fn circuit(&self) -> &Circuit<F, N, D> {
        &self.circuit
    }
}

/// The transcript of generalized PlonK: the verifier key and the public
/// inputs, then the rounds.
pub type GeneralizedTranscript<F, PC, const N: usize, const D: usize> = Public<
    VerifierKey<F, PC, N, D>,
    Public<
        PublicInputs<F>,
        Message<
            Wires<F, PC>,
            Challenge<
                Challenge<
                    Message<
                        GrandProduct<F, PC>,
                        Challenge<
                            Message<
                                Quotient<F, PC>,
                                Challenge<Message<Opened<F>, Message<Openings<F, PC>, End>>>,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >,
>;

/// Generalized PlonK over `F`, compiled with the univariate scheme `PC`,
/// for gates of degree at most `D` in `N` variables.
///
/// ```
/// use arrabbiata_generalized_plonk::GeneralizedPlonk;
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_poseidon::PoseidonSponge;
/// use arrabbiata_transcript::typed::{Pattern, Protocol};
/// type P = <GeneralizedPlonk<F, Ipa<Pallas, PoseidonSponge<F>>, 3, 2> as Protocol>::Transcript;
/// assert_eq!(<P as Pattern>::MESSAGES, 5);
/// assert_eq!(<P as Pattern>::CHALLENGES, 4);
/// ```
#[derive(Debug)]
pub struct GeneralizedPlonk<F, PC, const N: usize, const D: usize>(PhantomData<Parameters<F, PC>>);

/// The type parameters of [`GeneralizedPlonk`], as a marker.
type Parameters<F, PC> = fn() -> (F, PC);

impl<F, PC, const N: usize, const D: usize> Protocol for GeneralizedPlonk<F, PC, N, D>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
{
    type Transcript = GeneralizedTranscript<F, PC, N, D>;
}

/// The proof of generalized PlonK.
pub type Proof<F, PC, const N: usize, const D: usize> = ProofOf<GeneralizedPlonk<F, PC, N, D>>;

fn interpolate<F>(domain: &Domain<F>, values: &[F]) -> Result<UniPoly<F>, Error>
where
    F: TwoAdicField,
{
    domain.interpolate(values).ok_or(Error::DomainTooLarge)
}

/// The shifts `g^k`, `k < W`, of the cosets `g^k H` labelling the witness
/// columns. `g^i H` and `g^j H` are disjoint iff `g^((i - j) n) != 1`.
fn coset_shifts<F>(w: usize, n: usize) -> Result<Vec<F>, Error>
where
    F: TwoAdicField,
{
    let g = F::MULTIPLICATIVE_GENERATOR;
    let gn = g.pow(n as u64);
    let mut power = F::ONE;
    for _ in 1..w {
        power = power * gn;
        if power == F::ONE {
            return Err(Error::CosetShifts);
        }
    }
    Ok((0..w).map(|k| g.pow(k as u64)).collect())
}

/// The factor by which the quotient domain exceeds the circuit's: the
/// largest degree of a term of the identity divided by `n`.
///
/// ```
/// use arrabbiata_generalized_plonk::{Circuit, quotient_factor};
/// use arrabbiata_pasta::Fq as F;
/// // Three witness columns: the permutation term has degree 4.
/// let c = Circuit::<F, 1, 1>::new(2, 3, vec![], vec![], vec![], vec![], (0..6).collect(), 0, vec![]).unwrap();
/// assert_eq!(quotient_factor(&c), 4);
/// ```
#[must_use]
pub fn quotient_factor<F, const N: usize, const D: usize>(circuit: &Circuit<F, N, D>) -> usize
where
    F: Field,
{
    let mut factor = (circuit.witness_columns() + 1).max(2);
    for gate in circuit.gates() {
        factor = factor.max(circuit.column_degree(gate));
    }
    factor
}

/// The number of coefficients a polynomial commitment for `circuit` must
/// accept: the quotient's, or a column's if larger.
///
/// ```
/// use arrabbiata_generalized_plonk::{Circuit, max_coefficients};
/// use arrabbiata_pasta::Fq as F;
/// let c = Circuit::<F, 1, 1>::new(4, 3, vec![], vec![], vec![], vec![], (0..12).collect(), 0, vec![]).unwrap();
/// assert_eq!(max_coefficients(&c), 12);
/// ```
#[must_use]
pub fn max_coefficients<F, const N: usize, const D: usize>(circuit: &Circuit<F, N, D>) -> usize
where
    F: Field,
{
    ((quotient_factor(circuit) - 1) * circuit.num_rows()).max(circuit.num_rows())
}

/// Preprocess a circuit: interpolate and commit to the fixed, selector and
/// permutation polynomials.
///
/// # Errors
///
/// [`Error::DomainTooLarge`] if the field has no domain of the circuit
/// size, [`Error::CosetShifts`] if it has no `W` disjoint cosets of it.
///
/// ```
/// use arrabbiata_generalized_plonk::{Circuit, setup};
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 1], 0, vec![]).unwrap();
/// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// assert!(setup(&c, &pcs).is_ok());
/// ```
pub fn setup<F, PC, const N: usize, const D: usize>(
    circuit: &Circuit<F, N, D>,
    pcs: &PC,
) -> Result<Keys<F, PC, N, D>, Error>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
{
    let n = circuit.num_rows();
    let w = circuit.witness_columns();
    let domain = Domain::<F>::containing(n).ok_or(Error::DomainTooLarge)?;
    let shifts = coset_shifts::<F>(w, n)?;
    let interpolate_all = |columns: &[Vec<F>]| -> Result<Vec<UniPoly<F>>, Error> {
        columns.iter().map(|c| interpolate(&domain, c)).collect()
    };
    let fixed = interpolate_all(circuit.fixed())?;
    let selectors = interpolate_all(circuit.selectors())?;
    let sigma = circuit.sigma();
    let sigma_values: Vec<Vec<F>> = (0..w)
        .map(|k| {
            (0..n)
                .map(|i| {
                    let target = sigma[k * n + i];
                    shifts[target / n] * domain.element(target % n)
                })
                .collect()
        })
        .collect();
    let sigmas = interpolate_all(&sigma_values)?;
    let commit = |p: &UniPoly<F>| pcs.commit(p.coefficients());
    let vk = VerifierKey::new(Preprocessed {
        log_size: domain.log_size(),
        witness_columns: w,
        public_inputs: circuit.public_inputs(),
        variables: circuit.variables().to_vec(),
        gates: circuit.gates().to_vec(),
        bindings: circuit.bindings().to_vec(),
        shifts,
        fixed: fixed.iter().map(commit).collect(),
        selectors: selectors.iter().map(commit).collect(),
        sigmas: sigmas.iter().map(commit).collect(),
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
fn coset_evaluations<F>(p: &UniPoly<F>, ext: &Domain<F>, shift: F) -> Result<Vec<F>, Error>
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
        .ok_or(Error::DomainTooLarge)
}

/// `w^k` for the generator `w` of `domain` and any `k` in `Z`.
fn root_power<F>(domain: &Domain<F>, rotation: Rotation) -> F
where
    F: TwoAdicField,
{
    let n = domain.size() as i64;
    domain
        .generator()
        .pow((i64::from(rotation.offset())).rem_euclid(n) as u64)
}

/// The value of the gates' variable `v` at `zeta`, from the evaluations.
fn at_zeta<F>(v: &Variable, evaluations: &Opened<F>, rotated: &[Variable], public: &[F]) -> F
where
    F: Field,
{
    match v {
        Variable::Witness { column, rotation } if *rotation == Rotation::CURR => {
            evaluations.witness[*column]
        }
        Variable::Fixed { column, rotation } if *rotation == Rotation::CURR => {
            evaluations.fixed[*column]
        }
        Variable::Public(i) => public[*i],
        _ => match rotated.binary_search(v) {
            Ok(i) => evaluations.rotated[i],
            // Not reached: the rotated variables are computed from the
            // circuit's own variables.
            Err(_) => F::ZERO,
        },
    }
}

/// Prove that the witness columns `columns` satisfy the circuit of `pk`
/// for the public inputs `public`. Returns the proof and the transcript
/// after the protocol.
///
/// # Errors
///
/// [`Error::PublicInputs`] for a wrong number of public inputs,
/// [`Error::Shape`] for columns of the wrong shape,
/// [`Error::Unsatisfied`] if they do not satisfy the circuit.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_generalized_plonk::{Circuit, Error, prove, setup};
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// // Column 0 at row 1 is public input 0.
/// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 1], 1, vec![(0, 1)]).unwrap();
/// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// let (pk, _) = setup(&c, &pcs).unwrap();
/// let columns = [vec![F::ZERO, F::from_u64(7)]];
/// assert!(prove(&pk, &pcs, vesta_sponge(2), &[F::from_u64(7)], &columns).is_ok());
/// assert_eq!(prove(&pk, &pcs, vesta_sponge(2), &[F::from_u64(8)], &columns).unwrap_err(), Error::Unsatisfied);
/// ```
#[allow(clippy::type_complexity, clippy::too_many_lines)]
pub fn prove<F, PC, T, const N: usize, const D: usize>(
    pk: &ProverKey<F, PC, N, D>,
    pcs: &PC,
    transcript: T,
    public: &[F],
    columns: &[Vec<F>],
) -> Result<(Proof<F, PC, N, D>, T), Error>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
    T: Transcript<F>,
{
    let circuit = &pk.circuit;
    let key = &pk.vk.value;
    let domain = &pk.domain;
    let n = domain.size();
    let w = circuit.witness_columns();
    if public.len() != circuit.public_inputs() {
        return Err(Error::PublicInputs);
    }
    if columns.len() != w || columns.iter().any(|c| c.len() != n) {
        return Err(Error::Shape);
    }
    if !circuit.is_satisfied(columns, public) {
        return Err(Error::Unsatisfied);
    }
    let commit = |p: &UniPoly<F>| pcs.commit(p.coefficients());

    // Round 1: the witness columns.
    let wires: Vec<UniPoly<F>> = columns
        .iter()
        .map(|c| interpolate(domain, c))
        .collect::<Result<_, _>>()?;
    let tape = prover_tape::<F, T, GeneralizedPlonk<F, PC, N, D>>(transcript)
        .public(&pk.vk)
        .public(&PublicInputs(public.to_vec()))
        .send(Tagged::new(Commitments(wires.iter().map(commit).collect())));
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();

    // Round 2: the grand product of the copy permutation.
    let mut z_values = vec![F::ONE; n];
    for i in 0..n - 1 {
        let x = domain.element(i);
        let mut num = F::ONE;
        let mut den = F::ONE;
        for ((column, shift), sigma) in columns.iter().zip(&key.shifts).zip(&pk.sigma_values) {
            num = num * (column[i] + beta * *shift * x + gamma);
            den = den * (column[i] + beta * sigma[i] + gamma);
        }
        // den is zero only with probability about W n / |F| over gamma.
        z_values[i + 1] = z_values[i] * num * den.inverse().ok_or(Error::Unsatisfied)?;
    }
    let z = interpolate(domain, &z_values)?;
    let tape = tape.send(Tagged::new(commit(&z)));
    let (alpha, tape) = tape.challenge();

    // Round 3: the quotient, pointwise on the coset g H_E.
    let factor = quotient_factor(circuit);
    let ext = Domain::<F>::containing(factor * n).ok_or(Error::DomainTooLarge)?;
    let size = ext.size();
    let step = size / n;
    let shift = F::MULTIPLICATIVE_GENERATOR;
    let on_coset = |p: &UniPoly<F>| coset_evaluations(p, &ext, shift);
    let on_coset_all = |polys: &[UniPoly<F>]| -> Result<Vec<Vec<F>>, Error> {
        polys.iter().map(on_coset).collect()
    };
    let wire_values = on_coset_all(&wires)?;
    let fixed_values = on_coset_all(&pk.fixed)?;
    let selector_values = on_coset_all(&pk.selectors)?;
    let sigma_values = on_coset_all(&pk.sigmas)?;
    let z_coset = on_coset(&z)?;
    let mut first = vec![F::ZERO; n];
    first[0] = F::ONE;
    let l1 = on_coset(&interpolate(domain, &first)?)?;
    let mut bound = vec![F::ZERO; n];
    let mut bound_values = vec![F::ZERO; n];
    for (j, row) in circuit.bindings() {
        bound[*row] = F::ONE;
        bound_values[*row] = public[*j];
    }
    let lb = on_coset(&interpolate(domain, &bound)?)?;
    let pi = on_coset(&interpolate(domain, &bound_values)?)?;
    let shift_n = shift.pow(n as u64);
    let root_n = ext.generator().pow(n as u64);
    let mut vanishing_inverse = Vec::with_capacity(step);
    let mut power = F::ONE;
    for _ in 0..step {
        let zh = shift_n * power - F::ONE;
        vanishing_inverse.push(zh.inverse().ok_or(Error::CosetShifts)?);
        power = power * root_n;
    }
    let rotate = |values: &[F], point: usize, rotation: Rotation| -> F {
        let offset = (i64::from(rotation.offset()) * step as i64).rem_euclid(size as i64) as usize;
        values[(point + offset) % size]
    };
    let mut t_values = vec![F::ZERO; size];
    let mut x = shift;
    for (point, t_value) in t_values.iter_mut().enumerate() {
        let vars = circuit.point(|v| match v {
            Variable::Witness { column, rotation } => {
                rotate(&wire_values[*column], point, *rotation)
            }
            Variable::Fixed { column, rotation } => {
                rotate(&fixed_values[*column], point, *rotation)
            }
            Variable::Public(i) => public[*i],
        });
        let mut acc = F::ZERO;
        let mut power = F::ONE;
        for gate in circuit.gates() {
            let selector = match gate.selector() {
                Some(q) => selector_values[q][point],
                None => F::ONE,
            };
            acc = acc + power * selector * gate.polynomial().eval(&vars);
            power = power * alpha;
        }
        if w > 0 {
            acc = acc + power * (lb[point] * wire_values[0][point] - pi[point]);
        }
        power = power * alpha;
        let next = (point + step) % size;
        let mut left = z_coset[point];
        let mut right = z_coset[next];
        for k in 0..w {
            let wv = wire_values[k][point];
            left = left * (wv + beta * key.shifts[k] * x + gamma);
            right = right * (wv + beta * sigma_values[k][point] + gamma);
        }
        acc = acc + power * (left - right);
        power = power * alpha;
        acc = acc + power * l1[point] * (z_coset[point] - F::ONE);
        *t_value = acc * vanishing_inverse[point % step];
        x = x * ext.generator();
    }
    let t_shifted = ext.interpolate(&t_values).ok_or(Error::DomainTooLarge)?;
    let shift_inverse = shift.inverse().ok_or(Error::CosetShifts)?;
    let degree_bound = (factor - 1) * n;
    let mut t_coefficients = Vec::with_capacity(degree_bound);
    let mut power = F::ONE;
    for (k, c) in t_shifted.coefficients().iter().enumerate() {
        let value = *c * power;
        if k < degree_bound {
            t_coefficients.push(value);
        } else if !value.is_zero() {
            // Not reached for a satisfied circuit: the numerator vanishes
            // on H and has degree below factor * n.
            return Err(Error::Unsatisfied);
        }
        power = power * shift_inverse;
    }
    let t = UniPoly::new(t_coefficients);
    let tape = tape.send(Tagged::new(commit(&t)));
    let (zeta, tape) = tape.challenge();

    // Round 4: evaluations and openings.
    let rotated_vars = rotated(circuit.variables());
    let open = |p: &UniPoly<F>, point: &F| pcs.open(p.coefficients(), point);
    let opened = Opened {
        witness: wires.iter().map(|p| open(p, &zeta)).collect(),
        fixed: pk.fixed.iter().map(|p| open(p, &zeta)).collect(),
        selectors: pk.selectors.iter().map(|p| open(p, &zeta)).collect(),
        sigmas: pk.sigmas.iter().map(|p| open(p, &zeta)).collect(),
        z: open(&z, &zeta),
        t: open(&t, &zeta),
        z_next: open(&z, &(zeta * domain.generator())),
        rotated: rotated_vars
            .iter()
            .map(|v| match v {
                Variable::Witness { column, rotation } => {
                    open(&wires[*column], &(zeta * root_power(domain, *rotation)))
                }
                Variable::Fixed { column, rotation } => {
                    open(&pk.fixed[*column], &(zeta * root_power(domain, *rotation)))
                }
                Variable::Public(_) => unreachable!("rotated variables read columns"),
            })
            .collect(),
    };
    let evaluations = opened.clone().map(|(v, _)| v);
    let proofs = opened.map(|(_, p)| p);
    let (transcript, proof) = tape.send(evaluations).send(Tagged::new(proofs)).finish();
    Ok((proof, transcript))
}

/// Verify `proof` for the public inputs `public` against `vk`. Returns the
/// witness commitments the proof makes, and the transcript after the
/// protocol.
///
/// # Errors
///
/// [`Error::PublicInputs`] for a wrong number of public inputs,
/// [`Error::Shape`] if the proof has the wrong shape,
/// [`Error::Opening`] if an opening fails, [`Error::IdentityFailed`] if the
/// identity does not hold at `zeta`.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_generalized_plonk::{Circuit, prove, setup, verify};
/// use arrabbiata_ipa::Ipa;
/// use arrabbiata_pasta::{Fq as F, Pallas};
/// use arrabbiata_poseidon::{pallas_sponge, vesta_sponge};
/// let c = Circuit::<F, 1, 1>::new(2, 1, vec![], vec![], vec![], vec![], vec![0, 1], 1, vec![(0, 1)]).unwrap();
/// let pcs = Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), 4, vesta_sponge(1));
/// let (pk, vk) = setup(&c, &pcs).unwrap();
/// let (proof, _) = prove(&pk, &pcs, vesta_sponge(2), &[F::from_u64(7)], &[vec![F::ZERO, F::from_u64(7)]]).unwrap();
/// assert!(verify(&vk, &pcs, vesta_sponge(2), &[F::from_u64(7)], proof.clone()).is_ok());
/// assert!(verify(&vk, &pcs, vesta_sponge(2), &[F::from_u64(8)], proof).is_err());
/// ```
#[allow(clippy::type_complexity)]
pub fn verify<F, PC, T, const N: usize, const D: usize>(
    vk: &VerifierKey<F, PC, N, D>,
    pcs: &PC,
    transcript: T,
    public: &[F],
    proof: Proof<F, PC, N, D>,
) -> Result<(Vec<Com<F, PC>>, T), Error>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
    T: Transcript<F>,
{
    let key = &vk.value;
    if public.len() != key.public_inputs {
        return Err(Error::PublicInputs);
    }
    let domain = Domain::<F>::new(key.log_size).ok_or(Error::DomainTooLarge)?;
    let (wires, tape) = verifier_tape::<F, T, GeneralizedPlonk<F, PC, N, D>>(transcript, proof)
        .public(vk)
        .public(&PublicInputs(public.to_vec()))
        .read();
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();
    let (z_commitment, tape) = tape.read();
    let (alpha, tape) = tape.challenge();
    let (t_commitment, tape) = tape.read();
    let (zeta, tape) = tape.challenge();
    let (evaluations, tape) = tape.read();
    let (proofs, tape) = tape.read();
    let transcript = tape.finish();
    let (wires, z_commitment, t_commitment, proofs) = (
        wires.value.0,
        z_commitment.value,
        t_commitment.value,
        proofs.value,
    );

    let w = key.witness_columns;
    let rotated_vars = rotated(&key.variables);
    let counts = (w, key.fixed.len(), key.selectors.len(), rotated_vars.len());
    if wires.len() != w || !evaluations.has_shape(counts) || !proofs.has_shape(counts) {
        return Err(Error::Shape);
    }

    // The commitment and the point of every opening, in the shape of the
    // evaluations and the proofs.
    let at = |r: Rotation| zeta * root_power(&domain, r);
    let claims = Opened {
        witness: wires.iter().map(|c| (c, zeta)).collect(),
        fixed: key.fixed.iter().map(|c| (c, zeta)).collect(),
        selectors: key.selectors.iter().map(|c| (c, zeta)).collect(),
        sigmas: key.sigmas.iter().map(|c| (c, zeta)).collect(),
        z: (&z_commitment, zeta),
        t: (&t_commitment, zeta),
        z_next: (&z_commitment, at(Rotation::NEXT)),
        rotated: rotated_vars
            .iter()
            .map(|v| match v {
                Variable::Witness { column, rotation } => (&wires[*column], at(*rotation)),
                Variable::Fixed { column, rotation } => (&key.fixed[*column], at(*rotation)),
                Variable::Public(_) => unreachable!("rotated variables read columns"),
            })
            .collect(),
    };
    for (i, ((commitment, point), (value, opening))) in claims
        .iter()
        .zip(evaluations.iter().zip(proofs.iter()))
        .enumerate()
    {
        if !pcs.verify(commitment, point, *value, opening) {
            return Err(Error::Opening(i));
        }
    }

    // The identity at zeta.
    let mut vars = [F::ZERO; N];
    for (slot, v) in vars.iter_mut().zip(&key.variables) {
        *slot = at_zeta(v, &evaluations, &rotated_vars, public);
    }
    let mut lhs = F::ZERO;
    let mut power = F::ONE;
    for gate in &key.gates {
        let selector = match gate.selector() {
            Some(q) => evaluations.selectors[q],
            None => F::ONE,
        };
        lhs = lhs + power * selector * gate.polynomial().eval(&vars);
        power = power * alpha;
    }
    if w > 0 {
        let mut lb = F::ZERO;
        let mut pi = F::ZERO;
        for (j, row) in &key.bindings {
            let l = domain.lagrange_at(*row, zeta);
            lb = lb + l;
            pi = pi + public[*j] * l;
        }
        lhs = lhs + power * (lb * evaluations.witness[0] - pi);
    }
    power = power * alpha;
    let mut left = evaluations.z;
    let mut right = evaluations.z_next;
    for k in 0..w {
        let wv = evaluations.witness[k];
        left = left * (wv + beta * key.shifts[k] * zeta + gamma);
        right = right * (wv + beta * evaluations.sigmas[k] + gamma);
    }
    lhs = lhs + power * (left - right);
    power = power * alpha;
    lhs = lhs + power * domain.first_lagrange_at(zeta) * (evaluations.z - F::ONE);
    if lhs == evaluations.t * domain.vanishing_at(zeta) {
        Ok((wires, transcript))
    } else {
        Err(Error::IdentityFailed)
    }
}
