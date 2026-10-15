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
use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_mvpoly::MVPoly;
use arrabbiata_permutation::{self as permutation, Permutation};
use arrabbiata_poly::{Domain, Rotation, UniPoly};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, absorb_slice,
    typed::{
        Challenge, End, Label, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape,
    },
};

use crate::{
    Circuit, Error, Gate, Variable,
    circuit::absorb_parts,
    lookup::{self, Argument, At, Lookup},
};

/// The commitment type of a scheme.
type Com<F, PC> = <PC as PolynomialCommitment<F>>::Commitment;

/// The opening proof type of a scheme.
type Prf<F, PC> = <PC as PolynomialCommitment<F>>::Proof;

/// A value of type `T` produced with the commitment scheme `PC`: its
/// label combines the scheme's with `TAG`, so a proof made with one
/// scheme is not a proof for another.
pub struct Tagged<PC, T, const TAG: u64> {
    /// The value.
    pub value: T,
    scheme: PhantomData<fn() -> PC>,
}

impl<PC, T, const TAG: u64> Tagged<PC, T, TAG> {
    /// Tag `value` with the scheme `PC`.
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
/// `zeta w^k`; then the lookup openings, in the order of the lookups:
/// each committed lookup column (`h_1, h_2, Z` for plookup, `m, phi` for
/// LogUp) at `zeta` and `zeta w`, then, for plookup, each table column at
/// `zeta w`.
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
    /// The lookup openings.
    pub lookups: Vec<T>,
}

impl<T> Opened<T> {
    /// Every value, in the order of the openings.
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
            .chain(&self.lookups)
    }

    /// Every value, mutably, in the order of the openings.
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
            .chain(&mut self.lookups)
    }

    /// Whether there is one value per witness, fixed, selector,
    /// permutation, rotated polynomial and lookup opening, for the counts
    /// `(witness, permuted, fixed, selectors, rotated, lookups)`.
    fn has_shape(&self, (w, k, x, q, r, l): (usize, usize, usize, usize, usize, usize)) -> bool {
        self.witness.len() == w
            && self.fixed.len() == x
            && self.selectors.len() == q
            && self.sigmas.len() == k
            && self.rotated.len() == r
            && self.lookups.len() == l
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
            lookups: self.lookups.into_iter().map(&mut f).collect(),
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
        absorb_length(transcript, self.lookups.len());
        for value in self.iter() {
            value.absorb_into(transcript);
        }
    }
}

/// The column variables read at a rotation other than the current row,
/// with the rotation set to the one read, sorted and without repetition:
/// the openings at `zeta w^k` a proof makes.
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preprocessed<F, C, const N: usize, const D: usize>
where
    F: Field,
{
    log_size: u32,
    witness_columns: usize,
    permuted: Vec<usize>,
    public_inputs: usize,
    variables: Vec<Variable>,
    gates: Vec<Gate<F, N, D>>,
    bindings: Vec<(usize, usize)>,
    lookups: Vec<Lookup<F, N, D>>,
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
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        1 << self.log_size
    }

    /// The number of public inputs of a statement.
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
            &self.lookups,
            &self.permuted,
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
/// The commitments to the lookup columns drawn before `beta, gamma`
/// (`h_1, h_2` for plookup, `m` for LogUp), in the order of the lookups.
pub type LookupColumns<F, PC> = Tagged<PC, Commitments<Com<F, PC>>, 0x6c63>;
/// The commitments to the lookup columns drawn after `beta, gamma` (`Z`
/// for plookup, `phi` for LogUp), in the order of the lookups.
pub type LookupProducts<F, PC> = Tagged<PC, Commitments<Com<F, PC>>, 0x6c70>;
/// The commitment to the quotient `t`.
pub type Quotient<F, PC> = Tagged<PC, Com<F, PC>, 0x7474>;
/// The opening proofs.
pub type Openings<F, PC> = Tagged<PC, Opened<Prf<F, PC>>, 0x6f70>;

/// The keys produced by [`setup`].
pub type Keys<F, PC, const N: usize, const D: usize> =
    (ProverKey<F, PC, N, D>, VerifierKey<F, PC, N, D>);

/// The prover's view of a circuit: the verifier key, the circuit, and the
/// polynomials it commits to.
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
    pub const fn verifier_key(&self) -> &VerifierKey<F, PC, N, D> {
        &self.vk
    }

    /// The circuit.
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
                Message<
                    LookupColumns<F, PC>,
                    Challenge<
                        Challenge<
                            Message<
                                GrandProduct<F, PC>,
                                Message<
                                    LookupProducts<F, PC>,
                                    Challenge<
                                        Message<
                                            Quotient<F, PC>,
                                            Challenge<
                                                Message<Opened<F>, Message<Openings<F, PC>, End>>,
                                            >,
                                        >,
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

/// Generalized PlonK over `F`, compiled with the univariate scheme `PC`,
/// for gates of degree at most `D` in `N` variables.
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

/// The factor by which the quotient domain exceeds the circuit's: the
/// largest degree of a term of the identity divided by `n`.
#[must_use]
pub fn quotient_factor<F, const N: usize, const D: usize>(circuit: &Circuit<F, N, D>) -> usize
where
    F: Field,
{
    let mut factor = permutation::degree(circuit.permuted().len()).max(2);
    for gate in circuit.gates() {
        factor = factor.max(circuit.column_degree(gate));
    }
    // A lookup's terms multiply its query by two columns (plookup: Z and
    // the table; LogUp: the running sum and the table), and plookup's
    // other side multiplies three columns.
    for lookup in circuit.lookups() {
        factor = factor.max(circuit.query_degree(lookup) + 2).max(3);
    }
    factor
}

/// The number of coefficients a polynomial commitment for `circuit` must
/// accept: the quotient's, or a column's if larger.
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
    let k = circuit.permuted().len();
    let shifts = permutation::shifts::<F>(k, n).map_err(|_| Error::CosetShifts)?;
    let interpolate_all = |columns: &[Vec<F>]| -> Result<Vec<UniPoly<F>>, Error> {
        columns.iter().map(|c| interpolate(&domain, c)).collect()
    };
    let fixed = interpolate_all(circuit.fixed())?;
    let selectors = interpolate_all(circuit.selectors())?;
    let copies = Permutation::new(k, n, circuit.sigma().to_vec()).map_err(|_| Error::Shape)?;
    let sigma_values = permutation::labels(&copies, &shifts, &domain);
    let sigmas = interpolate_all(&sigma_values)?;
    let commit = |p: &UniPoly<F>| pcs.commit(p.coefficients());
    let vk = VerifierKey::new(Preprocessed {
        log_size: domain.log_size(),
        witness_columns: w,
        permuted: circuit.permuted().to_vec(),
        public_inputs: circuit.public_inputs(),
        variables: circuit.variables().to_vec(),
        gates: circuit.gates().to_vec(),
        bindings: circuit.bindings().to_vec(),
        lookups: circuit.lookups().to_vec(),
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

/// The number of lookup openings a proof of `lookups` makes: each
/// committed lookup column at `zeta` and `zeta w`, and plookup's table
/// columns at `zeta w`.
fn lookup_openings<F, const N: usize, const D: usize>(lookups: &[Lookup<F, N, D>]) -> usize
where
    F: Field,
{
    lookups
        .iter()
        .map(|l| {
            let (before, after) = l.argument().columns();
            let table = if l.argument() == Argument::Plookup {
                l.table().len()
            } else {
                0
            };
            2 * (before + after) + table
        })
        .sum()
}

/// What the prover computes for one lookup: the query and the table on
/// every row, and the polynomials it commits to before and after
/// `beta, gamma`.
struct LookupWitness<F> {
    queries: Vec<F>,
    table: Vec<F>,
    before: Vec<Vec<F>>,
    after: Vec<Vec<F>>,
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
/// A complete run, from a circuit to a verified proof, is in
/// `tests/examples.rs`.
#[allow(clippy::type_complexity, clippy::too_many_lines)]
pub fn prove<F, PC, T, const N: usize, const D: usize>(
    pk: &ProverKey<F, PC, N, D>,
    pcs: &PC,
    transcript: T,
    public: &[F],
    columns: &[Vec<F>],
) -> Result<(Proof<F, PC, N, D>, T), Error>
where
    F: TwoAdicField + PrimeField,
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
    let interpolate_all = |values: &[Vec<F>]| -> Result<Vec<UniPoly<F>>, Error> {
        values.iter().map(|c| interpolate(domain, c)).collect()
    };

    // Round 1: the witness columns, then the lookup combiner theta.
    let wires = interpolate_all(columns)?;
    let tape = prover_tape::<F, T, GeneralizedPlonk<F, PC, N, D>>(transcript)
        .public(&pk.vk)
        .public(&PublicInputs(public.to_vec()))
        .send(Tagged::new(Commitments(wires.iter().map(commit).collect())));
    let (theta, tape) = tape.challenge();

    // Round 2: the lookup columns drawn before beta, gamma.
    let row_point = |row: usize| {
        circuit.point(|v| match v {
            Variable::Witness { column, rotation } => columns[*column][rotation.apply(row, n)],
            Variable::Fixed { column, rotation } => {
                circuit.fixed()[*column][rotation.apply(row, n)]
            }
            Variable::Public(i) => public[*i],
        })
    };
    let mut lookups = Vec::with_capacity(circuit.lookups().len());
    for l in circuit.lookups() {
        let queries: Vec<F> = (0..n)
            .map(|row| {
                let selector = match l.selector() {
                    Some(q) => circuit.selectors()[q][row],
                    None => F::ONE,
                };
                lookup::query(l, &row_point(row), selector, theta)
            })
            .collect();
        let table: Vec<F> = (0..n)
            .map(|row| lookup::compress(l.table().iter().map(|k| circuit.fixed()[*k][row]), theta))
            .collect();
        let before = match l.argument() {
            Argument::Plookup => {
                let (h1, h2) = lookup::sorted_halves(&queries, &table)?;
                vec![h1, h2]
            }
            Argument::LogUp => vec![lookup::multiplicities(&queries, &table)?],
        };
        lookups.push(LookupWitness {
            queries,
            table,
            before,
            after: Vec::new(),
        });
    }
    let mut before_polys = Vec::new();
    for l in &lookups {
        before_polys.push(interpolate_all(&l.before)?);
    }
    let tape = tape.send(Tagged::new(Commitments(
        before_polys.iter().flatten().map(commit).collect(),
    )));
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();

    // Round 3: the grand product of the copy permutation, then the lookup
    // columns drawn after beta, gamma.
    let permuted_columns: Vec<&[F]> = circuit
        .permuted()
        .iter()
        .map(|c| columns[*c].as_slice())
        .collect();
    // A denominator is zero only with probability about k n / |F| over
    // gamma.
    let z_values = permutation::grand_product(
        &permuted_columns,
        &key.shifts,
        &pk.sigma_values,
        domain,
        beta,
        gamma,
    )
    .map_err(|_| Error::Unsatisfied)?;
    let z = interpolate(domain, &z_values)?;
    let tape = tape.send(Tagged::new(commit(&z)));
    for (l, data) in circuit.lookups().iter().zip(&mut lookups) {
        data.after = match l.argument() {
            Argument::Plookup => vec![lookup::plookup_product(
                &data.queries,
                &data.table,
                &data.before[0],
                &data.before[1],
                beta,
                gamma,
            )?],
            Argument::LogUp => vec![lookup::logup_sum(
                &data.queries,
                &data.table,
                &data.before[0],
                gamma,
            )?],
        };
    }
    let mut after_polys = Vec::new();
    for l in &lookups {
        after_polys.push(interpolate_all(&l.after)?);
    }
    let tape = tape.send(Tagged::new(Commitments(
        after_polys.iter().flatten().map(commit).collect(),
    )));
    let (alpha, tape) = tape.challenge();

    // Round 4: the quotient, pointwise on the coset g H_E.
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
    let mut lookup_values = Vec::with_capacity(lookups.len());
    for (before, after) in before_polys.iter().zip(&after_polys) {
        let mut values = on_coset_all(before)?;
        values.extend(on_coset_all(after)?);
        lookup_values.push(values);
    }
    let z_coset = on_coset(&z)?;
    let indicator = |row: usize| -> Result<Vec<F>, Error> {
        let mut values = vec![F::ZERO; n];
        values[row] = F::ONE;
        on_coset(&interpolate(domain, &values)?)
    };
    let l1 = indicator(0)?;
    let l_last = indicator(n - 1)?;
    let w_last = domain.element(n - 1);
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
        let witness_here: Vec<F> = circuit
            .permuted()
            .iter()
            .map(|c| wire_values[*c][point])
            .collect();
        let sigmas_here: Vec<F> = sigma_values.iter().map(|s| s[point]).collect();
        let at = permutation::At {
            x,
            witness: &witness_here,
            sigmas: &sigmas_here,
            z: z_coset[point],
            z_next: z_coset[next],
            l0: l1[point],
        };
        let [step_term, boundary] = permutation::terms(&at, &key.shifts, beta, gamma);
        acc = acc + power * step_term;
        power = power * alpha;
        acc = acc + power * boundary;
        for (l, values) in circuit.lookups().iter().zip(&lookup_values) {
            power = power * alpha;
            let selector = match l.selector() {
                Some(q) => selector_values[q][point],
                None => F::ONE,
            };
            let table =
                |p: usize| lookup::compress(l.table().iter().map(|k| fixed_values[*k][p]), theta);
            let at = At {
                f: lookup::query(l, &vars, selector, theta),
                t: (table(point), table(next)),
                columns: values.iter().map(|v| (v[point], v[next])).collect(),
                boundary: (l1[point], l_last[point], x - w_last),
            };
            let terms = lookup::terms(l.argument(), &at, beta, gamma);
            for (i, term) in terms.iter().enumerate() {
                if i > 0 {
                    power = power * alpha;
                }
                acc = acc + power * *term;
            }
        }
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

    // Round 5: evaluations and openings.
    let zeta_next = zeta * domain.generator();
    let rotated_vars = rotated(circuit.variables());
    let open = |p: &UniPoly<F>, point: &F| pcs.open(p.coefficients(), point);
    let mut lookup_openings = Vec::new();
    for ((l, before), after) in circuit
        .lookups()
        .iter()
        .zip(&before_polys)
        .zip(&after_polys)
    {
        for p in before.iter().chain(after) {
            lookup_openings.push(open(p, &zeta));
            lookup_openings.push(open(p, &zeta_next));
        }
        if l.argument() == Argument::Plookup {
            for k in l.table() {
                lookup_openings.push(open(&pk.fixed[*k], &zeta_next));
            }
        }
    }
    let opened = Opened {
        witness: wires.iter().map(|p| open(p, &zeta)).collect(),
        fixed: pk.fixed.iter().map(|p| open(p, &zeta)).collect(),
        selectors: pk.selectors.iter().map(|p| open(p, &zeta)).collect(),
        sigmas: pk.sigmas.iter().map(|p| open(p, &zeta)).collect(),
        z: open(&z, &zeta),
        t: open(&t, &zeta),
        z_next: open(&z, &zeta_next),
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
        lookups: lookup_openings,
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
/// A complete run, from a circuit to a verified proof, is in
/// `tests/examples.rs`.
#[allow(clippy::type_complexity, clippy::too_many_lines)]
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
    let n = domain.size();
    let (wires, tape) = verifier_tape::<F, T, GeneralizedPlonk<F, PC, N, D>>(transcript, proof)
        .public(vk)
        .public(&PublicInputs(public.to_vec()))
        .read();
    let (theta, tape) = tape.challenge();
    let (lookup_before, tape) = tape.read();
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();
    let (z_commitment, tape) = tape.read();
    let (lookup_after, tape) = tape.read();
    let (alpha, tape) = tape.challenge();
    let (t_commitment, tape) = tape.read();
    let (zeta, tape) = tape.challenge();
    let (evaluations, tape) = tape.read();
    let (proofs, tape) = tape.read();
    let transcript = tape.finish();
    let (wires, lookup_before, z_commitment, lookup_after, t_commitment, proofs) = (
        wires.value.0,
        lookup_before.value.0,
        z_commitment.value,
        lookup_after.value.0,
        t_commitment.value,
        proofs.value,
    );

    let w = key.witness_columns;
    let rotated_vars = rotated(&key.variables);
    let (before_count, after_count) = key.lookups.iter().fold((0, 0), |(b, a), l| {
        let (lb, la) = l.argument().columns();
        (b + lb, a + la)
    });
    let counts = (
        w,
        key.permuted.len(),
        key.fixed.len(),
        key.selectors.len(),
        rotated_vars.len(),
        lookup_openings(&key.lookups),
    );
    if wires.len() != w
        || lookup_before.len() != before_count
        || lookup_after.len() != after_count
        || !evaluations.has_shape(counts)
        || !proofs.has_shape(counts)
    {
        return Err(Error::Shape);
    }

    // The commitment and the point of every opening, in the shape of the
    // evaluations and the proofs.
    let at = |r: Rotation| zeta * root_power(&domain, r);
    let zeta_next = at(Rotation::NEXT);
    let mut lookup_claims = Vec::new();
    let (mut b, mut a) = (0, 0);
    for l in &key.lookups {
        let (lb, la) = l.argument().columns();
        for c in lookup_before[b..b + lb]
            .iter()
            .chain(&lookup_after[a..a + la])
        {
            lookup_claims.push((c, zeta));
            lookup_claims.push((c, zeta_next));
        }
        (b, a) = (b + lb, a + la);
        if l.argument() == Argument::Plookup {
            for k in l.table() {
                lookup_claims.push((&key.fixed[*k], zeta_next));
            }
        }
    }
    let claims = Opened {
        witness: wires.iter().map(|c| (c, zeta)).collect(),
        fixed: key.fixed.iter().map(|c| (c, zeta)).collect(),
        selectors: key.selectors.iter().map(|c| (c, zeta)).collect(),
        sigmas: key.sigmas.iter().map(|c| (c, zeta)).collect(),
        z: (&z_commitment, zeta),
        t: (&t_commitment, zeta),
        z_next: (&z_commitment, zeta_next),
        rotated: rotated_vars
            .iter()
            .map(|v| match v {
                Variable::Witness { column, rotation } => (&wires[*column], at(*rotation)),
                Variable::Fixed { column, rotation } => (&key.fixed[*column], at(*rotation)),
                Variable::Public(_) => unreachable!("rotated variables read columns"),
            })
            .collect(),
        lookups: lookup_claims,
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
    let l0 = domain.first_lagrange_at(zeta);
    let permuted_here: Vec<F> = key
        .permuted
        .iter()
        .map(|c| evaluations.witness[*c])
        .collect();
    let at = permutation::At {
        x: zeta,
        witness: &permuted_here,
        sigmas: &evaluations.sigmas,
        z: evaluations.z,
        z_next: evaluations.z_next,
        l0,
    };
    let [step, copy_boundary] = permutation::terms(&at, &key.shifts, beta, gamma);
    lhs = lhs + power * step;
    power = power * alpha;
    lhs = lhs + power * copy_boundary;
    let boundary = (
        l0,
        domain.lagrange_at(n - 1, zeta),
        zeta - domain.element(n - 1),
    );
    let mut offset = 0;
    for l in &key.lookups {
        power = power * alpha;
        let (lb, la) = l.argument().columns();
        let opened = &evaluations.lookups[offset..];
        let columns: Vec<(F, F)> = (0..lb + la)
            .map(|c| (opened[2 * c], opened[2 * c + 1]))
            .collect();
        let table_next = if l.argument() == Argument::Plookup {
            let next = &opened[2 * (lb + la)..2 * (lb + la) + l.table().len()];
            lookup::compress(next.iter().copied(), theta)
        } else {
            F::ZERO
        };
        let selector = match l.selector() {
            Some(q) => evaluations.selectors[q],
            None => F::ONE,
        };
        let table = lookup::compress(l.table().iter().map(|k| evaluations.fixed[*k]), theta);
        let at = At {
            f: lookup::query(l, &vars, selector, theta),
            t: (table, table_next),
            columns,
            boundary,
        };
        for (i, term) in lookup::terms(l.argument(), &at, beta, gamma)
            .iter()
            .enumerate()
        {
            if i > 0 {
                power = power * alpha;
            }
            lhs = lhs + power * *term;
        }
        offset += 2 * (lb + la)
            + if l.argument() == Argument::Plookup {
                l.table().len()
            } else {
                0
            };
    }
    if lhs == evaluations.t * domain.vanishing_at(zeta) {
        Ok((wires, transcript))
    } else {
        Err(Error::IdentityFailed)
    }
}
