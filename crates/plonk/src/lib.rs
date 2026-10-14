//! Vanilla PlonK, as a toy end-to-end example of the Arrabbiata stack.
//!
//! This crate compiles the PlonK polynomial IOP (Gabizon, Williamson,
//! Ciobotaru, ePrint 2019/953) with any univariate
//! [`PolynomialCommitment`], over any [`TwoAdicField`], with the typed
//! transcript of [`arrabbiata_transcript::typed`]. Circuits come from
//! [`arrabbiata_frontend`].
//!
//! ```
//! use arrabbiata_commit::{RevealCommitment, Univariate};
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_frontend::build;
//! use arrabbiata_plonk::{prove, setup, verify};
//! use arrabbiata_transcript::toy_sponge;
//!
//! // x^3 + x + 5 = 35, with x = 3 private and 35 public.
//! let (circuit, witness, ()) = build::<F, _, _>(|b| {
//!     let x = b.witness(F::new(3));
//!     let x2 = b.mul(&x, &x);
//!     let x3 = b.mul(&x2, &x);
//!     let sum = b.add(&x3, &x);
//!     let lhs = b.add_constant(&sum, F::new(5));
//!     let rhs = b.public_input(F::new(35));
//!     b.assert_equal(&lhs, &rhs);
//! });
//!
//! let pcs = RevealCommitment::<Univariate>::new();
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, toy_sponge(0), &witness).unwrap();
//! verify(&vk, &pcs, toy_sponge(0), witness.public_inputs(), proof).unwrap();
//! ```
//!
//! # The protocol
//!
//! The rounds, in the order of [`PlonkTranscript`]:
//!
//! 1. the prover commits to the wire polynomials `a, b, c`;
//! 2. the verifier sends `beta, gamma`; the prover commits to the grand
//!    product `z` of the copy-constraint argument;
//! 3. the verifier sends `alpha`; the prover commits to the quotient
//!    `t = (gate + PI + alpha * perm + alpha^2 * (z - 1) L_1) / Z_H`;
//! 4. the verifier sends `zeta`; the prover sends the evaluations at `zeta`
//!    (and `z` at `zeta * w`) with their opening proofs;
//! 5. the verifier checks the openings and the identity at `zeta`.
//!
//! # Deviations from the paper (why this is a toy)
//!
//! - No blinding factors: the proof is not zero-knowledge.
//! - No linearization and no batched openings: every polynomial is opened
//!   at `zeta` separately, selectors and permutation polynomials included.
//!   This changes the cost, not the soundness argument.
//! - The quotient is committed as one polynomial rather than three, which
//!   matters only for schemes with a degree bound (KZG).
//! - Polynomial products are computed by schoolbook multiplication.

#![no_std]

extern crate alloc;

use alloc::{vec, vec::Vec};
use core::{fmt, marker::PhantomData};

use arrabbiata_commit::{PolynomialCommitment, Univariate};
use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_frontend::{Circuit, Witness};
use arrabbiata_poly::{Domain, UniPoly};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, absorb_slice,
    typed::{
        Challenge, End, Label, Labeled, Message, ProofOf, Protocol, Public, prover_tape,
        verifier_tape,
    },
};

/// The commitment type of a scheme.
type Com<F, PC> = <PC as PolynomialCommitment<F>>::Commitment;

/// The keys produced by [`setup`].
pub type Keys<F, PC> = (ProverKey<F, Com<F, PC>>, VerifierKey<F, Com<F, PC>>);

/// Commitments to the wire polynomials `a, b, c`.
pub type Wires<C> = Labeled<Triple<C>, 0x7769>;
/// Commitment to the grand product `z`.
pub type GrandProduct<C> = Labeled<C, 0x7a7a>;
/// Commitment to the quotient `t`.
pub type Quotient<C> = Labeled<C, 0x7474>;

/// Three values of the same type, absorbed in order.
///
/// ```
/// use arrabbiata_plonk::Triple;
/// assert_eq!(Triple([1, 2, 3]).0[2], 3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Triple<C>(pub [C; 3]);

impl<F, C> Absorb<F> for Triple<C>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        self.0[0].absorb_into(transcript);
        self.0[1].absorb_into(transcript);
        self.0[2].absorb_into(transcript);
    }
}

/// The public inputs of a statement.
///
/// ```
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_plonk::PublicInputs;
/// assert_eq!(PublicInputs(vec![F::new(1)]).0.len(), 1);
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

/// The claimed evaluations at `zeta` (and `z` at `zeta * w`).
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_plonk::Evaluations;
/// let e = Evaluations { values: [F::ZERO; 13], z_shifted: F::ONE };
/// assert_eq!(e.z_shifted, F::ONE);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evaluations<F> {
    /// `a, b, c, q_L, q_R, q_O, q_M, q_C, S_1, S_2, S_3, z, t` at `zeta`,
    /// in the order of [`OPENED`].
    pub values: [F; 13],
    /// `z(zeta * w)`.
    pub z_shifted: F,
}

impl<F> Label for Evaluations<F> {
    const LABEL: u64 = 0x6576;
}

impl<F> Absorb<F> for Evaluations<F>
where
    F: Field,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        let mut i = 0;
        while i < 13 {
            transcript.absorb(self.values[i]);
            i += 1;
        }
        transcript.absorb(self.z_shifted);
    }
}

/// The names of the polynomials opened at `zeta`, in order.
///
/// ```
/// assert_eq!(arrabbiata_plonk::OPENED[11], "z");
/// ```
pub const OPENED: [&str; 13] = [
    "a", "b", "c", "q_L", "q_R", "q_O", "q_M", "q_C", "S_1", "S_2", "S_3", "z", "t",
];

/// The opening proofs of the evaluations.
///
/// ```
/// use arrabbiata_commit::NoProof;
/// use arrabbiata_plonk::Openings;
/// let o = Openings { at_zeta: [NoProof; 13], at_zeta_shifted: NoProof };
/// assert_eq!(o.at_zeta_shifted, NoProof);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Openings<P> {
    /// Proofs at `zeta`, in the order of [`OPENED`].
    pub at_zeta: [P; 13],
    /// Proof of `z` at `zeta * w`.
    pub at_zeta_shifted: P,
}

impl<P> Label for Openings<P> {
    const LABEL: u64 = 0x6f70;
}

impl<F, P> Absorb<F> for Openings<P>
where
    F: Field,
    P: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        let mut i = 0;
        while i < 13 {
            self.at_zeta[i].absorb_into(transcript);
            i += 1;
        }
        self.at_zeta_shifted.absorb_into(transcript);
    }
}

/// The verifier's view of a circuit: the domain, the coset shifts of the
/// permutation argument, and the commitments to the selectors and to the
/// permutation polynomials.
///
/// ```
/// use arrabbiata_commit::{RevealCommitment, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_frontend::build;
/// let (circuit, _, ()) = build::<F, _, _>(|b| {
///     b.witness(F::new(1));
/// });
/// let (_, vk) = arrabbiata_plonk::setup(&circuit, &RevealCommitment::<Univariate>::new()).unwrap();
/// assert_eq!(vk.num_rows(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKey<F, C> {
    log_size: u32,
    shifts: [F; 3],
    selectors: [C; 5],
    sigmas: [C; 3],
    public_rows: Vec<usize>,
}

impl<F, C> VerifierKey<F, C> {
    /// The number of rows of the circuit.
    ///
    /// ```
    /// use arrabbiata_commit::{RevealCommitment, Univariate};
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_frontend::build;
    /// let (c, _, ()) = build::<F, _, _>(|_| ());
    /// let (_, vk) = arrabbiata_plonk::setup(&c, &RevealCommitment::<Univariate>::new()).unwrap();
    /// assert_eq!(vk.num_rows(), c.num_rows());
    /// ```
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        1 << self.log_size
    }
}

impl<F, C> Label for VerifierKey<F, C> {
    const LABEL: u64 = 0x766b;
}

impl<F, C> Absorb<F> for VerifierKey<F, C>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_length(transcript, self.log_size as usize);
        let mut i = 0;
        while i < 3 {
            transcript.absorb(self.shifts[i]);
            i += 1;
        }
        let mut s = 0;
        while s < 5 {
            self.selectors[s].absorb_into(transcript);
            s += 1;
        }
        let mut k = 0;
        while k < 3 {
            self.sigmas[k].absorb_into(transcript);
            k += 1;
        }
        absorb_length(transcript, self.public_rows.len());
        let mut r = 0;
        while r < self.public_rows.len() {
            absorb_length(transcript, self.public_rows[r]);
            r += 1;
        }
    }
}

/// The prover's view of a circuit: the verifier key and the polynomials it
/// commits to.
///
/// ```
/// use arrabbiata_commit::{RevealCommitment, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_frontend::build;
/// let (circuit, _, ()) = build::<F, _, _>(|_| ());
/// let (pk, vk) = arrabbiata_plonk::setup(&circuit, &RevealCommitment::<Univariate>::new()).unwrap();
/// assert_eq!(pk.verifier_key(), &vk);
/// ```
#[derive(Clone, Debug)]
pub struct ProverKey<F, C> {
    vk: VerifierKey<F, C>,
    domain: Domain<F>,
    selectors: [UniPoly<F>; 5],
    sigmas: [UniPoly<F>; 3],
    sigma_values: [Vec<F>; 3],
}

impl<F, C> ProverKey<F, C> {
    /// The verifier key.
    ///
    /// ```
    /// use arrabbiata_commit::{RevealCommitment, Univariate};
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_frontend::build;
    /// let (c, _, ()) = build::<F, _, _>(|_| ());
    /// let (pk, _) = arrabbiata_plonk::setup(&c, &RevealCommitment::<Univariate>::new()).unwrap();
    /// assert_eq!(pk.verifier_key().num_rows(), 2);
    /// ```
    pub const fn verifier_key(&self) -> &VerifierKey<F, C> {
        &self.vk
    }
}

/// The transcript of vanilla PlonK. The prover and the verifier both follow
/// it, through [`Plonk`].
pub type PlonkTranscript<F, C, P> = Public<
    VerifierKey<F, C>,
    Public<
        PublicInputs<F>,
        Message<
            Wires<C>,
            Challenge<
                Challenge<
                    Message<
                        GrandProduct<C>,
                        Challenge<
                            Message<
                                Quotient<C>,
                                Challenge<Message<Evaluations<F>, Message<Openings<P>, End>>>,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >,
>;

/// Vanilla PlonK over `F`, compiled with the univariate scheme `PC`.
///
/// ```
/// use arrabbiata_commit::{RevealCommitment, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_plonk::Plonk;
/// use arrabbiata_transcript::typed::{Pattern, Protocol};
/// type P = <Plonk<F, RevealCommitment<Univariate>> as Protocol>::Transcript;
/// assert_eq!(<P as Pattern>::MESSAGES, 5);
/// assert_eq!(<P as Pattern>::CHALLENGES, 4);
/// ```
#[derive(Debug)]
pub struct Plonk<F, PC>(PhantomData<(F, PC)>);

impl<F, PC> Protocol for Plonk<F, PC>
where
    F: Field,
    PC: PolynomialCommitment<F, Family = Univariate>,
{
    type Transcript = PlonkTranscript<F, PC::Commitment, PC::Proof>;
}

/// Why setup, proving or verification failed.
///
/// ```
/// use arrabbiata_plonk::PlonkError;
/// assert_eq!(PlonkError::IdentityFailed.to_string(), "the PlonK identity does not hold at zeta");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlonkError {
    /// The field has no domain of the circuit's size.
    DomainTooLarge,
    /// The witness does not satisfy the circuit (the quotient does not
    /// exist).
    Unsatisfied,
    /// The number of public inputs does not match the circuit.
    PublicInputs,
    /// An opening proof does not verify; the index is into [`OPENED`], or 13
    /// for `z` at `zeta * w`.
    Opening(usize),
    /// The PlonK identity does not hold at `zeta`.
    IdentityFailed,
}

impl fmt::Display for PlonkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainTooLarge => write!(f, "the field has no domain of the circuit size"),
            Self::Unsatisfied => write!(f, "the witness does not satisfy the circuit"),
            Self::PublicInputs => write!(f, "wrong number of public inputs"),
            Self::Opening(i) => write!(f, "opening proof {i} does not verify"),
            Self::IdentityFailed => write!(f, "the PlonK identity does not hold at zeta"),
        }
    }
}

impl core::error::Error for PlonkError {}

/// `constant` as a degree-0 polynomial.
fn constant<F>(c: F) -> UniPoly<F>
where
    F: Field,
{
    UniPoly::new(vec![c])
}

/// Preprocess a circuit: interpolate and commit to the selectors and to the
/// permutation.
///
/// # Errors
///
/// [`PlonkError::DomainTooLarge`] if the field has no domain of the circuit
/// size.
///
/// ```
/// use arrabbiata_commit::{RevealCommitment, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_frontend::build;
/// let (c, _, ()) = build::<F, _, _>(|b| {
///     b.constant(F::new(2));
/// });
/// assert!(arrabbiata_plonk::setup(&c, &RevealCommitment::<Univariate>::new()).is_ok());
/// ```
pub fn setup<F, PC>(circuit: &Circuit<F>, pcs: &PC) -> Result<Keys<F, PC>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
{
    let n = circuit.num_rows();
    let domain = Domain::<F>::containing(n).ok_or(PlonkError::DomainTooLarge)?;
    // Coset shifts 1, g, g^2 for a generator g of F*: the cosets H, gH, g^2 H
    // are disjoint because g, g^2 and g^-1 are not in H (g has order p - 1).
    let g = F::MULTIPLICATIVE_GENERATOR;
    let shifts = [F::ONE, g, g * g];

    let gates = circuit.gates();
    let mut columns = [
        vec![F::ZERO; n],
        vec![F::ZERO; n],
        vec![F::ZERO; n],
        vec![F::ZERO; n],
        vec![F::ZERO; n],
    ];
    let mut row = 0;
    while row < n {
        columns[0][row] = gates[row].q_l;
        columns[1][row] = gates[row].q_r;
        columns[2][row] = gates[row].q_o;
        columns[3][row] = gates[row].q_m;
        columns[4][row] = gates[row].q_c;
        row += 1;
    }
    let selectors = columns.map(|column| {
        domain
            .interpolate(&column)
            .ok_or(PlonkError::DomainTooLarge)
    });
    let [s0, s1, s2, s3, s4] = selectors;
    let selectors = [s0?, s1?, s2?, s3?, s4?];

    // sigma_k(i) is the identity label of the cell sigma maps (k, i) to:
    // shift_(column) * w^(row).
    let sigma = circuit.sigma();
    let mut sigma_values = [vec![F::ZERO; n], vec![F::ZERO; n], vec![F::ZERO; n]];
    let mut cell = 0;
    while cell < 3 * n {
        let target = sigma[cell];
        sigma_values[cell / n][cell % n] = shifts[target / n] * domain.element(target % n);
        cell += 1;
    }
    let [p0, p1, p2] = sigma_values.clone().map(|values| {
        domain
            .interpolate(&values)
            .ok_or(PlonkError::DomainTooLarge)
    });
    let sigmas = [p0?, p1?, p2?];

    let vk = VerifierKey {
        log_size: domain.log_size(),
        shifts,
        selectors: selectors.clone().map(|p| pcs.commit(&coefficients(&p))),
        sigmas: sigmas.clone().map(|p| pcs.commit(&coefficients(&p))),
        public_rows: circuit.public_rows().to_vec(),
    };
    let pk = ProverKey {
        vk: vk.clone(),
        domain,
        selectors,
        sigmas,
        sigma_values,
    };
    Ok((pk, vk))
}

/// The coefficients of `p`, constant first.
fn coefficients<F>(p: &UniPoly<F>) -> Vec<F>
where
    F: Field,
{
    (0..p.len()).map(|k| p.coefficient(k)).collect()
}

/// `p(w X)`: the coefficients scaled by powers of `w`.
fn shift<F>(p: &UniPoly<F>, w: F) -> UniPoly<F>
where
    F: Field,
{
    let mut power = F::ONE;
    let mut out = Vec::with_capacity(p.len());
    for k in 0..p.len() {
        out.push(p.coefficient(k) * power);
        power = power * w;
    }
    UniPoly::new(out)
}

/// Prove that `witness` satisfies the circuit of `pk`.
///
/// # Errors
///
/// [`PlonkError::Unsatisfied`] if it does not (the quotient polynomial does
/// not exist), and [`PlonkError::PublicInputs`] if the public inputs do not
/// match the circuit.
///
/// ```
/// use arrabbiata_commit::{RevealCommitment, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_frontend::build;
/// use arrabbiata_plonk::{PlonkError, prove, setup};
/// use arrabbiata_transcript::toy_sponge;
///
/// // A false statement: 2 * 2 = 5.
/// let (circuit, witness, ()) = build::<F, _, _>(|b| {
///     let x = b.witness(F::new(2));
///     let y = b.mul(&x, &x);
///     let five = b.public_input(F::new(5));
///     b.assert_equal(&y, &five);
/// });
/// let pcs = RevealCommitment::<Univariate>::new();
/// let (pk, _) = setup(&circuit, &pcs).unwrap();
/// assert_eq!(prove(&pk, &pcs, toy_sponge(0), &witness).unwrap_err(), PlonkError::Unsatisfied);
/// ```
pub fn prove<F, PC, T>(
    pk: &ProverKey<F, Com<F, PC>>,
    pcs: &PC,
    transcript: T,
    witness: &Witness<F>,
) -> Result<ProofOf<Plonk<F, PC>>, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    T: Transcript<F>,
{
    let domain = &pk.domain;
    let n = domain.size();
    if witness.public_inputs().len() != pk.vk.public_rows.len() {
        return Err(PlonkError::PublicInputs);
    }
    let wires = witness.wires();

    // Round 1: wires.
    let [a, b, c] = [&wires[0], &wires[1], &wires[2]]
        .map(|column| domain.interpolate(column).ok_or(PlonkError::DomainTooLarge));
    let (a, b, c) = (a?, b?, c?);
    let wire_commitments = Triple([&a, &b, &c].map(|p| pcs.commit(&coefficients(p))));
    let tape = prover_tape::<F, T, Plonk<F, PC>>(transcript)
        .public(&pk.vk)
        .public(&PublicInputs(witness.public_inputs().to_vec()))
        .send(Labeled(wire_commitments));
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();

    // Round 2: grand product z(w^0) = 1, z(w^(i+1)) = z(w^i) * num_i / den_i.
    let mut z_values = vec![F::ONE; n];
    for i in 0..n - 1 {
        let x = domain.element(i);
        let mut num = F::ONE;
        let mut den = F::ONE;
        for ((column, shift), sigma) in wires.iter().zip(pk.vk.shifts).zip(&pk.sigma_values) {
            num = num * (column[i] + beta * shift * x + gamma);
            den = den * (column[i] + beta * sigma[i] + gamma);
        }
        // den is zero only with probability about 3n / |F| over gamma.
        z_values[i + 1] = z_values[i] * num * den.inverse().ok_or(PlonkError::Unsatisfied)?;
    }
    let z = domain
        .interpolate(&z_values)
        .ok_or(PlonkError::DomainTooLarge)?;
    let tape = tape.send(Labeled(pcs.commit(&coefficients(&z))));
    let (alpha, tape) = tape.challenge();

    // Round 3: quotient.
    let pi = domain
        .interpolate(&circuit_public_column(&pk.vk, witness.public_inputs(), n))
        .ok_or(PlonkError::DomainTooLarge)?;
    let [q_l, q_r, q_o, q_m, q_c] = &pk.selectors;
    let gate =
        &(&(&(&(q_l * &a) + &(q_r * &b)) + &(q_o * &c)) + &(&(q_m * &a) * &b)) + &(q_c + &pi);
    let x = UniPoly::new(vec![F::ZERO, F::ONE]);
    let term = |w: &UniPoly<F>, s: &UniPoly<F>| &(w + &s.scale(beta)) + &constant(gamma);
    let identity = |k: usize| x.scale(pk.vk.shifts[k]);
    let perm_left =
        &(&(&term(&a, &identity(0)) * &term(&b, &identity(1))) * &term(&c, &identity(2))) * &z;
    let z_shifted = shift(&z, domain.generator());
    let perm_right = &(&(&term(&a, &pk.sigmas[0]) * &term(&b, &pk.sigmas[1]))
        * &term(&c, &pk.sigmas[2]))
        * &z_shifted;
    let mut l1_values = vec![F::ZERO; n];
    l1_values[0] = F::ONE;
    let l1 = domain
        .interpolate(&l1_values)
        .ok_or(PlonkError::DomainTooLarge)?;
    let boundary = &(&z - &constant(F::ONE)) * &l1;
    let numerator =
        &(&gate + &(&perm_left - &perm_right).scale(alpha)) + &boundary.scale(alpha * alpha);
    let t = numerator
        .divide_by_vanishing(n)
        .ok_or(PlonkError::Unsatisfied)?;
    let tape = tape.send(Labeled(pcs.commit(&coefficients(&t))));
    let (zeta, tape) = tape.challenge();

    // Round 4: evaluations and openings.
    let opened: [&UniPoly<F>; 13] = [
        &a,
        &b,
        &c,
        q_l,
        q_r,
        q_o,
        q_m,
        q_c,
        &pk.sigmas[0],
        &pk.sigmas[1],
        &pk.sigmas[2],
        &z,
        &t,
    ];
    let openings = opened.map(|p| pcs.open(&coefficients(p), &zeta));
    let (z_shifted_value, z_shifted_proof) =
        pcs.open(&coefficients(&z), &(zeta * domain.generator()));
    let evaluations = Evaluations {
        values: openings.clone().map(|(v, _)| v),
        z_shifted: z_shifted_value,
    };
    let proofs = Openings {
        at_zeta: openings.map(|(_, p)| p),
        at_zeta_shifted: z_shifted_proof,
    };
    let (_, proof) = tape.send(evaluations).send(proofs).finish();
    Ok(proof)
}

/// The public input column for the verifier key's rows.
fn circuit_public_column<F, C>(vk: &VerifierKey<F, C>, public_inputs: &[F], n: usize) -> Vec<F>
where
    F: Field,
{
    let mut column = vec![F::ZERO; n];
    for (row, value) in vk.public_rows.iter().zip(public_inputs) {
        column[*row] = -*value;
    }
    column
}

/// Verify `proof` for the statement `public_inputs` against `vk`.
///
/// # Errors
///
/// [`PlonkError::PublicInputs`] if the number of public inputs is wrong,
/// [`PlonkError::Opening`] if an opening proof fails, and
/// [`PlonkError::IdentityFailed`] if the PlonK identity does not hold at
/// `zeta`.
///
/// ```
/// use arrabbiata_commit::{RevealCommitment, Univariate};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_frontend::build;
/// use arrabbiata_plonk::{PlonkError, prove, setup, verify};
/// use arrabbiata_transcript::toy_sponge;
///
/// let (circuit, witness, ()) = build::<F, _, _>(|b| {
///     let x = b.witness(F::new(4));
///     let y = b.mul(&x, &x);
///     let out = b.public_input(F::new(16));
///     b.assert_equal(&y, &out);
/// });
/// let pcs = RevealCommitment::<Univariate>::new();
/// let (pk, vk) = setup(&circuit, &pcs).unwrap();
/// let proof = prove(&pk, &pcs, toy_sponge(0), &witness).unwrap();
/// // The same proof does not verify for another statement.
/// assert!(verify(&vk, &pcs, toy_sponge(0), &[F::new(17)], proof.clone()).is_err());
/// assert!(verify(&vk, &pcs, toy_sponge(0), &[F::new(16)], proof).is_ok());
/// ```
pub fn verify<F, PC, T>(
    vk: &VerifierKey<F, Com<F, PC>>,
    pcs: &PC,
    transcript: T,
    public_inputs: &[F],
    proof: ProofOf<Plonk<F, PC>>,
) -> Result<(), PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate>,
    T: Transcript<F>,
{
    if public_inputs.len() != vk.public_rows.len() {
        return Err(PlonkError::PublicInputs);
    }
    let domain = Domain::<F>::new(vk.log_size).ok_or(PlonkError::DomainTooLarge)?;
    let (Labeled(wires), tape) = verifier_tape::<F, T, Plonk<F, PC>>(transcript, proof)
        .public(vk)
        .public(&PublicInputs(public_inputs.to_vec()))
        .read();
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();
    let (Labeled(z_commitment), tape) = tape.read();
    let (alpha, tape) = tape.challenge();
    let (Labeled(t_commitment), tape) = tape.read();
    let (zeta, tape) = tape.challenge();
    let (evaluations, tape) = tape.read();
    let (openings, tape) = tape.read();
    let _ = tape.finish();

    // Openings.
    let commitments: [&Com<F, PC>; 13] = [
        &wires.0[0],
        &wires.0[1],
        &wires.0[2],
        &vk.selectors[0],
        &vk.selectors[1],
        &vk.selectors[2],
        &vk.selectors[3],
        &vk.selectors[4],
        &vk.sigmas[0],
        &vk.sigmas[1],
        &vk.sigmas[2],
        &z_commitment,
        &t_commitment,
    ];
    for (i, ((commitment, value), proof)) in commitments
        .iter()
        .zip(evaluations.values)
        .zip(&openings.at_zeta)
        .enumerate()
    {
        if !pcs.verify(commitment, &zeta, value, proof) {
            return Err(PlonkError::Opening(i));
        }
    }
    let zeta_w = zeta * domain.generator();
    if !pcs.verify(
        &z_commitment,
        &zeta_w,
        evaluations.z_shifted,
        &openings.at_zeta_shifted,
    ) {
        return Err(PlonkError::Opening(13));
    }

    // The identity at zeta.
    let [a, b, c, q_l, q_r, q_o, q_m, q_c, s1, s2, s3, z, t] = evaluations.values;
    let mut pi = F::ZERO;
    for (row, value) in vk.public_rows.iter().zip(public_inputs) {
        pi = pi - *value * domain.lagrange_at(*row, zeta);
    }
    let gate = q_l * a + q_r * b + q_o * c + q_m * a * b + q_c + pi;
    let [k0, k1, k2] = vk.shifts;
    let perm = (a + beta * k0 * zeta + gamma)
        * (b + beta * k1 * zeta + gamma)
        * (c + beta * k2 * zeta + gamma)
        * z
        - (a + beta * s1 + gamma)
            * (b + beta * s2 + gamma)
            * (c + beta * s3 + gamma)
            * evaluations.z_shifted;
    let boundary = (z - F::ONE) * domain.first_lagrange_at(zeta);
    let lhs = gate + alpha * perm + alpha * alpha * boundary;
    if lhs == t * domain.vanishing_at(zeta) {
        Ok(())
    } else {
        Err(PlonkError::IdentityFailed)
    }
}
