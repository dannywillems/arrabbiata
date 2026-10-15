//! PlonK for the relaxation of a circuit: the SNARK a decider runs on an
//! accumulator.
//!
//! The statement is a [`Relaxation`]: the degree `D`, the slack `u`, the
//! coefficients `c`, the public inputs, and commitments to the witness
//! columns and to the error column. It is proved for a
//! [relaxed witness](arrabbiata_plonkish::RelaxedWitness) satisfying the
//! relaxation of the circuit ([`arrabbiata_plonkish::relaxed`]).
//!
//! The protocol is PlonK with two changes:
//!
//! - the witness and error commitments are part of the statement, not
//!   messages: the verifier already holds them (an accumulator carries
//!   them), and the prover opens them. A column must therefore be committed
//!   as `PC::commit(interpolate(column))`, the commitment PlonK makes to its
//!   polynomial, which is also linear in the column (a Lagrange-basis
//!   commitment, the one the accumulation scheme folds);
//! - the gate term is the relaxed gate condition
//!   `sum_g q_g sum_k c_k f^D_(g,k)(u, w) - E`, homogenized with the scalar
//!   `u` and combined by the given `c`, the error polynomial `E` opened at
//!   `zeta` alongside the others. The constraints of a row are combined by
//!   the statement's coefficients, not by a fresh challenge: that is the
//!   relaxation. `alpha` still separates the gate term from the
//!   public-input, permutation and boundary terms, which are unchanged:
//!   they are linear in the witness, so the accumulation scheme preserves
//!   them exactly.
//!
//! The homogenization multiplies by powers of the scalar `u` only, so the
//! quotient has the degree of the strict protocol's.
//!
//! Unlike [`crate::Plonk`], the messages carry the commitment scheme in
//! their type but not the sponge, so that one decider type runs on any
//! transcript, as a reduction of knowledge does.
//!
//! ```
//! use arrabbiata_commit::PolynomialCommitment;
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::{Builder, ipa, relaxed::{self, Relaxation}, setup, vanilla::{Arithmetic, Vanilla}};
//! use arrabbiata_plonkish::RelaxedWitness;
//! use arrabbiata_poly::Domain;
//! use arrabbiata_poseidon::vesta_sponge;
//!
//! // a b = c with 2 * 3 != 7: unsatisfied, but relaxed with E = -1 at u = 1.
//! let mut b = Builder::<F, Vanilla, 3, 5, 1>::new();
//! let one = F::ONE;
//! b.row::<Arithmetic, _>([F::from_u64(2), F::from_u64(3), F::from_u64(7)], [F::ZERO, F::ZERO, -one, one, F::ZERO]);
//! let (circuit, w) = b.build().unwrap();
//! let witness = RelaxedWitness::new(w.columns().clone(), vec![-one, F::ZERO]);
//! let pcs = ipa::pallas(&circuit);
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let domain = Domain::<F>::containing(circuit.num_rows()).unwrap();
//! let commit = |v: &[F]| pcs.commit(domain.interpolate(v).unwrap().coefficients());
//! let statement = Relaxation::new(
//!     2, one, vec![one], vec![],
//!     std::array::from_fn(|k| commit(&witness.columns()[k])),
//!     commit(witness.error()),
//! );
//! let (proof, _) = relaxed::prove(&pk, &pcs, vesta_sponge(0), &statement, &witness).unwrap();
//! relaxed::verify(&vk, &pcs, vesta_sponge(0), &statement, proof).unwrap();
//! ```

use alloc::vec::Vec;
use core::marker::PhantomData;

use arrabbiata_commit::{PolynomialCommitment, Univariate};
use arrabbiata_field::{Field, TwoAdicField};
use arrabbiata_plonkish::{RelaxedWitness, gate};
use arrabbiata_poly::{Domain, Expr, Scalars, UniPoly, evaluate_homogeneous};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, absorb_slice,
    typed::{
        Challenge, End, Label, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape,
    },
};

use crate::{
    Com, CosetView, GateSet, Opened, OpenedView, PlonkError, Prf, ProverKey, Scheme, VerifierKey,
    check_openings, grand_product, identity_rest, interpolate, interpolate_all, open_all, quotient,
    rotations,
};

/// The statement of relaxed PlonK: the relaxation of degree `degree` with
/// slack `u`, coefficients `coefficients` and public inputs `public`, and
/// the commitments to the `W` witness columns and to the error column.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::relaxed::Relaxation;
/// let s = Relaxation::new(2, F::ONE, vec![F::ONE], vec![], ["w"], "E");
/// assert_eq!((s.degree(), s.u(), s.error()), (2, F::ONE, &"E"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relaxation<F, C, const W: usize> {
    degree: usize,
    u: F,
    coefficients: Vec<F>,
    public: Vec<F>,
    wires: [C; W],
    error: C,
}

impl<F, C, const W: usize> Relaxation<F, C, W>
where
    F: Copy,
{
    /// The statement with the given components.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// let s = Relaxation::new(3, F::ONE, vec![], vec![F::ZERO], [0u8; 2], 1u8);
    /// assert_eq!(s.public(), &[F::ZERO]);
    /// ```
    pub const fn new(
        degree: usize,
        u: F,
        coefficients: Vec<F>,
        public: Vec<F>,
        wires: [C; W],
        error: C,
    ) -> Self {
        Self {
            degree,
            u,
            coefficients,
            public,
            wires,
            error,
        }
    }

    /// The degree of the relaxation.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// assert_eq!(Relaxation::new(5, F::ONE, vec![], vec![], [0u8; 0], 0u8).degree(), 5);
    /// ```
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// The slack.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// assert_eq!(Relaxation::new(1, F::from_u64(3), vec![], vec![], [0u8; 0], 0u8).u(), F::from_u64(3));
    /// ```
    pub const fn u(&self) -> F {
        self.u
    }

    /// The coefficients combining the constraints of a row.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// assert_eq!(Relaxation::new(1, F::ONE, vec![F::ONE], vec![], [0u8; 0], 0u8).coefficients(), &[F::ONE]);
    /// ```
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    /// The public inputs.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// assert!(Relaxation::new(1, F::ONE, vec![], vec![], [0u8; 0], 0u8).public().is_empty());
    /// ```
    pub fn public(&self) -> &[F] {
        &self.public
    }

    /// The commitments to the witness columns.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// assert_eq!(Relaxation::new(1, F::ONE, vec![], vec![], [4u8], 0u8).wires(), &[4]);
    /// ```
    pub const fn wires(&self) -> &[C; W] {
        &self.wires
    }

    /// The commitment to the error column.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::relaxed::Relaxation;
    /// assert_eq!(Relaxation::new(1, F::ONE, vec![], vec![], [0u8; 0], 9u8).error(), &9);
    /// ```
    pub const fn error(&self) -> &C {
        &self.error
    }
}

impl<F, C, const W: usize> Label for Relaxation<F, C, W> {
    const LABEL: u64 = 0x726c_7878;
}

impl<F, C, const W: usize> Absorb<F> for Relaxation<F, C, W>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        absorb_length(transcript, self.degree);
        transcript.absorb(self.u);
        absorb_slice(transcript, &self.coefficients);
        absorb_slice(transcript, &self.public);
        absorb_length(transcript, W);
        for c in &self.wires {
            c.absorb_into(transcript);
        }
        self.error.absorb_into(transcript);
    }
}

/// The openings of relaxed PlonK: those of PlonK, then the error
/// polynomial at `zeta`.
///
/// ```
/// use arrabbiata_plonk::{Opened, relaxed::RelaxedOpened};
/// let opened = Opened::<u8, 1, 0, 0> {
///     witness: [1], fixed: [], selectors: [], sigmas: [2], z: 3, t: 4,
///     z_next: 5, witness_rotated: vec![], fixed_rotated: vec![],
/// };
/// let r = RelaxedOpened { opened, error: 6 };
/// assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedOpened<T, const W: usize, const X: usize, const Q: usize> {
    /// The openings of PlonK.
    pub opened: Opened<T, W, X, Q>,
    /// The error polynomial at `zeta`.
    pub error: T,
}

impl<T, const W: usize, const X: usize, const Q: usize> RelaxedOpened<T, W, X, Q> {
    /// Every value, in the order of the openings: PlonK's, then the error.
    ///
    /// ```
    /// use arrabbiata_plonk::{Opened, relaxed::RelaxedOpened};
    /// let opened = Opened::<u8, 0, 0, 0> {
    ///     witness: [], fixed: [], selectors: [], sigmas: [], z: 0, t: 0,
    ///     z_next: 0, witness_rotated: vec![], fixed_rotated: vec![],
    /// };
    /// assert_eq!(RelaxedOpened { opened, error: 1 }.iter().count(), 4);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.opened.iter().chain(core::iter::once(&self.error))
    }
}

impl<T, const W: usize, const X: usize, const Q: usize> Label for RelaxedOpened<T, W, X, Q> {
    const LABEL: u64 = 0x7265_6576;
}

impl<F, T, const W: usize, const X: usize, const Q: usize> Absorb<F> for RelaxedOpened<T, W, X, Q>
where
    F: Field,
    T: Absorb<F>,
{
    fn absorb_into<Tr>(&self, transcript: &mut Tr)
    where
        Tr: Transcript<F>,
    {
        self.opened.absorb_into(transcript);
        self.error.absorb_into(transcript);
    }
}

/// Commitment to the grand product `z`, from `PC`.
pub type GrandProduct<F, PC> = Scheme<PC, Com<F, PC>, 0x7a7a>;
/// Commitment to the quotient `t`, from `PC`.
pub type Quotient<F, PC> = Scheme<PC, Com<F, PC>, 0x7474>;
/// The opening proofs, from `PC`.
pub type Openings<F, PC, const W: usize, const X: usize, const Q: usize> =
    Scheme<PC, RelaxedOpened<Prf<F, PC>, W, X, Q>, 0x6f70>;

/// The transcript of relaxed PlonK: the verifier key and the statement,
/// then the rounds of PlonK from the copy challenges on.
pub type RelaxedTranscript<F, PC, G, const W: usize, const X: usize, const Q: usize> = Public<
    VerifierKey<F, PC, G, W, X, Q>,
    Public<
        Relaxation<F, Com<F, PC>, W>,
        Challenge<
            Challenge<
                Message<
                    GrandProduct<F, PC>,
                    Challenge<
                        Message<
                            Quotient<F, PC>,
                            Challenge<
                                Message<
                                    RelaxedOpened<F, W, X, Q>,
                                    Message<Openings<F, PC, W, X, Q>, End>,
                                >,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >,
>;

/// The type parameters of [`RelaxedPlonk`], as a marker that owns none of
/// them.
type Parameters<F, PC, G> = fn() -> (F, PC, G);

/// Relaxed PlonK over `F` for the gate set `G`, compiled with `PC`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa::PallasIpa, relaxed::RelaxedPlonk, vanilla::Vanilla};
/// use arrabbiata_transcript::typed::{Pattern, Protocol};
/// type P = <RelaxedPlonk<F, PallasIpa, Vanilla, 3, 5, 1> as Protocol>::Transcript;
/// // z, t, the evaluations and the openings: the wires are in the statement.
/// assert_eq!(<P as Pattern>::MESSAGES, 4);
/// assert_eq!(<P as Pattern>::CHALLENGES, 4);
/// ```
#[derive(Debug)]
pub struct RelaxedPlonk<F, PC, G, const W: usize, const X: usize, const Q: usize>(
    PhantomData<Parameters<F, PC, G>>,
);

impl<F, PC, G, const W: usize, const X: usize, const Q: usize> Protocol
    for RelaxedPlonk<F, PC, G, W, X, Q>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
    G: GateSet<F, W, X>,
{
    type Transcript = RelaxedTranscript<F, PC, G, W, X, Q>;
}

/// The statement's shape against the circuit: a degree at least the gate
/// set's, one coefficient per constraint of the largest gadget, one public
/// input per public row.
fn check_statement<F, C, G, const W: usize, const X: usize, const Q: usize>(
    statement: &Relaxation<F, C, W>,
    public_rows: usize,
) -> Result<(), PlonkError>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    if statement.degree < gate::degree::<F, G, W, X>()
        || statement.coefficients.len() != gate::combiner_length::<F, G, W, X>()
    {
        return Err(PlonkError::Shape);
    }
    if statement.public.len() != public_rows {
        return Err(PlonkError::PublicInputs);
    }
    Ok(())
}

/// The relaxed gate term `sum_g q_g sum_k c_k f^D_(g,k)(u, view) - e` for
/// the selector values `selectors` and the error value `e`.
fn relaxed_gates<F, A, const W: usize, const X: usize>(
    constraints: &[Vec<Expr<F, W, X>>],
    view: &A,
    selectors: &[F],
    error: F,
    (degree, u, coefficients): (usize, F, &[F]),
) -> F
where
    F: Field,
    A: arrabbiata_poly::Assignment<F, Scalars<F>, W, X>,
{
    let ring = Scalars::new();
    let mut acc = F::ZERO;
    for (selector, gadget) in selectors.iter().zip(constraints) {
        let mut combined = F::ZERO;
        for (c, k) in gadget.iter().zip(coefficients) {
            // The degree is checked against the gate set beforehand.
            let h = match evaluate_homogeneous(&ring, c, view, &u, degree) {
                Some(h) => h,
                None => F::ZERO,
            };
            combined = combined + *k * h;
        }
        acc = acc + *selector * combined;
    }
    acc - error
}

/// Prove `statement` for the relaxed witness `witness`.
///
/// Returns the proof and the transcript after the protocol.
///
/// # Errors
///
/// [`PlonkError::Shape`] if the statement does not fit the circuit or the
/// witness has the wrong shape, [`PlonkError::PublicInputs`] for a wrong
/// number of public inputs, [`PlonkError::Unsatisfied`] if the witness
/// does not satisfy the relaxation or does not open the statement's
/// commitments.
///
/// ```
/// use arrabbiata_commit::PolynomialCommitment;
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{Builder, PlonkError, ipa, relaxed::{self, Relaxation}, setup, vanilla::Vanilla};
/// use arrabbiata_plonkish::RelaxedWitness;
/// use arrabbiata_poseidon::vesta_sponge;
/// let (circuit, w) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
/// let pcs = ipa::pallas(&circuit);
/// let (pk, _) = setup(&circuit, &pcs).unwrap();
/// let witness = RelaxedWitness::embed(&w);
/// let zero = pcs.commit(&[]);
/// // The degree of the arithmetic gate is 2.
/// let low = Relaxation::new(1, F::ONE, vec![F::ONE], vec![], std::array::from_fn(|_| zero), zero);
/// assert_eq!(relaxed::prove(&pk, &pcs, vesta_sponge(0), &low, &witness).unwrap_err(), PlonkError::Shape);
/// ```
#[allow(clippy::type_complexity)]
pub fn prove<F, PC, T, G, const W: usize, const X: usize, const Q: usize>(
    pk: &ProverKey<F, PC, G, W, X, Q>,
    pcs: &PC,
    transcript: T,
    statement: &Relaxation<F, Com<F, PC>, W>,
    witness: &RelaxedWitness<F, W>,
) -> Result<(ProofOf<RelaxedPlonk<F, PC, G, W, X, Q>>, T), PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
    T: Transcript<F>,
    G: GateSet<F, W, X>,
{
    let key = &pk.vk.value;
    check_statement::<F, _, G, W, X, Q>(statement, key.public_rows.len())?;
    let n = pk.domain.size();
    if witness.columns().iter().any(|c| c.len() != n) || witness.error().len() != n {
        return Err(PlonkError::Shape);
    }
    if !pk.circuit.is_relaxed_satisfied(
        witness,
        &statement.public,
        statement.u,
        &statement.coefficients,
        statement.degree,
    ) {
        return Err(PlonkError::Unsatisfied);
    }
    let commit = |p: &UniPoly<F>| pcs.commit(p.coefficients());
    let wires = interpolate_all(&pk.domain, witness.columns())?;
    let error = interpolate(&pk.domain, witness.error())?;
    if (0..W).any(|k| commit(&wires[k]) != statement.wires[k]) || commit(&error) != statement.error
    {
        return Err(PlonkError::Unsatisfied);
    }

    let tape = prover_tape::<F, T, RelaxedPlonk<F, PC, G, W, X, Q>>(transcript)
        .public(&pk.vk)
        .public(statement);
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();
    let z = grand_product(pk, witness.columns(), beta, gamma)?;
    let tape = tape.send(Scheme::new(commit(&z)));
    let (alpha, tape) = tape.challenge();

    let constraints: Vec<Vec<Expr<F, W, X>>> = (0..Q).map(|g| G::constraints(g)).collect();
    let relaxation = (
        statement.degree,
        statement.u,
        statement.coefficients.as_slice(),
    );
    let gates = |view: &CosetView<'_, F, W, X>, selectors: &[F], extra: &[F]| -> (F, F) {
        (
            relaxed_gates(&constraints, view, selectors, extra[0], relaxation),
            alpha,
        )
    };
    let t = quotient(
        pk,
        &wires,
        &z,
        &statement.public,
        core::slice::from_ref(&error),
        (beta, gamma, alpha),
        true,
        gates,
    )?;
    let tape = tape.send(Scheme::new(commit(&t)));
    let (zeta, tape) = tape.challenge();

    let opened = open_all(pk, pcs, &wires, &z, &t, zeta);
    let error_opened = pcs.open(error.coefficients(), &zeta);
    let evaluations = RelaxedOpened {
        opened: opened.clone().map(|(v, _)| v),
        error: error_opened.0,
    };
    let proofs = RelaxedOpened {
        opened: opened.map(|(_, p)| p),
        error: error_opened.1,
    };
    let (transcript, proof) = tape.send(evaluations).send(Scheme::new(proofs)).finish();
    Ok((proof, transcript))
}

/// Verify `proof` for `statement` against `vk`; returns the transcript
/// after the protocol.
///
/// # Errors
///
/// [`PlonkError::Shape`] or [`PlonkError::PublicInputs`] if the statement
/// does not fit the circuit, [`PlonkError::Opening`] if an opening fails
/// (the error polynomial's index is the last), and
/// [`PlonkError::IdentityFailed`] if the identity does not hold at `zeta`.
///
/// ```
/// use arrabbiata_commit::PolynomialCommitment;
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{Builder, PlonkError, ipa, relaxed::{self, Relaxation}, setup, vanilla::Vanilla};
/// use arrabbiata_plonkish::RelaxedWitness;
/// use arrabbiata_poseidon::vesta_sponge;
/// let (circuit, w) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
/// let pcs = ipa::pallas(&circuit);
/// let (pk, vk) = setup(&circuit, &pcs).unwrap();
/// let zero = pcs.commit(&[]);
/// let statement = Relaxation::new(2, F::ONE, vec![F::ONE], vec![], std::array::from_fn(|_| zero), zero);
/// let (proof, _) = relaxed::prove(&pk, &pcs, vesta_sponge(0), &statement, &RelaxedWitness::embed(&w)).unwrap();
/// relaxed::verify(&vk, &pcs, vesta_sponge(0), &statement, proof.clone()).unwrap();
/// // Another slack: the transcript, hence every challenge, differs.
/// let other = Relaxation::new(2, F::from_u64(2), vec![F::ONE], vec![], std::array::from_fn(|_| zero), zero);
/// assert!(relaxed::verify(&vk, &pcs, vesta_sponge(0), &other, proof).is_err());
/// ```
pub fn verify<F, PC, T, G, const W: usize, const X: usize, const Q: usize>(
    vk: &VerifierKey<F, PC, G, W, X, Q>,
    pcs: &PC,
    transcript: T,
    statement: &Relaxation<F, Com<F, PC>, W>,
    proof: ProofOf<RelaxedPlonk<F, PC, G, W, X, Q>>,
) -> Result<T, PlonkError>
where
    F: TwoAdicField,
    PC: PolynomialCommitment<F, Family = Univariate> + Label,
    T: Transcript<F>,
    G: GateSet<F, W, X>,
{
    let key = &vk.value;
    check_statement::<F, _, G, W, X, Q>(statement, key.public_rows.len())?;
    let domain = Domain::<F>::new(key.log_size).ok_or(PlonkError::DomainTooLarge)?;
    let tape = verifier_tape::<F, T, RelaxedPlonk<F, PC, G, W, X, Q>>(transcript, proof)
        .public(vk)
        .public(statement);
    let (beta, tape) = tape.challenge();
    let (gamma, tape) = tape.challenge();
    let (z_commitment, tape) = tape.read();
    let (alpha, tape) = tape.challenge();
    let (t_commitment, tape) = tape.read();
    let (zeta, tape) = tape.challenge();
    let (evaluations, tape) = tape.read();
    let (proofs, tape) = tape.read();
    let transcript = tape.finish();
    let proofs = proofs.value;

    check_openings::<F, PC, G, W, X, Q>(
        key,
        pcs,
        &domain,
        (&statement.wires, &z_commitment.value, &t_commitment.value),
        zeta,
        &evaluations.opened,
        &proofs.opened,
    )?;
    if !pcs.verify(&statement.error, &zeta, evaluations.error, &proofs.error) {
        return Err(PlonkError::Opening(evaluations.opened.iter().count()));
    }

    let (rotated_witness, rotated_fixed) = rotations::<F, G, W, X>();
    let view = OpenedView {
        opened: &evaluations.opened,
        rotated_witness: &rotated_witness,
        rotated_fixed: &rotated_fixed,
    };
    let constraints: Vec<Vec<Expr<F, W, X>>> = (0..Q).map(|g| G::constraints(g)).collect();
    let gates = relaxed_gates(
        &constraints,
        &view,
        &evaluations.opened.selectors,
        evaluations.error,
        (statement.degree, statement.u, &statement.coefficients),
    );
    let lhs = gates
        + identity_rest(
            key,
            &domain,
            &evaluations.opened,
            &statement.public,
            (beta, gamma, alpha),
            alpha,
            zeta,
        );
    if lhs == evaluations.opened.t * domain.vanishing_at(zeta) {
        Ok(transcript)
    } else {
        Err(PlonkError::IdentityFailed)
    }
}
