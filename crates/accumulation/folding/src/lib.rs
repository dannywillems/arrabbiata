//! Accumulation of homogeneous polynomial relations of arbitrary degree.
//!
//! # The scheme
//!
//! Let the relaxed relation of a [`ConstraintSystem`] have degree `d` (see
//! [`ConstraintSystem::DEGREE`]): on every row `i`,
//! `P(u, c, w_i) = sum_j c_j f_j^D(u, w_i) = E_i`, with `P` homogeneous of
//! degree `d` in the instance-and-witness variables `(u, c, w)`. Given an
//! accumulator `(u1, c1, w1, E1)` and a fresh (or another accumulated)
//! instance `(u2, c2, w2, E2)`, the line through them is a polynomial in a
//! scalar `r`:
//!
//! ```text
//! P(u1 + r u2, c1 + r c2, w1 + r w2) = E1 + r T_1 + ... + r^(d-1) T_(d-1) + r^d E2.
//! ```
//!
//! Homogeneity is what pins the two ends: the coefficient of `r^0` is
//! `P(u1, c1, w1) = E1` and the coefficient of `r^d` is
//! `P(u2, c2, w2) = E2`. The prover commits to the cross-terms `T_k`, the
//! verifier draws `r` after them, and both parties set
//!
//! ```text
//! u = u1 + r u2,   c = c1 + r c2,   [w] = [w1] + r [w2],
//! [E] = [E1] + sum_k r^k [T_k] + r^d [E2],
//! ```
//!
//! the verifier on commitments (using that commitment is a module morphism),
//! the prover on the vectors themselves. The identity above says the folded
//! pair satisfies the relaxed relation whenever both inputs do
//! (completeness), for every `r`.
//!
//! This is the generalization of Nova described in the Arrabbiata note:
//! Nova is the special case `d = 2` with one constraint and no combiner
//! ([`Nova`]); the note's combination of constraints by folded powers of a
//! challenge is [`arrabbiata_relation::aggregation::Powers`].
//!
//! # Types
//!
//! [`Fold`] is the protocol, as a type. Its transcript is declared once, as
//! [`FoldingTranscript`], and both [`Fold::prove`] and [`Fold::verify`] run
//! on it, so they cannot disagree on the order of the tape or on what is
//! absorbed. The number of cross-terms `T` is a const parameter checked
//! against the relation's degree at compile time.
//!
//! ```
//! use arrabbiata_commit::IdentityCommitment;
//! use arrabbiata_folding::Fold;
//! use arrabbiata_relation::{aggregation::Powers, examples};
//! use arrabbiata_transcript::toy_sponge;
//! use arrabbiata_field::Goldilocks as F;
//!
//! // Fibonacci: degree 2 constraints, two of them, combined with Powers,
//! // so the relation has degree 3 and a fold has 2 cross-terms.
//! type Scheme = Fold<F, IdentityCommitment, Powers, 2, 1, 2, 2, 2>;
//!
//! let cs = examples::fibonacci(8);
//! let ck = IdentityCommitment;
//! let fresh = |x, y| {
//!     let trace = examples::fibonacci_trace(&cs, x, y);
//!     let instance = cs.commit(&ck, &trace, &mut toy_sponge(1));
//!     cs.relax(&ck, instance, trace)
//! };
//! let (acc, acc_w) = fresh(1, 2);
//! let (new, new_w) = fresh(3, 4);
//!
//! let (folded, folded_w, proof, _) =
//!     Scheme::prove(&cs, &ck, toy_sponge(0), (&acc, &acc_w), (&new, &new_w)).unwrap();
//! let (verified, _) = Scheme::verify(&cs, &ck, toy_sponge(0), &acc, &new, proof);
//!
//! assert_eq!(folded, verified);
//! cs.check_relaxed(&ck, &verified, &folded_w).unwrap();
//! ```

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::{Field, powers};
use arrabbiata_poly::{Assignment, FixedColumn, Polynomials, Rotation, UniPoly, WitnessColumn};
use arrabbiata_relation::{
    ConstraintSystem, RelaxedInstance, RelaxedWitness,
    aggregation::{Aggregation, Single},
    error::ShapeError,
};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_length, tag,
    typed::{
        Challenge, End, Label, Message, ProofOf, Protocol, Public, prover_tape, verifier_tape,
    },
};

/// The commitments to the cross-terms `T_1, ..., T_N` of a fold.
///
/// # Type parameters
///
/// - `C`: the type of a commitment.
/// - `N`: the number of cross-terms, `d - 1` for a relation of degree `d`.
///
/// ```
/// use arrabbiata_folding::CrossTerms;
/// let t = CrossTerms::new([1u8, 2]);
/// assert_eq!(t.commitments(), &[1, 2]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossTerms<C, const N: usize>([C; N]);

impl<C, const N: usize> CrossTerms<C, N> {
    /// The cross-term commitments, `T_1` first.
    ///
    /// ```
    /// use arrabbiata_folding::CrossTerms;
    /// assert_eq!(CrossTerms::new(["T1"]).commitments()[0], "T1");
    /// ```
    pub const fn new(commitments: [C; N]) -> Self {
        Self(commitments)
    }

    /// The cross-term commitments, `T_1` first.
    ///
    /// ```
    /// use arrabbiata_folding::CrossTerms;
    /// assert_eq!(CrossTerms::new([0u8; 3]).commitments().len(), 3);
    /// ```
    pub const fn commitments(&self) -> &[C; N] {
        &self.0
    }
}

impl<C, const N: usize> Label for CrossTerms<C, N> {
    const LABEL: u64 = tag::FOLDING_PROOF;
}

impl<F, C, const N: usize> Absorb<F> for CrossTerms<C, N>
where
    F: Field,
    C: Absorb<F>,
{
    fn absorb_into<T>(&self, transcript: &mut T)
    where
        T: Transcript<F>,
    {
        transcript.absorb(F::from_u64(tag::FOLDING_PROOF));
        absorb_length(transcript, N);
        let mut k = 0;
        while k < N {
            self.0[k].absorb_into(transcript);
            k += 1;
        }
    }
}

/// The relaxed instance type of the scheme.
///
/// # Type parameters
///
/// - `F`: the field the relation is over (the scalar field of the commitment
///   scheme).
/// - `PC`: the homomorphic vector commitment scheme; `PC::Commitment` is the
///   type of the column, error and cross-term commitments.
/// - `A`: the aggregation strategy combining the `M` constraints into one
///   equation per row ([`Single`] or
///   [`Powers`](arrabbiata_relation::aggregation::Powers)).
/// - `W`: the number of witness columns (committed and folded).
/// - `M`: the number of constraints.
pub type Accumulator<F, PC, A, const W: usize, const M: usize> = RelaxedInstance<
    F,
    <PC as HomomorphicCommitment<F>>::Commitment,
    <A as Aggregation<F, M>>::Coefficients,
    W,
>;

/// The transcript of one fold: the circuit, the two instances, the
/// cross-terms, then the folding challenge.
///
/// # Type parameters
///
/// - `F`: the field the relation is over (the scalar field of the commitment
///   scheme).
/// - `PC`: the homomorphic vector commitment scheme; `PC::Commitment` is the
///   type of the column, error and cross-term commitments.
/// - `A`: the aggregation strategy combining the `M` constraints into one
///   equation per row ([`Single`] or
///   [`Powers`](arrabbiata_relation::aggregation::Powers)).
/// - `W`: the number of witness columns (committed and folded).
/// - `X`: the number of fixed columns (selectors; shared by both instances,
///   never folded).
/// - `M`: the number of constraints.
/// - `D`: the maximum degree of a constraint in the witness columns; each
///   constraint is homogenized to this degree with `u`.
/// - `T`: the number of cross-terms, `d - 1` for the relation's degree
///   `d = D + A::EXTRA_DEGREE` (checked at compile time where `Fold` is used).
///
/// Declaring the statement (`Public`) before the message is what binds `r`
/// to everything the verifier folds; this is the weak Fiat-Shamir pitfall
/// excluded by the type.
///
/// ```
/// use arrabbiata_commit::{IdentityCommitment, Opened};
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_folding::FoldingTranscript;
/// use arrabbiata_relation::aggregation::Single;
/// use arrabbiata_transcript::typed::Pattern;
///
/// type P = FoldingTranscript<F, IdentityCommitment, Single, 1, 1, 1, 2, 1>;
/// assert_eq!(<P as Pattern>::MESSAGES, 1);
/// assert_eq!(<P as Pattern>::CHALLENGES, 1);
/// ```
pub type FoldingTranscript<
    F,
    PC,
    A,
    const W: usize,
    const X: usize,
    const M: usize,
    const D: usize,
    const T: usize,
> = Public<
    ConstraintSystem<F, A, W, X, M, D>,
    Public<
        Accumulator<F, PC, A, W, M>,
        Public<
            Accumulator<F, PC, A, W, M>,
            Message<CrossTerms<<PC as HomomorphicCommitment<F>>::Commitment, T>, Challenge<End>>,
        >,
    >,
>;

/// The folding protocol for circuits with `W` witness and `X` fixed columns
/// and `M` constraints of degree at most `D`, aggregated by `A`, committed
/// with `PC`; a fold produces `T` cross-terms, and `T` must equal
/// `D + A::EXTRA_DEGREE - 1` (checked at compile time).
///
/// # Type parameters
///
/// - `F`: the field the relation is over (the scalar field of the commitment
///   scheme).
/// - `PC`: the homomorphic vector commitment scheme; `PC::Commitment` is the
///   type of the column, error and cross-term commitments.
/// - `A`: the aggregation strategy combining the `M` constraints into one
///   equation per row ([`Single`] or
///   [`Powers`](arrabbiata_relation::aggregation::Powers)).
/// - `W`: the number of witness columns (committed and folded).
/// - `X`: the number of fixed columns (selectors; shared by both instances,
///   never folded).
/// - `M`: the number of constraints.
/// - `D`: the maximum degree of a constraint in the witness columns; each
///   constraint is homogenized to this degree with `u`.
/// - `T`: the number of cross-terms, `d - 1` for the relation's degree
///   `d = D + A::EXTRA_DEGREE` (checked at compile time where `Fold` is used).
///
/// ```
/// use arrabbiata_commit::IdentityCommitment;
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_folding::Fold;
/// use arrabbiata_relation::{aggregation::Powers, examples};
/// use arrabbiata_transcript::toy_sponge;
///
/// type Scheme = Fold<F, IdentityCommitment, Powers, 2, 1, 2, 2, 2>;
/// let cs = examples::fibonacci(4);
/// let ck = IdentityCommitment;
/// let trace = examples::fibonacci_trace(&cs, 1, 1);
/// let (i, w) = cs.relax(&ck, cs.commit(&ck, &trace, &mut toy_sponge(0)), trace);
/// let (folded, _, proof, _) = Scheme::prove(&cs, &ck, toy_sponge(0), (&i, &w), (&i, &w)).unwrap();
/// assert_eq!(Scheme::verify(&cs, &ck, toy_sponge(0), &i, &i, proof).0, folded);
/// ```
///
/// A wrong number of cross-terms is a compile-time error. Here a degree-2
/// relation with no combiner (Nova) is declared with two cross-terms:
///
/// ```compile_fail
/// use arrabbiata_commit::IdentityCommitment;
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_folding::Fold;
/// use arrabbiata_relation::{aggregation::Single, examples};
/// use arrabbiata_transcript::toy_sponge;
///
/// type Wrong = Fold<F, IdentityCommitment, Single, 1, 1, 1, 2, 2>;
/// let cs = examples::squaring(4);
/// let ck = IdentityCommitment;
/// let trace = examples::squaring_trace(&cs, 2);
/// let (i, w) = cs.relax(&ck, cs.commit(&ck, &trace, &mut toy_sponge(0)), trace);
/// let _ = Wrong::prove(&cs, &ck, toy_sponge(0), (&i, &w), (&i, &w));
/// ```
#[derive(Debug)]
pub struct Fold<
    F,
    PC,
    A,
    const W: usize,
    const X: usize,
    const M: usize,
    const D: usize,
    const T: usize,
>(PhantomData<(F, PC, A)>);

/// Nova: degree 2, one constraint, no combiner, one cross-term.
///
/// # Type parameters
///
/// - `F`: the field the relation is over (the scalar field of the commitment
///   scheme).
/// - `PC`: the homomorphic vector commitment scheme; `PC::Commitment` is the
///   type of the column, error and cross-term commitments.
/// - `W`: the number of witness columns (committed and folded).
/// - `X`: the number of fixed columns (selectors; shared by both instances,
///   never folded).
///
/// With the row-local constraint `a * b - c`, the relaxed relation is
/// `a * b - u * c = E`, Nova's relaxed R1CS, and the single cross-term is
/// Nova's `T = a1 b2 + a2 b1 - u1 c2 - u2 c1`.
///
/// ```
/// use arrabbiata_commit::IdentityCommitment;
/// use arrabbiata_field::Goldilocks as F;
/// use arrabbiata_folding::Nova;
/// use arrabbiata_relation::examples;
/// use arrabbiata_transcript::toy_sponge;
///
/// let cs = examples::squaring(4);
/// let ck = IdentityCommitment;
/// let relaxed = |x| {
///     let trace = examples::squaring_trace(&cs, x);
///     cs.relax(&ck, cs.commit(&ck, &trace, &mut toy_sponge(0)), trace)
/// };
/// let (a, aw) = relaxed(2);
/// let (b, bw) = relaxed(3);
/// let (_, w, _, _) =
///     Nova::<F, IdentityCommitment, 1, 1>::prove(&cs, &ck, toy_sponge(0), (&a, &aw), (&b, &bw)).unwrap();
/// // The folded error is no longer zero: the accumulator is relaxed.
/// assert!(w.error().iter().any(|e| *e != F::new(0)));
/// ```
pub type Nova<F, PC, const W: usize, const X: usize> = Fold<F, PC, Single, W, X, 1, 2, 1>;

impl<F, PC, A, const W: usize, const X: usize, const M: usize, const D: usize, const T: usize>
    Protocol for Fold<F, PC, A, W, X, M, D, T>
where
    F: Field,
    PC: HomomorphicCommitment<F>,
    A: Aggregation<F, M>,
{
    type Transcript = FoldingTranscript<F, PC, A, W, X, M, D, T>;
}

/// Two traces on the line `acc + r * fresh`, as an assignment in `F[r]`.
struct LineView<'a, F, const W: usize, const X: usize> {
    acc: &'a [Vec<F>; W],
    fresh: &'a [Vec<F>; W],
    fixed: &'a [Vec<F>; X],
    row: usize,
    num_rows: usize,
}

impl<F, const W: usize, const X: usize> Assignment<F, Polynomials<F>, W, X>
    for LineView<'_, F, W, X>
where
    F: Field,
{
    fn witness(
        &self,
        _: &Polynomials<F>,
        column: WitnessColumn<W>,
        rotation: Rotation,
    ) -> UniPoly<F> {
        let row = rotation.apply(self.row, self.num_rows);
        UniPoly::line(
            self.acc[column.index()][row],
            self.fresh[column.index()][row],
        )
    }

    fn fixed(&self, column: FixedColumn<X>, rotation: Rotation) -> F {
        self.fixed[column.index()][rotation.apply(self.row, self.num_rows)]
    }
}

/// A vector of exactly `N` elements as an array.
fn array_from<C, const N: usize>(values: Vec<C>) -> [C; N] {
    match <[C; N]>::try_from(values) {
        Ok(array) => array,
        Err(_) => unreachable!("the vector has exactly N elements"),
    }
}

impl<F, PC, A, const W: usize, const X: usize, const M: usize, const D: usize, const T: usize>
    Fold<F, PC, A, W, X, M, D, T>
where
    F: Field,
    PC: HomomorphicCommitment<F>,
    A: Aggregation<F, M>,
{
    const DEGREE_MATCHES: () = assert!(
        T + 1 == D + A::EXTRA_DEGREE,
        "the number of cross-terms must be the degree of the relation minus one"
    );

    /// The cross-term vectors `T_1, ..., T_T`, one value per row each.
    fn cross_terms(
        cs: &ConstraintSystem<F, A, W, X, M, D>,
        acc: (&Accumulator<F, PC, A, W, M>, &RelaxedWitness<F, W>),
        fresh: (&Accumulator<F, PC, A, W, M>, &RelaxedWitness<F, W>),
    ) -> [Vec<F>; T] {
        let ring = Polynomials::new();
        let u = UniPoly::line(acc.0.u(), fresh.0.u());
        let coefficients = match A::coefficient(acc.0.coefficients(), 0) {
            None => None,
            Some(_) => {
                let mut lines = Vec::with_capacity(M);
                let mut j = 0;
                while j < M {
                    let c1 = A::coefficient(acc.0.coefficients(), j).unwrap_or(F::ONE);
                    let c2 = A::coefficient(fresh.0.coefficients(), j).unwrap_or(F::ONE);
                    lines.push(UniPoly::line(c1, c2));
                    j += 1;
                }
                Some(lines)
            }
        };
        let coefficients_ref = coefficients.as_deref();
        let mut terms = Vec::with_capacity(T);
        let mut k = 0;
        while k < T {
            terms.push(Vec::with_capacity(cs.num_rows()));
            k += 1;
        }
        let mut row = 0;
        while row < cs.num_rows() {
            let view = LineView {
                acc: acc.1.columns(),
                fresh: fresh.1.columns(),
                fixed: cs.fixed_columns(),
                row,
                num_rows: cs.num_rows(),
            };
            let g = cs.evaluate_row(&ring, &view, &u, coefficients_ref);
            let mut k = 0;
            while k < T {
                terms[k].push(g.coefficient(k + 1));
                k += 1;
            }
            row += 1;
        }
        array_from(terms)
    }

    /// The folded instance, computed from the two instances, the cross-term
    /// commitments and the challenge. Shared by prover and verifier, so the
    /// prover's instance is the verifier's by construction.
    fn fold_instances(
        ck: &PC,
        acc: &Accumulator<F, PC, A, W, M>,
        fresh: &Accumulator<F, PC, A, W, M>,
        cross_terms: &CrossTerms<PC::Commitment, T>,
        r: F,
    ) -> Accumulator<F, PC, A, W, M> {
        let mut columns = acc.columns().clone();
        let mut column = 0;
        while column < W {
            columns[column] = ck.add(
                &acc.columns()[column],
                &ck.scale(&fresh.columns()[column], r),
            );
            column += 1;
        }
        // [E] = [E1] + sum_k r^k [T_k] + r^(T+1) [E2]
        let r_powers = powers(r, T + 2);
        let mut error = acc.error().clone();
        let mut k = 0;
        while k < T {
            error = ck.add(
                &error,
                &ck.scale(&cross_terms.commitments()[k], r_powers[k + 1]),
            );
            k += 1;
        }
        error = ck.add(&error, &ck.scale(fresh.error(), r_powers[T + 1]));
        RelaxedInstance::new(
            acc.u() + r * fresh.u(),
            A::fold(acc.coefficients(), fresh.coefficients(), r),
            columns,
            error,
        )
    }

    /// Fold `fresh` into `acc`.
    ///
    /// Returns the folded instance and witness, the proof (the cross-term
    /// commitments, as the [`ProofOf`] the verifier expects), and the
    /// transcript, so that a caller can continue the protocol on it.
    ///
    /// # Errors
    ///
    /// A [`ShapeError`] if a witness does not have the shape of the circuit.
    ///
    /// ```
    /// use arrabbiata_commit::IdentityCommitment;
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_folding::Nova;
    /// use arrabbiata_relation::examples;
    /// use arrabbiata_transcript::toy_sponge;
    ///
    /// let cs = examples::squaring(4);
    /// let ck = IdentityCommitment;
    /// let trace = examples::squaring_trace(&cs, 5);
    /// let (inst, wit) = cs.relax(&ck, cs.commit(&ck, &trace, &mut toy_sponge(0)), trace);
    /// let (folded, folded_w, _proof, _sponge) =
    ///     Nova::<F, _, 1, 1>::prove(&cs, &ck, toy_sponge(0), (&inst, &wit), (&inst, &wit)).unwrap();
    /// cs.check_relaxed(&ck, &folded, &folded_w).unwrap();
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn prove<Tr>(
        cs: &ConstraintSystem<F, A, W, X, M, D>,
        ck: &PC,
        transcript: Tr,
        acc: (&Accumulator<F, PC, A, W, M>, &RelaxedWitness<F, W>),
        fresh: (&Accumulator<F, PC, A, W, M>, &RelaxedWitness<F, W>),
    ) -> Result<
        (
            Accumulator<F, PC, A, W, M>,
            RelaxedWitness<F, W>,
            ProofOf<Self>,
            Tr,
        ),
        ShapeError,
    >
    where
        Tr: Transcript<F>,
    {
        let () = Self::DEGREE_MATCHES;
        check_shape(cs, acc.1)?;
        check_shape(cs, fresh.1)?;

        let terms = Self::cross_terms(cs, acc, fresh);
        let mut commitments = Vec::with_capacity(T);
        let mut k = 0;
        while k < T {
            commitments.push(ck.commit(&terms[k]));
            k += 1;
        }
        let cross_terms = CrossTerms::new(array_from(commitments));

        let (r, tape) = prover_tape::<F, Tr, Self>(transcript)
            .public(cs)
            .public(acc.0)
            .public(fresh.0)
            .send(cross_terms.clone())
            .challenge();
        let (transcript, proof) = tape.finish();

        let instance = Self::fold_instances(ck, acc.0, fresh.0, &cross_terms, r);
        let witness = fold_witness::<F, W, T>(acc.1, fresh.1, &terms, r);
        Ok((instance, witness, proof, transcript))
    }

    /// The verifier's fold: the folded instance, computed from the two
    /// instances and the proof alone.
    ///
    /// The verifier performs no check: soundness is that the folded instance
    /// is satisfiable only if both inputs are (with overwhelming probability
    /// over `r`), so the check is deferred to whoever decides the final
    /// accumulator.
    ///
    /// ```
    /// use arrabbiata_commit::IdentityCommitment;
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_folding::Nova;
    /// use arrabbiata_relation::examples;
    /// use arrabbiata_transcript::toy_sponge;
    ///
    /// let cs = examples::squaring(4);
    /// let ck = IdentityCommitment;
    /// let trace = examples::squaring_trace(&cs, 5);
    /// let (inst, wit) = cs.relax(&ck, cs.commit(&ck, &trace, &mut toy_sponge(0)), trace);
    /// let (p_inst, _, proof, _) =
    ///     Nova::<F, _, 1, 1>::prove(&cs, &ck, toy_sponge(0), (&inst, &wit), (&inst, &wit)).unwrap();
    /// let (v_inst, _) = Nova::<F, _, 1, 1>::verify(&cs, &ck, toy_sponge(0), &inst, &inst, proof);
    /// assert_eq!(p_inst, v_inst);
    /// ```
    pub fn verify<Tr>(
        cs: &ConstraintSystem<F, A, W, X, M, D>,
        ck: &PC,
        transcript: Tr,
        acc: &Accumulator<F, PC, A, W, M>,
        fresh: &Accumulator<F, PC, A, W, M>,
        proof: ProofOf<Self>,
    ) -> (Accumulator<F, PC, A, W, M>, Tr)
    where
        Tr: Transcript<F>,
    {
        let () = Self::DEGREE_MATCHES;
        let (cross_terms, tape) = verifier_tape::<F, Tr, Self>(transcript, proof)
            .public(cs)
            .public(acc)
            .public(fresh)
            .read();
        let (r, tape) = tape.challenge();
        let transcript = tape.finish();
        (
            Self::fold_instances(ck, acc, fresh, &cross_terms, r),
            transcript,
        )
    }
}

fn check_shape<F, A, const W: usize, const X: usize, const M: usize, const D: usize>(
    cs: &ConstraintSystem<F, A, W, X, M, D>,
    witness: &RelaxedWitness<F, W>,
) -> Result<(), ShapeError>
where
    F: Field,
    A: Aggregation<F, M>,
{
    let mut column = 0;
    while column < W {
        if witness.columns()[column].len() != cs.num_rows() {
            return Err(ShapeError::WitnessColumnLength {
                column,
                expected: cs.num_rows(),
                found: witness.columns()[column].len(),
            });
        }
        column += 1;
    }
    if witness.error().len() != cs.num_rows() {
        return Err(ShapeError::ErrorLength {
            expected: cs.num_rows(),
            found: witness.error().len(),
        });
    }
    Ok(())
}

/// `w = w1 + r w2`, `E = E1 + sum_k r^k T_k + r^(T+1) E2`.
fn fold_witness<F, const W: usize, const T: usize>(
    acc: &RelaxedWitness<F, W>,
    fresh: &RelaxedWitness<F, W>,
    cross_terms: &[Vec<F>; T],
    r: F,
) -> RelaxedWitness<F, W>
where
    F: Field,
{
    let mut columns = acc.columns().clone();
    let mut column = 0;
    while column < W {
        let mut row = 0;
        while row < columns[column].len() {
            columns[column][row] = columns[column][row] + r * fresh.columns()[column][row];
            row += 1;
        }
        column += 1;
    }
    let r_powers = powers(r, T + 2);
    let mut error = acc.error().to_vec();
    let mut row = 0;
    while row < error.len() {
        let mut k = 0;
        while k < T {
            error[row] = error[row] + r_powers[k + 1] * cross_terms[k][row];
            k += 1;
        }
        error[row] = error[row] + r_powers[T + 1] * fresh.error()[row];
        row += 1;
    }
    RelaxedWitness::new(columns, error)
}
