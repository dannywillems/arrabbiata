//! A generalized-PlonK decider for accumulated PlonKish circuits.
//!
//! A *decider* for an accumulation scheme is an argument of knowledge for
//! its relaxed relation `R'`: a reduction `R' -> R_top` in the category of
//! reductions ([`arrabbiata_reduction`]). It depends on `R'` only, not on
//! how an accumulator was reached, which is why it lives apart from the
//! accumulation scheme.
//!
//! # The relaxed relation is a generalized circuit
//!
//! [`Decider`] proves the relaxed relation of `arrabbiata_plonkish_folding`
//! with generalized PlonK ([`arrabbiata_generalized_plonk`]). The relaxation
//! of a circuit `C` of degree `D` is the generalized circuit
//! [`relaxation`]`(C)`:
//!
//! - the witness columns of `C`, then the error column `E`;
//! - the fixed and selector columns of `C`, and one more selector, `idle`,
//!   equal to `1 - sum_g q_g`: on a row turning no gadget on it is 1;
//! - the public inputs of `C`, bound to their rows as in `C`, then the
//!   slack `u` and the coefficients `c_0, ..., c_(m-1)`, which no row
//!   binds: they are scalars the gates read;
//! - for each gadget `g`, the gate `sum_k c_k f^D_(g,k)(u, w) - E`, turned
//!   on by `q_g`, where `f^D` homogenizes the constraint `f` with `u`; and
//!   the gate `E`, turned on by `idle`;
//! - the copy permutation of `C`, on the columns of `C` only: the error
//!   column carries no copy, so it is not permuted.
//!
//! Since the selectors of `C` are one-hot, the selectors of the relaxation
//! form a partition of unity, and a witness `(w, E)` satisfies it for the
//! public inputs `(x, u, c)` exactly when it satisfies the relaxation of
//! `C` ([`arrabbiata_plonkish::relaxed`]). The circuit does not depend on
//! the accumulator: `u` and `c` are inputs, so it is keyed once.
//!
//! The gates have degree `D + 1` in the folded variables `(u, c, w)` and,
//! with the fixed columns that multiply them, at most `D + 1 + e` in all
//! their variables, with `e` the largest number of fixed columns in a
//! monomial of a constraint. Only the columns count for the size of the
//! quotient: `u` and `c` are scalars.
//!
//! # The link to the accumulator
//!
//! The scheme commits to a column as the polynomial interpolating it, the
//! commitment PlonK makes to its witness columns. The verifier checks that
//! the proof's witness commitments are the accumulator's, then the proof.
//! The link is the law that a scheme's two `commit` agree on the same
//! coefficients ([`Shared`]), tested for IPA and KZG.
//!
//! Composed with the embedding `R -> R'` of the accumulation scheme, the
//! decider is an argument for the circuit relation itself ([`snark`]).
//!
//! Complete runs: `tests/examples.rs` (incremental verification, a SNARK by
//! composition).

#![no_std]

extern crate alloc;

use alloc::{vec, vec::Vec};
use core::fmt;

use arrabbiata_commit::{BatchOpening, HomomorphicCommitment, Univariate};
use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_generalized_plonk::{
    self as generalized, Circuit as GeneralizedCircuit, Gate, GeneralizedTranscript, Proof,
    ProverKey, Variable, VerifierKey,
    plonkish::{columns_read, variable},
};
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_plonkish::{
    Circuit, RelaxedWitness,
    gate::{GateSet, combiner_length},
};
use arrabbiata_plonkish_folding::{Embed, Params, Relaxed, RelaxedInstance};
use arrabbiata_poly::{Expr, Rotation};
use arrabbiata_reduction::{Output, OutputOf, Reduction, Relation, Then, Top};
use arrabbiata_transcript::{
    Transcript,
    typed::{Label, Tail, Tape},
};

/// A commitment scheme that commits to vectors homomorphically and to
/// polynomials with batched evaluation proofs, with one commitment type:
/// what the fold and the decider share. The law that the two `commit` agree on the
/// same coefficients is what links an accumulator to its decider; it is a
/// property of the scheme, tested for IPA and KZG.
pub trait Shared<F>:
    HomomorphicCommitment<F>
    + BatchOpening<
        F,
        Family = Univariate,
        Commitment = <Self as HomomorphicCommitment<F>>::Commitment,
    > + Label
where
    F: TwoAdicField,
{
}

impl<F, PC> Shared<F> for PC
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>
        + BatchOpening<
            F,
            Family = Univariate,
            Commitment = <PC as HomomorphicCommitment<F>>::Commitment,
        > + Label,
{
}

/// Why building, proving or verifying the decider failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `N2` or `D2` is too small for the relaxation of the circuit.
    Size,
    /// Generalized PlonK failed.
    Plonk(generalized::Error),
    /// The proof's (or the witness's) commitment `k` (a witness column for
    /// `k < W`, the error column for `k = W`) is not the accumulator's.
    Commitment(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Size => write!(
                f,
                "too few variables or too low a degree for the relaxation"
            ),
            Self::Plonk(e) => write!(f, "generalized PlonK: {e}"),
            Self::Commitment(k) => write!(f, "the proof's commitment {k} is not the accumulator's"),
        }
    }
}

impl core::error::Error for Error {}

impl From<generalized::Error> for Error {
    fn from(e: generalized::Error) -> Self {
        Self::Plonk(e)
    }
}

/// The homogenization of `expr` to degree `degree` with the slack variable
/// `u`, as a polynomial in the variables numbered by `index`: the sum of
/// `u^(degree - j)` times the component of degree `j` in the witness
/// columns (fixed columns are coefficients, of degree 0). `None` if
/// `degree` is below the degree of `expr` in the witness or the result
/// does not fit `N, D`.
fn homogenize<F, I, const W: usize, const X: usize, const N: usize, const D: usize>(
    expr: &Expr<F, W, X>,
    index: &I,
    u: &Sparse<F, N, D>,
    degree: usize,
) -> Option<Sparse<F, N, D>>
where
    F: Field,
    I: Fn(Variable) -> Option<usize>,
{
    /// The polynomial of `e`, homogeneous of the returned degree in
    /// `(u, witness)`.
    fn go<F, I, const W: usize, const X: usize, const N: usize, const D: usize>(
        e: &Expr<F, W, X>,
        index: &I,
        u: &Sparse<F, N, D>,
    ) -> Option<(Sparse<F, N, D>, usize)>
    where
        F: Field,
        I: Fn(Variable) -> Option<usize>,
    {
        let pad = |p: Sparse<F, N, D>, by: usize| -> Option<Sparse<F, N, D>> {
            let mut p = p;
            for _ in 0..by {
                p = p.checked_mul(u)?;
            }
            Some(p)
        };
        match e {
            Expr::Constant(c) => Some((Sparse::constant(*c), 0)),
            Expr::Variable(v) => {
                let x = Sparse::variable(Var::try_new(index(variable(v))?)?)?;
                let d = usize::from(matches!(v, arrabbiata_poly::Variable::Witness(_, _)));
                Some((x, d))
            }
            Expr::Neg(a) => {
                let (p, d) = go(a, index, u)?;
                Some((-p, d))
            }
            Expr::Add(a, b) => {
                let (p, da) = go(a, index, u)?;
                let (q, db) = go(b, index, u)?;
                let d = da.max(db);
                Some((pad(p, d - da)? + pad(q, d - db)?, d))
            }
            Expr::Mul(a, b) => {
                let (p, da) = go(a, index, u)?;
                let (q, db) = go(b, index, u)?;
                Some((p.checked_mul(&q)?, da + db))
            }
        }
    }
    let (p, d) = go(expr, index, u)?;
    if d > degree {
        return None;
    }
    let mut p = p;
    for _ in d..degree {
        p = p.checked_mul(u)?;
    }
    Some(p)
}

/// The relaxation of degree `degree` of a PlonKish circuit, as a
/// generalized circuit with gates of degree at most `D2` in `N2`
/// variables: see the [crate documentation](crate).
///
/// Its public inputs are those of `circuit`, then `u`, then the
/// [combiner length](combiner_length) coefficients; its witness columns
/// those of `circuit`, then the error column.
///
/// # Errors
///
/// [`Error::Size`] if `N2` or `D2` is too small, or `degree` is below the
/// degree of the gate set.
pub fn relaxation<
    F,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N2: usize,
    const D2: usize,
>(
    circuit: &Circuit<F, G, W, X, Q>,
    degree: usize,
) -> Result<GeneralizedCircuit<F, N2, D2>, Error>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let n = circuit.num_rows();
    let public = circuit.public_rows().len();
    let m = combiner_length::<F, G, W, X>();
    let error = Variable::Witness {
        column: W,
        rotation: Rotation::CURR,
    };
    let mut variables = columns_read::<F, G, W, X>();
    variables.push(error);
    variables.push(Variable::Public(public));
    for k in 0..m {
        variables.push(Variable::Public(public + 1 + k));
    }
    if variables.len() > N2 {
        return Err(Error::Size);
    }
    let index = |v: Variable| variables.iter().position(|w| *w == v);
    let var = |v: Variable| -> Result<Sparse<F, N2, D2>, Error> {
        Sparse::variable(Var::try_new(index(v).ok_or(Error::Size)?).ok_or(Error::Size)?)
            .ok_or(Error::Size)
    };
    let u = var(Variable::Public(public))?;
    let e = var(error)?;
    let mut gates = Vec::with_capacity(Q + 1);
    for g in 0..Q {
        let mut p = -e.clone();
        for (k, c) in G::constraints(g).iter().enumerate() {
            let h = homogenize(c, &index, &u, degree).ok_or(Error::Size)?;
            let ck = var(Variable::Public(public + 1 + k))?;
            p = p + ck.checked_mul(&h).ok_or(Error::Size)?;
        }
        let selector = Some(g);
        gates.push(Gate::new(selector, p));
    }
    // The idle selector is the last one, after the Q of the circuit.
    let idle_selector = Some(Q);
    gates.push(Gate::new(idle_selector, e));
    let mut selectors = circuit.selector_columns().to_vec();
    let mut idle = vec![F::ONE; n];
    for column in &selectors {
        for (i, q) in column.iter().enumerate() {
            idle[i] = idle[i] - *q;
        }
    }
    selectors.push(idle);
    // Built with no copies, then given the circuit's copies on its own
    // columns: the error column carries none, so it is not permuted.
    let sigma = (0..(W + 1) * n).collect();
    let num_rows = n;
    // The columns of the circuit, then the error.
    let witness_columns = W + 1;
    let fixed = circuit.fixed().to_vec();
    // The public inputs of the circuit, then u and the m coefficients.
    let public_inputs = public + 1 + m;
    let bindings = circuit.public_rows().iter().copied().enumerate().collect();
    GeneralizedCircuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        sigma,
        public_inputs,
        bindings,
    )
    .and_then(|relaxed| {
        let permuted = (0..W).collect();
        let copies = circuit.sigma().to_vec();
        relaxed.with_permutation(permuted, copies)
    })
    .map_err(Error::Plonk)
}

/// The decider: generalized PlonK on the relaxation of the circuit of
/// `params`, keyed once.
///
/// # Type parameters
///
/// Those of [`Params`], then:
///
/// - `N2`: the number of variables of the relaxation's gates, at least the
///   number of columns the gate set reads (at their rotations), plus 2 (the
///   error and `u`), plus the combiner length;
/// - `D2`: their degree, at least `D + 1` plus the largest number of fixed
///   columns in a monomial of a constraint.
///
/// As a reduction it goes from the relaxed relation of `params` to
/// [`Top`].
///
/// Complete runs: `tests/examples.rs` (incremental verification, a SNARK by
/// composition).
pub struct Decider<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
    const N2: usize,
    const D2: usize,
> where
    F: TwoAdicField + PrimeField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    params: &'a Params<F, PC, G, W, X, Q, N, D>,
    pk: ProverKey<F, PC, N2, D2>,
    vk: VerifierKey<F, PC, N2, D2>,
}

impl<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
    const N2: usize,
    const D2: usize,
> Decider<'a, F, PC, G, W, X, Q, N, D, N2, D2>
where
    F: TwoAdicField + PrimeField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    /// The decider for the scheme with parameters `params`: the relaxation
    /// of its circuit, set up with the scheme's key.
    ///
    /// # Errors
    ///
    /// [`Error::Size`] if `N2` or `D2` is too small, and the errors of
    /// [`arrabbiata_generalized_plonk::setup`].
    pub fn new(params: &'a Params<F, PC, G, W, X, Q, N, D>) -> Result<Self, Error> {
        let circuit = relaxation::<F, G, W, X, Q, N2, D2>(params.circuit(), D)?;
        let (pk, vk) = generalized::setup(&circuit, params.ck())?;
        Ok(Self { params, pk, vk })
    }

    /// The verifier key of the decider.
    pub const fn verifier_key(&self) -> &VerifierKey<F, PC, N2, D2> {
        &self.vk
    }

    /// The relaxation of the circuit, as a generalized circuit.
    pub const fn circuit(&self) -> &GeneralizedCircuit<F, N2, D2> {
        self.pk.circuit()
    }

    /// The public inputs of the relaxation for an accumulator: its public
    /// inputs, its slack, its coefficients.
    fn public(
        instance: &RelaxedInstance<F, <PC as HomomorphicCommitment<F>>::Commitment, W>,
    ) -> Vec<F> {
        let mut public = instance.public().to_vec();
        public.push(instance.u());
        public.extend_from_slice(instance.coefficients());
        public
    }

    /// The accumulator's commitments, the error last.
    fn commitments(
        instance: &RelaxedInstance<F, <PC as HomomorphicCommitment<F>>::Commitment, W>,
    ) -> Vec<<PC as HomomorphicCommitment<F>>::Commitment> {
        let mut c = instance.columns().to_vec();
        c.push(instance.error().clone());
        c
    }
}

impl<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
    const N2: usize,
    const D2: usize,
> Reduction<F> for Decider<'a, F, PC, G, W, X, Q, N, D, N2, D2>
where
    F: TwoAdicField + PrimeField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    type Source = Relaxed<'a, F, PC, G, W, X, Q, N, D>;
    type Target = Top;
    type Proof = Proof<F, PC, N2, D2>;
    type Error = Error;
    type Pattern = GeneralizedTranscript<F, PC, N2, D2>;

    fn source(&self) -> Self::Source {
        self.params.relaxed()
    }

    fn target(&self) -> Top {
        Top
    }

    fn prove<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        instance: &<Self::Source as Relation>::Instance,
        witness: &RelaxedWitness<F, W>,
    ) -> Result<OutputOf<F, Self, Tape<F, T, R>>, Error>
    where
        T: Transcript<F>,
    {
        let mut columns = witness.columns().to_vec();
        columns.push(witness.error().to_vec());
        for (k, (column, c)) in columns.iter().zip(Self::commitments(instance)).enumerate() {
            let opens = self.params.commit_column(column).is_ok_and(|own| own == c);
            if !opens {
                return Err(Error::Commitment(k));
            }
        }
        let (proof, tape) = generalized::prove_tape(
            &self.pk,
            self.params.ck(),
            tape,
            &Self::public(instance),
            &columns,
        )?;
        Ok(Output {
            instance: (),
            witness: (),
            proof,
            transcript: tape,
        })
    }

    fn verify<T, R>(
        &self,
        tape: Tape<F, T, Tail<Self::Pattern, R>>,
        instance: &<Self::Source as Relation>::Instance,
        proof: Self::Proof,
    ) -> Result<((), Tape<F, T, R>), Error>
    where
        T: Transcript<F>,
    {
        let (wires, tape) = generalized::verify_tape(
            &self.vk,
            self.params.ck(),
            tape,
            &Self::public(instance),
            proof,
        )?;
        for (k, (ours, theirs)) in wires.iter().zip(Self::commitments(instance)).enumerate() {
            if *ours != theirs {
                return Err(Error::Commitment(k));
            }
        }
        Ok(((), tape))
    }
}

/// A SNARK for the circuit relation: the decider after the embedding,
/// `Dec . Embed : R -> R_top`.
pub type Snark<
    'a,
    'd,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
    const N2: usize,
    const D2: usize,
> = Then<Embed<'a, F, PC, G, W, X, Q, N, D>, &'d Decider<'a, F, PC, G, W, X, Q, N, D, N2, D2>>;

/// The SNARK `Dec . Embed` for the circuit of `decider`: an argument of
/// knowledge for the circuit relation, obtained from the category by
/// composition, with no protocol of its own.
///
/// Complete runs: `tests/examples.rs` (incremental verification, a SNARK by
/// composition).
pub const fn snark<
    'a,
    'd,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
    const N2: usize,
    const D2: usize,
>(
    decider: &'d Decider<'a, F, PC, G, W, X, Q, N, D, N2, D2>,
) -> Snark<'a, 'd, F, PC, G, W, X, Q, N, D, N2, D2>
where
    F: TwoAdicField + PrimeField,
    PC: Shared<F>,
    G: GateSet<F, W, X>,
{
    Then(Embed(decider.params), decider)
}
