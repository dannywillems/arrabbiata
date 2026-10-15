//! The parameters of the scheme, and the two relations it relates.

use alloc::vec::Vec;

use arrabbiata_commit::HomomorphicCommitment;
use arrabbiata_field::TwoAdicField;
use arrabbiata_mvpoly::{Layout, Sparse, from_expr};
use arrabbiata_plonkish::{
    Circuit, RelaxedWitness, Witness,
    gate::{self, GateSet},
};
use arrabbiata_poly::{Domain, FixedColumn, Rotation};
use arrabbiata_reduction::Relation;
use arrabbiata_transcript::{Absorb, Transcript};

use crate::{Digest, Error, RelaxedInstance};

/// The parameters of the scheme for one circuit: the circuit, the
/// commitment key, the domain of the circuit, the layout of the
/// constraints' variables, and a digest of the circuit for the transcript.
///
/// # Type parameters
///
/// - `F`: the field of the circuit, the scalar field of the commitments.
/// - `PC`: the homomorphic commitment scheme.
/// - `G`, `W`, `X`, `Q`: the gate set, the numbers of witness columns,
///   fixed columns and selectors of the circuit.
/// - `N`: the number of variables of a constraint polynomial, at least `W`
///   times the number of rotations the gate set reads.
/// - `D`: the degree of the relaxation, at least the degree of the gate set
///   in the witness; a fold has `D` cross-terms.
pub struct Params<
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
> where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    circuit: Circuit<F, G, W, X, Q>,
    ck: PC,
    domain: Domain<F>,
    layout: Layout<W, N>,
    /// The constraint polynomials of each gadget that reads no fixed
    /// column, the same on every row.
    uniform: Vec<Option<Vec<Sparse<F, N, D>>>>,
    combiner: usize,
    digest: Digest<F>,
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Params<F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    /// The parameters for `circuit` with the key `ck`. The digest of the
    /// circuit is squeezed from `transcript` after absorbing the circuit:
    /// prover and verifier must use the same one, and the fold's
    /// transcript then absorbs a single element instead of the circuit.
    ///
    /// # Errors
    ///
    /// [`Error::Degree`] if `D` is below the degree of the gate set,
    /// [`Error::Layout`] if `N` cannot hold the columns at their rotations,
    /// [`Error::Domain`] if the field has no domain of the circuit size.
    pub fn new<T>(circuit: Circuit<F, G, W, X, Q>, ck: PC, mut transcript: T) -> Result<Self, Error>
    where
        T: Transcript<F>,
    {
        if gate::degree::<F, G, W, X>() > D {
            return Err(Error::Degree);
        }
        let layout =
            Layout::<W, N>::new(&gate::witness_rotations::<F, G, W, X>()).ok_or(Error::Layout)?;
        let domain = Domain::<F>::containing(circuit.num_rows()).ok_or(Error::Domain)?;
        if domain.size() != circuit.num_rows() {
            return Err(Error::Domain);
        }
        let mut uniform = Vec::with_capacity(Q);
        for selector in 0..Q {
            if gate::reads_fixed::<F, G, W, X>(selector) {
                uniform.push(None);
            } else {
                let zero = |_, _| F::ZERO;
                let mut polys = Vec::new();
                for c in G::constraints(selector) {
                    polys.push(from_expr(&c, &layout, &zero).ok_or(Error::Degree)?);
                }
                uniform.push(Some(polys));
            }
        }
        circuit.absorb_into(&mut transcript);
        let digest = Digest::new(transcript.squeeze());
        Ok(Self {
            circuit,
            ck,
            domain,
            layout,
            uniform,
            combiner: gate::combiner_length::<F, G, W, X>(),
            digest,
        })
    }

    /// The circuit.
    pub const fn circuit(&self) -> &Circuit<F, G, W, X, Q> {
        &self.circuit
    }

    /// The commitment key.
    pub const fn ck(&self) -> &PC {
        &self.ck
    }

    /// The digest of the circuit, absorbed by every transcript of the
    /// scheme.
    pub const fn digest(&self) -> &Digest<F> {
        &self.digest
    }

    /// The number of coefficients combining the constraints of a row: the
    /// largest number of constraints of a gadget.
    pub const fn combiner_length(&self) -> usize {
        self.combiner
    }

    /// The relation `R` of the circuit.
    pub const fn satisfiability(&self) -> Satisfiability<'_, F, PC, G, W, X, Q, N, D> {
        Satisfiability(self)
    }

    /// The relaxed relation `R'` of the circuit.
    pub const fn relaxed(&self) -> Relaxed<'_, F, PC, G, W, X, Q, N, D> {
        Relaxed(self)
    }

    /// The commitment to a column of `n` values: the commitment to the
    /// polynomial of degree below `n` interpolating it on the domain of the
    /// circuit. It is linear in the column.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if the column does not have `n` values.
    pub fn commit_column(&self, values: &[F]) -> Result<PC::Commitment, Error> {
        let p = self.domain.interpolate(values).ok_or(Error::Shape)?;
        Ok(self.ck.commit(p.coefficients()))
    }

    /// The zero of the module of commitments, the commitment to a zero
    /// column, computed as `0 * c` for a given commitment `c` rather than
    /// by committing to `n` zeros.
    pub(crate) fn zero_like(&self, c: &PC::Commitment) -> PC::Commitment {
        self.ck.scale(c, F::ZERO)
    }

    /// The point of `columns` on `row` under the layout: one value per
    /// column and rotation.
    pub(crate) fn point(&self, columns: &[Vec<F>; W], row: usize) -> [F; N] {
        self.layout.point(columns, row)
    }

    /// The constraint polynomials of the gadget `selector` on `row`.
    pub(crate) fn polynomials(
        &self,
        selector: usize,
        row: usize,
    ) -> Result<Vec<Sparse<F, N, D>>, Error> {
        if let Some(Some(polys)) = self.uniform.get(selector) {
            return Ok(polys.clone());
        }
        let n = self.circuit.num_rows();
        let fixed = self.circuit.fixed();
        let at = |c: FixedColumn<X>, r: Rotation| fixed[c.index()][r.apply(row, n)];
        let mut polys = Vec::new();
        for c in G::constraints(selector) {
            polys.push(from_expr(&c, &self.layout, &at).ok_or(Error::Degree)?);
        }
        Ok(polys)
    }
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    core::fmt::Debug for Params<F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Params")
            .field("num_rows", &self.circuit.num_rows())
            .field("degree", &D)
            .field("variables", &N)
            .field("combiner", &self.combiner)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

/// The relation `R` of a circuit: a witness satisfies it, with the
/// instance as its public inputs.
pub struct Satisfiability<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(pub &'a Params<F, PC, G, W, X, Q, N, D>)
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>;

/// The relaxed relation `R'` of a circuit: the instance's commitments open
/// to the witness, which satisfies the relaxation of degree `D` for the
/// instance's slack, coefficients and public inputs.
///
/// Complete runs: `tests/examples.rs` (an incremental computation, two
/// accumulators folded); laws: `tests/accumulation.rs`.
pub struct Relaxed<
    'a,
    F,
    PC,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(pub &'a Params<F, PC, G, W, X, Q, N, D>)
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>;

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize> Clone
    for Satisfiability<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize> Copy
    for Satisfiability<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize> Clone
    for Relaxed<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize> Copy
    for Relaxed<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Relation for Satisfiability<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    type Instance = Vec<F>;
    type Witness = Witness<F, W>;
    type Error = Error;

    fn check(&self, public: &Vec<F>, witness: &Witness<F, W>) -> Result<(), Error> {
        if witness.public_inputs() != public.as_slice() {
            return Err(Error::PublicInputs);
        }
        if self.0.circuit.is_satisfied(witness) {
            Ok(())
        } else {
            Err(Error::Unsatisfied)
        }
    }
}

impl<F, PC, G, const W: usize, const X: usize, const Q: usize, const N: usize, const D: usize>
    Relation for Relaxed<'_, F, PC, G, W, X, Q, N, D>
where
    F: TwoAdicField,
    PC: HomomorphicCommitment<F>,
    G: GateSet<F, W, X>,
{
    type Instance = RelaxedInstance<F, PC::Commitment, W>;
    type Witness = RelaxedWitness<F, W>;
    type Error = Error;

    fn check(
        &self,
        instance: &RelaxedInstance<F, PC::Commitment, W>,
        witness: &RelaxedWitness<F, W>,
    ) -> Result<(), Error> {
        let p = self.0;
        if instance.coefficients().len() != p.combiner
            || instance.public().len() != p.circuit.public_rows().len()
        {
            return Err(Error::Shape);
        }
        for (k, (column, commitment)) in
            witness.columns().iter().zip(instance.columns()).enumerate()
        {
            if p.commit_column(column)? != *commitment {
                return Err(Error::Commitment(k));
            }
        }
        if p.commit_column(witness.error())? != *instance.error() {
            return Err(Error::Commitment(W));
        }
        if p.circuit.is_relaxed_satisfied(
            witness,
            instance.public(),
            instance.u(),
            instance.coefficients(),
            D,
        ) {
            Ok(())
        } else {
            Err(Error::Unsatisfied)
        }
    }
}
