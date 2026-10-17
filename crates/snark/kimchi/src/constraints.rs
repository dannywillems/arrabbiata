//! Kimchi's constraint system (`kimchi/src/circuits/constraints.rs`,
//! `ConstraintSystem` and its `Builder`): a list of gates, padded to a
//! domain, with the zero-knowledge rows, the permutation shifts and the
//! fixed columns the prover and the verifier index are built from.
//!
//! [`Builder::build`] follows Kimchi's steps:
//!
//! 1. the lookup domain: the rows of the fixed tables, of the runtime tables
//!    and of the built-in tables the gates use (the 12-bit range-check table
//!    for RangeCheck0, RangeCheck1, Rot64 and ForeignFieldMul; the 4-bit XOR
//!    table for Xor16), plus a dummy zero row when no table has id 0;
//! 2. the zero-knowledge rows: 3, or with chunks of `max_poly_size` rows the
//!    fixpoint of `zk_rows = (16 c + 5) / 7` and `n = 2^ceil(log2(circuit +
//!    zk_rows))`, with `c = n / max_poly_size` chunks;
//! 3. the domain `d1` of size `n`, and `d2`, `d4`, `d8` (whose generators
//!    square into one another);
//! 4. the padding: Zero gates with identity wiring up to `n` rows;
//! 5. the shifts ([`crate::permutation::shifts`]);
//! 6. the lookup index ([`crate::lookup::LookupIndex`]) when the gates look
//!    something up.
//!
//! The columns are given as evaluations on `d1`
//! ([`ConstraintSystem::sigmas`], [`ConstraintSystem::selector`],
//! [`ConstraintSystem::coefficient`]), and [`ConstraintSystem::extend`]
//! carries one to a larger domain, as Kimchi stores them (over `d8`, and
//! over `d4` for the Generic and CompleteAdd selectors).

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;

use arrabbiata_field::{PrimeField, TwoAdicField};
use arrabbiata_poly::domain::Domain;

use crate::{
    gates::Gates,
    lookup::{LookupError, LookupIndex},
    permutation::{PERMUTS, shifts},
};

/// The number of witness columns.
pub const COLUMNS: usize = 15;

/// The rows of the built-in 12-bit range-check table.
pub const RANGE_CHECK_TABLE_SIZE: usize = 1 << 12;

/// The rows of the built-in 4-bit XOR table.
pub const XOR_TABLE_SIZE: usize = 1 << 8;

/// Kimchi's gate types, in Kimchi's order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateType {
    /// No constraint.
    #[default]
    Zero,
    /// Two PlonK half-gates.
    Generic,
    /// Five Poseidon rounds.
    Poseidon,
    /// Complete affine addition.
    CompleteAdd,
    /// Variable-base scalar multiplication.
    VarBaseMul,
    /// Endomorphism-optimized scalar multiplication.
    EndoMul,
    /// The scalar of an EndoMul.
    EndoMulScalar,
    /// Lookups of the `Lookup` pattern, no constraint of its own.
    Lookup,
    /// Three 88-bit range checks, first row.
    RangeCheck0,
    /// Three 88-bit range checks, last two rows.
    RangeCheck1,
    /// Foreign-field addition.
    ForeignFieldAdd,
    /// Foreign-field multiplication.
    ForeignFieldMul,
    /// 16 bits of a XOR.
    Xor16,
    /// A 64-bit rotation.
    Rot64,
}

/// The built-in table a gate's lookups read, if any (Kimchi's
/// `LookupPattern::from_gate` then `LookupPattern::table`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateTable {
    /// The 12-bit range-check table.
    RangeCheck,
    /// The 4-bit XOR table.
    Xor,
}

impl GateType {
    /// The built-in table this gate's lookups read.
    #[must_use]
    pub const fn table(self) -> Option<GateTable> {
        match self {
            Self::RangeCheck0 | Self::RangeCheck1 | Self::Rot64 | Self::ForeignFieldMul => {
                Some(GateTable::RangeCheck)
            }
            Self::Xor16 => Some(GateTable::Xor),
            _ => None,
        }
    }

    /// Whether Kimchi builds this gate's selector only when the circuit
    /// uses it (a feature flag), rather than always.
    #[must_use]
    pub const fn is_optional(self) -> bool {
        matches!(
            self,
            Self::RangeCheck0
                | Self::RangeCheck1
                | Self::ForeignFieldAdd
                | Self::ForeignFieldMul
                | Self::Xor16
                | Self::Rot64
        )
    }
}

/// A cell of the permuted columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Wire {
    /// The row.
    pub row: usize,
    /// The column, below [`PERMUTS`].
    pub col: usize,
}

impl Wire {
    /// The wires of a row copied to nothing: each cell to itself.
    #[must_use]
    pub fn for_row(row: usize) -> [Self; PERMUTS] {
        core::array::from_fn(|col| Self { row, col })
    }
}

/// A row: its gate, where each permuted cell is wired to (the next cell of
/// its cycle), and its coefficients (missing ones are zero).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitGate<F> {
    /// The gate.
    pub typ: GateType,
    /// For each permuted cell, the next cell of its cycle.
    pub wires: [Wire; PERMUTS],
    /// The coefficients, at most [`COLUMNS`].
    pub coeffs: Vec<F>,
}

impl<F> CircuitGate<F> {
    /// A row.
    #[must_use]
    pub const fn new(typ: GateType, wires: [Wire; PERMUTS], coeffs: Vec<F>) -> Self {
        Self { typ, wires, coeffs }
    }
}

/// Why a row of a witness does not satisfy its gate (Kimchi's
/// `CircuitGateError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitGateError {
    /// Constraint `n` (from 1, in Kimchi's order) of the gate does not
    /// hold.
    Constraint(GateType, usize),
    /// The wire of column `n` points outside the permuted columns.
    WireColumn(GateType, usize),
    /// The cell `src` differs from the cell `dst` it is wired to.
    CopyConstraint {
        /// The gate.
        typ: GateType,
        /// The cell of the row.
        src: Wire,
        /// The cell it is wired to.
        dst: Wire,
    },
    /// The row is outside the witness.
    FailedToGetWitnessForRow(GateType, usize),
}

impl fmt::Display for CircuitGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constraint(typ, n) => write!(f, "Invalid {typ:?} constraint: {n}"),
            Self::WireColumn(typ, n) => write!(f, "Invalid {typ:?} wire column: {n}"),
            Self::CopyConstraint { typ, src, dst } => write!(
                f,
                "Invalid {typ:?} copy constraint: {},{} -> {},{}",
                src.row, src.col, dst.row, dst.col
            ),
            Self::FailedToGetWitnessForRow(typ, row) => {
                write!(f, "Failed to get {typ:?} witness for row {row}")
            }
        }
    }
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// Whether row `row` of `witness` satisfies this gate, as Kimchi's
    /// `CircuitGate::verify_witness` decides it: the wiring of the
    /// permuted cells of the row, then the gate's constraints by `gates`,
    /// on the row and the next (zero past the end). Like Kimchi's, it
    /// checks no constraint for the Zero, Generic and Lookup gates, and no
    /// lookup.
    ///
    /// # Errors
    ///
    /// The first [`CircuitGateError`].
    pub fn verify_witness<G>(
        &self,
        row: usize,
        witness: &[Vec<F>; COLUMNS],
        gates: &G,
    ) -> Result<(), CircuitGateError>
    where
        G: Gates<F>,
    {
        let typ = self.typ;
        if witness.iter().any(|column| column.len() <= row) {
            return Err(CircuitGateError::FailedToGetWitnessForRow(typ, row));
        }
        for (col, wire) in self.wires.iter().enumerate() {
            if wire.col >= PERMUTS {
                return Err(CircuitGateError::WireColumn(typ, col));
            }
            let there = witness[wire.col].get(wire.row);
            if there != Some(&witness[col][row]) {
                return Err(CircuitGateError::CopyConstraint {
                    typ,
                    src: Wire { row, col },
                    dst: *wire,
                });
            }
        }
        if matches!(typ, GateType::Zero | GateType::Generic | GateType::Lookup) {
            return Ok(());
        }
        let cell = |column: usize, rotation: usize| {
            witness[column]
                .get(row + rotation)
                .copied()
                .unwrap_or(F::ZERO)
        };
        let coefficient =
            |column: usize, _: usize| self.coeffs.get(column).copied().unwrap_or(F::ZERO);
        let values = gates.evaluate(typ, &cell, &coefficient);
        match values.iter().position(|v| !v.is_zero()) {
            Some(i) => Err(CircuitGateError::Constraint(typ, i + 1)),
            None => Ok(()),
        }
    }
}

impl<F> CircuitGate<F>
where
    F: PrimeField,
{
    /// Whether row `row` satisfies this Generic gate's two half-gates, the
    /// first less the public input of its row (Kimchi's `verify_generic`).
    ///
    /// # Errors
    ///
    /// `"generic: incorrect gate"`, Kimchi's message, when the gate is not a
    /// Generic gate or a half-gate does not hold.
    pub fn verify_generic(
        &self,
        row: usize,
        witness: &[Vec<F>; COLUMNS],
        public: &[F],
    ) -> Result<(), String> {
        let incorrect = || String::from("generic: incorrect gate");
        if self.typ != GateType::Generic {
            return Err(incorrect());
        }
        let cell = |c: usize| witness[c].get(row).copied().unwrap_or(F::ZERO);
        let coefficient = |i: usize| self.coeffs.get(i).copied().unwrap_or(F::ZERO);
        for half in 0..2 {
            let (k, r) = (5 * half, 3 * half);
            let sum = coefficient(k) * cell(r)
                + coefficient(k + 1) * cell(r + 1)
                + coefficient(k + 2) * cell(r + 2);
            let mul = coefficient(k + 3) * cell(r) * cell(r + 1);
            let x = if half == 0 {
                public.get(row).copied().unwrap_or(F::ZERO)
            } else {
                F::ZERO
            };
            if !(sum + mul + coefficient(k + 4) - x).is_zero() {
                return Err(incorrect());
            }
        }
        Ok(())
    }

    /// Whether row `row` satisfies this gate, as Kimchi's
    /// `CircuitGate::verify` decides it: nothing for Zero and Lookup,
    /// [`CircuitGate::verify_generic`] for Generic, and
    /// [`CircuitGate::verify_witness`] for the others. (Kimchi checks
    /// Poseidon, CompleteAdd, VarBaseMul, EndoMul and EndoMulScalar with
    /// verifiers of their own: they accept the same rows and word their
    /// rejections differently.)
    ///
    /// # Errors
    ///
    /// The rejection, as a message.
    pub fn verify<G>(
        &self,
        row: usize,
        witness: &[Vec<F>; COLUMNS],
        gates: &G,
        public: &[F],
    ) -> Result<(), String>
    where
        G: Gates<F>,
    {
        match self.typ {
            GateType::Zero | GateType::Lookup => Ok(()),
            GateType::Generic => self.verify_generic(row, witness, public),
            _ => self
                .verify_witness(row, witness, gates)
                .map_err(|e| e.to_string()),
        }
    }
}

/// Why a witness does not satisfy a constraint system (Kimchi's
/// `GateError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateError {
    /// The two cells are wired together and differ.
    DisconnectedWires(Wire, Wire),
    /// The public row's gate does not take its left cell.
    IncorrectPublic(usize),
    /// The gate of `row` rejects it.
    Custom {
        /// The row.
        row: usize,
        /// The gate's message.
        err: String,
    },
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DisconnectedWires(a, b) => write!(
                f,
                "the wires ({}, {}) and ({}, {}) differ",
                a.row, a.col, b.row, b.col
            ),
            Self::IncorrectPublic(row) => write!(f, "public row {row} has the wrong gate"),
            Self::Custom { row, err } => write!(f, "row {row}: {err}"),
        }
    }
}

/// Copy constraints between the cells of a list of gates.
pub trait Connect {
    /// Join the cycles of the cells `(row, col)` `a` and `b` by exchanging
    /// their successors (Kimchi's `connect_cell_pair`): two cells on
    /// distinct cycles end up on one.
    ///
    /// # Panics
    ///
    /// If a row is out of range or a column is not below [`PERMUTS`].
    fn connect_cell_pair(&mut self, a: (usize, usize), b: (usize, usize));

    /// Put the cells `a` and `b` on one cycle: [`Connect::connect_cell_pair`]
    /// when they are on two, nothing when they are already on one (where
    /// the swap would split the cycle). The sound way to wire a cell to
    /// another; the cycles are a partition, and this is its join.
    ///
    /// # Panics
    ///
    /// If a row is out of range or a column is not below [`PERMUTS`].
    fn join(&mut self, a: (usize, usize), b: (usize, usize));

    /// Kimchi's `connect_64bit`, meant to wire the cells 1 and 2 of
    /// `start_row`, the two high limbs of a RangeCheck0 row, to
    /// `(zero_row, 0)`.
    ///
    /// It does not: of its three swaps, the third joins two cells already on
    /// one cycle, which splits the cycle, and `(start_row, 1)` is left
    /// wired to itself. Only `(start_row, 2)` reaches the zero cell, so the
    /// row's value is bounded by `2^76`, not `2^64`. Kept as Kimchi has it
    /// (proof-systems `9227444890`), since this constructor is a port; the
    /// first two swaps alone would wire all three cells together.
    ///
    /// # Panics
    ///
    /// If a row is out of range.
    fn connect_64bit(&mut self, zero_row: usize, start_row: usize) {
        self.connect_cell_pair((start_row, 1), (start_row, 2));
        self.connect_cell_pair((start_row, 2), (zero_row, 0));
        self.connect_cell_pair((zero_row, 0), (start_row, 1));
    }
}

impl<F> Connect for [CircuitGate<F>] {
    fn connect_cell_pair(&mut self, a: (usize, usize), b: (usize, usize)) {
        let next_a = self[a.0].wires[a.1];
        self[a.0].wires[a.1] = self[b.0].wires[b.1];
        self[b.0].wires[b.1] = next_a;
    }

    fn join(&mut self, a: (usize, usize), b: (usize, usize)) {
        let start = Wire { row: a.0, col: a.1 };
        let mut at = self[a.0].wires[a.1];
        while at != start {
            if (at.row, at.col) == b {
                return;
            }
            at = self[at.row].wires[at.col];
        }
        if a != b {
            self.connect_cell_pair(a, b);
        }
    }
}

/// A fixed lookup table: its id and its columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupTable<F> {
    /// The table id.
    pub id: i32,
    /// The columns, of equal length.
    pub data: Vec<Vec<F>>,
}

/// A runtime table: its id and its first (fixed) column; the second is
/// the prover's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeTableCfg<F> {
    /// The table id.
    pub id: i32,
    /// The first column.
    pub first_column: Vec<F>,
}

/// The four domains: `d1` of size `n`, and `d2`, `d4`, `d8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationDomains<F> {
    /// Size `n`.
    pub d1: Domain<F>,
    /// Size `2n`.
    pub d2: Domain<F>,
    /// Size `4n`.
    pub d4: Domain<F>,
    /// Size `8n`.
    pub d8: Domain<F>,
}

impl<F> EvaluationDomains<F>
where
    F: TwoAdicField,
{
    /// The domains of size `n`, `2n`, `4n`, `8n` with `n` the least power
    /// of two at least `rows`, or `None` when the field has no subgroup of
    /// order `8n`.
    #[must_use]
    pub fn create(rows: usize) -> Option<Self> {
        let d1 = Domain::containing(rows)?;
        let log = d1.log_size();
        Some(Self {
            d1,
            d2: Domain::new(log + 1)?,
            d4: Domain::new(log + 2)?,
            d8: Domain::new(log + 3)?,
        })
    }
}

/// Why a constraint system cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Kimchi needs at least two gates.
    TooFewGates,
    /// A wire names a cell out of the circuit.
    InvalidWire {
        /// The row of the wire.
        row: usize,
        /// The column of the wire.
        col: usize,
    },
    /// A gate has more than [`COLUMNS`] coefficients.
    TooManyCoefficients {
        /// The row of the gate.
        row: usize,
    },
    /// The field has no domain of the size the circuit needs.
    DomainTooLarge {
        /// The rows needed.
        rows: usize,
    },
    /// The lookup tables are inconsistent.
    Lookup(LookupError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewGates => write!(f, "a circuit needs at least two gates"),
            Self::InvalidWire { row, col } => {
                write!(
                    f,
                    "a wire names the cell ({row}, {col}), out of the circuit"
                )
            }
            Self::TooManyCoefficients { row } => {
                write!(
                    f,
                    "the gate of row {row} has more than {COLUMNS} coefficients"
                )
            }
            Self::DomainTooLarge { rows } => write!(f, "no domain of {rows} rows in the field"),
            Self::Lookup(e) => write!(f, "lookup: {e}"),
        }
    }
}

/// The rows `zk_rows` must exceed for `num_chunks` chunks (Kimchi's
/// `zk_rows_strict_lower_bound`): each chunk of each permuted column and of
/// `z` reveals two evaluations, so `zk_rows >= (2 (PERMUTS + 1) c - 2) /
/// PERMUTS + 1`.
#[must_use]
pub const fn zk_rows_strict_lower_bound(num_chunks: usize) -> usize {
    (2 * (PERMUTS + 1) * num_chunks - 2) / PERMUTS
}

/// The parameters of a constraint system before it is built.
#[derive(Clone, Debug)]
pub struct Builder<F> {
    gates: Vec<CircuitGate<F>>,
    public: usize,
    prev_challenges: usize,
    lookup_tables: Vec<LookupTable<F>>,
    runtime_tables: Option<Vec<RuntimeTableCfg<F>>>,
    max_poly_size: Option<usize>,
}

impl<F> Builder<F>
where
    F: PrimeField + TwoAdicField,
{
    /// The number of public inputs (default 0).
    #[must_use]
    pub const fn public(mut self, public: usize) -> Self {
        self.public = public;
        self
    }

    /// The number of previous IPA challenges the proofs carry (default 0).
    #[must_use]
    pub const fn prev_challenges(mut self, prev_challenges: usize) -> Self {
        self.prev_challenges = prev_challenges;
        self
    }

    /// The fixed lookup tables (default none).
    #[must_use]
    pub fn lookup(mut self, tables: Vec<LookupTable<F>>) -> Self {
        self.lookup_tables = tables;
        self
    }

    /// The runtime tables (default none).
    #[must_use]
    pub fn runtime(mut self, tables: Option<Vec<RuntimeTableCfg<F>>>) -> Self {
        self.runtime_tables = tables;
        self
    }

    /// The rows of a commitment chunk (default: one chunk whatever the
    /// size).
    #[must_use]
    pub const fn max_poly_size(mut self, max_poly_size: Option<usize>) -> Self {
        self.max_poly_size = max_poly_size;
        self
    }

    /// The rows the lookup argument needs: every table row, the dummy
    /// zero row when no table has id 0.
    fn lookup_domain_size(&self) -> usize {
        let mut size = 0;
        let mut has_table_with_id_0 = false;
        for table in &self.lookup_tables {
            if table.id == 0 {
                has_table_with_id_0 = true;
            }
            size += table.data.first().map_or(0, Vec::len);
        }
        if let Some(runtime) = &self.runtime_tables {
            size += runtime.iter().map(|t| t.first_column.len()).sum::<usize>();
        }
        let uses = |t: GateTable| self.gates.iter().any(|g| g.typ.table() == Some(t));
        if uses(GateTable::Xor) {
            size += XOR_TABLE_SIZE;
        }
        if uses(GateTable::RangeCheck) {
            size += RANGE_CHECK_TABLE_SIZE;
        }
        if has_table_with_id_0 { size } else { size + 1 }
    }

    /// The zero-knowledge rows and the rows the domain must hold.
    fn zk_rows(&self) -> Result<(usize, usize), Error> {
        // One more row than the tables: it closes the lookup aggregation.
        let circuit = self.gates.len().max(self.lookup_domain_size() + 1);
        let mut zk_rows = 3;
        let mut rows = circuit + zk_rows;
        if let Some(max_poly_size) = self.max_poly_size {
            loop {
                let size = rows
                    .checked_next_power_of_two()
                    .ok_or(Error::DomainTooLarge { rows })?;
                let num_chunks = if size < max_poly_size {
                    1
                } else {
                    size / max_poly_size
                };
                zk_rows = zk_rows_strict_lower_bound(num_chunks) + 1;
                rows = circuit + zk_rows;
                if size >= rows {
                    break;
                }
            }
        }
        Ok((zk_rows, rows))
    }

    /// The constraint system.
    ///
    /// # Errors
    ///
    /// [`Error::TooFewGates`] below two gates, [`Error::InvalidWire`] for a
    /// wire out of the circuit, [`Error::TooManyCoefficients`], and
    /// [`Error::DomainTooLarge`] when the field has no domain of the size.
    pub fn build(self) -> Result<ConstraintSystem<F>, Error> {
        if self.gates.len() < 2 {
            return Err(Error::TooFewGates);
        }
        for (row, gate) in self.gates.iter().enumerate() {
            if gate.coeffs.len() > COLUMNS {
                return Err(Error::TooManyCoefficients { row });
            }
            for wire in &gate.wires {
                if wire.row >= self.gates.len() || wire.col >= PERMUTS {
                    return Err(Error::InvalidWire {
                        row: wire.row,
                        col: wire.col,
                    });
                }
            }
        }
        let (zk_rows, rows) = self.zk_rows()?;
        let domain = EvaluationDomains::create(rows).ok_or(Error::DomainTooLarge { rows })?;
        let n = domain.d1.size();
        let mut gates = self.gates;
        let used: Vec<GateType> = {
            let mut used: Vec<GateType> = gates.iter().map(|g| g.typ).collect();
            used.sort_unstable();
            used.dedup();
            used
        };
        let first_padding = gates.len();
        gates.extend(
            (first_padding..n)
                .map(|row| CircuitGate::new(GateType::Zero, Wire::for_row(row), vec![])),
        );
        let lookup = LookupIndex::create(
            &gates,
            &self.lookup_tables,
            self.runtime_tables.as_deref(),
            n,
            zk_rows,
        )
        .map_err(Error::Lookup)?;
        Ok(ConstraintSystem {
            public: self.public,
            prev_challenges: self.prev_challenges,
            domain,
            gates,
            zk_rows,
            shift: shifts(domain.d1.log_size()),
            lookup_tables: self.lookup_tables,
            runtime_tables: self.runtime_tables,
            max_poly_size: self.max_poly_size,
            lookup,
            used,
        })
    }
}

/// A built constraint system.
#[derive(Clone, Debug)]
pub struct ConstraintSystem<F> {
    /// The public inputs: the first rows of column 0.
    pub public: usize,
    /// The previous IPA challenges the proofs carry.
    pub prev_challenges: usize,
    /// The domains.
    pub domain: EvaluationDomains<F>,
    /// The gates, padded with Zero gates to `d1`.
    pub gates: Vec<CircuitGate<F>>,
    /// The zero-knowledge rows, the last of `d1`.
    pub zk_rows: usize,
    /// The permutation shifts.
    pub shift: [F; PERMUTS],
    /// The fixed lookup tables.
    pub lookup_tables: Vec<LookupTable<F>>,
    /// The runtime tables.
    pub runtime_tables: Option<Vec<RuntimeTableCfg<F>>>,
    /// The rows of a commitment chunk.
    pub max_poly_size: Option<usize>,
    /// The lookup index, when the circuit looks something up.
    pub lookup: Option<LookupIndex<F>>,
    used: Vec<GateType>,
}

impl<F> ConstraintSystem<F>
where
    F: PrimeField + TwoAdicField,
{
    /// Whether `witness`, padded with zero rows to the domain, satisfies
    /// every gate with the public inputs `public` (Kimchi's
    /// `ConstraintSystem::verify_witness`): row by row, the wiring of the
    /// row's permuted cells, that a public row takes its left cell, then
    /// [`CircuitGate::verify`].
    ///
    /// # Errors
    ///
    /// The first [`GateError`].
    pub fn verify_witness<G>(
        &self,
        witness: &[Vec<F>; COLUMNS],
        public: &[F],
        gates: &G,
    ) -> Result<(), GateError>
    where
        G: Gates<F>,
    {
        let n = self.domain.d1.size();
        let witness: [Vec<F>; COLUMNS] = core::array::from_fn(|c| {
            let mut column = witness[c].clone();
            column.resize(n.max(column.len()), F::ZERO);
            column
        });
        for (row, gate) in self.gates.iter().enumerate() {
            for (col, wire) in gate.wires.iter().enumerate() {
                if wire.col >= PERMUTS {
                    return Err(GateError::Custom {
                        row,
                        err: format!("a wire can only be connected to the first {PERMUTS} columns"),
                    });
                }
                if witness[col][row] != witness[wire.col][wire.row] {
                    return Err(GateError::DisconnectedWires(Wire { row, col }, *wire));
                }
            }
            if row < self.public && gate.coeffs.first() != Some(&F::ONE) {
                return Err(GateError::IncorrectPublic(row));
            }
            gate.verify(row, &witness, gates, public)
                .map_err(|err| GateError::Custom { row, err })?;
        }
        Ok(())
    }
}

impl<F> ConstraintSystem<F>
where
    F: PrimeField + TwoAdicField,
{
    /// A builder over `gates`, with no public input, no previous challenge,
    /// no table, one chunk.
    #[must_use]
    pub const fn create(gates: Vec<CircuitGate<F>>) -> Builder<F> {
        Builder {
            gates,
            public: 0,
            prev_challenges: 0,
            lookup_tables: Vec::new(),
            runtime_tables: None,
            max_poly_size: None,
        }
    }

    /// Whether a gate of the circuit (before padding) is of type `typ`:
    /// for an optional selector, whether Kimchi builds it.
    #[must_use]
    pub fn uses(&self, typ: GateType) -> bool {
        self.used.binary_search(&typ).is_ok()
    }

    /// The label of a cell: `shift[col] w^row`.
    #[must_use]
    pub fn cell_to_field(&self, wire: Wire) -> F {
        self.shift[wire.col] * self.domain.d1.element(wire.row)
    }

    /// The permutation columns on `d1`: at each cell, the label of the
    /// cell it is wired to. Kimchi then zeroes the rows `n + 2 - zk_rows`
    /// to `n - 2`, so that `z` is random there (none for three zk rows).
    #[must_use]
    pub fn sigmas(&self) -> [Vec<F>; PERMUTS] {
        let n = self.domain.d1.size();
        let mut sigmas: [Vec<F>; PERMUTS] = core::array::from_fn(|col| {
            self.gates
                .iter()
                .map(|gate| self.cell_to_field(gate.wires[col]))
                .collect()
        });
        for sigma in &mut sigmas {
            for value in sigma.iter_mut().take(n - 1).skip(n + 2 - self.zk_rows) {
                *value = F::ZERO;
            }
        }
        sigmas
    }

    /// The selector of `typ` on `d1`: 1 on its rows, 0 elsewhere.
    #[must_use]
    pub fn selector(&self, typ: GateType) -> Vec<F> {
        self.gates
            .iter()
            .map(|gate| if gate.typ == typ { F::ONE } else { F::ZERO })
            .collect()
    }

    /// The coefficient column `i` on `d1`.
    ///
    /// # Panics
    ///
    /// If `i` is not below [`COLUMNS`].
    #[must_use]
    pub fn coefficient(&self, i: usize) -> Vec<F> {
        assert!(i < COLUMNS, "{COLUMNS} coefficient columns");
        self.gates
            .iter()
            .map(|gate| gate.coeffs.get(i).copied().unwrap_or(F::ZERO))
            .collect()
    }

    /// The evaluations on `to` of the polynomial of degree below `n` with
    /// the evaluations `column` on `d1`, or `None` if `column` does not
    /// have `n` entries or `to` is smaller than `d1`.
    #[must_use]
    pub fn extend(&self, column: &[F], to: &Domain<F>) -> Option<Vec<F>> {
        let polynomial = self.domain.d1.interpolate(column)?;
        to.evaluate(&polynomial)
    }
}
