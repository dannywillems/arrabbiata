//! The builder: the environment that reads a gadget as a generalized PlonK
//! circuit and, given an input, as its witness.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};
use core::fmt;

use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::{self as generalized, Argument, Circuit, Failure, Gate, Lookup};
use arrabbiata_permutation::Permutation;

use crate::{
    env::{Cell, Env, TypedGadget},
    expr::{Atom, Expr, Wire},
    shape::{Of, Shape},
};

/// The columns of the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The number of witness columns.
    pub witness: usize,
    /// The number of fixed columns.
    pub fixed: usize,
    /// The number of witness columns, from the first, that can be copied
    /// between (Kimchi: 7 of its 15).
    pub permuted: usize,
}

/// What the builder rejects. Each names the gadget at fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A column outside the layout.
    Column {
        /// The gadget.
        gadget: &'static str,
        /// The column.
        column: usize,
    },
    /// A cell written twice (or a fixed cell given two values).
    Overwrite {
        /// The gadget.
        gadget: &'static str,
        /// The column.
        column: usize,
        /// The absolute row.
        row: usize,
    },
    /// A constraint does not hold on the witness.
    Unsatisfied {
        /// The gadget.
        gadget: &'static str,
        /// The absolute row of the constraint.
        row: usize,
    },
    /// Two instances of the gadget have different constraints at the same
    /// row of the gadget: a per-instance value was used as a constant
    /// instead of being put in a fixed column.
    Inconsistent {
        /// The gadget.
        gadget: &'static str,
    },
    /// The gadget's sub-gadgets took more rows than it declares.
    Rows {
        /// The gadget.
        gadget: &'static str,
    },
    /// A copy or a public input of a variable that is not a single cell
    /// (or, for a public input, not a cell of the first column).
    NotACell {
        /// The gadget.
        gadget: &'static str,
    },
    /// A copy from or to a column the permutation does not cover.
    NotPermuted {
        /// The gadget.
        gadget: &'static str,
        /// The column.
        column: usize,
    },
    /// An input of the circuit was used in a constraint without being put
    /// in a cell first.
    Unplaced {
        /// The gadget.
        gadget: &'static str,
    },
    /// The gates read more column variables than the circuit's `N`.
    Variables {
        /// The number needed.
        needed: usize,
    },
    /// A constraint of degree more than the circuit's `D`.
    Degree {
        /// The gadget.
        gadget: &'static str,
        /// Its degree.
        degree: usize,
    },
    /// A value read from a table is not one of its rows.
    NotInTable {
        /// The gadget.
        gadget: &'static str,
        /// The absolute row of the read.
        row: usize,
    },
    /// A table read with another content, or with a query of the wrong
    /// width.
    Table {
        /// The table.
        name: &'static str,
    },
    /// Generalized PlonK rejected the circuit.
    Circuit(generalized::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column { gadget, column } => {
                write!(f, "{gadget}: column {column} is outside the layout")
            }
            Self::Overwrite {
                gadget,
                column,
                row,
            } => {
                write!(f, "{gadget}: cell ({column}, {row}) is written twice")
            }
            Self::Unsatisfied { gadget, row } => {
                write!(f, "{gadget}: a constraint fails on row {row}")
            }
            Self::Inconsistent { gadget } => write!(
                f,
                "{gadget}: instances have different constraints; put per-instance values in fixed columns"
            ),
            Self::Rows { gadget } => {
                write!(f, "{gadget}: sub-gadgets take more rows than declared")
            }
            Self::NotACell { gadget } => {
                write!(f, "{gadget}: copied or exposed value is not a cell")
            }
            Self::NotPermuted { gadget, column } => {
                write!(f, "{gadget}: column {column} is not permuted")
            }
            Self::Unplaced { gadget } => {
                write!(f, "{gadget}: an input is used before being placed")
            }
            Self::Variables { needed } => write!(f, "the gates need {needed} variables"),
            Self::Degree { gadget, degree } => {
                write!(f, "{gadget}: a constraint has degree {degree}")
            }
            Self::NotInTable { gadget, row } => {
                write!(f, "{gadget}: a value read on row {row} is not in its table")
            }
            Self::Table { name } => write!(
                f,
                "table {name}: another content, or a query of the wrong width"
            ),
            Self::Circuit(e) => write!(f, "generalized PlonK: {e}"),
        }
    }
}

/// A compiled circuit and, when an input was given, its witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled<F, const N: usize, const D: usize>
where
    F: Field,
{
    /// The circuit.
    pub circuit: Circuit<F, N, D>,
    /// The witness columns (zero without an input).
    pub columns: Vec<Vec<F>>,
    /// The public inputs, in the order they were made public (zero without
    /// an input).
    pub public: Vec<F>,
    /// Each gate kind of the circuit, by its selector: the gadget and the
    /// row of the gadget its gates are on.
    pub kinds: Vec<(&'static str, isize)>,
}

/// The first requirement a witness of a compiled circuit breaks
/// ([`Compiled::check`]), named by gadget: Kimchi's `CircuitGateError`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    /// The columns have the wrong sizes.
    Shape,
    /// The cell at `(column, row)` differs from a cell it is copied to.
    Copy {
        /// The witness column.
        column: usize,
        /// The row.
        row: usize,
    },
    /// Constraint `index` (from 0, in the order asserted) of the gates of
    /// `gadget` at its row `offset` does not hold at `row`.
    Constraint {
        /// The gadget.
        gadget: &'static str,
        /// The row of the gadget.
        offset: isize,
        /// The absolute row.
        row: usize,
        /// The index of the constraint among the gate kind's.
        index: usize,
    },
    /// Read `index` (from 0) of the same gates looks up a tuple that is
    /// not in its table.
    Read {
        /// The gadget.
        gadget: &'static str,
        /// The row of the gadget.
        offset: isize,
        /// The absolute row.
        row: usize,
        /// The index of the read among the gate kind's.
        index: usize,
    },
    /// Public input `input` is not its cell.
    Public {
        /// The public input.
        input: usize,
    },
}

impl<F, const N: usize, const D: usize> Compiled<F, N, D>
where
    F: Field,
{
    /// The first requirement `columns` break with the circuit's public
    /// inputs, rows in order and, on a row, its copies, then its
    /// constraints, then its reads.
    ///
    /// # Errors
    ///
    /// The [`Violation`].
    pub fn check(&self, columns: &[Vec<F>]) -> Result<(), Violation> {
        let Some(failure) = self.circuit.failure(columns, &self.public) else {
            return Ok(());
        };
        // The position of item `i` among those of the same selector.
        let label = |selectors: &[Option<usize>], i: usize| {
            let q = selectors[i];
            let index = selectors[..i].iter().filter(|s| **s == q).count();
            let (gadget, offset) = q
                .and_then(|q| self.kinds.get(q))
                .copied()
                .unwrap_or(("", 0));
            (gadget, offset, index)
        };
        Err(match failure {
            Failure::Shape => Violation::Shape,
            Failure::Copy { column, row } => Violation::Copy { column, row },
            Failure::Binding { input } => Violation::Public { input },
            Failure::Gate { gate, row } => {
                let selectors: Vec<Option<usize>> =
                    self.circuit.gates().iter().map(Gate::selector).collect();
                let (gadget, offset, index) = label(&selectors, gate);
                Violation::Constraint {
                    gadget,
                    offset,
                    row,
                    index,
                }
            }
            Failure::Lookup { lookup, row } => {
                let selectors: Vec<Option<usize>> = self
                    .circuit
                    .lookups()
                    .iter()
                    .map(|l| l.selector())
                    .collect();
                let (gadget, offset, index) = label(&selectors, lookup);
                Violation::Read {
                    gadget,
                    offset,
                    row,
                    index,
                }
            }
        })
    }
}

/// The gates of one gadget at one row of the gadget, shared by every
/// instance.
#[derive(Clone, Debug)]
struct Kind<F> {
    gadget: &'static str,
    /// The row of the gadget the gates are on.
    offset: isize,
    /// The constraints and the table reads, in the order asserted.
    entries: Vec<Entry<F>>,
    /// The instance that defined the constraints.
    owner: usize,
    /// The rows the gates are on.
    rows: BTreeSet<usize>,
}

/// What a gadget asserts on a row: a constraint, or a read of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Entry<F> {
    Constraint(Expr<F>),
    Read(&'static str, Vec<Expr<F>>),
}

/// A read-only table: its rows, stored in fixed columns, one column per
/// entry of a row. A read asserts that a tuple of variables is one of the
/// rows (plookup, as Kimchi's lookups are).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rom<F> {
    name: &'static str,
    columns: Vec<usize>,
    rows: Vec<Vec<F>>,
}

impl<F> Rom<F>
where
    F: Field,
{
    /// The table `name` of `rows`, stored in the fixed `columns` (a row has
    /// one entry per column). The columns hold the table on every row of the
    /// circuit, repeating the first row below the last.
    #[must_use]
    pub const fn new(name: &'static str, columns: Vec<usize>, rows: Vec<Vec<F>>) -> Self {
        Self {
            name,
            columns,
            rows,
        }
    }

    /// The name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The fixed columns.
    #[must_use]
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// The rows.
    #[must_use]
    pub fn rows(&self) -> &[Vec<F>] {
        &self.rows
    }
}

/// A check postponed until the witness is complete: a constraint or a read
/// over a cell written after it.
#[derive(Clone, Debug)]
struct Deferred<F> {
    gadget: &'static str,
    row: usize,
    entry: Entry<F>,
}

/// A gadget being synthesized.
#[derive(Clone, Debug)]
struct Frame {
    gadget: &'static str,
    base: usize,
    id: usize,
    /// The constraints of each kind this instance has asserted so far.
    seen: BTreeMap<usize, usize>,
}

/// The environment of compilation.
///
/// It is created by [`circuit`], [`witness`] and [`evaluate`], which run
/// the gadget in it; it is not built directly.
#[derive(Clone, Debug)]
pub struct Builder<F> {
    layout: Layout,
    with_witness: bool,
    /// Whether broken constraints and reads are let through (the witness of
    /// [`assign`]), rather than reported.
    permissive: bool,
    cursor: usize,
    frames: Vec<Frame>,
    instances: usize,
    witness: BTreeMap<(usize, usize), F>,
    written: BTreeSet<(usize, usize)>,
    fixed: BTreeMap<(usize, usize), F>,
    copies: Vec<((usize, usize), (usize, usize))>,
    public: Vec<(usize, Option<F>)>,
    kinds: Vec<Kind<F>>,
    kind_index: BTreeMap<(&'static str, isize), usize>,
    tables: BTreeMap<&'static str, Rom<F>>,
    deferred: Vec<Deferred<F>>,
    last_row: usize,
    error: Option<Error>,
}

impl<F> Builder<F>
where
    F: Field,
{
    const fn new(layout: Layout, with_witness: bool) -> Self {
        Self {
            layout,
            with_witness,
            permissive: false,
            cursor: 0,
            frames: Vec::new(),
            instances: 0,
            witness: BTreeMap::new(),
            written: BTreeSet::new(),
            fixed: BTreeMap::new(),
            copies: Vec::new(),
            public: Vec::new(),
            kinds: Vec::new(),
            kind_index: BTreeMap::new(),
            tables: BTreeMap::new(),
            deferred: Vec::new(),
            last_row: 0,
            error: None,
        }
    }

    fn fail(&mut self, error: Error) {
        let broken = matches!(error, Error::Unsatisfied { .. } | Error::NotInTable { .. });
        if self.permissive && broken {
            return;
        }
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn gadget(&self) -> &'static str {
        self.frames.last().map_or("circuit", |f| f.gadget)
    }

    fn absolute(&self, at: Cell) -> (usize, usize) {
        let base = self.frames.last().map_or(0, |f| f.base);
        (at.column, base + at.row)
    }

    /// Write the witness cell `cell`, once.
    fn write(&mut self, cell: (usize, usize), value: Option<F>) -> Wire<F> {
        let (column, row) = cell;
        let gadget = self.gadget();
        if column >= self.layout.witness {
            self.fail(Error::Column { gadget, column });
        } else if !self.written.insert(cell) {
            self.fail(Error::Overwrite {
                gadget,
                column,
                row,
            });
        }
        if let Some(v) = value {
            self.witness.insert(cell, v);
        }
        self.last_row = self.last_row.max(row);
        Wire::new(Expr::Witness { column, row }, value)
    }

    /// Check the kinds this instance shares with an earlier instance.
    fn close(&mut self, frame: &Frame) {
        for (kind, count) in &frame.seen {
            let k = &self.kinds[*kind];
            if k.owner != frame.id && *count != k.entries.len() {
                let gadget = k.gadget;
                self.fail(Error::Inconsistent { gadget });
            }
        }
    }

    /// The anchor of an entry first reading row `first_read`: the gadget's
    /// first row, unless the entry reads a row before it.
    fn anchor(&self, first_read: Option<usize>) -> usize {
        let base = self.frames.last().map_or(0, |f| f.base);
        match first_read {
            Some(row) if row < base => row,
            _ => base,
        }
    }

    /// Record `entry` (relative to `anchor`) in the current gadget's kind for
    /// the row `anchor`, checking it against the kind's first instance.
    fn record(&mut self, anchor: usize, entry: Entry<F>) {
        let gadget = self.gadget();
        let Some(frame) = self.frames.last() else {
            return;
        };
        let offset = anchor.cast_signed() - frame.base.cast_signed();
        let id = frame.id;
        let kind = match self.kind_index.get(&(gadget, offset)) {
            Some(k) => *k,
            None => {
                self.kinds.push(Kind {
                    gadget,
                    offset,
                    entries: Vec::new(),
                    owner: id,
                    rows: BTreeSet::new(),
                });
                let k = self.kinds.len() - 1;
                self.kind_index.insert((gadget, offset), k);
                k
            }
        };
        let frame = self.frames.last_mut().unwrap_or_else(|| unreachable!());
        let count = *frame.seen.get(&kind).unwrap_or(&0);
        frame.seen.insert(kind, count + 1);
        let k = &mut self.kinds[kind];
        k.rows.insert(anchor);
        if k.owner == id {
            k.entries.push(entry);
        } else if k.entries.get(count) != Some(&entry) {
            self.fail(Error::Inconsistent { gadget });
        }
    }

    /// The value of an expression in the finished witness.
    fn value_of(&self, expr: &Expr<F>) -> Option<F> {
        let witness =
            |c: usize, r: usize| Some(self.witness.get(&(c, r)).copied().unwrap_or(F::ZERO));
        let fixed = |c: usize, r: usize| Some(self.fixed.get(&(c, r)).copied().unwrap_or(F::ZERO));
        expr.evaluate(&witness, &fixed)
    }

    /// Whether `values` is a row of the table `name`.
    fn in_table(&self, name: &str, values: &[F]) -> bool {
        self.tables
            .get(name)
            .is_some_and(|t| t.rows.iter().any(|row| row.as_slice() == values))
    }

    /// The postponed checks, on the finished witness.
    fn check_deferred(&mut self) {
        let deferred = core::mem::take(&mut self.deferred);
        for d in &deferred {
            let holds = match &d.entry {
                Entry::Constraint(e) => self.value_of(e) == Some(F::ZERO),
                Entry::Read(name, query) => {
                    let values: Option<Vec<F>> = query.iter().map(|e| self.value_of(e)).collect();
                    values.is_some_and(|v| self.in_table(name, &v))
                }
            };
            if !holds {
                let (gadget, row) = (d.gadget, d.row);
                match d.entry {
                    Entry::Constraint(_) => self.fail(Error::Unsatisfied { gadget, row }),
                    Entry::Read(..) => self.fail(Error::NotInTable { gadget, row }),
                }
            }
        }
    }

    fn finish<const N: usize, const D: usize>(mut self) -> Result<Compiled<F, N, D>, Error> {
        if self.with_witness {
            self.check_deferred();
        }
        if let Some(e) = self.error {
            return Err(e);
        }
        let n = domain_size(self.cursor, self.last_row, &self.tables);
        // The tables, in their fixed columns, on every row.
        for table in self.tables.values() {
            for (j, column) in table.columns.iter().enumerate() {
                if *column >= self.layout.fixed {
                    return Err(Error::Column {
                        gadget: table.name,
                        column: *column,
                    });
                }
                for row in 0..n {
                    let entry = table.rows.get(row).unwrap_or(&table.rows[0])[j];
                    if let Some(old) = self.fixed.insert((*column, row), entry)
                        && old != entry
                    {
                        return Err(Error::Overwrite {
                            gadget: table.name,
                            column: *column,
                            row,
                        });
                    }
                }
            }
        }
        // The column variables, in one numbering for every gate and read.
        let mut atoms = Vec::new();
        for kind in &self.kinds {
            for entry in &kind.entries {
                match entry {
                    Entry::Constraint(c) => c.atoms(&mut atoms),
                    Entry::Read(_, query) => query.iter().for_each(|e| e.atoms(&mut atoms)),
                }
            }
        }
        atoms.sort_unstable();
        atoms.dedup();
        if atoms.len() > N {
            return Err(Error::Variables {
                needed: atoms.len(),
            });
        }
        let index: BTreeMap<Atom, usize> = atoms.iter().enumerate().map(|(i, a)| (*a, i)).collect();
        let variables = atoms.iter().map(|a| a.variable()).collect();
        let mut gates = Vec::new();
        let mut lookups = Vec::new();
        let mut selectors = Vec::new();
        for (k, kind) in self.kinds.iter().enumerate() {
            let mut selector = vec![F::ZERO; n];
            for row in &kind.rows {
                selector[*row] = F::ONE;
            }
            selectors.push(selector);
            let gadget = kind.gadget;
            let sparse = |e: &Expr<F>| {
                let degree = e.degree();
                e.to_sparse::<N, D>(&index)
                    .ok_or(Error::Degree { gadget, degree })
            };
            for entry in &kind.entries {
                match entry {
                    Entry::Constraint(c) => gates.push(Gate::new(Some(k), sparse(c)?)),
                    Entry::Read(name, query) => {
                        let inputs = query.iter().map(sparse).collect::<Result<Vec<_>, _>>()?;
                        let table = self
                            .tables
                            .get(name)
                            .map(|t| t.columns.clone())
                            .unwrap_or_default();
                        lookups.push(Lookup::new(Argument::Plookup, Some(k), inputs, table));
                    }
                }
            }
        }
        let columns: Vec<Vec<F>> = (0..self.layout.witness)
            .map(|c| {
                (0..n)
                    .map(|r| self.witness.get(&(c, r)).copied().unwrap_or(F::ZERO))
                    .collect()
            })
            .collect();
        let fixed = (0..self.layout.fixed)
            .map(|c| {
                (0..n)
                    .map(|r| self.fixed.get(&(c, r)).copied().unwrap_or(F::ZERO))
                    .collect()
            })
            .collect();
        let public_inputs = self.public.len();
        let bindings = self
            .public
            .iter()
            .enumerate()
            .map(|(i, (row, _))| (i, *row))
            .collect();
        let public = self
            .public
            .iter()
            .map(|(_, v)| v.unwrap_or(F::ZERO))
            .collect();
        let every_cell_alone = (0..self.layout.witness * n).collect();
        let circuit = Circuit::new(
            n,
            self.layout.witness,
            fixed,
            selectors,
            variables,
            gates,
            every_cell_alone,
            public_inputs,
            bindings,
        )
        .map_err(Error::Circuit)?;
        let cell = |(column, row): (usize, usize)| column * n + row;
        let copies: Vec<(usize, usize)> = self
            .copies
            .iter()
            .map(|(a, b)| (cell(*a), cell(*b)))
            .collect();
        let permuted = self.layout.permuted;
        let sigma = Permutation::from_copies(permuted, n, &copies)
            .map_err(|_| Error::Circuit(generalized::Error::Shape))?;
        let circuit = circuit
            .with_permutation((0..permuted).collect(), sigma.sigma().to_vec())
            .map_err(Error::Circuit)?;
        let circuit = if lookups.is_empty() {
            circuit
        } else {
            circuit.with_lookups(lookups).map_err(Error::Circuit)?
        };
        let kinds = self.kinds.iter().map(|k| (k.gadget, k.offset)).collect();
        Ok(Compiled {
            circuit,
            columns,
            public,
            kinds,
        })
    }
}

impl<F> Env<F> for Builder<F>
where
    F: Field,
{
    type Var = Wire<F>;

    fn constant(&mut self, value: F) -> Wire<F> {
        Wire::new(Expr::Constant(value), Some(value))
    }

    fn fixed(&mut self, at: Cell, value: F) -> Wire<F> {
        let (column, row) = self.absolute(at);
        let gadget = self.gadget();
        if column >= self.layout.fixed {
            self.fail(Error::Column { gadget, column });
        } else if let Some(old) = self.fixed.insert((column, row), value)
            && old != value
        {
            self.fail(Error::Overwrite {
                gadget,
                column,
                row,
            });
        }
        self.last_row = self.last_row.max(row);
        Wire::new(Expr::Fixed { column, row }, Some(value))
    }

    fn define(&mut self, at: Cell, value: Wire<F>) -> Wire<F> {
        let cell = self.absolute(at);
        let wire = self.write(cell, value.value);
        self.assert_zero(wire.clone() - value);
        wire
    }

    fn advice<const K: usize>(
        &mut self,
        at: Cell,
        inputs: [&Wire<F>; K],
        hint: impl FnOnce([F; K]) -> F,
    ) -> Wire<F> {
        let values: Option<Vec<F>> = inputs.iter().map(|w| w.value).collect();
        let value = values
            .filter(|_| self.with_witness)
            .map(|v| hint(core::array::from_fn(|i| v[i])));
        let cell = self.absolute(at);
        self.write(cell, value)
    }

    fn copy(&mut self, at: Cell, value: &Wire<F>) -> Wire<F> {
        let gadget = self.gadget();
        let Some(source) = value.cell() else {
            self.fail(Error::NotACell { gadget });
            return value.clone();
        };
        let target = self.absolute(at);
        if source == target {
            return value.clone();
        }
        for column in [source.0, target.0] {
            if column >= self.layout.permuted {
                self.fail(Error::NotPermuted { gadget, column });
            }
        }
        self.copies.push((source, target));
        self.write(target, value.value)
    }

    fn connect(&mut self, a: &Wire<F>, b: &Wire<F>) {
        let gadget = self.gadget();
        let (Some(x), Some(y)) = (a.cell(), b.cell()) else {
            self.fail(Error::NotACell { gadget });
            return;
        };
        for column in [x.0, y.0] {
            if column >= self.layout.permuted {
                self.fail(Error::NotPermuted { gadget, column });
            }
        }
        if self.with_witness && a.value.is_some() && b.value.is_some() && a.value != b.value {
            self.fail(Error::Unsatisfied { gadget, row: x.1 });
        }
        if x != y {
            self.copies.push((x, y));
        }
    }

    fn place(&mut self, at: Cell, value: &Wire<F>) -> Wire<F> {
        let permuted = self.layout.permuted;
        let target = self.absolute(at);
        match value.cell() {
            // Already there.
            Some(cell) if cell == target => value.clone(),
            Some((column, _)) if column < permuted && at.column < permuted => self.copy(at, value),
            _ => self.define(at, value.clone()),
        }
    }

    fn public(&mut self, value: &Wire<F>) {
        match value.cell() {
            Some((0, row)) => self.public.push((row, value.value)),
            _ => {
                let gadget = self.gadget();
                self.fail(Error::NotACell { gadget });
            }
        }
    }

    fn assert_zero(&mut self, value: Wire<F>) {
        let gadget = self.gadget();
        if value.expr.reads_input() {
            self.fail(Error::Unplaced { gadget });
            return;
        }
        let Some(first_read) = value.expr.min_row() else {
            // A constraint on constants only.
            if value.expr != Expr::Constant(F::ZERO) {
                let row = self.cursor;
                self.fail(Error::Unsatisfied { gadget, row });
            }
            return;
        };
        // A constraint lives on its gadget's first row, reading the later
        // rows by rotation, as a Kimchi gate does; one reading a row before
        // the gadget lives on the first row it reads.
        let anchor = self.anchor(Some(first_read));
        if self.with_witness {
            match value.value {
                Some(v) if v != F::ZERO => self.fail(Error::Unsatisfied {
                    gadget,
                    row: first_read,
                }),
                // A cell this constraint reads is written later.
                None => self.deferred.push(Deferred {
                    gadget,
                    row: first_read,
                    entry: Entry::Constraint(value.expr.clone()),
                }),
                Some(_) => {}
            }
        }
        let relative = value.expr.relative_to(anchor);
        self.record(anchor, Entry::Constraint(relative));
    }

    fn read(&mut self, table: &Rom<F>, query: &[Wire<F>]) {
        let gadget = self.gadget();
        if query.iter().any(|w| w.expr.reads_input()) {
            self.fail(Error::Unplaced { gadget });
            return;
        }
        match self.tables.get(table.name) {
            Some(known) if known != table => {
                let name = table.name;
                self.fail(Error::Table { name });
                return;
            }
            Some(_) => {}
            None => {
                self.tables.insert(table.name, table.clone());
            }
        }
        if query.len() != table.columns.len() {
            let name = table.name;
            self.fail(Error::Table { name });
            return;
        }
        let first_read = query.iter().filter_map(|w| w.expr.min_row()).min();
        let anchor = self.anchor(first_read);
        let row = first_read.unwrap_or(anchor);
        let exprs: Vec<Expr<F>> = query.iter().map(|w| w.expr.clone()).collect();
        if self.with_witness {
            let values: Option<Vec<F>> = query.iter().map(|w| w.value).collect();
            match values {
                Some(v) if !self.in_table(table.name, &v) => {
                    self.fail(Error::NotInTable { gadget, row })
                }
                None => self.deferred.push(Deferred {
                    gadget,
                    row,
                    entry: Entry::Read(table.name, exprs.clone()),
                }),
                Some(_) => {}
            }
        }
        let relative = exprs.iter().map(|e| e.relative_to(anchor)).collect();
        self.record(anchor, Entry::Read(table.name, relative));
    }

    fn cell(&mut self, at: Cell) -> Wire<F> {
        let (column, row) = self.absolute(at);
        if column >= self.layout.witness {
            let gadget = self.gadget();
            self.fail(Error::Column { gadget, column });
        }
        self.last_row = self.last_row.max(row);
        let value = self.witness.get(&(column, row)).copied();
        Wire::new(Expr::Witness { column, row }, value)
    }

    fn call<G>(&mut self, gadget: &G, input: Of<G::Input, Wire<F>>) -> Of<G::Output, Wire<F>>
    where
        G: TypedGadget<F>,
    {
        let base = self.cursor;
        let id = self.instances;
        self.instances += 1;
        self.frames.push(Frame {
            gadget: G::NAME,
            base,
            id,
            seen: BTreeMap::new(),
        });
        let output = gadget.synthesize(self, input);
        if let Some(frame) = self.frames.pop() {
            self.close(&frame);
        }
        let end = base + G::ROWS;
        if self.cursor > end {
            self.fail(Error::Rows { gadget: G::NAME });
        }
        self.cursor = self.cursor.max(end);
        output
    }
}

fn inputs<F, S>(values: Option<&Of<S, F>>) -> Of<S, Wire<F>>
where
    F: Field,
    S: Shape,
{
    let wires: Vec<Wire<F>> = match values {
        Some(v) => S::flatten(v)
            .into_iter()
            .enumerate()
            .map(|(i, x)| Wire::new(Expr::Input(i), Some(x)))
            .collect(),
        None => (0..S::SIZE)
            .map(|i| Wire::new(Expr::Input(i), None))
            .collect(),
    };
    S::unflatten(&wires)
}

/// The circuit of `gadget`, without a witness: what the setup needs.
///
/// # Errors
///
/// An [`Error`] naming the gadget at fault.
pub fn circuit<F, G, const N: usize, const D: usize>(
    gadget: &G,
    layout: Layout,
) -> Result<Circuit<F, N, D>, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, false);
    let input = inputs::<F, G::Input>(None);
    let _ = builder.call(gadget, input);
    builder.finish().map(|c| c.circuit)
}

/// The compiled circuit of `gadget`, without a witness: [`circuit`] with the
/// gadget kind of each selector, the symbolic interpretation of the gadget
/// alone (its witness columns empty). Because a gadget's layout never
/// depends on a value, this is the circuit [`witness`] compiles on every
/// input.
///
/// # Errors
///
/// An [`Error`] naming the gadget at fault.
pub fn compile<F, G, const N: usize, const D: usize>(
    gadget: &G,
    layout: Layout,
) -> Result<Compiled<F, N, D>, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, false);
    let input = inputs::<F, G::Input>(None);
    let _ = builder.call(gadget, input);
    builder.finish()
}

/// A compiled circuit with its witness, and the gadget's output.
pub type Witnessed<F, G, const N: usize, const D: usize> =
    (Compiled<F, N, D>, Of<<G as TypedGadget<F>>::Output, F>);

/// The circuit of `gadget`, its witness on `input`, and its output.
///
/// The circuit is the one [`circuit`] gives: it does not depend on the
/// input.
///
/// # Errors
///
/// An [`Error`] naming the gadget at fault; [`Error::Unsatisfied`] at the
/// first constraint the witness breaks.
pub fn witness<F, G, const N: usize, const D: usize>(
    gadget: &G,
    layout: Layout,
    input: &Of<G::Input, F>,
) -> Result<Witnessed<F, G, N, D>, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, true);
    let wires = inputs::<F, G::Input>(Some(input));
    let output = builder.call(gadget, wires);
    let values: Vec<F> = G::Output::flatten(&output)
        .iter()
        .map(|w| w.value.unwrap_or(F::ZERO))
        .collect();
    let compiled = builder.finish()?;
    Ok((compiled, G::Output::unflatten(&values)))
}

/// The circuit of `gadget` and the witness it computes on `input`, whether
/// or not the witness satisfies the circuit: a witness generator's output
/// (Kimchi's `create_*_witness`), for a test that checks a broken one.
/// [`Compiled::check`] then names what it breaks.
///
/// # Errors
///
/// As [`witness`], except [`Error::Unsatisfied`] and [`Error::NotInTable`].
pub fn assign<F, G, const N: usize, const D: usize>(
    gadget: &G,
    layout: Layout,
    input: &Of<G::Input, F>,
) -> Result<Witnessed<F, G, N, D>, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, true);
    builder.permissive = true;
    let wires = inputs::<F, G::Input>(Some(input));
    let output = builder.call(gadget, wires);
    let values: Vec<F> = G::Output::flatten(&output)
        .iter()
        .map(|w| w.value.unwrap_or(F::ZERO))
        .collect();
    let compiled = builder.finish()?;
    Ok((compiled, G::Output::unflatten(&values)))
}

/// The output of `gadget` on `input`: the function the gadget computes, from
/// the same `synthesize` as its constraints.
///
/// # Errors
///
/// As [`witness`].
pub fn evaluate<F, G>(
    gadget: &G,
    layout: Layout,
    input: &Of<G::Input, F>,
) -> Result<Of<G::Output, F>, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, true);
    let wires = inputs::<F, G::Input>(Some(input));
    let output = builder.call(gadget, wires);
    if let Some(e) = builder.error {
        return Err(e);
    }
    let values: Vec<F> = G::Output::flatten(&output)
        .iter()
        .map(|w| w.value.unwrap_or(F::ZERO))
        .collect();
    Ok(G::Output::unflatten(&values))
}

/// The figures of one gate kind: the constraints one gadget asserts on one
/// of its rows, shared by every instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateFigures {
    /// The gadget.
    pub gadget: &'static str,
    /// The row of the gadget the gates are on.
    pub offset: isize,
    /// The number of constraints.
    pub constraints: usize,
    /// The degree of each constraint, in order: the total degree in the
    /// cells (witness and fixed alike), constants of degree 0, the way
    /// Kimchi's `Expr::degree(1, 0)` counts it. The selector is not
    /// included (the quotient sees one more).
    pub degrees: Vec<usize>,
    /// The largest of [`GateFigures::degrees`].
    pub max_degree: usize,
    /// The number of rows the gates are on: the instances.
    pub instances: usize,
    /// The witness cells the constraints read, as (column, rotation).
    pub witness_cells: Vec<(usize, usize)>,
    /// The fixed cells the constraints read, as (column, rotation).
    pub fixed_cells: Vec<(usize, usize)>,
    /// The largest rotation read (0: the row only, 1: the next row too).
    pub max_rotation: usize,
    /// The table of each read, in order: one lookup each.
    pub reads: Vec<&'static str>,
    /// The witness cells the reads look up, as (column, rotation).
    pub read_cells: Vec<(usize, usize)>,
}

/// The figures of a circuit, without compiling it to a fixed `N` and `D`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Figures {
    /// The rows the gadgets take.
    pub rows: usize,
    /// The number of rows of the table: the next power of two.
    pub domain_size: usize,
    /// Each gate kind, in the order the builder met them; each is one
    /// selector.
    pub gates: Vec<GateFigures>,
    /// The number of column variables the gates read: generalized PlonK's
    /// smallest `N`.
    pub variables: usize,
    /// The largest constraint degree: generalized PlonK's smallest `D`.
    pub max_degree: usize,
    /// The number of copy constraints.
    pub copies: usize,
    /// The number of public inputs.
    pub public_inputs: usize,
    /// The tables read: name, rows, columns.
    pub tables: Vec<(&'static str, usize, usize)>,
}

impl Figures {
    /// The gate kinds of the gadget named `gadget`.
    #[must_use]
    pub fn gates_of(&self, gadget: &str) -> Vec<&GateFigures> {
        self.gates.iter().filter(|g| g.gadget == gadget).collect()
    }
}

/// The figures of `gadget`'s circuit: its gate kinds (constraints, degrees,
/// instances, cells read), its rows, variables, copies and public inputs.
///
/// # Errors
///
/// As [`circuit`].
pub fn figures<F, G>(gadget: &G, layout: Layout) -> Result<Figures, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, false);
    let input = inputs::<F, G::Input>(None);
    let _ = builder.call(gadget, input);
    if let Some(e) = builder.error {
        return Err(e);
    }
    let mut all_atoms = Vec::new();
    let gates: Vec<GateFigures> = builder
        .kinds
        .iter()
        .map(|kind| {
            let constraints: Vec<&Expr<F>> = kind
                .entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Constraint(c) => Some(c),
                    Entry::Read(..) => None,
                })
                .collect();
            let reads: Vec<(&'static str, &Vec<Expr<F>>)> = kind
                .entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Read(name, query) => Some((*name, query)),
                    Entry::Constraint(_) => None,
                })
                .collect();
            let degrees: Vec<usize> = constraints.iter().map(|c| c.degree()).collect();
            let mut atoms = Vec::new();
            for c in &constraints {
                c.atoms(&mut atoms);
            }
            let mut read_atoms = Vec::new();
            for (_, query) in &reads {
                query.iter().for_each(|e| e.atoms(&mut read_atoms));
            }
            read_atoms.sort_unstable();
            read_atoms.dedup();
            all_atoms.extend(read_atoms.iter().copied());
            let read_cells = read_atoms
                .iter()
                .filter_map(|a| match *a {
                    Atom::Witness { column, rotation } => Some((column, rotation)),
                    Atom::Fixed { .. } => None,
                })
                .collect();
            atoms.sort_unstable();
            atoms.dedup();
            all_atoms.extend(atoms.iter().copied());
            let mut witness_cells = Vec::new();
            let mut fixed_cells = Vec::new();
            for a in &atoms {
                match *a {
                    Atom::Witness { column, rotation } => witness_cells.push((column, rotation)),
                    Atom::Fixed { column, rotation } => fixed_cells.push((column, rotation)),
                }
            }
            let max_rotation = witness_cells
                .iter()
                .chain(&fixed_cells)
                .map(|(_, r)| *r)
                .max()
                .unwrap_or(0);
            GateFigures {
                gadget: kind.gadget,
                offset: kind.offset,
                constraints: constraints.len(),
                reads: reads.iter().map(|(name, _)| *name).collect(),
                read_cells,
                max_degree: degrees.iter().copied().max().unwrap_or(0),
                degrees,
                instances: kind.rows.len(),
                witness_cells,
                fixed_cells,
                max_rotation,
            }
        })
        .collect();
    all_atoms.sort_unstable();
    all_atoms.dedup();
    let rows = builder.cursor.max(builder.last_row + 1);
    Ok(Figures {
        rows,
        domain_size: domain_size(builder.cursor, builder.last_row, &builder.tables),
        tables: builder
            .tables
            .values()
            .map(|t| (t.name, t.rows.len(), t.columns.len()))
            .collect(),
        max_degree: gates.iter().map(|g| g.max_degree).max().unwrap_or(0),
        gates,
        variables: all_atoms.len(),
        copies: builder.copies.len(),
        public_inputs: builder.public.len(),
    })
}

/// The rows of the table: the gadgets' rows, and for plookup a free last
/// row and a row per table entry, rounded up to a power of two.
fn domain_size<F>(
    cursor: usize,
    last_row: usize,
    tables: &BTreeMap<&'static str, Rom<F>>,
) -> usize {
    let reserve = usize::from(!tables.is_empty());
    let table_rows = tables.values().map(|t| t.rows.len() + 1).max().unwrap_or(0);
    cursor
        .max(last_row + 1 + reserve)
        .max(table_rows)
        .max(2)
        .next_power_of_two()
}

/// The constraints and table reads of one gate kind, as polynomials in the
/// cells relative to the gate's row: what a differential test evaluates
/// against another implementation of the same gate.
#[derive(Clone, Debug)]
pub struct GateConstraints<F> {
    /// The gadget.
    pub gadget: &'static str,
    /// The row of the gadget the gates are on.
    pub offset: isize,
    constraints: Vec<Expr<F>>,
    reads: Vec<(&'static str, Vec<Expr<F>>)>,
}

impl<F> GateConstraints<F>
where
    F: Field,
{
    /// The value of each constraint, in the order asserted, at the witness
    /// cells `witness(column, rotation)` and the fixed cells
    /// `fixed(column, rotation)`, rotation 0 being the gate's row.
    pub fn evaluate(
        &self,
        witness: &dyn Fn(usize, usize) -> F,
        fixed: &dyn Fn(usize, usize) -> F,
    ) -> Vec<F> {
        let w = |c: usize, r: usize| Some(witness(c, r));
        let f = |c: usize, r: usize| Some(fixed(c, r));
        self.constraints
            .iter()
            .map(|c| c.evaluate(&w, &f).unwrap_or(F::ZERO))
            .collect()
    }

    /// The tuple each read looks up, in order, with its table, at the same
    /// cells.
    pub fn reads(
        &self,
        witness: &dyn Fn(usize, usize) -> F,
        fixed: &dyn Fn(usize, usize) -> F,
    ) -> Vec<(&'static str, Vec<F>)> {
        let w = |c: usize, r: usize| Some(witness(c, r));
        let f = |c: usize, r: usize| Some(fixed(c, r));
        self.reads
            .iter()
            .map(|(name, query)| {
                let values = query
                    .iter()
                    .map(|e| e.evaluate(&w, &f).unwrap_or(F::ZERO))
                    .collect();
                (*name, values)
            })
            .collect()
    }
}

/// The gate kinds of `gadget`'s circuit with their constraints, in the
/// order the builder met them.
///
/// # Errors
///
/// As [`circuit`].
pub fn gates<F, G>(gadget: &G, layout: Layout) -> Result<Vec<GateConstraints<F>>, Error>
where
    F: Field,
    G: TypedGadget<F>,
{
    let mut builder = Builder::new(layout, false);
    let input = inputs::<F, G::Input>(None);
    let _ = builder.call(gadget, input);
    if let Some(e) = builder.error {
        return Err(e);
    }
    Ok(builder
        .kinds
        .iter()
        .map(|kind| GateConstraints {
            gadget: kind.gadget,
            offset: kind.offset,
            constraints: kind
                .entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Constraint(c) => Some(c.clone()),
                    Entry::Read(..) => None,
                })
                .collect(),
            reads: kind
                .entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Read(name, query) => Some((*name, query.clone())),
                    Entry::Constraint(_) => None,
                })
                .collect(),
        })
        .collect())
}
