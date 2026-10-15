//! The builder: the environment that reads a gadget as a generalized PlonK
//! circuit and, given an input, as its witness.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};
use core::fmt;

use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::{self as generalized, Circuit, Gate};
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
}

/// The gates of one gadget at one row of the gadget, shared by every
/// instance.
#[derive(Clone, Debug)]
struct Kind<F> {
    gadget: &'static str,
    constraints: Vec<Expr<F>>,
    /// The instance that defined the constraints.
    owner: usize,
    /// The rows the gates are on.
    rows: BTreeSet<usize>,
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
            last_row: 0,
            error: None,
        }
    }

    fn fail(&mut self, error: Error) {
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
            if k.owner != frame.id && *count != k.constraints.len() {
                let gadget = k.gadget;
                self.fail(Error::Inconsistent { gadget });
            }
        }
    }

    fn finish<const N: usize, const D: usize>(self) -> Result<Compiled<F, N, D>, Error> {
        if let Some(e) = self.error {
            return Err(e);
        }
        let n = self
            .cursor
            .max(self.last_row + 1)
            .max(2)
            .next_power_of_two();
        // The column variables, in one numbering for every gate.
        let mut atoms = Vec::new();
        for kind in &self.kinds {
            for c in &kind.constraints {
                c.atoms(&mut atoms);
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
        let mut selectors = Vec::new();
        for (k, kind) in self.kinds.iter().enumerate() {
            let mut selector = vec![F::ZERO; n];
            for row in &kind.rows {
                selector[*row] = F::ONE;
            }
            selectors.push(selector);
            for c in &kind.constraints {
                let degree = c.degree();
                let gadget = kind.gadget;
                let polynomial = c
                    .to_sparse::<N, D>(&index)
                    .ok_or(Error::Degree { gadget, degree })?;
                gates.push(Gate::new(Some(k), polynomial));
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
        Ok(Compiled {
            circuit,
            columns,
            public,
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
        for column in [source.0, target.0] {
            if column >= self.layout.permuted {
                self.fail(Error::NotPermuted { gadget, column });
            }
        }
        self.copies.push((source, target));
        self.write(target, value.value)
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
        let Some(anchor) = value.expr.min_row() else {
            // A constraint on constants only.
            if value.expr != Expr::Constant(F::ZERO) {
                let row = self.cursor;
                self.fail(Error::Unsatisfied { gadget, row });
            }
            return;
        };
        if self.with_witness && value.value != Some(F::ZERO) {
            self.fail(Error::Unsatisfied {
                gadget,
                row: anchor,
            });
        }
        let Some(frame) = self.frames.last() else {
            return;
        };
        let offset = anchor.cast_signed() - frame.base.cast_signed();
        let id = frame.id;
        let relative = value.expr.relative_to(anchor);
        let kind = match self.kind_index.get(&(gadget, offset)) {
            Some(k) => *k,
            None => {
                self.kinds.push(Kind {
                    gadget,
                    constraints: Vec::new(),
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
            k.constraints.push(relative);
        } else if k.constraints.get(count) != Some(&relative) {
            self.fail(Error::Inconsistent { gadget });
        }
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
