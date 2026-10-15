//! A typed API to write vanilla PlonK circuits.
//!
//! Circuit code reads like the computation it constrains. Every operation
//! returns a [`Wire`] carrying its value (for the witness) and its position
//! (for the constraints):
//!
//! # Arithmetization
//!
//! The target is vanilla PlonK: three wire columns `a, b, c`, five selectors
//! and the gate
//!
//! ```text
//! q_L a + q_R b + q_O c + q_M a b + q_C + PI = 0
//! ```
//!
//! on every row, plus copy constraints between cells, enforced by a
//! permutation `sigma` of the `3n` cells. Each operation of the builder
//! emits one row; its inputs are tied to the cells that produced them by
//! copy constraints.
//!
//! # Branding
//!
//! A wire belongs to the builder that created it. This is enforced by the
//! type checker with a *generative* lifetime brand (the technique of
//! `GhostCell`, Yanovski et al., ICFP 2021): [`build`] runs its closure with
//! a builder whose lifetime `'id` is fresh and invariant, so wires of two
//! circuits have incomparable types, and no wire can escape the closure.
//!
//! ```compile_fail
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_frontend::vanilla::build;
//! build::<F, _, _>(|outer| {
//!     let x = outer.witness(F::new(1));
//!     // A wire of the outer circuit cannot be used in another one.
//!     build::<F, _, _>(|inner| {
//!         inner.mul(&x, &x);
//!     });
//! });
//! ```
//!
//! ```compile_fail
//! use arrabbiata_field::Goldilocks as F;
//! use arrabbiata_frontend::vanilla::build;
//! // A wire cannot outlive its builder.
//! let (_, _, leaked) = build::<F, _, _>(|b| b.witness(F::new(1)));
//! ```

use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use arrabbiata_field::Field;

/// The invariant lifetime brand. `fn(&'id ()) -> &'id ()` is invariant in
/// `'id`, so two brands unify only if they are the same lifetime.
type Brand<'id> = PhantomData<fn(&'id ()) -> &'id ()>;

/// A wire column of the vanilla PlonK arithmetization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Column {
    /// The left input `a`.
    A,
    /// The right input `b`.
    B,
    /// The output `c`.
    C,
}

impl Column {
    /// All columns, in order.
    pub const ALL: [Self; 3] = [Self::A, Self::B, Self::C];

    /// The index of the column: 0, 1 or 2.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
        }
    }
}

/// A cell of the trace: a column at a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    /// The column.
    pub column: Column,
    /// The row.
    pub row: usize,
}

impl Cell {
    /// The position of the cell in the flattened trace `a || b || c` of
    /// `num_rows` rows: `column * num_rows + row`.
    #[must_use]
    pub const fn index(self, num_rows: usize) -> usize {
        self.column.index() * num_rows + self.row
    }
}

/// A value in a circuit under construction: its value, and the cell that
/// holds it. Branded by its builder (see the crate documentation).
#[derive(Clone, Copy, Debug)]
pub struct Wire<'id, F> {
    value: F,
    cell: Cell,
    brand: Brand<'id>,
}

impl<F> Wire<'_, F>
where
    F: Field,
{
    /// The value the wire carries.
    #[must_use]
    pub const fn value(&self) -> F {
        self.value
    }

    /// The cell holding the value.
    #[must_use]
    pub const fn cell(&self) -> Cell {
        self.cell
    }
}

/// The selectors of one row: `q_L a + q_R b + q_O c + q_M a b + q_C`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gate<F> {
    /// Left selector.
    pub q_l: F,
    /// Right selector.
    pub q_r: F,
    /// Output selector.
    pub q_o: F,
    /// Multiplication selector.
    pub q_m: F,
    /// Constant selector.
    pub q_c: F,
}

impl<F> Gate<F>
where
    F: Field,
{
    /// The gate with every selector zero: no constraint on the row.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            q_l: F::ZERO,
            q_r: F::ZERO,
            q_o: F::ZERO,
            q_m: F::ZERO,
            q_c: F::ZERO,
        }
    }

    /// `q_L a + q_R b + q_O c + q_M a b + q_C`.
    #[must_use]
    pub fn evaluate(&self, a: F, b: F, c: F) -> F {
        self.q_l * a + self.q_r * b + self.q_o * c + self.q_m * a * b + self.q_c
    }
}

/// A circuit under construction. Obtained through [`build`].
#[derive(Debug)]
pub struct Builder<'id, F> {
    gates: Vec<Gate<F>>,
    rows: Vec<[F; 3]>,
    copies: Vec<(Cell, Cell)>,
    public: Vec<(usize, F)>,
    brand: Brand<'id>,
}

/// Run `f` on a fresh builder, and return the circuit, its witness and the
/// result of `f`.
///
/// The `for<'id>` bound makes the brand of the builder fresh: see the crate
/// documentation.
pub fn build<F, R, Body>(f: Body) -> (Circuit<F>, Witness<F>, R)
where
    F: Field,
    Body: for<'id> FnOnce(&mut Builder<'id, F>) -> R,
{
    let mut builder = Builder {
        gates: Vec::new(),
        rows: Vec::new(),
        copies: Vec::new(),
        public: Vec::new(),
        brand: PhantomData,
    };
    let result = f(&mut builder);
    let (circuit, witness) = builder.finish();
    (circuit, witness, result)
}

impl<'id, F> Builder<'id, F>
where
    F: Field,
{
    fn row(&mut self, gate: Gate<F>, values: [F; 3]) -> usize {
        self.gates.push(gate);
        self.rows.push(values);
        self.rows.len() - 1
    }

    fn wire(&self, value: F, column: Column, row: usize) -> Wire<'id, F> {
        Wire {
            value,
            cell: Cell { column, row },
            brand: PhantomData,
        }
    }

    /// A gate with inputs `x` (in column `a`) and `y` (in column `b`) and
    /// output `out` (in column `c`).
    fn binary(
        &mut self,
        gate: Gate<F>,
        x: &Wire<'id, F>,
        y: &Wire<'id, F>,
        out: F,
    ) -> Wire<'id, F> {
        let row = self.row(gate, [x.value, y.value, out]);
        self.copies.push((
            x.cell,
            Cell {
                column: Column::A,
                row,
            },
        ));
        self.copies.push((
            y.cell,
            Cell {
                column: Column::B,
                row,
            },
        ));
        self.wire(out, Column::C, row)
    }

    /// A private input.
    pub fn witness(&mut self, value: F) -> Wire<'id, F> {
        let row = self.row(Gate::empty(), [value, F::ZERO, F::ZERO]);
        self.wire(value, Column::A, row)
    }

    /// A public input: part of the statement, given to the verifier.
    pub fn public_input(&mut self, value: F) -> Wire<'id, F> {
        // q_L a + PI = 0 with PI = -value on this row.
        let gate = Gate {
            q_l: F::ONE,
            ..Gate::empty()
        };
        let row = self.row(gate, [value, F::ZERO, F::ZERO]);
        self.public.push((row, value));
        self.wire(value, Column::A, row)
    }

    /// The constant `c`.
    pub fn constant(&mut self, c: F) -> Wire<'id, F> {
        let gate = Gate {
            q_l: F::ONE,
            q_c: -c,
            ..Gate::empty()
        };
        let row = self.row(gate, [c, F::ZERO, F::ZERO]);
        self.wire(c, Column::A, row)
    }

    /// `x + y`.
    pub fn add(&mut self, x: &Wire<'id, F>, y: &Wire<'id, F>) -> Wire<'id, F> {
        let gate = Gate {
            q_l: F::ONE,
            q_r: F::ONE,
            q_o: -F::ONE,
            ..Gate::empty()
        };
        self.binary(gate, x, y, x.value + y.value)
    }

    /// `x - y`.
    pub fn sub(&mut self, x: &Wire<'id, F>, y: &Wire<'id, F>) -> Wire<'id, F> {
        let gate = Gate {
            q_l: F::ONE,
            q_r: -F::ONE,
            q_o: -F::ONE,
            ..Gate::empty()
        };
        self.binary(gate, x, y, x.value - y.value)
    }

    /// `x * y`.
    pub fn mul(&mut self, x: &Wire<'id, F>, y: &Wire<'id, F>) -> Wire<'id, F> {
        let gate = Gate {
            q_m: F::ONE,
            q_o: -F::ONE,
            ..Gate::empty()
        };
        self.binary(gate, x, y, x.value * y.value)
    }

    /// `x + c` for a constant `c`.
    pub fn add_constant(&mut self, x: &Wire<'id, F>, c: F) -> Wire<'id, F> {
        let gate = Gate {
            q_l: F::ONE,
            q_o: -F::ONE,
            q_c: c,
            ..Gate::empty()
        };
        let out = x.value + c;
        let row = self.row(gate, [x.value, F::ZERO, out]);
        self.copies.push((
            x.cell,
            Cell {
                column: Column::A,
                row,
            },
        ));
        self.wire(out, Column::C, row)
    }

    /// `c * x` for a constant `c`.
    pub fn scale(&mut self, x: &Wire<'id, F>, c: F) -> Wire<'id, F> {
        let gate = Gate {
            q_l: c,
            q_o: -F::ONE,
            ..Gate::empty()
        };
        let out = c * x.value;
        let row = self.row(gate, [x.value, F::ZERO, out]);
        self.copies.push((
            x.cell,
            Cell {
                column: Column::A,
                row,
            },
        ));
        self.wire(out, Column::C, row)
    }

    /// Constrain `x = y`, by a copy constraint (no row).
    pub fn assert_equal(&mut self, x: &Wire<'id, F>, y: &Wire<'id, F>) {
        self.copies.push((x.cell, y.cell));
    }

    /// Constrain `x = 0`.
    pub fn assert_zero(&mut self, x: &Wire<'id, F>) {
        let gate = Gate {
            q_l: F::ONE,
            ..Gate::empty()
        };
        let row = self.row(gate, [x.value, F::ZERO, F::ZERO]);
        self.copies.push((
            x.cell,
            Cell {
                column: Column::A,
                row,
            },
        ));
    }

    /// Pad to a power of two (at least 2) and compute the permutation.
    fn finish(self) -> (Circuit<F>, Witness<F>) {
        let mut num_rows = 2;
        while num_rows < self.rows.len() {
            num_rows *= 2;
        }
        let mut gates = self.gates;
        let mut wires = [
            vec![F::ZERO; num_rows],
            vec![F::ZERO; num_rows],
            vec![F::ZERO; num_rows],
        ];
        let mut row = 0;
        while row < self.rows.len() {
            let mut k = 0;
            while k < 3 {
                wires[k][row] = self.rows[row][k];
                k += 1;
            }
            row += 1;
        }
        while gates.len() < num_rows {
            gates.push(Gate::empty());
        }
        let sigma = permutation(&self.copies, num_rows);
        let mut public_rows = Vec::with_capacity(self.public.len());
        let mut public_values = Vec::with_capacity(self.public.len());
        let mut i = 0;
        while i < self.public.len() {
            public_rows.push(self.public[i].0);
            public_values.push(self.public[i].1);
            i += 1;
        }
        (
            Circuit {
                num_rows,
                gates,
                sigma,
                public_rows,
            },
            Witness {
                wires,
                public: public_values,
            },
        )
    }
}

/// The permutation of the `3n` cells whose cycles are the equivalence
/// classes of the copy constraints: each cell maps to the next cell of its
/// class, the last to the first.
fn permutation(copies: &[(Cell, Cell)], num_rows: usize) -> Vec<usize> {
    let size = 3 * num_rows;
    // Union-find over cell indices.
    let mut parent: Vec<usize> = (0..size).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (x, y) in copies {
        let rx = find(&mut parent, x.index(num_rows));
        let ry = find(&mut parent, y.index(num_rows));
        if rx != ry {
            parent[rx] = ry;
        }
    }
    // Group cells by class, in increasing order, and link each to the next.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); size];
    for cell in 0..size {
        let root = find(&mut parent, cell);
        members[root].push(cell);
    }
    let mut sigma: Vec<usize> = (0..size).collect();
    for class in &members {
        for (i, cell) in class.iter().enumerate() {
            sigma[*cell] = class[(i + 1) % class.len()];
        }
    }
    sigma
}

/// A vanilla PlonK circuit: the gates of every row, the copy permutation, and
/// the rows of the public inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuit<F> {
    num_rows: usize,
    gates: Vec<Gate<F>>,
    sigma: Vec<usize>,
    public_rows: Vec<usize>,
}

/// The assignment of a circuit: the three wire columns and the public input
/// values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness<F> {
    wires: [Vec<F>; 3],
    public: Vec<F>,
}

impl<F> Witness<F> {
    /// A witness given directly by its wire columns and public inputs, for
    /// witnesses computed outside the builder. Nothing is checked here;
    /// [`Circuit::is_satisfied`] and the prover check it against a circuit.
    #[must_use]
    pub const fn new(wires: [Vec<F>; 3], public: Vec<F>) -> Self {
        Self { wires, public }
    }

    /// The wire columns `a`, `b`, `c`.
    #[must_use]
    pub const fn wires(&self) -> &[Vec<F>; 3] {
        &self.wires
    }

    /// The public input values, in the order of [`Circuit::public_rows`].
    #[must_use]
    pub fn public_inputs(&self) -> &[F] {
        &self.public
    }
}

impl<F> Circuit<F>
where
    F: Field,
{
    /// The number of rows, a power of two.
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The gates, one per row.
    #[must_use]
    pub fn gates(&self) -> &[Gate<F>] {
        &self.gates
    }

    /// The copy permutation on cell indices (see [`Cell::index`]).
    #[must_use]
    pub fn sigma(&self) -> &[usize] {
        &self.sigma
    }

    /// The rows of the public inputs.
    #[must_use]
    pub fn public_rows(&self) -> &[usize] {
        &self.public_rows
    }

    /// The public input column `PI`: `-x_i` on the row of the `i`-th public
    /// input, zero elsewhere, so that a public-input row reads
    /// `a - x_i = 0`.
    #[must_use]
    pub fn public_input_column(&self, public_inputs: &[F]) -> Vec<F> {
        let mut column = vec![F::ZERO; self.num_rows];
        let mut i = 0;
        while i < self.public_rows.len() && i < public_inputs.len() {
            column[self.public_rows[i]] = -public_inputs[i];
            i += 1;
        }
        column
    }

    /// Whether `witness` satisfies every gate (with the public inputs) and
    /// every copy constraint.
    #[must_use]
    pub fn is_satisfied(&self, witness: &Witness<F>) -> bool {
        if self.public_rows.len() != witness.public.len() {
            return false;
        }
        let pi = self.public_input_column(&witness.public);
        let mut row = 0;
        while row < self.num_rows {
            let [a, b, c] = [&witness.wires[0], &witness.wires[1], &witness.wires[2]];
            if !(self.gates[row].evaluate(a[row], b[row], c[row]) + pi[row]).is_zero() {
                return false;
            }
            row += 1;
        }
        let mut cell = 0;
        while cell < 3 * self.num_rows {
            let target = self.sigma[cell];
            let here = witness.wires[cell / self.num_rows][cell % self.num_rows];
            let there = witness.wires[target / self.num_rows][target % self.num_rows];
            if here != there {
                return false;
            }
            cell += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use proptest::prelude::*;
    use std::{collections::BTreeSet, vec::Vec};

    use super::*;

    fn cell(index: usize, num_rows: usize) -> Cell {
        Cell {
            column: Column::ALL[index / num_rows],
            row: index % num_rows,
        }
    }

    /// The connected components of the copy graph, by breadth-first search:
    /// an oracle independent of the union-find in `permutation`.
    fn components(copies: &[(usize, usize)], size: usize) -> Vec<BTreeSet<usize>> {
        let mut seen = vec![false; size];
        let mut out = Vec::new();
        for start in 0..size {
            if seen[start] {
                continue;
            }
            let mut component = BTreeSet::new();
            let mut queue = vec![start];
            seen[start] = true;
            while let Some(x) = queue.pop() {
                component.insert(x);
                for &(a, b) in copies {
                    for (from, to) in [(a, b), (b, a)] {
                        if from == x && !seen[to] {
                            seen[to] = true;
                            queue.push(to);
                        }
                    }
                }
            }
            out.push(component);
        }
        out
    }

    proptest! {
        /// sigma is a permutation, and its cycles are exactly the connected
        /// components of the copy constraints.
        #[test]
        fn sigma_cycles_are_copy_classes(
            log_rows in 1u32..4,
            raw in prop::collection::vec((any::<usize>(), any::<usize>()), 0..20),
        ) {
            let n = 1usize << log_rows;
            let size = 3 * n;
            let copies: Vec<(usize, usize)> = raw.iter().map(|(a, b)| (a % size, b % size)).collect();
            let cells: Vec<(Cell, Cell)> =
                copies.iter().map(|&(a, b)| (cell(a, n), cell(b, n))).collect();
            let sigma = permutation(&cells, n);

            // A bijection of 0..3n.
            let image: BTreeSet<usize> = sigma.iter().copied().collect();
            prop_assert_eq!(image.len(), size);
            prop_assert!(sigma.iter().all(|&s| s < size));

            // Each cycle is a component, and each component a cycle.
            for component in components(&copies, size) {
                let start = *component.iter().next().unwrap();
                let mut cycle = BTreeSet::new();
                let mut x = start;
                loop {
                    cycle.insert(x);
                    x = sigma[x];
                    if x == start {
                        break;
                    }
                }
                prop_assert_eq!(cycle, component);
            }
        }
    }
}
