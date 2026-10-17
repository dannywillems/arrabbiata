//! Kimchi's lookups (`kimchi/src/circuits/lookup`): the patterns, the
//! built-in tables, and the lookup index the constraint system carries.
//!
//! A gate's rows look up tuples of their cells in tables, by PATTERN
//! ([`LookupPattern`]): Xor16 rows look up four `(left, right, out)`
//! nybble triples in the XOR table, RangeCheck0, RangeCheck1 (both rows)
//! and Rot64 rows look up columns 3-6 in the 12-bit table, the second row
//! of ForeignFieldMul columns 7-10, and Lookup rows three `(index, value)`
//! pairs of columns `(1, 2), (3, 4), (5, 6)` in the table whose id is in
//! column 0.
//!
//! The index ([`LookupIndex`]) concatenates every table (the given ones,
//! then the built-in ones the patterns need, the range check before XOR,
//! then the runtime tables with a zero second column) into the columns of
//! one joint table, padded with zeros, with a column of table ids when
//! some table has a nonzero id, a selector per pattern used, and, with
//! runtime tables, a selector that is 0 exactly on the runtime rows and on
//! the zero-knowledge rows.

use alloc::{vec, vec::Vec};
use core::fmt;

use arrabbiata_field::PrimeField;

use crate::constraints::{CircuitGate, GateType, LookupTable, RuntimeTableCfg};

/// The id of the XOR table.
pub const XOR_TABLE_ID: i32 = 0;

/// The id of the 12-bit range-check table.
pub const RANGE_CHECK_TABLE_ID: i32 = 1;

/// The ways gates read tables, in Kimchi's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LookupPattern {
    /// Xor16: four nybble triples.
    Xor,
    /// Lookup: three `(index, value)` pairs in the table of column 0.
    Lookup,
    /// RangeCheck0, RangeCheck1, Rot64: columns 3-6 in the 12-bit table.
    RangeCheck,
    /// The second row of ForeignFieldMul: columns 7-10 in the 12-bit table.
    ForeignFieldMul,
}

/// The patterns, in Kimchi's order.
pub const PATTERNS: [LookupPattern; 4] = [
    LookupPattern::Xor,
    LookupPattern::Lookup,
    LookupPattern::RangeCheck,
    LookupPattern::ForeignFieldMul,
];

/// Which table a joint lookup reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupTableId {
    /// A fixed id.
    Constant(i32),
    /// The id in a witness column of the row.
    WitnessColumn(usize),
}

/// One lookup: a tuple of cells of the row (`(column, next row?)`) and the
/// table it must be a row of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointLookup {
    /// The table.
    pub table_id: LookupTableId,
    /// The cells, one per table column.
    pub entry: Vec<usize>,
}

impl LookupPattern {
    /// The pattern of a gate's current row (`next` false) or of the row
    /// after it (`next` true), Kimchi's `LookupPattern::from_gate`.
    #[must_use]
    pub const fn from_gate(typ: GateType, next: bool) -> Option<Self> {
        match (typ, next) {
            (GateType::Lookup, false) => Some(Self::Lookup),
            (GateType::RangeCheck0 | GateType::Rot64, false) | (GateType::RangeCheck1, _) => {
                Some(Self::RangeCheck)
            }
            (GateType::ForeignFieldMul, _) => Some(Self::ForeignFieldMul),
            (GateType::Xor16, false) => Some(Self::Xor),
            _ => None,
        }
    }

    /// The id of the built-in table the pattern reads, if any.
    #[must_use]
    pub const fn table_id(self) -> Option<i32> {
        match self {
            Self::Xor => Some(XOR_TABLE_ID),
            Self::Lookup => None,
            Self::RangeCheck | Self::ForeignFieldMul => Some(RANGE_CHECK_TABLE_ID),
        }
    }

    /// The lookups of a row of the pattern.
    #[must_use]
    pub const fn max_lookups_per_row(self) -> usize {
        match self {
            Self::Xor | Self::RangeCheck | Self::ForeignFieldMul => 4,
            Self::Lookup => 3,
        }
    }

    /// The width of its widest lookup.
    #[must_use]
    pub const fn max_joint_size(self) -> usize {
        match self {
            Self::Xor => 3,
            Self::Lookup => 2,
            Self::RangeCheck | Self::ForeignFieldMul => 1,
        }
    }

    /// The lookups of a row of the pattern, all on the row itself.
    #[must_use]
    pub fn lookups(self) -> Vec<JointLookup> {
        match self {
            Self::Xor => (0..4)
                .map(|i| JointLookup {
                    table_id: LookupTableId::Constant(XOR_TABLE_ID),
                    entry: vec![3 + i, 7 + i, 11 + i],
                })
                .collect(),
            Self::Lookup => (0..3)
                .map(|i| JointLookup {
                    table_id: LookupTableId::WitnessColumn(0),
                    entry: vec![2 * i + 1, 2 * i + 2],
                })
                .collect(),
            Self::RangeCheck => (3..=6)
                .map(|column| JointLookup {
                    table_id: LookupTableId::Constant(RANGE_CHECK_TABLE_ID),
                    entry: vec![column],
                })
                .collect(),
            Self::ForeignFieldMul => (7..=10)
                .map(|column| JointLookup {
                    table_id: LookupTableId::Constant(RANGE_CHECK_TABLE_ID),
                    entry: vec![column],
                })
                .collect(),
        }
    }
}

/// The XOR table of 4-bit words, `(i, j, i ^ j)`, in Kimchi's order
/// (reversed, so that its last row is the zero row).
#[must_use]
pub fn xor_table<F: PrimeField>() -> LookupTable<F> {
    let mut data = vec![Vec::new(), Vec::new(), Vec::new()];
    for i in 0..16u64 {
        for j in 0..16u64 {
            data[0].push(F::from_u64(i));
            data[1].push(F::from_u64(j));
            data[2].push(F::from_u64(i ^ j));
        }
    }
    for column in &mut data {
        column.reverse();
    }
    LookupTable {
        id: XOR_TABLE_ID,
        data,
    }
}

/// The 12-bit range-check table, `0, 1, ..., 4095`.
#[must_use]
pub fn range_check_table<F: PrimeField>() -> LookupTable<F> {
    LookupTable {
        id: RANGE_CHECK_TABLE_ID,
        data: vec![(0..1u64 << 12).map(F::from_u64).collect()],
    }
}

/// The patterns a circuit uses and what follows from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupInfo {
    /// The patterns used, in Kimchi's order.
    pub patterns: Vec<LookupPattern>,
    /// The most lookups of a row.
    pub max_per_row: usize,
    /// The widest lookup.
    pub max_joint_size: usize,
    /// Whether runtime tables are used.
    pub uses_runtime_tables: bool,
}

impl LookupInfo {
    /// The patterns of `gates`, or `None` when they look nothing up.
    #[must_use]
    pub fn from_gates<F>(gates: &[CircuitGate<F>], uses_runtime_tables: bool) -> Option<Self> {
        let patterns: Vec<LookupPattern> = PATTERNS
            .iter()
            .copied()
            .filter(|p| {
                gates.iter().any(|g| {
                    LookupPattern::from_gate(g.typ, false) == Some(*p)
                        || LookupPattern::from_gate(g.typ, true) == Some(*p)
                })
            })
            .collect();
        if patterns.is_empty() {
            return None;
        }
        Some(Self {
            max_per_row: patterns
                .iter()
                .map(|p| p.max_lookups_per_row())
                .max()
                .unwrap_or(0),
            max_joint_size: patterns
                .iter()
                .map(|p| p.max_joint_size())
                .max()
                .unwrap_or(0),
            patterns,
            uses_runtime_tables,
        })
    }

    /// Whether some lookup reads more than one column (Kimchi's
    /// `joint_lookup_used`).
    #[must_use]
    pub fn joint_lookup_used(&self) -> bool {
        self.patterns.iter().any(|p| p.max_joint_size() > 1)
    }

    /// The lookups of each of the `n` rows of `gates`.
    #[must_use]
    pub fn by_row<F>(&self, gates: &[CircuitGate<F>]) -> Vec<Vec<JointLookup>> {
        let mut rows = vec![Vec::new(); gates.len() + 1];
        for (i, gate) in gates.iter().enumerate() {
            if let Some(p) = LookupPattern::from_gate(gate.typ, false) {
                rows[i] = p.lookups();
            }
            if let Some(p) = LookupPattern::from_gate(gate.typ, true) {
                rows[i + 1] = p.lookups();
            }
        }
        rows.truncate(gates.len());
        rows
    }
}

/// Why the lookup index cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupError {
    /// A table has columns of different lengths.
    InconsistentTableLength,
    /// The joint table does not fit before the zero-knowledge rows.
    LookupTableTooLong {
        /// Its rows.
        length: usize,
        /// The rows allowed.
        maximum_allowed: usize,
    },
    /// A table with id 0 has no row of zeros.
    TableIdZeroMustHaveZeroEntry,
    /// Two tables share an id.
    LookupTableIdCollision,
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InconsistentTableLength => write!(f, "a table has columns of different lengths"),
            Self::LookupTableTooLong {
                length,
                maximum_allowed,
            } => write!(
                f,
                "the joint table has {length} rows, at most {maximum_allowed} fit"
            ),
            Self::TableIdZeroMustHaveZeroEntry => {
                write!(f, "the table with id 0 must have a row of zeros")
            }
            Self::LookupTableIdCollision => write!(f, "two tables share an id"),
        }
    }
}

/// The field element of a table id (a negative id is the negation).
#[must_use]
pub fn id_to_field<F: PrimeField>(id: i32) -> F {
    let magnitude = F::from_u64(u64::from(id.unsigned_abs()));
    if id < 0 { -magnitude } else { magnitude }
}

/// The lookup index of a constraint system (Kimchi's
/// `LookupConstraintSystem`), every column on `d1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupIndex<F> {
    /// The patterns.
    pub info: LookupInfo,
    /// The selector of each pattern used, in Kimchi's order.
    pub selectors: Vec<(LookupPattern, Vec<F>)>,
    /// The joint table's columns.
    pub table: Vec<Vec<F>>,
    /// The table ids, when some table has a nonzero id.
    pub table_ids: Option<Vec<F>>,
    /// With runtime tables, 0 on their rows and on the zero-knowledge rows.
    pub runtime_selector: Option<Vec<F>>,
    /// The first row of the runtime tables in the joint table.
    pub runtime_table_offset: Option<usize>,
    /// The runtime tables.
    pub runtime_tables: Option<Vec<RuntimeTableCfg<F>>>,
}

impl<F> LookupIndex<F>
where
    F: PrimeField,
{
    /// The index of the padded `gates` on `n` rows, or `None` when nothing
    /// is looked up.
    ///
    /// # Errors
    ///
    /// A [`LookupError`] for inconsistent, colliding or too long tables.
    pub fn create(
        gates: &[CircuitGate<F>],
        fixed_tables: &[LookupTable<F>],
        runtime_tables: Option<&[RuntimeTableCfg<F>]>,
        n: usize,
        zk_rows: usize,
    ) -> Result<Option<Self>, LookupError> {
        let Some(info) = LookupInfo::from_gates(gates, runtime_tables.is_some()) else {
            return Ok(None);
        };
        let max_num_entries = n - zk_rows - 1;
        let mut selectors: Vec<(LookupPattern, Vec<F>)> = info
            .patterns
            .iter()
            .map(|p| (*p, vec![F::ZERO; n]))
            .collect();
        let mut set = |p: LookupPattern, row: usize| {
            if let Some((_, s)) = selectors.iter_mut().find(|(q, _)| *q == p)
                && row < n
            {
                s[row] = F::ONE;
            }
        };
        let mut gate_tables: Vec<i32> = Vec::new();
        for (i, gate) in gates.iter().enumerate().take(n) {
            for (next, row) in [(false, i), (true, i + 1)] {
                if let Some(p) = LookupPattern::from_gate(gate.typ, next) {
                    set(p, row);
                    if let Some(id) = p.table_id()
                        && !gate_tables.contains(&id)
                    {
                        gate_tables.push(id);
                    }
                }
            }
        }
        // Kimchi collects them in a BTreeSet whose order puts the range
        // check before XOR ("for backwards compatibility",
        // `GateLookupTable`'s `Ord`).
        gate_tables.sort_unstable_by_key(|id| u8::from(*id == XOR_TABLE_ID));
        let mut tables: Vec<LookupTable<F>> = fixed_tables.to_vec();
        for id in gate_tables {
            tables.push(if id == XOR_TABLE_ID {
                xor_table()
            } else {
                range_check_table()
            });
        }
        let mut ids: Vec<i32> = tables.iter().map(|t| t.id).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1]) {
            return Err(LookupError::LookupTableIdCollision);
        }
        let length = |t: &LookupTable<F>| t.data.first().map_or(0, Vec::len);
        let mut has_table_id_0 = false;
        let (runtime_table_offset, runtime_selector) = match runtime_tables {
            None => (None, None),
            Some(runtime) => {
                let mut runtime_ids: Vec<i32> = runtime.iter().map(|t| t.id).collect();
                runtime_ids.sort_unstable();
                if runtime_ids.windows(2).any(|w| w[0] == w[1]) {
                    return Err(LookupError::LookupTableIdCollision);
                }
                let offset: usize = tables.iter().map(length).sum();
                let runtime_len: usize = runtime.iter().map(|t| t.first_column.len()).sum();
                let mut selector = vec![F::ONE; n];
                for value in selector.iter_mut().skip(offset).take(runtime_len) {
                    *value = F::ZERO;
                }
                for value in selector.iter_mut().rev().take(zk_rows) {
                    *value = F::ZERO;
                }
                for table in runtime {
                    if table.id == 0 {
                        has_table_id_0 = true;
                    }
                    let placeholders = vec![F::ZERO; table.first_column.len()];
                    tables.push(LookupTable {
                        id: table.id,
                        data: vec![table.first_column.clone(), placeholders],
                    });
                }
                (Some(offset), Some(selector))
            }
        };
        let width = tables
            .iter()
            .map(|t| t.data.len())
            .max()
            .unwrap_or(0)
            .max(info.max_joint_size);
        let mut columns: Vec<Vec<F>> = vec![Vec::new(); width];
        let mut table_ids: Vec<F> = Vec::new();
        let mut non_zero_table_id = false;
        let mut has_table_id_0_with_zero_entry = false;
        for table in &tables {
            let rows = length(table);
            if table.id == 0 {
                has_table_id_0 = true;
                let zero_row = (0..rows).any(|r| table.data.iter().all(|c| c[r].is_zero()));
                has_table_id_0_with_zero_entry |= zero_row;
            } else {
                non_zero_table_id = true;
            }
            table_ids.extend(core::iter::repeat_n(id_to_field::<F>(table.id), rows));
            for (i, column) in table.data.iter().enumerate() {
                if column.len() != rows {
                    return Err(LookupError::InconsistentTableLength);
                }
                columns[i].extend(column);
            }
            for column in columns.iter_mut().skip(table.data.len()) {
                column.extend(core::iter::repeat_n(F::ZERO, rows));
            }
        }
        if has_table_id_0 && !has_table_id_0_with_zero_entry {
            return Err(LookupError::TableIdZeroMustHaveZeroEntry);
        }
        if columns[0].len() >= max_num_entries {
            return Err(LookupError::LookupTableTooLong {
                length: columns[0].len(),
                maximum_allowed: max_num_entries - 1,
            });
        }
        for column in &mut columns {
            column.resize(n, F::ZERO);
        }
        table_ids.resize(n, F::ZERO);
        Ok(Some(Self {
            info,
            selectors,
            table: columns,
            table_ids: non_zero_table_id.then_some(table_ids),
            runtime_selector,
            runtime_table_offset,
            runtime_tables: runtime_tables.map(<[RuntimeTableCfg<F>]>::to_vec),
        }))
    }

    /// The selector of `pattern`, if used.
    #[must_use]
    pub fn selector(&self, pattern: LookupPattern) -> Option<&[F]> {
        self.selectors
            .iter()
            .find(|(p, _)| *p == pattern)
            .map(|(_, s)| s.as_slice())
    }
}
