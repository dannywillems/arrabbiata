//! Tables and reads, and cells read before they are written.
//!
//! - a 4-bit range check by a read of a 16-row table: values in range
//!   satisfy the circuit, a value out of range is rejected on its row while
//!   the witness is computed, and changing a read cell breaks the compiled
//!   circuit; the reads of every instance share the gadget's selector, and
//!   a proof verifies;
//! - two tables under one name are rejected;
//! - a constraint over a cell another gadget writes later is checked once
//!   the witness is complete;
//! - the figures count the reads and the tables.

use arrabbiata_field::Field;
use arrabbiata_gadget::{
    Array, Cell, Env, Error, Layout, Of, Rom, Scalar, TypedGadget, Unit, figures, witness,
};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_pasta::{Fq as F, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

/// The table in fixed column 0; nothing else uses fixed columns.
const LAYOUT: Layout = Layout {
    witness: 2,
    fixed: 1,
    permuted: 2,
};
const N: usize = 6;
const D: usize = 2;

/// The numbers `0 .. 16`.
fn nibbles() -> Rom<F> {
    let rows = (0..16).map(|v| vec![F::from_u64(v)]).collect();
    let table_column = vec![0];
    Rom::new("nibbles", table_column, rows)
}

/// `x` in column 0, checked to be a nibble.
#[derive(Clone, Debug)]
struct Nibble {
    table: Rom<F>,
}

impl TypedGadget<F> for Nibble {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "nibble";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        env.read(&self.table, core::slice::from_ref(&x));
        x
    }
}

/// Four nibbles, their sum public.
#[derive(Clone, Debug)]
struct Nibbles {
    table: Rom<F>,
}

impl TypedGadget<F> for Nibbles {
    type Input = Array<Scalar, 4>;
    type Output = Scalar;
    const NAME: &'static str = "nibbles";
    const ROWS: usize = 5;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        input: Of<Array<Scalar, 4>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let gadget = Nibble {
            table: self.table.clone(),
        };
        let [a, b, c, d] = input.map(|v| env.call(&gadget, v));
        let sum = env.define(Cell::new(0, 4), a + b + c + d);
        env.public(&sum);
        sum
    }
}

fn arb_nibble() -> impl Strategy<Value = u64> {
    0u64..16
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn values_in_range_satisfy(values in prop::array::uniform4(arb_nibble())) {
        let gadget = Nibbles { table: nibbles() };
        let input = values.map(F::from_u64);
        let (compiled, sum) = witness::<F, _, N, D>(&gadget, LAYOUT, &input).unwrap();
        prop_assert_eq!(sum, F::from_u64(values.iter().sum()));
        prop_assert!(compiled.circuit.is_satisfied(&compiled.columns, &compiled.public));
        // A read cell out of range, with the sum moved along: only the
        // lookup catches it.
        let mut columns = compiled.columns.clone();
        columns[0][1] = F::from_u64(16 + values[1]);
        columns[0][4] = columns[0][4] + F::from_u64(16);
        let public = [compiled.public[0] + F::from_u64(16)];
        prop_assert!(!compiled.circuit.is_satisfied(&columns, &public));
    }

    #[test]
    fn values_out_of_range_are_rejected(position in 0usize..4, value in 16u64..1000) {
        let gadget = Nibbles { table: nibbles() };
        let mut input = [F::ONE; 4];
        input[position] = F::from_u64(value);
        let found = witness::<F, _, N, D>(&gadget, LAYOUT, &input).unwrap_err();
        prop_assert_eq!(found, Error::NotInTable { gadget: "nibble", row: position });
    }
}

#[test]
fn the_reads_share_the_gadgets_selector() {
    let gadget = Nibbles { table: nibbles() };
    let f = figures(&gadget, LAYOUT).unwrap();
    let [nibble] = f.gates_of("nibble")[..] else {
        panic!("one kind")
    };
    assert_eq!(nibble.constraints, 0);
    assert_eq!(nibble.reads, vec!["nibbles"]);
    assert_eq!(nibble.read_cells, vec![(0, 0)]);
    assert_eq!(nibble.instances, 4);
    assert_eq!(f.tables, vec![("nibbles", 16, 1)]);
    // The table needs 16 rows and plookup a free one: 32.
    assert_eq!(f.domain_size, 32);
}

fn ipa(size: usize) -> Ipa<Pallas, PoseidonSponge<F>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_range_check_is_proved() {
    let gadget = Nibbles { table: nibbles() };
    let input = [3, 15, 0, 9].map(F::from_u64);
    let (compiled, sum) = witness::<F, _, N, D>(&gadget, LAYOUT, &input).unwrap();
    let pcs = ipa(max_coefficients(&compiled.circuit));
    let (prover_key, verifier_key) = setup(&compiled.circuit, &pcs).unwrap();
    let (proof, _) = prove(
        &prover_key,
        &pcs,
        vesta_sponge(PROOF_DOMAIN),
        &compiled.public,
        &compiled.columns,
    )
    .unwrap();
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &[sum],
            proof.clone()
        )
        .is_ok()
    );
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &[sum + F::ONE],
            proof
        )
        .is_err()
    );
}

/// The same name with other rows.
#[derive(Clone, Debug)]
struct TwoTables;

impl TypedGadget<F> for TwoTables {
    type Input = Unit;
    type Output = Unit;
    const NAME: &'static str = "two-tables";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, (): Of<Unit, E::Var>) -> Of<Unit, E::Var> {
        let x = env.advice(Cell::new(0, 0), [], |[]| F::ZERO);
        env.read(&nibbles(), core::slice::from_ref(&x));
        let rows = vec![vec![F::ZERO]];
        let other = Rom::new("nibbles", vec![0], rows);
        env.read(&other, &[x]);
    }
}

#[test]
fn a_table_has_one_content() {
    let found = figures(&TwoTables, LAYOUT).unwrap_err();
    assert_eq!(found, Error::Table { name: "nibbles" });
}

/// A gadget constraining the next row's cell to its own, before the next
/// gadget writes it.
#[derive(Clone, Copy, Debug)]
struct AheadOf;

impl TypedGadget<F> for AheadOf {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "ahead";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&input], |[v]| v);
        let next = env.cell(Cell::new(1, 1));
        env.assert_equal(next, x.clone());
        x
    }
}

/// `AheadOf`, then a row writing `value` where it looks.
#[derive(Clone, Copy, Debug)]
struct Later {
    value: u64,
}

impl TypedGadget<F> for Later {
    type Input = Scalar;
    type Output = Unit;
    const NAME: &'static str = "later";
    const ROWS: usize = 2;

    fn synthesize<E: Env<F>>(&self, env: &mut E, input: Of<Scalar, E::Var>) -> Of<Unit, E::Var> {
        env.call(&AheadOf, input);
        let value = F::from_u64(self.value);
        env.advice(Cell::new(1, 1), [], |[]| value);
    }
}

#[test]
fn a_cell_written_later_is_checked_at_the_end() {
    let seven = F::from_u64(7);
    let (compiled, ()) = witness::<F, _, N, D>(&Later { value: 7 }, LAYOUT, &seven).unwrap();
    assert!(
        compiled
            .circuit
            .is_satisfied(&compiled.columns, &compiled.public)
    );
    let found = witness::<F, _, N, D>(&Later { value: 8 }, LAYOUT, &seven).unwrap_err();
    assert_eq!(
        found,
        Error::Unsatisfied {
            gadget: "ahead",
            row: 0
        }
    );
}
