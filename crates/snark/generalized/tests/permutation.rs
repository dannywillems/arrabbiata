//! The permutation argument on some columns only: a sum of 4-bit numbers.
//!
//! Each row `i` adds `b_i` to a running sum, `c_i = a_i + b_i`, and checks
//! that `b_i` is a 4-bit number: four helper columns hold its bits,
//! `b_i = sum_j 2^j bit_(i,j)` and `bit (bit - 1) = 0`. The running sum is
//! carried from row to row by copies `c_i -> a_(i+1)`; the first `a` is
//! bound to the public input `0` and the last to the claimed total. Only
//! the columns `a, b, c` carry copies, so only they are permuted: the bits
//! are constrained by gates alone. This is how Kimchi uses its 15 columns,
//! 7 of which are permuted.
//!
//! - the honest sum is proved, with the permutation on 3 columns, and the
//!   quotient is then smaller than with all 7 permuted;
//! - breaking one copy (keeping every gate satisfied) is rejected;
//! - restricting the permutation does not change which witnesses satisfy
//!   the circuit, as long as the copies are between the permuted columns.

use arrabbiata_field::Field;
use arrabbiata_generalized_plonk::{
    Circuit, Error, Gate, Variable, max_coefficients, prove, quotient_factor, setup, verify,
};
use arrabbiata_ipa::Ipa;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_pasta::{Fq as F, Pallas};
use arrabbiata_permutation::Permutation;
use arrabbiata_poly::Rotation;
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};
use proptest::prelude::*;

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

const ROWS: usize = 8;
/// The additions: every row but the last, which holds the total.
const TERMS: usize = ROWS - 1;
const BITS: usize = 4;
/// `a, b, c` and the bits of `b`.
const COLUMNS: usize = 3 + BITS;

/// The variables: every column on the current row, in column order.
fn variables() -> Vec<Variable> {
    (0..COLUMNS)
        .map(|column| Variable::Witness {
            column,
            rotation: Rotation::CURR,
        })
        .collect()
}

fn var(i: usize) -> Sparse<F, COLUMNS, 2> {
    Sparse::variable(Var::try_new(i).unwrap()).unwrap()
}

/// The gates, all on the selector of the addition rows: `c = a + b`,
/// `b = sum_j 2^j bit_j`, and each bit boolean.
fn gates() -> Vec<Gate<F, COLUMNS, 2>> {
    let on_additions = Some(0);
    let (a, b, c) = (var(0), var(1), var(2));
    let mut gates = vec![Gate::new(on_additions, c - a - b.clone())];
    let mut decomposition = b;
    for j in 0..BITS {
        let bit = var(3 + j);
        let weight = F::from_u64(1 << j);
        decomposition = decomposition - bit.scale(weight);
        let boolean = bit.checked_mul(&(bit.clone() - Sparse::one())).unwrap();
        gates.push(Gate::new(on_additions, boolean));
    }
    gates.push(Gate::new(on_additions, decomposition));
    gates
}

/// The copies `c_i -> a_(i+1)`, as cells of the columns `a, b, c`.
fn running_sum_copies() -> Vec<(usize, usize)> {
    let a = 0;
    let c = 2;
    (0..TERMS)
        .map(|i| (c * ROWS + i, a * ROWS + i + 1))
        .collect()
}

/// The circuit, its copies on the columns `permuted`.
fn summation(permuted: &[usize]) -> Circuit<F, COLUMNS, 2> {
    let mut additions = vec![F::ONE; ROWS];
    additions[ROWS - 1] = F::ZERO;
    let num_rows = ROWS;
    let witness_columns = COLUMNS;
    let fixed = vec![];
    let selectors = vec![additions];
    let variables = variables();
    let gates = gates();
    let no_copies = (0..COLUMNS * ROWS).collect();
    // Public input 0 is the start (row 0), public input 1 the total.
    let public_inputs = 2;
    let bindings = vec![(0, 0), (1, ROWS - 1)];
    let circuit = Circuit::new(
        num_rows,
        witness_columns,
        fixed,
        selectors,
        variables,
        gates,
        no_copies,
        public_inputs,
        bindings,
    )
    .unwrap();
    // The copies only involve a and c, which are the first columns in
    // `permuted` (it starts with 0, 1, 2), so their cell indices do not move.
    let copies = Permutation::from_copies(permuted.len(), ROWS, &running_sum_copies()).unwrap();
    circuit
        .with_permutation(permuted.to_vec(), copies.sigma().to_vec())
        .unwrap()
}

/// The witness of the sum of `terms`, and its public inputs `(0, total)`.
fn witness(terms: &[u64; TERMS]) -> (Vec<Vec<F>>, Vec<F>) {
    let mut columns = vec![vec![F::ZERO; ROWS]; COLUMNS];
    let mut total = 0u64;
    for (i, term) in terms.iter().enumerate() {
        columns[0][i] = F::from_u64(total);
        columns[1][i] = F::from_u64(*term);
        total += term;
        columns[2][i] = F::from_u64(total);
        for j in 0..BITS {
            columns[3 + j][i] = F::from_u64((term >> j) & 1);
        }
    }
    columns[0][ROWS - 1] = F::from_u64(total);
    (columns, vec![F::ZERO, F::from_u64(total)])
}

fn ipa(size: usize) -> Ipa<Pallas, PoseidonSponge<F>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript)
}

fn proves(c: &Circuit<F, COLUMNS, 2>, columns: &[Vec<F>], public: &[F]) -> Result<(), Error> {
    let pcs = ipa(max_coefficients(c));
    let (pk, vk) = setup(c, &pcs)?;
    let (proof, _) = prove(&pk, &pcs, vesta_sponge(PROOF_DOMAIN), public, columns)?;
    verify(&vk, &pcs, vesta_sponge(PROOF_DOMAIN), public, proof).map(|_| ())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// The sum is proved with the copies on `a, b, c` only, and the quotient
    /// is smaller than with every column permuted.
    #[test]
    fn sums_are_proved(terms in prop::array::uniform7(0u64..16)) {
        let sparse = summation(&[0, 1, 2]);
        let dense = summation(&(0..COLUMNS).collect::<Vec<_>>());
        let (columns, public) = witness(&terms);
        prop_assert_eq!(proves(&sparse, &columns, &public), Ok(()));
        prop_assert!(quotient_factor(&sparse) < quotient_factor(&dense));
        prop_assert_eq!(quotient_factor(&sparse), 4);
    }

    /// One copy broken, every gate still satisfied: `a_(i+1)` and `c_(i+1)`
    /// both shifted, so row `i + 1` still adds up but no longer continues
    /// row `i`.
    #[test]
    fn a_broken_copy_is_rejected(terms in prop::array::uniform7(0u64..16), at in 0usize..TERMS - 1, delta in 1u64..100) {
        let c = summation(&[0, 1, 2]);
        let (mut columns, public) = witness(&terms);
        let shift = F::from_u64(delta);
        // The running sum a from row at + 1 on (the total included), and c
        // on the addition rows after at.
        for a in &mut columns[0][at + 1..] {
            *a = *a + shift;
        }
        for c in &mut columns[2][at + 1..TERMS] {
            *c = *c + shift;
        }
        // The total moved with it.
        let public = [public[0], public[1] + shift];
        prop_assert!(!c.is_satisfied(&columns, &public));
        prop_assert_eq!(proves(&c, &columns, &public), Err(Error::Unsatisfied));
    }

    /// Which witnesses satisfy the circuit does not depend on how many
    /// columns are permuted, when the copies are between permuted columns.
    #[test]
    fn restriction_preserves_satisfaction(terms in prop::array::uniform7(0u64..16), column in 0usize..COLUMNS, row in 0usize..ROWS, delta in 0u64..3) {
        let sparse = summation(&[0, 1, 2]);
        let dense = summation(&(0..COLUMNS).collect::<Vec<_>>());
        let (mut columns, public) = witness(&terms);
        columns[column][row] = columns[column][row] + F::from_u64(delta);
        prop_assert_eq!(sparse.is_satisfied(&columns, &public), dense.is_satisfied(&columns, &public));
    }
}
