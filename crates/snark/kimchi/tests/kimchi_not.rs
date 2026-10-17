//! Kimchi's `kimchi/src/tests/not.rs`, test by test, with the same inputs
//! and outputs. The two circuits are typed gadgets: the all-ones public
//! input, then either its XOR with the word (the checked NOT) or Generic
//! rows `ones - x` (the unchecked NOT). Their rows are the symbolic run,
//! their witness the numeric run, both checked against Kimchi's run of the
//! same test (`gate_tests/`), with every verdict.

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem},
    gates::Gates,
};
use arrabbiata_kimchi_gadgets::{
    generic::{Generic, GenericPair, GenericSpec, PublicInput},
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{
    Case, ForBits, bit_length, case, debug, for_bits, kimchi_rows, kimchi_witness, pallas_gates,
    prove_and_verify, rows, vesta_gates,
};

/// `2^bits - 1`.
fn all_ones<F: Field>(bits: usize) -> F {
    F::from_u64(2).pow(bits as u64) - F::ONE
}

/// The NOT half-gate, `l - r = o`.
fn negation<F: Field>() -> GenericSpec<F> {
    GenericSpec::Add {
        left: None,
        right: Some(-F::ONE),
        output: None,
    }
}

/// The checked NOT of a word of `B` bits: the all-ones public input on
/// row 0, the word in column 0 of row 1, then their XOR (Kimchi's
/// `extend_not_gadget_checked_length` after its public row). Input
/// `[ones, x]`.
#[derive(Clone, Debug)]
struct NotXor<F, const B: usize>(Xor<F, B>);

impl<F: PrimeField, const B: usize> TypedGadget<F> for NotXor<F, B> {
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "not-xor";
    const ROWS: usize = 1 + Xor::<F, B>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        [ones, x]: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let ones = env.call(&PublicInput, ones);
        let x = env.advice(Cell::new(0, 1), [&x], |[v]| v);
        env.call(&self.0, [x, ones])
    }
}

/// One row of NOTs: the word (or the two words) where the half-gates read
/// them, then our Generic gate on `(ones, x)`. Input `(ones, [x1, x2])`,
/// `x2` unused for a single NOT.
#[derive(Clone, Copy, Debug)]
struct NotRow {
    double: bool,
}

impl<F: PrimeField> TypedGadget<F> for NotRow {
    type Input = (Scalar, Array<Scalar, 2>);
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "not-row";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (ones, [x1, x2]): Of<Self::Input, E::Var>,
    ) -> Of<Array<Scalar, 2>, E::Var> {
        let x1 = env.advice(Cell::new(1, 0), [&x1], |[v]| v);
        if self.double {
            let x2 = env.advice(Cell::new(4, 0), [&x2], |[v]| v);
            let pair = GenericPair {
                first: negation(),
                second: negation(),
            };
            env.call(&pair, ([ones.clone(), x1], [ones, x2]))
        } else {
            let not = env.call(&Generic { spec: negation() }, [ones, x1]);
            [not.clone(), not]
        }
    }
}

/// The unchecked NOT of `N` words: the all-ones public input on row 0,
/// then their NOTs two per row (Kimchi's
/// `extend_not_gadget_unchecked_length` after its public row). Input
/// `(ones, xs)`.
#[derive(Clone, Copy, Debug)]
struct NotGnrc<const N: usize>;

impl<F: PrimeField, const N: usize> TypedGadget<F> for NotGnrc<N> {
    type Input = (Scalar, Array<Scalar, N>);
    type Output = Unit;
    const NAME: &'static str = "not-gnrc";
    const ROWS: usize = 1 + N.div_ceil(2);

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (ones, xs): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let ones = env.call(&PublicInput, ones);
        for pair in xs.chunks(2) {
            let double = pair.len() == 2;
            let second = pair.get(1).unwrap_or(&pair[0]).clone();
            env.call(
                &NotRow { double },
                (ones.clone(), [pair[0].clone(), second]),
            );
        }
    }
}

/// The checked NOT of `input` on words of `B` bits.
struct NotXorCase<F> {
    input: [F; 2],
}

impl<F: PrimeField> ForBits for NotXorCase<F> {
    type Output = (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]);

    fn call<const B: usize>(self) -> Self::Output {
        let gadget = NotXor(Xor::<F, B> { table: xor_table() });
        let rows = NotXor::<F, B>::ROWS;
        (
            kimchi_rows(&gadget, rows),
            kimchi_witness(&gadget, &self.input, rows),
        )
    }
}

/// The checked NOT of `x` on `bits` bits (at least `bitlen(x)`): rows and
/// witness (Kimchi's `create_not_witness_checked_length`).
fn checked<F: PrimeField>(x: F, bits: usize) -> (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]) {
    for_bits(
        bits,
        NotXorCase {
            input: [all_ones(bits), x],
        },
    )
}

/// The unchecked NOT of `xs` on `bits` bits: rows and witness (Kimchi's
/// `create_not_witness_unchecked_length`).
fn unchecked<F: PrimeField>(xs: &[F], bits: usize) -> (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]) {
    fn of<F: PrimeField, const N: usize>(
        xs: &[F],
        ones: F,
    ) -> (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]) {
        let input: [F; N] = xs.try_into().unwrap();
        let rows = <NotGnrc<N> as TypedGadget<F>>::ROWS;
        (
            kimchi_rows(&NotGnrc::<N>, rows),
            kimchi_witness(&NotGnrc::<N>, &(ones, input), rows),
        )
    }
    let ones = all_ones(bits);
    match xs.len() {
        1 => of::<F, 1>(xs, ones),
        2 => of::<F, 2>(xs, ones),
        5 => of::<F, 5>(xs, ones),
        n => panic!("no NOT of {n} words here"),
    }
}

/// Every row's verdict is Kimchi's.
fn verify_rows<F, G>(case: &Case, cs: &ConstraintSystem<F>, witness: &[Vec<F>; COLUMNS], gates: &G)
where
    F: PrimeField,
    G: Gates<F>,
{
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, witness, gates);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "row {row}"
        );
    }
}

/// Kimchi's `setup_not_xor`, its inputs from Kimchi's run, and with
/// `verify` its `test_not_xor`.
fn not_xor<F, G>(name: &str, gates: &G, verify: bool) -> ([Vec<F>; COLUMNS], ConstraintSystem<F>)
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let input: F = case.elements("in")[0];
    let bits = (case.number("given") == 1).then(|| case.number("bits"));
    let real = bit_length(input).max(bits.unwrap_or(0));
    let (circuit, witness) = checked(input, real);
    let cs = ConstraintSystem::create(circuit).public(1).build().unwrap();
    assert_eq!(rows(&witness), case.witnesses[0], "{name}");
    // Kimchi's check_not_xor: the output is the NOT on `real` bits.
    assert_eq!(witness[2][1], all_ones::<F>(real) - input);
    if verify {
        verify_rows(&case, &cs, &witness, gates);
    }
    (witness, cs)
}

/// Kimchi's `setup_not_gnrc`, and with `verify` its `test_not_gnrc`.
fn not_gnrc<F, G>(name: &str, gates: &G, verify: bool) -> ([Vec<F>; COLUMNS], ConstraintSystem<F>)
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let inputs: Vec<F> = case.elements("inputs");
    let bits = case.number("bits");
    let (circuit, witness) = unchecked(&inputs, bits);
    let cs = ConstraintSystem::create(circuit).public(1).build().unwrap();
    assert_eq!(rows(&witness), case.witnesses[0], "{name}");
    // Kimchi's check_not_gnrc.
    for (i, input) in inputs.iter().enumerate() {
        let (column, row) = if i % 2 == 0 {
            (2, 1 + i / 2)
        } else {
            (5, 1 + (i - 1) / 2)
        };
        assert_eq!(witness[column][row], all_ones::<F>(bits) - *input);
    }
    if verify {
        verify_rows(&case, &cs, &witness, gates);
    }
    (witness, cs)
}

#[test]
fn test_prove_and_verify_not_xor() {
    let case = case("test_prove_and_verify_not_xor");
    let bits = 64;
    let (gates, witness) = checked(case.elements::<Fp>("in")[0], bits);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![all_ones(bits)]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_prove_and_verify_five_not_gnrc() {
    let case = case("test_prove_and_verify_five_not_gnrc");
    let bits = 64;
    let (gates, witness) = unchecked(&case.elements::<Fp>("inputs"), bits);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, vec![all_ones(bits)]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_not_xor_all_crumb() {
    let gates = vesta_gates();
    for i in 0..16 {
        not_xor::<Fp, _>(&format!("test_not_xor_all_crumb {i} none"), &gates, true);
        for c in (4..=16).step_by(4) {
            not_xor::<Fp, _>(&format!("test_not_xor_all_crumb {i} {c}"), &gates, true);
        }
    }
}

#[test]
fn test_not_xor_crumbs_random() {
    let gates = vesta_gates();
    for i in 2..=7 {
        let bits = 1 << i;
        not_xor::<Fp, _>(
            &format!("test_not_xor_crumbs_random {bits} some"),
            &gates,
            true,
        );
        not_xor::<Fp, _>(
            &format!("test_not_xor_crumbs_random {bits} none"),
            &gates,
            true,
        );
    }
}

#[test]
fn test_not_xor_big_random() {
    not_xor::<Fp, _>("test_not_xor_big_random vesta", &vesta_gates(), true);
    not_xor::<Fq, _>("test_not_xor_big_random pallas", &pallas_gates(), true);
}

#[test]
fn test_not_gnrc_double() {
    not_gnrc::<Fp, _>("test_not_gnrc_double vesta", &vesta_gates(), true);
    not_gnrc::<Fq, _>("test_not_gnrc_double pallas", &pallas_gates(), true);
}

#[test]
fn test_not_gnrc_single() {
    not_gnrc::<Fp, _>("test_not_gnrc_single vesta", &vesta_gates(), true);
    not_gnrc::<Fq, _>("test_not_gnrc_single pallas", &pallas_gates(), true);
}

#[test]
fn test_not_gnrc_vector() {
    not_gnrc::<Fp, _>("test_not_gnrc_vector vesta", &vesta_gates(), true);
    not_gnrc::<Fq, _>("test_not_gnrc_vector pallas", &pallas_gates(), true);
}

#[test]
fn test_bad_not_gnrc() {
    let gates = vesta_gates();
    let case = case("test_bad_not_gnrc");
    let (mut witness, cs) = not_gnrc::<Fp, _>("test_bad_not_gnrc", &gates, false);
    witness[0][0] = witness[0][0] + Fp::ONE;
    let ours = cs.gates[0].verify_witness(0, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("step0"));
    witness[0][1] = witness[0][1] + Fp::ONE;
    let ours = cs.gates[1].verify(1, &witness, &gates, &[]);
    assert_eq!(debug(&ours), case.verdict("step1"));
}

#[test]
fn test_bad_not_xor() {
    let gates = vesta_gates();
    let case = case("test_bad_not_xor");
    let (mut witness, cs) = not_xor::<Fp, _>("test_bad_not_xor", &gates, false);
    witness[0][0] = witness[0][0] + Fp::ONE;
    let ours = cs.gates[0].verify_witness(0, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("step0"));
    witness[1][1] = witness[1][1] + Fp::ONE;
    let ours = cs.gates[1].verify_witness(1, &witness, &gates);
    assert_eq!(debug(&ours), case.verdict("step1"));
    witness[0][0] = Fp::ZERO;
    witness[1][1] = Fp::ZERO;
    for column in &mut witness[7..=10] {
        column[1] = Fp::ZERO;
    }
    let result = prove_and_verify(cs.gates.clone(), witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
