//! Kimchi's `kimchi/src/tests/xor.rs`, test by test, with the same inputs
//! and outputs. Each test's circuit is a typed gadget (our XOR, its inputs
//! placed or public): its rows are the symbolic run, its witness the
//! numeric run, and both are checked against Kimchi's run of the same test
//! (`gate_tests/`), with every verdict.

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField, TwoAdicField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Placed, Scalar, TypedGadget};
use arrabbiata_kimchi::{
    constraints::{COLUMNS, CircuitGate, ConstraintSystem, GateType, Wire},
    gates::Gates,
};
use arrabbiata_kimchi_gadgets::{
    generic::PublicInput,
    xor::{Xor, xor_table},
};
use arrabbiata_pasta::{Fp, Fq};
use gate_tests::{
    Case, ForBits, case, debug, for_bits, kimchi_rows, kimchi_witness, pallas_gates,
    prove_and_verify, rows, vesta_gates,
};

/// The XOR of two words of `B` bits, their cells in columns 0 and 1 of the
/// first row: Kimchi's `create_xor_gadget`.
fn xor<F: PrimeField, const B: usize>() -> Placed<Xor<F, B>> {
    Placed {
        gadget: Xor { table: xor_table() },
        cells: vec![Cell::new(0, 0), Cell::new(1, 0)],
    }
}

/// Two public inputs on rows 0 and 1, then their XOR: the circuit of
/// Kimchi's `test_extend_xor` and `test_xor_finalization`.
#[derive(Clone, Debug)]
struct PublicXor<F, const B: usize>(Xor<F, B>);

impl<F: PrimeField, const B: usize> TypedGadget<F> for PublicXor<F, B> {
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "public-xor";
    const ROWS: usize = 2 + Xor::<F, B>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        words: Of<Array<Scalar, 2>, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let [a, b] = words;
        let a = env.call(&PublicInput, a);
        let b = env.call(&PublicInput, b);
        env.call(&self.0, [a, b])
    }
}

/// The rows and witness of the XOR of `input` on `bits` bits.
struct XorCase<F> {
    input: [F; 2],
    public: bool,
}

impl<F: PrimeField> ForBits for XorCase<F> {
    type Output = (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS]);

    fn call<const B: usize>(self) -> Self::Output {
        if self.public {
            let gadget = PublicXor(Xor::<F, B> { table: xor_table() });
            let rows = PublicXor::<F, B>::ROWS;
            (
                kimchi_rows(&gadget, rows),
                kimchi_witness(&gadget, &self.input, rows),
            )
        } else {
            let gadget = xor::<F, B>();
            let rows = Xor::<F, B>::ROWS;
            (
                kimchi_rows(&gadget, rows),
                kimchi_witness(&gadget, &self.input, rows),
            )
        }
    }
}

/// Kimchi's `setup_xor` on the inputs of `case`: the rows and the witness,
/// which is Kimchi's.
fn setup<F>(case: &Case, public: bool) -> (Vec<CircuitGate<F>>, [Vec<F>; COLUMNS])
where
    F: PrimeField + TwoAdicField,
{
    let input: [F; 2] = case.elements("in").try_into().unwrap();
    let bits = case.number("bits");
    let (gates, witness) = for_bits(bits, XorCase { input, public });
    if let Some(kimchi) = case.witnesses.first() {
        assert_eq!(&rows(&witness), kimchi);
    }
    (gates, witness)
}

/// Every row's verdict on `witness` is Kimchi's.
fn verify_rows<F, G>(
    case: &Case,
    gates: Vec<CircuitGate<F>>,
    public: usize,
    witness: &[Vec<F>; COLUMNS],
    kimchi: &G,
) where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let cs = ConstraintSystem::create(gates)
        .public(public)
        .build()
        .unwrap();
    for (row, gate) in cs.gates.iter().enumerate().take(witness[0].len()) {
        let ours = gate.verify_witness(row, witness, kimchi);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("row{row}")),
            "row {row}"
        );
    }
}

/// Kimchi's `test_xor`.
fn test_xor<F, G>(name: &str, kimchi: &G) -> [Vec<F>; COLUMNS]
where
    F: PrimeField + TwoAdicField,
    G: Gates<F>,
{
    let case = case(name);
    let (gates, witness) = setup::<F>(&case, false);
    verify_rows(&case, gates, 0, &witness, kimchi);
    witness
}

fn two_to(bits: u64) -> Fp {
    Fp::from_u64(2).pow(bits)
}

#[test]
fn test_prove_and_verify_xor() {
    let case = case("test_prove_and_verify_xor");
    let (gates, witness) = setup::<Fp>(&case, false);
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_xor64_alternating() {
    let witness = test_xor::<Fp, _>("test_xor64_alternating", &vesta_gates());
    assert_eq!(witness[2][0], two_to(64) - Fp::ONE);
    assert_eq!(witness[2][1], two_to(48) - Fp::ONE);
    assert_eq!(witness[2][2], two_to(32) - Fp::ONE);
    assert_eq!(witness[2][3], two_to(16) - Fp::ONE);
    assert_eq!(witness[2][4], Fp::ZERO);
}

#[test]
fn test_xor64_zeros() {
    let witness = test_xor::<Fp, _>("test_xor64_zeros", &vesta_gates());
    assert_eq!(witness[2][0], Fp::ZERO);
}

#[test]
fn test_xor64_zero_one() {
    let witness = test_xor::<Fp, _>("test_xor64_zero_one", &vesta_gates());
    assert_eq!(witness[2][0], two_to(64) - Fp::ONE);
}

/// A random test, on Vesta then on Pallas.
fn random(name: &str) {
    test_xor::<Fp, _>(&format!("{name} vesta"), &vesta_gates());
    test_xor::<Fq, _>(&format!("{name} pallas"), &pallas_gates());
}

#[test]
fn test_xor8_random() {
    random("test_xor8_random");
}

#[test]
fn test_xor16_random() {
    random("test_xor16_random");
}

#[test]
fn test_xor32_random() {
    random("test_xor32_random");
}

#[test]
fn test_xor64_random() {
    random("test_xor64_random");
}

#[test]
fn test_xor128_random() {
    random("test_xor128_random");
}

#[test]
fn test_bad_xor_decompsition() {
    let case = case("test_bad_xor_decompsition");
    let (gates, mut witness) = setup::<Fp>(&case, false);
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let kimchi = vesta_gates();
    for col in 0..COLUMNS {
        witness[col][0] = witness[col][0] + Fp::ONE;
        let ours = cs.gates[0].verify_witness(0, &witness, &kimchi);
        assert_eq!(
            debug(&ours),
            case.verdict(&format!("col{col}")),
            "column {col}"
        );
        witness[col][0] = witness[col][0] - Fp::ONE;
    }
    let ours = cs.gates[0].verify_witness(0, &witness, &kimchi);
    assert_eq!(debug(&ours), case.verdict("undone"));
}

#[test]
fn test_extend_xor() {
    let case = case("test_extend_xor");
    let (gates, witness) = setup::<Fp>(&case, true);
    verify_rows(&case, gates, 2, &witness, &vesta_gates());
}

#[test]
fn test_bad_xor() {
    let case = case("test_bad_xor");
    let (gates, mut witness) = setup::<Fp>(&case, false);
    witness[2][0] = Fp::ZERO;
    for i in 1..=4 {
        witness[COLUMNS - i][0] = Fp::ZERO;
    }
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_xor_finalization() {
    let case = case("test_xor_finalization");
    let two_64 = Fp::from_u64(1 << 32) * Fp::from_u64(1 << 32);
    let word = |hi: u64, lo: u64| Fp::from_u64(hi) * two_64 + Fp::from_u64(lo);
    let input = [
        word(0xDC81_1727_DAF2_2EC1, 0x5927_D6AA_275F_406B),
        word(0xA4F4_417A_F072_DF90, 0x16A1_EAB4_58DA_80D1),
    ];
    let (gates, witness) = for_bits(
        128,
        XorCase {
            input,
            public: true,
        },
    );
    assert_eq!(rows(&witness), case.witnesses[0]);
    verify_rows(&case, gates.clone(), 2, &witness, &vesta_gates());
    let public = vec![witness[0][0], witness[0][1]];
    let result = prove_and_verify(gates, witness, public);
    assert_eq!(debug(&result), case.verdict("prove"));
}

/// Not one of Kimchi's tests: the XOR's last row is the zero constant
/// (Generic, coefficient 1), its three first cells on one copy cycle, and
/// a public input is presented as Kimchi's Generic row of coefficient 1.
#[test]
fn the_last_row_is_a_zero_constant_and_public_rows_are_generic() {
    let gates = kimchi_rows(&xor::<Fp, 16>(), 2);
    assert_eq!(gates[1].typ, GateType::Generic);
    assert_eq!(gates[1].coeffs, vec![Fp::ONE]);
    let start = Wire { row: 1, col: 0 };
    let mut cycle = vec![start];
    let mut at = gates[1].wires[0];
    while at != start {
        cycle.push(at);
        at = gates[at.row].wires[at.col];
    }
    cycle.sort();
    assert_eq!(cycle, Wire::for_row(1)[..3].to_vec());
    let public = kimchi_rows(&PublicXor(Xor::<Fp, 16> { table: xor_table() }), 4);
    let first = (public[0].typ, public[0].coeffs.clone());
    assert_eq!(first, (GateType::Generic, vec![Fp::ONE]));
    assert_eq!(public[0].wires[0], Wire { row: 2, col: 0 });
}
