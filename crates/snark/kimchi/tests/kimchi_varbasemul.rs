//! Kimchi's `kimchi/src/tests/varbasemul.rs`, with the same inputs and
//! outputs. The circuit of ten multiplications by scalars of 255 bits is one
//! typed gadget, our VarBaseMul gates unwired as Kimchi lays them out: its
//! rows are the symbolic run, its witness the numeric run (each result
//! `(1 + 2^255 + 2 x) T` and equal to Kimchi's, the Blake2b512 digest
//! Kimchi's), and the proof's verdict (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::{Affine, Projective};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::constraints::COLUMNS;
use arrabbiata_kimchi_gadgets::varbasemul::{
    ACCUMULATORS, BITS_PER_GATE, N, N_NEXT, ROWS_PER_GATE, VarBaseMulGate,
};
use arrabbiata_pasta::{Fp, Fq, Pallas};
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, kimchi_rows, kimchi_witness, prove_and_verify, rows};

/// The scalars of Kimchi's test.
const SCALARS: usize = 10;
/// The bits of each scalar.
const BITS: usize = 255;
/// The rows of each multiplication.
const ROWS_PER_SCALAR: usize = ROWS_PER_GATE * BITS / BITS_PER_GATE;

/// Kimchi's circuit: each scalar's multiplication of `T` from `A`, its
/// gates unwired as Kimchi's test leaves them (each gate writes `T`, its
/// accumulator and `n`). Input `(T, A, bits)`, the bits of each scalar
/// most significant first.
#[derive(Clone, Copy, Debug)]
struct Multiplications;

impl<F: PrimeField> TypedGadget<F> for Multiplications {
    type Input = (
        Array<Scalar, 2>,
        Array<Scalar, 2>,
        Array<Scalar, { SCALARS * BITS }>,
    );
    type Output = Unit;
    const NAME: &'static str = "kimchi-varbasemul-test";
    const ROWS: usize = SCALARS * ROWS_PER_SCALAR;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (t, a, bits): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let write = |env: &mut E, column: usize, row: usize, v: &E::Var| {
            env.advice(Cell::new(column, row), [v], |[x]| x)
        };
        let (_, x0) = ACCUMULATORS[0];
        for (s, scalar) in bits.chunks(BITS).enumerate() {
            let mut acc = a.clone();
            let mut n = env.constant(F::ZERO);
            for (g, five) in scalar.chunks(BITS_PER_GATE).enumerate() {
                // The `g`th gate of scalar `s` is on this row.
                let row = s * ROWS_PER_SCALAR + g * ROWS_PER_GATE;
                let base = [write(env, 0, row, &t[0]), write(env, 1, row, &t[1])];
                let start = [
                    write(env, x0, row, &acc[0]),
                    write(env, x0 + 1, row, &acc[1]),
                ];
                let count = write(env, N, row, &n);
                let five: [E::Var; BITS_PER_GATE] = core::array::from_fn(|k| five[k].clone());
                (acc, n) = env.call(&VarBaseMulGate, (base, start, count, five));
            }
        }
    }
}

/// The coordinates of `p`.
fn coordinates(p: &Projective<Pallas>) -> (Fp, Fp) {
    let affine: Affine<Pallas> = p.to_affine();
    (affine.x().unwrap(), affine.y().unwrap())
}

/// The Blake2b512 digest of the witness as the tool prints it.
fn digest(witness: &[Vec<Fp>; COLUMNS]) -> String {
    let lines: Vec<String> = rows(witness).iter().map(|row| row.join(" ")).collect();
    Blake2b512::digest(lines.join("\n").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `x`, an integer below both moduli, in `Fp`.
fn integer(x: Fq) -> Fp {
    let two_64 = Fp::from_u64(2).pow(64);
    (0..Fq::LIMBS)
        .rev()
        .fold(Fp::ZERO, |acc, i| acc * two_64 + Fp::from_u64(x.limb(i)))
}

#[test]
fn varbase_mul_test() {
    gate_tests::with_stack(varbase_mul_test_body);
}

fn varbase_mul_test_body() {
    let case = case("varbase_mul_test");
    // Kimchi's T, mina-curves' Pallas generator (1, y), not ours.
    let [x, y]: [Fp; 2] = case.elements("base").try_into().unwrap();
    let t = Affine::<Pallas>::new(x, y).unwrap().to_projective();
    let xs: Vec<Fq> = case.elements("x");
    assert_eq!(xs.len(), SCALARS);
    let accs: Vec<Fp> = case.elements("acc");
    let ns: Vec<Fp> = case.elements("n");
    let n = <Multiplications as TypedGadget<Fp>>::ROWS;
    let gates = kimchi_rows(&Multiplications, n);
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let (xt, yt) = coordinates(&t);
    let (x0, y0) = coordinates(&(t + t));
    let bits: Vec<Fp> = xs
        .iter()
        .flat_map(|x| {
            (0..BITS)
                .rev()
                .map(move |k| (x.limb(k / 64) >> (k % 64)) & 1)
        })
        .map(Fp::from_u64)
        .collect();
    let input = ([xt, yt], [x0, y0], bits.try_into().unwrap());
    let w = kimchi_witness(&Multiplications, &input, n);
    let (_, last) = ACCUMULATORS[BITS_PER_GATE];
    for (i, x) in xs.iter().enumerate() {
        // The last gate's result is on its second row; its n' on its first.
        let end = (i + 1) * ROWS_PER_SCALAR - 1;
        let acc = (w[last][end], w[last + 1][end]);
        let shift = Fq::from_u64(2).pow(BITS as u64);
        let expected = coordinates(&(t * (Fq::ONE + shift + *x + *x)));
        assert_eq!(acc, expected, "scalar {i}");
        assert_eq!(acc, (accs[2 * i], accs[2 * i + 1]), "Kimchi's scalar {i}");
        let count = w[N_NEXT][end - 1];
        assert_eq!(count, integer(*x));
        assert_eq!(count, ns[i]);
    }
    let kimchi_digest = case.input("digest");
    assert_eq!(kimchi_digest[0], w[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&w));
    let result = prove_and_verify(gates, w, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
