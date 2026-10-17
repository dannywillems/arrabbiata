//! Kimchi's `kimchi/src/tests/endomul.rs`, test by test, with the same
//! inputs and outputs. The circuit of 100 scalar multiplications is one
//! typed gadget, our EndoMul rows unwired as Kimchi lays them out: its rows
//! are the symbolic run, its witness the numeric run (each result `[k] T`
//! for the challenge's scalar, the Blake2b512 digest against Kimchi's, the
//! proof's verdict); and the 16-bit regression with Kimchi's result and
//! witness (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::{Affine, Projective, glv::Glv};
use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::{constraints::COLUMNS, oracle::ScalarChallenge};
use arrabbiata_kimchi_gadgets::endomul::{BITS_PER_ROW, EndoMulGate, column};
use arrabbiata_pasta::{Fp, Fq, Pallas};
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, kimchi_rows, kimchi_witness, prove_and_verify, rows};

/// The scalar challenges of Kimchi's test.
const SCALARS: usize = 100;
/// The bits of each challenge.
const BITS: usize = 128;

/// One multiplication of `T` by `B` bits (most significant first) from the
/// accumulator `P`, on `B / 4` EndoMul rows and the row after, which holds
/// the result: Kimchi's `endosclmul::gen_witness`. Its rows are unwired as
/// Kimchi's test leaves them: each writes `T`, the first `P` and `n = 0`.
/// Input `(T, P, bits)`.
#[derive(Clone, Copy, Debug)]
struct Multiplication<F, const B: usize> {
    endo: F,
}

impl<F: PrimeField, const B: usize> TypedGadget<F> for Multiplication<F, B> {
    type Input = (Array<Scalar, 2>, Array<Scalar, 2>, Array<Scalar, B>);
    type Output = (Array<Scalar, 2>, Scalar);
    const NAME: &'static str = "kimchi-endomul-test";
    const ROWS: usize = B / BITS_PER_ROW + 1;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (t, p, bits): Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let write = |env: &mut E, column: usize, row: usize, v: &E::Var| {
            env.advice(Cell::new(column, row), [v], |[x]| x)
        };
        let mut acc = [
            write(env, column::XP, 0, &p[0]),
            write(env, column::YP, 0, &p[1]),
        ];
        let zero = env.constant(F::ZERO);
        let mut n = write(env, column::N, 0, &zero);
        let gate = EndoMulGate { endo: self.endo };
        for (row, four) in bits.chunks(BITS_PER_ROW).enumerate() {
            let t = [
                write(env, column::XT, row, &t[0]),
                write(env, column::YT, row, &t[1]),
            ];
            let four: [E::Var; BITS_PER_ROW] = core::array::from_fn(|k| four[k].clone());
            // The `row`th call is on row `row`.
            let (next, n_next) = env.call(&gate, (t, acc, n, four));
            acc = next;
            n = n_next;
        }
        (acc, n)
    }
}

/// Kimchi's circuit: each challenge's multiplication on its own rows.
#[derive(Clone, Copy, Debug)]
struct Multiplications<F> {
    endo: F,
}

impl<F: PrimeField> TypedGadget<F> for Multiplications<F> {
    type Input = (
        Array<Scalar, 2>,
        Array<Scalar, 2>,
        Array<Scalar, { SCALARS * BITS }>,
    );
    type Output = Unit;
    const NAME: &'static str = "kimchi-endomul-tests";
    const ROWS: usize = SCALARS * Multiplication::<F, BITS>::ROWS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        (t, p, bits): Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        let one = Multiplication::<F, BITS> { endo: self.endo };
        for scalar in bits.chunks(BITS) {
            let scalar: [E::Var; BITS] = core::array::from_fn(|k| scalar[k].clone());
            env.call(&one, (t.clone(), p.clone(), scalar));
        }
    }
}

/// The coordinates of `p`.
fn coordinates(p: &Projective<Pallas>) -> (Fp, Fp) {
    let affine: Affine<Pallas> = p.to_affine();
    (affine.x().unwrap(), affine.y().unwrap())
}

/// Kimchi's base point `T`, its Pallas generator `(1, y)` (mina-curves),
/// which is not ours, `(-1, 2)` (as in zcash's pasta_curves).
fn base(case: &gate_tests::Case) -> Projective<Pallas> {
    let [x, y]: [Fp; 2] = case.elements("base").try_into().unwrap();
    assert_eq!(x, Fp::ONE);
    Affine::<Pallas>::new(x, y).unwrap().to_projective()
}

/// Kimchi's starting accumulator: `2 (phi(T) + T)`.
fn acc0(t: &Projective<Pallas>) -> (Fp, Fp) {
    let sum = t.endomorphism() + *t;
    coordinates(&(sum + sum))
}

/// The low 128 bits of `x`, most significant first.
fn bits_msb(x: Fq) -> Vec<bool> {
    (0..128)
        .rev()
        .map(|i| (x.limb(i / 64) >> (i % 64)) & 1 == 1)
        .collect()
}

/// `x`, a 128-bit integer, in `Fp`.
fn integer(x: Fq) -> Fp {
    let two_64 = Fp::from_u64(2).pow(64);
    Fp::from_u64(x.limb(1)) * two_64 + Fp::from_u64(x.limb(0))
}

/// The Blake2b512 digest of the witness as the tool prints it.
fn digest(witness: &[Vec<Fp>; COLUMNS]) -> String {
    let lines: Vec<String> = rows(witness).iter().map(|row| row.join(" ")).collect();
    Blake2b512::digest(lines.join("\n").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn endomul_test() {
    gate_tests::with_stack(endomul_test_body);
}

fn endomul_test_body() {
    let case = case("endomul_test");
    let endo_q: Fp = case.elements("endo_q")[0];
    let endo_r: Fq = case.elements("endo_r")[0];
    assert_eq!(endo_q, Pallas::beta(), "Kimchi's endo_q is Pallas's beta");
    assert_eq!(
        endo_r,
        Pallas::lambda(),
        "Kimchi's endo_r is Pallas's lambda"
    );
    let xs: Vec<Fq> = case.elements("x");
    let accs: Vec<Fp> = case.elements("acc");
    let ns: Vec<Fp> = case.elements("n");
    assert_eq!(xs.len(), SCALARS);
    let t = base(&case);
    let (x0, y0) = acc0(&t);
    let (xt, yt) = coordinates(&t);
    let gadget = Multiplications { endo: endo_q };
    let n = <Multiplications<Fp> as TypedGadget<Fp>>::ROWS;
    let gates = kimchi_rows(&gadget, n);
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let bits: Vec<Fp> = xs
        .iter()
        .flat_map(|x| bits_msb(*x))
        .map(|b| Fp::from_u64(u64::from(b)))
        .collect();
    let input = ([xt, yt], [x0, y0], bits.try_into().unwrap());
    let witness = kimchi_witness(&gadget, &input, n);
    let rows_per_scalar = n / SCALARS;
    for (i, x) in xs.iter().enumerate() {
        let after = (i + 1) * rows_per_scalar - 1;
        let acc = (witness[column::XP][after], witness[column::YP][after]);
        let scalar = ScalarChallenge(*x).to_field(endo_r);
        assert_eq!(acc, coordinates(&(t * scalar)), "scalar {i}");
        assert_eq!(acc, (accs[2 * i], accs[2 * i + 1]), "Kimchi's scalar {i}");
        assert_eq!(witness[column::N][after], integer(*x));
        assert_eq!(witness[column::N][after], ns[i]);
    }
    let kimchi_digest = case.input("digest");
    assert_eq!(kimchi_digest[0], witness[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&witness));
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_endomul_regression() {
    // 0b1010_0011_1100_0101 = 41925, most significant first.
    let case = case("test_endomul_regression");
    let bits = [
        true, false, true, false, false, false, true, true, true, true, false, false, false, true,
        false, true,
    ];
    let t = base(&case);
    let (x0, y0) = acc0(&t);
    let (xt, yt) = coordinates(&t);
    let gadget = Multiplication::<Fp, 16> {
        endo: Pallas::beta(),
    };
    let input = ([xt, yt], [x0, y0], bits.map(|b| Fp::from_u64(u64::from(b))));
    let witness = kimchi_witness(&gadget, &input, 5);
    let res_acc = (witness[column::XP][4], witness[column::YP][4]);
    let res_n = witness[column::N][4];
    let acc: Vec<Fp> = case.elements("acc");
    assert_eq!(res_acc, (acc[0], acc[1]));
    assert_eq!(res_n, Fp::from_u64(41925));
    assert_eq!(res_n, case.elements::<Fp>("n")[0]);
    assert_eq!(rows(&witness), case.witnesses[0]);
    // Row 0 reads the bits 1, 0, 1, 0, and n is 10 after it.
    let first = [
        witness[11][0],
        witness[12][0],
        witness[13][0],
        witness[14][0],
    ];
    assert_eq!(first, [Fp::ONE, Fp::ZERO, Fp::ONE, Fp::ZERO]);
    assert_eq!(witness[6][1], Fp::from_u64(10));
}
