//! Kimchi's `kimchi/src/tests/ec.rs`, with the same inputs and outputs: 100
//! doublings, 100 additions and 100 sums `P + (-P)` of multiples of two
//! points Kimchi draws, as one typed gadget of our CompleteAdd rows: its
//! rows (the symbolic run) Kimchi's, its witness (the numeric run) of
//! Kimchi's Blake2b512 digest, and the proof's verdict (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::{Affine, Projective};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi::constraints::COLUMNS;
use arrabbiata_kimchi_gadgets::complete_add::CompleteAdd;
use arrabbiata_pasta::{Fp, Fq, Pallas};
use blake2::{Blake2b512, Digest};
use gate_tests::{case, debug, prove_and_verify, rows};

/// The two points in columns 0-3, then our CompleteAdd gate.
#[derive(Clone, Copy, Debug)]
struct AddRow;

impl TypedGadget<Fp> for AddRow {
    type Input = Array<Scalar, 4>;
    type Output = Array<Scalar, 3>;
    const NAME: &'static str = "add-row";
    const ROWS: usize = 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        points: Of<Array<Scalar, 4>, E::Var>,
    ) -> Of<Array<Scalar, 3>, E::Var> {
        let cells = core::array::from_fn(|i| env.advice(Cell::new(i, 0), [&points[i]], |[v]| v));
        env.call(&CompleteAdd, cells)
    }
}

/// The additions of Kimchi's test: 100 doublings, 100 additions, 100 sums
/// to infinity.
const ADDITIONS: usize = 300;

/// Kimchi's circuit: one CompleteAdd row per pair of points, unwired.
#[derive(Clone, Copy, Debug)]
struct Additions;

impl TypedGadget<Fp> for Additions {
    type Input = Array<Array<Scalar, 4>, ADDITIONS>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-ec-test";
    const ROWS: usize = ADDITIONS;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        pairs: Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        for points in pairs {
            env.call(&AddRow, points);
        }
    }
}

/// The coordinates of `p`.
fn coordinates(p: &Projective<Pallas>) -> [Fp; 2] {
    let affine: Affine<Pallas> = p.to_affine();
    [affine.x().unwrap(), affine.y().unwrap()]
}

/// `[p, 2 p, .., n p]`.
fn multiples(p: Projective<Pallas>, n: usize) -> Vec<Projective<Pallas>> {
    let mut acc = p;
    let mut all = Vec::with_capacity(n);
    for _ in 0..n {
        all.push(acc);
        acc = acc + p;
    }
    all
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
fn ec_test() {
    gate_tests::with_stack(ec_test_body);
}

fn ec_test_body() {
    let case = case("ec_test");
    let (num_doubles, num_additions, num_infs) = (100, 100, 100);
    // Kimchi's generator, mina-curves' (1, y), not ours.
    let [x, y]: [Fp; 2] = case.elements("base").try_into().unwrap();
    let g = Affine::<Pallas>::new(x, y).unwrap().to_projective();
    let [r1, r2]: [Fq; 2] = case.elements("r").try_into().unwrap();
    let ps = multiples(g * r1, num_additions);
    let qs = multiples(g * r2, num_additions);
    assert_eq!(num_doubles + num_additions + num_infs, ADDITIONS);
    let gates = gate_tests::kimchi_rows(&Additions, ADDITIONS);
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let pairs = ps.iter().take(num_doubles).map(|p| (*p, *p));
    let pairs = pairs.chain(ps.iter().zip(&qs).map(|(p, q)| (*p, *q)));
    let pairs = pairs.chain(ps.iter().take(num_infs).map(|p| (*p, -*p)));
    let input: Vec<[Fp; 4]> = pairs
        .map(|(p, q)| {
            let [x1, y1] = coordinates(&p);
            let [x2, y2] = coordinates(&q);
            [x1, y1, x2, y2]
        })
        .collect();
    let input: [[Fp; 4]; ADDITIONS] = input.try_into().unwrap();
    let witness = gate_tests::kimchi_witness(&Additions, &input, ADDITIONS);
    let kimchi_digest = case.input("digest");
    assert_eq!(kimchi_digest[0], witness[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&witness));
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
