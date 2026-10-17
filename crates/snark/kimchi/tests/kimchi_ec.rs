//! Kimchi's `kimchi/src/tests/ec.rs`, with the same inputs and outputs: 100
//! doublings, 100 additions and 100 sums `P + (-P)` of multiples of two
//! points Kimchi draws, each row computed by our CompleteAdd gate, the
//! witness's Blake2b512 digest Kimchi's, and the proof's verdict
//! (`gate_tests/`).

mod gate_tests;
mod kimchi_runs;

use arrabbiata_curve::{Affine, Projective};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, assign};
use arrabbiata_kimchi::constraints::{COLUMNS, CircuitGate, GateType, Wire};
use arrabbiata_kimchi_gadgets::{LAYOUT, complete_add::CompleteAdd};
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

/// The column variables of the row.
const VARIABLES: usize = 11;
/// The degree of the CompleteAdd constraints.
const DEGREE: usize = 3;

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
    let case = case("ec_test");
    let (num_doubles, num_additions, num_infs) = (100, 100, 100);
    // Kimchi's generator, mina-curves' (1, y), not ours.
    let [x, y]: [Fp; 2] = case.elements("base").try_into().unwrap();
    let g = Affine::<Pallas>::new(x, y).unwrap().to_projective();
    let [r1, r2]: [Fq; 2] = case.elements("r").try_into().unwrap();
    let ps = multiples(g * r1, num_additions);
    let qs = multiples(g * r2, num_additions);
    let gates: Vec<CircuitGate<Fp>> = (0..num_doubles + num_additions + num_infs)
        .map(|row| CircuitGate::new(GateType::CompleteAdd, Wire::for_row(row), vec![]))
        .collect();
    gate_tests::assert_kimchi_rows(&gates, &case.gates[0]);
    let pairs = ps.iter().take(num_doubles).map(|p| (*p, *p));
    let pairs = pairs.chain(ps.iter().zip(&qs).map(|(p, q)| (*p, *q)));
    let pairs = pairs.chain(ps.iter().take(num_infs).map(|p| (*p, -*p)));
    let mut witness: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);
    for (p, q) in pairs {
        let [x1, y1] = coordinates(&p);
        let [x2, y2] = coordinates(&q);
        let (compiled, _) =
            assign::<Fp, _, VARIABLES, DEGREE>(&AddRow, LAYOUT, &[x1, y1, x2, y2]).unwrap();
        for (column, values) in witness.iter_mut().zip(&compiled.columns) {
            column.push(values[0]);
        }
    }
    let kimchi_digest = case.input("digest");
    assert_eq!(kimchi_digest[0], witness[0].len().to_string());
    assert_eq!(kimchi_digest[1], digest(&witness));
    let result = prove_and_verify(gates, witness, vec![]);
    assert_eq!(debug(&result), case.verdict("prove"));
}
