//! The behaviour of each public item, one test per item.

use arrabbiata_curve::glv::Glv;
use arrabbiata_field::{Field, FieldModel, Native, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Placed, Scalar, TypedGadget, evaluate, measure};
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    oracle::{ChallengeModel, ScalarChallenge},
    prover::{PointEvaluations, ProverIndex},
    srs::Srs,
};
use arrabbiata_kimchi_gadgets::LAYOUT;
use arrabbiata_pasta::{Fp, Fq, Vesta};
use arrabbiata_pickles::{
    curve::{self, CHALLENGE_BITS, EndoScale, SCALAR_BITS, Shape},
    group::{self, GroupModel},
    model::{Affine, Basic, Circuit, Constant, Hint, Inverse, LOAD, Load, load},
    scalar::{
        Constants, Inputs, OUTPUTS, Outputs, ScalarError, b_poly, combine, combined_inner_product,
        evaluations, flatten,
    },
    transcript::{Permute, Split128},
    wrap,
};
use arrabbiata_poseidon::kimchi::pallas_base;
use arrabbiata_transcript::Permutation;

/// The index of two Generic rows.
fn index() -> ProverIndex<Vesta> {
    let gates: Vec<CircuitGate<Fp>> = (0..2)
        .map(|row| {
            let coeffs = vec![Fp::ONE, Fp::ONE, -Fp::ONE];
            CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs)
        })
        .collect();
    let cs = ConstraintSystem::create(gates).build().unwrap();
    let depth = cs.domain.d1.size();
    ProverIndex::create(cs, Srs::create(depth))
}

/// `f(x, y)` computed in the circuit model, with the rows it takes.
fn run<F>(f: F, x: Fp, y: Fp) -> (Fp, usize)
where
    F: Fn(&mut dyn FnMut(Op) -> Value, Value, Value) -> Value + Copy,
{
    let gadget = Generic2 { f };
    let rows = measure(&gadget, LAYOUT).unwrap();
    (evaluate(&gadget, LAYOUT, &[x, y]).unwrap(), rows)
}

/// The operations a test program asks the model for.
#[derive(Clone, Copy, Debug)]
enum Op {
    Add(usize, usize),
    Mul(usize, usize),
    Inverse(usize),
    Constant(Fp),
    Low(usize),
    Endo(usize),
}

/// A value of a test program: an index into the values computed so far.
type Value = usize;

/// A gadget running a test program `f` in the circuit model.
#[derive(Clone, Copy)]
struct Generic2<F> {
    f: F,
}

impl<F> TypedGadget<Fp> for Generic2<F>
where
    F: Fn(&mut dyn FnMut(Op) -> Value, Value, Value) -> Value + Copy,
{
    type Input = Array<Scalar, 2>;
    type Output = Scalar;
    const NAME: &'static str = "pickles-test-program";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Self::Input, E::Var>,
    ) -> Of<Scalar, E::Var> {
        let cells = load(env, &input);
        let mut model = Circuit::new(env);
        let mut values: Vec<Affine<E::Var, Fp>> = cells.into_iter().map(Affine::of).collect();
        let mut ask = |op: Op| {
            let v = match op {
                Op::Add(a, b) => model.add(&values[a], &values[b]),
                Op::Mul(a, b) => model.mul(&values[a], &values[b]),
                Op::Inverse(a) => model.inverse(&values[a]),
                Op::Constant(c) => model.constant(c),
                Op::Low(a) => model.lowest_128_bits(&values[a]),
                Op::Endo(a) => model.endo(&values[a], Vesta::lambda()),
            };
            values.push(v);
            values.len() - 1
        };
        let out = (self.f)(&mut ask, 0, 1);
        let result = values[out].clone();
        model.cell(&result)
    }
}

#[test]
fn model_basic() {
    // l x + r y + m x y + c in the output cell.
    let gadget = Placed {
        gadget: Basic {
            coefficients: [
                Fp::from_u64(2),
                Fp::from_u64(3),
                -Fp::ONE,
                Fp::ONE,
                Fp::from_u64(5),
            ],
        },
        cells: vec![Cell::new(0, 0), Cell::new(1, 0)],
    };
    let out = evaluate(&gadget, LAYOUT, &[Fp::from_u64(4), Fp::from_u64(7)]).unwrap();
    assert_eq!(out, Fp::from_u64(2 * 4 + 3 * 7 + 4 * 7 + 5));
}

#[test]
fn model_inverse() {
    let gadget = Placed {
        gadget: Inverse,
        cells: vec![Cell::new(0, 0)],
    };
    let out = evaluate(&gadget, LAYOUT, &Fp::from_u64(4)).unwrap();
    assert_eq!(out * Fp::from_u64(4), Fp::ONE);
    // Zero has no inverse: no witness satisfies the row.
    assert!(arrabbiata_gadget::witness::<Fp, _, 64, 7>(&gadget, LAYOUT, &Fp::ZERO).is_err());
}

#[test]
fn model_constant() {
    let gadget = Constant {
        value: Fp::from_u64(9),
    };
    assert_eq!(evaluate(&gadget, LAYOUT, &()).unwrap(), Fp::from_u64(9));
}

#[test]
fn model_load() {
    // One row of seven cells, no gate.
    let input: [Fp; LOAD] = core::array::from_fn(|i| Fp::from_u64(i as u64));
    assert_eq!(evaluate(&Load, LAYOUT, &input).unwrap(), input);
    assert_eq!(<Load as TypedGadget<Fp>>::ROWS, 1);
}

#[test]
fn model_load_function() {
    // Nine values take two rows.
    #[derive(Clone, Copy)]
    struct Nine;
    impl TypedGadget<Fp> for Nine {
        type Input = Array<Scalar, 9>;
        type Output = Array<Scalar, 9>;
        const NAME: &'static str = "pickles-test-nine";
        const ROWS: usize = 2;
        fn synthesize<E: Env<Fp>>(
            &self,
            env: &mut E,
            input: Of<Self::Input, E::Var>,
        ) -> Of<Self::Output, E::Var> {
            let cells = load(env, &input);
            core::array::from_fn(|i| cells[i].clone())
        }
    }
    assert_eq!(measure(&Nine, LAYOUT), Ok(2));
    let input: [Fp; 9] = core::array::from_fn(|i| Fp::from_u64(i as u64 + 1));
    assert_eq!(evaluate(&Nine, LAYOUT, &input).unwrap(), input);
}

#[test]
fn model_affine() {
    let c: Affine<(), Fp> = Affine::constant(Fp::from_u64(3));
    assert!(c.cell.is_none());
    assert_eq!(c.offset, Fp::from_u64(3));
    let x: Affine<u8, Fp> = Affine::of(7);
    assert_eq!((x.cell, x.scale, x.offset), (Some(7), Fp::ONE, Fp::ZERO));
}

#[test]
fn model_circuit() {
    // (x + 1)(y + 2) / (x + y): a product and an addition one row each,
    // the inverse one, and the constant shifts free.
    let (value, rows) = run(
        |ask, x, y| {
            let one = ask(Op::Constant(Fp::ONE));
            let two = ask(Op::Constant(Fp::from_u64(2)));
            let a = ask(Op::Add(x, one));
            let b = ask(Op::Add(y, two));
            let product = ask(Op::Mul(a, b));
            let sum = ask(Op::Add(x, y));
            let inverse = ask(Op::Inverse(sum));
            ask(Op::Mul(product, inverse))
        },
        Fp::from_u64(3),
        Fp::from_u64(5),
    );
    let expected = Fp::from_u64(4 * 7) * Fp::from_u64(8).inverse().unwrap();
    assert_eq!(value, expected);
    // Load, product, sum, inverse, last product.
    assert_eq!(rows, 5);
}

#[test]
fn scalar_constants() {
    // An index without lookups: its domain and the evaluations' shape.
    let k = Constants::of(&index().verifier_index).unwrap();
    assert_eq!(k.domain_size, 8);
    assert_eq!(k.omega.pow(8), Fp::ONE);
    assert!(k.shape.optional_selectors.is_empty());
    assert_eq!(k.shape.w[0].zeta.len(), 1);
}

#[test]
fn scalar_error() {
    assert_eq!(
        ScalarError::Lookups.to_string(),
        "the scalar half's circuit does not cover lookups"
    );
    assert_eq!(
        ScalarError::Domain.to_string(),
        "the index's domain is not one of the field's"
    );
    let size = ScalarError::InputSize {
        declared: 1,
        expected: 2,
    };
    assert_eq!(
        size.to_string(),
        "an input of 1 values where the index gives 2"
    );
}

#[test]
fn scalar_combine() {
    // 1 + 2 y + 3 y^2 at y = 10.
    let chunks = [1, 2, 3].map(Fp::from_u64);
    assert_eq!(
        combine(&mut Native, &chunks, &Fp::from_u64(10)),
        Fp::from_u64(321)
    );
}

#[test]
fn scalar_b_poly() {
    // Kimchi's b_poly, as arrabbiata-ipa computes it.
    let chals = [2, 3, 5].map(Fp::from_u64);
    let x = Fp::from_u64(7);
    assert_eq!(
        b_poly(&mut Native, &chals, &x),
        arrabbiata_ipa::b_poly(&chals, x)
    );
}

#[test]
fn scalar_evaluations_and_combined_inner_product() {
    // The opening's order: previous accumulators, public, ft, then the
    // proof's; combined by xi across and r between the points.
    let point = |a: u64, b: u64| PointEvaluations {
        zeta: vec![Fp::from_u64(a)],
        zeta_omega: vec![Fp::from_u64(b)],
    };
    let k = Constants::of(&index().verifier_index).unwrap();
    let mut n = 0;
    let evals = k.shape.map(&mut |()| {
        n += 1;
        Fp::from_u64(n)
    });
    let points = [Fp::from_u64(2), Fp::from_u64(3)];
    let ft = [Fp::from_u64(100), Fp::from_u64(200)];
    let prev = vec![vec![Fp::ONE]];
    let all = evaluations(&mut Native, &evals, &prev, ft, &points);
    assert_eq!(all.len(), 1 + 2 + 1 + 6 + 15 + 15 + 6);
    assert_eq!(all[0], [vec![Fp::from_u64(3)], vec![Fp::from_u64(4)]]);
    assert_eq!(all[2], [vec![ft[0]], vec![ft[1]]]);
    let two = vec![
        [point(1, 2).zeta, point(1, 2).zeta_omega],
        [vec![Fp::from_u64(3)], vec![Fp::from_u64(4)]],
    ];
    let (xi, r) = (Fp::from_u64(10), Fp::from_u64(100));
    assert_eq!(
        combined_inner_product(&mut Native, &two, &xi, &r),
        Fp::from_u64(1 + 100 * 2 + 10 * (3 + 100 * 4))
    );
}

#[test]
fn scalar_inputs_outputs_and_flatten() {
    // The flat order: seven scalars, the challenges, the evaluations.
    let k = Constants::of(&index().verifier_index).unwrap();
    let evals = k.shape.map(&mut |()| Fp::from_u64(9));
    let inputs = Inputs {
        alpha: Fp::from_u64(1),
        beta: Fp::from_u64(2),
        gamma: Fp::from_u64(3),
        zeta: Fp::from_u64(4),
        digest: Fp::from_u64(5),
        ft_eval1: Fp::from_u64(7),
        prev_challenges: vec![vec![Fp::from_u64(8)]],
        bulletproof_challenges: vec![Fp::from_u64(11)],
        evals,
    };
    let flat = flatten(&inputs);
    assert_eq!(flat.len(), 6 + 1 + 1 + 88);
    assert_eq!(flat[..8], [1, 2, 3, 4, 5, 7, 8, 11].map(Fp::from_u64));
    let outputs = Outputs {
        linearization: arrabbiata_kimchi::verifier::Linearization {
            ft_eval0: Fp::ONE,
            perm: Fp::ONE,
            zeta_to_domain_size: Fp::ONE,
            zeta_to_srs_length: Fp::ONE,
        },
        combined_inner_product: Fp::ZERO,
        b: Fp::ZERO,
        xi: Fp::ZERO,
    };
    assert_eq!(outputs.clone(), outputs);
    assert_eq!(OUTPUTS, 7);
}

#[test]
fn transcript_permute() {
    // Kimchi's permutation on its gate's rows, the output on the row after.
    let poseidon = pallas_base();
    let gadget = Placed {
        gadget: Permute {
            poseidon: &poseidon,
        },
        cells: vec![Cell::new(0, 0), Cell::new(1, 0), Cell::new(2, 0)],
    };
    let input = [1, 2, 3].map(Fp::from_u64);
    let out = evaluate(&gadget, LAYOUT, &input).unwrap();
    let mut state = input;
    poseidon.permute(&mut state);
    assert_eq!(out, state);
    assert_eq!(gadget.rows(), 12);
}

#[test]
fn transcript_split_128() {
    // The canonical split: x = lo + 2^128 hi.
    let gadget = Placed {
        gadget: Split128,
        cells: vec![Cell::new(0, 0)],
    };
    let x = Fp::from_limbs(&[1, 2, 3, 4]).unwrap();
    let [lo, hi] = evaluate(&gadget, LAYOUT, &x).unwrap();
    assert_eq!(lo, Fp::from_limbs(&[1, 2]).unwrap());
    assert_eq!(hi, Fp::from_limbs(&[3, 4]).unwrap());
    assert!(arrabbiata_gadget::witness::<Fp, _, 64, 7>(&gadget, LAYOUT, &x).is_ok());
}

#[test]
fn transcript_circuit_challenges() {
    // In the circuit model: the lowest 128 bits of x, then its scalar.
    let x = Fp::from_limbs(&[5, 6, 7, 0]).unwrap();
    let (low, _) = run(|ask, x, _| ask(Op::Low(x)), x, Fp::ZERO);
    assert_eq!(low, Fp::from_limbs(&[5, 6]).unwrap());
    let (scalar, rows) = run(|ask, x, _| ask(Op::Endo(x)), low, Fp::ZERO);
    assert_eq!(scalar, ScalarChallenge(low).to_field(Vesta::lambda()));
    // Load, the constants 0 and 2 (2 wired to both a0 and b0), eight
    // EndoMulScalar rows, the combination.
    assert_eq!(rows, 12);
    assert_eq!(
        ChallengeModel::<Fp>::endo(&mut Native, &low, Vesta::lambda()),
        scalar
    );
}

#[test]
fn group_group_model() {
    // Natively: points of the curve, scalars of its field; the endomorphism
    // scaling of a 128-bit challenge and its inverse undo each other.
    let g = arrabbiata_curve::Projective::<Vesta>::generator();
    let m = &mut Native;
    let two = GroupModel::<Vesta>::scale_point(m, &g, &Fp::from_u64(2));
    assert_eq!(GroupModel::<Vesta>::add_points(m, &g, &g), two);
    assert_eq!(GroupModel::<Vesta>::sub_points(m, &two, &g), g);
    let chal = Fq::from_u64(0x1234_5678);
    let there = GroupModel::<Vesta>::endo_scale(m, &g, &chal);
    assert_eq!(GroupModel::<Vesta>::endo_inv_scale(m, &there, &chal), g);
    let c = GroupModel::<Vesta>::coordinates_of(m, &g);
    assert_eq!(GroupModel::<Vesta>::point_of(m, &c.x, &c.y), g);
    assert_eq!(GroupModel::<Vesta>::constant_point(m, &g), g);
    assert_eq!(GroupModel::<Vesta>::neg_point(m, &g), -g);
    assert!(!GroupModel::<Vesta>::shifted_elements(m, &Fp::ONE).is_empty());
}

#[test]
fn group_constants() {
    // Of an index without lookups: its digest, no Lagrange commitment
    // without public input, one chunk.
    let index = index();
    let base =
        arrabbiata_poseidon::kimchi::KimchiSponge::new(arrabbiata_poseidon::kimchi::vesta_base());
    let c = group::Constants::new(&index.verifier_index, &index.srs, &base).unwrap();
    assert_eq!(c.index_digest, index.verifier_index.digest(base.clone()));
    assert_eq!((c.lagrange.len(), c.chunks), (0, 1));
    assert_eq!(c.before_witness.len(), 6);
}

#[test]
fn group_proof_in_deferred_in_outputs() {
    // The shapes the group half reads and returns; their behaviour is
    // tests/group.rs's.
    let g = arrabbiata_curve::Projective::<Vesta>::generator();
    let proof = group::ProofIn {
        previous: vec![],
        witness: core::array::from_fn(|_| vec![g]),
        z: vec![g],
        t: vec![g],
        rounds: vec![(g, g)],
        delta: g,
        z1: Fp::ONE,
        z2: Fp::ONE,
        sg: g,
    };
    let deferred = group::DeferredIn {
        public: vec![],
        perm: Fp::ONE,
        zeta_to_srs_length: Fp::ONE,
        zeta_to_domain_size: Fp::ONE,
        xi: Fq::ONE,
        combined_inner_product: Fp::ONE,
        b: Fp::ONE,
    };
    assert_eq!((proof.rounds.len(), deferred.public.len()), (1, 0));
}

/// The shape of a one-chunk proof of 4 rounds, no previous accumulator,
/// no public input.
const SHAPE: Shape = Shape {
    previous: 0,
    chunks: 1,
    quotient_chunks: 7,
    rounds: 4,
    public: 0,
};

#[test]
fn curve_shape() {
    // 25 + 2 * 4 points of two coordinates, two scalars, six values; five
    // draws, four prechallenges, sg.
    assert_eq!((SHAPE.inputs(), SHAPE.outputs()), (74, 11));
    assert_eq!((CHALLENGE_BITS, SCALAR_BITS), (128, 255));
}

#[test]
fn curve_flatten() {
    let g = arrabbiata_curve::Projective::<Vesta>::generator();
    let proof = group::ProofIn {
        previous: vec![],
        witness: core::array::from_fn(|_| vec![g]),
        z: vec![g],
        t: vec![g; 7],
        rounds: vec![(g, g); 4],
        delta: g,
        z1: Fp::ONE,
        z2: Fp::ONE,
        sg: g,
    };
    let deferred = group::DeferredIn {
        public: vec![],
        perm: Fp::ONE,
        zeta_to_srs_length: Fp::ONE,
        zeta_to_domain_size: Fp::ONE,
        xi: Fq::ONE,
        combined_inner_product: Fp::ONE,
        b: Fp::ONE,
    };
    let flat = curve::flatten(&proof, &deferred, |_| [Fq::ONE, Fq::ONE], |_| Fq::ZERO);
    assert_eq!(flat.len(), SHAPE.inputs());
}

#[test]
fn curve_group_half_and_size_error() {
    // A size that is not the shape's is refused with the sizes it needs.
    let index = index();
    let base =
        arrabbiata_poseidon::kimchi::KimchiSponge::new(arrabbiata_poseidon::kimchi::vesta_base());
    let constants = group::Constants::new(&index.verifier_index, &index.srs, &base).unwrap();
    let poseidon = arrabbiata_poseidon::kimchi::vesta_base();
    let refused = curve::GroupHalf::<Vesta, 1, 11>::new(constants, &poseidon, SHAPE).err();
    assert_eq!(
        refused,
        Some(curve::SizeError {
            inputs: 74,
            outputs: 11,
        })
    );
}

#[test]
fn curve_endo_scale() {
    // 32 rows of four bits, and the row its result lands on.
    assert_eq!(<EndoScale<Fq> as TypedGadget<Fq>>::ROWS, 33);
}

/// The group map of `t` in the circuit model.
#[derive(Clone, Copy, Debug)]
struct MapInCircuit;

impl TypedGadget<Fq> for MapInCircuit {
    type Input = Array<Scalar, 1>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "map-in-circuit";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<Fq>>(
        &self,
        env: &mut E,
        input: Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let cells = load(env, &input);
        let mut m = Circuit::new(env);
        let t = Affine::of(cells[0].clone());
        let map = arrabbiata_kimchi::srs::GroupMap::<Vesta>::setup();
        let (x, y) = map.to_group_in(&mut m, &t);
        [m.cell(&x), m.cell(&y)]
    }
}

#[test]
fn curve_group_map_model() {
    // In the circuit, the native point of the map: a witnessed choice and
    // ordinate, constrained to the curve.
    let t = Fq::from_u64(1234);
    let p = arrabbiata_kimchi::srs::GroupMap::<Vesta>::setup().to_group(t);
    let out = evaluate(&MapInCircuit, LAYOUT, &[t]).unwrap();
    assert_eq!(out, [p.x().unwrap(), p.y().unwrap()]);
    let (_, witnessed) =
        arrabbiata_gadget::witness::<Fq, _, 64, 7>(&MapInCircuit, LAYOUT, &[t]).unwrap();
    assert_eq!(witnessed, out);
}

/// A witnessed square of the input, asserted equal to `x * x`, or (when
/// `wrong`) to `x * x + 1`.
#[derive(Clone, Copy, Debug)]
struct Squared {
    wrong: bool,
}

impl TypedGadget<Fq> for Squared {
    type Input = Array<Scalar, 1>;
    type Output = Scalar;
    const NAME: &'static str = "squared";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<Fq>>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> E::Var {
        let cells = load(env, &input);
        let mut m = Circuit::new(env);
        let x = Affine::of(cells[0].clone());
        let hinted = m.witness([&x], |[v]| v * v);
        let squared = m.mul(&x, &x);
        let target = if self.wrong {
            m.add(&squared, &Affine::constant(Fq::ONE))
        } else {
            squared
        };
        m.assert_equal(&hinted, &target);
        m.cell(&hinted)
    }
}

#[test]
fn model_hint_witness_assert_equal() {
    // A hint is a witness cell of the inputs' values; an equality is a copy
    // constraint, so a false one has no satisfying witness.
    let x = Fq::from_u64(9);
    assert_eq!(
        evaluate(&Squared { wrong: false }, LAYOUT, &[x]).unwrap(),
        Fq::from_u64(81)
    );
    assert!(
        arrabbiata_gadget::witness::<Fq, _, 64, 7>(&Squared { wrong: false }, LAYOUT, &[x]).is_ok()
    );
    assert!(
        arrabbiata_gadget::witness::<Fq, _, 64, 7>(&Squared { wrong: true }, LAYOUT, &[x]).is_err()
    );
    assert_eq!(<Hint<fn([Fq; 1]) -> Fq, 1> as TypedGadget<Fq>>::ROWS, 1);
}

#[test]
fn wrap_shifted_and_unshifted() {
    // The Type1 shift and its inverse: s = 2 t + 2^255 + 1, for scalars
    // across the field, including the edges.
    for s in [
        Fp::ZERO,
        Fp::ONE,
        -Fp::ONE,
        Fp::from_u64(5),
        Fp::from_u64(1 << 40) + Fp::ONE,
    ] {
        assert_eq!(wrap::unshifted(wrap::shifted(s)), s);
    }
}

#[test]
fn wrap_statement_round_trip() {
    // A statement is its public input: of inverts to_vec, and a wrong
    // length has no statement.
    let v = |k: u64| Fq::from_u64(k);
    let st = wrap::Statement {
        step_public: vec![v(1), v(2)],
        alpha: v(3),
        beta: v(4),
        gamma: v(5),
        zeta: v(6),
        xi: v(7),
        digest: v(8),
        perm: v(9),
        zeta_to_srs_length: v(10),
        zeta_to_domain_size: v(11),
        combined_inner_product: v(12),
        b: v(13),
        sg: [v(14), v(15)],
        prechallenges: vec![v(16), v(17), v(18)],
    };
    let flat = st.to_vec();
    assert_eq!(flat.len(), 2 + 13 + 3);
    assert_eq!(wrap::Statement::of(2, 3, &flat), Some(st));
    assert_eq!(wrap::Statement::of(2, 4, &flat), None);
}

#[test]
fn wrap_shape() {
    // One public input, one chunk, a quotient of seven, three rounds.
    let shape = wrap::Shape {
        public: 1,
        chunks: 1,
        quotient_chunks: 7,
        rounds: 3,
    };
    assert_eq!((shape.statement(), shape.witness()), (17, 62));
}

#[test]
fn wrap_gates_of_both_fields() {
    // Kimchi's gates over each field of the cycle, with its curve's beta.
    let _ = wrap::step_gates();
    let _ = wrap::wrap_gates();
    assert_eq!((wrap::VARIABLES, wrap::DEGREE), (64, 7));
}

#[test]
fn wrap_new_refuses_a_wrong_size() {
    let index = index();
    let poseidon = arrabbiata_poseidon::kimchi::vesta_base();
    let shape = wrap::Shape {
        public: 0,
        chunks: 1,
        quotient_chunks: 7,
        rounds: 3,
    };
    let refused =
        wrap::Wrap::<1, 1>::new(&index.verifier_index, &index.srs, &poseidon, shape).err();
    assert_eq!(refused, Some(wrap::Error::Size));
}
