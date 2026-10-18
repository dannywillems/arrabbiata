//! The scalar half, written once, in its two models: natively it is
//! `arrabbiata-kimchi`'s verifier arithmetic (an independent
//! implementation, the oracle); as a circuit, its output is the native one,
//! its witness satisfies it, and Kimchi proves and verifies it.

use arrabbiata_curve::{glv::Glv, msm};
use arrabbiata_field::{Field, FieldModel, Native};
use arrabbiata_gadget::{
    Array, Cell, Env, Of, Scalar, TypedGadget, Unit, evaluate, measure, witness,
};
use arrabbiata_ipa::{RecursionChallenge, b_poly_coefficients};
use arrabbiata_kimchi::{
    compiled,
    constraints::{CircuitGate, ConstraintSystem},
    deferred::expand_deferred,
    gates::KimchiGates,
    prover::{ProverIndex, ProverProof, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
    verifier::{Challenges, oracles, verify},
};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    generic::{Generic, GenericSpec},
    poseidon::{PoseidonPermutation, ROWS_PER_PERMUTATION},
};
use arrabbiata_pasta::{Fp, Pallas, Vesta};
use arrabbiata_pickles::{
    model::{Affine, Circuit, load},
    scalar::{Constants, Inputs, OUTPUTS, ScalarError, ScalarHalf, flatten, scalar_half},
};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, KimchiSponge, WIDTH, pallas_base, vesta_base};

/// The column variables and the degree of the compiled circuits.
const VARIABLES: usize = 64;
const DEGREE: usize = 7;

/// A Poseidon permutation of the input in columns 0-2 of its first row,
/// its output on the row after its rounds, which it reserves.
#[derive(Clone, Copy, Debug)]
struct Permutation<'a> {
    poseidon: &'a KimchiPoseidon<Fp>,
}

impl TypedGadget<Fp> for Permutation<'_> {
    type Input = Array<Scalar, WIDTH>;
    type Output = Array<Scalar, WIDTH>;
    const NAME: &'static str = "pickles-test-permutation";
    const ROWS: usize = ROWS_PER_PERMUTATION + 1;

    fn synthesize<E: Env<Fp>>(
        &self,
        env: &mut E,
        input: Of<Self::Input, E::Var>,
    ) -> Of<Self::Output, E::Var> {
        let state = core::array::from_fn(|k| env.advice(Cell::new(k, 0), [&input[k]], |[v]| v));
        env.call(
            &PoseidonPermutation {
                poseidon: self.poseidon,
            },
            state,
        )
    }
}

/// A circuit to verify: a Poseidon permutation of its input, then the
/// product of the first two outputs.
#[derive(Clone, Copy, Debug)]
struct Fixture<'a> {
    poseidon: &'a KimchiPoseidon<Fp>,
}

impl TypedGadget<Fp> for Fixture<'_> {
    type Input = Array<Scalar, WIDTH>;
    type Output = Unit;
    const NAME: &'static str = "pickles-test-fixture";
    const ROWS: usize = 13;

    fn synthesize<E: Env<Fp>>(&self, env: &mut E, input: Of<Self::Input, E::Var>) {
        let out = env.call(
            &Permutation {
                poseidon: self.poseidon,
            },
            input,
        );
        let spec = GenericSpec::Mul {
            output: None,
            mul: None,
        };
        let _ = env.call(&Generic { spec }, [out[0].clone(), out[1].clone()]);
    }
}

/// Kimchi's gates over `Fp`.
fn gates() -> KimchiGates<Fp> {
    KimchiGates::new(&pallas_base(), Pallas::beta()).unwrap()
}

/// The index and witness of the rows of `gadget` on `input`, with
/// `prev` previous accumulators and the chunk size `max_poly_size`.
fn indexed<G>(
    gadget: &G,
    input: &Of<G::Input, Fp>,
    prev: usize,
    max_poly_size: Option<usize>,
) -> (ProverIndex<Vesta>, [Vec<Fp>; 15])
where
    G: TypedGadget<Fp>,
{
    let rows = gadget.rows();
    let circuit: Vec<CircuitGate<Fp>> =
        compiled::circuit_rows::<Fp, _, VARIABLES, DEGREE>(gadget, rows, 0).unwrap();
    let (columns, _) =
        compiled::witness_columns::<Fp, _, VARIABLES, DEGREE>(gadget, input, rows).unwrap();
    let cs = ConstraintSystem::create(circuit)
        .prev_challenges(prev)
        .max_poly_size(max_poly_size)
        .build()
        .unwrap();
    let depth = max_poly_size.unwrap_or(cs.domain.d1.size());
    (ProverIndex::create(cs, Srs::create(depth)), columns)
}

/// A previous accumulator on `srs`: arbitrary challenges and the
/// commitment of their challenge polynomial.
fn accumulator(srs: &Srs<Vesta>) -> RecursionChallenge<Vesta> {
    let rounds = srs.g.len().trailing_zeros() as u64;
    let chals: Vec<Fp> = (0..rounds).map(|i| Fp::from_u64(3 + 2 * i)).collect();
    let comm = msm(&srs.g, &b_poly_coefficients(&chals));
    RecursionChallenge { chals, comm }
}

/// A proof of `index` on `columns`, carrying `prev`.
fn proved(
    index: &ProverIndex<Vesta>,
    columns: [Vec<Fp>; 15],
    prev: &[RecursionChallenge<Vesta>],
) -> ProverProof<Vesta> {
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(31));
    prove(
        index,
        columns,
        &[],
        prev,
        &gates(),
        (&base, &scalar),
        &mut rng,
    )
    .unwrap()
}

/// The scalar half's inputs on `proof`, from the verifier's own oracles,
/// and its expected outputs, from `arrabbiata-kimchi`'s deferred values.
fn inputs_of(
    index: &ProverIndex<Vesta>,
    proof: &ProverProof<Vesta>,
) -> (Inputs<Fp>, [Fp; OUTPUTS]) {
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let vi = &index.verifier_index;
    let sponges = (&base, &scalar);
    let o = oracles(vi, &index.srs, proof, &[], &gates(), sponges).unwrap();
    let values = expand_deferred(vi, &index.srs, proof, &[], &gates(), sponges).unwrap();
    let inputs = Inputs {
        challenges: Challenges {
            alpha: o.alpha,
            beta: o.beta,
            gamma: o.gamma,
            zeta: o.zeta,
            joint_combiner: None,
        },
        xi: o.v,
        r: o.u,
        ft_eval1: proof.ft_eval1,
        prev_challenges: proof
            .prev_challenges
            .iter()
            .map(|p| p.chals.clone())
            .collect(),
        bulletproof_challenges: values.bulletproof_challenges.clone(),
        evals: proof.evals.clone(),
    };
    let expected = [
        o.ft_eval0,
        values.plonk.perm,
        values.plonk.zeta_to_domain_size,
        values.plonk.zeta_to_srs_length,
        values.combined_inner_product,
        values.b,
    ];
    (inputs, expected)
}

/// The scalar half on the proof of `index`: natively, the oracle's values;
/// as a circuit of input size `N`, the same output, a satisfied witness,
/// and a Kimchi proof that verifies.
fn both_models<const N: usize>(index: &ProverIndex<Vesta>, proof: &ProverProof<Vesta>) {
    let gates = gates();
    let (inputs, expected) = inputs_of(index, proof);
    let constants = Constants::of(&index.verifier_index).unwrap();
    let native = scalar_half(&mut Native, &constants, &inputs, &gates);
    let lin = &native.linearization;
    let natively = [
        lin.ft_eval0,
        lin.perm,
        lin.zeta_to_domain_size,
        lin.zeta_to_srs_length,
        native.combined_inner_product,
        native.b,
    ];
    assert_eq!(natively, expected, "native");

    let prev: Vec<usize> = inputs.prev_challenges.iter().map(Vec::len).collect();
    let rounds = inputs.bulletproof_challenges.len();
    let gadget = ScalarHalf::<Fp, N>::new(constants, &gates, prev, rounds).unwrap();
    let flat: [Fp; N] = flatten(&inputs).try_into().unwrap();
    let out = evaluate(&gadget, LAYOUT, &flat).unwrap();
    assert_eq!(out, expected, "circuit output");
    let (_, witnessed) = witness::<Fp, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &flat).unwrap();
    assert_eq!(witnessed, expected, "witness");

    // The circuit is a Kimchi circuit: Kimchi proves it and verifies it.
    let (circuit_index, columns) = indexed(&gadget, &flat, 0, None);
    let circuit_proof = proved(&circuit_index, columns, &[]);
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(1));
    let verdict = verify(
        &circuit_index.verifier_index,
        &circuit_index.srs,
        &circuit_proof,
        &[],
        &gates,
        (&base, &scalar),
        &mut rng,
    );
    assert_eq!(verdict, Ok(()), "the scalar half's circuit proves");
}

/// The fixture's input.
const INPUT: [u64; WIDTH] = [1, 2, 3];

#[test]
fn the_scalar_half_is_the_verifiers_in_both_models() {
    let poseidon = pallas_base();
    let fixture = Fixture {
        poseidon: &poseidon,
    };
    let input = INPUT.map(Fp::from_u64);
    let (index, columns) = indexed(&fixture, &input, 0, None);
    let proof = proved(&index, columns, &[]);
    both_models::<99>(&index, &proof);
}

#[test]
fn the_scalar_half_with_chunks_and_a_previous_accumulator() {
    let poseidon = pallas_base();
    let fixture = Fixture {
        poseidon: &poseidon,
    };
    let input = INPUT.map(Fp::from_u64);
    let (index, columns) = indexed(&fixture, &input, 1, Some(8));
    let previous = accumulator(&index.srs);
    let proof = proved(&index, columns, &[previous]);
    both_models::<365>(&index, &proof);
}

#[test]
fn the_circuit_model_lays_one_row_per_product() {
    // Constants, scalings and additions of constants are free; a product
    // of two cells is one Generic row.
    #[derive(Clone, Copy, Debug)]
    struct Square;
    impl TypedGadget<Fp> for Square {
        type Input = Scalar;
        type Output = Scalar;
        const NAME: &'static str = "pickles-test-square";
        const ROWS: usize = 3;
        fn synthesize<E: Env<Fp>>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
            let cells = load(env, &[x]);
            let mut m = Circuit::new(env);
            let x = Affine::of(cells[0].clone());
            let y = m.scale(&x, Fp::from_u64(3));
            let y = m.shift(&y, Fp::ONE);
            let square = m.mul(&y, &y);
            m.cell(&square)
        }
    }
    assert_eq!(measure(&Square, LAYOUT), Ok(2));
    let out = evaluate(&Square, LAYOUT, &Fp::from_u64(2)).unwrap();
    assert_eq!(out, Fp::from_u64(49));
}

#[test]
fn an_input_of_the_wrong_size_is_refused() {
    let poseidon = pallas_base();
    let fixture = Fixture {
        poseidon: &poseidon,
    };
    let input = INPUT.map(Fp::from_u64);
    let (index, _) = indexed(&fixture, &input, 0, None);
    let gates = gates();
    let constants = Constants::of(&index.verifier_index).unwrap();
    let error = ScalarHalf::<Fp, 3>::new(constants, &gates, vec![], 5).unwrap_err();
    assert_eq!(
        error,
        ScalarError::InputSize {
            declared: 3,
            expected: 100
        }
    );
}
