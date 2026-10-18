//! The group half, written once: natively it is `arrabbiata-kimchi`'s
//! group half (the oracle), on Kimchi proofs with and without chunks and
//! previous accumulators; a false deferred value or opening is caught.

use core::cell::RefCell;

use arrabbiata_curve::{glv::Glv, msm};
use arrabbiata_field::{Field, Native, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Scalar, TypedGadget, Unit, evaluate, witness};
use arrabbiata_ipa::{RecursionChallenge, b_poly_coefficients, generic::shift_scalar};
use arrabbiata_kimchi::{
    compiled,
    constraints::{CircuitGate, ConstraintSystem},
    deferred::{expand_deferred, group_half},
    gates::KimchiGates,
    oracle::{BaseReading, ScalarChallenge, coordinates},
    prover::{ProverIndex, ProverProof, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    generic::{Generic, GenericSpec},
    poseidon::{PoseidonPermutation, ROWS_PER_PERMUTATION},
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_pickles::{
    curve::{GroupHalf, Shape, flatten},
    group::{Constants, DeferredIn, ProofIn, group_half_in},
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

/// A 128-bit scalar, as the base-field value it is.
fn to_base(x: Fp) -> Fq {
    Fq::from_limbs(&[x.limb(0), x.limb(1)]).unwrap()
}

/// A 128-bit base-field value, as the scalar it is.
fn to_scalar(x: Fq) -> Fp {
    Fp::from_limbs(&[x.limb(0), x.limb(1)]).unwrap()
}

/// What the group half reads on a proof, and the deferred values.
type GroupInputs = (
    Constants<Vesta>,
    ProofIn<arrabbiata_curve::Projective<Vesta>, Fp>,
    DeferredIn<Fq, Fp>,
    arrabbiata_kimchi::deferred::DeferredValues<Fp>,
);

/// The group half's inputs on `proof`: its commitments and opening, and
/// the deferred values `arrabbiata-kimchi` expands.
fn inputs_of(index: &ProverIndex<Vesta>, proof: &ProverProof<Vesta>, public: &[Fp]) -> GroupInputs {
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let vi = &index.verifier_index;
    let values =
        expand_deferred(vi, &index.srs, proof, public, &gates(), (&base, &scalar)).unwrap();
    let constants = Constants::new(vi, &index.srs, &base).unwrap();
    let opening = &proof.proof;
    let proof_in = ProofIn {
        previous: proof.prev_challenges.iter().map(|p| p.comm).collect(),
        witness: core::array::from_fn(|i| proof.w_comm[i].chunks.clone()),
        z: proof.z_comm.chunks.clone(),
        t: proof.t_comm.chunks.clone(),
        rounds: opening.lr.clone(),
        delta: opening.delta,
        z1: opening.z1,
        z2: opening.z2,
        sg: opening.sg,
    };
    let deferred = DeferredIn {
        public: public.to_vec(),
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: to_base(values.xi.0),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };
    (constants, proof_in, deferred, values)
}

/// The group half natively on `proof`: the draws are the deferred ones,
/// the opening holds, `sg` is Kimchi's; a false `b` or `z1` fails it.
fn natively(index: &ProverIndex<Vesta>, proof: &ProverProof<Vesta>, public: &[Fp]) {
    let (constants, proof_in, deferred, values) = inputs_of(index, proof, public);
    let poseidon = vesta_base();
    let model = RefCell::new(Native);
    let out = group_half_in::<Vesta, _>(&model, &poseidon, &constants, &proof_in, &deferred);
    assert_eq!(out.left, out.right, "the opening holds");
    let drawn = &out.drawn;
    assert_eq!(ScalarChallenge(to_scalar(drawn.alpha)), values.plonk.alpha);
    assert_eq!(to_scalar(drawn.beta), values.plonk.beta);
    assert_eq!(to_scalar(drawn.gamma), values.plonk.gamma);
    assert_eq!(ScalarChallenge(to_scalar(drawn.zeta)), values.plonk.zeta);
    let lambda = Vesta::lambda();
    let mapped: Vec<Fp> = out
        .prechallenges
        .iter()
        .map(|c| ScalarChallenge(to_scalar(*c)).to_field(lambda))
        .collect();
    assert_eq!(mapped, values.bulletproof_challenges);
    let base = KimchiSponge::new(vesta_base());
    let oracle = group_half(
        &index.verifier_index,
        &index.srs,
        proof,
        public,
        &values,
        &base,
    )
    .unwrap();
    assert_eq!(
        BaseReading::<Vesta>::read_digest(&Native, &drawn.digest),
        oracle.digest
    );
    assert_eq!(out.sg, oracle.sg);

    let wrong_b = DeferredIn {
        b: deferred.b + Fp::ONE,
        ..deferred.clone()
    };
    let out = group_half_in::<Vesta, _>(&model, &poseidon, &constants, &proof_in, &wrong_b);
    assert_ne!(out.left, out.right, "a false b");
    let wrong_z1 = ProofIn {
        z1: proof_in.z1 + Fp::ONE,
        ..proof_in.clone()
    };
    let out = group_half_in::<Vesta, _>(&model, &poseidon, &constants, &wrong_z1, &deferred);
    assert_ne!(out.left, out.right, "a false z1");
}

/// The fixture's input.
const INPUT: [u64; WIDTH] = [1, 2, 3];

#[test]
fn the_group_half_is_the_verifiers_natively() {
    let poseidon = pallas_base();
    let fixture = Fixture {
        poseidon: &poseidon,
    };
    let input = INPUT.map(Fp::from_u64);
    let (index, columns) = indexed(&fixture, &input, 0, None);
    let proof = proved(&index, columns, &[]);
    natively(&index, &proof, &[]);
}

#[test]
fn the_group_half_with_chunks_and_a_previous_accumulator() {
    let poseidon = pallas_base();
    let fixture = Fixture {
        poseidon: &poseidon,
    };
    let input = INPUT.map(Fp::from_u64);
    let (index, columns) = indexed(&fixture, &input, 1, Some(8));
    let prev = [accumulator(&index.srs)];
    let proof = proved(&index, columns, &prev);
    natively(&index, &proof, &[]);
}

#[test]
fn the_group_half_with_a_public_input() {
    // Row 0 binds w0 to the public input; row 1 is w0 + w1 = w2.
    let gates: Vec<CircuitGate<Fp>> = [vec![Fp::ONE], vec![Fp::ONE, Fp::ONE, -Fp::ONE]]
        .into_iter()
        .enumerate()
        .map(|(row, coeffs)| {
            CircuitGate::new(
                arrabbiata_kimchi::constraints::GateType::Generic,
                arrabbiata_kimchi::constraints::Wire::for_row(row),
                coeffs,
            )
        })
        .collect();
    let cs = ConstraintSystem::create(gates).public(1).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Vesta>::create(8));
    let public = [Fp::from_u64(5)];
    let columns: [Vec<Fp>; 15] = core::array::from_fn(|c| match c {
        0 => vec![public[0], Fp::from_u64(1)],
        1 => vec![Fp::ZERO, Fp::from_u64(2)],
        2 => vec![Fp::ZERO, Fp::from_u64(3)],
        _ => vec![Fp::ZERO; 2],
    });
    let base = KimchiSponge::new(vesta_base());
    let scalar = KimchiSponge::new(pallas_base());
    let mut rng = Arkworks(SplitMix64(7));
    let proof = prove(
        &index,
        columns,
        &[],
        &[],
        &self::gates(),
        (&base, &scalar),
        &mut rng,
    )
    .unwrap();
    natively(&index, &proof, &public);
    as_a_circuit::<71, 10>(&index, &proof, &public);
}

/// The column variables and the degree of the group half's circuits.
const GROUP_VARIABLES: usize = 64;
const GROUP_DEGREE: usize = 7;

/// A scalar of Vesta, as the circuit over Fq holds it: its shifted value
/// (`Shifted_value.Type1`), which fits Fq.
fn shifted(s: &Fp) -> Fq {
    let t = shift_scalar::<Vesta>(*s);
    Fq::from_limbs(&(0..4).map(|i| t.limb(i)).collect::<Vec<_>>()).unwrap()
}

/// The group half as a circuit of input size `N` and output size `M` on
/// `proof`: the native outputs, a witness that satisfies it; a false `b`
/// leaves no satisfying witness.
fn as_a_circuit<const N: usize, const M: usize>(
    index: &ProverIndex<Vesta>,
    proof: &ProverProof<Vesta>,
    public: &[Fp],
) {
    let (constants, proof_in, deferred, _) = inputs_of(index, proof, public);
    let poseidon = vesta_base();
    let model = RefCell::new(Native);
    let native = group_half_in::<Vesta, _>(&model, &poseidon, &constants, &proof_in, &deferred);
    let mut expected = vec![
        native.drawn.alpha,
        native.drawn.beta,
        native.drawn.gamma,
        native.drawn.zeta,
        native.drawn.digest,
    ];
    expected.extend(&native.prechallenges);
    let (sx, sy) = coordinates(&native.sg);
    expected.extend([sx, sy]);

    let shape = Shape {
        previous: proof_in.previous.len(),
        chunks: proof_in.z.len(),
        quotient_chunks: proof_in.t.len(),
        rounds: proof_in.rounds.len(),
        public: public.len(),
        width: 1,
    };
    let gadget = GroupHalf::<Vesta, N, M>::new(constants, &poseidon, shape).unwrap();
    let point = |p: &arrabbiata_curve::Projective<Vesta>| {
        let (x, y) = coordinates(p);
        [x, y]
    };
    let flat: [Fq; N] = flatten(&proof_in, &deferred_in_fq(&deferred), point, |s| {
        vec![shifted(s)]
    })
    .try_into()
    .unwrap();
    let out = evaluate(&gadget, LAYOUT, &flat).unwrap();
    assert_eq!(out.to_vec(), expected, "circuit output");
    let (_, witnessed) =
        witness::<Fq, _, GROUP_VARIABLES, GROUP_DEGREE>(&gadget, LAYOUT, &flat).unwrap();
    assert_eq!(witnessed.to_vec(), expected, "witness");

    let wrong = DeferredIn {
        b: deferred.b + Fp::ONE,
        ..deferred.clone()
    };
    let flat: [Fq; N] = flatten(&proof_in, &deferred_in_fq(&wrong), point, |s| {
        vec![shifted(s)]
    })
    .try_into()
    .unwrap();
    // Refused by the group half's own constraint: the copy wiring the IPA's
    // two sides together, not by a gadget inside it.
    let refused = witness::<Fq, _, GROUP_VARIABLES, GROUP_DEGREE>(&gadget, LAYOUT, &flat).err();
    assert!(
        matches!(
            refused,
            Some(arrabbiata_gadget::Error::Unsatisfied {
                gadget: "pickles-group-half",
                ..
            })
        ),
        "a false b breaks the final equation: {refused:?}"
    );
}

/// The deferred values with `xi` kept as it is (a base-field value) and the
/// scalars as they are, for [`flatten`]'s scalar map.
fn deferred_in_fq(d: &DeferredIn<Fq, Fp>) -> DeferredIn<Fq, Fp> {
    d.clone()
}

#[test]
fn the_group_half_as_a_circuit() {
    let poseidon = pallas_base();
    let fixture = Fixture {
        poseidon: &poseidon,
    };
    let input = INPUT.map(Fp::from_u64);
    let (index, columns) = indexed(&fixture, &input, 0, None);
    let proof = proved(&index, columns, &[]);
    let rounds = proof.proof.lr.len();
    assert_eq!(rounds, 4, "the shape below is for 4 rounds");
    as_a_circuit::<74, 11>(&index, &proof, &[]);
}
