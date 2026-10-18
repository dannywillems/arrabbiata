//! The group half of a Pallas proof, natively and as a circuit over `Fp`,
//! the other side of the cycle: its scalars, of the larger field `Fq`, are
//! held as Type2 shifted values (high bits, low bit), scaled as Pickles'
//! `scale_fast2`.

use core::cell::RefCell;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::{Field, Native, PrimeField};
use arrabbiata_gadget::{evaluate, witness};
use arrabbiata_kimchi::{
    constraints::{CircuitGate, ConstraintSystem, GateType, Wire},
    deferred::expand_deferred,
    oracle::{ScalarChallenge, coordinates},
    prover::{ProverIndex, prove},
    rng::{Arkworks, SplitMix64},
    srs::Srs,
};
use arrabbiata_kimchi_gadgets::LAYOUT;
use arrabbiata_pasta::{Fp, Fq, Pallas};
use arrabbiata_pickles::{
    curve::{GroupHalf, Shape, flatten, held, scalar_width},
    group::{Constants, DeferredIn, ProofIn, PublicWord, group_half_in},
    statement::Word,
    wrap::wrap_gates,
};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base, vesta_base};

/// A 128-bit scalar of `Fq` as the element of `Fp` it is.
fn to_base(x: Fq) -> Fp {
    Fp::from_limbs(&[x.limb(0), x.limb(1)]).unwrap()
}

#[test]
fn the_group_half_of_a_pallas_proof_as_a_circuit_over_fp() {
    // Row 0 binds w0 to the public input; row 1 is w0 + w1 = w2.
    let gates: Vec<CircuitGate<Fq>> = [vec![Fq::ONE], vec![Fq::ONE, Fq::ONE, -Fq::ONE]]
        .into_iter()
        .enumerate()
        .map(|(row, coeffs)| CircuitGate::new(GateType::Generic, Wire::for_row(row), coeffs))
        .collect();
    let cs = ConstraintSystem::create(gates).public(1).build().unwrap();
    let index = ProverIndex::create(cs, Srs::<Pallas>::create(8));
    let public = [Fq::from_u64(5)];
    let columns: [Vec<Fq>; 15] = core::array::from_fn(|c| match c {
        0 => vec![public[0], Fq::from_u64(1)],
        1 => vec![Fq::ZERO, Fq::from_u64(2)],
        2 => vec![Fq::ZERO, Fq::from_u64(3)],
        _ => vec![Fq::ZERO; 2],
    });
    let base = KimchiSponge::new(pallas_base());
    let scalar = KimchiSponge::new(vesta_base());
    let mut rng = Arkworks(SplitMix64(17));
    let proof = prove(
        &index,
        columns,
        &[],
        &[],
        &wrap_gates(),
        (&base, &scalar),
        &mut rng,
    )
    .unwrap();

    let vi = &index.verifier_index;
    let values = expand_deferred(
        vi,
        &index.srs,
        &proof,
        &public,
        &wrap_gates(),
        (&base, &scalar),
    )
    .unwrap();
    let constants = Constants::new(vi, &index.srs, &base).unwrap();
    let opening = &proof.proof;
    let proof_in = ProofIn {
        previous: vec![],
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
        public: public
            .iter()
            .map(|p| PublicWord {
                value: Word::of(*p).reduce(),
                bound: Some(254),
            })
            .collect(),
        perm: values.plonk.perm,
        zeta_to_srs_length: values.plonk.zeta_to_srs_length,
        zeta_to_domain_size: values.plonk.zeta_to_domain_size,
        xi: to_base(values.xi.0),
        combined_inner_product: values.combined_inner_product,
        b: values.b,
    };

    // Natively: the opening holds and the draws are the deferred ones.
    let poseidon = pallas_base();
    let model = RefCell::new(Native);
    let native = group_half_in::<Pallas, _>(&model, &poseidon, &constants, &proof_in, &deferred);
    assert_eq!(native.left, native.right);
    let lambda = Pallas::lambda();
    let mapped: Vec<Fq> = native
        .prechallenges
        .iter()
        .map(|c| ScalarChallenge(Fq::from_limbs(&[c.limb(0), c.limb(1)]).unwrap()).to_field(lambda))
        .collect();
    assert_eq!(mapped, values.bulletproof_challenges);

    // As a circuit over Fp, the scalars of Fq held as Type2: the native
    // outputs, and a satisfied witness.
    assert_eq!(scalar_width::<Pallas>(), 2);
    let shape = Shape {
        previous: 0,
        chunks: 1,
        quotient_chunks: proof_in.t.len(),
        rounds: proof_in.rounds.len(),
        public: 1,
        width: 2,
    };
    assert_eq!((shape.inputs(), shape.outputs()), (78, 10));
    let gadget =
        GroupHalf::<Pallas, 78, 10>::new(constants, &poseidon, shape, vec![Some(254)]).unwrap();
    let point = |p: &Projective<Pallas>| {
        let (x, y) = coordinates(p);
        [x, y]
    };
    let flat: [Fp; 78] = flatten(&proof_in, &deferred, point, |s| held::<Pallas>(*s))
        .try_into()
        .unwrap();
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
    let out = evaluate(&gadget, LAYOUT, &flat).unwrap();
    assert_eq!(out.to_vec(), expected);
    assert!(witness::<Fp, _, 64, 7>(&gadget, LAYOUT, &flat).is_ok());

    // A false b, even or odd in its low bit, breaks the final equation.
    for delta in [Fq::ONE, Fq::from_u64(2)] {
        let wrong = DeferredIn {
            b: deferred.b + delta,
            ..deferred.clone()
        };
        let flat: [Fp; 78] = flatten(&proof_in, &wrong, point, |s| held::<Pallas>(*s))
            .try_into()
            .unwrap();
        assert!(witness::<Fp, _, 64, 7>(&gadget, LAYOUT, &flat).is_err());
    }
}
