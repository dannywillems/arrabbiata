//! A gadget from definition to a verified proof: a MiMC-style hash,
//! `x -> (x + k_i)^5` for round constants `k_0, ..., k_6`.
//!
//! - one round is a gadget of one row: `t = x + k` with `k` in a fixed
//!   column, `t^2` and `t^4` in two cells, and `t^5` on the next row, where
//!   the next round reads it;
//! - the hash places its private input, runs the rounds, and makes the
//!   result public;
//! - the seven rounds share one selector, because their constraints are the
//!   same and only the fixed column changes;
//! - the proof says "I know a preimage of this hash" and verifies against
//!   the hash only.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Cell, Env, Layout, Of, Scalar, TypedGadget, circuit, compile, witness};
use arrabbiata_generalized_plonk::{max_coefficients, prove, setup, verify};
use arrabbiata_ipa::Ipa;
use arrabbiata_pasta::{Fq as F, Pallas};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};

/// The domain tag of the sponge deriving the IPA generators.
const GENERATORS_DOMAIN: u64 = 0;
/// The domain tag of IPA's own transcript.
const IPA_DOMAIN: u64 = 1;
/// The domain tag of the proofs' transcripts.
const PROOF_DOMAIN: u64 = 2;

const ROUNDS: usize = 7;

/// The table: the round's value and its two powers, the round constant, and
/// copies on the first column (none are needed here).
const LAYOUT: Layout = Layout {
    witness: 3,
    fixed: 1,
    permuted: 1,
};

/// Generalized PlonK's bounds: the gates read at most 8 column variables,
/// and have degree at most 3.
const VARIABLES: usize = 8;
const DEGREE: usize = 3;

/// One round: `x -> (x + k)^5`, `x` on column 0 of the row.
#[derive(Clone, Copy, Debug)]
struct Round {
    constant: F,
}

impl TypedGadget<F> for Round {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "mimc-round";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let k = env.fixed(Cell::new(0, 0), self.constant);
        let t = x + k;
        let t2 = env.define(Cell::new(1, 0), t.clone() * t.clone());
        let t4 = env.define(Cell::new(2, 0), t2.clone() * t2);
        env.define(Cell::new(0, 1), t4 * t)
    }
}

/// The hash of a private input, made public.
#[derive(Clone, Debug)]
struct Hash {
    constants: [F; ROUNDS],
}

impl TypedGadget<F> for Hash {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "mimc";
    const ROWS: usize = ROUNDS + 1;

    fn synthesize<E>(&self, env: &mut E, preimage: Of<Scalar, E::Var>) -> Of<Scalar, E::Var>
    where
        E: Env<F>,
    {
        let mut x = env.advice(Cell::new(0, 0), [&preimage], |[p]| p);
        for constant in self.constants {
            x = env.call(&Round { constant }, x);
        }
        env.public(&x);
        x
    }
}

fn hash() -> Hash {
    Hash {
        constants: core::array::from_fn(|i| F::from_u64(1_000 + i as u64)),
    }
}

/// The hash, computed directly.
fn reference(constants: &[F; ROUNDS], preimage: F) -> F {
    constants.iter().fold(preimage, |x, k| {
        let t = x + *k;
        t * t * t * t * t
    })
}

fn ipa(size: usize) -> Ipa<Pallas, PoseidonSponge<F>> {
    let mut generators = pallas_sponge(GENERATORS_DOMAIN);
    let size = size.next_power_of_two();
    let ipa_transcript = vesta_sponge(IPA_DOMAIN);
    Ipa::<Pallas, _>::setup(&mut generators, size, ipa_transcript)
}

#[test]
fn a_preimage_is_proved() {
    let gadget = hash();
    let preimage = F::from_u64(42);
    let (compiled, output) =
        witness::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &preimage).unwrap();
    let expected = reference(&gadget.constants, preimage);
    assert_eq!(output, expected);
    assert_eq!(compiled.public, vec![expected]);

    let pcs = ipa(max_coefficients(&compiled.circuit));
    let (prover_key, verifier_key) = setup(&compiled.circuit, &pcs).unwrap();
    let (proof, _) = prove(
        &prover_key,
        &pcs,
        vesta_sponge(PROOF_DOMAIN),
        &compiled.public,
        &compiled.columns,
    )
    .unwrap();
    let claimed = compiled.public.clone();
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &claimed,
            proof.clone()
        )
        .is_ok()
    );
    let another_hash = [expected + F::ONE];
    assert!(
        verify(
            &verifier_key,
            &pcs,
            vesta_sponge(PROOF_DOMAIN),
            &another_hash,
            proof
        )
        .is_err()
    );
}

#[test]
fn the_rounds_share_one_selector() {
    let gadget = hash();
    let c = circuit::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT).unwrap();
    // One kind of gate: the round's three definitions.
    assert_eq!(c.selectors().len(), 1);
    assert_eq!(c.gates().len(), 3);
    let selected = c.selectors()[0].iter().filter(|s| **s == F::ONE).count();
    assert_eq!(selected, ROUNDS);
    // The round constants are the fixed column, one per round row.
    let constants: Vec<F> = c.fixed()[0][..ROUNDS].to_vec();
    assert_eq!(constants, gadget.constants.to_vec());
}

#[test]
fn the_circuit_does_not_depend_on_the_witness() {
    let gadget = hash();
    let without = circuit::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT).unwrap();
    for preimage in [F::ZERO, F::ONE, F::from_u64(123_456_789)] {
        let (with, _) = witness::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &preimage).unwrap();
        assert_eq!(with.circuit, without);
    }
}

#[test]
fn compile_is_the_symbolic_run() {
    // The circuit and the kind of each selector, from the gadget alone: the
    // gate kinds a witness run names, and no witness.
    let gadget = hash();
    let compiled = compile::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT).unwrap();
    assert_eq!(
        compiled.circuit,
        circuit::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT).unwrap()
    );
    let (with, _) = witness::<F, _, VARIABLES, DEGREE>(&gadget, LAYOUT, &F::ONE).unwrap();
    assert_eq!(compiled.kinds, with.kinds);
    assert!(
        compiled
            .columns
            .iter()
            .all(|column| column.iter().all(|x| *x == F::ZERO))
    );
}
