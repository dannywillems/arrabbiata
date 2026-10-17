//! Kimchi's `kimchi/src/tests/generic.rs`, test by test, with the same
//! inputs and outputs. The circuit is one typed gadget (the public inputs,
//! then our Generic pairs on Kimchi's values): its rows are the symbolic
//! run, its witness the numeric run, each proof's verdict against Kimchi's
//! run (`gate_tests/`). `test_generic_gate_kzg` (Kimchi's `bn254` feature,
//! a KZG proof) has no counterpart: our Kimchi proves with the IPA only.

mod gate_tests;
mod kimchi_runs;

use arrabbiata_field::{Field, PrimeField};
use arrabbiata_gadget::{Array, Cell, Env, Of, Placed, Scalar, TypedGadget, Unit};
use arrabbiata_kimchi_gadgets::generic::{GenericPair, GenericSpec, PublicInput};
use arrabbiata_pasta::Fp;
use gate_tests::{case, debug, kimchi_rows, kimchi_witness, prove_and_verify, rows};

/// The rows of `l + 3 r = o` beside `2 l r = o`, and of `l = 3` beside
/// `l = 5`, each kind ten times.
const REPEATS: usize = 10;

/// Kimchi's `create_circuit` and `fill_in_witness` as one gadget: `P`
/// public inputs, then `l + 3 r = o` beside `2 l r = o` on `l = 11, r =
/// 23`, then `l = 3` beside `l = 5`.
#[derive(Clone, Copy, Debug)]
struct KimchiGeneric<const P: usize>;

impl<F: PrimeField, const P: usize> TypedGadget<F> for KimchiGeneric<P> {
    type Input = Array<Scalar, P>;
    type Output = Unit;
    const NAME: &'static str = "kimchi-generic-test";
    const ROWS: usize = P + 2 * REPEATS;

    fn synthesize<E: Env<F>>(
        &self,
        env: &mut E,
        public: Of<Self::Input, E::Var>,
    ) -> Of<Unit, E::Var> {
        for p in public {
            env.call(&PublicInput, p);
        }
        let cells = vec![
            Cell::new(0, 0),
            Cell::new(1, 0),
            Cell::new(3, 0),
            Cell::new(4, 0),
        ];
        let operations = Placed {
            gadget: GenericPair {
                first: GenericSpec::Add {
                    left: None,
                    right: Some(F::from_u64(3)),
                    output: None,
                },
                second: GenericSpec::Mul {
                    output: None,
                    mul: Some(F::from_u64(2)),
                },
            },
            cells: cells.clone(),
        };
        let constants = Placed {
            gadget: GenericPair {
                first: GenericSpec::Const(F::from_u64(3)),
                second: GenericSpec::Const(F::from_u64(5)),
            },
            cells,
        };
        let operands = |env: &mut E, l: u64, r: u64| {
            [env.constant(F::from_u64(l)), env.constant(F::from_u64(r))]
        };
        for _ in 0..REPEATS {
            let x = operands(env, 11, 23);
            let y = operands(env, 11, 23);
            env.call(&operations, (x, y));
        }
        for _ in 0..REPEATS {
            let x = operands(env, 3, 0);
            let y = operands(env, 5, 0);
            env.call(&constants, (x, y));
        }
    }
}

/// One of the four: the circuit with `P` public rows, its witness,
/// TestFramework's verdict.
fn generic<const P: usize>(name: &str, public: [Fp; P]) {
    let case = case(name);
    assert_eq!(public.to_vec(), case.elements::<Fp>("public"));
    let gadget = KimchiGeneric::<P>;
    let n = <KimchiGeneric<P> as TypedGadget<Fp>>::ROWS;
    let gates = kimchi_rows(&gadget, n);
    let witness = kimchi_witness(&gadget, &public, n);
    assert_eq!(rows(&witness), case.witnesses[0]);
    let result = prove_and_verify(gates, witness, public.to_vec());
    assert_eq!(debug(&result), case.verdict("prove"));
}

#[test]
fn test_generic_gate() {
    generic("test_generic_gate", []);
}

#[test]
fn test_generic_gate_pub() {
    generic("test_generic_gate_pub", [Fp::from_u64(3); 5]);
}

#[test]
fn test_generic_gate_pub_all_zeros() {
    generic("test_generic_gate_pub_all_zeros", [Fp::ZERO; 5]);
}

#[test]
fn test_generic_gate_pub_empty() {
    generic("test_generic_gate_pub_empty", []);
}
