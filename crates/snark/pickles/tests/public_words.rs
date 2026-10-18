//! A public word in the circuit: its canonical integer, pinned by its bound
//! (`assert_below`), then acted on a constant point as a Type2 scalar
//! (`act_shifted`), which must equal the native action at the window's
//! edges.

use arrabbiata_curve::Projective;
use arrabbiata_field::{Field, Native};
use arrabbiata_gadget::{Array, Env, Of, Scalar, TypedGadget, evaluate, measure, witness};
use arrabbiata_kimchi::oracle::coordinates;
use arrabbiata_kimchi_gadgets::LAYOUT;
use arrabbiata_pasta::{Fp, Fq, Vesta};
use arrabbiata_pickles::{
    curve::Shifted,
    group::{GroupModel, PublicWord},
    model::{Affine, Circuit, load},
    statement::{WINDOW_BITS, Word},
};

/// `x` constrained below `2^bits`.
#[derive(Clone, Copy)]
struct Below {
    bits: u32,
}

impl TypedGadget<Fq> for Below {
    type Input = Array<Scalar, 1>;
    type Output = Scalar;
    const NAME: &'static str = "pickles-test-below";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<Fq>>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> E::Var {
        let x = load(env, &input).remove(0);
        let x = Affine::of(x);
        let mut model = Circuit::new(env);
        model.assert_below(&x, self.bits);
        model.cell(&x)
    }
}

/// The action of a word of the given bound on the generator of Vesta.
#[derive(Clone, Copy)]
struct Act {
    bound: u32,
}

impl TypedGadget<Fq> for Act {
    type Input = Array<Scalar, 1>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "pickles-test-act";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<Fq>>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> [E::Var; 2] {
        let w = load(env, &input).remove(0);
        let mut model = Circuit::new(env);
        let word = PublicWord {
            value: Affine::of(w),
            bound: self.bound,
        };
        let g = Projective::<Vesta>::generator();
        GroupModel::<Vesta>::act_shifted(&mut model, &g, &word)
    }
}

/// `2^n - 1` in `Fq`.
fn all_ones(n: u32) -> Fq {
    (Fq::ONE + Fq::ONE).pow(u64::from(n)) - Fq::ONE
}

#[test]
fn a_word_is_below_its_bound_or_has_no_witness() {
    for bits in [1, 2, 64, 128, 129, 200, WINDOW_BITS] {
        let gadget = Below { bits };
        let top = all_ones(bits);
        assert!(
            witness::<Fq, _, 64, 7>(&gadget, LAYOUT, &[top]).is_ok(),
            "{bits}"
        );
        assert!(witness::<Fq, _, 64, 7>(&gadget, LAYOUT, &[Fq::ZERO]).is_ok());
        let past = top + Fq::ONE;
        assert!(
            witness::<Fq, _, 64, 7>(&gadget, LAYOUT, &[past]).is_err(),
            "{bits}"
        );
    }
    // -1 is the integer q - 1, past every bound: the residue is not the word.
    let gadget = Below { bits: WINDOW_BITS };
    assert!(witness::<Fq, _, 64, 7>(&gadget, LAYOUT, &[-Fq::ONE]).is_err());
}

#[test]
fn the_action_of_a_word_is_the_native_one() {
    // At the window's edges: 0, 1, 2^254 - 1, for each bound that admits
    // them.
    let g = Projective::<Vesta>::generator();
    let cases = [
        (Fq::ZERO, 1),
        (Fq::ONE, 1),
        (Fq::from_u64(6), 128),
        (all_ones(127), 128),
        (all_ones(128), 128),
        (all_ones(WINDOW_BITS), WINDOW_BITS),
    ];
    for (w, bound) in cases {
        let gadget = Act { bound };
        let native =
            GroupModel::<Vesta>::act_shifted(&mut Native, &g, &PublicWord { value: w, bound });
        let (x, y) = coordinates(&native);
        assert_eq!(evaluate(&gadget, LAYOUT, &[w]).unwrap(), [x, y], "{w:?}");
        assert!(
            witness::<Fq, _, 64, 7>(&gadget, LAYOUT, &[w]).is_ok(),
            "{w:?}"
        );
    }
    // A word past its bound has no witness, nor one past the window: the
    // cell -1 is the integer q - 1, which is not below 2^254.
    let past = all_ones(128) + Fq::ONE;
    assert!(witness::<Fq, _, 64, 7>(&Act { bound: 128 }, LAYOUT, &[past]).is_err());
    let gadget = Act { bound: WINDOW_BITS };
    assert!(witness::<Fq, _, 64, 7>(&gadget, LAYOUT, &[-Fq::ONE]).is_err());
}

/// A Type1 scaling of the generator of Vesta by the cell it reads.
#[derive(Clone, Copy)]
struct Scale;

impl TypedGadget<Fq> for Scale {
    type Input = Array<Scalar, 1>;
    type Output = Array<Scalar, 2>;
    const NAME: &'static str = "pickles-test-scale";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<Fq>>(&self, env: &mut E, input: Of<Self::Input, E::Var>) -> [E::Var; 2] {
        let t = load(env, &input).remove(0);
        let mut model = Circuit::new(env);
        let g = Projective::<Vesta>::generator();
        let base = GroupModel::<Vesta>::constant_point(&mut model, &g);
        let scalar = Shifted {
            high: Affine::of(t),
            low: None,
        };
        GroupModel::<Vesta>::scale_point(&mut model, &base, &scalar)
    }
}

#[test]
fn the_scaling_is_canonical() {
    // VarBaseMul reads the canonical integer of the cell: below 2^254 it
    // scales by [2 t + 2^255 + 1]; at 2^254, whose bits would need the top
    // one, there is no witness, rather than one on the bits of t + q.
    let g = Projective::<Vesta>::generator();
    let t = all_ones(WINDOW_BITS);
    let s = Word::of(t).reduce::<Fp>();
    let expected = g * (s + s + (Fp::ONE + Fp::ONE).pow(255) + Fp::ONE);
    let (x, y) = coordinates(&expected);
    assert_eq!(evaluate(&Scale, LAYOUT, &[t]).unwrap(), [x, y]);
    assert!(witness::<Fq, _, 64, 7>(&Scale, LAYOUT, &[t]).is_ok());
    let past = t + Fq::ONE;
    assert!(witness::<Fq, _, 64, 7>(&Scale, LAYOUT, &[past]).is_err());
}
