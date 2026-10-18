//! A gadget's constraints in a model of the signature of fields, and the
//! rows a gadget takes.

use arrabbiata_field::{Field, FieldModel, Native};
use arrabbiata_gadget::{
    Cell, Chain, Env, Layout, Of, Repeat, Scalar, TypedGadget, gates, measure,
};
use arrabbiata_pasta::Fp as F;

const LAYOUT: Layout = Layout {
    witness: 3,
    fixed: 1,
    permuted: 3,
};

/// `y = x^3 + k`, on one row: `x` and `y` in cells, `k` fixed.
#[derive(Clone, Copy, Debug)]
struct Cube {
    k: F,
}

impl TypedGadget<F> for Cube {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "cube";
    const ROWS: usize = 1;

    fn synthesize<E: Env<F>>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let x = env.advice(Cell::new(0, 0), [&x], |[v]| v);
        let k = env.fixed(Cell::new(0, 0), self.k);
        let y = env.advice(Cell::new(1, 0), [&x, &k], |[v, c]| v * v * v + c);
        env.assert_zero(x.clone() * x.clone() * x - y.clone() + k);
        y
    }
}

/// A gadget whose rows depend on its parameter: `count` cubes.
#[derive(Clone, Copy, Debug)]
struct Cubes {
    count: usize,
}

impl TypedGadget<F> for Cubes {
    type Input = Scalar;
    type Output = Scalar;
    const NAME: &'static str = "cubes";
    const ROWS: usize = 0;

    fn rows(&self) -> usize {
        measure(self, LAYOUT).unwrap_or(0)
    }

    fn synthesize<E: Env<F>>(&self, env: &mut E, x: Of<Scalar, E::Var>) -> Of<Scalar, E::Var> {
        let mut y = x;
        for _ in 0..self.count {
            y = env.call(&Cube { k: F::ONE }, y);
        }
        y
    }
}

/// A model counting its products.
#[derive(Default)]
struct Counting {
    products: usize,
}

impl FieldModel<F> for Counting {
    type Value = F;

    fn constant(&mut self, c: F) -> F {
        c
    }

    fn add(&mut self, a: &F, b: &F) -> F {
        *a + *b
    }

    fn mul(&mut self, a: &F, b: &F) -> F {
        self.products += 1;
        *a * *b
    }

    fn neg(&mut self, a: &F) -> F {
        -*a
    }

    fn inverse(&mut self, a: &F) -> F {
        a.inverse().unwrap_or(F::ZERO)
    }
}

#[test]
fn gate_constraints_interpret() {
    // In the native model, the constraint's values; in another, the same
    // term, computed there: here two products for x^3.
    let kinds = gates::<F, _>(&Cube { k: F::from_u64(5) }, LAYOUT).unwrap();
    let cube = &kinds[0];
    let witness = |c: usize, _: usize| [F::from_u64(2), F::from_u64(13)][c];
    let fixed = |_: usize, _: usize| F::from_u64(5);
    let native = cube.interpret(&mut Native, &witness, &fixed);
    assert_eq!(native, cube.evaluate(&witness, &fixed));
    assert_eq!(native, vec![F::ZERO]);
    let mut counting = Counting::default();
    let counted = cube.interpret(&mut counting, &witness, &fixed);
    assert_eq!(counted, native);
    assert_eq!(counting.products, 2);
}

#[test]
fn rows_and_measure() {
    // A gadget of a type has the type's rows; a family says its own, which
    // measure counts; composites add their parts'.
    assert_eq!(Cube { k: F::ONE }.rows(), 1);
    assert_eq!(measure(&Cube { k: F::ONE }, LAYOUT), Ok(0));
    let three = Cubes { count: 3 };
    assert_eq!(measure(&three, LAYOUT), Ok(3));
    assert_eq!(three.rows(), 3);
    let chain = Chain {
        first: three,
        second: Cubes { count: 2 },
    };
    assert_eq!(chain.rows(), 5);
    let repeated = Repeat::<_, 2> { gadget: three };
    assert_eq!(repeated.rows(), 6);
    let out = arrabbiata_gadget::evaluate(&chain, LAYOUT, &F::ONE).unwrap();
    let mut x = F::ONE;
    for _ in 0..5 {
        x = x * x * x + F::ONE;
    }
    assert_eq!(out, x);
}
