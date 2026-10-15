//! PlonKish circuits as generalized circuits.
//!
//! A circuit of [`arrabbiata_plonkish`] is a generalized circuit whose
//! gates are the constraints of its gadgets, each turned on by its
//! gadget's selector, whose variables are the columns its constraints read
//! (at their rotations), and whose public inputs are bound to witness
//! column 0 on its public rows. [`from_plonkish`] is that map; it
//! preserves and reflects satisfaction.

use alloc::{collections::BTreeSet, vec::Vec};

use arrabbiata_field::Field;
use arrabbiata_mvpoly::{MVPoly, Sparse, Var};
use arrabbiata_plonkish::{
    Circuit as PlonkishCircuit, GateSet,
    gate::{self},
};
use arrabbiata_poly::{Expr, Variable as Read};

use crate::{Circuit, Error, Gate, Variable};

/// A variable of a PlonKish constraint as a generalized variable.
///
/// ```
/// use arrabbiata_generalized_plonk::{Variable, plonkish::variable};
/// use arrabbiata_poly::{Rotation, Variable as Read, WitnessColumn};
/// let v = variable::<2, 0>(&Read::Witness(WitnessColumn::new::<1>(), Rotation::NEXT));
/// assert_eq!(v, Variable::Witness { column: 1, rotation: Rotation::NEXT });
/// ```
#[must_use]
pub fn variable<const W: usize, const X: usize>(v: &Read<W, X>) -> Variable {
    match v {
        Read::Witness(c, r) => Variable::Witness {
            column: c.index(),
            rotation: *r,
        },
        Read::Fixed(c, r) => Variable::Fixed {
            column: c.index(),
            rotation: *r,
        },
    }
}

/// The polynomial of `expr`, its variables numbered by `index`; `None` if
/// a variable has no index or the degree exceeds `D`.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_generalized_plonk::{Variable, plonkish::polynomial};
/// use arrabbiata_mvpoly::{MVPoly, Sparse};
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
/// let a = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let p: Sparse<F, 1, 2> = polynomial(&(a.clone() * a), &|_: Variable| Some(0)).unwrap();
/// assert_eq!(p.eval(&[F::from_u64(3)]), F::from_u64(9));
/// ```
pub fn polynomial<F, I, const W: usize, const X: usize, const N: usize, const D: usize>(
    expr: &Expr<F, W, X>,
    index: &I,
) -> Option<Sparse<F, N, D>>
where
    F: Field,
    I: Fn(Variable) -> Option<usize>,
{
    match expr {
        Expr::Constant(c) => Some(Sparse::constant(*c)),
        Expr::Variable(v) => Sparse::variable(Var::try_new(index(variable(v))?)?),
        Expr::Neg(a) => Some(-polynomial::<F, I, W, X, N, D>(a, index)?),
        Expr::Add(a, b) => Some(
            polynomial::<F, I, W, X, N, D>(a, index)? + polynomial::<F, I, W, X, N, D>(b, index)?,
        ),
        Expr::Mul(a, b) => {
            polynomial::<F, I, W, X, N, D>(a, index)?
                .checked_mul(&polynomial::<F, I, W, X, N, D>(b, index)?)
        }
    }
}

/// The columns, at their rotations, some constraint of `G` reads, sorted
/// and without repetition.
///
/// ```
/// use arrabbiata_generalized_plonk::{Variable, plonkish::columns_read};
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::vanilla::Vanilla;
/// // a, b, c and the five fixed columns, on the current row.
/// assert_eq!(columns_read::<F, Vanilla, 3, 5>().len(), 8);
/// ```
#[must_use]
pub fn columns_read<F, G, const W: usize, const X: usize>() -> Vec<Variable>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut set = BTreeSet::new();
    let mut vars = Vec::new();
    for selector in 0..G::SELECTORS {
        for c in G::constraints(selector) {
            gate::variables(&c, &mut vars);
        }
    }
    for v in &vars {
        set.insert(variable(v));
    }
    set.into_iter().collect()
}

/// The generalized circuit of a PlonKish circuit, with gates of degree at
/// most `D` in `N` variables.
///
/// # Errors
///
/// [`Error::Shape`] if `N` is below the number of columns the gate set
/// reads (at their rotations) or `D` below the degree of a constraint in
/// all its variables, fixed columns included.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_generalized_plonk::plonkish::from_plonkish;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::vanilla;
/// let (c, w, ()) = build::<F, _, _>(|b| {
///     let x = b.witness(F::from_u64(3));
///     let y = b.mul(&x, &x);
///     let out = b.public_input(F::from_u64(9));
///     b.assert_equal(&y, &out);
/// });
/// let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
/// // 8 columns read; q_M a b has degree 3 in all of them.
/// let g = from_plonkish::<F, _, 3, 5, 1, 8, 3>(&circuit).unwrap();
/// assert!(g.is_satisfied(witness.columns(), witness.public_inputs()));
/// assert!(from_plonkish::<F, _, 3, 5, 1, 8, 2>(&circuit).is_err());
/// ```
pub fn from_plonkish<
    F,
    G,
    const W: usize,
    const X: usize,
    const Q: usize,
    const N: usize,
    const D: usize,
>(
    circuit: &PlonkishCircuit<F, G, W, X, Q>,
) -> Result<Circuit<F, N, D>, Error>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let variables = columns_read::<F, G, W, X>();
    let index = |v: Variable| variables.iter().position(|u| *u == v);
    let mut gates = Vec::new();
    for selector in 0..Q {
        for c in G::constraints(selector) {
            let p = polynomial::<F, _, W, X, N, D>(&c, &index).ok_or(Error::Shape)?;
            gates.push(Gate::new(Some(selector), p));
        }
    }
    let bindings = circuit.public_rows().iter().copied().enumerate().collect();
    Circuit::new(
        circuit.num_rows(),
        W,
        circuit.fixed().to_vec(),
        circuit.selector_columns().to_vec(),
        variables,
        gates,
        circuit.sigma().to_vec(),
        circuit.public_rows().len(),
        bindings,
    )
}
