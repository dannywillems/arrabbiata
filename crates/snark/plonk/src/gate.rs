//! Gadgets and gate sets.
//!
//! A [`Gadget`] is a named list of polynomial constraints over the `W`
//! witness columns and `X` fixed columns of a row and of the next row. A
//! circuit activates at most one gadget per row, through a selector: the
//! protocol multiplies every constraint of gadget `g` by the selector
//! column `q_g`, so a gadget constrains exactly the rows that turn it on.
//!
//! A [`GateSet`] is a type-level list of gadgets, `(G0, (G1, (G2, ())))`.
//! The selector of a gadget is its position in the list, so selector
//! indices are assigned by the compiler instead of by hand, and a builder
//! can only turn on a gadget the circuit's gate set contains ([`Member`]).
//! This is the design of the original Arrabbiata gadgets (one selector per
//! gadget, constraints gated automatically, one gadget per row) with the
//! index of a selector derived from its type.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::gate::{Gadget, GateSet, Member};
//! use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
//!
//! /// `b = a^2` on the current row.
//! struct Square;
//! impl Gadget<F, 2, 0> for Square {
//!     const NAME: &'static str = "square";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::Curr);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::Curr);
//!         vec![b - a.clone() * a]
//!     }
//! }
//!
//! /// `a' = a + b` on the next row.
//! struct Step;
//! impl Gadget<F, 2, 0> for Step {
//!     const NAME: &'static str = "step";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::Curr);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::Curr);
//!         let next = Expr::witness(WitnessColumn::new::<0>(), Rotation::Next);
//!         vec![next - a - b]
//!     }
//! }
//!
//! type Gates = (Square, (Step, ()));
//! assert_eq!(<Gates as GateSet<F, 2, 0>>::SELECTORS, 2);
//! assert_eq!(<Gates as Member<Step, _>>::INDEX, 1);
//! ```

use alloc::vec::Vec;
use core::marker::PhantomData;

use arrabbiata_field::Field;
use arrabbiata_poly::{Expr, Variable};

/// A gadget: constraints that hold on every row whose selector is on.
///
/// The constraints may read the `W` witness and `X` fixed columns at the
/// current and the next row. They must not mention the selector: the
/// protocol multiplies each of them by it.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::gate::Gadget;
/// use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
///
/// /// The Poseidon S-box: `b = a^5`.
/// struct SBox;
/// impl Gadget<F, 2, 0> for SBox {
///     const NAME: &'static str = "sbox";
///     fn constraints() -> Vec<Expr<F, 2, 0>> {
///         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::Curr);
///         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::Curr);
///         let a2 = a.clone() * a.clone();
///         vec![b - a2.clone() * a2 * a]
///     }
/// }
/// assert_eq!(SBox::constraints()[0].degree(), 5);
/// ```
pub trait Gadget<F, const W: usize, const X: usize>
where
    F: Field,
{
    /// A name, for errors and documentation.
    const NAME: &'static str;

    /// The constraints, each required to vanish on the rows that turn the
    /// gadget on.
    ///
    /// ```
    /// use arrabbiata_field::Field;
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{gate::Gadget, vanilla::Arithmetic};
    /// assert_eq!(<Arithmetic as Gadget<F, 3, 5>>::constraints().len(), 1);
    /// ```
    fn constraints() -> Vec<Expr<F, W, X>>;
}

/// A list of gadgets, as a type: `()` is empty and `(H, T)` puts `H` in
/// front of `T`. The gadget at position `i` is turned on by selector `i`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{gate::GateSet, vanilla::Vanilla};
/// assert_eq!(<() as GateSet<F, 3, 5>>::SELECTORS, 0);
/// assert_eq!(<Vanilla as GateSet<F, 3, 5>>::SELECTORS, 1);
/// ```
pub trait GateSet<F, const W: usize, const X: usize>
where
    F: Field,
{
    /// The number of gadgets, hence of selector columns.
    const SELECTORS: usize;

    /// The constraints of the gadget turned on by `selector`; empty when
    /// `selector >= SELECTORS`.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{gate::GateSet, vanilla::Vanilla};
    /// assert_eq!(<Vanilla as GateSet<F, 3, 5>>::constraints(0).len(), 1);
    /// assert!(<Vanilla as GateSet<F, 3, 5>>::constraints(1).is_empty());
    /// ```
    fn constraints(selector: usize) -> Vec<Expr<F, W, X>>;

    /// The name of the gadget turned on by `selector`.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonk::{gate::GateSet, vanilla::Vanilla};
    /// assert_eq!(<Vanilla as GateSet<F, 3, 5>>::name(0), Some("arithmetic"));
    /// assert_eq!(<Vanilla as GateSet<F, 3, 5>>::name(1), None);
    /// ```
    fn name(selector: usize) -> Option<&'static str>;
}

impl<F, const W: usize, const X: usize> GateSet<F, W, X> for ()
where
    F: Field,
{
    const SELECTORS: usize = 0;

    fn constraints(_: usize) -> Vec<Expr<F, W, X>> {
        Vec::new()
    }

    fn name(_: usize) -> Option<&'static str> {
        None
    }
}

impl<F, H, T, const W: usize, const X: usize> GateSet<F, W, X> for (H, T)
where
    F: Field,
    H: Gadget<F, W, X>,
    T: GateSet<F, W, X>,
{
    const SELECTORS: usize = 1 + T::SELECTORS;

    fn constraints(selector: usize) -> Vec<Expr<F, W, X>> {
        if selector == 0 {
            H::constraints()
        } else {
            T::constraints(selector - 1)
        }
    }

    fn name(selector: usize) -> Option<&'static str> {
        if selector == 0 {
            Some(H::NAME)
        } else {
            T::name(selector - 1)
        }
    }
}

/// The index of a gadget at the head of a gate set.
///
/// ```
/// use arrabbiata_plonk::gate::{Here, Member};
/// use arrabbiata_plonk::vanilla::{Arithmetic, Vanilla};
/// assert_eq!(<Vanilla as Member<Arithmetic, Here>>::INDEX, 0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Here;

/// The index of a gadget in the tail of a gate set, at index `I` there.
///
/// ```
/// use arrabbiata_plonk::gate::{Here, Member, There};
/// struct A;
/// struct B;
/// assert_eq!(<(A, (B, ())) as Member<B, There<Here>>>::INDEX, 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct There<I>(PhantomData<I>);

/// `Self` contains the gadget `G`, at the position witnessed by `I`.
///
/// `I` is inferred: write `Member<G, _>`, or let a builder call infer it.
/// A gadget outside the gate set has no index, so it cannot be turned on:
///
/// ```compile_fail
/// use arrabbiata_plonk::gate::Member;
/// struct A;
/// struct B;
/// let _ = <(A, ()) as Member<B, _>>::INDEX;
/// ```
///
/// ```
/// use arrabbiata_plonk::gate::Member;
/// struct A;
/// struct B;
/// assert_eq!(<(A, (B, ())) as Member<A, _>>::INDEX, 0);
/// ```
pub trait Member<G, I> {
    /// The position of `G`, which is the index of its selector.
    const INDEX: usize;
}

impl<H, T> Member<H, Here> for (H, T) {
    const INDEX: usize = 0;
}

impl<H, T, G, I> Member<G, There<I>> for (H, T)
where
    T: Member<G, I>,
{
    const INDEX: usize = 1 + T::INDEX;
}

/// The degree of `e` in all its variables, fixed columns included: the
/// degree of the polynomial it denotes once every column is interpolated.
pub(crate) fn total_degree<F, const W: usize, const X: usize>(e: &Expr<F, W, X>) -> usize {
    match e {
        Expr::Constant(_) => 0,
        Expr::Variable(_) => 1,
        Expr::Neg(a) => total_degree(a),
        Expr::Add(a, b) => {
            let (da, db) = (total_degree(a), total_degree(b));
            if da >= db { da } else { db }
        }
        Expr::Mul(a, b) => total_degree(a) + total_degree(b),
    }
}

/// Every variable of `e`, in syntactic order, with repetitions.
pub(crate) fn variables<F, const W: usize, const X: usize>(
    e: &Expr<F, W, X>,
    out: &mut Vec<Variable<W, X>>,
) {
    match e {
        Expr::Constant(_) => {}
        Expr::Variable(v) => out.push(*v),
        Expr::Neg(a) => variables(a, out),
        Expr::Add(a, b) | Expr::Mul(a, b) => {
            variables(a, out);
            variables(b, out);
        }
    }
}
