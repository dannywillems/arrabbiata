//! Gadgets and gate sets.
//!
//! A [`Gadget`] is a named list of polynomial constraints over the `W`
//! witness columns and `X` fixed columns, each read at a fixed offset from
//! the constrained row (a [`Rotation`]). A
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
//! use arrabbiata_plonkish::gate::{Gadget, GateSet, Member};
//! use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
//!
//! /// `b = a^2` on the current row.
//! struct Square;
//! impl Gadget<F, 2, 0> for Square {
//!     const NAME: &'static str = "square";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
//!         vec![b - a.clone() * a]
//!     }
//! }
//!
//! /// `a' = a + b` on the next row.
//! struct Step;
//! impl Gadget<F, 2, 0> for Step {
//!     const NAME: &'static str = "step";
//!     fn constraints() -> Vec<Expr<F, 2, 0>> {
//!         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
//!         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
//!         let next = Expr::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
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
use arrabbiata_poly::{Expr, Rotation, Variable};
use arrabbiata_transcript::{Transcript, absorb_length};

/// The selector of one of `Q` gadgets: an index below `Q`.
///
/// A row of a circuit holds an `Option<Selector<Q>>`: no gadget, or exactly
/// one. The selector columns are derived from these, so they are one-hot by
/// construction.
///
/// ```
/// use arrabbiata_plonkish::gate::Selector;
/// let s = Selector::<3>::new::<2>();
/// assert_eq!(s.index(), 2);
/// assert_eq!(Selector::<3>::try_new(3), None);
/// ```
///
/// An index out of range is rejected at compile time:
///
/// ```compile_fail
/// use arrabbiata_plonkish::gate::Selector;
/// let _ = Selector::<3>::new::<3>();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Selector<const Q: usize>(usize);

impl<const Q: usize> Selector<Q> {
    /// The selector `I`, checked against `Q` at compile time.
    ///
    /// ```
    /// use arrabbiata_plonkish::gate::Selector;
    /// assert_eq!(Selector::<2>::new::<0>().index(), 0);
    /// ```
    #[must_use]
    pub const fn new<const I: usize>() -> Self {
        const { assert!(I < Q, "selector index out of range") };
        Self(I)
    }

    /// The selector `index`, or `None` if it is not below `Q`.
    ///
    /// ```
    /// use arrabbiata_plonkish::gate::Selector;
    /// assert!(Selector::<2>::try_new(1).is_some());
    /// assert!(Selector::<2>::try_new(2).is_none());
    /// ```
    #[must_use]
    pub const fn try_new(index: usize) -> Option<Self> {
        if index < Q { Some(Self(index)) } else { None }
    }

    /// The index, below `Q`.
    ///
    /// ```
    /// use arrabbiata_plonkish::gate::Selector;
    /// assert_eq!(Selector::<4>::new::<3>().index(), 3);
    /// ```
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A gadget: constraints that hold on every row whose selector is on.
///
/// The constraints may read the `W` witness and `X` fixed columns at any
/// [`Rotation`] of the constrained row. They must not mention the
/// selector: the
/// protocol multiplies each of them by it.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::gate::Gadget;
/// use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
///
/// /// The Poseidon S-box: `b = a^5`.
/// struct SBox;
/// impl Gadget<F, 2, 0> for SBox {
///     const NAME: &'static str = "sbox";
///     fn constraints() -> Vec<Expr<F, 2, 0>> {
///         let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
///         let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
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
    /// use arrabbiata_plonkish::{gate::Gadget, vanilla::Arithmetic};
    /// assert_eq!(<Arithmetic as Gadget<F, 3, 5>>::constraints().len(), 1);
    /// ```
    fn constraints() -> Vec<Expr<F, W, X>>;
}

/// A list of gadgets, as a type: `()` is empty and `(H, T)` puts `H` in
/// front of `T`. The gadget at position `i` is turned on by selector `i`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::GateSet, vanilla::Vanilla};
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
    /// use arrabbiata_plonkish::{gate::GateSet, vanilla::Vanilla};
    /// assert_eq!(<Vanilla as GateSet<F, 3, 5>>::constraints(0).len(), 1);
    /// assert!(<Vanilla as GateSet<F, 3, 5>>::constraints(1).is_empty());
    /// ```
    fn constraints(selector: usize) -> Vec<Expr<F, W, X>>;

    /// The name of the gadget turned on by `selector`.
    ///
    /// ```
    /// use arrabbiata_pasta::Fq as F;
    /// use arrabbiata_plonkish::{gate::GateSet, vanilla::Vanilla};
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
/// use arrabbiata_plonkish::gate::{Here, Member};
/// use arrabbiata_plonkish::vanilla::{Arithmetic, Vanilla};
/// assert_eq!(<Vanilla as Member<Arithmetic, Here>>::INDEX, 0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Here;

/// The index of a gadget in the tail of a gate set, at index `I` there.
///
/// ```
/// use arrabbiata_plonkish::gate::{Here, Member, There};
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
/// use arrabbiata_plonkish::gate::Member;
/// struct A;
/// struct B;
/// let _ = <(A, ()) as Member<B, _>>::INDEX;
/// ```
///
/// ```
/// use arrabbiata_plonkish::gate::Member;
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
///
/// It differs from [`Expr::degree`], the degree in the witness columns
/// alone, exactly when a fixed column multiplies a witness column.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::{Gadget, total_degree}, vanilla::Arithmetic};
/// let gate = &<Arithmetic as Gadget<F, 3, 5>>::constraints()[0];
/// // q_M a b: degree 3 in all columns, 2 in the witness.
/// assert_eq!(total_degree(gate), 3);
/// assert_eq!(gate.degree(), 2);
/// ```
#[must_use]
pub fn total_degree<F, const W: usize, const X: usize>(e: &Expr<F, W, X>) -> usize {
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

/// The smallest and largest row offsets the gadget turned on by `selector`
/// reads, `(0, 0)` for a gadget that reads only its own row.
pub(crate) fn span<F, G, const W: usize, const X: usize>(selector: usize) -> (i32, i32)
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut vars = Vec::new();
    for c in G::constraints(selector) {
        variables(&c, &mut vars);
    }
    vars.iter()
        .map(|v| match v {
            Variable::Witness(_, r) | Variable::Fixed(_, r) => r.offset(),
        })
        .fold((0, 0), |(lo, hi), k| (lo.min(k), hi.max(k)))
}

/// Every variable of `e`, in syntactic order, with repetitions, appended
/// to `out`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::gate::variables;
/// use arrabbiata_poly::{Expr, Rotation, Variable, WitnessColumn};
/// let a = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let mut out = Vec::new();
/// variables(&(a.clone() * a), &mut out);
/// assert_eq!(out.len(), 2);
/// assert!(matches!(out[0], Variable::Witness(_, Rotation::CURR)));
/// ```
pub fn variables<F, const W: usize, const X: usize>(
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

/// The degree of a gate set in the witness columns: the largest
/// [`Expr::degree`] of a constraint of one of its gadgets, `0` for a gate
/// set without constraints. Fixed columns are coefficients, of degree 0.
///
/// This is the degree to which a relaxation homogenizes every constraint.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::degree, vanilla::Vanilla};
/// assert_eq!(degree::<F, Vanilla, 3, 5>(), 2);
/// assert_eq!(degree::<F, (), 3, 5>(), 0);
/// ```
#[must_use]
pub fn degree<F, G, const W: usize, const X: usize>() -> usize
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut d = 0;
    for gadget in 0..G::SELECTORS {
        for c in G::constraints(gadget) {
            d = d.max(c.degree());
        }
    }
    d
}

/// The largest number of constraints of a gadget of the gate set: the
/// number of coefficients combining the constraints of a row. One-hot
/// selectors are what makes the maximum, and not the sum, enough: a row
/// turns on one gadget, so the gadgets can share the coefficients.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::combiner_length, vanilla::Vanilla};
/// assert_eq!(combiner_length::<F, Vanilla, 3, 5>(), 1);
/// assert_eq!(combiner_length::<F, (), 3, 5>(), 0);
/// ```
#[must_use]
pub fn combiner_length<F, G, const W: usize, const X: usize>() -> usize
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut m = 0;
    for gadget in 0..G::SELECTORS {
        m = m.max(G::constraints(gadget).len());
    }
    m
}

/// The rotations at which some constraint of the gate set reads a witness
/// column, sorted by offset and without repetition.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::{Gadget, witness_rotations}, vanilla::Vanilla};
/// use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
///
/// /// `a' = a`.
/// struct Repeat;
/// impl Gadget<F, 1, 0> for Repeat {
///     const NAME: &'static str = "repeat";
///     fn constraints() -> Vec<Expr<F, 1, 0>> {
///         let a = |r| Expr::witness(WitnessColumn::new::<0>(), r);
///         vec![a(Rotation::NEXT) - a(Rotation::CURR)]
///     }
/// }
/// assert_eq!(witness_rotations::<F, (Repeat, ()), 1, 0>(), [Rotation::CURR, Rotation::NEXT]);
/// assert_eq!(witness_rotations::<F, Vanilla, 3, 5>(), [Rotation::CURR]);
/// ```
#[must_use]
pub fn witness_rotations<F, G, const W: usize, const X: usize>() -> Vec<Rotation>
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut out: Vec<Rotation> = Vec::new();
    let mut vars = Vec::new();
    for gadget in 0..G::SELECTORS {
        for c in G::constraints(gadget) {
            variables(&c, &mut vars);
        }
    }
    for v in vars {
        if let Variable::Witness(_, r) = v
            && !out.contains(&r)
        {
            out.push(r);
        }
    }
    out.sort_by_key(|r| r.offset());
    out
}

/// Whether some constraint of the gadget turned on by `selector` reads a
/// fixed column.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::reads_fixed, vanilla::Vanilla};
/// assert!(reads_fixed::<F, Vanilla, 3, 5>(0));
/// assert!(!reads_fixed::<F, Vanilla, 3, 5>(1));
/// ```
#[must_use]
pub fn reads_fixed<F, G, const W: usize, const X: usize>(selector: usize) -> bool
where
    F: Field,
    G: GateSet<F, W, X>,
{
    let mut vars = Vec::new();
    for c in G::constraints(selector) {
        variables(&c, &mut vars);
    }
    vars.iter().any(|v| matches!(v, Variable::Fixed(_, _)))
}

/// Absorb `e` in prefix form, one tag per node: an injective encoding of
/// the expression tree, so a transcript that absorbs the constraints of a
/// gate set is bound to the relation and not only to its columns.
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::gate::absorb_expr;
/// use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
/// use arrabbiata_poseidon::vesta_sponge;
/// use arrabbiata_transcript::Transcript;
/// let a = Expr::<F, 1, 0>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
/// let (mut s, mut t) = (vesta_sponge(0), vesta_sponge(0));
/// absorb_expr(&(a.clone() * a.clone()), &mut s);
/// absorb_expr(&(a.clone() + a), &mut t);
/// assert_ne!(s.squeeze(), t.squeeze());
/// ```
pub fn absorb_expr<F, T, const W: usize, const X: usize>(e: &Expr<F, W, X>, transcript: &mut T)
where
    F: Field,
    T: Transcript<F>,
{
    let rotation = |r: Rotation| r.code() as usize;
    match e {
        Expr::Constant(c) => {
            absorb_length(transcript, 0);
            transcript.absorb(*c);
        }
        Expr::Variable(Variable::Witness(col, r)) => {
            absorb_length(transcript, 1);
            absorb_length(transcript, col.index());
            absorb_length(transcript, rotation(*r));
        }
        Expr::Variable(Variable::Fixed(col, r)) => {
            absorb_length(transcript, 2);
            absorb_length(transcript, col.index());
            absorb_length(transcript, rotation(*r));
        }
        Expr::Neg(a) => {
            absorb_length(transcript, 3);
            absorb_expr(a, transcript);
        }
        Expr::Add(a, b) => {
            absorb_length(transcript, 4);
            absorb_expr(a, transcript);
            absorb_expr(b, transcript);
        }
        Expr::Mul(a, b) => {
            absorb_length(transcript, 5);
            absorb_expr(a, transcript);
            absorb_expr(b, transcript);
        }
    }
}

/// Absorb the constraints of every gadget of `G`, in the order of `G`,
/// each list prefixed with its length.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonkish::{gate::absorb_gates, vanilla::Vanilla};
/// use arrabbiata_poseidon::vesta_sponge;
/// use arrabbiata_transcript::Transcript;
/// let (mut s, mut t) = (vesta_sponge(0), vesta_sponge(0));
/// absorb_gates::<F, Vanilla, _, 3, 5>(&mut s);
/// absorb_gates::<F, (), _, 3, 5>(&mut t);
/// assert_ne!(s.squeeze(), t.squeeze());
/// ```
pub fn absorb_gates<F, G, T, const W: usize, const X: usize>(transcript: &mut T)
where
    F: Field,
    G: GateSet<F, W, X>,
    T: Transcript<F>,
{
    for gadget in 0..G::SELECTORS {
        let constraints = G::constraints(gadget);
        absorb_length(transcript, constraints.len());
        for c in &constraints {
            absorb_expr(c, transcript);
        }
    }
}
