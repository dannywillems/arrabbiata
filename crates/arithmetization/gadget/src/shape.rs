//! Shapes: the types of a gadget's input and output.
//!
//! A shape is a type constructor `V -> Of<V>` holding a fixed number of
//! `V`s, so one shape types both the symbolic variables a gadget reads while
//! building constraints and the field elements it reads while computing a
//! witness. Every shape of size `k` is isomorphic to `V^k`; [`Shape::flatten`]
//! and [`Shape::unflatten`] are that isomorphism, and products of shapes are
//! tuples (of two, three or four shapes).
//!
//! Composition is checked on the shape, not on its instances: a gadget whose
//! output shape is `Array<Scalar, 3>` composes with one whose input shape is
//! `Array<Scalar, 3>`, for every variable type at once.
//!
//! [`Array<S, N>`] holds `N` values of the shape `S`, so its type says what
//! it holds: `Array<Scalar, 2>` is two variables, `Array<Array<Scalar, 4>, 3>`
//! three groups of four. As a functor it is the composite `V -> [S(V); N]`.

use alloc::vec::Vec;

/// A fixed-size container type.
pub trait Shape {
    /// The number of elements.
    const SIZE: usize;

    /// The container of `V`s.
    type Of<V: Clone>: Clone;

    /// The elements, in order.
    fn flatten<V: Clone>(value: &Self::Of<V>) -> Vec<V>;

    /// The container of the first [`Shape::SIZE`] elements of `values`.
    ///
    /// # Panics
    ///
    /// If `values` has fewer than [`Shape::SIZE`] elements.
    fn unflatten<V: Clone>(values: &[V]) -> Self::Of<V>;
}

/// `S`'s container of `V`s.
pub type Of<S, V> = <S as Shape>::Of<V>;

/// No element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unit;

impl Shape for Unit {
    const SIZE: usize = 0;
    type Of<V: Clone> = ();

    fn flatten<V: Clone>((): &()) -> Vec<V> {
        Vec::new()
    }

    fn unflatten<V: Clone>(_: &[V]) {}
}

/// One element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar;

impl Shape for Scalar {
    const SIZE: usize = 1;
    type Of<V: Clone> = V;

    fn flatten<V: Clone>(value: &V) -> Vec<V> {
        alloc::vec![value.clone()]
    }

    fn unflatten<V: Clone>(values: &[V]) -> V {
        values[0].clone()
    }
}

/// `N` values of the shape `S`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Array<S, const N: usize>(core::marker::PhantomData<S>);

impl<S, const N: usize> Shape for Array<S, N>
where
    S: Shape,
{
    const SIZE: usize = N * S::SIZE;
    type Of<V: Clone> = [S::Of<V>; N];

    fn flatten<V: Clone>(value: &Self::Of<V>) -> Vec<V> {
        value.iter().flat_map(S::flatten).collect()
    }

    fn unflatten<V: Clone>(values: &[V]) -> Self::Of<V> {
        core::array::from_fn(|i| S::unflatten(&values[i * S::SIZE..]))
    }
}

impl<A, B> Shape for (A, B)
where
    A: Shape,
    B: Shape,
{
    const SIZE: usize = A::SIZE + B::SIZE;
    type Of<V: Clone> = (A::Of<V>, B::Of<V>);

    fn flatten<V: Clone>((a, b): &Self::Of<V>) -> Vec<V> {
        let mut values = A::flatten(a);
        values.extend(B::flatten(b));
        values
    }

    fn unflatten<V: Clone>(values: &[V]) -> Self::Of<V> {
        let (a, b) = values.split_at(A::SIZE);
        (A::unflatten(a), B::unflatten(b))
    }
}

impl<A, B, C> Shape for (A, B, C)
where
    A: Shape,
    B: Shape,
    C: Shape,
{
    const SIZE: usize = A::SIZE + B::SIZE + C::SIZE;
    type Of<V: Clone> = (A::Of<V>, B::Of<V>, C::Of<V>);

    fn flatten<V: Clone>((a, b, c): &Self::Of<V>) -> Vec<V> {
        let mut values = A::flatten(a);
        values.extend(B::flatten(b));
        values.extend(C::flatten(c));
        values
    }

    fn unflatten<V: Clone>(values: &[V]) -> Self::Of<V> {
        let (a, rest) = values.split_at(A::SIZE);
        let (b, c) = rest.split_at(B::SIZE);
        (A::unflatten(a), B::unflatten(b), C::unflatten(c))
    }
}

impl<A, B, C, D> Shape for (A, B, C, D)
where
    A: Shape,
    B: Shape,
    C: Shape,
    D: Shape,
{
    const SIZE: usize = A::SIZE + B::SIZE + C::SIZE + D::SIZE;
    type Of<V: Clone> = (A::Of<V>, B::Of<V>, C::Of<V>, D::Of<V>);

    fn flatten<V: Clone>((a, b, c, d): &Self::Of<V>) -> Vec<V> {
        let mut values = A::flatten(a);
        values.extend(B::flatten(b));
        values.extend(C::flatten(c));
        values.extend(D::flatten(d));
        values
    }

    fn unflatten<V: Clone>(values: &[V]) -> Self::Of<V> {
        let (a, rest) = values.split_at(A::SIZE);
        let (b, rest) = rest.split_at(B::SIZE);
        let (c, d) = rest.split_at(C::SIZE);
        (
            A::unflatten(a),
            B::unflatten(b),
            C::unflatten(c),
            D::unflatten(d),
        )
    }
}
