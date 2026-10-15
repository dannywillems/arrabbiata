//! Heterogeneous lists: the value-level counterpart of a protocol pattern.
//!
//! The proof of a Fiat-Shamir protocol is the list of the prover's messages,
//! each of its own type. It is therefore an element of a product type whose
//! factors are fixed by the protocol: an `HCons<M1, HCons<M2, HNil>>`. See
//! [`crate::typed::Pattern::Proof`].

/// The empty list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HNil;

/// A list with head `H` and tail `T`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HCons<H, T> {
    /// The first element.
    pub head: H,
    /// The rest of the list.
    pub tail: T,
}

impl<H, T> HCons<H, T> {
    /// The list `head :: tail`.
    pub const fn new(head: H, tail: T) -> Self {
        Self { head, tail }
    }
}

/// A heterogeneous list.
pub trait HList {
    /// The number of elements, known at compile time.
    const LEN: usize;
}

impl HList for HNil {
    const LEN: usize = 0;
}

impl<H, T> HList for HCons<H, T>
where
    T: HList,
{
    const LEN: usize = 1 + T::LEN;
}

/// List reversal onto an accumulator: `rev(xs) ++ acc`.
///
/// The prover writes its messages as it goes, newest first (a snoc list);
/// reversing yields the proof in protocol order.
pub trait Reverse<Acc> {
    /// The type of `rev(self) ++ Acc`.
    type Output;

    /// `rev(self) ++ acc`.
    fn reverse_onto(self, acc: Acc) -> Self::Output;
}

impl<Acc> Reverse<Acc> for HNil {
    type Output = Acc;

    fn reverse_onto(self, acc: Acc) -> Acc {
        acc
    }
}

impl<H, T, Acc> Reverse<Acc> for HCons<H, T>
where
    T: Reverse<HCons<H, Acc>>,
{
    type Output = <T as Reverse<HCons<H, Acc>>>::Output;

    fn reverse_onto(self, acc: Acc) -> Self::Output {
        self.tail.reverse_onto(HCons::new(self.head, acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_reverses() {
        let list = HCons::new(1u8, HCons::new("two", HCons::new(3u64, HNil)));
        let reversed: HCons<u64, HCons<&str, HCons<u8, HNil>>> = list.reverse_onto(HNil);
        assert_eq!(reversed.head, 3);
        assert_eq!(reversed.tail.head, "two");
        assert_eq!(reversed.tail.tail.head, 1);
        const { assert!(<HCons<u8, HCons<u8, HNil>> as HList>::LEN == 2) };
    }
}
