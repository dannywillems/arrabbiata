//! `Rotation` is an element of `Z` acting on the rows `Z/n` by
//! translation. These are the laws of that action.

use arrabbiata_poly::Rotation;
use proptest::prelude::*;

fn arb_rotation() -> impl Strategy<Value = Rotation> {
    prop_oneof![
        (-8i32..=8).prop_map(Rotation::new),
        any::<i32>().prop_map(Rotation::new),
        Just(Rotation::new(i32::MIN)),
        Just(Rotation::new(i32::MAX)),
    ]
}

/// A row count and a row in it.
fn arb_row() -> impl Strategy<Value = (usize, usize)> {
    (1usize..=64).prop_flat_map(|n| (Just(n), 0..n))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The identity acts trivially.
    #[test]
    fn identity((n, row) in arb_row()) {
        prop_assert_eq!(Rotation::CURR.apply(row, n), row);
    }

    /// `apply` lands in `0..n`.
    #[test]
    fn closed(r in arb_rotation(), (n, row) in arb_row()) {
        prop_assert!(r.apply(row, n) < n);
    }

    /// Compatibility: `apply(a + b) = apply(b) . apply(a)`.
    #[test]
    fn compatible(a in arb_rotation(), b in arb_rotation(), (n, row) in arb_row()) {
        if let Some(ab) = a.compose(b) {
            prop_assert_eq!(ab.apply(row, n), b.apply(a.apply(row, n), n));
        }
    }

    /// Composition is commutative and associative where defined.
    #[test]
    fn composition_is_commutative_and_associative(
        a in arb_rotation(),
        b in arb_rotation(),
        c in arb_rotation(),
    ) {
        prop_assert_eq!(a.compose(b), b.compose(a));
        let left = a.compose(b).and_then(|ab| ab.compose(c));
        let right = b.compose(c).and_then(|bc| a.compose(bc));
        if let (Some(l), Some(r)) = (left, right) {
            prop_assert_eq!(l, r);
        }
    }

    /// `compose` fails exactly when the offset leaves `i32`.
    #[test]
    fn compose_is_checked(a in arb_rotation(), b in arb_rotation()) {
        let sum = a.offset() as i64 + b.offset() as i64;
        let fits = i32::try_from(sum).is_ok();
        prop_assert_eq!(a.compose(b).is_some(), fits);
        if fits {
            prop_assert_eq!(a.compose(b).unwrap().offset() as i64, sum);
        }
    }

    /// The inverse undoes the rotation, on both sides.
    #[test]
    fn inverse(r in arb_rotation(), (n, row) in arb_row()) {
        match r.inverse() {
            Some(inv) => {
                prop_assert_eq!(inv.apply(r.apply(row, n), n), row);
                prop_assert_eq!(r.apply(inv.apply(row, n), n), row);
                prop_assert_eq!(r.compose(inv), Some(Rotation::CURR));
            }
            None => prop_assert_eq!(r.offset(), i32::MIN),
        }
    }

    /// The action factors through `Z/n`: offsets congruent mod `n` act the
    /// same, and `n` acts as the identity.
    #[test]
    fn periodic(k in -1000i32..1000, (n, row) in arb_row(), m in -4i32..4) {
        let shifted = Rotation::new(k + m * n as i32);
        prop_assert_eq!(shifted.apply(row, n), Rotation::new(k).apply(row, n));
        prop_assert_eq!(Rotation::new(n as i32).apply(row, n), row);
    }

    /// Each rotation is a bijection of `0..n`: a translation of `Z/n`.
    #[test]
    fn bijection(r in arb_rotation(), n in 1usize..=64) {
        let mut hit = vec![false; n];
        for row in 0..n {
            let image = r.apply(row, n);
            prop_assert!(!hit[image]);
            hit[image] = true;
        }
    }

    /// The action of one step is the cyclic successor.
    #[test]
    fn next_and_prev_are_the_successor_and_predecessor((n, row) in arb_row()) {
        prop_assert_eq!(Rotation::NEXT.apply(row, n), (row + 1) % n);
        prop_assert_eq!(Rotation::PREV.apply(row, n), (row + n - 1) % n);
    }

    /// The zigzag code is injective, and inverted by `(c >> 1) ^ -(c & 1)`.
    #[test]
    fn code_is_injective(a in arb_rotation(), b in arb_rotation()) {
        prop_assert_eq!(a.code() == b.code(), a == b);
        let c = a.code() as i64;
        prop_assert_eq!((c >> 1) ^ -(c & 1), a.offset() as i64);
        prop_assert!(a.code() <= u32::MAX as u64);
    }

    /// Rotations are ordered by offset.
    #[test]
    fn order_is_the_order_of_offsets(a in any::<i32>(), b in any::<i32>()) {
        prop_assert_eq!(Rotation::new(a).cmp(&Rotation::new(b)), a.cmp(&b));
    }
}

/// Known answers, with the extreme offsets.
#[test]
fn known_answers() {
    let cases = [
        (0, 0, 1, 0),
        (5, 3, 8, 0),
        (-1, 0, 8, 7),
        (-9, 0, 8, 7),
        (17, 2, 8, 3),
        (i32::MAX, 0, 8, 7),
        (i32::MIN, 0, 8, 0),
        (i32::MIN, 1, 3, 2),
        (1, 0, 1, 0),
    ];
    for (k, row, n, expected) in cases {
        assert_eq!(Rotation::new(k).apply(row, n), expected, "{k} {row} {n}");
    }
}
