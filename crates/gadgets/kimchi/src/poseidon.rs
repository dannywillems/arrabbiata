//! Kimchi's Poseidon gate: five rounds of the permutation on one row.
//!
//! A row holds the state before its first round and after each of its five
//! rounds, three columns each, in the order [`STATE_ORDER`]: the input in
//! columns 0-2, then the states after rounds 1, 2, 3 in columns 6-8, 9-11,
//! 12-14, the state after round 4 in columns 3-5 (so that the permutation
//! can reach it), and the state after round 5 in columns 0-2 of the next
//! row, where the next gate reads it. The round constants of the row's five
//! rounds are its 15 coefficients, `coefficient[3 r + i]` being constant
//! `i` of round `r`.
//!
//! Each round is `x -> MDS x^7 + c`, so a row has 15 constraints of degree
//! 7, the same on every row: 11 rows ([`PoseidonPermutation`]) are the 55
//! rounds of [`arrabbiata_poseidon::kimchi`], under one selector.

use arrabbiata_field::Field;
use arrabbiata_gadget::{Array, Cell, Env, Of, TypedGadget};
use arrabbiata_poseidon::kimchi::{KimchiPoseidon, ROUNDS, WIDTH};

/// The rounds on one row.
pub const ROUNDS_PER_ROW: usize = 5;

/// The rows of one permutation.
pub const ROWS_PER_PERMUTATION: usize = ROUNDS / ROUNDS_PER_ROW;

/// The slot of the state before round `r` of a row (and, for `r = 0`, of
/// the state after the row's last round, on the next row): slot `s` is
/// columns `3 s .. 3 s + 3`.
pub const STATE_ORDER: [usize; ROUNDS_PER_ROW] = [0, 2, 3, 4, 1];

/// `x^7`.
fn sbox<V>(x: V) -> V
where
    V: Clone + core::ops::Mul<Output = V>,
{
    let square = x.clone() * x.clone();
    let fourth = square.clone() * square.clone();
    x * square * fourth
}

/// One row: rounds `first_round .. first_round + 5` of `poseidon`.
///
/// Its input must be in columns 0-2 of its row already, as the previous
/// row's output is (and as [`PoseidonPermutation`] puts the first input):
/// placing it there inside the gate would give the gate a constraint the
/// other rows do not have.
#[derive(Clone, Copy, Debug)]
pub struct PoseidonRow<'a, F> {
    /// The permutation, whose MDS matrix the constraints use.
    pub poseidon: &'a KimchiPoseidon<F>,
    /// The first round of the row.
    pub first_round: usize,
}

impl<F> TypedGadget<F> for PoseidonRow<'_, F>
where
    F: Field,
{
    type Input = Array<WIDTH>;
    type Output = Array<WIDTH>;
    const NAME: &'static str = "kimchi-poseidon";
    const ROWS: usize = 1;

    fn synthesize<E>(
        &self,
        env: &mut E,
        input: Of<Array<WIDTH>, E::Var>,
    ) -> Of<Array<WIDTH>, E::Var>
    where
        E: Env<F>,
    {
        let mds = self.poseidon.mds();
        let mut state = input;
        for r in 0..ROUNDS_PER_ROW {
            let constants = self.poseidon.round_constants()[self.first_round + r];
            let powered = state.clone().map(sbox);
            let slot = STATE_ORDER[(r + 1) % ROUNDS_PER_ROW];
            let row = usize::from(r + 1 == ROUNDS_PER_ROW);
            state = core::array::from_fn(|i| {
                let constant = env.fixed(Cell::new(WIDTH * r + i, 0), constants[i]);
                let mut next = constant;
                for (m, x) in mds[i].iter().zip(&powered) {
                    let m = env.constant(*m);
                    next = next + m * x.clone();
                }
                env.define(Cell::new(WIDTH * slot + i, row), next)
            });
        }
        state
    }
}

/// The 55 rounds, on 11 rows, the output on the first three columns of the
/// row after.
#[derive(Clone, Copy, Debug)]
pub struct PoseidonPermutation<'a, F> {
    /// The permutation.
    pub poseidon: &'a KimchiPoseidon<F>,
}

impl<F> TypedGadget<F> for PoseidonPermutation<'_, F>
where
    F: Field,
{
    type Input = Array<WIDTH>;
    type Output = Array<WIDTH>;
    const NAME: &'static str = "kimchi-poseidon-permutation";
    const ROWS: usize = ROWS_PER_PERMUTATION;

    fn synthesize<E>(
        &self,
        env: &mut E,
        input: Of<Array<WIDTH>, E::Var>,
    ) -> Of<Array<WIDTH>, E::Var>
    where
        E: Env<F>,
    {
        // The input on the first row, where the first gate reads it; the
        // gates then find their input in place.
        let mut state = core::array::from_fn(|i| env.place(Cell::new(i, 0), &input[i]));
        for row in 0..ROWS_PER_PERMUTATION {
            let gate = PoseidonRow {
                poseidon: self.poseidon,
                first_round: ROUNDS_PER_ROW * row,
            };
            state = env.call(&gate, state);
        }
        state
    }
}
