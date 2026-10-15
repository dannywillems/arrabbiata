//! Kimchi's Zero gate: a row with no constraints of its own, which holds
//! cells the gate above reads as its next row (the bound of a foreign
//! field addition, the last limbs of a range check).

use arrabbiata_field::Field;
use arrabbiata_gadget::{Env, Of, TypedGadget, Unit};

/// One row, no constraints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Zero;

impl<F> TypedGadget<F> for Zero
where
    F: Field,
{
    type Input = Unit;
    type Output = Unit;
    const NAME: &'static str = "kimchi-zero";
    const ROWS: usize = 1;

    fn synthesize<E>(&self, _env: &mut E, (): Of<Unit, E::Var>) -> Of<Unit, E::Var>
    where
        E: Env<F>,
    {
    }
}
