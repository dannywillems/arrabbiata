//! Kimchi's Generic gate (`polynomials/generic.rs`): two PlonK half-gates
//! per row, by what each computes.

use alloc::vec::Vec;

use arrabbiata_field::Field;
use arrabbiata_kimchi_gadgets::generic::GenericSpec;

use crate::{
    constraints::{CircuitGate, GateType, Wire},
    permutation::PERMUTS,
};

/// A half-gate (Kimchi's `GenericGateSpec`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenericGateSpec<F> {
    /// One of the half-gates the gadget computes.
    Spec(GenericSpec<F>),
    /// A public input: `left = x`, the public input added by the protocol.
    Pub,
}

impl<F> GenericGateSpec<F>
where
    F: Field,
{
    /// The five coefficients `[l, r, o, m, c]`.
    #[must_use]
    pub fn coefficients(&self) -> [F; 5] {
        match self {
            Self::Spec(spec) => spec.coefficients(),
            Self::Pub => [F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        }
    }
}

impl<F> CircuitGate<F>
where
    F: Field,
{
    /// A Generic row with the half-gates `first` and `second` (none: zero
    /// coefficients), Kimchi's `create_generic_gadget`.
    #[must_use]
    pub fn create_generic_gadget(
        wires: [Wire; PERMUTS],
        first: GenericGateSpec<F>,
        second: Option<GenericGateSpec<F>>,
    ) -> Self {
        let mut coeffs: Vec<F> = first.coefficients().to_vec();
        let rest = second.map_or([F::ZERO; 5], |s| s.coefficients());
        coeffs.extend(rest);
        Self::new(GateType::Generic, wires, coeffs)
    }
}
