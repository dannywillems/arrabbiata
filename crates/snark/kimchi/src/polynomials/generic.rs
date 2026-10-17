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

/// Kimchi's `generic::testing`: the Generic circuit its tests prove.
pub mod testing {
    use alloc::vec::Vec;

    use arrabbiata_field::Field;
    use arrabbiata_kimchi_gadgets::generic::GenericSpec;

    use super::GenericGateSpec;
    use crate::constraints::{COLUMNS, CircuitGate, Wire};

    /// The left half-gate of the add and mul rows: `l + 3 r = o`.
    fn add<F: Field>() -> GenericSpec<F> {
        GenericSpec::Add {
            left: None,
            right: Some(F::from_u64(3)),
            output: None,
        }
    }

    /// The right half-gate: `2 l r = o`.
    fn mul<F: Field>() -> GenericSpec<F> {
        GenericSpec::Mul {
            output: None,
            mul: Some(F::from_u64(2)),
        }
    }

    /// From `start_row`: `public` public rows, 10 rows of `l + 3 r = o` and
    /// `2 l r = o`, 10 rows of `l = 3` and `l = 5` (Kimchi's
    /// `create_circuit`).
    #[must_use]
    pub fn create_circuit<F: Field>(start_row: usize, public: usize) -> Vec<CircuitGate<F>> {
        let mut rows = start_row..;
        let mut gates = Vec::new();
        for r in rows.by_ref().take(public) {
            let gate =
                CircuitGate::create_generic_gadget(Wire::for_row(r), GenericGateSpec::Pub, None);
            gates.push(gate);
        }
        for r in rows.by_ref().take(10) {
            let first = GenericGateSpec::Spec(add());
            let second = GenericGateSpec::Spec(mul());
            gates.push(CircuitGate::create_generic_gadget(
                Wire::for_row(r),
                first,
                Some(second),
            ));
        }
        for r in rows.take(10) {
            let first = GenericGateSpec::Spec(GenericSpec::Const(F::from_u64(3)));
            let second = GenericGateSpec::Spec(GenericSpec::Const(F::from_u64(5)));
            gates.push(CircuitGate::create_generic_gadget(
                Wire::for_row(r),
                first,
                Some(second),
            ));
        }
        gates
    }

    /// The witness of [`create_circuit`] from `start_row` (Kimchi's
    /// `fill_in_witness`): the public inputs, `11, 23` into both half-gates
    /// with the outputs the half-gates compute, and the constants.
    ///
    /// # Panics
    ///
    /// If a column is shorter than the circuit.
    pub fn fill_in_witness<F: Field>(
        start_row: usize,
        witness: &mut [Vec<F>; COLUMNS],
        public: &[F],
    ) {
        let mut rows = start_row..;
        for p in public {
            let Some(r) = rows.next() else { break };
            witness[0][r] = *p;
        }
        let (l, r_value) = (F::from_u64(11), F::from_u64(23));
        let output = |spec: GenericSpec<F>| match spec.output(l, r_value) {
            Some(o) => o,
            None => unreachable!("the add and mul half-gates have an output"),
        };
        for r in rows.by_ref().take(10) {
            let values = [l, r_value, output(add()), l, r_value, output(mul())];
            for (c, v) in values.into_iter().enumerate() {
                witness[c][r] = v;
            }
        }
        for r in rows.take(10) {
            witness[0][r] = F::from_u64(3);
            witness[3][r] = F::from_u64(5);
        }
    }
}
