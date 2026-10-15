//! Vanilla PlonK as an instance of the generic arithmetization.
//!
//! The arithmetization of the PlonK paper is one gadget, [`Arithmetic`],
//! over three witness columns `a, b, c` and five fixed columns
//! `q_L, q_R, q_O, q_M, q_C`:
//!
//! ```text
//! q_L a + q_R b + q_O c + q_M a b + q_C = 0.
//! ```
//!
//! Circuits written with [`arrabbiata_frontend`] compile to it with
//! [`circuit`] and [`witness`]: every gate row turns the gadget on, and
//! public inputs (column `a` of their row) are bound by the public-input
//! identity of the circuit instead of by a gate.

use alloc::{vec, vec::Vec};

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla as frontend;
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};

use crate::{
    CircuitError,
    circuit::{Circuit, Witness},
    gate::{Gadget, Selector},
};

/// The arithmetic gate of the PlonK paper,
/// `q_L a + q_R b + q_O c + q_M a b + q_C`, with `a, b, c` the witness
/// columns and `q_L, q_R, q_O, q_M, q_C` the fixed columns, in this order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Arithmetic;

impl<F> Gadget<F, 3, 5> for Arithmetic
where
    F: Field,
{
    const NAME: &'static str = "arithmetic";

    fn constraints() -> Vec<Expr<F, 3, 5>> {
        type E<F> = Expr<F, 3, 5>;
        let a = E::<F>::witness(WitnessColumn::new::<0>(), Rotation::CURR);
        let b = E::<F>::witness(WitnessColumn::new::<1>(), Rotation::CURR);
        let c = E::<F>::witness(WitnessColumn::new::<2>(), Rotation::CURR);
        let q_l = E::<F>::fixed(FixedColumn::new::<0>(), Rotation::CURR);
        let q_r = E::<F>::fixed(FixedColumn::new::<1>(), Rotation::CURR);
        let q_o = E::<F>::fixed(FixedColumn::new::<2>(), Rotation::CURR);
        let q_m = E::<F>::fixed(FixedColumn::new::<3>(), Rotation::CURR);
        let q_c = E::<F>::fixed(FixedColumn::new::<4>(), Rotation::CURR);
        vec![q_l * a.clone() + q_r * b.clone() + q_o * c + q_m * a * b + q_c]
    }
}

/// The gate set of vanilla PlonK: the arithmetic gadget alone.
pub type Vanilla = (Arithmetic, ());

/// A vanilla PlonK circuit.
pub type VanillaCircuit<F> = Circuit<F, Vanilla, 3, 5, 1>;

/// Compile a circuit of [`arrabbiata_frontend`] to vanilla PlonK.
///
/// # Errors
///
/// [`CircuitError::Shape`] if the frontend circuit is malformed (it is not,
/// when it comes from [`arrabbiata_frontend::vanilla::build`]).
pub fn circuit<F>(circuit: &frontend::Circuit<F>) -> Result<VanillaCircuit<F>, CircuitError>
where
    F: Field,
{
    let n = circuit.num_rows();
    let mut fixed = [
        vec![F::ZERO; n],
        vec![F::ZERO; n],
        vec![F::ZERO; n],
        vec![F::ZERO; n],
        vec![F::ZERO; n],
    ];
    for (row, gate) in circuit.gates().iter().enumerate() {
        fixed[0][row] = gate.q_l;
        fixed[1][row] = gate.q_r;
        fixed[2][row] = gate.q_o;
        fixed[3][row] = gate.q_m;
        fixed[4][row] = gate.q_c;
    }
    // A public row is bound by the public-input identity; its gate (which
    // the frontend writes as `a + PI = 0`) is turned off.
    let mut gadgets = vec![Some(Selector::new::<0>()); n];
    for row in circuit.public_rows() {
        gadgets[*row] = None;
    }
    Circuit::new(
        n,
        fixed,
        gadgets,
        circuit.sigma().to_vec(),
        circuit.public_rows().to_vec(),
    )
}

/// Compile a witness of [`arrabbiata_frontend`] to vanilla PlonK.
#[must_use]
pub fn witness<F>(witness: &frontend::Witness<F>) -> Witness<F, 3>
where
    F: Field,
{
    Witness::new(witness.wires().clone(), witness.public_inputs().to_vec())
}
