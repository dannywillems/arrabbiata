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
//! identity of the generic protocol instead of by a gate.
//!
//! ```
//! use arrabbiata_field::Field;
//! use arrabbiata_frontend::vanilla::build;
//! use arrabbiata_pasta::Fq as F;
//! use arrabbiata_plonk::{ipa, prove, setup, vanilla, verify};
//!
//! // x^3 + x + 5 = 35, with x = 3 private and 35 public.
//! let (c, w, ()) = build::<F, _, _>(|b| {
//!     let x = b.witness(F::from_u64(3));
//!     let x2 = b.mul(&x, &x);
//!     let x3 = b.mul(&x2, &x);
//!     let sum = b.add(&x3, &x);
//!     let lhs = b.add_constant(&sum, F::from_u64(5));
//!     let rhs = b.public_input(F::from_u64(35));
//!     b.assert_equal(&lhs, &rhs);
//! });
//! let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
//! let pcs = ipa::pallas(&circuit);
//! let (pk, vk) = setup(&circuit, &pcs).unwrap();
//! let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
//! verify(&vk, &pcs, ipa::pallas_transcript(), &[F::from_u64(35)], proof).unwrap();
//! ```

use alloc::{vec, vec::Vec};

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla as frontend;
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};

use crate::{
    PlonkError,
    circuit::{Circuit, Witness},
    gate::{Gadget, Selector},
};

/// The arithmetic gate of the PlonK paper,
/// `q_L a + q_R b + q_O c + q_M a b + q_C`, with `a, b, c` the witness
/// columns and `q_L, q_R, q_O, q_M, q_C` the fixed columns, in this order.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{gate::Gadget, vanilla::Arithmetic};
/// let gate = &<Arithmetic as Gadget<F, 3, 5>>::constraints()[0];
/// // Degree 2 in the witness (q_M a b).
/// assert_eq!(gate.degree(), 2);
/// ```
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
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{gate::GateSet, vanilla::Vanilla};
/// assert_eq!(<Vanilla as GateSet<F, 3, 5>>::SELECTORS, 1);
/// ```
pub type Vanilla = (Arithmetic, ());

/// A vanilla PlonK circuit.
///
/// ```
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::vanilla::{self, VanillaCircuit};
/// let (c, _, ()) = build::<F, _, _>(|_| ());
/// let circuit: VanillaCircuit<F> = vanilla::circuit(&c).unwrap();
/// assert_eq!(circuit.num_rows(), c.num_rows());
/// ```
pub type VanillaCircuit<F> = Circuit<F, Vanilla, 3, 5, 1>;

/// Vanilla PlonK over `F`, with the commitment scheme `PC` and the sponge
/// `S`.
///
/// ```
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::{ipa::PallasIpa, vanilla::VanillaPlonk};
/// use arrabbiata_poseidon::PoseidonSponge;
/// use arrabbiata_transcript::typed::{Pattern, Protocol};
/// type P = <VanillaPlonk<F, PallasIpa, PoseidonSponge<F>> as Protocol>::Transcript;
/// assert_eq!(<P as Pattern>::MESSAGES, 5);
/// ```
pub type VanillaPlonk<F, PC, S> = crate::Plonk<F, PC, S, Vanilla, 3, 5, 1>;

/// Compile a circuit of [`arrabbiata_frontend`] to vanilla PlonK.
///
/// # Errors
///
/// [`PlonkError::Shape`] if the frontend circuit is malformed (it is not,
/// when it comes from [`arrabbiata_frontend::vanilla::build`]).
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::vanilla;
/// let (c, w, ()) = build::<F, _, _>(|b| {
///     let x = b.witness(F::from_u64(2));
///     let y = b.mul(&x, &x);
///     let out = b.public_input(F::from_u64(4));
///     b.assert_equal(&y, &out);
/// });
/// let circuit = vanilla::circuit(&c).unwrap();
/// assert!(circuit.is_satisfied(&vanilla::witness(&w)));
/// ```
pub fn circuit<F>(circuit: &frontend::Circuit<F>) -> Result<VanillaCircuit<F>, PlonkError>
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
///
/// ```
/// use arrabbiata_field::Field;
/// use arrabbiata_frontend::vanilla::build;
/// use arrabbiata_pasta::Fq as F;
/// use arrabbiata_plonk::vanilla;
/// let (_, w, ()) = build::<F, _, _>(|b| {
///     b.public_input(F::from_u64(4));
/// });
/// assert_eq!(vanilla::witness(&w).public_inputs(), &[F::from_u64(4)]);
/// ```
#[must_use]
pub fn witness<F>(witness: &frontend::Witness<F>) -> Witness<F, 3>
where
    F: Field,
{
    Witness::new(witness.wires().clone(), witness.public_inputs().to_vec())
}
