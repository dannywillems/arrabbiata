//! The constraints of each gate type, as a parameter of the prover.
//!
//! The quotient needs, at each point of a large enough domain, the value
//! of every constraint of the gate of each type; [`Gates`] is that map.
//! [`KimchiGates`] is Kimchi's: the constraints of `arrabbiata-kimchi-gadgets`,
//! checked there equal to Kimchi's polynomials, in Kimchi's order (so that
//! constraint `i` of a gate is the one Kimchi weights by `alpha^i`).

use alloc::vec::Vec;

use arrabbiata_field::PrimeField;
use arrabbiata_gadget::{Error, GateConstraints};
use arrabbiata_kimchi_gadgets::harness::{NAMES, gate_kinds};
use arrabbiata_poseidon::kimchi::KimchiPoseidon;

use crate::constraints::GateType;

/// The constraints of each gate type.
pub trait Gates<F> {
    /// The values of the constraints of `typ`, in order, at the cells
    /// `witness(column, rotation)` (rotation 0 the gate's row, 1 the next)
    /// and the coefficients `coefficient(column, rotation)`; empty for a
    /// gate with no constraint of its own.
    fn evaluate(
        &self,
        typ: GateType,
        witness: &dyn Fn(usize, usize) -> F,
        coefficient: &dyn Fn(usize, usize) -> F,
    ) -> Vec<F>;
}

/// Kimchi's gate type of each of Kimchi's gate names.
const TYPES: [(&str, GateType); 12] = [
    ("Generic", GateType::Generic),
    ("Poseidon", GateType::Poseidon),
    ("CompleteAdd", GateType::CompleteAdd),
    ("VarBaseMul", GateType::VarBaseMul),
    ("EndoMul", GateType::EndoMul),
    ("EndoMulScalar", GateType::EndoMulScalar),
    ("RangeCheck0", GateType::RangeCheck0),
    ("RangeCheck1", GateType::RangeCheck1),
    ("ForeignFieldAdd", GateType::ForeignFieldAdd),
    ("ForeignFieldMul", GateType::ForeignFieldMul),
    ("Xor16", GateType::Xor16),
    ("Rot64", GateType::Rot64),
];

/// The Kimchi gate type of the gadget named `gadget` (`kimchi-xor16` is
/// `Xor16`), by the harness's names; `kimchi-zero` is `Zero`.
#[must_use]
pub fn gate_type_of(gadget: &str) -> Option<GateType> {
    if gadget == "kimchi-zero" {
        return Some(GateType::Zero);
    }
    let kimchi = NAMES.iter().find(|(_, ours)| *ours == gadget)?.0;
    TYPES.iter().find(|(n, _)| *n == kimchi).map(|(_, t)| *t)
}

/// Kimchi's gates over `F`.
#[derive(Clone, Debug)]
pub struct KimchiGates<F> {
    kinds: Vec<(GateType, GateConstraints<F>)>,
}

impl<F> KimchiGates<F>
where
    F: PrimeField,
{
    /// The gates with Kimchi's Poseidon parameters over `F` and the
    /// endomorphism coefficient `endo` of the EndoMul gate (the GLV `beta`
    /// of the curve whose base field is `F`).
    ///
    /// # Errors
    ///
    /// An [`Error`] of the gadget builder, which the harnesses do not raise.
    pub fn new(poseidon: &KimchiPoseidon<F>, endo: F) -> Result<Self, Error> {
        let kinds = gate_kinds(poseidon, endo)?
            .into_iter()
            .filter_map(|(name, kind)| {
                TYPES
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, typ)| (*typ, kind))
            })
            .collect();
        Ok(Self { kinds })
    }
}

impl<F> Gates<F> for KimchiGates<F>
where
    F: PrimeField,
{
    fn evaluate(
        &self,
        typ: GateType,
        witness: &dyn Fn(usize, usize) -> F,
        coefficient: &dyn Fn(usize, usize) -> F,
    ) -> Vec<F> {
        self.kinds
            .iter()
            .find(|(t, _)| *t == typ)
            .map_or_else(Vec::new, |(_, kind)| kind.evaluate(witness, coefficient))
    }
}
