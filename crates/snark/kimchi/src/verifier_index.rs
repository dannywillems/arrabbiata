//! Kimchi's verifier index (`kimchi/src/verifier_index.rs`, built by
//! `ProverIndex::verifier_index`): the circuit's parameters and the
//! commitments to its fixed columns.
//!
//! Every column is committed from its evaluations on `d1`, chunked by the
//! SRS ([`Srs::commit_non_hiding`]). Six selectors, those Kimchi always
//! builds (Generic, Poseidon, CompleteAdd, VarBaseMul, EndoMul,
//! EndoMulScalar), are also masked by `h` with blinder 1 (Kimchi's
//! `mask_fixed`), so that a selector that is zero everywhere still commits
//! to a point other than the identity; the permutation and coefficient
//! columns and the optional selectors are not. The optional selectors are
//! present only when the circuit uses their gate.
//!
//! The lookup part of the index (Kimchi's `LookupVerifierIndex`) comes with
//! the lookups.

use arrabbiata_curve::Curve;
use arrabbiata_field::{PrimeField, TwoAdicField};
use arrabbiata_transcript::Transcript;

use crate::{
    constraints::{COLUMNS, ConstraintSystem, GateType},
    oracle::FqSponge,
    permutation::PERMUTS,
    srs::{PolyComm, Srs},
};

/// The verifier index of a circuit over the scalar field of `C`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierIndex<C>
where
    C: Curve,
{
    /// The rows of `d1`.
    pub domain_size: usize,
    /// The chunk size, the SRS size.
    pub max_poly_size: usize,
    /// The zero-knowledge rows.
    pub zk_rows: usize,
    /// The public inputs.
    pub public: usize,
    /// The previous IPA challenges the proofs carry.
    pub prev_challenges: usize,
    /// The permutation shifts.
    pub shift: [C::Scalar; PERMUTS],
    /// The permutation columns.
    pub sigma_comm: [PolyComm<C>; PERMUTS],
    /// The coefficient columns.
    pub coefficients_comm: [PolyComm<C>; COLUMNS],
    /// The Generic selector, masked.
    pub generic_comm: PolyComm<C>,
    /// The Poseidon selector, masked.
    pub psm_comm: PolyComm<C>,
    /// The CompleteAdd selector, masked.
    pub complete_add_comm: PolyComm<C>,
    /// The VarBaseMul selector, masked.
    pub mul_comm: PolyComm<C>,
    /// The EndoMul selector, masked.
    pub emul_comm: PolyComm<C>,
    /// The EndoMulScalar selector, masked.
    pub endomul_scalar_comm: PolyComm<C>,
    /// The RangeCheck0 selector, if used.
    pub range_check0_comm: Option<PolyComm<C>>,
    /// The RangeCheck1 selector, if used.
    pub range_check1_comm: Option<PolyComm<C>>,
    /// The ForeignFieldAdd selector, if used.
    pub foreign_field_add_comm: Option<PolyComm<C>>,
    /// The ForeignFieldMul selector, if used.
    pub foreign_field_mul_comm: Option<PolyComm<C>>,
    /// The Xor16 selector, if used.
    pub xor_comm: Option<PolyComm<C>>,
    /// The Rot64 selector, if used.
    pub rot_comm: Option<PolyComm<C>>,
}

impl<C> VerifierIndex<C>
where
    C: Curve,
    C::Scalar: TwoAdicField,
{
    /// The verifier index of `cs` on `srs`.
    ///
    /// # Panics
    ///
    /// If the SRS is empty.
    #[must_use]
    pub fn create(cs: &ConstraintSystem<C::Scalar>, srs: &Srs<C>) -> Self
    where
        C::Scalar: PrimeField,
    {
        let d1 = cs.domain.d1;
        let rows = d1.size();
        let commit = |column: &[C::Scalar]| {
            let polynomial = match d1.interpolate(column) {
                Some(p) => p,
                None => unreachable!("every column has one value per row"),
            };
            srs.commit_non_hiding(polynomial.coefficients(), rows)
        };
        let fixed = |typ: GateType| srs.mask_fixed(commit(&cs.selector(typ)));
        let optional = |typ: GateType| cs.uses(typ).then(|| commit(&cs.selector(typ)));
        let sigmas = cs.sigmas();
        Self {
            domain_size: rows,
            max_poly_size: srs.max_poly_size(),
            zk_rows: cs.zk_rows,
            public: cs.public,
            prev_challenges: cs.prev_challenges,
            shift: cs.shift,
            sigma_comm: core::array::from_fn(|i| commit(&sigmas[i])),
            coefficients_comm: core::array::from_fn(|i| commit(&cs.coefficient(i))),
            generic_comm: fixed(GateType::Generic),
            psm_comm: fixed(GateType::Poseidon),
            complete_add_comm: fixed(GateType::CompleteAdd),
            mul_comm: fixed(GateType::VarBaseMul),
            emul_comm: fixed(GateType::EndoMul),
            endomul_scalar_comm: fixed(GateType::EndoMulScalar),
            range_check0_comm: optional(GateType::RangeCheck0),
            range_check1_comm: optional(GateType::RangeCheck1),
            foreign_field_add_comm: optional(GateType::ForeignFieldAdd),
            foreign_field_mul_comm: optional(GateType::ForeignFieldMul),
            xor_comm: optional(GateType::Xor16),
            rot_comm: optional(GateType::Rot64),
        }
    }
}

impl<C> VerifierIndex<C>
where
    C: Curve,
    C::Base: PrimeField,
{
    /// Kimchi's digest of the index: every commitment absorbed into a fresh
    /// base-field sponge over `sponge`, in Kimchi's order (sigmas,
    /// coefficients, the six masked selectors, then the optional ones as
    /// RangeCheck0, RangeCheck1, ForeignFieldMul, ForeignFieldAdd, Xor16,
    /// Rot64), then a squeeze. The parameters are not absorbed.
    #[must_use]
    pub fn digest<S>(&self, sponge: S) -> C::Base
    where
        S: Transcript<C::Base>,
    {
        let mut fq = FqSponge::<C, S>::new(sponge);
        let mut absorb = |c: &PolyComm<C>| fq.absorb_g(&c.chunks);
        for c in &self.sigma_comm {
            absorb(c);
        }
        for c in &self.coefficients_comm {
            absorb(c);
        }
        for c in [
            &self.generic_comm,
            &self.psm_comm,
            &self.complete_add_comm,
            &self.mul_comm,
            &self.emul_comm,
            &self.endomul_scalar_comm,
        ] {
            absorb(c);
        }
        for c in [
            &self.range_check0_comm,
            &self.range_check1_comm,
            &self.foreign_field_mul_comm,
            &self.foreign_field_add_comm,
            &self.xor_comm,
            &self.rot_comm,
        ]
        .into_iter()
        .flatten()
        {
            absorb(c);
        }
        fq.digest_fq()
    }
}
