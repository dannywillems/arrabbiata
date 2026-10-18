//! Kimchi's proofs on the workspace codec (`arrabbiata-codec`): the bytes
//! of `rmp_serde::to_vec` of Kimchi's `ProverProof`, every struct the array
//! of its fields in declaration order, every scalar and point arkworks'
//! compressed bytes.
//!
//! The impls here are products of the codecs of the fields; the fields'
//! codecs are the containers' and the scalars' and points' of their own
//! crates. Kimchi's proof differs from ours in shape only, and that is all
//! the impls below carry:
//!
//! - Kimchi's evaluations list every optional selector, every lookup
//!   column and five sorted columns, each an option, in a fixed order;
//!   ours keep the present ones. The encoding writes Kimchi's 26 fields;
//!   the decoding keeps the present ones, in Kimchi's order, which is the
//!   order our prover makes them in, so decoding inverts encoding.
//! - Kimchi's previous accumulator commits to `b` as a commitment; ours as
//!   a point, which is a commitment of one chunk.

use alloc::vec::Vec;

use arrabbiata_codec::{Codec, Decodable, Encodable, Error, Input, msgpack};
use arrabbiata_curve::{Curve, Projective};
use arrabbiata_field::{PrimeField, Sqrt};
use arrabbiata_ipa::RecursionChallenge;

use crate::{
    constraints::GateType,
    lookup::LookupPattern,
    prover::{
        LookupCommitments, LookupEvaluations, OpeningProof, PointEvaluations, ProofEvaluations,
        ProverProof,
    },
    srs::PolyComm,
};

/// The sorted lookup columns Kimchi's evaluations have room for.
const SORTED: usize = 5;

/// The fields of Kimchi's evaluations.
const EVALUATION_FIELDS: usize = 26;

/// The optional selectors, in the order of Kimchi's evaluations.
const OPTIONAL_SELECTORS: [GateType; 6] = [
    GateType::RangeCheck0,
    GateType::RangeCheck1,
    GateType::ForeignFieldAdd,
    GateType::ForeignFieldMul,
    GateType::Xor16,
    GateType::Rot64,
];

/// The lookup selectors, in the order of Kimchi's evaluations.
const LOOKUP_SELECTORS: [LookupPattern; 4] = [
    LookupPattern::Xor,
    LookupPattern::Lookup,
    LookupPattern::RangeCheck,
    LookupPattern::ForeignFieldMul,
];

impl<C> Encodable for PolyComm<C>
where
    C: Curve<Base: PrimeField>,
{
    fn write(&self, out: &mut Vec<u8>) {
        (&self.chunks,).write(out);
    }
}

impl<C> Decodable<()> for PolyComm<C>
where
    C: Curve<Base: PrimeField + Sqrt>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        let (chunks,) = <(Vec<Projective<C>>,)>::read(input, ())?;
        Ok(Self { chunks })
    }
}

impl<F> Encodable for PointEvaluations<F>
where
    F: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        (&self.zeta, &self.zeta_omega).write(out);
    }
}

impl<F> Decodable<()> for PointEvaluations<F>
where
    F: Decodable<()>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        let (zeta, zeta_omega) = <(Vec<F>, Vec<F>)>::read(input, ())?;
        Ok(Self { zeta, zeta_omega })
    }
}

impl<C> Encodable for OpeningProof<C>
where
    C: Curve<Base: PrimeField, Scalar: Encodable>,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, 5);
        self.lr.write(out);
        self.delta.write(out);
        self.z1.write(out);
        self.z2.write(out);
        self.sg.write(out);
    }
}

impl<C> Decodable<()> for OpeningProof<C>
where
    C: Curve<Base: PrimeField + Sqrt, Scalar: Decodable<()>>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        msgpack::read_fields(input, 5)?;
        Ok(Self {
            lr: Vec::read(input, ())?,
            delta: Projective::read(input, ())?,
            z1: C::Scalar::read(input, ())?,
            z2: C::Scalar::read(input, ())?,
            sg: Projective::read(input, ())?,
        })
    }
}

impl<C> Encodable for LookupCommitments<C>
where
    C: Curve<Base: PrimeField>,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, 3);
        self.sorted.write(out);
        self.aggreg.write(out);
        self.runtime.write(out);
    }
}

impl<C> Decodable<()> for LookupCommitments<C>
where
    C: Curve<Base: PrimeField + Sqrt>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        msgpack::read_fields(input, 3)?;
        Ok(Self {
            sorted: Vec::read(input, ())?,
            aggreg: PolyComm::read(input, ())?,
            runtime: Option::read(input, ())?,
        })
    }
}

impl<F> Encodable for ProofEvaluations<F>
where
    F: Encodable,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, EVALUATION_FIELDS);
        // Kimchi's public evaluations are an option, always present.
        Some(&self.public).write(out);
        self.w.write(out);
        self.z.write(out);
        self.s.write(out);
        self.coefficients.write(out);
        for x in [
            &self.generic_selector,
            &self.poseidon_selector,
            &self.complete_add_selector,
            &self.mul_selector,
            &self.emul_selector,
            &self.endomul_scalar_selector,
        ] {
            x.write(out);
        }
        for typ in OPTIONAL_SELECTORS {
            let mut found = None;
            for (t, x) in &self.optional_selectors {
                if *t == typ {
                    found = Some(x);
                }
            }
            found.write(out);
        }
        let l = self.lookup.as_ref();
        l.map(|l| &l.aggregation).write(out);
        l.map(|l| &l.table).write(out);
        let sorted: [Option<&PointEvaluations<F>>; SORTED] =
            core::array::from_fn(|i| l.and_then(|l| l.sorted.get(i)));
        sorted.write(out);
        l.and_then(|l| l.runtime_table.as_ref()).write(out);
        l.and_then(|l| l.runtime_selector.as_ref()).write(out);
        for pattern in LOOKUP_SELECTORS {
            let mut found = None;
            if let Some(l) = l {
                for (p, x) in &l.selectors {
                    if *p == pattern {
                        found = Some(x);
                    }
                }
            }
            found.write(out);
        }
    }
}

impl<F> Decodable<()> for ProofEvaluations<F>
where
    F: Decodable<()>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        msgpack::read_fields(input, EVALUATION_FIELDS)?;
        let Some(public) = Option::read(input, ())? else {
            return Err(Error::Unexpected("the public evaluations"));
        };
        let w = Decodable::read(input, ())?;
        let z = PointEvaluations::read(input, ())?;
        let s = Decodable::read(input, ())?;
        let coefficients = Decodable::read(input, ())?;
        let generic_selector = PointEvaluations::read(input, ())?;
        let poseidon_selector = PointEvaluations::read(input, ())?;
        let complete_add_selector = PointEvaluations::read(input, ())?;
        let mul_selector = PointEvaluations::read(input, ())?;
        let emul_selector = PointEvaluations::read(input, ())?;
        let endomul_scalar_selector = PointEvaluations::read(input, ())?;
        let mut optional_selectors = Vec::new();
        for typ in OPTIONAL_SELECTORS {
            if let Some(x) = Option::read(input, ())? {
                optional_selectors.push((typ, x));
            }
        }
        let aggregation: Option<PointEvaluations<F>> = Option::read(input, ())?;
        let table: Option<PointEvaluations<F>> = Option::read(input, ())?;
        let sorted: [Option<PointEvaluations<F>>; SORTED] = Decodable::read(input, ())?;
        let runtime_table = Option::read(input, ())?;
        let runtime_selector = Option::read(input, ())?;
        let mut selectors = Vec::new();
        for pattern in LOOKUP_SELECTORS {
            if let Some(x) = Option::read(input, ())? {
                selectors.push((pattern, x));
            }
        }
        // Our lookup evaluations are all present or all absent; Kimchi's
        // fields of a proof without lookups are all `nil`.
        let lookup = match (aggregation, table) {
            (Some(aggregation), Some(table)) => Some(LookupEvaluations {
                aggregation,
                table,
                sorted: sorted.into_iter().flatten().collect(),
                runtime_table,
                runtime_selector,
                selectors,
            }),
            (None, None) => {
                let none = sorted.iter().all(Option::is_none)
                    && runtime_table.is_none()
                    && runtime_selector.is_none()
                    && selectors.is_empty();
                if !none {
                    return Err(Error::Invalid("lookup evaluations without the aggregation"));
                }
                None
            }
            _ => return Err(Error::Invalid("the aggregation and the table together")),
        };
        Ok(Self {
            public,
            w,
            z,
            s,
            coefficients,
            generic_selector,
            poseidon_selector,
            complete_add_selector,
            mul_selector,
            emul_selector,
            endomul_scalar_selector,
            optional_selectors,
            lookup,
        })
    }
}

impl<C> Encodable for ProverProof<C>
where
    C: Curve<Base: PrimeField, Scalar: Encodable>,
{
    fn write(&self, out: &mut Vec<u8>) {
        msgpack::write_array(out, 5);
        // The commitments.
        msgpack::write_array(out, 4);
        self.w_comm.write(out);
        self.z_comm.write(out);
        self.t_comm.write(out);
        self.lookup.write(out);
        self.proof.write(out);
        self.evals.write(out);
        self.ft_eval1.write(out);
        // The previous accumulators: `b`'s commitment of one chunk.
        msgpack::write_array(out, self.prev_challenges.len());
        for p in &self.prev_challenges {
            let comm = PolyComm {
                chunks: alloc::vec![p.comm],
            };
            (&p.chals, &comm).write(out);
        }
    }
}

impl<C> Decodable<()> for ProverProof<C>
where
    C: Curve<Base: PrimeField + Sqrt, Scalar: Decodable<()>>,
{
    fn read(input: &mut Input<'_>, (): ()) -> Result<Self, Error> {
        msgpack::read_fields(input, 5)?;
        msgpack::read_fields(input, 4)?;
        let w_comm = Decodable::read(input, ())?;
        let z_comm = PolyComm::read(input, ())?;
        let t_comm = PolyComm::read(input, ())?;
        let lookup = Option::read(input, ())?;
        let proof = OpeningProof::read(input, ())?;
        let evals = ProofEvaluations::read(input, ())?;
        let ft_eval1 = C::Scalar::read(input, ())?;
        let accumulators: Vec<(Vec<C::Scalar>, PolyComm<C>)> = Vec::read(input, ())?;
        let mut prev_challenges = Vec::with_capacity(accumulators.len());
        for (chals, comm) in accumulators {
            match comm.chunks[..] {
                [comm] => prev_challenges.push(RecursionChallenge { chals, comm }),
                _ => return Err(Error::Invalid("a previous accumulator of one chunk")),
            }
        }
        Ok(Self {
            w_comm,
            z_comm,
            t_comm,
            lookup,
            evals,
            ft_eval1,
            proof,
            prev_challenges,
        })
    }
}

/// `proof` as Kimchi serializes it.
#[must_use]
pub fn to_bytes<C>(proof: &ProverProof<C>) -> Vec<u8>
where
    C: Curve<Base: PrimeField, Scalar: Encodable>,
{
    arrabbiata_codec::encode(proof)
}

/// The proof Kimchi serialized as `bytes`.
///
/// # Errors
///
/// An [`Error`] when the bytes are not the canonical encoding of a proof
/// over `C`.
pub fn from_bytes<C>(bytes: &[u8]) -> Result<ProverProof<C>, Error>
where
    C: Curve<Base: PrimeField + Sqrt, Scalar: Codec>,
{
    arrabbiata_codec::decode(bytes)
}
