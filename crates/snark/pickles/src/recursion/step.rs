//! The step side of the recursion: what a step proof states and hashes,
//! shared by the step circuit that proves it and the wrap circuit that
//! reads it back.
//!
//! - [`Statement`]: Pickles' `Types.Step.Statement` for one predecessor,
//!   in the order of its `spec`: the unfinalized proof ([`Unfinalized`]),
//!   the messages for the next step proof, the predecessor's messages for
//!   the next wrap proof.
//! - [`MessagesForNextStepProof`]: the application state, the wrap key
//!   ([`index_elements`]), and the accumulators of the wrap proofs
//!   verified, crossed over as Mina's `Per_proof_witness` pairs them: the
//!   `sg` of each wrap proof with the challenges its statement deferred
//!   ([`deferred_challenges`]).

use alloc::vec::Vec;

use arrabbiata_curve::{Projective, glv::Glv};
use arrabbiata_field::Field;
use arrabbiata_kimchi::{
    oracle::{FrSponge, ScalarChallenge, coordinates},
    prover::{ProofEvaluations, ProverProof},
    verifier_index::VerifierIndex,
};
use arrabbiata_pasta::{Fp, Fq, Pallas, Vesta};
use arrabbiata_poseidon::kimchi::{KimchiSponge, pallas_base};

use crate::{
    statement::{Bool, Challenge, Digest, Sort, Type2, Word},
    wrap,
};

/// A wrap proof left for the next wrap circuit to finalize: its deferred
/// values, scalars of `Fq` as Type2 words `[high, low]`, and its draws
/// (Pickles' `Per_proof.In_circuit`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unfinalized<V> {
    /// The combined inner product.
    pub combined_inner_product: [V; 2],
    /// `b`.
    pub b: [V; 2],
    /// `zeta^m`.
    pub zeta_to_srs_length: [V; 2],
    /// `zeta^n`.
    pub zeta_to_domain_size: [V; 2],
    /// `perm`.
    pub perm: [V; 2],
    /// The base sponge's digest before the evaluations.
    pub sponge_digest_before_evaluations: V,
    /// `beta`.
    pub beta: V,
    /// `gamma`.
    pub gamma: V,
    /// The 128-bit challenge `alpha` comes from.
    pub alpha: V,
    /// The 128-bit challenge `zeta` comes from.
    pub zeta: V,
    /// The 128-bit challenge `xi` comes from.
    pub xi: V,
    /// The IPA's round challenges, unmapped.
    pub bulletproof_challenges: Vec<V>,
    /// Whether the next wrap circuit finalizes it: one, for a real proof.
    pub should_finalize: V,
}

/// What the step proof proves, in Pickles' order, for one predecessor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement<V> {
    /// The wrap proof verified, unfinalized.
    pub unfinalized: Unfinalized<V>,
    /// The digest of the messages for the next step proof
    /// ([`MessagesForNextStepProof`]).
    pub messages_for_next_step_proof: V,
    /// The predecessor's digest of the messages for the next wrap proof,
    /// passed along.
    pub messages_for_next_wrap_proof: V,
}

/// The values the statement holds besides the prechallenges.
const FIXED: usize = 10 + 1 + 5 + 1 + 2;

impl<V: Clone> Statement<V> {
    /// The statement as the step proof's public input, in order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<V> {
        let u = &self.unfinalized;
        let mut out = Vec::new();
        for f in [
            &u.combined_inner_product,
            &u.b,
            &u.zeta_to_srs_length,
            &u.zeta_to_domain_size,
            &u.perm,
        ] {
            out.extend(f.iter().cloned());
        }
        out.extend([
            u.sponge_digest_before_evaluations.clone(),
            u.beta.clone(),
            u.gamma.clone(),
            u.alpha.clone(),
            u.zeta.clone(),
            u.xi.clone(),
        ]);
        out.extend(u.bulletproof_challenges.iter().cloned());
        out.extend([
            u.should_finalize.clone(),
            self.messages_for_next_step_proof.clone(),
            self.messages_for_next_wrap_proof.clone(),
        ]);
        out
    }

    /// The statement of `values`, [`Statement::to_vec`]'s inverse for a
    /// wrap proof of `rounds` rounds; `None` for another length.
    #[must_use]
    pub fn of(rounds: usize, values: &[V]) -> Option<Self> {
        if values.len() != FIXED + rounds {
            return None;
        }
        let v = |i: usize| values[i].clone();
        let pair = |i: usize| [v(i), v(i + 1)];
        Some(Self {
            unfinalized: Unfinalized {
                combined_inner_product: pair(0),
                b: pair(2),
                zeta_to_srs_length: pair(4),
                zeta_to_domain_size: pair(6),
                perm: pair(8),
                sponge_digest_before_evaluations: v(10),
                beta: v(11),
                gamma: v(12),
                alpha: v(13),
                zeta: v(14),
                xi: v(15),
                bulletproof_challenges: values[16..16 + rounds].to_vec(),
                should_finalize: v(16 + rounds),
            },
            messages_for_next_step_proof: v(17 + rounds),
            messages_for_next_wrap_proof: v(18 + rounds),
        })
    }
}

/// The bound of each word of a statement of `rounds` rounds, in order: the
/// sorts' ([`Sort::BOUNDS`]).
#[must_use]
pub fn bounds(rounds: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..5 {
        out.extend(Type2::<Pallas>::BOUNDS);
    }
    out.extend(Digest::BOUNDS);
    for _ in 0..5 + rounds {
        out.extend(Challenge::BOUNDS);
    }
    out.extend(Bool::BOUNDS);
    for _ in 0..2 {
        out.extend(Digest::BOUNDS);
    }
    out
}

/// The wrap key as the elements the messages hash: the coordinates of its
/// commitments, permutation, coefficients, then the selectors (Pickles'
/// `index_to_field_elements`, as read).
#[must_use]
pub fn index_elements(index: &VerifierIndex<Pallas>) -> Vec<Fp> {
    let mut comms: Vec<&Projective<Pallas>> = Vec::new();
    for c in index.sigma_comm.iter().chain(&index.coefficients_comm) {
        comms.extend(&c.chunks);
    }
    for c in [
        &index.generic_comm,
        &index.psm_comm,
        &index.complete_add_comm,
        &index.mul_comm,
        &index.emul_comm,
        &index.endomul_scalar_comm,
    ] {
        comms.extend(&c.chunks);
    }
    comms
        .into_iter()
        .flat_map(|p| {
            let (x, y) = coordinates(p);
            [x, y]
        })
        .collect()
}

/// What the next step proof needs of this one, hashed into the statement:
/// the application state, the wrap key, the `sg` of each wrap proof
/// verified, and the challenges each wrap statement deferred, those of the
/// step proof it wrapped, mapped into `Fp` (Pickles'
/// `Messages_for_next_step_proof`, prepared by `Ipa.Step.compute_challenges`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessagesForNextStepProof {
    /// The wrap key ([`index_elements`]).
    pub index: Vec<Fp>,
    /// The application state.
    pub app_state: Vec<Fp>,
    /// The wrap proofs' accumulator commitments, as coordinates.
    pub challenge_polynomial_commitments: Vec<[Fp; 2]>,
    /// The challenges the wrap statements deferred, mapped.
    pub old_bulletproof_challenges: Vec<Vec<Fp>>,
}

impl MessagesForNextStepProof {
    /// The elements hashed: the key, the state, then each commitment's
    /// coordinates followed by its challenges (Pickles' `to_field_elements`).
    #[must_use]
    pub fn field_elements(&self) -> Vec<Fp> {
        let mut out = self.index.clone();
        out.extend(&self.app_state);
        for (comm, chals) in self
            .challenge_polynomial_commitments
            .iter()
            .zip(&self.old_bulletproof_challenges)
        {
            out.extend(comm);
            out.extend(chals);
        }
        out
    }

    /// The digest: Kimchi's sponge over `Fp`, absorbing the elements, then
    /// one squeeze (Pickles' `Tick_field_sponge.digest`).
    #[must_use]
    pub fn digest(&self) -> Fp {
        let mut sponge = FrSponge::new(KimchiSponge::new(pallas_base()));
        sponge.absorb_multiple(&self.field_elements());
        sponge.digest()
    }
}

/// The application rule: the next state of a state.
#[must_use]
pub fn rule(state: Fp) -> Fp {
    state + Fp::ONE
}

/// The challenges a wrap statement deferred, mapped into `Fp` by the
/// endomorphism of Vesta: those of the step proof it wrapped.
#[must_use]
pub fn deferred_challenges(statement: &wrap::Statement<Fq>) -> Vec<Fp> {
    let lambda = Vesta::lambda();
    statement
        .bulletproof_challenges
        .iter()
        .map(|c| ScalarChallenge(to_fp(*c)).to_field(lambda))
        .collect()
}

/// What finalizing a wrap proof needs beyond the statement: its
/// evaluations and `ft(zeta w)`.
#[derive(Clone, Debug)]
pub struct WrapEvaluations {
    /// The wrap proof's evaluations.
    pub evals: ProofEvaluations<Fq>,
    /// `ft(zeta w)`.
    pub ft_eval1: Fq,
}

impl WrapEvaluations {
    /// The evaluations `proof` carries.
    #[must_use]
    pub fn of(proof: &ProverProof<Pallas>) -> Self {
        Self {
            evals: proof.evals.clone(),
            ft_eval1: proof.ft_eval1,
        }
    }
}

/// A value of `Fq` below `2^254` as the element of `Fp` with its integer.
pub(crate) fn to_fp(x: Fq) -> Fp {
    Word::of(x).reduce()
}

/// A value of `Fp` below `2^254` as the element of `Fq` with its integer.
pub(crate) fn to_fq(x: Fp) -> Fq {
    Word::of(x).reduce()
}

/// The Type2 words of a scalar of `Fq`, in `Fp`.
pub(crate) fn type2(s: Fq) -> [Fp; 2] {
    let words = Type2::<Pallas>::encode(&s);
    [words[0].reduce(), words[1].reduce()]
}

/// `2^255 + 1` in `Fp`: Type1's shift.
pub(crate) fn type1_shift() -> Fp {
    (Fp::ONE + Fp::ONE).pow(255) + Fp::ONE
}

/// An unfinalized proof's challenges mapped by the endomorphism of Pallas:
/// the scalars of `Fq` its accumulator's polynomial is made of.
pub fn computed(prechallenges: &[Fp]) -> Vec<Fq> {
    let lambda = Pallas::lambda();
    prechallenges
        .iter()
        .map(|c| ScalarChallenge(Word::of(*c).reduce::<Fq>()).to_field(lambda))
        .collect()
}
