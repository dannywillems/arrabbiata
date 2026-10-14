//! KZG over BN254: completeness, evaluation binding (against a prover who
//! does not know tau), and the module structure of commitments.

use arrabbiata_bn254::{Bn254, Fr, G1};
use arrabbiata_commit::{HomomorphicCommitment, PolynomialCommitment};
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_kzg::Kzg;
use proptest::prelude::*;
use std::sync::OnceLock;

fn kzg() -> &'static Kzg<Bn254> {
    static KZG: OnceLock<Kzg<Bn254>> = OnceLock::new();
    KZG.get_or_init(|| Kzg::insecure_setup(16, Fr::from_u64(0x1234_5678_9abc_def0)))
}

fn arb_f() -> impl Strategy<Value = Fr> {
    prop_oneof![
        Just(Fr::ZERO),
        Just(Fr::ONE),
        any::<[u64; 4]>().prop_map(|l| {
            l.iter().fold(Fr::ZERO, |acc, x| {
                acc * Fr::from_u64(u64::MAX) + Fr::from_u64(*x)
            })
        }),
    ]
}

fn eval(p: &[Fr], x: Fr) -> Fr {
    p.iter().rev().fold(Fr::ZERO, |acc, c| acc * x + *c)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn completeness(p in prop::collection::vec(arb_f(), 0..=16), z in arb_f()) {
        let c = PolynomialCommitment::commit(kzg(), &p);
        let (v, proof) = kzg().open(&p, &z);
        prop_assert_eq!(v, eval(&p, z));
        prop_assert!(kzg().verify(&c, &z, v, &proof));
    }

    /// A proof does not verify for another value or another point, and a
    /// random point is not a proof.
    #[test]
    fn binding(p in prop::collection::vec(arb_f(), 1..=16), z in arb_f(), delta in 1u64..1000) {
        let d = Fr::from_u64(delta);
        let c = PolynomialCommitment::commit(kzg(), &p);
        let (v, proof) = kzg().open(&p, &z);
        prop_assert!(!kzg().verify(&c, &z, v + d, &proof));
        if eval(&p, z + d) != v {
            prop_assert!(!kzg().verify(&c, &(z + d), v, &proof));
        }
        let forged = proof + Projective::<G1>::generator();
        prop_assert!(!kzg().verify(&c, &z, v, &forged));
    }

    #[test]
    fn commitment_is_a_morphism(a in prop::collection::vec(arb_f(), 16), b in prop::collection::vec(arb_f(), 16), s in arb_f()) {
        let folded: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x + s * *y).collect();
        let ck = kzg();
        prop_assert_eq!(
            HomomorphicCommitment::commit(ck, &folded),
            ck.add(&HomomorphicCommitment::commit(ck, &a), &ck.scale(&HomomorphicCommitment::commit(ck, &b), s))
        );
    }
}

/// Whoever knows tau can open to anything: the reason the setup must come
/// from a ceremony. Kept as a test so the threat model is executable.
#[test]
fn knowing_tau_breaks_binding() {
    let tau = Fr::from_u64(0x1234_5678_9abc_def0);
    let p = [Fr::from_u64(5), Fr::from_u64(7)];
    let c = PolynomialCommitment::commit(kzg(), &p);
    let (z, fake) = (Fr::from_u64(3), Fr::from_u64(1000));
    // pi = [(p(tau) - fake) / (tau - z)] G.
    let q = (eval(&p, tau) - fake) * (tau - z).inverse().unwrap();
    let forged = Projective::<G1>::generator() * q;
    assert!(kzg().verify(&c, &z, fake, &forged));
}
