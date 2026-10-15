//! Regression vectors for the PlonK transcript.
//!
//! Proving is deterministic, so a fixed circuit, witness, scheme and sponge
//! give a fixed proof. These vectors pin the evaluations, which depend on
//! every challenge, hence on everything absorbed before them: the message
//! labels (scheme, sponge and tag), the verifier key with the gate set's
//! constraints, the public inputs, and every commitment, in order. A change
//! to any of them changes `zeta` and fails these tests.
//!
//! They were produced by this implementation (2026-10-14); they pin the
//! transcript, so that a change to it is deliberate, not its correctness,
//! which the other tests establish. Regenerate them with
//! `PRINT_VECTORS=1 cargo test -p arrabbiata-plonk --test transcript --
//! --nocapture` after an intended change.

use arrabbiata_bn254::Fr;
use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::build;
use arrabbiata_pasta::Fq;
use arrabbiata_plonk::{Builder, Gadget, Opened, ipa, kzg, prove, setup, vanilla, verify};
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};

/// `t(zeta)`, `z(zeta w)` and the first witness polynomial at `zeta`.
fn fingerprint<F: Field, const W: usize, const X: usize, const Q: usize>(
    e: &Opened<F, W, X, Q>,
) -> [String; 3] {
    [
        format!("{:?}", e.t),
        format!("{:?}", e.z_next),
        format!("{:?}", e.witness.first().copied().unwrap_or(F::ZERO)),
    ]
}

fn check(name: &str, got: [String; 3], expected: [&str; 3]) {
    if std::env::var("PRINT_VECTORS").is_ok() {
        println!("{name}: {got:?}");
        return;
    }
    assert_eq!(got, expected.map(String::from), "{name}");
}

/// `x^3 + x + 5 = 35` with `x = 3`, compiled to vanilla PlonK.
fn cubic<F: Field>() -> (vanilla::VanillaCircuit<F>, arrabbiata_plonk::Witness<F, 3>) {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(3));
        let x2 = b.mul(&x, &x);
        let x3 = b.mul(&x2, &x);
        let sum = b.add(&x3, &x);
        let lhs = b.add_constant(&sum, F::from_u64(5));
        let rhs = b.public_input(F::from_u64(35));
        b.assert_equal(&lhs, &rhs);
    });
    (vanilla::circuit(&c).unwrap(), vanilla::witness(&w))
}

#[test]
fn vanilla_ipa_pallas() {
    let (circuit, witness) = cubic::<Fq>();
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    let got = fingerprint(&proof.tail.tail.tail.head);
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            witness.public_inputs(),
            proof
        )
        .is_ok()
    );
    check("vanilla_ipa_pallas", got, VANILLA_IPA_PALLAS);
}

#[test]
fn vanilla_kzg_bn254() {
    let (circuit, witness) = cubic::<Fr>();
    let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0x7e57));
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
    let got = fingerprint(&proof.tail.tail.tail.head);
    assert!(verify(&vk, &pcs, kzg::transcript(), witness.public_inputs(), proof).is_ok());
    check("vanilla_kzg_bn254", got, VANILLA_KZG_BN254);
}

/// `b = a(-1) + f(+2) a(+1)`: rotations on both sides, fixed included.
struct Stencil;
impl Gadget<Fq, 2, 1> for Stencil {
    const NAME: &'static str = "stencil";
    fn constraints() -> Vec<Expr<Fq, 2, 1>> {
        let a = |k| Expr::witness(WitnessColumn::new::<0>(), Rotation::new(k));
        let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
        let f = Expr::fixed(FixedColumn::new::<0>(), Rotation::new(2));
        vec![b - a(-1) - f * a(1)]
    }
}

#[test]
fn rotations_ipa_pallas() {
    let a: Vec<Fq> = (0..8).map(|i| Fq::from_u64(i * i + 1)).collect();
    let f: Vec<Fq> = (0..8).map(|i| Fq::from_u64(7 * i + 3)).collect();
    let mut b = Builder::<Fq, (Stencil, ()), 2, 1, 1>::cyclic();
    for i in 0..8 {
        let v = a[(i + 7) % 8] + f[(i + 2) % 8] * a[(i + 1) % 8];
        b.row::<Stencil, _>([a[i], v], [f[i]]);
    }
    b.public(3);
    let (circuit, witness) = b.build().unwrap();
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    let got = fingerprint(&proof.tail.tail.tail.head);
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            witness.public_inputs(),
            proof
        )
        .is_ok()
    );
    check("rotations_ipa_pallas", got, ROTATIONS_IPA_PALLAS);
}

const VANILLA_IPA_PALLAS: [&str; 3] = [
    "0x3c4d8e3f0dfb10cf3e53b28821b41c66f15125663b58160aa65d26b5458af4fa",
    "0x2899abb201af6a52359adb631f6033b06ceb22556bf501044d9382dabe5f5152",
    "0x2f6bfa1314459e6da7d83a60bd6948308b9269f174e9a4d45c8de41b75f538ba",
];
const VANILLA_KZG_BN254: [&str; 3] = [
    "0x236ba3d03e6394b0ba6002cf1044bf7e88d86d9e951b2cdc32cf35daab9034ba",
    "0x16afa636039ac0669e47fb6ea6da9760709234534c5e82cd0619307fb3dfdcfa",
    "0x1ba573ec7be6866dced34756fba5bfd60966933855dbd00421d95c50200dfb6b",
];
/// `z(zeta w) = 1`: the circuit has no copy, so the grand product is 1.
const ROTATIONS_IPA_PALLAS: [&str; 3] = [
    "0x3b87a509d0feca8e6e5a9acd3fbc6e910d5aff0ec4b75b471ff8cc932368b5c8",
    "0x0000000000000000000000000000000000000000000000000000000000000001",
    "0x39d35c6c920255fd602ca8b45ea66e2e73f24525d3a609be778fe5a3f7c8de27",
];
