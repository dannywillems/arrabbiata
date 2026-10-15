//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_bn254::Fr;
use arrabbiata_field::{Field, Goldilocks as FieldGoldilocks};
use arrabbiata_frontend::vanilla::build;
use arrabbiata_pasta::{Fq as F, Fq};
use arrabbiata_plonk::{
    CircuitError, Columns, Opened, Plonk, PlonkError, PublicInputs, QuotientSizing, Scheme, With,
    circuit::{Builder, Cell},
    gate::Gadget,
    ipa,
    ipa::PallasIpa,
    kzg, prove, setup, vanilla,
    vanilla::{Vanilla, VanillaPlonk},
    verify,
};
use arrabbiata_poly::{Expr, Rotation, WitnessColumn};
use arrabbiata_poseidon::PoseidonSponge;
use arrabbiata_transcript::{
    Transcript,
    typed::{Label, Pattern, Protocol},
};

#[test]
fn ipa_overview() {
    let (c, w, ()) = build::<Fq, _, _>(|b| {
        let x = b.witness(Fq::from_u64(3));
        let y = b.mul(&x, &x);
        let out = b.public_input(Fq::from_u64(9));
        b.assert_equal(&y, &out);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    verify(
        &vk,
        &pcs,
        ipa::pallas_transcript(),
        &[Fq::from_u64(9)],
        proof,
    )
    .unwrap();
}

#[test]
fn ipa_pallas() {
    let (c, _, ()) = build::<Fq, _, _>(|_| ());
    let circuit = vanilla::circuit(&c).unwrap();
    assert_eq!(ipa::pallas(&circuit).size(), 8);
}

#[test]
fn ipa_bn254() {
    let (c, _, ()) = build::<Fr, _, _>(|_| ());
    let circuit = vanilla::circuit(&c).unwrap();
    assert_eq!(ipa::bn254(&circuit).size(), 8);
}

#[test]
fn ipa_pallas_transcript() {
    assert_eq!(
        ipa::pallas_transcript().squeeze(),
        ipa::pallas_transcript().squeeze()
    );
}

#[test]
fn ipa_bn254_transcript() {
    assert_eq!(
        ipa::bn254_transcript().squeeze(),
        ipa::bn254_transcript().squeeze()
    );
}

#[test]
fn kzg_overview() {
    let (c, w, ()) = build::<Fr, _, _>(|b| {
        let x = b.witness(Fr::from_u64(3));
        let y = b.mul(&x, &x);
        let out = b.public_input(Fr::from_u64(9));
        b.assert_equal(&y, &out);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    let pcs = kzg::insecure_srs(&circuit, Fr::from_u64(0xdead_beef));
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, kzg::transcript(), &witness).unwrap();
    verify(&vk, &pcs, kzg::transcript(), &[Fr::from_u64(9)], proof).unwrap();
}

#[test]
fn kzg_insecure_srs() {
    let (c, _, ()) = build::<Fr, _, _>(|_| ());
    let circuit = vanilla::circuit(&c).unwrap();
    let srs = kzg::insecure_srs(&circuit, Fr::ONE + Fr::ONE);
    assert_eq!(srs.size(), circuit.max_coefficients());
}

#[test]
fn kzg_transcript() {
    assert_eq!(kzg::transcript().squeeze(), kzg::transcript().squeeze());
}

#[test]
fn crate_overview() {
    /// The Poseidon S-box, `b = a^5`.
    struct SBox;
    impl Gadget<F, 2, 0> for SBox {
        const NAME: &'static str = "sbox";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let a = Expr::witness(WitnessColumn::new::<0>(), Rotation::CURR);
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            let a2 = a.clone() * a.clone();
            vec![b - a2.clone() * a2 * a]
        }
    }

    /// `a' = b`: the output of a row is the input of the next.
    struct Chain;
    impl Gadget<F, 2, 0> for Chain {
        const NAME: &'static str = "chain";
        fn constraints() -> Vec<Expr<F, 2, 0>> {
            let b = Expr::witness(WitnessColumn::new::<1>(), Rotation::CURR);
            let next = Expr::witness(WitnessColumn::new::<0>(), Rotation::NEXT);
            vec![next - b]
        }
    }

    // 2 -> 2^5 -> 2^25, with 2 and 2^25 public.
    type Gates = (SBox, (Chain, ()));
    let mut b = Builder::<F, Gates, 2, 0, 2>::new();
    let x = F::from_u64(2);
    let y = x.pow(5);
    let z = y.pow(5);
    let input = b.plain_row([x, F::ZERO], []);
    let r0 = b.row::<SBox, _>([x, y], []);
    let r1 = b.row::<SBox, _>([y, z], []);
    let output = b.plain_row([z, F::ZERO], []);
    let (a, c) = (WitnessColumn::new::<0>(), WitnessColumn::new::<1>());
    b.copy(Cell::new(a, input), Cell::new(a, r0));
    b.copy(Cell::new(c, r0), Cell::new(a, r1));
    b.copy(Cell::new(c, r1), Cell::new(a, output));
    b.public(input);
    b.public(output);
    let (circuit, witness) = b.build().unwrap();

    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    verify(&vk, &pcs, ipa::pallas_transcript(), &[x, z], proof).unwrap();
}

#[test]
fn scheme() {
    let m: Scheme<PallasIpa, u8, 1> = Scheme::new(7);
    assert_eq!(m.value, 7);
}

#[test]
fn scheme_new() {
    let m = Scheme::<PallasIpa, _, 1>::new("x");
    assert_eq!(m.value, "x");
}

#[test]
fn with() {
    type Setup = With<PallasIpa, PoseidonSponge<Fq>>;
    assert_eq!(Setup::LABEL, <PallasIpa as Label>::LABEL);
}

#[test]
fn columns() {
    assert_eq!(Columns([1, 2, 3]).0[2], 3);
}

#[test]
fn public_inputs() {
    assert_eq!(PublicInputs(vec![FieldGoldilocks::from_u64(1)]).0.len(), 1);
}

#[test]
fn opened() {
    let o = Opened::<u8, 1, 1, 1> {
        witness: [0],
        fixed: [1],
        selectors: [2],
        sigmas: [3],
        z: 4,
        t: 5,
        z_next: 6,
        witness_rotated: vec![],
        fixed_rotated: vec![],
    };
    assert_eq!(
        o.iter().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6]
    );
}

#[test]
fn opened_iter() {
    let o = Opened::<u8, 2, 0, 0> {
        witness: [1, 2],
        fixed: [],
        selectors: [],
        sigmas: [3, 4],
        z: 5,
        t: 6,
        z_next: 7,
        witness_rotated: vec![8],
        fixed_rotated: vec![],
    };
    assert_eq!(o.iter().count(), 8);
}

#[test]
fn opened_iter_mut() {
    let mut o = Opened::<u8, 1, 0, 0> {
        witness: [1],
        fixed: [],
        selectors: [],
        sigmas: [2],
        z: 3,
        t: 4,
        z_next: 5,
        witness_rotated: vec![],
        fixed_rotated: vec![],
    };
    *o.iter_mut().nth(4).unwrap() = 9;
    assert_eq!(o.z_next, 9);
}

#[test]
fn opened_rotated_witness() {
    // Vanilla PlonK reads only the current row.
    assert!(Opened::<F, 3, 5, 1>::rotated_witness::<F, Vanilla>().is_empty());
}

#[test]
fn opened_rotated_fixed() {
    assert!(Opened::<F, 3, 5, 1>::rotated_fixed::<F, Vanilla>().is_empty());
}

#[test]
fn preprocessed() {
    let (c, _, ()) = build::<F, _, _>(|b| {
        b.witness(F::from_u64(1));
    });
    let circuit = vanilla::circuit(&c).unwrap();
    let (_, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    assert_eq!(vk.value.num_rows(), 2);
}

#[test]
fn preprocessed_num_rows() {
    let (c, _, ()) = build::<F, _, _>(|_| ());
    let circuit = vanilla::circuit(&c).unwrap();
    let (_, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    assert_eq!(vk.value.num_rows(), circuit.num_rows());
}

#[test]
fn preprocessed_public_rows() {
    let (c, _, ()) = build::<F, _, _>(|b| {
        b.public_input(F::ONE);
    });
    let circuit = vanilla::circuit(&c).unwrap();
    let (_, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    assert_eq!(vk.value.public_rows(), circuit.public_rows());
}

#[test]
fn prover_key() {
    let (c, _, ()) = build::<F, _, _>(|_| ());
    let circuit = vanilla::circuit(&c).unwrap();
    let (pk, vk) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    assert_eq!(pk.verifier_key(), &vk);
}

#[test]
fn prover_key_verifier_key() {
    let (c, _, ()) = build::<F, _, _>(|_| ());
    let circuit = vanilla::circuit(&c).unwrap();
    let (pk, _) = arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).unwrap();
    assert_eq!(pk.verifier_key().value.num_rows(), 2);
}

#[test]
fn plonk() {
    type P = <Plonk<F, PallasIpa, PoseidonSponge<F>, Vanilla, 3, 5, 1> as Protocol>::Transcript;
    assert_eq!(<P as Pattern>::MESSAGES, 5);
    assert_eq!(<P as Pattern>::CHALLENGES, 4);
}

#[test]
fn plonk_error() {
    assert_eq!(
        PlonkError::IdentityFailed.to_string(),
        "the PlonK identity does not hold at zeta"
    );
}

#[test]
fn plonk_error_impl() {
    assert_eq!(
        PlonkError::from(CircuitError::Rotation),
        PlonkError::Rotation
    );
}

#[test]
fn quotient_sizing() {
    let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    // q * q_M a b has degree 4, and so has z (a + ..) (b + ..) (c + ..).
    assert_eq!(c.quotient_factor(), 4);
    assert_eq!(c.max_coefficients(), 3 * c.num_rows());
}

#[test]
fn plonk_error_quotient_factor() {
    let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    assert_eq!(c.quotient_factor(), 4);
}

#[test]
fn plonk_error_max_coefficients() {
    let (c, _) = Builder::<F, Vanilla, 3, 5, 1>::new().build().unwrap();
    assert_eq!(c.max_coefficients(), 3 * c.num_rows());
}

#[test]
fn setup_works() {
    let (c, _, ()) = build::<F, _, _>(|b| {
        b.constant(F::from_u64(2));
    });
    let circuit = vanilla::circuit(&c).unwrap();
    assert!(arrabbiata_plonk::setup(&circuit, &ipa::pallas(&circuit)).is_ok());
}

#[test]
fn prove_works() {
    // A false statement: 2 * 2 = 5.
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(2));
        let y = b.mul(&x, &x);
        let five = b.public_input(F::from_u64(5));
        b.assert_equal(&y, &five);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    let pcs = ipa::pallas(&circuit);
    let (pk, _) = setup(&circuit, &pcs).unwrap();
    assert_eq!(
        prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap_err(),
        PlonkError::Unsatisfied
    );
}

#[test]
fn verify_works() {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(4));
        let y = b.mul(&x, &x);
        let out = b.public_input(F::from_u64(16));
        b.assert_equal(&y, &out);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    // The same proof does not verify for another statement.
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            &[F::from_u64(17)],
            proof.clone()
        )
        .is_err()
    );
    assert!(
        verify(
            &vk,
            &pcs,
            ipa::pallas_transcript(),
            &[F::from_u64(16)],
            proof
        )
        .is_ok()
    );
}

#[test]
fn vanilla_overview() {
    // x^3 + x + 5 = 35, with x = 3 private and 35 public.
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::from_u64(3));
        let x2 = b.mul(&x, &x);
        let x3 = b.mul(&x2, &x);
        let sum = b.add(&x3, &x);
        let lhs = b.add_constant(&sum, F::from_u64(5));
        let rhs = b.public_input(F::from_u64(35));
        b.assert_equal(&lhs, &rhs);
    });
    let (circuit, witness) = (vanilla::circuit(&c).unwrap(), vanilla::witness(&w));
    let pcs = ipa::pallas(&circuit);
    let (pk, vk) = setup(&circuit, &pcs).unwrap();
    let proof = prove(&pk, &pcs, ipa::pallas_transcript(), &witness).unwrap();
    verify(
        &vk,
        &pcs,
        ipa::pallas_transcript(),
        &[F::from_u64(35)],
        proof,
    )
    .unwrap();
}

#[test]
fn vanilla_vanilla_plonk() {
    type P = <VanillaPlonk<F, PallasIpa, PoseidonSponge<F>> as Protocol>::Transcript;
    assert_eq!(<P as Pattern>::MESSAGES, 5);
}
