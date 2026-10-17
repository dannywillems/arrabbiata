//! Kimchi's Fiat-Shamir oracles on fixed scripts, for the differential
//! test of arrabbiata-kimchi's `oracle` module: `DefaultFqSponge` over
//! Vesta's base field (`fq_kimchi`), `DefaultFrSponge` over its scalar
//! field (`fp_kimchi`), and `ScalarChallenge::to_field` with Vesta's
//! `endo_r`. Field elements as 64 hex digits, big-endian; `fq` and `fp`
//! tag the field.
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField};
use kimchi::plonk_sponge::FrSponge;
use mina_curves::pasta::{Fp, Fq, Pallas, PallasParameters, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi as SC,
    pasta::{fp_kimchi, fq_kimchi},
    sponge::{DefaultFqSponge, DefaultFrSponge, FqSponge, ScalarChallenge},
};

const FULL_ROUNDS: usize = 55;

fn hex<F: PrimeField>(x: F) -> String {
    x.into_bigint().to_bytes_be().iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let endo_r = kimchi::curve::vesta_endos().1;
    println!("endo_r fp {}", hex(endo_r));
    let g = Vesta::generator();
    let points: Vec<Vesta> = (1..=3u64)
        .map(|k| (g * Fp::from(k)).into_affine())
        .collect();
    for p in &points {
        println!("point fq {} {}", hex(p.x), hex(p.y));
    }

    // The Fq sponge: absorb two points, challenge, absorb two scalars,
    // challenge, challenge_fq, absorb it back, two challenges (the second
    // from the buffered limbs? no: each takes a fresh squeeze), the
    // identity, digest.
    let mut fq = DefaultFqSponge::<VestaParameters, SC, FULL_ROUNDS>::new(fq_kimchi::static_params());
    fq.absorb_g(&points[..2]);
    println!("fq.challenge fp {}", hex(fq.challenge()));
    fq.absorb_fr(&[Fp::from(7u64), -Fp::from(1u64)]);
    println!("fq.challenge fp {}", hex(fq.challenge()));
    let f = fq.challenge_fq();
    println!("fq.challenge_fq fq {}", hex(f));
    fq.absorb_fq(&[f]);
    println!("fq.challenge fp {}", hex(fq.challenge()));
    println!("fq.challenge fp {}", hex(fq.challenge()));
    fq.absorb_g(&[Vesta::zero(), points[2]]);
    println!("fq.digest fp {}", hex(fq.clone().digest()));
    println!("fq.digest_fq fq {}", hex(fq.digest_fq()));

    // The Fr sponge.
    let mut fr = DefaultFrSponge::<Fp, SC, FULL_ROUNDS>::from(fp_kimchi::static_params());
    fr.absorb(&Fp::from(11u64));
    fr.absorb_multiple(&[Fp::from(12u64), -Fp::from(13u64)]);
    let c = fr.challenge();
    println!("fr.challenge fp {}", hex(c.inner()));
    println!("fr.challenge.to_field fp {}", hex(c.to_field(&endo_r)));
    let c = fr.challenge();
    println!("fr.challenge fp {}", hex(c.inner()));
    println!("fr.digest fp {}", hex(fr.digest()));

    // The Fq sponge of Pallas (base Fp, scalars Fq, the larger modulus):
    // absorb_fr splits each scalar into its high bits and its low bit.
    let mut pallas = DefaultFqSponge::<PallasParameters, SC, FULL_ROUNDS>::new(fp_kimchi::static_params());
    pallas.absorb_g(&[Pallas::generator()]);
    pallas.absorb_fr(&[Fq::from(7u64), -Fq::from(1u64), Fq::from(1u64)]);
    println!("pallas.challenge fq {}", hex(pallas.challenge()));
    println!("pallas.digest fq {}", hex(pallas.digest()));
    let gp = Pallas::generator();
    println!("pallas.point fp {} {}", hex(gp.x), hex(gp.y));

    // ScalarChallenge::to_field on fixed 128-bit values.
    for limbs in [[0u64, 0], [1, 0], [0xdead_beef_0123_4567, 0x0f0f_f0f0_aaaa_5555], [u64::MAX, u64::MAX]] {
        let c = ScalarChallenge::<Fp>::from_limbs(limbs);
        println!(
            "to_field {:016x} {:016x} fp {}",
            limbs[0],
            limbs[1],
            hex(c.to_field(&endo_r))
        );
    }
}
