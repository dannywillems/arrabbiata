//! Kimchi's constraints and witnesses, for a differential test of
//! arrabbiata-kimchi-gadgets.
//!
//! Part 1: for each gate, at 4 pseudo-random points (all 15 cells of the
//! current and next rows and the 15 coefficients drawn from a fixed
//! seed), the value of each constraint as Kimchi's `constraint_checks`
//! computes it over the field, with Kimchi's constants (the MDS matrix of
//! `fp_kimchi`, the endomorphism coefficient of Pallas).
//!
//! Part 2: the rows Kimchi's witness generators produce on fixed inputs.
//!
//! Field elements are printed as 64 hex digits, big-endian.
use ark_ff::{BigInteger, Field, PrimeField, UniformRand};
use kimchi::circuits::{
    argument::{Argument, ArgumentEnv, ArgumentWitness},
    berkeley_columns::BerkeleyChallenges,
    expr::{Cache, Constants},
    polynomial::COLUMNS,
    polynomials::{
        complete_add::CompleteAdd, endomul_scalar::EndomulScalar, endosclmul::EndosclMul,
        foreign_field_add::{self, circuitgates::ForeignFieldAdd, witness::FFOps},
        foreign_field_mul::{self, circuitgates::ForeignFieldMul},
        generic::Generic, poseidon::Poseidon,
        range_check::{self, circuitgates::{RangeCheck0, RangeCheck1}},
        rot::{self, RotMode, Rot64},
        varbasemul::VarbaseMul,
        xor::{self, Xor16},
    },
};
use mina_curves::pasta::Fp;
use num_bigint::BigUint;
use rand::{SeedableRng, rngs::StdRng};

fn hex(x: Fp) -> String {
    let bytes = x.into_bigint().to_bytes_be();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn line(tag: &str, xs: &[Fp]) {
    let items: Vec<String> = xs.iter().map(|x| hex(*x)).collect();
    println!("{tag} {}", items.join(" "));
}

fn constants() -> Constants<Fp> {
    Constants {
        endo_coefficient: kimchi::curve::pallas_endos().0,
        mds: &mina_poseidon::pasta::fp_kimchi::static_params().mds,
        zk_rows: 3,
    }
}

fn evaluate<A: Argument<Fp>>(name: &str, rng: &mut StdRng) {
    for point in 0..4 {
        let curr: [Fp; COLUMNS] = core::array::from_fn(|_| Fp::rand(rng));
        let next: [Fp; COLUMNS] = core::array::from_fn(|_| Fp::rand(rng));
        let coeffs: Vec<Fp> = (0..15).map(|_| Fp::rand(rng)).collect();
        let challenges = BerkeleyChallenges {
            alpha: Fp::ONE,
            beta: Fp::ONE,
            gamma: Fp::ONE,
            joint_combiner: Fp::ONE,
        };
        let env = ArgumentEnv::<Fp, Fp>::create(
            ArgumentWitness { curr, next },
            coeffs.clone(),
            constants(),
            challenges,
        );
        let values = A::constraint_checks(&env, &mut Cache::default());
        println!("constraints {name} {point}");
        line("curr", &curr);
        line("next", &next);
        line("coeffs", &coeffs);
        line("values", &values);
    }
}

fn rows(name: &str, witness: &[Vec<Fp>; COLUMNS]) {
    println!("witness {name} {}", witness[0].len());
    for r in 0..witness[0].len() {
        let row: Vec<Fp> = (0..COLUMNS).map(|c| witness[c][r]).collect();
        line("row", &row);
    }
}

fn big(hex: &str) -> BigUint {
    BigUint::parse_bytes(hex.as_bytes(), 16).unwrap()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(20261016);
    evaluate::<Generic<Fp>>("Generic", &mut rng);
    evaluate::<Poseidon<Fp>>("Poseidon", &mut rng);
    evaluate::<CompleteAdd<Fp>>("CompleteAdd", &mut rng);
    evaluate::<VarbaseMul<Fp>>("VarBaseMul", &mut rng);
    evaluate::<EndosclMul<Fp>>("EndoMul", &mut rng);
    evaluate::<EndomulScalar<Fp>>("EndoMulScalar", &mut rng);
    evaluate::<RangeCheck0<Fp>>("RangeCheck0", &mut rng);
    evaluate::<RangeCheck1<Fp>>("RangeCheck1", &mut rng);
    evaluate::<ForeignFieldAdd<Fp>>("ForeignFieldAdd", &mut rng);
    evaluate::<ForeignFieldMul<Fp>>("ForeignFieldMul", &mut rng);
    evaluate::<Xor16<Fp>>("Xor16", &mut rng);
    evaluate::<Rot64<Fp>>("Rot64", &mut rng);

    // Part 2: witnesses on fixed inputs.
    let two_87 = Fp::from(2u64).pow([87]);
    rows(
        "RangeCheck",
        &range_check::witness::create_multi(two_87 + Fp::from(12345u64), Fp::from(7u64), Fp::from(1u64 << 40)),
    );
    rows(
        "Xor",
        &xor::create_xor_witness(Fp::from(0xdead_beef_0123_4567u64), Fp::from(0x0f0f_f0f0_aaaa_5555u64), 64),
    );
    let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);
    rot::extend_rot(&mut w, 0x0123_4567_89ab_cdef, 3, RotMode::Left);
    rows("RotLeft3", &w);
    let mut w: [Vec<Fp>; COLUMNS] = core::array::from_fn(|_| vec![]);
    rot::extend_rot(&mut w, 0x0123_4567_89ab_cdef, 17, RotMode::Right);
    rows("RotRight17", &w);
    let secp = big("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f");
    let a = big("7a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9");
    let b = big("f1e2d3c4b5a6978877665544332211000ffeeddccbbaa99887766554433221100");
    rows(
        "ForeignFieldAddSecp",
        &foreign_field_add::witness::create_chain(&[a.clone(), b.clone()], &[FFOps::Add], secp.clone()),
    );
    rows(
        "ForeignFieldSubSecp",
        &foreign_field_add::witness::create_chain(&[b.clone() % &secp, a.clone()], &[FFOps::Sub], secp.clone()),
    );
    let (mul, _) = foreign_field_mul::witness::create::<Fp>(&a, &(b % &secp), &secp);
    rows("ForeignFieldMulSecp", &mul);
}
