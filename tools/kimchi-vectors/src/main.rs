//! Kimchi's own figures for each gate: the number of constraints its
//! `constraint_checks` returns, Kimchi's `CONSTRAINTS`, and the degree of
//! each constraint as Kimchi's `Expr::degree(1, 0)` computes it.
use kimchi::circuits::{
    argument::Argument,
    expr::Cache,
    polynomials::{
        complete_add::CompleteAdd, endomul_scalar::EndomulScalar, endosclmul::EndosclMul,
        foreign_field_add::circuitgates::ForeignFieldAdd,
        foreign_field_mul::circuitgates::ForeignFieldMul, generic::Generic, poseidon::Poseidon,
        range_check::circuitgates::{RangeCheck0, RangeCheck1}, rot::Rot64,
        varbasemul::VarbaseMul, xor::Xor16,
    },
};
use ark_ff::{BigInteger, PrimeField};
use mina_curves::pasta::{Fp, Pallas, Vesta};
use poly_commitment::ipa::endos;

fn figures<A: Argument<Fp>>(name: &str) {
    let mut cache = Cache::default();
    let constraints = A::constraints(&mut cache);
    let degrees: Vec<u64> = constraints.iter().map(|c| c.degree(1, 0)).collect();
    let max = degrees.iter().max().copied().unwrap_or(0);
    println!("{name} CONSTRAINTS={} returned={} max_degree={max} degrees={degrees:?}", A::CONSTRAINTS, constraints.len());
}

fn limbs<F: PrimeField>(x: F) -> Vec<String> {
    x.into_bigint().as_ref().iter().map(|l| format!("0x{l:016x}")).collect()
}

fn main() {
    let (q, r) = endos::<Pallas>();
    println!("Pallas endo_q={:?} endo_r={:?}", limbs(q), limbs(r));
    let (q, r) = endos::<Vesta>();
    println!("Vesta endo_q={:?} endo_r={:?}", limbs(q), limbs(r));
    let _ = BigInteger::num_bits(&Fp::from(1u64).into_bigint());
    figures::<Generic<Fp>>("Generic");
    figures::<Poseidon<Fp>>("Poseidon");
    figures::<CompleteAdd<Fp>>("CompleteAdd");
    figures::<VarbaseMul<Fp>>("VarBaseMul");
    figures::<EndosclMul<Fp>>("EndoMul");
    figures::<EndomulScalar<Fp>>("EndoMulScalar");
    figures::<RangeCheck0<Fp>>("RangeCheck0");
    figures::<RangeCheck1<Fp>>("RangeCheck1");
    figures::<ForeignFieldAdd<Fp>>("ForeignFieldAdd");
    figures::<ForeignFieldMul<Fp>>("ForeignFieldMul");
    figures::<Xor16<Fp>>("Xor16");
    figures::<Rot64<Fp>>("Rot64");
}
