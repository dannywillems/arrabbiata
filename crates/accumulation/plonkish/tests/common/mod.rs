//! Circuits shared by the tests: a vanilla one from the frontend, and a
//! custom gate set of degree 5 reading the previous and the next row, a
//! fixed column, with a gadget of two constraints.

#![allow(dead_code)]

use arrabbiata_field::Field;
use arrabbiata_frontend::vanilla::build;
use arrabbiata_ipa::Ipa;
use arrabbiata_pasta::{Fq as F, Pallas};
use arrabbiata_plonkish::{
    Builder, Cell, Circuit, Gadget, Witness,
    vanilla::{self, Vanilla},
};
use arrabbiata_plonkish_folding::Params;
use arrabbiata_poly::{Expr, FixedColumn, Rotation, WitnessColumn};
use arrabbiata_poseidon::{PoseidonSponge, pallas_sponge, vesta_sponge};

pub type Pcs = Ipa<Pallas, PoseidonSponge<F>>;

/// The commitment key for circuits of `n` rows.
pub fn ck(n: usize) -> Pcs {
    Ipa::<Pallas, _>::setup(&mut pallas_sponge(0), n, vesta_sponge(1))
}

/// `x^3 + k x + 5 = y`, with `x` and `k` private and `y` public: one
/// circuit for all values.
pub fn vanilla(x: u64, k: u64) -> (Circuit<F, Vanilla, 3, 5, 1>, Witness<F, 3>) {
    let (c, w, ()) = build::<F, _, _>(|b| {
        let xv = F::from_u64(x);
        let kv = F::from_u64(k);
        let xw = b.witness(xv);
        let kw = b.witness(kv);
        let x2 = b.mul(&xw, &xw);
        let x3 = b.mul(&x2, &xw);
        let kx = b.mul(&kw, &xw);
        let sum = b.add(&x3, &kx);
        let lhs = b.add_constant(&sum, F::from_u64(5));
        let rhs = b.public_input(xv * xv * xv + kv * xv + F::from_u64(5));
        b.assert_equal(&lhs, &rhs);
    });
    (vanilla::circuit(&c).unwrap(), vanilla::witness(&w))
}

pub type VanillaParams = Params<F, Pcs, Vanilla, 3, 5, 1, 3, 2>;

pub fn vanilla_params(domain: u64) -> VanillaParams {
    let (circuit, _) = vanilla(0, 0);
    let n = circuit.num_rows();
    Params::new(circuit, ck(n), vesta_sponge(domain)).unwrap()
}

pub const W: usize = 2;
pub const X: usize = 1;
pub const Q: usize = 3;

fn a(k: i32) -> Expr<F, W, X> {
    Expr::witness(WitnessColumn::new::<0>(), Rotation::new(k))
}

fn b(k: i32) -> Expr<F, W, X> {
    Expr::witness(WitnessColumn::new::<1>(), Rotation::new(k))
}

/// `b = a^5`.
pub struct Pow5;
impl Gadget<F, W, X> for Pow5 {
    const NAME: &'static str = "pow5";
    fn constraints() -> Vec<Expr<F, W, X>> {
        let a2 = a(0) * a(0);
        vec![b(0) - a2.clone() * a2 * a(0)]
    }
}

/// `b = q a(-1) + a`, with `q` the fixed column.
pub struct Affine;
impl Gadget<F, W, X> for Affine {
    const NAME: &'static str = "affine";
    fn constraints() -> Vec<Expr<F, W, X>> {
        let q = Expr::fixed(FixedColumn::new::<0>(), Rotation::CURR);
        vec![b(0) - q * a(-1) - a(0)]
    }
}

/// `b = a^2` and `a(+1) = b`: two constraints.
pub struct SquareOn;
impl Gadget<F, W, X> for SquareOn {
    const NAME: &'static str = "square-on";
    fn constraints() -> Vec<Expr<F, W, X>> {
        vec![b(0) - a(0) * a(0), a(1) - b(0)]
    }
}

pub type Gates = (Pow5, (Affine, (SquareOn, ())));

/// `((q x + x^5)^2)` published, from `x`, with the weight `q` fixed by
/// the circuit.
pub fn custom(x: u64, q: u64) -> (Circuit<F, Gates, W, X, Q>, Witness<F, W>) {
    let x = F::from_u64(x);
    let q = F::from_u64(q);
    let y = x.pow(5);
    let s = q * x + y;
    let mut builder = Builder::<F, Gates, W, X, Q>::new();
    let r0 = builder.row::<Pow5, _>([x, y], [F::ZERO]);
    let r1 = builder.row::<Affine, _>([y, s], [q]);
    let r2 = builder.row::<SquareOn, _>([s, s * s], [F::ZERO]);
    let r3 = builder.plain_row([s * s, F::ZERO], [F::ZERO]);
    let (ca, cb) = (WitnessColumn::new::<0>(), WitnessColumn::new::<1>());
    builder.copy(Cell::new(cb, r0), Cell::new(ca, r1));
    builder.copy(Cell::new(cb, r1), Cell::new(ca, r2));
    builder.public(r3);
    builder.build().unwrap()
}

/// Rotations -1, 0, +1 over two columns: 6 variables; degree 5.
pub type CustomParams = Params<F, Pcs, Gates, W, X, Q, 6, 5>;

pub fn custom_params(q: u64, domain: u64) -> CustomParams {
    let (circuit, _) = custom(0, q);
    let n = circuit.num_rows();
    Params::new(circuit, ck(n), vesta_sponge(domain)).unwrap()
}
