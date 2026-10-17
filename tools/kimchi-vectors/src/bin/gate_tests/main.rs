//! Kimchi's gate unit tests (kimchi/src/tests/*.rs), run with fixed seeds,
//! printing what each draws and what Kimchi computes, so that
//! arrabbiata-kimchi can port each test with the same inputs and outputs.
//!
//! Lines:
//! - `test <name>`: a test starts;
//! - `input <key> <values...>`: a drawn or fixed input (field elements as
//!   64 hex digits big-endian, integers in decimal);
//! - `witness <rows>` then `row <15 values>`: Kimchi's witness;
//! - `verdict <label> <Debug of the result>`: one assertion's outcome.
//!
//! Seeds: `make_test_rng(None)` becomes `StdRng::from_seed([k; 32])` with
//! `k` given per test and curve below.

mod and;
mod common;
mod ec;
mod ffadd;
mod endomul;
mod endomul_scalar;
mod generic;
mod not;
mod poseidon;
mod range_check;
mod rot;
mod varbasemul;
mod xor;

fn main() {
    xor::run();
    rot::run();
    range_check::run();
    generic::run();
    not::run();
    and::run();
    and::serialization();
    endomul_scalar::run();
    endomul::run();
    varbasemul::run();
    ec::run();
    poseidon::run();
    ffadd::run();
}
