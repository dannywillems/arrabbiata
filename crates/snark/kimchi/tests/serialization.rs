//! Kimchi's serialized proofs: the 40 fixtures of proof-systems at
//! `KIMCHI_REV` (`kimchi/src/tests/fixtures`, `vectors/kimchi_fixtures.txt`),
//! each the `rmp_serde` bytes of a proof Kimchi made. Every one decodes to
//! our proof and encodes back to Kimchi's bytes, byte for byte: Kimchi's
//! bytes are in the domain of our decoder and fixed by `write . read`, the
//! canonicity law on bytes neither side of this test wrote.

mod kimchi_runs;

use std::sync::LazyLock;

use arrabbiata_codec::{Decodable, Error, Input, laws, msgpack};
use arrabbiata_kimchi::{
    prover::ProverProof,
    serialization::{from_bytes, to_bytes},
};
use arrabbiata_pasta::{Pallas, Vesta};

/// Kimchi's fixtures, by name.
static FIXTURES: LazyLock<Vec<(String, Vec<u8>)>> = LazyLock::new(|| {
    kimchi_runs::read("kimchi_fixtures.txt")
        .lines()
        .map(|l| {
            let mut words = l.split_whitespace();
            assert_eq!(words.next(), Some("fixture"));
            let name = words.next().unwrap().to_string();
            let hex = words.next().unwrap();
            let bytes = (0..hex.len() / 2)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
                .collect();
            (name, bytes)
        })
        .collect()
});

/// The proof bytes of a fixture: the first field of Kimchi's `RawFixture`,
/// `rmp_serde` bytes of `{ proof_bytes, verifier_index_bytes,
/// public_inputs_bytes, num_public_inputs, feature_flags, .. }`, its byte
/// vectors arrays of integers.
fn proof_bytes(fixture: &[u8]) -> Result<Vec<u8>, Error> {
    let mut input = Input::new(fixture);
    let fields = msgpack::read_array(&mut input)?;
    let proof = Vec::<u8>::read(&mut input, ())?;
    for _ in 1..fields {
        msgpack::skip(&mut input)?;
    }
    assert_eq!(input.remaining(), 0, "one fixture per file");
    Ok(proof)
}

/// The fixture's proof over `C` decodes, and encodes back to its bytes.
fn check<C>(name: &str, bytes: &[u8])
where
    C: arrabbiata_curve::Curve<
            Base: arrabbiata_field::PrimeField + arrabbiata_field::Sqrt,
            Scalar: arrabbiata_codec::Codec,
        >,
{
    let proof: ProverProof<C> =
        from_bytes(bytes).unwrap_or_else(|e| panic!("{name}: Kimchi's bytes do not decode: {e}"));
    assert!(to_bytes(&proof) == bytes, "{name}: not Kimchi's bytes back");
    assert_eq!(laws::canonical::<ProverProof<C>, ()>(bytes, ()), Ok(()));
    assert_eq!(laws::round_trip(&proof, ()), Ok(()));
}

#[test]
fn kimchis_fixtures_are_read_and_written_byte_for_byte() {
    assert_eq!(FIXTURES.len(), 40);
    for (name, fixture) in FIXTURES.iter() {
        let bytes = proof_bytes(fixture).unwrap_or_else(|e| panic!("{name}: {e}"));
        if name.contains("pallas") {
            check::<Pallas>(name, &bytes);
        } else {
            check::<Vesta>(name, &bytes);
        }
    }
}

#[test]
fn a_fixture_on_the_wrong_curve_is_refused() {
    // Vesta's proof read over Pallas: its points are not Pallas's.
    let (_, fixture) = FIXTURES
        .iter()
        .find(|(n, _)| n == "and_prove_and_verify_vesta")
        .unwrap();
    let bytes = proof_bytes(fixture).unwrap();
    assert!(from_bytes::<Pallas>(&bytes).is_err());
}

#[test]
fn a_cut_or_padded_proof_is_refused() {
    let (_, fixture) = &FIXTURES[0];
    let bytes = proof_bytes(fixture).unwrap();
    let over = |b: &[u8]| from_bytes::<Pallas>(b).err();
    assert_eq!(over(&bytes[..bytes.len() - 1]), Some(Error::Truncated));
    let mut padded = bytes.clone();
    padded.push(0);
    assert_eq!(over(&padded), Some(Error::Trailing(1)));
}
