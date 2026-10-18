//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

use arrabbiata_bn254::{Bn254, Fr, G1, G2};
use arrabbiata_commit::PolynomialCommitment;
use arrabbiata_curve::Projective;
use arrabbiata_field::Field;
use arrabbiata_kzg::{
    Kzg, PairingAccumulator,
    accumulation::{FoldPattern, OpenPattern, Openings, Pairs},
};
use arrabbiata_transcript::{
    Absorb, Transcript, absorb_slice,
    typed::{Label, Pattern},
};

#[test]
fn crate_overview() {
    // A test setup whose secret is known: never use it in a deployment.
    let kzg = Kzg::<Bn254>::insecure_setup(8, Fr::from_u64(123_456_789));
    let p = [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
    let c = kzg.commit(&p);
    let (v, proof) = kzg.open(&p, &Fr::from_u64(2));
    assert_eq!(v, Fr::from_u64(17));
    assert!(kzg.verify(&c, &Fr::from_u64(2), v, &proof));
}

#[test]
fn kzg() {
    assert_eq!(Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(5)).size(), 4);
}

#[test]
fn kzg_insecure_setup() {
    let _ = Kzg::<Bn254>::insecure_setup(2, Fr::from_u64(42));
}

#[test]
fn kzg_from_powers() {
    let tau = Fr::from_u64(9);
    let (g, h) = (Projective::<G1>::generator(), Projective::<G2>::generator());
    let kzg = Kzg::<Bn254>::from_powers(vec![g, g * tau], h, h * tau);
    assert_eq!(kzg.size(), 2);
}

#[test]
fn kzg_size() {
    assert_eq!(
        Kzg::<Bn254>::insecure_setup(16, Fr::ONE + Fr::ONE).size(),
        16
    );
}

/// A transcript that keeps what it absorbs.
#[derive(Default, PartialEq, Eq, Debug)]
struct Record(Vec<Fr>);

impl Transcript<Fr> for Record {
    fn absorb(&mut self, value: Fr) {
        self.0.push(value);
    }

    fn squeeze(&mut self) -> Fr {
        Fr::from_u64(self.0.len() as u64)
    }
}

#[test]
fn openings() {
    // One message: the quotient commitments, absorbed as a slice.
    let g = Projective::<G1>::generator();
    let openings = Openings::<Bn254>(vec![g, g + g]);
    let mut record = Record::default();
    openings.absorb_into(&mut record);
    let mut expected = Record::default();
    absorb_slice(&mut expected, &openings.0);
    assert_eq!(record, expected);
    assert_ne!(Openings::<Bn254>::LABEL, Pairs::<Bn254>::LABEL);
}

#[test]
fn pairs() {
    // The pairing accumulators, absorbed as a slice.
    let kzg = Kzg::<Bn254>::insecure_setup(4, Fr::from_u64(5));
    let (_, proof) = kzg.open(&[Fr::ONE, Fr::ONE], &Fr::from_u64(2));
    let pair = PairingAccumulator::<Bn254> { p: proof, w: proof };
    let pairs = Pairs::<Bn254>(vec![pair.clone(), pair]);
    let mut record = Record::default();
    pairs.absorb_into(&mut record);
    let mut expected = Record::default();
    absorb_slice(&mut expected, &pairs.0);
    assert_eq!(record, expected);
}

#[test]
fn open_pattern() {
    // Combine's challenge, the openings sent, then `r`.
    assert_eq!(<OpenPattern<Bn254> as Pattern>::MESSAGES, 1);
    assert_eq!(<OpenPattern<Bn254> as Pattern>::CHALLENGES, 2);
}

#[test]
fn fold_pattern() {
    // The pairs public, then `r`.
    assert_eq!(<FoldPattern<Bn254> as Pattern>::MESSAGES, 0);
    assert_eq!(<FoldPattern<Bn254> as Pattern>::CHALLENGES, 1);
}
