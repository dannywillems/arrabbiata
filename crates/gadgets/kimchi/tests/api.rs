//! The behaviour of each public item of the `harness` module, one test per
//! item (the gates themselves are tested in their own files).

use arrabbiata_curve::glv::Glv;
use arrabbiata_gadget::gates;
use arrabbiata_kimchi_gadgets::{
    LAYOUT,
    harness::{Harness, NAMES, gate_kinds, of},
};
use arrabbiata_pasta::{Fp, Pallas};
use arrabbiata_poseidon::kimchi::pallas_base;

#[test]
fn harness() {
    // A harness meets its gate's kind at row 0.
    let kinds = gates::<Fp, _>(&Harness::<Fp>::CompleteAdd, LAYOUT).unwrap();
    assert!(
        kinds
            .iter()
            .any(|k| k.gadget == "kimchi-complete-add" && k.offset == 0)
    );
}

#[test]
fn harness_of() {
    let poseidon = pallas_base();
    let endo = Pallas::beta();
    assert!(matches!(
        of("kimchi-endomul", &poseidon, endo),
        Harness::EndoMul(e) if e == endo
    ));
    assert!(matches!(of("kimchi-xor16", &poseidon, endo), Harness::Xor));
}

#[test]
fn names() {
    assert_eq!(NAMES.len(), 12);
    assert_eq!(NAMES[0], ("Generic", "kimchi-generic"));
}

#[test]
fn kinds_of_every_gate() {
    let kinds = gate_kinds(&pallas_base(), Pallas::beta()).unwrap();
    let names: Vec<&str> = kinds.iter().map(|(name, _)| *name).collect();
    let expected: Vec<&str> = NAMES.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, expected);
}
