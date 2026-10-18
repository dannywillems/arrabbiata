//! Kimchi's committed test fixtures (`kimchi/src/tests/fixtures/*.bin`, each
//! `rmp_serde::to_vec` of a `RawFixture`: a proof, its verifier index and
//! its public inputs), printed one per line as `fixture <name> <hex>`, in
//! name order, for the serialization tests.
use std::{fs, path::Path};

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../proof-systems/kimchi/src/tests/fixtures");
    let mut files: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    for path in files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = fs::read(&path).unwrap();
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        println!("fixture {name} {hex}");
    }
}
