//! Kimchi's runs: the vectors `make kimchi-vectors` writes from
//! o1-labs/proof-systems at the commit in `tools/kimchi-vectors/KIMCHI_REV`.
//! They are generated, not committed. They are read from `tests/vectors/`,
//! or from the directory `KIMCHI_VECTORS` names, for a copy of the tree
//! that leaves ignored files out (cargo-mutants' build directories).

use std::{env, fs, path::PathBuf};

/// The vector file `name`.
///
/// # Panics
///
/// If it is missing: `make kimchi-vectors` writes it.
pub fn read(name: &str) -> String {
    let dir = env::var_os("KIMCHI_VECTORS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors"),
        PathBuf::from,
    );
    let path = dir.join(name);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}; run `make kimchi-vectors`", path.display()))
}
