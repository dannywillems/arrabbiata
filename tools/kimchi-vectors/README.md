# kimchi-vectors

Kimchi's runs, printed as the vectors arrabbiata's regression tests read.
The tool builds against o1-labs/proof-systems at the commit in
[`KIMCHI_REV`](KIMCHI_REV). The vectors are generated, not committed:

    make kimchi-vectors

`make kimchi-checkout` fetches that one commit (depth 1, without its
submodules: the Optimism submodule of `o1vm/` is not used by Kimchi) into
`tools/proof-systems/`, and checks that its `HEAD` is `KIMCHI_REV` and its
tree is clean. The tool depends on it by path. `Cargo.lock` started as
proof-systems' own lock at that commit, so the shared dependencies are at
the versions Kimchi is tested with, and the build runs `--locked`. The
checkout sits beside the tool, not inside it: a path dependency inside a
workspace's directory would become a member of that workspace.

CI generates the vectors in the `kimchi-generate` job (the composite action
`.github/actions/kimchi-vectors`), cached on the tool, the commit, the
Makefile and `arrabbiata_proofs.txt`, and uploads them as the
`kimchi-vectors` artifact. The test jobs restore them from that cache.
The tests read them from `tests/vectors/`, or from the directory
`KIMCHI_VECTORS` names: cargo-mutants copies the tree without its ignored
files, so the Mutants workflow points its runs at a copy outside the tree.

Every run is deterministic: fixed seeds (`StdRng::from_seed([k; 32])`,
SplitMix64), so two runs write the same bytes.

## what each binary writes

| binary | vector |
| --- | --- |
| `shifts` | `crates/snark/kimchi/tests/vectors/kimchi_shifts.txt` |
| `constraint_system` | `.../kimchi_constraint_system.txt` |
| `verifier_index` | `.../kimchi_verifier_index.txt` |
| `oracles` | `.../kimchi_oracles.txt` |
| `gate_tests` | `.../kimchi_gate_tests.txt` |
| `prover <arrabbiata_proofs.txt>` | `.../kimchi_prover.txt` (stdout), `.../kimchi_verifies_arrabbiata.txt` (stderr) |
| `differential` | `crates/gadgets/kimchi/tests/vectors/kimchi_differential.txt` |

Two binaries print tables kept inline in the tests, not files:
`kimchi-vectors` (the constraint counts and degrees of each gate, the
`KIMCHI` table of `crates/gadgets/kimchi/tests/figures.rs`) and
`ipa_vectors` (`b_poly`, `b_poly_coefficients`, `combined_inner_product`,
in `crates/commitment/ipa/tests/kimchi.rs`).

- `shifts`: `cs.shift` for domains of 16, 128 and 1024 rows.
- `constraint_system`: four circuits through `ConstraintSystem::build` and
  a `ProverIndex`; the domain, `zk_rows`, the generators of d1 and d8, and
  the Blake2b512 digest of every column of `column_evaluations`.
- `verifier_index`: the same circuits' verifier indexes, every commitment
  chunk as affine coordinates.
- `oracles`: fixed scripts through `DefaultFqSponge`, `DefaultFrSponge` and
  `ScalarChallenge::to_field`.
- `gate_tests`: Kimchi's gate unit tests (`kimchi/src/tests/`), one `test`
  block each: the inputs, the witnesses, Kimchi's rows (`gates`, trailing
  padding left out), every verdict (`Debug` of Kimchi's result) and, for
  the serialization regression, the proof.
- `prover`: proofs of the generic, chunked, gadgets, recursion and lookup
  circuits, every commitment, evaluation and oracle; given our dump, Kimchi's
  verifier on our proofs, on a tampered copy and on its own.
- `differential`: every gate's `constraint_checks` at random points, and
  Kimchi's witness generators on fixed inputs.
