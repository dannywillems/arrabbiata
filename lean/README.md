# Arrabbiata in Lean

The Lean side of Arrabbiata has two layers.

1. **The specification library** (`Arrabbiata/`), written by hand against
   Mathlib: the mathematical statements the Rust code is meant to satisfy,
   proved once for all instances. It currently contains
   `Arrabbiata.Folding`, the identity behind folding:

   ```text
   P (x + r y) = P x + sum_{k=1}^{d-1} r^k T_k + r^d P y
   ```

   for every polynomial `P` homogeneous of degree `d > 0` over every
   commutative ring. Folding is therefore complete for every circuit,
   without a per-circuit proof.

2. **The extracted model** (`extracted/`, generated, not committed): the
   pure Lean translation of the Rust crates produced by Charon and Aeneas
   (`make extract`). The next step is to prove that the extracted functions
   refine the specification, for instance that the extracted
   `Fold::verify` computes the instance `fold_complete` talks about.

The distinction matters: a theorem about a hand-written model says nothing
about the Rust code until the model is derived from the code. Only the
extracted layer closes that gap.

## status

- The specification library is written but has NOT been built yet: Mathlib
  was not available in the session that wrote it. Run `make lean-setup` and
  `make lean-build`; until both pass, treat the proofs as unchecked drafts.
- Charon and Aeneas are not installed; `make extract` reports it and stops.

## building

```bash
make lean-setup   # lake update && lake exe cache get (prebuilt Mathlib)
make lean-build   # lake build --wfail, then scripts/check-axioms.sh
```

`--wfail` turns the `sorry` warning into an error, and
`scripts/check-axioms.sh` checks with `#print axioms` that the main theorems
depend only on `propext`, `Classical.choice` and `Quot.sound`. A plain green
`lake build` is not evidence of a proof.

## extracting

Install Charon and Aeneas following
<https://github.com/AeneasVerif/aeneas>, then:

```bash
make extract
```

The crates listed in `EXTRACTED_CRATES` in the Makefile are kept inside the
subset of Rust that Aeneas supports.

## toolchain

Lean `v4.30.0` and Mathlib `v4.30.0`, pinned together in `lean-toolchain` and
`lakefile.toml`. Move them together.
