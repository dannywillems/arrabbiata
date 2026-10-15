# Arrabbiata

Auto-research project.

A library to write circuits in Rust and to fold them, designed so that the
code can be extracted to Lean (with [Charon] and [Aeneas]) and reasoned about
there.

Arrabbiata is generic over the field, the polynomial commitment scheme, the
sponge, and the aggregation of constraints. It has no dependency outside the
Rust standard library; property tests use [proptest] as a dev-dependency only.

API documentation, with an example on every public item:
<https://dannywillems.github.io/arrabbiata/>.

## status

Early, but end to end. Implemented and tested:

- prime fields in Montgomery form, generic over the modulus (Goldilocks,
  the Pasta fields, the BN254 fields), and quadratic extensions;
- elliptic curves with complete addition formulas: the Pasta cycle (Pallas,
  Vesta), BN254 (`G1`, `G2`) with its pairing (checked against the
  EIP-196/197 vectors), and Grumpkin, which forms a cycle with BN254;
- the Poseidon permutation over any prime field, matching the Zcash test
  vectors on the Pasta fields and the reference vectors of its authors on
  BN254;
- two real polynomial commitment schemes: IPA over any curve (no trusted
  setup) and KZG over any pairing;
- constraint systems, the relaxed relation, and folding of homogeneous
  relations of any degree (Nova is the degree-2 instance), with real
  (Pedersen or KZG) commitments;
- the typed Fiat-Shamir transcript;
- reductions of knowledge as types: relations, reductions, their
  composition and products, arguments as reductions to the trivial
  relation;
- PlonKish circuits (gadgets, one-hot selectors, rotations, copies, public
  inputs) and their relaxation;
- PlonK over any number of witness columns, fixed columns and selectors
  (custom gates of any degree, reading any row at a fixed offset), with
  vanilla PlonK as an instance; instantiated with KZG on BN254 and with IPA
  on Pallas and on BN254;
- generalized PlonK: any number of witness columns, fixed columns,
  selectors and public inputs, with gates that are any multivariate
  polynomials of bounded degree (public inputs read as scalars), into which
  PlonKish circuits embed; with lookup arguments, plookup and LogUp, for
  vector lookups of polynomial queries into fixed tables;
- the folding scheme of the Arrabbiata note on PlonKish circuits of any
  degree, with the cross-terms computed from multivariate polynomials, copy
  constraints and public inputs folded linearly, and a decider that proves
  the relaxed relation as a generalized PlonK circuit: accumulate any number
  of steps, then prove them with one proof;
- the Lean statement and proof of completeness of folding.

Not yet implemented: the IVC verifier circuit on the cycle, zero-knowledge
(blinding), and
performance work (the pairing, MSMs and polynomial products use the plain
textbook algorithms).

## layout

Crates are grouped by the abstraction they implement; each group has a
README, and each crate a single responsibility. An abstraction (a trait
crate) sits beside its instances, so a new scheme or curve is a new crate in
the right group.

| directory | crate | contents | extracted to Lean |
| --- | --- | --- | --- |
| `crates/algebra` | `arrabbiata-field` | the `Field` traits; Montgomery prime fields; quadratic extensions; Goldilocks | yes |
| | `arrabbiata-mvpoly` | multivariate polynomials in normal form (sparse, and dense through the primes), cross-terms of folding, Hasse derivatives | no |
| | `arrabbiata-poly` | typed columns, constraint expressions, graded evaluation and homogenization, univariate polynomials, NTT domains | yes |
| `crates/ec` | `arrabbiata-curve` | the `Curve` and `Pairing` traits, complete formulas, MSM, the group laws | not yet |
| | `arrabbiata-pasta` | Pallas, Vesta and their fields | not yet |
| | `arrabbiata-bn254` | BN254 fields, `G1`, `G2`, `Fq12`, the pairing | not yet |
| | `arrabbiata-grumpkin` | Grumpkin, the curve forming a cycle with BN254 | not yet |
| `crates/oracle` | `arrabbiata-transcript` | sponges over a permutation; the typed Fiat-Shamir transcript | yes |
| | `arrabbiata-poseidon` | Poseidon with Grain-generated constants; Pasta and BN254 instances | not yet |
| `crates/commitment` | `arrabbiata-commit` | homomorphic vector commitments; polynomial commitments by family | yes |
| | `arrabbiata-ipa` | the inner-product argument, over any curve | not yet |
| | `arrabbiata-kzg` | KZG, over any pairing | not yet |
| `crates/reduction` | `arrabbiata-reduction` | reductions of knowledge: relations, reductions, composition, products, arguments | no |
| `crates/arithmetization` | `arrabbiata-relation` | constraint systems, relaxed relations, aggregation strategies | yes |
| | `arrabbiata-plonkish` | PlonKish circuits: gadgets, one-hot selectors, rotations, copies; their relaxation; the vanilla gate | no |
| | `arrabbiata-frontend` | typed circuit builders: `vanilla`, for vanilla PlonK | no |
| `crates/accumulation` | `arrabbiata-folding` | folding of homogeneous relations of any degree; Nova as a special case | yes |
| | `arrabbiata-plonkish-folding` | accumulation of PlonKish circuits: the embedding `R -> R'` and the fold `R' x R' -> R'`, cross-terms from `mvpoly` | no |
| `crates/decider` | `arrabbiata-plonk-decider` | the decider `R' -> R_top` of `plonkish-folding`: the relaxation as a generalized PlonK circuit | no |
| `crates/snark` | `arrabbiata-plonk` | PlonK with custom gates over any number of columns; vanilla PlonK as an instance; KZG (`kzg`) and IPA (`ipa`) | no |
| | `arrabbiata-generalized-plonk` | generalized PlonK: any numbers of columns, selectors and public inputs, gates any multivariate polynomials of degree at most `D`; plookup and LogUp lookups | no |

"Extracted" means the crate is kept inside the subset of Rust that Aeneas
supports: no `unsafe`, no trait objects, no closures or iterator chains in
non-test code.

## examples

Every example lives in the API documentation, where it is compiled and run
as a doctest, so it cannot drift from the code. Good entry points:

- folding two instances: the crate documentation of `arrabbiata-folding`;
- accumulating PlonKish circuits, and deciding the accumulator with one
  PlonK proof: the crate documentation of `arrabbiata-plonkish-folding` and
  of `arrabbiata-plonk-decider`;
- reductions of knowledge and their composition: the crate documentation of
  `arrabbiata-reduction`;
- writing custom gates and proving a circuit with PlonK: the crate
  documentation of `arrabbiata-plonk`;
- gates as multivariate polynomials reading public inputs: the crate
  documentation of `arrabbiata-generalized-plonk`;
- vanilla PlonK from a circuit of `arrabbiata_frontend::vanilla`:
  `arrabbiata_plonk::vanilla`, with
  `arrabbiata_plonk::ipa` and `arrabbiata_plonk::kzg`;
- the typed transcript: `arrabbiata_transcript::typed`.

## what the type system enforces

- A column index is checked against the number of columns of its circuit,
  at compile time for literals.
- An aggregation strategy without combiner only accepts one constraint
  (`Single` implements `Aggregation<F, 1>` and nothing else).
- The number of cross-terms of a folding scheme matches the degree of its
  relation, at compile time.
- A prover and a verifier run the same transcript: both are derived from one
  `Protocol::Transcript` type, the statement must be absorbed before any
  challenge, a sponge state cannot be reused, and the proof is a typed list
  of exactly the messages the protocol declares.
- A PlonK prover only accepts a univariate polynomial commitment scheme; a
  folding scheme only accepts a commitment that is a module morphism (which a
  lattice commitment, homomorphic only for short vectors, is not).
- A PlonK proof is typed by its commitment scheme: a KZG proof does not
  type-check against an IPA verifier, even over the same field and curve.
- A PlonK circuit only turns on gadgets of its gate set, and its number of
  selectors is the number of gadgets; the index of a selector is the
  position of its gadget in the gate set type.
- A wire of a circuit cannot be used in another circuit, or escape its
  builder (generative lifetime brand).
- Reductions compose only when the target of the first is the source of
  the second (`Then`), and an argument is exactly a reduction whose target
  is the trivial relation (`Argument`).

Each of these is pinned by a `compile_fail` doctest.

## formal verification

The `lean/` directory holds the Lean library, built in CI with the toolchain
of the Aeneas Lean library. It proves the identity the folding scheme rests
on, for every homogeneous polynomial over every commutative ring, so folding
is complete for every circuit at once; the axiom census shows only `propext`,
`Classical.choice` and `Quot.sound`. See
[`lean/README.md`](lean/README.md) and [`docs/architecture.md`](docs/architecture.md).

## development

```bash
make help          # list targets
make test          # unit, property, doc and compile-fail tests
make lint          # clippy, warnings denied
make doc-examples  # require an example on every public item (nightly)
make check-format  # nightly rustfmt and taplo
make mutants       # mutation testing (PACKAGE=, FILE=, SHARD=k/n)
make mutants-diff  # mutation testing of the changes since BASE (origin/main)
make lean-setup    # fetch Mathlib (once)
make lean-build    # build the Lean library and check its axioms
```

CI runs the same targets, on stable and beta. Mutation testing
([cargo-mutants](https://mutants.rs)) runs in its own workflow: on the
changed code of each pull request, and on the whole workspace weekly.

## license

MIT. See [LICENSE](LICENSE).

## references

- Kothapalli, Setty, Tzialla. Nova: Recursive Zero-Knowledge Arguments from
  Folding Schemes. CRYPTO 2022. <https://eprint.iacr.org/2021/370>
- Bunz, Chen. ProtoStar: Generic Efficient Accumulation/Folding for Special
  Sound Protocols. ASIACRYPT 2023. <https://eprint.iacr.org/2023/620>
- Kothapalli, Parno. Algebraic Reductions of Knowledge. CRYPTO 2023.
  <https://eprint.iacr.org/2022/009>
- Bunz, Chiesa, Lin, Mishra, Spooner. Proof-Carrying Data without Succinct
  Arguments. CRYPTO 2021. <https://eprint.iacr.org/2020/1618>
- Willems. Behind Nova: cross-terms computation for high degree gates.
  <https://hackmd.io/qq_Awc1AR3ywzkruE4Wq9Q>
- Gabizon, Williamson, Ciobotaru. PlonK. <https://eprint.iacr.org/2019/953>
- Gabizon, Williamson. plookup: A simplified polynomial protocol for lookup
  tables. <https://eprint.iacr.org/2020/315>
- Haboeck. Multivariate lookups based on logarithmic derivatives.
  <https://eprint.iacr.org/2022/1530>
- Ho, Protzenko. Aeneas: Rust verification by functional translation. ICFP
  2022. <https://github.com/AeneasVerif/aeneas>

[Charon]: https://github.com/AeneasVerif/charon
[Aeneas]: https://github.com/AeneasVerif/aeneas
[proptest]: https://github.com/proptest-rs/proptest
