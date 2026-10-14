# Arrabbiata

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
  Vesta) and BN254 (`G1`, `G2`) with its pairing;
- the Poseidon permutation over any prime field, matching the Zcash test
  vectors on the Pasta fields;
- two real polynomial commitment schemes: IPA over any curve (no trusted
  setup) and KZG over any pairing;
- constraint systems, the relaxed relation, and folding of homogeneous
  relations of any degree (Nova is the degree-2 instance), with real
  (Pedersen or KZG) commitments;
- the typed Fiat-Shamir transcript;
- vanilla PlonK, instantiated with KZG on BN254 and with IPA on Pallas and
  on BN254;
- the Lean statement and proof of completeness of folding.

Not yet implemented: the IVC verifier circuit on the cycle, the permutation
argument inside the folded relation, zero-knowledge (blinding), and
performance work (the pairing, MSMs and polynomial products use the plain
textbook algorithms). `IdentityCommitment` and `RevealCommitment` remain as
reference schemes for tests: binding, but neither hiding nor succinct.

## layout

Crates are grouped by the abstraction they implement; each group has a
README, and each crate a single responsibility. An abstraction (a trait
crate) sits beside its instances, so a new scheme or curve is a new crate in
the right group.

| directory | crate | contents | extracted to Lean |
| --- | --- | --- | --- |
| `crates/algebra` | `arrabbiata-field` | the `Field` traits; Montgomery prime fields; quadratic extensions; Goldilocks | yes |
| | `arrabbiata-poly` | typed columns, constraint expressions, graded evaluation and homogenization, univariate polynomials, NTT domains | yes |
| `crates/ec` | `arrabbiata-curve` | the `Curve` and `Pairing` traits, complete formulas, MSM, the group laws | not yet |
| | `arrabbiata-pasta` | Pallas, Vesta and their fields | not yet |
| | `arrabbiata-bn254` | BN254 fields, `G1`, `G2`, `Fq12`, the pairing | not yet |
| `crates/oracle` | `arrabbiata-transcript` | sponges over a permutation; the typed Fiat-Shamir transcript | yes |
| | `arrabbiata-poseidon` | Poseidon with Grain-generated constants; Pasta and BN254 instances | not yet |
| `crates/commitment` | `arrabbiata-commit` | homomorphic vector commitments; polynomial commitments by family; reference schemes | yes |
| | `arrabbiata-ipa` | the inner-product argument, over any curve | not yet |
| | `arrabbiata-kzg` | KZG, over any pairing | not yet |
| `crates/arithmetization` | `arrabbiata-relation` | constraint systems, relaxed relations, aggregation strategies | yes |
| | `arrabbiata-frontend` | a typed circuit builder (vanilla PlonK arithmetization) | no |
| `crates/accumulation` | `arrabbiata-folding` | folding of homogeneous relations of any degree; Nova as a special case | yes |
| `crates/snark` | `arrabbiata-plonk` | vanilla PlonK, with KZG (`kzg`) and IPA (`ipa`) | no |

"Extracted" means the crate is kept inside the subset of Rust that Aeneas
supports: no `unsafe`, no trait objects, no closures or iterator chains in
non-test code.

## example: fold two instances

```rust
use arrabbiata_commit::IdentityCommitment;
use arrabbiata_field::Goldilocks as F;
use arrabbiata_folding::Fold;
use arrabbiata_relation::{aggregation::Powers, examples};
use arrabbiata_transcript::toy_sponge;

// Two constraints of degree 2, combined by folded powers of a challenge:
// the relation has degree 3, and a fold has 2 cross-terms.
type Scheme = Fold<F, IdentityCommitment, Powers, 2, 1, 2, 2, 2>;

let cs = examples::fibonacci(8);
let ck = IdentityCommitment;
let fresh = |x, y| {
    let trace = examples::fibonacci_trace(&cs, x, y);
    let instance = cs.commit(&ck, &trace, &mut toy_sponge(1));
    cs.relax(&ck, instance, trace)
};
let (acc, acc_w) = fresh(1, 2);
let (new, new_w) = fresh(3, 4);

let (folded, folded_w, proof, _) =
    Scheme::prove(&cs, &ck, toy_sponge(0), (&acc, &acc_w), (&new, &new_w)).unwrap();
let (verified, _) = Scheme::verify(&cs, &ck, toy_sponge(0), &acc, &new, proof);
assert_eq!(folded, verified);
cs.check_relaxed(&ck, &verified, &folded_w).unwrap();
```

## example: write a circuit and prove it with PlonK

```rust
use arrabbiata_commit::{RevealCommitment, Univariate};
use arrabbiata_field::Goldilocks as F;
use arrabbiata_frontend::build;
use arrabbiata_plonk::{prove, setup, verify};
use arrabbiata_transcript::toy_sponge;

let (circuit, witness, ()) = build::<F, _, _>(|b| {
    let x = b.witness(F::new(3));
    let x2 = b.mul(&x, &x);
    let x3 = b.mul(&x2, &x);
    let sum = b.add(&x3, &x);
    let lhs = b.add_constant(&sum, F::new(5));
    let rhs = b.public_input(F::new(35));
    b.assert_equal(&lhs, &rhs);
});

let pcs = RevealCommitment::<Univariate>::new();
let (pk, vk) = setup(&circuit, &pcs).unwrap();
let proof = prove(&pk, &pcs, toy_sponge(0), &witness).unwrap();
verify(&vk, &pcs, toy_sponge(0), witness.public_inputs(), proof).unwrap();
```

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
- A wire of a circuit cannot be used in another circuit, or escape its
  builder (generative lifetime brand).

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
make lean-setup    # fetch Mathlib (once)
make lean-build    # build the Lean library and check its axioms
```

CI runs the same targets, on stable and beta.

## license

MIT. See [LICENSE](LICENSE).

## references

- Kothapalli, Setty, Tzialla. Nova: Recursive Zero-Knowledge Arguments from
  Folding Schemes. CRYPTO 2022. <https://eprint.iacr.org/2021/370>
- Bunz, Chen. ProtoStar: Generic Efficient Accumulation/Folding for Special
  Sound Protocols. ASIACRYPT 2023. <https://eprint.iacr.org/2023/620>
- Willems. Behind Nova: cross-terms computation for high degree gates.
  <https://hackmd.io/qq_Awc1AR3ywzkruE4Wq9Q>
- Gabizon, Williamson, Ciobotaru. PlonK. <https://eprint.iacr.org/2019/953>
- Ho, Protzenko. Aeneas: Rust verification by functional translation. ICFP
  2022. <https://github.com/AeneasVerif/aeneas>

[Charon]: https://github.com/AeneasVerif/charon
[Aeneas]: https://github.com/AeneasVerif/aeneas
[proptest]: https://github.com/proptest-rs/proptest
