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

Early. The algebra, the relations, the folding scheme, the typed transcript
and a toy vanilla PlonK are implemented and tested. The concrete field is the
64-bit Goldilocks field; the curves of a cycle, a real polynomial commitment
scheme, and the IVC verifier circuit are not implemented yet. The commitment
schemes in the repository are reference schemes (publish the committed
vector); they are binding but neither hiding nor succinct, and exist to test
the protocols independently of the cryptography.

## crates

| crate | contents | extracted to Lean |
| --- | --- | --- |
| `arrabbiata-field` | the `Field` trait; Goldilocks | yes |
| `arrabbiata-poly` | typed columns, constraint expressions, graded evaluation and homogenization, NTT domains | yes |
| `arrabbiata-transcript` | sponges over a permutation; typed Fiat-Shamir transcripts | yes |
| `arrabbiata-commit` | homomorphic vector commitments; polynomial commitments by family (univariate, multilinear) | yes |
| `arrabbiata-relation` | constraint systems, relaxed relations, aggregation strategies | yes |
| `arrabbiata-folding` | folding of homogeneous relations of any degree; Nova as a special case | yes |
| `arrabbiata-frontend` | a typed circuit builder (vanilla PlonK arithmetization) | no |
| `arrabbiata-plonk` | vanilla PlonK, as an end-to-end example | no |

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
- A wire of a circuit cannot be used in another circuit, or escape its
  builder (generative lifetime brand).

Each of these is pinned by a `compile_fail` doctest.

## formal verification

The `lean/` directory holds the Lean library. It currently proves the identity
the folding scheme rests on, for every homogeneous polynomial over every
commutative ring, so folding is complete for every circuit at once. See
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
