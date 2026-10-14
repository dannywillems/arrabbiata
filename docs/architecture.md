# architecture

This document explains how the crates fit together, the mathematics each one
implements, what the type system enforces, and how much of it is tested,
proved, or only assumed. Claims labelled "our framing" are interpretations
written for this project, not results quoted from a source.

## goal

People write circuits in Rust with Arrabbiata, fold them, and prove things
about them in Lean. The Rust code is extracted to Lean with Charon and Aeneas,
so a Lean theorem about the extracted code is a theorem about the code that
runs. The library supplies the Lean theorems that hold for every circuit (for
instance, completeness of folding), so a user only proves what is specific to
their own circuit.

Three consequences shape the Rust code:

- The core crates are generic over small traits (`Field`, `Algebra`,
  `Transcript`, the commitment traits) that map to Lean structures. The Lean
  side instantiates `Field` with Mathlib's fields, so every lemma proved over
  the abstract structure holds for every concrete field.
- The core crates stay in the subset of Rust that Aeneas extracts: no
  `unsafe` (forbidden workspace-wide), no trait objects, no closures or
  iterator chains outside tests, plain `while` loops.
- Nothing depends on an external crate. Every primitive is implemented in the
  repository, behind a trait, so that each can be replaced (a post-quantum
  commitment, another permutation) without touching the protocols.

## the model-theoretic reading (our framing)

Each trait is a signature, its documented laws are the axioms, and each
implementation is a structure that should satisfy them. On the Rust side,
satisfaction is sampled: every law is a property test, and a shrunk
counterexample is a countermodel. On the Lean side it is decided: a law is a
theorem. Which of the two each law currently gets is recorded below.

## the crates

### `arrabbiata-field`

`Field` is a finite field with total operations (`inverse` returns an
`Option`). The axioms are executable predicates in `laws`, reused by every
concrete field's property tests. Goldilocks (`p = 2^64 - 2^32 + 1`) is the
reference instance; its multiplicative generator is checked in a test by
verifying that it is not a `q`-th power for every prime `q` dividing `p - 1`,
and that the factorization of `p - 1` is complete.

### `arrabbiata-poly`

An `Expr` is a term of the free commutative `F`-algebra on the variables of a
circuit: a multivariate polynomial as a syntax tree. Three evaluations are
defined, all generic over an `F`-algebra `A`:

- `evaluate`: the unique algebra morphism extending an assignment of the
  variables (the universal property of the polynomial ring).
- `evaluate_graded`: the homogeneous components `f_0, ..., f_d` of `f` with
  respect to the degree in the witness variables. Equivalently, `f_k(a)` is
  the coefficient of `t^k` in `f(t a)`, so graded evaluation is evaluation
  in `A[t]` along `w -> t a(w)`.
- `evaluate_homogeneous`: the homogenization `f^D(u, a) = sum_k u^(D-k) f_k(a)`,
  homogeneous of degree `D` in `(u, witness)`, with `f^D(1, a) = f(a)`.

Fixed columns (selectors) have degree zero: they are not folded, so they
behave as coefficients. The same code evaluates a row in `F` (to check a
witness) and in `F[r]` (to compute the cross-terms of a fold), because the
target algebra is a parameter. The algebra is passed as a value implementing
`Algebra` rather than as a bound on the element type, which avoids coherence
conflicts between `F` and `F[r]` and corresponds to a Lean structure
instance.

Laws tested: the components sum to the value; component `k` is homogeneous of
degree `k`; dehomogenization; homogeneity of `f^D`; the folding identity for
one expression; the ring laws of `F[r]`; the NTT agrees with naive evaluation
and is inverted by interpolation.

### `arrabbiata-relation`

A `ConstraintSystem<F, A, W, X, M, D>` has `W` witness and `X` fixed
columns and `M` constraints of degree at most `D`, aggregated by the strategy
`A`. Its relaxed relation is, on every row `i`,

```text
P(u, c, w_i) = sum_j c_j f_j^D(u, w_i) = E_i,
```

homogeneous of degree `d = D + A::EXTRA_DEGREE` in `(u, c, w)`. The
aggregation strategies:

- `Single`: no coefficient, `d = D`. Implemented only for `M = 1`.
- `Powers`: `c_j = alpha^j` on a fresh instance, then folded linearly like
  any instance variable, so `d = D + 1`. This is the combination of
  constraints of the Arrabbiata note (after ProtoStar): powers of a challenge
  cannot be folded as powers, so each power becomes its own variable.

Nova's relaxed R1CS is `Single` with `D = 2`: for a row constraint
`a * b - c`, the relaxed relation is `a * b - u * c = E`.

### `arrabbiata-folding`

The identity of the crate documentation:

```text
P(x + r y) = P(x) + sum_{k=1}^{d-1} r^k T_k + r^d P(y)
```

for `P` homogeneous of degree `d`. The prover computes `T_k` by evaluating
each row in `F[r]` along the line `x + r y`, and commits to them; the
verifier folds the commitments, using that commitment is a module morphism.
`Fold<F, PC, A, W, X, M, D, T>` is the protocol; `T`, the number of
cross-terms, is checked against `d - 1` at compile time. `Nova<F, PC, W, X>`
is the alias `Fold<F, PC, Single, W, X, 1, 2, 1>`.

Completeness is the identity above: it is proved in Lean
(`Arrabbiata.fold_complete`), for every homogeneous polynomial over every
commutative ring, and property-tested in Rust on arbitrary relaxed pairs
(not only fresh ones). Soundness (our framing of the standard argument):
`d + 1` accepting transcripts with distinct challenges determine the
polynomial in `r` by interpolation (the Vandermonde matrix of distinct
points is invertible), hence both witnesses, provided commitments are
binding. Neither the extractor nor the knowledge-soundness theorem is
formalized yet.

### `arrabbiata-field`: Montgomery fields and extensions

`MontField<C, N>` is a prime field on `N` limbs in Montgomery form. A
configuration states only the modulus and a quadratic non-residue; `R`,
`R^2`, `-p^-1 mod 2^64` and the two-adicity are computed by `const fn`, so
the modulus is the single hand-written number, and it is checked from
outside (primality, and the order of a curve over it). Multiplication is
CIOS; inversion is Fermat; square roots are Tonelli-Shanks. The tests compare
every operation with an independent reference (schoolbook product, binary
long division) on 1-, 2- and 4-limb moduli. `Quadratic<C>` builds
`F[u] / (u^2 - beta)`, inverting through the norm, which vanishes only at 0
exactly because `beta` is not a square.

### `arrabbiata-curve`, `arrabbiata-pasta`, `arrabbiata-bn254`

Curves `y^2 = x^3 + b` over any field, in homogeneous projective
coordinates, with the complete addition formula of Renes, Costello and
Batina (EUROCRYPT 2016, Algorithm 7). Completeness holds on curves whose
group has odd order, which covers Pallas, Vesta, BN254 `G1` and the BN254
twist (of order `r (2p - r)`, both factors odd). One formula with no branch
on point values is what makes the arithmetic short and uniform; it is
checked against textbook affine addition.

The Pasta curves form a cycle: Pallas over `Fp` has `q` points, Vesta over
`Fq` has `p`. BN254 adds `Fq2`, the twist `G2`, `Fq12 = Fq2[w] / (w^6 - xi)`,
and the reduced Tate pairing. The pairing is written for obvious
correctness: a Miller loop over the bits of `r` with the point arithmetic in
`G1` (affine, over `Fq`), line values in `Fq12`, vertical lines dropped
because their values lie in `Fq6`, which the factor `p^6 - 1` of the final
exponent kills (denominator elimination), and a plain exponentiation by
`(p^12 - 1) / r`. Tests check bilinearity, non-degeneracy, the order of the
target group, and that the final exponent times `r` is `p^12 - 1`.

The `Pairing` trait abstracts the pairing (two groups with a common scalar
field and a product-equals-one check), so KZG is written once for every
pairing.

### `arrabbiata-poseidon`

Poseidon is a family: its round constants and MDS matrix are the output of
the Grain LFSR seeded with the parameters, so an implementation must run the
generator exactly as the reference does. Ours follows the reference script
as implemented by `halo2_poseidon` (80-bit LFSR, self-shrinking, rejection
sampling for the round constants, reduction for the Cauchy MDS points), and
reproduces the Zcash permutation test vectors on both Pasta fields. The
BN254 instances use the same generator; no compatibility with another BN254
Poseidon is claimed.

### `arrabbiata-ipa` and `arrabbiata-kzg`

Both implement `PolynomialCommitment` with `Family = Univariate`, and
`HomomorphicCommitment`, so both serve PlonK and folding.

- IPA, over any curve: Pedersen vector commitments to the coefficients, and
  the Bulletproofs inner-product argument as in Halo, with the value bound by
  `U' = xi U`. No trusted setup; generators come from Poseidon over the base
  field. The verifier is linear in the degree. Not hiding.
- KZG, over any pairing: `C = [p(tau)] G`, opening proof `[q(tau)] G` for
  `q = (p - v) / (X - z)`, checked with one pairing product. Binding needs
  nobody to know `tau`; `insecure_setup` exists for tests, and a test shows
  how knowing `tau` opens a commitment to any value.

### `arrabbiata-transcript`

Two layers. `Transcript` is the untyped interface, implemented by a duplex
sponge over any `Permutation`. The typed layer makes the protocol a type:

- a pattern is a type-level list of `Public<M, _>` (statement, absorbed by
  both parties, not in the proof), `Message<M, _>` (prover message) and
  `Challenge<_>`, ending in `End`;
- `Then<P, Q>` concatenates patterns;
- `Pattern::Proof` projects a pattern onto its prover messages, an `HList`;
- `Prover` and `Verifier` tapes follow the pattern step by step, consuming
  themselves, so steps happen in order and no state is reused, and the
  prover can only finish with a value of exactly the proof type;
- `Protocol` names one transcript type from which both tapes are derived, so
  a prover and a verifier of the same protocol cannot disagree on it.

Our framing: patterns are words of the free monoid on the steps, `Then` is
the monoid operation with unit `End`, and `Proof` is a monoid morphism into
types under product: `Proof(Then<P, Q>)` is `Proof(P)` followed by
`Proof(Q)`. A tape is the state of the automaton reading the word; affine
moves make the state linear, which is the discipline of session types
(Honda, *Types for dyadic interaction*, CONCUR 1993). The design follows the
typed Fiat-Shamir experiment of o1-labs/proof-systems#2094.

The toy permutation is NOT cryptographic and serves the tests of the generic
code; protocols use the Poseidon sponge.

### `arrabbiata-commit`

Two separate traits, because protocols need different things:

- `HomomorphicCommitment`: vectors over `F`, with commitments forming an
  `F`-module and `commit` a module morphism. Folding requires it.
- `PolynomialCommitment`: polynomials of a `PolynomialFamily` (`Univariate`:
  coefficients, point in `F`; `Multilinear`: hypercube evaluations, point in
  `F^k`) with evaluation proofs. PlonK requires `Family = Univariate`.

Keeping them apart is deliberate. Lattice commitments (Greyhound, Hachi,
Akita; ePrint 2026/1983) are multilinear and homomorphic only for
combinations of small norm, so they are polynomial commitments but must not
be folded with arbitrary field challenges; the type system rejects that use.
A norm-aware folding (in the style of LatticeFold) would be a third trait.

The reference schemes (`IdentityCommitment`, `RevealCommitment`) publish the
committed object: binding, not hiding, not succinct.

### `arrabbiata-frontend`

A circuit builder for the vanilla PlonK arithmetization. Wires carry their
value and their cell; copy constraints become a permutation whose cycles are
the equivalence classes of cells. Wires are branded by their builder with a
generative lifetime: `build` takes a closure `for<'id> FnOnce(&mut
Builder<'id, F>) -> R`, the rank-2 type of `runST` (Launchbury and Peyton
Jones, *Lazy functional state threads*, PLDI 1994), as used by GhostCell
(Yanovski et al., ICFP 2021). Wires of two circuits have incomparable types,
and no wire escapes its builder.

### `arrabbiata-plonk`

Vanilla PlonK (ePrint 2019/953) over any two-adic field and any univariate
polynomial commitment, on the typed transcript. It is a toy: no blinding (not
zero-knowledge), no linearization, unbatched openings, schoolbook
multiplication.

## deviations and next steps

- **Copy constraints in folding.** The folding relation has no permutation
  argument yet, so folding a PlonK circuit ignores its wiring. The relation
  needs the permutation (or a lookup) argument as part of the folded relation.
- **Circuit digest.** The folding transcript absorbs the whole constraint
  system on every fold. An IVC verifier needs a digest instead.
- **The verifier circuit on the cycle.** The fields, curves, Poseidon and the
  commitments of the cycle exist; the in-circuit verifier (non-native
  arithmetic, EC gadgets, Poseidon gadget) does not yet.
- **Performance.** Naive MSM (no Pippenger), Tate pairing over the bits of
  `r` (not optimal ate), final exponentiation without Frobenius, schoolbook
  polynomial products, unbatched PlonK openings.
- **Generalized PlonK**, beyond the vanilla gate.
- **Extraction.** Charon and Aeneas are not run yet. The first pilot is
  `arrabbiata-poly` (recursion over boxed expressions, generic algebras), to
  measure what extracts before relying on it.
- **Aeneas-sensitive constructs** present in extracted crates, to check in
  the pilot: `core::ops` operator traits on the field, inline `const`
  blocks, `core::array::from_fn` with a closure in
  `ConstraintSystem::commit`, and `TryFrom<Vec<T>>` for arrays.

## laws and where they stand

| law | tested (sampled) | proved in Lean |
| --- | --- | --- |
| field axioms (Goldilocks, Montgomery fields) | yes, against a long-division reference | by Mathlib for `ZMod p` |
| curve group laws, complete formulas = affine | yes | no |
| pairing bilinearity, non-degeneracy, order | yes | no |
| Poseidon = reference permutation | yes, Zcash test vectors | no |
| IPA and KZG completeness, binding, morphism | yes | no |
| graded and homogeneous evaluation | yes | no |
| folding identity | yes, on arbitrary relaxed pairs | yes, `fold_complete` (built in CI) |
| prover and verifier fold to the same instance | yes | by construction (shared function) |
| commitment is a module morphism (reference scheme) | yes | no |
| PCS completeness and evaluation binding (reference) | yes | no |
| multilinear extension interpolates its table | yes | no |
| PlonK completeness | yes, random circuits, with the reference scheme, KZG and IPA | no |
| folding with Pedersen and KZG commitments | yes | no |
| PlonK rejects tampered evaluations, statements, transcripts | yes | no |
| folding knowledge soundness | sanity tests only | no |
