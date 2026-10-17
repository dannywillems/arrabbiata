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

The crates live under `crates/<abstraction>/<crate>`: `algebra`, `ec`,
`oracle`, `commitment`, `iop`, `reduction`, `arithmetization`,
`accumulation`, `decider` and `snark` (see the README of each directory). The sections
below follow the crates.

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

GLV (`glv`): every curve here has `j = 0`, so `phi(x, y) = (beta x, y)`, for
a cube root of unity `beta` of the base field, is an endomorphism, acting on
the prime-order group as a cube root of unity `lambda` of the scalars. A
scalar splits as `k = k1 + k2 lambda` against a short basis of the lattice
`{(a, b) : a + b lambda = 0 mod r}` (Guide to ECC, Algorithm 3.74), with
halves of at most 128 bits, and `[k] P = [k1] P + [k2] phi(P)` runs both in
one loop of 128 doublings. The divisions of the rounding are precomputed
(`round(2^256 b / r)`), and the rest is arithmetic modulo `2^256`, exact
because the halves are small. Pallas, Vesta, BN254 `G1` and `G2` and
Grumpkin carry their constants, generated and checked by a script: `beta`
is the multiplicative generator to the power `(p - 1) / 3` (Kimchi's
convention), and `lambda` the eigenvalue of `phi` on the generator. Pallas'
and Vesta's are Kimchi's `endos`, as Kimchi prints them; on BN254, `phi` acts
on `G2` as `lambda^2` of `G1`'s `lambda`. The laws tested on every curve:
`beta` and `lambda` are primitive cube roots of unity, `phi` keeps points on
the curve, is additive, has order 3 and is `[lambda]`; the split recomposes
the scalar; and GLV multiplication is scalar multiplication.

### `arrabbiata-poseidon`

Poseidon is a family: its round constants and MDS matrix are the output of
the Grain LFSR seeded with the parameters, so an implementation must run the
generator exactly as the reference does. Ours follows the reference script
as implemented by `halo2_poseidon` (80-bit LFSR, self-shrinking, rejection
sampling for the round constants, reduction for the Cauchy MDS points), and
reproduces the Zcash permutation test vectors on both Pasta fields. The
BN254 instances use the same generator; no compatibility with another BN254
Poseidon is claimed.

`kimchi` is the instance of Kimchi (o1-labs/proof-systems, `mina-poseidon`):
width 3, rate 2, 55 full rounds, `x^7`, and each round S-box, then MDS, then
round constants, with no key before the first round. Its constants are not
Grain's for these parameters: they are transcribed from proof-systems
(`fp_kimchi.rs`, `fq_kimchi.rs`) as limbs, checked below the modulus at
compile time. Its sponge is Kimchi's: a zero state, no domain, and two
squeezes per permutation. It reproduces `kimchi.json` over `Fp` and the
empty-sponge challenge over `Fq`.

### `arrabbiata-ipa` and `arrabbiata-kzg`

Both implement `PolynomialCommitment` with `Family = Univariate`, and
`HomomorphicCommitment`, so both serve PlonK and folding.

- IPA, over any curve: Pedersen vector commitments to the coefficients, and
  the Bulletproofs inner-product argument as in Halo and Kimchi, with the
  value bound by `U' = xi U`. No trusted setup; generators come from Poseidon
  over the base field. Not hiding. The rounds fold as Kimchi's (`a_lo +
  u^-1 a_hi`, `b_lo + u b_hi`, `G_lo + u G_hi`) and the proof carries `sg`,
  the folded generator. One proof covers several points: the evaluation
  vector is `sum_j evalscale^j (1, x_j, x_j^2, ...)`, and with the
  polynomials combined by `polyscale` the value is Kimchi's
  `combined_inner_product`. The verifier is two halves: `succinct`, `O(k)`,
  which trusts `sg` and returns the accumulator `(u, sg)`
  (`RecursionChallenge`), and `decide`, the MSM `sg = <s, G>` (`decide_all`
  for many, with one MSM). `b_poly`, `b_poly_coefficients` and
  `combined_inner_product` agree with Kimchi's on Kimchi's own output.
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

Two oracles implement `Transcript`: the duplex sponge over a field
permutation (Poseidon, in `arrabbiata-poseidon`, also Kimchi's sponge), and
`hash::HashTranscript`, a hash chain over any `ByteHash` (a 256-bit hash of
bytes: SHA-256 in the tests, a dev-dependency only), for protocols, such as
post-quantum ones, that have no reason to use an arithmetic sponge. A
challenge reduces at least twice the field's bits, so it is uniform up to
`2^-|p|` in the random oracle model. Every protocol is generic over the
oracle, and the accumulation laws run on all three. The repository has no
non-cryptographic stand-in.

### `arrabbiata-commit`

Two separate traits, because protocols need different things:

- `HomomorphicCommitment`: vectors over `F`, with commitments forming an
  `F`-module and `commit` a module morphism. Folding requires it.
- `PolynomialCommitment`: polynomials of a `PolynomialFamily` (`Univariate`:
  coefficients, point in `F`; `Multilinear`: hypercube evaluations, point in
  `F^k`) with evaluation proofs. PlonK requires `Family = Univariate`. A
  family also names an owned form of its points (`OwnedPoint`: `F`, or
  `Coordinates` for `F^k`), which claims and accumulators hold, so that
  accumulation (`arrabbiata-halo`) is written once for every family.

Keeping them apart is deliberate. Lattice commitments (Greyhound, Hachi,
Akita; ePrint 2026/1983) are multilinear and homomorphic only for
combinations of small norm, so they are polynomial commitments but must not
be folded with arbitrary field challenges; the type system rejects that use.
A norm-aware folding (in the style of LatticeFold) would be a third trait.

The schemes are IPA (Pedersen vector commitments with a logarithmic
opening, transparent) and KZG (over any pairing, trusted setup).

### `arrabbiata-mvpoly`

Polynomials in `N` variables of degree at most `D`, in normal form, after
the `mvpoly` crate of o1-labs/proof-systems. One trait, `MVPoly<F, N, D>`,
two representations: `Sparse` (the nonzero terms, keyed by monomial) and
`Dense` (every coefficient, the monomial `x^e` indexed by
`prod_i p_i^(e_i)` over the first `N` primes, an injective monoid
morphism by unique factorization). Everything but storage is written once
in the trait, so the two cannot drift; they are tested to be isomorphic.
The product is partial, `checked_mul`, as `F[x]_(<= D)` is not closed
under it.

Cross-terms are the coefficients of `P^D(u_1 + r u_2, x_1 + r x_2)` in
`r`, by the binomial expansion of each monomial; they are the directional
Hasse derivatives of `P^D`, so the expansion needs no division and holds
in any characteristic. `combined_cross_terms` combines constraints by
powers of a folded combiner, as Arrabbiata's folding does. `from_expr`
turns a constraint into its polynomial on a row, under a `Layout` of
rotations (the current and the next row, in Arrabbiata's layout). The
cross-terms agree, coefficient by coefficient, with those the folding
prover computes by evaluating the relation in `F[r]`: two independent
computations of the same object.

### `arrabbiata-frontend`

Frontends, one module per arithmetization. `vanilla` is the circuit
builder for the vanilla PlonK arithmetization. Wires carry their
value and their cell; copy constraints become a permutation whose cycles are
the equivalence classes of cells. Wires are branded by their builder with a
generative lifetime: `build` takes a closure `for<'id> FnOnce(&mut
Builder<'id, F>) -> R`, the rank-2 type of `runST` (Launchbury and Peyton
Jones, *Lazy functional state threads*, PLDI 1994), as used by GhostCell
(Yanovski et al., ICFP 2021). Wires of two circuits have incomparable types,
and no wire escapes its builder.

### `arrabbiata-plonk`

PlonK (ePrint 2019/953) generalized to custom gates, over any two-adic
field, any univariate polynomial commitment and any sponge, on the typed
transcript: `Plonk<F, PC, S, G, W, X, Q>` with `W` witness columns, `X`
fixed columns and a gate set `G` of `Q` gadgets.

The arithmetization follows the original Arrabbiata gadgets: each gadget is
a list of constraints turned on by its own selector, the protocol
multiplying every constraint by its gadget's selector. The gate set is a
type-level list `(G0, (G1, ()))`, so the index of a selector is the
position of its gadget, computed by the compiler (`Member<G, I>`, with the
position `I` inferred), and a builder cannot turn on a gadget outside the
gate set. `Q` is checked against the length of the list at compile time.

- **One gadget per row, by construction.** A row stores an
  `Option<Selector<Q>>`, an element of the coproduct `1 + Q`, and the
  selector columns are derived from it, so they are one-hot in every
  circuit that can be written down. This is a design choice, not a
  requirement of the protocol: every gated constraint has its own power of
  `alpha`, so the identity would hold term by term for any selector values.
  On `H` the one-hot selectors are orthogonal idempotents of
  `F[X]/(X^n - 1)`, and the gated identity is a case split.
- **Rotations.** A constraint reads a column at any offset `k` in `Z` from
  the constrained row (`Rotation`), acting on the rows `Z/n` by
  translation, i.e. the automorphism `X |-> w^k X` of `F[X]/(X^n - 1)`.
  Each pair `(column, k)` a gate set reads is opened at `zeta w^k`.
- **Rows: a cycle for the protocol, a path for the builder.** The protocol
  reads rows on `Z/n`, so a rotation past the last row wraps to the first.
  `Builder::new` lays rows on a path and rejects a gadget reading a row
  outside the ones added, so padding to a power of two is never observed
  and adding rows never changes what existing ones mean. `Builder::cyclic`
  opts into the cycle (and needs a power of two rows); there the relation
  is invariant under rotating all rows, as for a shift of finite type.
- **Copies** range over all `W n` witness cells, with the cosets `g^k H`
  labelling the columns. `sigma` is a function of the partition the copies
  generate, and a witness satisfies them iff it is constant on the
  classes.
- **Public inputs** are bound to witness column 0 by the term
  `L_P w_0 - PI`; a circuit with no witness column has none.

The quotient is computed pointwise on a coset of a domain of size at least
`d n`, where `d` bounds the total degree of a gated constraint (fixed
columns included) and of the permutation term (`W + 1`), so a gate of any
degree costs one NTT per column. Vanilla PlonK is the instance `W = 3`,
`X = 5`, one arithmetic gadget (`vanilla`), which circuits of
`arrabbiata_frontend::vanilla` compile to.

The tests build Poseidon (one row per round, against the Zcash and the
reference vectors) and elliptic-curve scalar multiplication on Vesta and
Grumpkin (against the native curves) as gate sets, and prove them with IPA
and KZG.

It is a toy: no blinding (not zero-knowledge), no linearization, the
quotient committed in one piece.

The openings are batched. After the evaluations the verifier sends `nu`,
and the prover makes one proof per distinct point (`opening_groups`). The
protocol asks of its scheme only `BatchOpening`, an argument for `k`
evaluation claims at one point, and every scheme has one: `Unbatched<PC>`
proves the claims one by one. A scheme whose commitment is linear
(`LinearCommitment`: IPA, KZG) reduces the `k` claims to one, the
combination with powers of `nu`, and opens that. A wrong value survives
only for the at most `k - 1` values of `nu` that are roots of a nonzero
polynomial (Schwartz-Zippel). That is the same random linear combination
as folding. Linearity is therefore a property of the scheme, not of the
SNARK: a scheme with Merkle commitments, such as FRI, batches by combining
codewords where the verifier queries them, and fits the same trait with a
proof of its own.

Compiled with an accumulation instead (`accumulating`, a separate
implementation), generalized PlonK is a reduction `Satisfiability ->
Accumulated`: the same IOP rounds (shared as `Rounds<.., Tail>`, generic
over the rest of the transcript, so the batched protocol's transcript is
unchanged), then every polynomial at every point it reads (`zeta`, and
`zeta w^r` for each rotation, `rotations`), handed to the scheme's
`Succinct`. The verifier checks the identity on the sparse view of those
values and outputs the accumulator without deciding it, so
`Then(Accumulating, Decide)` is the SNARK, and `Then(Each(Accumulating),
Fold)` turns many proofs into one accumulator, decided once. It is tested
on IPA (three oracles) and KZG (two), with rotations `PREV` and `NEXT` and
both lookup arguments.

### `arrabbiata-permutation`

The copy-constraint argument of PlonK (ePrint 2019/953), written once and
used by both PlonK and generalized PlonK. A `Permutation` acts on the cells
of `k` chosen columns; `from_copies` cycles through each class of the
equivalence relation the copies generate, so it depends on the partition
only, and an assignment respects it exactly when it is constant on the
classes. The cells are labelled `k_c w^i` with shifts in distinct cosets of
the domain; the grand product `z` multiplies `(w + beta id + gamma)` over
`(w + beta sigma + gamma)` row by row, and the two identities
`z prod (w + beta k x + gamma) = z(w x) prod (w + beta sigma + gamma)` and
`L_0 (z - 1) = 0` hold on `H` exactly for respected columns (except with
probability about `k n / |F|`). The product step has degree `k + 1`, so
permuting only the columns that carry copies shrinks the quotient: Kimchi
permutes 7 of its 15 columns, and generalized PlonK takes the set of
permuted columns (`Circuit::with_permutation`); the decider leaves its error
column out. Our framing: the cyclic group `<sigma>` acts on the cells with
the classes as orbits, and the grand product compares two multisets of
labelled values, so it is a multiset-equality argument, the same tool as
plookup's.

The laws are property-tested: the partition determines `sigma`; respected
iff constant on classes; the product closes iff respected (sampled); the
identities vanish on every row exactly then; the labels of distinct cells
are distinct. PlonK's transcript is unchanged by the extraction (its
regression vectors pass).

### `arrabbiata-reduction`

Reductions of knowledge (Kothapalli and Parno, *Algebraic reductions of
knowledge*, CRYPTO 2023) as types. A `Relation` is a check on (instance,
witness) pairs; a `Reduction<F>` from `Source` to `Target` is a
non-interactive, publicly reducible protocol on a transcript over `F`: the
prover maps a pair to a pair and a proof, the verifier maps the instance and
the proof to the output instance.

Relations and reductions form a symmetric monoidal category (Kothapalli,
PhD thesis, Construction 8.1 and Theorem 8.1), and the crate is that
category: `Identity`, sequential composition `Then` (their Theorem 1),
products of relations `Product` and of reductions `Parallel` (Theorem 2),
the unit `Unit`, and `Top = {(true, bottom)}`. An argument of knowledge is a
reduction to `Top` (Definition 3); `Argument` is the trait of those. That
`Then` only composes a reduction whose target is the next one's source is a
bound, pinned by a `compile_fail` doctest. `laws::complete` checks the two
executable laws of a reduction on a pair (the verifier's instance is the
prover's, the output is in the target), and the category laws (units,
associativity) are tested on the folding scheme of linear relations.

The reason for the crate is the shape of accumulation. Our framing: an
accumulation scheme for `R` is a relaxation `R'` with three morphisms,

```text
Embed : R -> R'     Fold : R' x R' -> R'     Dec : R' -> R_top
```

and everything else is composition: accumulating a step is
`Fold . (Embed x Id) : R x R' -> R'`, and `Dec . Embed` is a SNARK for `R`.
By Yoneda (thesis, Theorem 8.2), a reduction `R -> R'` is the same as a
way to turn proofs for `R'` into proofs for `R`, which is why a decider
needs to know `R'` only. Another accumulation scheme is another choice of
`R'` and of the three morphisms, in its own crate.

The finite powers `R^*` of a relation are `Batch` (lists of pairs), with
`Single : R -> R^*` and `Concat : R^* x R^* -> R^*` as unit and
multiplication (the free monoid), and `Each(A) : R1^* -> R2^*` running a
reduction on every element. `Each` is a functor (it preserves identities
and `Then`) and `Concat` a monoid; both are tested.

### `arrabbiata-halo`

Halo's accumulation of polynomial openings (Bowe, Grigg, Hopwood, ePrint
2019/1021; Bunz, Chiesa, Mishra and Spooner, ePrint 2020/499, who give
accumulation schemes for `PC_DL`, the IPA, and `PC_AGM`, KZG), for any
scheme with an `Accumulation`, built only from morphisms of
`arrabbiata-reduction` and over any oracle (`Transcript<F>`) and any
family of polynomials.

The trait asks for four things, and nothing else (no linearity, no
particular family, no particular oracle): a succinct verifier of a batch of
evaluation claims that outputs an accumulator, a decider for the
accumulator relation, a fold `R_acc^* -> R_acc` with a succinct verifier,
and soundness of the three. Three relations (`Evaluations`, `Opened` for
one polynomial at several points, `Accumulated`), four morphisms, and
composites that are the same for every scheme:

```text
Succinct : Evaluations   -> Accumulated     the scheme's opening, succinct check
Fold     : Accumulated^* -> Accumulated     the scheme's fold
Decide   : Accumulated   -> Top             the scheme's decider, run once
Combine  : Evaluations   -> Opened          linear schemes only (polyscale)

Open       = Succinct ; Decide
Accumulate = Each(Succinct) ; Fold
DecideAll  = Fold ; Decide
```

The two schemes meet the trait through different properties, which is the
checklist for another scheme (a lattice or hash-based one):

| | IPA (`PC_DL`) | KZG (`PC_AGM`) |
| --- | --- | --- |
| batch of `k` polynomials | `Combine` (linear commitment) | `Combine` (linear commitment) |
| `m` points | one IPA, `evalscale` | one proof per point, pairs combined by `r` |
| accumulator | `(u, sg)`, claims `sg = commit(b_u)` | `(P, W)`, claims `e(P, H) = e(W, [tau] H)` |
| decider | MSM of the size of the key | one pairing check |
| fold | `Lift ; Succinct`: `b_u(z)` costs `O(k)`, one more IPA | random combination, no proof: the accepted pairs are a subgroup |

`Combine` needs commitments that combine with arbitrary field scalars;
lattice commitments combine only with small-norm scalars and hash-based
ones not at all, so it is a property some schemes have and the trait does
not require it. The laws are generic functions run on IPA (with the
Poseidon duplex sponge and with Kimchi's sponge) and on KZG over BN254:
every morphism and composite is complete, `Accumulate` is `Each(Succinct)`
then `Fold`, a wrong value is rejected, and a false accumulator, which no
succinct check sees, makes the fold fail or its output false. Kimchi
merges previous accumulators into the next proof's opening at `zeta, zeta
w` instead of a fresh point; that saves `Fold`'s IPA and is the protocol's
to make sound (the accumulators must be absorbed before `zeta`), so it
belongs to the Kimchi SNARK, not here.

### `arrabbiata-plonkish`

PlonKish circuits, moved out of the PlonK crate so that the SNARK, the
accumulation scheme and the decider share one arithmetization: gadgets,
gate sets as type-level lists, one-hot selectors, rotations, copies and
public inputs (described under `arrabbiata-plonk` below), the builder, the
vanilla gate, and a circuit's encoding for a transcript (`Absorb`).

Its new part is the relaxation (`relaxed`). For a degree `D` at least the
gate set's (in the witness; fixed columns are coefficients), a relaxed
witness `(w, E)` satisfies a circuit for a slack `u` and coefficients `c`
when, on every row turning on gadget `g`,
`sum_k c_k f^D_(g,k)(u, w) = E_i` (and `E_i = 0` on other rows), copied
cells are equal and column 0 holds the public inputs. The gadgets share the
coefficients because the selectors are one-hot: `c` needs as many entries
as the largest gadget has constraints, not their sum.

Our framing of what `R'` is: homogenizing `f` of degree at most `D` in `N`
variables to `f^D(u, x) = u^D f(x / u)` is an injection into the
homogeneous polynomials of degree `D` in `N + 1` variables, inverted by
setting `u = 1`. The zero set of `f^D` is the affine cone over the
projective closure of the zero set of `f`, which is recovered as the chart
`u = 1`. The relaxation replaces `f^D = 0` by `f^D = E`, the graph of the
defect, which is closed under the line through two of its points; the
coefficients `c` add one to the degree. The tests check the chart (a
witness satisfies the relaxation at `u = 1, E = 0` for every `c`),
homogeneity (scaling `(u, w, x)` by `t` and `E` by `t^D`), and that random
coefficients detect an unsatisfied witness except at the root.

### `arrabbiata-plonkish-folding`

The folding scheme of the Arrabbiata note on PlonKish circuits, as the
three morphisms above minus the decider.

- `Satisfiability` is `R`: a witness satisfies the circuit, for the
  instance's public inputs.
- `Relaxed` is `R'`: the instance `(u, c, x, [w], [E])` is opened by a
  relaxed witness satisfying the relaxation of degree `D`.
- `Embed : R -> R'` commits to the witness columns, draws the combiner `a`
  after them, and outputs `(1, (1, a, a^2, ...), x, [w], [0])`. It is
  complete because the chart holds for every `c`, and knowledge-sound
  because `a` is drawn after the commitments.
- `Fold : R' x R' -> R'` commits to the `D` cross-terms, draws `r`, and
  folds every component linearly. The cross-terms are computed per row from
  the constraints in normal form (`arrabbiata-mvpoly`), only for the gadget
  the row turns on: one-hot selectors make the other gadgets' terms zero
  (the branched-gate observation of Protostar, section 3.6.1).
- `accumulate` is `Fold . (Embed x Id)`, built from the category.

A column is committed as the polynomial interpolating it on the domain of
the circuit (a Lagrange-basis commitment): linear in the column, so the
fold works on commitments, and equal to the commitment PlonK makes, so the
decider opens it. Copy constraints and public inputs are linear, so they
hold on the whole line through two instances: they need no error term, and
the decider checks them once. The transcripts absorb a digest of the
circuit, squeezed once at setup, not the circuit itself, so the verifiers
do `W + D + 2` scalar multiplications and no polynomial evaluation.

The tests run chains of up to four steps on a vanilla circuit and on a
custom gate set of degree 5 reading the previous and the next row, a fixed
column, with a gadget of two constraints. The checks are completeness
(`laws::complete` at every step), the chart, the folding of two
accumulators, and soundness probes: an unsatisfied step, a corrupted
accumulator, a corrupted cross-term, swapped instances, and another
circuit's digest each leave `R'`. The completeness tests check `mvpoly`'s
cross-terms against the relaxation computed independently, by graded
evaluation of the constraints.

### `arrabbiata-generalized-plonk`

Generalized PlonK: any number of witness columns, fixed columns,
selectors and public inputs (all chosen at runtime), and gates that are any
multivariate polynomials of degree at most `D` in `N` variables
(`arrabbiata-mvpoly`). A variable names a witness or fixed column at a
rotation of the constrained row, or a public input; a gate is turned on by
a selector, or holds on every row. Public inputs are scalars the gates read,
and some are also bound to cells of witness column 0. Selectors are columns
with any values: one-hot selectors are circuits, not a requirement.

The protocol is PlonK's polynomial IOP with this gate identity, on the
typed transcript, generic over the commitment and the sponge. Each variable
becomes, on the coset where the quotient is computed, the rotated column
(or the scalar input); the quotient is sized by the largest number of
column variables in a monomial, so a gate's degree in its public inputs
costs nothing. A proof opens every column at `zeta`, and each column a gate
reads at a rotation `k` at `zeta w^k`.

PlonKish circuits embed into it (`plonkish::from_plonkish`): the variables
are the columns the gate set reads at their rotations, the gates the
gadgets' constraints under their selectors, the public inputs those of the
public rows. Our framing: this is a faithful embedding of relations, tested
in both directions (satisfaction is preserved and reflected) on random
vanilla programs and on gate sets reading other rows and fixed columns, and
the two protocols accept the same statements.

**Lookups.** A lookup asks that on every row its selector turns on, the
tuple of its input polynomials (in the gates' variables) is a row of a table
of fixed columns. Tuples are compressed by a challenge `theta` drawn after
the witness, and a row with the selector off queries the table's first row,
so the query column is defined everywhere. Two arguments prove it:

- *plookup* (Gabizon and Williamson, ePrint 2020/315): the sorted merge of
  queries and table in two overlapping halves `h_1, h_2`, and a grand
  product `Z` whose telescoping checks that the consecutive pairs of the
  merge are those of the table plus repeats. Our adaptation to a domain of
  `n` rows reserves the last one (the paper uses a domain of `n + 1` for
  `n` queries), so a plookup is gated and off on the last row.
- *LogUp* (Haboeck, ePrint 2022/1530): multiplicities `m` and a running sum
  `phi` of `1 / (gamma - f) - m / (gamma - t)` that must close on the
  cycle, which is the fractional identity
  `sum_i 1 / (gamma - f_i) = sum_j m_j / (gamma - t_j)`; it needs the
  characteristic to exceed the number of queries, and uses every row.

Our framing of the difference: plookup reduces a subset claim to a
*multiset equality* of consecutive pairs (a grand product, the same tool
as the permutation argument), LogUp to an identity of *rational functions*
(sums of poles), which is additive rather than multiplicative and so needs
no sorting; that additivity is what makes LogUp the one that folds well.
Both are prover-side work in `O(n log n)` (a sort or a count, then NTTs)
and one more round of commitments; the quotient grows to accommodate the
query's degree plus two.

The tests cover range checks, vector lookups into a XOR table (where every
coordinate of a wrong tuple is in its column but the tuple is not a row,
which only the compression by `theta` catches), queries that are
polynomials of degree 2, rows with the selector off, repeated queries and
tables with repeated rows, every lookup opening, misplaced lookup
commitments, the rejection of malformed lookups, and KZG.

The tests check completeness with IPA and KZG, that a statement with
another public input is rejected, that every evaluation and every opening of
a proof is checked, gates reading public inputs at every degree up to the
bound, ungated and overlapping gates, and the rejection of malformed
circuits.

### `arrabbiata-plonk-decider`

The decider `Dec : R' -> R_top` of `plonkish-folding`. It writes the
relaxation of degree `D` of a circuit `C` as a generalized circuit
(`relaxation`) and proves it with generalized PlonK:

- witness columns: those of `C`, then the error column `E`;
- selectors: those of `C`, then `idle = 1 - sum_g q_g`;
- public inputs: those of `C` (bound to their rows), then `u` and the
  coefficients `c`, which the gates read as scalars;
- gates: `sum_k c_k f^D_(g,k)(u, w) - E` under `q_g`, and `E` under `idle`;
- copies: those of `C`, the identity on `E`.

The homogenization `f^D` is computed syntactically on the constraint, each
subexpression homogeneous in `(u, w)` and the lower one padded by powers of
`u`; fixed columns stay variables of degree 0 in the folding. Because the
selectors of `C` are one-hot, the selectors of the relaxation form a
partition of unity, which is what makes the error vanish on the rows that
turn no gadget on; a test checks that the generalized circuit and the
relaxation of `arrabbiata-plonkish` agree on accumulators and on every
single change of the error. The circuit does not depend on an accumulator
(`u` and `c` are inputs), so it is keyed once.

The link to the accumulator: the verifier checks that the proof's witness
commitments are the accumulator's columns and error, which holds because
the scheme commits to a column as the polynomial interpolating it and a
scheme's two `commit` agree on the same coefficients (`Shared`), a law
tested for IPA and KZG. The SNARK `Dec . Embed` (`snark`) is obtained by
composition. The tests decide accumulators of up to four steps (vanilla,
the custom gate set, KZG on BN254), reject every perturbed instance (slack,
coefficient, public input, column or error commitment), a changed
evaluation of the error, and another accumulator's proof, compose the
decider after accumulation, and check that the honest prover refuses an
accumulator outside `R'`.

### `arrabbiata-gadget` and `arrabbiata-kimchi-gadgets`

A gadget (`TypedGadget`) has an input and an output shape and one
`synthesize`, generic in the environment `Env`. The environment is a
signature: a ring of variables and the operations `constant`, `fixed`,
`define`, `advice`, `copy`, `place`, `public`, `assert_zero`, `call`. A
generic `synthesize` cannot inspect a variable, so it is one term of that
signature, and the builder interprets it in two models at once: a variable
is a `Wire`, the pair of a polynomial in the cells and its value. Ring
operations act on both, and evaluation is a ring homomorphism, so the
witness is the constraints evaluated: `evaluate` is the gadget's function
and there is no second implementation of it (our framing: a tagless-final
term, read in the product of two models).

The API is shaped against the usual errors of circuit writing:

- every write says how its cell is constrained: `define` (to a value),
  `copy` (to another cell, by the permutation), `place` (whichever applies,
  and nothing when the value is already there), `advice` (not at all, until
  an `assert_zero`); the under-constrained cell is the one `advice` makes,
  and it is named;
- cells are relative to the gadget, gadgets are laid out row after row by
  `call`, and a cell written twice is an error;
- the instances of a gadget share their gates and one selector: a gate
  kind is the gadget and the row of the constraint within it, its
  constraints are those of the first instance, and every other instance
  must produce the same ones (`Error::Inconsistent` otherwise); what
  varies goes in fixed columns (`fixed`), as Kimchi's coefficients do;
- a constraint the witness breaks is reported while the witness is
  computed, with the gadget and the row;
- shapes compose by type: `Chain` needs the second gadget's input shape to
  be the first's output shape, for every variable type at once, and
  `Repeat` iterates a gadget whose shapes agree.

`circuit` builds the circuit without a witness, `witness` with one, and a
test checks that they agree: the circuit does not depend on the witness.
The other laws tested: the witness satisfies the circuit, `evaluate` of a
chain is the composite, changing a written cell or breaking a copy is
caught, and each misuse is rejected with its error.

`arrabbiata-kimchi-gadgets` lays Kimchi's gates on Kimchi's table (15
witness columns, the first 7 permuted, 15 coefficient columns). The
Poseidon gate is five rounds on a row in Kimchi's `STATE_ORDER`, its 15
constraints `x -> MDS x^7 + c` of degree 7 and its round constants in the
coefficients; 11 rows are the permutation, under one selector. The tests
check Kimchi's permutation, the witness layout row by row, the two-input
vector of proof-systems' `kimchi.json`, and a proof of knowledge of a
preimage.

The Generic gate is Kimchi's two half-gates on columns 0-2 and 3-5,
`c0 w0 + c1 w1 + c2 w2 + c3 w0 w1 + c4` and the same on the next five
coefficients (`GenericRow`, degree 3). `GenericSpec` mirrors Kimchi's
`GenericGateSpec` (`Add`, `Mul`, `Const`, `Plus`) and sets the coefficients
as `create_generic_gadget` does; `Generic` and `GenericPair` compute one or
two operations, the output in the half's third cell, pinned by the
constraint. Inputs are wired by copies, as in Kimchi. A public input is a
binding of the compiled circuit (`PublicInput`), which Kimchi presents as its
`Pub` row, a Generic row of coefficient 1. The tests
rebuild Kimchi's own test circuit (`create_circuit`, `fill_in_witness`)
cell for cell, check that the operations solve their half-gates, and prove
a root of `x^3 + x + 5 = 35`.

The CompleteAdd gate adds two affine points of `y^2 = x^3 + b` on one row
(`x1, y1, x2, y2, x3, y3, inf, same_x, s, inf_z, x21_inv` in columns 0-10)
with Kimchi's seven constraints of degree at most 3: `same_x` and the
inverse of `x2 - x1`, the slope of the tangent or of the chord, the sum,
and `inf = same_x and y1 != y2`. The witness is computed as Kimchi's test
(`tests/ec.rs`) writes it, including the doubling of `P` in the sum cells of
`P + (-P)`. The tests reproduce Kimchi's three kinds of rows cell for cell,
check that every determined cell is caught when changed and that the free
cells are exactly Kimchi's (`x21_inv` when `x1 = x2`, `inf_z` when
`y1 = y2`), and prove that four private points sum to a public point.

The VarBaseMul gate is five steps `A -> 2 A + (2 b - 1) T` of a
variable-base scalar multiplication on two rows, with Kimchi's 21
constraints (the decomposition `n' = 32 n + bits`, then per bit booleanity,
the slope, and the output's two coordinates, of degree up to 6). The tests
reproduce Kimchi's `tests/varbasemul.rs` (255 bits from `2 G` give
`(1 + 2^255 + 2 x) G` and `n = x`, every cell of the 102 rows equal to
Kimchi's witness), check that a gate determines all its cells, and prove a
multiplication with a public base and result.

A constraint lives on its gadget's first row and reads later rows by
rotation, as a Kimchi gate does; so each leaf gadget is one selector.

The EndoMul gate is four bits per row of an endomorphism-optimized scalar
multiplication: each pair of bits picks one of `T`, `-T`, `phi(T)`,
`-phi(T)` with Kimchi's `endo`, the curve's GLV `beta`, and the row applies
`P -> (P + Q) + P` twice, with Kimchi's 12 constraints of degree up to 4.
The tests reproduce Kimchi's `tests/endomul.rs` (128 bits from
`2 (phi(G) + G)` give `[k] G` for `k = ScalarChallenge::to_field(x)` with the
GLV `lambda`, and `n = x`, every cell equal to Kimchi's `gen_witness`), check
that a row determines its cells, and prove `Q = [k] T` with `T` and `Q`
public.

The EndoMulScalar gate computes, in the other field, the scalar EndoMul
multiplies by: eight two-bit crumbs per row update `(n, a, b)` through the
cubics `c` and `d` that interpolate Kimchi's tables on `{0, 1, 2, 3}`, with
`a lambda + b = ScalarChallenge::to_field(x)`; 11 constraints, each crumb
range-checked by `x (x - 1) (x - 2) (x - 3)`. The tests reproduce Kimchi's
`tests/endomul_scalar.rs` cell for cell, check that over `Fq` with Pallas'
`lambda` the scalar is EndoMul's, that a row determines its cells and a
crumb of 4 is rejected, and prove that a public `(a, b)` comes from a
private challenge.

Tables and reads: a `Rom` is a read-only table stored in fixed columns
(on every row, the first row repeated below the last), and `env.read`
asserts that a tuple is one of its rows. A read lives on its gadget's first
row and shares the gadget's selector, like a constraint, and is checked for
consistency across instances the same way; it compiles to a plookup lookup
of generalized PlonK. `env.cell` reads a cell another gadget writes later
(Kimchi's RangeCheck0 reads the next row); a constraint or a read over a
value not yet known is checked once the witness is complete.

The RangeCheck gates check three values of up to 88 bits on four rows
(RangeCheck0, RangeCheck0, RangeCheck1, Zero), each split into 12-bit limbs,
looked up in the table of `0 .. 2^12`, and 2-bit crumbs, constrained by
`x (x - 1) (x - 2) (x - 3)`, with Kimchi's 10 and 21 constraints and its
lookup pattern (columns 3-6, and on the Zero row the copies of the first
two values' top limbs). The tests check Kimchi's `create_multi` layout cell
for cell, that a value of `2^88` or more is rejected, and that a limb out
of range with the recombination kept exact is caught by the lookup alone.
They do not prove: the table makes the circuit 8192 rows, and with one IPA
opening per polynomial that takes hours; batched openings are the next
step that matters for Kimchi.

The Xor16 gate XORs 16 bits per row: the three words in columns 0-2, their
nybbles in 3-14, the words shifted by 16 on the next row, three
recombination constraints of degree 1, and four lookups of
`(in1_k, in2_k, out_k)` in the 4-bit XOR table (Kimchi's order, `(0, 0, 0)`
last). `Xor<BITS>` chains the rows and closes with Kimchi's zero row, a
Generic `w0 = 0` whose three first cells are wired together by
`env.connect` (Kimchi's `connect_cell_pair`). The tests check Kimchi's
`init_xor` layout cell for cell, that a wrong nybble is caught by the
lookup alone and a wider word by the zero row, and prove a XOR with a
public result (512 rows; about 90 seconds with unbatched openings, about
9 with batched ones). RangeCheck is proved too, on the 8192 rows its 12-bit
table needs. The openings are no longer the cost there; the commitments
and the quotient at that size take about six minutes, so that test runs
only with `make test-slow`. Each table has fixed columns of its own
(the range table one, the XOR table three); Kimchi shares them with a table
id, which is the next step for lookups.

Rot64 rotates a 64-bit word by a known offset as Kimchi does: `word 2^rot
= excess 2^64 + shifted` and `rotated = shifted + excess`, with `shifted`
and `excess` range-checked by two RangeCheck0 rows whose two high limbs are
wired to a Generic zero row, and `excess < 2^rot` proved by decomposing
`bound = excess - 2^rot + 2^64` on the Rot64 row itself (its 12-bit limbs
looked up, its crumbs constrained); `2^rot` is coefficient 0, so every
offset shares one selector, and a right rotation is a left one by `64 -
r`. The tests check Kimchi's `extend_rot` layout cell for cell, that the
output is `u64::rotate_left` / `rotate_right` for every offset from 0 to
64, and that the only other split of `word 2^rot` (moving `2^64` from
`shifted` to `excess`, with `rotated` and `bound` adjusted to keep the
equations) is rejected.

ForeignFieldAdd computes `left +/- right = q m + result` modulo a foreign
`m` on three 88-bit limbs, the low two compacted into 176 bits so one carry
suffices (`q (q - s) = 0`, the carry in `{-1, 0, 1}`, two limb equations;
the limbs of `m` and the sign are coefficients 0-3). `ForeignFieldAdd` is
one operation made sound on its own: two Generic rows of constants (`1`,
`2^88`, `0`), the gate, the bound gate (the same gate, right input `(0, 0,
2^88)` and overflow `1`, both copied from the constants, so it computes
`result + 2^264 - m`), Kimchi's Zero row holding the bound, and range checks
of the result and the bound: `bound < 2^264` is `result < m`. Kimchi leaves
the constant copies and the range checks to the circuit designer; here they
are in the gadget. The witness needs 264-bit integers, written as four
88-bit limbs in `u128`s (`foreign_field`), since the repository has no
bignum dependency. The tests check Kimchi's `create_chain` rows cell for
cell for secp256k1's and BN254's base fields and `2^64 - 59`, the result
against independent 64-bit-word arithmetic, and that the bound alone
accepts exactly the `x < m`, the rejection coming from the range check.

ForeignFieldMul checks `a b = q m + r` twice: modulo the native field
(`a_n b_n + q_n f'_n - q_n 2^264 = r_n`, with `f' = 2^264 - m`) and modulo
`2^264` limb by limb, through the intermediate products `p0, p1, p2` and
the carries `v0` (2 bits) and `v1` (91 bits, in seven 12-bit lookups,
three crumbs and a bit); the two together give the equation over the
integers by the Chinese remainder theorem when `m^2 < 2^264 n`.
`ForeignFieldMul` is one multiplication sound on its own: the gate and its
Zero row, then the checks Kimchi leaves to the designer
(`ExternalChecks`): range checks of the quotient and of `(q'2, p10,
p110)`, the compact range check of the remainder (RangeCheck0's compact
mode, now a parameter of the gate), the remainder's high bound `r2 + 2^88 -
m2 - 1` on a Generic `Plus` row, and its range check. The witness divides
a 528-bit product by `m`, with `Nat`, a natural number in 64-bit words
(binary long division). The tests check Kimchi's two rows cell for cell,
the remainder against an independent double-and-add, and `a b + c` through
ForeignFieldAdd.

The differential test against proof-systems (`tests/differential.rs`)
runs on Kimchi's own output, produced by a tool against proof-systems
`9227444890`. For every gate, at four pseudo-random points (all cells of
the row and the next, and all coefficients random, so nowhere near a
satisfying witness), each constraint of our gate kind takes exactly the
value Kimchi's `constraint_checks` computes, in Kimchi's order, with
Kimchi's constants (the `fp_kimchi` MDS matrix, Pallas's endomorphism
coefficient). Two distinct polynomials of degree at most 7 agree at a
random point of a 255-bit field with probability about `7 / 2^255`, so the
constraints are the same polynomials, which the figures (counts and
degrees) do not establish. On fixed inputs, our gadgets also fill the
rows Kimchi's witness generators fill (multi-range check, 64-bit XOR,
rotations left and right, foreign field addition and subtraction,
multiplication), cell for cell. The builder exposes `gates`, each gate
kind's constraints and reads as polynomials to evaluate, for this.

Terms are built with Kimchi's simplifications (constants fold, `0 + e`,
`0 e`, `1 e`), which keep the polynomial and make the syntactic degree
agree with Kimchi's on the same construction; without them a Horner
evaluation from `0` counts one degree too many, which the figures test
caught on EndoMulScalar.
`figures` gives, per gate kind, the constraint count, the degree of each
constraint (Kimchi's `Expr::degree(1, 0)` convention), the instances and
the cells read, and per circuit the rows, variables, copies and public
inputs; a test checks them against the compiled circuit, and
`tests/figures.rs` of `arrabbiata-kimchi-gadgets` against Kimchi's, as
Kimchi computes them: the counts and degrees are the output of running each
gate's `Argument::constraints` and `Expr::degree(1, 0)` on proof-systems
`9227444890`, for all twelve gates with constraints (the ones not yet
written here are the targets).

### `arrabbiata-kimchi`

Kimchi's SNARK as a separate implementation over the shared components,
built milestone by milestone (plan 0006) and checked against
proof-systems' own output at each step. Milestone 1 is Kimchi's two-sponge
Fiat-Shamir (`oracle`): `FqSponge` over the base field (points absorbed as
their affine coordinates, the identity as `(0, 0)`; scalars embedded, or
split into high bits and low bit when the scalar modulus is the larger;
128-bit challenges from the two low limbs of a squeezed element; the
digest read as a scalar), `FrSponge` over the scalar field, and
`ScalarChallenge::to_field`, the map of 128 bits to `a lambda + b` that
EndoMul computes in circuit. Both sponges are generic over the underlying
sponge; on Kimchi's `KimchiSponge` every value equals Kimchi's
`DefaultFqSponge`, `DefaultFrSponge` and `ScalarChallenge` on the same
scripts, for Vesta and for Pallas (where the scalar split is exercised),
and our `Vesta::lambda()` is Kimchi's `endo_r`.

Every circuit of the port, including the circuit each of Kimchi's gate
tests builds, is one typed gadget, run twice: symbolically
(`arrabbiata_gadget::compile`), which gives the constraints, the copy
partition, the lookups and the fixed columns, and numerically on an input
(`assign`), which gives the witness. `compiled` reads Kimchi's rows off the
symbolic run: a row's gate type is the kind of the selector on it, its
coefficients the fixed columns, its wires the copy permutation (each
permuted cell wired to the next of its cycle), and a public input is
Kimchi's Generic row of coefficient 1. Two presentations of one circuit
agree up to the order inside each cycle, which the permutation argument
does not see, so the tests compare rows up to the partition. There are no
Kimchi-shaped constructors or witness functions: the witness is the
gadget's, and a tampered witness is ours, edited.

## deviations and next steps

- **Copy constraints in folding.** `arrabbiata-folding` (the relation of
  `arithmetization/relation`) has no permutation argument, so folding there
  ignores wiring. `arrabbiata-plonkish-folding` folds the copies linearly
  and its decider checks them with PlonK's permutation argument.
- **Circuit digest.** `arrabbiata-folding` absorbs the whole constraint
  system on every fold; `arrabbiata-plonkish-folding` absorbs a digest.
- **The decider is a toy PlonK.** No blinding, and the
  error column is committed in full. Its gates have the fixed columns as
  variables, so their degree grows with the fixed factors of a constraint. The accumulation verifier is
  sublinear, but nothing yet runs it inside a circuit (IVC).
- **The verifier circuit on the cycle.** The fields, curves, Poseidon and the
  commitments of the cycle exist; the in-circuit verifier (non-native
  arithmetic, EC gadgets, Poseidon gadget) does not yet.
- **Performance.** Tate pairing over the bits of `r` (not optimal ate),
  final exponentiation without Frobenius, schoolbook polynomial products in
  folding, single-threaded commitments and quotient (about six minutes
  for an 8192-row PlonK proof).
- **PlonK beyond custom gates.** Lookups.
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
| BN254 add, scalar multiplication, pairing check = EIP-196/197 | yes, the 49 vectors shipped with go-ethereum | no |
| Grumpkin has the order of the BN254 base field | yes, `[r - 1] G = -G` | no |
| rotations act on rows as `Z` on `Z/n` | yes | no |
| Poseidon = reference permutation | yes, Zcash vectors (Pasta), the authors' `hadeshash` vectors (BN254, widths 3 and 5), and proof-systems' `kimchi.json` (Kimchi's instance) | no |
| IPA and KZG completeness, binding, morphism | yes | no |
| IPA at several points: complete, sound; `succinct` trusts only `sg`, `decide` checks it; `decide_all` = `decide` for each | yes | no |
| `b_poly`, `b_poly_coefficients`, `combined_inner_product` = Kimchi's | yes, Kimchi's output | no |
| lists of a relation: `Each` is a functor, `Concat` a monoid | yes | no |
| Halo, for IPA and KZG and two oracles: every morphism and composite complete; a false accumulator caught by the next fold | yes | no |
| graded and homogeneous evaluation | yes | no |
| folding identity | yes, on arbitrary relaxed pairs | yes, `fold_complete` (built in CI) |
| prover and verifier fold to the same instance | yes | by construction (shared function) |
| commitment is a module morphism (Pedersen) | yes | no |
| multilinear extension interpolates its table | yes | no |
| PlonK completeness | yes, random circuits, vanilla and custom gates, KZG and IPA | no |
| folding with Pedersen and KZG commitments | yes | no |
| mvpoly: ring laws, evaluation morphism, sparse = dense | yes | no |
| mvpoly cross-terms = folding prover's cross-terms | yes, on the example relations | no |
| Hasse derivatives: Taylor, `H^(i) H^(j) = C(i+j, i) H^(i+j)` | yes | no |
| PlonK rejects tampered evaluations, statements, transcripts | yes | no |
| PlonK identity rejects a cheating prover (gates, copies, public inputs) | yes, every single-cell violation, with and without rotations, mutation-checked | no |
| copies: `sigma` depends only on the partition; satisfied iff constant on classes | yes | no |
| vanilla compilation preserves and reflects satisfaction | yes | no |
| path circuits ignore padding; cyclic circuits are rotation-invariant | yes | no |
| Poseidon and curve scalar multiplication as gate sets = native | yes | no |
| folding knowledge soundness | sanity tests only | no |
| category of reductions: completeness of `Then` and `Parallel`, unit and associativity laws | yes, on the folding of linear relations | no |
| relaxation: the chart `u = 1, E = 0` for every `c`, homogeneity, detection by random `c` | yes | no |
| embedding and PlonKish fold: complete, verifier = prover, output in `R'` | yes, chains of up to 4 steps, degree 2 and 5, rotations, fixed columns | no |
| PlonKish fold: `mvpoly` cross-terms = relaxation evaluated independently | yes, through completeness | no |
| PlonKish fold soundness probes (bad step, bad accumulator, bad cross-term, swapped instances, digest) | yes | no |
| generalized PlonK: completeness, every opening and evaluation checked, public inputs read by gates | yes, IPA and KZG | no |
| gadgets: the witness satisfies the circuit, the circuit does not depend on the witness, `evaluate` is functorial in `Chain` | yes | no |
| Kimchi Poseidon gate = Kimchi's permutation and layout | yes, `kimchi.json` | no |
| Kimchi Generic gate = Kimchi's coefficients, on Kimchi's test circuit and witness | yes | no |
| Kimchi CompleteAdd = Kimchi's rows; its free cells are exactly Kimchi's | yes | no |
| Kimchi VarBaseMul = Kimchi's 255-bit test, cell for cell; every cell determined | yes | no |
| Kimchi EndoMul = Kimchi's 128-bit test, cell for cell; every cell determined | yes | no |
| Kimchi EndoMulScalar = Kimchi's test, cell for cell; its scalar is EndoMul's | yes | no |
| Kimchi RangeCheck0/1 = Kimchi's layout and lookups; only the lookup catches a large limb | yes | no |
| Kimchi Xor16 = Kimchi's layout; only the lookup catches a wrong nybble | yes | no |
| every Kimchi gate's constraints = Kimchi's polynomials (evaluated at random points) | yes, Kimchi's `constraint_checks` output | no |
| Kimchi's witness generators = our gadgets' witnesses on fixed inputs | yes, Kimchi's output | no |
| gadget reads: in-table values pass, out-of-table rejected on their row, proofs with lookups | yes | no |
| GLV: `phi = [lambda]`, the split recomposes `k`, GLV multiplication = scalar multiplication | yes, on every curve | no |
| gate figures (constraints, degrees, cells read) = Kimchi's | yes, against Kimchi's own output | no |
| lookups (plookup, LogUp): complete; reject a query outside the table, a wrong vector tuple, changed openings, moved commitments | yes, IPA and KZG | no |
| permutation: `sigma` depends on the partition only; respected iff constant on classes; the product closes and the identities vanish iff respected | yes | no |
| permuting a subset of columns: same satisfied witnesses, smaller quotient | yes, on a sum of 4-bit numbers | no |
| PlonKish circuits embed into generalized PlonK, preserving and reflecting satisfaction; both protocols accept the same statements | yes, random vanilla programs and rotated gate sets | no |
| the decider's generalized circuit = the relaxation of `arrabbiata-plonkish` | yes, on accumulators and every change of the error | no |
| decider: complete on accumulators, rejects perturbed instances, evaluations and foreign proofs | yes, IPA and KZG | no |
| `HomomorphicCommitment::commit = PolynomialCommitment::commit` | yes, IPA and KZG | no |
| `Dec . Embed` is an argument for the circuit relation | yes | no |
