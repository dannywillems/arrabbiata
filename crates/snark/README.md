# SNARKs

Succinct non-interactive arguments, each compiled from a polynomial IOP with a
polynomial commitment scheme on the typed transcript.

- `plonk` is PlonK with custom gates over any number of columns and
  selectors, with vanilla PlonK as one instance.
- `generalized` is generalized PlonK: any numbers of witness columns, fixed
  columns, selectors and public inputs, and gates that are any multivariate
  polynomials of bounded degree, reading public inputs as scalars, and
  lookups (plookup and LogUp) of polynomial queries into fixed tables. The
  circuits of `plonk` embed into it, and so does the relaxation the decider
  of `accumulation/plonkish` proves.
- `kimchi` is Kimchi (o1-labs/proof-systems), checked equal to Kimchi's
  proofs and bytes, with its verifier as a reduction to Halo's accumulation
  and factored into the group half and the scalar half Pickles defers
  between.
- `pickles` is Pickles, Mina's recursion over Kimchi, built from that
  factorization: so far the scalar half written once over the field
  signature, natively and as a Kimchi circuit.
