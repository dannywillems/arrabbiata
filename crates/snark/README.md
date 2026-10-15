# SNARKs

Succinct non-interactive arguments, each compiled from a polynomial IOP with a
polynomial commitment scheme on the typed transcript.

- `plonk` is PlonK with custom gates over any number of columns and
  selectors, with vanilla PlonK as one instance.
- `generalized` is generalized PlonK: any numbers of witness columns, fixed
  columns, selectors and public inputs, and gates that are any multivariate
  polynomials of bounded degree, reading public inputs as scalars. The
  circuits of `plonk` embed into it, and so does the relaxation the decider
  of `accumulation/plonkish` proves.
