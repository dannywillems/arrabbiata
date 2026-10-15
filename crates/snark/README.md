# SNARKs

Succinct non-interactive arguments, each compiled from a polynomial IOP with a
polynomial commitment scheme on the typed transcript. `plonk` is PlonK
with custom gates over any number of columns and selectors, vanilla PlonK
being one instance, and relaxed PlonK, the SNARK for the relaxation of a
circuit that the decider of `accumulation/plonkish` runs.
