# decider

Deciders: arguments of knowledge for the relaxed relation of an
accumulation scheme, a reduction `R' -> R_top`. A decider depends on `R'`
only, not on how an accumulator was reached, so it lives here rather than
with the scheme. `plonk` is the decider of `accumulation/plonkish`: it
writes the relaxation of a circuit as a generalized PlonK circuit (the
slack and the coefficients are public inputs, the error a witness column)
and proves it with `snark/generalized`.
