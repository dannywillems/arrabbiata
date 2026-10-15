# accumulation

Accumulation schemes: a relaxation `R'` of a relation `R`, an embedding
`R -> R'` and a fold `R' x R' -> R'`, so that one final check, the decider
(see `../decider`), covers every accumulated instance.

- `folding` folds the homogeneous relations of `arithmetization/relation`
  of any degree, with Nova as the degree-2 instance.
- `plonkish` accumulates the PlonKish circuits of
  `arithmetization/plonkish`, with the cross-terms computed from
  multivariate polynomials and the copy constraints folded linearly; its
  decider is `decider/plonk`.

Other accumulation schemes (multi-instance folding, split accumulation,
...) will sit beside them, as other morphisms of the same shape.
