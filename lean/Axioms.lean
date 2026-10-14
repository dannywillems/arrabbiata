import Arrabbiata

/-! Axiom census of the main theorems. `scripts/check-axioms.sh` fails if any
of them depends on `sorryAx` or on an axiom outside the standard three. -/

#print axioms Arrabbiata.fold_complete
#print axioms Arrabbiata.foldPoly_eval
