/-
Copyright (c) 2026 Danny Willems. All rights reserved.
Released under the MIT license as described in the file LICENSE.
-/
module

public import Mathlib

@[expose] public section

/-!
# Completeness of folding for homogeneous polynomial relations

The folding scheme of `arrabbiata-folding` rests on one algebraic identity.
Let `P` be a multivariate polynomial over a commutative ring `R`, homogeneous
of degree `d > 0`, and let `x`, `y` be two assignments of its variables. Then
`r ↦ P (x + r • y)` is a polynomial in `r` of degree at most `d` whose constant
coefficient is `P x` and whose coefficient of `r ^ d` is `P y`:

  `P (x + r • y) = P x + ∑_{k=1}^{d-1} r ^ k * T k + r ^ d * P y`.

The `T k` are the cross-terms the prover commits to. The identity holds for
every `P`, so folding is complete for every circuit at once: in the Rust code,
the variables of `P` are the homogenizing scalar `u`, the aggregation
coefficients `c`, and the witness cells of a row, and `P` is
`∑ j, c j * f_j^D (u, w)` (`ConstraintSystem::evaluate_row`).

Nova's relaxed R1CS is the case `d = 2`.

## Main definitions

- `Arrabbiata.foldPoly`: the polynomial `r ↦ P (x + r • y)`, in `R[X]`.

## Main statements

- `Arrabbiata.foldPoly_eval`: it evaluates to `P (x + r • y)`.
- `Arrabbiata.foldPoly_coeff_zero`: its constant coefficient is `P x`.
- `Arrabbiata.foldPoly_natDegree_le`, `Arrabbiata.foldPoly_coeff_degree`: for
  `P` homogeneous of degree `d`, it has degree at most `d` and its coefficient
  of `X ^ d` is `P y`.
- `Arrabbiata.fold_complete`: the folding identity.

## References

* Kothapalli, Setty, Tzialla, *Nova: Recursive Zero-Knowledge Arguments from
  Folding Schemes*, CRYPTO 2022 (ePrint 2021/370).
* Bunz, Chen, *ProtoStar: Generic Efficient Accumulation/Folding for Special
  Sound Protocols*, ASIACRYPT 2023 (ePrint 2023/620).
-/

open Polynomial

namespace Arrabbiata

variable {σ R : Type*} [CommRing R]

/-- The line through `x` and `y` in `R[X]`: coordinate `i` is `x i + X * y i`.
Used by `foldPoly`. -/
noncomputable def line (x y : σ → R) (i : σ) : R[X] :=
  C (x i) + X * C (y i)

/-- `r ↦ P (x + r • y)`, as a polynomial in `r`. Its coefficients are the
cross-terms of a fold; models the evaluation of a row in `F[r]` performed by
the folding prover. -/
noncomputable def foldPoly (P : MvPolynomial σ R) (x y : σ → R) : R[X] :=
  MvPolynomial.eval₂ C (line x y) P

/-- The folding polynomial evaluates to the folded point: the prover's
polynomial in `r` agrees with evaluation on the folded witness. -/
theorem foldPoly_eval (P : MvPolynomial σ R) (x y : σ → R) (r : R) :
    (foldPoly P x y).eval r = MvPolynomial.eval (x + r • y) P := by
  simp only [foldPoly, line, MvPolynomial.eval₂_eq, MvPolynomial.eval_eq, eval_finsetSum,
    eval_mul, eval_C, eval_prod, eval_pow, eval_add, eval_X, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul]

/-- The constant coefficient is the accumulator's value: `T 0 = P x`.
Used by `fold_complete`. -/
theorem foldPoly_coeff_zero (P : MvPolynomial σ R) (x y : σ → R) :
    (foldPoly P x y).coeff 0 = MvPolynomial.eval x P := by
  rw [coeff_zero_eq_eval_zero, foldPoly_eval, zero_smul, add_zero]

/-- A coordinate of the line has degree at most one. -/
theorem natDegree_line_le (x y : σ → R) (i : σ) : (line x y i).natDegree ≤ 1 := by
  unfold line
  compute_degree

/-- The coefficient of `X` in a coordinate of the line is `y i`. -/
theorem coeff_line_one (x y : σ → R) (i : σ) : (line x y i).coeff 1 = y i := by
  simp [line]

/-- A product of powers of coordinates of the line has degree at most the sum
of the exponents, and its top coefficient is the corresponding product of
powers of `y`. The monomial-level heart of `foldPoly_coeff_degree`. -/
theorem prod_line_pow (x y : σ → R) (e : σ → ℕ) (s : Finset σ) :
    (∏ i ∈ s, line x y i ^ e i).natDegree ≤ ∑ i ∈ s, e i ∧
      (∏ i ∈ s, line x y i ^ e i).coeff (∑ i ∈ s, e i) = ∏ i ∈ s, y i ^ e i := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    obtain ⟨hdeg, hcoeff⟩ := ih
    have hpow_deg : (line x y a ^ e a).natDegree ≤ e a := by
      calc (line x y a ^ e a).natDegree ≤ e a * (line x y a).natDegree := natDegree_pow_le
        _ ≤ e a * 1 := Nat.mul_le_mul_left _ (natDegree_line_le x y a)
        _ = e a := mul_one _
    have hpow_coeff : (line x y a ^ e a).coeff (e a) = y a ^ e a := by
      have h := coeff_pow_of_natDegree_le (m := e a) (natDegree_line_le x y a)
      rw [mul_one] at h
      rw [h, coeff_line_one]
    rw [Finset.prod_insert ha, Finset.sum_insert ha, Finset.prod_insert ha]
    refine ⟨?_, ?_⟩
    · exact (natDegree_mul_le).trans (Nat.add_le_add hpow_deg hdeg)
    · rw [coeff_mul_add_eq_of_natDegree_le hpow_deg hdeg, hpow_coeff, hcoeff]

/-- For `P` homogeneous of degree `d`, the folding polynomial has degree at
most `d`: there are exactly `d - 1` cross-terms. Used by `fold_complete`. -/
theorem foldPoly_natDegree_le {P : MvPolynomial σ R} {d : ℕ} (hP : P.IsHomogeneous d)
    (x y : σ → R) : (foldPoly P x y).natDegree ≤ d := by
  classical
  rw [foldPoly, MvPolynomial.eval₂_eq]
  refine natDegree_sum_le_of_forall_le _ _ fun m hm => ?_
  refine (natDegree_C_mul_le _ _).trans ?_
  have hdeg : ∑ i ∈ m.support, m i = d := by
    have h := hP (MvPolynomial.mem_support_iff.mp hm)
    simpa [Finsupp.weight_apply, Finsupp.sum] using h
  exact hdeg ▸ (prod_line_pow x y m m.support).1

/-- For `P` homogeneous of degree `d`, the coefficient of `X ^ d` is the
fresh instance's value: `T d = P y`. Used by `fold_complete`. -/
theorem foldPoly_coeff_degree {P : MvPolynomial σ R} {d : ℕ} (hP : P.IsHomogeneous d)
    (x y : σ → R) : (foldPoly P x y).coeff d = MvPolynomial.eval y P := by
  classical
  rw [foldPoly, MvPolynomial.eval₂_eq, finsetSum_coeff, MvPolynomial.eval_eq]
  refine Finset.sum_congr rfl fun m hm => ?_
  have hdeg : ∑ i ∈ m.support, m i = d := by
    have h := hP (MvPolynomial.mem_support_iff.mp hm)
    simpa [Finsupp.weight_apply, Finsupp.sum] using h
  rw [coeff_C_mul, ← hdeg, (prod_line_pow x y m m.support).2]

/-- **Completeness of folding.** For `P` homogeneous of degree `d > 0`,
evaluating at the folded point `x + r • y` gives the accumulator's value, plus
the cross-terms weighted by powers of `r`, plus `r ^ d` times the fresh
value. With `P x = E_1` and `P y = E_2` this is the update
`E = E_1 + sum_k r ^ k T k + r ^ d E_2` computed by `Fold::prove` and
`Fold::verify`. -/
theorem fold_complete {P : MvPolynomial σ R} {d : ℕ} (hP : P.IsHomogeneous d) (hd : 0 < d)
    (x y : σ → R) (r : R) :
    MvPolynomial.eval (x + r • y) P =
      MvPolynomial.eval x P + (∑ k ∈ Finset.Ico 1 d, (foldPoly P x y).coeff k * r ^ k) +
        MvPolynomial.eval y P * r ^ d := by
  rw [← foldPoly_eval, eval_eq_sum_range' (Nat.lt_succ_of_le (foldPoly_natDegree_le hP x y)),
    Finset.sum_range_succ, Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot hd,
    foldPoly_coeff_zero, foldPoly_coeff_degree hP, pow_zero, mul_one]

end Arrabbiata
