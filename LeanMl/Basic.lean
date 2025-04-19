import Mathlib



 /-
 Exercises from the Mathematics in Lean Textbook.
 See https://leanprover-community.github.io/mathematics_in_lean/

 Each section corresponds to a "sheet" in the GitHub repo;
 c.f.,
 -/


/-
* * * * *  Section 2: Real Numbers * * * * *
-/


/-
Sheet 1: Real number basics
-/

example : (2 : ℝ) + 2 = 4 := by
  norm_num


example : (2 : ℝ) + 2 ≠ 5 := by
  norm_num

example : (2 : ℝ) + 2 < 5 := by
  norm_num

example : ∃ x : ℝ, 3 * x + 7 = 12 := by
  use 5/3
  norm_num

example : ∃ x : ℝ, 3 * x + 7 ≠ 12 := by
  use 1
  norm_num

example : ∃ x y : ℝ, 2 * x + 3 * y = 7 ∧ x + 2 * y = 4 := by
  use 2, 1
  norm_num


/-
Sheet 2: Algebra on the Reals
-/

example (x y : ℝ) : (x + y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 := by
  ring

example : ∀ a b : ℝ, ∃ x, (a + b) ^ 3 = a ^ 3 + x * a ^ 2 * b + 3 * a * b ^ 2 + b ^ 3 := by
  intro a b
  use 3
  ring

example : ∃ x : ℝ, ∀ y, y + y = x * y := by
  use 2
  intro y
  ring


example : ∀ x : ℝ, ∃ y, x + y = 2 := by
  intro x
  use 2 - x
  ring

example : ∀ x : ℝ, ∃ y, x + y ≠ 2 := by
  intro x
  use -x
  ring
  simp


/-
Sheet 3: Limits of sequences
-/


-- The ℝ sequence n ↦ n^2+3
def f : ℕ → ℝ := fun n ↦ n ^ 2 + 3

/-
Here's the definition of the limit of a sequence.
-/

/-- If `a(n)` is a sequence of reals and `t` is a real, `TendsTo a t`
is the assertion that the limit of `a(n)` as `n → ∞` is `t`. -/
def TendsTo (a : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ ε > 0, ∃ B : ℕ, ∀ n, B ≤ n → |a n - t| < ε

/-
-- If your goal is `TendsTo a t` and you want to replace it with
-- `∀ ε > 0, ∃ B, …` then you can do this with `rw tendsTo_def`. -/
theorem tendsTo_def {a : ℕ → ℝ} {t : ℝ} :
    TendsTo a t ↔ ∀ ε, 0 < ε → ∃ B : ℕ, ∀ n, B ≤ n → |a n - t| < ε := by
  rfl  -- true by definition


/-- The limit of the constant sequence with value 37 is 37. -/
theorem tendsTo_thirtyseven : TendsTo (fun n ↦ 37) 37 :=
  by
  rw [tendsTo_def]
  intro ε hε
  use 100
  intro n hn
  norm_num
  exact hε

/-- The limit of the constant sequence with value `c` is `c`. -/
theorem tendsTo_const (c : ℝ) : TendsTo (fun n ↦ c) c :=
  by
  intro ε hε
  dsimp only
  use 37
  intro n hn
  ring_nf
  norm_num
  exact hε

/-- If `a(n)` tends to `t` then `a(n) + c` tends to `t + c` -/
theorem tendsTo_add_const {a : ℕ → ℝ} {t : ℝ} (c : ℝ) (h : TendsTo a t) :
    TendsTo (fun n => a n + c) (t + c) :=
  by
  rw [tendsTo_def]
  intro ε hε
  rw [tendsTo_def] at h
  specialize h ε
