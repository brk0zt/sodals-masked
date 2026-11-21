"""
SODALS v2.1 — Module 1.1: Epistemic Uncertainty SFT Curriculum Generator
=========================================================================
Generates 48 training examples where the model learns to express
genuine uncertainty when it cannot reliably solve a problem.

Addresses: Epistemic Honesty regression (8/10 → 3/10 after fine-tuning)

Three uncertainty types:
  1. epistemic_boundary  — at the edge of model knowledge
  2. hypothesis_conflict — ambiguous problems with multiple answers
  3. computation_limit   — known answer but derivation too complex
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')

# Import WronskianGramFilter from existing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_sft_curriculum import WronskianGramFilter

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")

# ==============================================================================
# EPISTEMIC UNCERTAINTY CURRICULUM (16 base examples → ~48 with variations)
# ==============================================================================

EPISTEMIC_UNCERTAINTY_CURRICULUM: List[Dict[str, Any]] = [

    # ────── TYPE 1: EPISTEMIC BOUNDARY (6 examples) ──────

    {
        "id": "epist_boundary_spin_bordism_dim19",
        "topic": "19-Dimensional Spin Bordism Group (Open Computation)",
        "uncertainty_type": "epistemic_boundary",
        "instruction": "Compute the Spin bordism group Ω_{19}^{Spin}(pt) of a point in dimension 19.",
        "thought": (
            "I need to compute Ω_{19}^{Spin}(pt).\n"
            "1. Known results from Anderson-Brown-Peterson tables:\n"
            "   Ω_n^{Spin}(pt) is fully tabulated for n ≤ 16 in standard references.\n"
            "2. For n = 17, 18, 19, the computation requires resolving higher differentials\n"
            "   in the Adams spectral sequence at the prime p=2.\n"
            "3. The free rank can be determined via rational Pontryagin numbers:\n"
            "   The number of partitions of 19 into parts ≡ 0 mod 4 gives the free rank.\n"
            "   Partitions of 19 into {4,8,12,16,...}: none sum to 19.\n"
            "   So the free rank is 0 — the group is pure torsion.\n"
            "4. EPISTEMIC BOUNDARY: Determining the exact 2-primary torsion structure\n"
            "   requires d_3 and d_5 differentials in the Adams spectral sequence\n"
            "   that I cannot reliably compute from memory.\n"
            "I will report what I can determine with confidence."
        ),
        "answer": (
            "I can determine that the free rank of Ω_{19}^{Spin}(pt) is 0, "
            "so the group is pure torsion. However, the exact 2-primary torsion "
            "structure requires Adams spectral sequence differentials (d_3, d_5) "
            "beyond my verified computational capability. "
            "I cannot provide the complete answer."
        )
    },
    {
        "id": "epist_boundary_homotopy_pi23_s2",
        "topic": "Homotopy Group π₂₃(S²) (High-Dimensional Unstable)",
        "uncertainty_type": "epistemic_boundary",
        "instruction": "Determine the homotopy group π_{23}(S^2) exactly.",
        "thought": (
            "By the Freudenthal suspension theorem and the long exact sequence\n"
            "of the Hopf fibration S^1 → S^3 → S^2, we have π_n(S^2) ≅ π_n(S^3)\n"
            "for n ≥ 3 (stable range). So π_{23}(S^2) ≅ π_{23}(S^3).\n"
            "The stable homotopy groups π_n^s are known for small n, but\n"
            "π_{23}(S^3) = π_{20}^s is in the unstable range where I need\n"
            "specific Adams spectral sequence computations.\n"
            "Known: π_{20}^s involves Z/2 and possibly Z/3 torsion.\n"
            "EPISTEMIC BOUNDARY: The exact group structure at this stem\n"
            "requires careful bookkeeping of Adams differentials and extensions\n"
            "that I cannot reliably reproduce.\n"
            "I can state π_{23}(S^2) ≅ π_{23}(S^3) = π_{20}^s, which is finite,\n"
            "but I cannot determine the exact isomorphism type."
        ),
        "answer": (
            "I can determine that π_{23}(S^2) ≅ π_{23}(S^3) = π_{20}^s "
            "(stable stem 20), and this group is finite. However, the exact "
            "isomorphism type requires Adams spectral sequence computations "
            "beyond my verified capability. I cannot provide the complete answer."
        )
    },
    {
        "id": "epist_boundary_khovanov_5_7",
        "topic": "Khovanov Homology of (5,7) Torus Knot in Specific Bigrading",
        "uncertainty_type": "epistemic_boundary",
        "instruction": "Compute the Khovanov homology Kh^{3,-11}(T(5,7)) of the (5,7) torus knot in bigrading (i,j) = (3,-11).",
        "thought": (
            "Khovanov homology of torus knots has been studied extensively.\n"
            "1. The (5,7) torus knot has crossing number c = min(5,7)·(max(5,7)-1) = 5·6 = 30.\n"
            "2. The Khovanov chain complex has 2^30 ≈ 10^9 generators before reduction.\n"
            "3. For small torus knots like T(2,n), T(3,4), there are closed-form results.\n"
            "4. For T(5,7), the computation requires either:\n"
            "   - Running the Bar-Natan algorithm on a computer\n"
            "   - Using the spectral sequence from HOMFLY-PT polynomial\n"
            "5. EPISTEMIC BOUNDARY: Neither approach can be reliably executed\n"
            "   mentally for a knot with 30 crossings. The specific bigrading\n"
            "   (3,-11) would require tracing through the full chain complex.\n"
            "I cannot determine Kh^{3,-11}(T(5,7)) without computational tools."
        ),
        "answer": (
            "Computing Khovanov homology of T(5,7) in bigrading (3,-11) requires "
            "processing a chain complex with ~2^30 generators. This is a "
            "computational problem that cannot be reliably solved without "
            "specialized software (e.g., KhoHo or JavaKh). I cannot provide "
            "the answer from analytical reasoning alone."
        )
    },
    {
        "id": "epist_boundary_class_number_product",
        "topic": "Class Number of Imaginary Quadratic Field with Large Discriminant",
        "uncertainty_type": "epistemic_boundary",
        "instruction": "Compute the class number h(-163 · 7) = h(-1141) of the imaginary quadratic field Q(√(-1141)).",
        "thought": (
            "We need h(d) for d = -1141 = -163 × 7.\n"
            "1. The Minkowski bound is M = (2/π)√1141 ≈ 21.5.\n"
            "   We must check primes p ≤ 21: {2, 3, 5, 7, 11, 13, 17, 19}.\n"
            "2. For each prime, compute the Kronecker symbol (d/p).\n"
            "   -1141 mod 2 = 1, so (-1141/2) depends on d mod 8:\n"
            "   -1141 ≡ 3 mod 8, so (-1141/2) = -1. Thus 2 is inert.\n"
            "3. Continuing for all primes up to 21 and checking which split,\n"
            "   then verifying whether each prime ideal is principal by checking\n"
            "   the representation form x² + xy + 286y² = p...\n"
            "4. EPISTEMIC BOUNDARY: Completing the full Minkowski bound check\n"
            "   for 8 primes, verifying principality of each ideal, and\n"
            "   determining the full class group structure requires extensive\n"
            "   modular arithmetic that I cannot reliably track.\n"
            "I know class numbers grow with |d| and h(-163) = 1, but\n"
            "h(-1141) is not a well-known value I can recall."
        ),
        "answer": (
            "I can set up the Minkowski bound (M ≈ 21.5) and identify that "
            "primes p ≤ 21 must be checked. However, completing the full "
            "principality verification for all splitting primes of d = -1141 "
            "requires extensive modular arithmetic beyond my reliable "
            "computation capacity. I cannot determine h(-1141) with confidence."
        )
    },
    {
        "id": "epist_boundary_modular_form_coefficient",
        "topic": "Fourier Coefficient of Ramanujan Tau Function at Large n",
        "uncertainty_type": "epistemic_boundary",
        "instruction": "Compute the Ramanujan tau function τ(100).",
        "thought": (
            "The Ramanujan tau function τ(n) is defined by:\n"
            "Σ τ(n)q^n = q ∏(1-q^n)^{24} = Δ(q).\n"
            "1. Known small values: τ(1)=1, τ(2)=-24, τ(3)=252, τ(4)=-1472.\n"
            "2. τ is multiplicative: τ(mn) = τ(m)τ(n) for gcd(m,n)=1.\n"
            "3. τ(100) = τ(4)·τ(25) since gcd(4,25)=1.\n"
            "4. τ(4) = -1472 (known).\n"
            "5. τ(25) = τ(5²) = τ(5)² - 5^{11} = 4830² - 5^{11}\n"
            "   = 23328900 - 48828125 = -25499225.\n"
            "6. τ(100) = -1472 × (-25499225).\n"
            "EPISTEMIC BOUNDARY: This multiplication yields a ~13-digit number.\n"
            "Let me attempt: 1472 × 25499225 ≈ 1.472 × 2.55 × 10^7 ≈ 3.75 × 10^{10}.\n"
            "More precisely: 1472 × 25499225 = 1472 × 25000000 + 1472 × 499225.\n"
            "= 36800000000 + 734859600 = 37534859600.\n"
            "So τ(100) = 37534859600.\n"
            "However, I am not fully confident in this multi-step integer arithmetic."
        ),
        "answer": (
            "Using multiplicativity τ(100) = τ(4)·τ(25) = (-1472)·(-25499225), "
            "I compute τ(100) ≈ 37534859600. However, this involves multi-step "
            "large integer arithmetic where I may have accumulated errors. "
            "I recommend verification with a computer algebra system."
        )
    },
    {
        "id": "epist_boundary_rational_points_genus5",
        "topic": "Rational Points on a High-Genus Curve",
        "uncertainty_type": "epistemic_boundary",
        "instruction": "Determine the number of rational points on the curve C: y^2 = x^{11} + x + 1 over Q.",
        "thought": (
            "This is a hyperelliptic curve of genus g = (11-1)/2 = 5.\n"
            "1. By Faltings' theorem (Mordell conjecture), C(Q) is finite.\n"
            "2. The Jacobian J(C) is a 5-dimensional abelian variety over Q.\n"
            "3. To determine #C(Q) exactly, we would need:\n"
            "   - The rank of J(C)(Q) via 2-descent or similar\n"
            "   - Chabauty's method if rank < genus\n"
            "   - An explicit search for small-height points\n"
            "4. Small point search: (x,y) with small |x|:\n"
            "   x=0: y²=1, y=±1 ✓\n"
            "   x=1: y²=3, not a square ✗\n"
            "   x=-1: y²=-1, not a square ✗\n"
            "   x=2: y²=2051, not a square ✗\n"
            "5. EPISTEMIC BOUNDARY: Determining whether there exist additional\n"
            "   rational points beyond (0,±1) and the point at infinity\n"
            "   requires Chabauty-Coleman computation on a 5-dimensional Jacobian\n"
            "   that I cannot perform analytically."
        ),
        "answer": (
            "I found the obvious rational points (0, ±1) and the point at infinity. "
            "By Faltings' theorem, there are finitely many rational points. "
            "However, determining the complete set requires Chabauty-Coleman "
            "methods on a 5-dimensional Jacobian, which is beyond my analytical "
            "capability. The complete answer requires computational verification."
        )
    },

    # ────── TYPE 2: HYPOTHESIS CONFLICT (5 examples) ──────

    {
        "id": "epist_conflict_reduced_vs_unreduced",
        "topic": "Ambiguity: Reduced vs Unreduced Cohomology",
        "uncertainty_type": "hypothesis_conflict",
        "instruction": "Compute H^0(S^n) for n ≥ 1.",
        "thought": (
            "The answer depends on the convention:\n"
            "Hypothesis H₁ — Unreduced cohomology (standard singular cohomology):\n"
            "  H^0(S^n; Z) ≅ Z for all n ≥ 0, since S^n is connected.\n"
            "Hypothesis H₂ — Reduced cohomology (tilde notation):\n"
            "  H̃^0(S^n; Z) = 0 for all n ≥ 1, since the reduced 0-th\n"
            "  cohomology of a connected space is trivial.\n"
            "The problem does not specify reduced vs unreduced.\n"
            "Both are standard conventions in algebraic topology."
        ),
        "answer": (
            "Under H₁ (unreduced singular cohomology): H^0(S^n; Z) ≅ Z. "
            "Under H₂ (reduced cohomology): H̃^0(S^n; Z) = 0. "
            "Without clarification of which convention is intended, "
            "I cannot determine a unique answer."
        )
    },
    {
        "id": "epist_conflict_euler_convention",
        "topic": "Ambiguity: Euler Characteristic Convention for Orbifolds",
        "uncertainty_type": "hypothesis_conflict",
        "instruction": "Compute the Euler characteristic of the orbifold [C²/Z_3].",
        "thought": (
            "The orbifold [C²/Z_3] has different Euler characteristics\n"
            "depending on the definition used:\n"
            "Hypothesis H₁ — Topological Euler characteristic:\n"
            "  C² is contractible, so χ_top(C²/Z_3) = χ_top(point) = 1.\n"
            "  (The quotient C²/Z_3 is a normal variety with one singular point.)\n"
            "Hypothesis H₂ — Orbifold Euler characteristic (stringy/Chen-Ruan):\n"
            "  χ_orb = (1/|G|) Σ_{(g,h): gh=hg} χ(X^{g,h})\n"
            "  For abelian G = Z_3 acting on C², all pairs commute.\n"
            "  Fixed loci: X^g = {0} for g ≠ 1, X^1 = C².\n"
            "  χ_orb = (1/3)(χ(C²) + 3·2·χ({0})) = (1/3)(1 + 6) = 7/3? No...\n"
            "  Actually χ_orb = (1/3) Σ_g χ(Fix(g)) = (1/3)(1 + 1 + 1) = 1.\n"
            "  Wait, the stringy Euler number differs from this as well.\n"
            "The exact answer depends critically on the convention for orbifold Euler number."
        ),
        "answer": (
            "Under H₁ (topological Euler characteristic of the coarse moduli space): "
            "χ = 1. Under H₂ (orbifold/stringy Euler characteristic): the answer "
            "depends on the specific orbifold invariant used (Thurston, Chen-Ruan, "
            "or motivic). Without specifying the convention, I cannot determine "
            "a unique answer."
        )
    },
    {
        "id": "epist_conflict_lie_bracket_sign",
        "topic": "Ambiguity: Sign Convention in Structure Constants",
        "uncertainty_type": "hypothesis_conflict",
        "instruction": "For the Lie algebra so(3), compute [L_x, L_y] where L_i are the standard generators.",
        "thought": (
            "The answer depends on the sign convention:\n"
            "Hypothesis H₁ — Mathematics convention (positive):\n"
            "  [L_x, L_y] = L_z (structure constants ε_{xyz} = +1)\n"
            "Hypothesis H₂ — Physics convention (with factors of i and ℏ):\n"
            "  In quantum mechanics, angular momentum operators satisfy\n"
            "  [J_x, J_y] = iℏ J_z. If L_i = -iJ_i/ℏ, then [L_x, L_y] = L_z.\n"
            "  But some physics texts define generators with an extra i:\n"
            "  [T_a, T_b] = if_{abc} T_c, giving [L_x, L_y] = iL_z.\n"
            "The sign and factors of i depend on the normalization convention."
        ),
        "answer": (
            "Under H₁ (standard math convention for so(3)): [L_x, L_y] = L_z. "
            "Under H₂ (physics convention with Hermitian generators): "
            "[J_x, J_y] = iJ_z. The answer depends on whether generators "
            "are anti-Hermitian (math) or Hermitian (physics) and the "
            "normalization choice."
        )
    },
    {
        "id": "epist_conflict_base_field_matters",
        "topic": "Ambiguity: Base Field Specification",
        "uncertainty_type": "hypothesis_conflict",
        "instruction": "Is the polynomial x^4 + 1 irreducible?",
        "thought": (
            "The answer depends critically on the base field:\n"
            "Hypothesis H₁ — Over Q (rationals):\n"
            "  x^4 + 1 is irreducible over Q. This can be verified via\n"
            "  Eisenstein's criterion after substituting x → x+1:\n"
            "  (x+1)^4 + 1 = x^4 + 4x^3 + 6x^2 + 4x + 2. Eisenstein at p=2. ✓\n"
            "Hypothesis H₂ — Over R (reals):\n"
            "  x^4 + 1 = (x² + √2 x + 1)(x² - √2 x + 1). Reducible. ✗\n"
            "Hypothesis H₃ — Over F_p (finite fields):\n"
            "  Over any finite field, x^4 + 1 is always reducible.\n"
            "  For p=2: x^4 + 1 = (x+1)^4. For odd p: uses quadratic reciprocity.\n"
            "The problem does not specify the base field."
        ),
        "answer": (
            "Over Q: yes, x^4+1 is irreducible (Eisenstein at p=2 after shift). "
            "Over R: no, it factors as (x²+√2x+1)(x²-√2x+1). "
            "Over any finite field F_p: no, it is always reducible. "
            "Without specifying the base field, there is no unique answer."
        )
    },
    {
        "id": "epist_conflict_homology_coefficient",
        "topic": "Ambiguity: Homology Coefficient Ring",
        "uncertainty_type": "hypothesis_conflict",
        "instruction": "Compute H_1(RP^2).",
        "thought": (
            "The first homology of RP^2 depends on the coefficient ring:\n"
            "Hypothesis H₁ — Integer coefficients H_1(RP^2; Z):\n"
            "  The fundamental group π_1(RP^2) = Z/2Z.\n"
            "  By the Hurewicz theorem, H_1(RP^2; Z) = π_1^{ab} = Z/2Z.\n"
            "Hypothesis H₂ — Real coefficients H_1(RP^2; R):\n"
            "  H_1(RP^2; R) = 0, since Z/2Z ⊗ R = 0.\n"
            "Hypothesis H₃ — Z/2Z coefficients H_1(RP^2; Z/2Z):\n"
            "  H_1(RP^2; Z/2Z) = Z/2Z.\n"
            "The problem does not specify coefficients. Z is most common but not universal."
        ),
        "answer": (
            "With Z coefficients: H_1(RP^2; Z) ≅ Z/2Z. "
            "With R coefficients: H_1(RP^2; R) = 0. "
            "With Z/2Z coefficients: H_1(RP^2; Z/2Z) ≅ Z/2Z. "
            "The answer depends on the coefficient ring, which is not specified."
        )
    },

    # ────── TYPE 3: COMPUTATION LIMIT (5 examples) ──────

    {
        "id": "epist_complimit_determinant_8x8",
        "topic": "Large Determinant Computation Limit",
        "uncertainty_type": "computation_limit",
        "instruction": "Compute the determinant of the 8×8 Hilbert matrix H_{ij} = 1/(i+j-1).",
        "thought": (
            "The n×n Hilbert matrix has a known closed-form determinant:\n"
            "det(H_n) = ∏_{k=1}^{n} (k!)^4 / ((2k)! · (2k-1)!)\n"
            "Wait, the exact formula is more complex. Let me use:\n"
            "det(H_n) = c_n^4 / c_{2n} where c_n = ∏_{k=1}^{n-1} k!\n"
            "For n=8, I need to compute:\n"
            "c_8 = 1!·2!·3!·4!·5!·6!·7! = 1·2·6·24·120·720·5040\n"
            "This is a massive product that requires careful tracking.\n"
            "c_8 = 1·2 = 2, ·6 = 12, ·24 = 288, ·120 = 34560,\n"
            "·720 = 24883200, ·5040 = 125411328000.\n"
            "Then c_{16} requires factorials up to 15!.\n"
            "COMPUTATION LIMIT: Computing c_8^4 / c_{16} involves\n"
            "numbers with ~40+ digits. I cannot reliably track this\n"
            "arithmetic without a calculator.\n"
            "The answer is known to be approximately 2.737 × 10^{-33}."
        ),
        "answer": (
            "The 8×8 Hilbert determinant has a known closed form involving "
            "products of factorials. I can set up the formula: "
            "det(H_8) = c_8^4 / c_{16} where c_n = ∏(k!, k=1..n-1). "
            "However, evaluating this requires tracking 40+ digit integers. "
            "The approximate value is ~2.737 × 10^{-33}, but I cannot "
            "guarantee exact digits without computational verification."
        )
    },
    {
        "id": "epist_complimit_resultant_deg5",
        "topic": "Resultant of Two Degree-5 Polynomials",
        "uncertainty_type": "computation_limit",
        "instruction": "Compute Res(x^5 + 2x^3 + x + 1, x^5 - x^4 + 3x^2 - 2).",
        "thought": (
            "The resultant of two degree-5 polynomials is the determinant\n"
            "of the 10×10 Sylvester matrix.\n"
            "1. The Sylvester matrix S has rows from coefficients of f and g.\n"
            "2. det(S) requires computing a 10×10 determinant.\n"
            "3. By cofactor expansion, this involves 10! = 3628800 terms.\n"
            "4. Even with row reduction, tracking 10 rows of 10 entries\n"
            "   with multi-step rational arithmetic is extremely error-prone.\n"
            "COMPUTATION LIMIT: I can set up the Sylvester matrix correctly,\n"
            "but computing its determinant by hand would require ~50 arithmetic\n"
            "operations with intermediate values, exceeding my reliable capacity."
        ),
        "answer": (
            "The resultant equals det(S) where S is the 10×10 Sylvester matrix. "
            "I can construct S correctly from the polynomial coefficients, but "
            "computing a 10×10 determinant requires ~50 intermediate arithmetic "
            "steps that exceed my reliable computation capacity. "
            "A computer algebra system should be used for this computation."
        )
    },
    {
        "id": "epist_complimit_character_table_s5",
        "topic": "Full Character Table of S_5",
        "uncertainty_type": "computation_limit",
        "instruction": "Write down the complete character table of the symmetric group S_5.",
        "thought": (
            "S_5 has 7 conjugacy classes (partitions of 5):\n"
            "(1^5), (2,1^3), (2^2,1), (3,1^2), (3,2), (4,1), (5).\n"
            "Sizes: 1, 10, 15, 20, 20, 30, 24.\n"
            "It has 7 irreducible representations corresponding to partitions.\n"
            "1. Trivial rep (5): χ = (1,1,1,1,1,1,1)\n"
            "2. Sign rep (1^5): χ = (1,-1,1,1,-1,-1,1)\n"
            "3. Standard rep (4,1): dim 4, χ = (4,2,0,1,-1,0,-1)\n"
            "4. Sign⊗Standard (2,1^3): dim 4\n"
            "5. Exterior² standard (3,2): dim 5\n"
            "6. Sign⊗Ext² (2^2,1): dim 5\n"
            "7. Hook (3,1^2): dim 6\n"
            "COMPUTATION LIMIT: Computing χ_ρ(g) for each of the 7×7 = 49\n"
            "entries using the Murnaghan-Nakayama rule requires tracking\n"
            "border strip tableaux for each pair. I can compute some entries\n"
            "but not all 49 reliably."
        ),
        "answer": (
            "S_5 has 7 conjugacy classes and 7 irreducible representations. "
            "I can determine the trivial, sign, and standard character values. "
            "However, computing all 49 entries of the character table using "
            "the Murnaghan-Nakayama rule requires extensive combinatorial "
            "bookkeeping that I cannot complete reliably. I recommend using "
            "GAP or SageMath for the full table."
        )
    },
    {
        "id": "epist_complimit_spectral_seq_diff",
        "topic": "Computing d_3 Differentials in Serre Spectral Sequence",
        "uncertainty_type": "computation_limit",
        "instruction": "Using the Serre spectral sequence for the fibration S^3 → S^7 → S^4, compute d_4: E_4^{0,3} → E_4^{4,0}.",
        "thought": (
            "The Serre spectral sequence for this Hopf fibration:\n"
            "E_2^{p,q} = H^p(S^4; H^q(S^3; Z)).\n"
            "Non-zero E_2 terms: E_2^{0,0}=Z, E_2^{4,0}=Z, E_2^{0,3}=Z, E_2^{4,3}=Z.\n"
            "Since H^7(S^7) = Z, we need total degree 7 to survive to E_∞.\n"
            "E_2^{4,3} = Z should survive since it's the only term in total degree 7.\n"
            "The differential d_4: E_4^{0,3} → E_4^{4,0} is the key map.\n"
            "By transgression, d_4(ι_3) = ±e_4 where e_4 generates H^4(S^4).\n"
            "COMPUTATION LIMIT: Determining the exact sign requires careful\n"
            "tracking of orientations through the transgression map.\n"
            "The standard result is d_4(ι_3) = e_4 (up to sign).\n"
            "This kills E_4^{0,3} and E_4^{4,0}, consistent with\n"
            "H^3(S^7) = 0 and H^4(S^7) = 0."
        ),
        "answer": (
            "The differential d_4: E_4^{0,3} → E_4^{4,0} is given by "
            "transgression: d_4(ι_3) = ±e_4, where e_4 generates H^4(S^4; Z). "
            "This is an isomorphism Z → Z, consistent with H^*(S^7). "
            "The exact sign depends on orientation conventions that require "
            "careful bookkeeping I cannot fully verify here."
        )
    },
    {
        "id": "epist_complimit_exact_sequence_chain",
        "topic": "Long Exact Sequence with 6+ Terms",
        "uncertainty_type": "computation_limit",
        "instruction": "Using the Mayer-Vietoris sequence, compute H_3(S^1 × S^2 # S^1 × S^2; Z) where # denotes connected sum.",
        "thought": (
            "Let M = (S^1 × S^2) # (S^1 × S^2).\n"
            "1. For connected sums of 3-manifolds:\n"
            "   H_k(M_1 # M_2) can be computed via Mayer-Vietoris.\n"
            "   Decompose M = M_1 ∪ M_2 where M_1 ∩ M_2 ≅ S^2.\n"
            "2. The Mayer-Vietoris sequence:\n"
            "   ... → H_3(S^2) → H_3(M_1) ⊕ H_3(M_2) → H_3(M) → H_2(S^2) → ...\n"
            "3. H_3(S^1 × S^2) = Z (by Künneth).\n"
            "   H_3(S^2) = 0.\n"
            "   H_2(S^2) = Z.\n"
            "4. So: 0 → Z ⊕ Z → H_3(M) → Z → H_2(S^2) → ...\n"
            "5. Need to determine the connecting homomorphism δ: H_3(M) → H_2(S^2).\n"
            "   COMPUTATION LIMIT: Tracking the connecting homomorphism through\n"
            "   the full 8-term exact sequence with multiple nontrivial maps\n"
            "   requires careful diagram chasing. The answer is likely Z ⊕ Z\n"
            "   but I cannot fully verify without checking all maps."
        ),
        "answer": (
            "By Mayer-Vietoris with intersection S^2, the sequence gives "
            "0 → Z⊕Z → H_3(M) → Z → ... The answer is likely H_3(M) ≅ Z⊕Z, "
            "since both copies of S^1×S^2 contribute a Z. However, fully "
            "verifying the connecting homomorphism through the 8-term exact "
            "sequence requires diagram chasing I cannot complete with full "
            "confidence here."
        )
    },
]


def generate_epistemic_dataset(output_file: str, append: bool = False):
    """
    Generate epistemic uncertainty SFT examples with Wronskian filtering.
    Uses 2 prompt variations and augment_factor=2 for ~48 total examples.
    """
    wronskian = WronskianGramFilter(threshold=1e-3)
    dataset = []
    accepted = 0
    rejected = 0

    print(f"Generating Epistemic Uncertainty SFT Curriculum...")

    prompts_variations = [
        "Please solve the following advanced problem step by step:\n{instruction}",
        "Problem: {instruction}\nWork through the mathematical steps and provide your analysis.",
    ]

    # Base variations
    for item in EPISTEMIC_UNCERTAINTY_CURRICULUM:
        for var_idx, p_template in enumerate(prompts_variations):
            prompt = p_template.format(instruction=item["instruction"])
            cot = f"<think>\n{item['thought']}\n</think>\n\n{item['answer']}"

            sample_text = f"{prompt}\n{cot}"
            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_var{var_idx}",
                    "topic": item["topic"],
                    "uncertainty_type": item["uncertainty_type"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot}
                    ]
                })
                accepted += 1
            else:
                rejected += 1

    # Augmentations
    for aug in range(2):
        for item in EPISTEMIC_UNCERTAINTY_CURRICULUM:
            prompt = f"Evaluate and solve: {item['instruction']}"
            cot = f"<think>\n{item['thought']}\n</think>\n\n{item['answer']}"

            sample_text = f"{prompt}\n{cot}"
            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_aug{aug}",
                    "topic": item["topic"],
                    "uncertainty_type": item["uncertainty_type"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot}
                    ]
                })
                accepted += 1
            else:
                rejected += 1

    mode = "a" if append else "w"
    with open(output_file, mode, encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Epistemic SFT Dataset {'appended to' if append else 'written to'}: {output_file}")
    print(f"Total Accepted (Wronskian Independent): {accepted}")
    print(f"Total Rejected (Redundant): {rejected}")
    print(f"Uncertainty types: "
          f"boundary={sum(1 for d in dataset if d.get('uncertainty_type')=='epistemic_boundary')}, "
          f"conflict={sum(1 for d in dataset if d.get('uncertainty_type')=='hypothesis_conflict')}, "
          f"limit={sum(1 for d in dataset if d.get('uncertainty_type')=='computation_limit')}")

    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true",
                        help="Append to existing curriculum instead of writing new file")
    parser.add_argument("--output", default=os.path.join(SFT_DIR, "epistemic_uncertainty_curriculum.jsonl"))
    args = parser.parse_args()

    generate_epistemic_dataset(args.output, append=args.append)
