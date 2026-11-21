"""
SODALS v2.1 — Module 1.3: Falsification and Rollback CoT Curriculum Generator
=============================================================================
Generates 32 mathematically rigorous falsification and rollback CoT examples
across 4 domains (Topology, Algebra, Analysis, Physics).

CoT Structure:
  1. Initial Heuristic / Naive Attempt
  2. CONFLICT DETECTED: Explicit contradiction or constraint violation
  3. ROLLBACK: Discarding invalid assumptions and resetting state
  4. CORRECTED APPROACH: Systematic first-principles derivation
  5. Clean Final Answer

Addresses:
  - Self-Correction regression (7/10 -> 3/10)
  - Proof Abandonment (jumping to unverified answers)
  - Dirty state mixing / lack of rollback discipline
"""

import json
import os
import sys
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_sft_curriculum import WronskianGramFilter

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")

FALSIFICATION_CURRICULUM: List[Dict[str, Any]] = [
    # ── DOMAIN A: TOPOLOGY & GEOMETRY (8 examples) ──
    {
        "id": "falsif_top_euler_grassmannian_gr25",
        "domain": "Topology & Geometry",
        "instruction": "Compute the Euler characteristic χ(Gr(2, 5)) of the real Grassmannian of 2-planes in R^5.",
        "thought": (
            "Let me consider the real Grassmannian Gr(2, 5).\n"
            "Initial attempt: Apply the standard combinatorial formula for complex Grassmannians or even-dimensional spheres.\n"
            "For complex Gr_C(k, n), χ = C(n, k) = C(5, 2) = 10.\n"
            "Can we assume χ(Gr_R(2, 5)) = C(floor(5/2), floor(2/2)) = C(2, 1) = 2?\n"
            "CONFLICT DETECTED: The formula χ(Gr_R(2k, 2n)) = C(n, k) holds when both dimensions are even. Here n=5 is odd!\n"
            "Furthermore, dim Gr_R(2, 5) = 2 * (5 - 2) = 6, which is even, so Poincare duality does not force χ to vanish, but the even-parity shortcut is invalid.\n"
            "ROLLBACK: Discard the naive parity shortcut and count Schubert cell dimensions over R.\n"
            "CORRECTED APPROACH: Schubert cells in Gr(2, 5) correspond to partitions λ = (λ_1, λ_2) with 3 >= λ_1 >= λ_2 >= 0.\n"
            "The dimension of the cell e_λ is |λ| = λ_1 + λ_2.\n"
            "The cells are:\n"
            "- dim 0: (0,0) [1 cell, even]\n"
            "- dim 1: (1,0) [1 cell, odd]\n"
            "- dim 2: (2,0), (1,1) [2 cells, even]\n"
            "- dim 3: (3,0), (2,1) [2 cells, odd]\n"
            "- dim 4: (3,1), (2,2) [2 cells, even]\n"
            "- dim 5: (3,2) [1 cell, odd]\n"
            "- dim 6: (3,3) is impossible since λ_2 <= 2; max is (3,2) which has dim 5... wait!\n"
            "Max cell is (3,3) in a 2x3 box? Box size is k x (n-k) = 2 x 3. Max partition is (3,3) of size 6.\n"
            "Let's recount cells by dimension:\n"
            "- dim 0: (0,0) -> 1 (even)\n"
            "- dim 1: (1,0) -> 1 (odd)\n"
            "- dim 2: (2,0), (1,1) -> 2 (even)\n"
            "- dim 3: (3,0), (2,1) -> 2 (odd)\n"
            "- dim 4: (3,1), (2,2) -> 2 (even)\n"
            "- dim 5: (3,2) -> 1 (odd)\n"
            "- dim 6: (3,3) -> 1 (even)\n"
            "Total cells = 1+1+2+2+2+1+1 = 10 = C(5,2).\n"
            "For real Grassmannians, χ is given by the number of Schubert cells that are orientable / cycle-generating, which corresponds to partitions where all parts λ_i have the same parity as their bounding row constraints.\n"
            "By the classic result of Ehresmann (1934) and Hiller (1980), for Gr_R(2, 2m+1), χ(Gr_R(2, 2m+1)) = χ(Gr_R(2, 2m)) = C(m, 1) = m. Here m = 2, so χ(Gr_R(2, 5)) = 2.\n"
            "Let's verify via the double cover S^1 x S^1 fixed points: χ = C(floor(5/2), floor(2/2)) = C(2, 1) = 2."
        ),
        "answer": "2"
    },
    {
        "id": "falsif_top_fundamental_group_klein_puncture",
        "domain": "Topology & Geometry",
        "instruction": "Compute the fundamental group π₁(K ∖ {p}) of a punctured Klein bottle K.",
        "thought": (
            "We want to compute π₁(K ∖ {p}).\n"
            "Initial attempt: The unpunctured Klein bottle has presentation π₁(K) = ⟨a, b | a b a b⁻¹ = 1⟩.\n"
            "Removing a point removes the 2-cell relator in the CW decomposition, so maybe π₁(K ∖ {p}) = ⟨a, b | ⟩ = F₂ (free group of rank 2).\n"
            "Wait, let's verify if the standard cell structure has the puncture in the interior of the single 2-cell.\n"
            "In the standard planar polygon for K, we have a square with edges identified as a b a b⁻¹. The boundary is the 1-skeleton consisting of two circles a and b meeting at a vertex.\n"
            "Removing an interior point from the 2-cell retracts the 2-cell onto its boundary.\n"
            "Therefore, K ∖ {p} deformation retracts onto the 1-skeleton X¹ = S¹ ∨ S¹.\n"
            "The fundamental group of S¹ ∨ S¹ is indeed the free group on 2 generators F₂ = ⟨a, b | ∅⟩.\n"
            "CONFLICT CHECK: Did we verify if the puncture is a single point or affects the homology? H₁(K ∖ {p}) = Ab(F₂) = Z ⊕ Z, whereas H₁(K) = Z ⊕ Z/2Z. Removing the point removes the torsion relation 2a = 0 (or abab⁻¹ = 1 -> a² = 1), so H₁ becomes torsion-free rank 2.\n"
            "This is completely consistent with the deformation retraction onto S¹ ∨ S¹."
        ),
        "answer": "F_2"
    },
    {
        "id": "falsif_top_cohomology_ring_rp3",
        "domain": "Topology & Geometry",
        "instruction": "Determine the cup product x ⌣ y of the two non-trivial 1-dimensional cohomology classes in H¹(RP³; Z₂) and H²(RP³; Z₂).",
        "thought": (
            "We analyze H*(RP³; Z₂).\n"
            "Initial attempt: H*(RP³; Z₂) ≅ Z₂[x]/(x⁴) where deg(x) = 1.\n"
            "Then the generator of H² is x².\n"
            "So the cup product of the generator x ∈ H¹(RP³; Z₂) and y = x² ∈ H²(RP³; Z₂) would be x ⌣ y = x · x² = x³ ∈ H³(RP³; Z₂).\n"
            "CONFLICT DETECTED: What if the coefficients were Z instead of Z₂? In H*(RP³; Z), H¹ = 0, H² = Z₂, H³ = Z. In that case, H¹ is trivial so any cup product from H¹ is 0!\n"
            "ROLLBACK: Check the exact question statement. The prompt explicitly specifies Z₂ coefficients: H¹(RP³; Z₂) and H²(RP³; Z₂).\n"
            "CORRECTED APPROACH: Over Z₂, the cohomology ring of real projective space is standardly H*(RPⁿ; Z₂) ≅ Z₂[x]/(xⁿ⁺¹), where |x| = 1.\n"
            "For n = 3: H*(RP³; Z₂) ≅ Z₂[x]/(x⁴).\n"
            "The unique non-zero element in H¹ is x, and the unique non-zero element in H² is x².\n"
            "Their cup product is x ⌣ x² = x³, which is the unique non-zero fundamental class in H³(RP³; Z₂)."
        ),
        "answer": "x^3"
    },
    {
        "id": "falsif_top_stiefel_whitney_tangent_s2xs2",
        "domain": "Topology & Geometry",
        "instruction": "Compute the total Stiefel-Whitney class w(T(S² × S²)).",
        "thought": (
            "We want w(T(S² × S²)).\n"
            "Initial attempt: T(S² × S²) ≅ pr₁*(TS²) ⊕ pr₂*(TS²).\n"
            "Using the Whitney sum formula: w(T(S² × S²)) = w(pr₁*TS²) ⌣ w(pr₂*TS²).\n"
            "For S², w(TS²) = 1 + w₁(TS²) + w₂(TS²).\n"
            "Since S² is orientable, w₁(TS²) = 0.\n"
            "What about w₂(TS²)? w₂(TS²) is the mod 2 reduction of the Euler class e(TS²).\n"
            "e(TS²)[S²] = χ(S²) = 2.\n"
            "Reducing mod 2: 2 ≡ 0 mod 2. So w₂(TS²) = 0!\n"
            "CONFLICT CHECK: If w₁(TS²) = 0 and w₂(TS²) = 0, then w(TS²) = 1.\n"
            "Then w(T(S² × S²)) = (1)(1) = 1.\n"
            "Let's double check if w₂(TS²) is indeed 0: For any orientable 2-manifold M, w₂(TM) reduction of e(TM) mod 2 equals χ(M) mod 2.\n"
            "Since χ(S²) = 2 ≡ 0 (mod 2), w₂(TS²) = 0 ∈ H²(S²; Z₂).\n"
            "Therefore w(TS²) = 1 + 0 + 0 = 1.\n"
            "By the product formula, w(T(S² × S²)) = 1."
        ),
        "answer": "1"
    },
    {
        "id": "falsif_top_homotopy_relative_exact",
        "domain": "Topology & Geometry",
        "instruction": "Compute π₂(S², S¹).",
        "thought": (
            "Consider the pair (S², S¹) where S¹ is the equator of S².\n"
            "Initial attempt: π₂(S², S¹) ≅ π₂(S²/S¹)? S²/S¹ is the wedge S² ∨ S², so π₂(S² ∨ S²) ≅ Z ⊕ Z.\n"
            "CONFLICT DETECTED: For homotopy groups, π_n(X, A) is generally NOT isomorphic to π_n(X/A) unless excision applies in the stable range! Excision fails for homotopy groups in low dimensions.\n"
            "ROLLBACK: Must use the long exact sequence of the pair (S², S¹):\n"
            "... → π₂(S¹) → π₂(S²) → π₂(S², S¹) → π₁(S¹) → π₁(S²) → ...\n"
            "CORRECTED APPROACH:\n"
            "1. π₂(S¹) = 0 (universal cover of S¹ is R, which is contractible, so π_k(S¹) = 0 for k ≥ 2).\n"
            "2. π₂(S²) ≅ Z.\n"
            "3. π₁(S¹) ≅ Z.\n"
            "4. π₁(S²) = 0 (S² is simply connected).\n"
            "Substitute into the exact sequence:\n"
            "0 → Z → π₂(S², S¹) → Z → 0.\n"
            "Since Z is a free abelian group, this short exact sequence of groups splits:\n"
            "0 → Z → π₂(S², S¹) → Z → 0 ⟹ π₂(S², S¹) ≅ Z ⊕ Z."
        ),
        "answer": "Z + Z"
    },
    {
        "id": "falsif_top_mayer_vietoris_klein",
        "domain": "Topology & Geometry",
        "instruction": "Compute H₂(K; Z) for the Klein bottle K using the Mayer-Vietoris sequence.",
        "thought": (
            "We want H₂(K; Z) for the Klein bottle K.\n"
            "Initial attempt: Decompose K into two Mobius bands M₁ and M₂ glued along their boundary circles S¹.\n"
            "H₂(M₁) = 0, H₂(M₂) = 0, H₂(M₁ ∩ M₂) = H₂(S¹) = 0.\n"
            "Mayer-Vietoris gives:\n"
            "0 → H₂(K) → H₁(S¹) --(i_1*, i_2*)--> H₁(M₁) ⊕ H₁(M₂) → H₁(K) → ...\n"
            "H₁(S¹) ≅ Z with generator γ.\n"
            "H₁(M₁) ≅ Z with generator α₁, where the boundary γ wraps twice around the core: i_1*(γ) = 2α₁.\n"
            "Similarly i_2*(γ) = 2α₂.\n"
            "The map is Φ: Z → Z ⊕ Z, 1 ↦ (2, -2) [or (2, 2)].\n"
            "Since Φ(1) = (2, -2) ≠ (0, 0), ker(Φ) = 0.\n"
            "From the exact sequence: 0 → H₂(K) → ker(Φ) = 0 ⟹ H₂(K) = 0.\n"
            "CONFLICT CHECK: Is K an orientable closed surface? No, it is non-orientable and compact without boundary.\n"
            "Top homology of any closed connected non-orientable n-manifold over Z is always 0.\n"
            "Thus H₂(K; Z) = 0."
        ),
        "answer": "0"
    },
    {
        "id": "falsif_top_euler_characteristic_connected_sum",
        "domain": "Topology & Geometry",
        "instruction": "Compute the Euler characteristic χ(T² # T² # RP²).",
        "thought": (
            "We want χ of the connected sum of two 2-tori and one real projective plane.\n"
            "Initial attempt: χ(A # B) = χ(A) + χ(B).\n"
            "χ(T²) = 0, χ(T²) = 0, χ(RP²) = 1, so χ = 0 + 0 + 1 = 1?\n"
            "CONFLICT DETECTED: The formula χ(A # B) = χ(A) + χ(B) is false for surfaces!\n"
            "When taking connected sum of 2-manifolds, we remove two open 2-disks (each having χ = 1) and glue along S¹ (χ(S¹) = 0).\n"
            "By inclusion-exclusion: χ(A # B) = (χ(A) - 1) + (χ(B) - 1) + 0 = χ(A) + χ(B) - 2.\n"
            "ROLLBACK: Discard naive sum and apply the correct connected sum formula for 2-manifolds:\n"
            "χ(M₁ # M₂ # ... # M_k) = Σ χ(M_i) - 2(k - 1).\n"
            "CORRECTED APPROACH:\n"
            "Here k = 3 surfaces: M₁ = T², M₂ = T², M₃ = RP².\n"
            "χ(T²) = 0\n"
            "χ(T²) = 0\n"
            "χ(RP²) = 1\n"
            "χ(T² # T² # RP²) = 0 + 0 + 1 - 2*(3 - 1) = 1 - 4 = -3.\n"
            "Let's verify: T² # T² # RP² is homeomorphic to a connected sum of 2*2 + 1 = 5 copies of RP² (since T² # RP² ≅ RP² # RP² # RP²).\n"
            "For #₅ RP²: χ = 2 - 5 = -3. Perfect match."
        ),
        "answer": "-3"
    },
    {
        "id": "falsif_top_degree_antipodal_map",
        "domain": "Topology & Geometry",
        "instruction": "Compute the degree of the antipodal map a: Sⁿ → Sⁿ, x ↦ -x.",
        "thought": (
            "We want deg(a) for a(x) = -x on Sⁿ ⊂ Rⁿ⁺¹.\n"
            "Initial attempt: The map reflects all coordinates: (x₀, ..., x_n) ↦ (-x₀, ..., -x_n).\n"
            "This is a composition of (n+1) single-coordinate reflections r_i(x_i) = -x_i.\n"
            "Each single reflection reverses orientation of Rⁿ⁺¹, so it has degree -1 on Sⁿ.\n"
            "Therefore deg(a) = (-1)^(n+1).\n"
            "Let's test small n:\n"
            "- n = 0: S⁰ = {-1, 1}. a(1) = -1, a(-1) = 1. The map swaps the two points. In reduced 0-homology, degree is (-1)¹ = -1.\n"
            "- n = 1: S¹ (circle). a is rotation by π. Rotations are homotopic to identity, so degree is +1. Formula gives (-1)² = +1. Correct!\n"
            "- n = 2: S² (sphere). a is reflection through origin. Formula gives (-1)³ = -1. Correct!\n"
            "CONFLICT CHECK: Did we confuse n with the ambient dimension n+1? Ambient dimension is n+1, so there are n+1 reflections. The formula (-1)^(n+1) is completely sound."
        ),
        "answer": "(-1)^(n+1)"
    },

    # ── DOMAIN B: ALGEBRA & NUMBER THEORY (8 examples) ──
    {
        "id": "falsif_alg_sylow_subgroups_order_30",
        "domain": "Algebra & Number Theory",
        "instruction": "Determine the number of Sylow 5-subgroups n₅ in a simple group of order 30, or prove no such group exists.",
        "thought": (
            "Let |G| = 30 = 2 · 3 · 5.\n"
            "Initial attempt: By Sylow's theorem, n₅ ≡ 1 (mod 5) and n₅ | 6. Possible values are n₅ ∈ {1, 6}.\n"
            "If G is simple, n₅ ≠ 1, so n₅ = 6.\n"
            "Each Sylow 5-subgroup has order 5 (prime), so any two distinct Sylow 5-subgroups intersect trivially at {e}.\n"
            "6 Sylow 5-subgroups contain 6 · (5 - 1) = 24 elements of order 5.\n"
            "Now look at Sylow 3-subgroups: n₃ ≡ 1 (mod 3) and n₃ | 10. Possible n₃ ∈ {1, 10}.\n"
            "If G is simple, n₃ = 10. Each Sylow 3-subgroup has order 3, so distinct ones intersect trivially.\n"
            "10 Sylow 3-subgroups contain 10 · (3 - 1) = 20 elements of order 3.\n"
            "CONFLICT DETECTED: Total elements required = 24 (order 5) + 20 (order 3) + 1 (identity) = 45 > 30 = |G|!\n"
            "ROLLBACK: We assumed a simple group of order 30 could have both n₅ = 6 and n₃ = 10.\n"
            "CORRECTED APPROACH: The element count 45 strictly exceeds the group order 30.\n"
            "Therefore, it is impossible for both n₅ = 6 and n₃ = 10 to hold simultaneously.\n"
            "At least one of n₅ or n₃ must be 1, meaning G has a normal Sylow subgroup.\n"
            "Hence, no simple group of order 30 exists."
        ),
        "answer": "0"
    },
    {
        "id": "falsif_alg_galois_group_x4_minus_2",
        "domain": "Algebra & Number Theory",
        "instruction": "Determine the Galois group Gal(Q(∜2, i) / Q).",
        "thought": (
            "Let f(x) = x⁴ - 2 over Q. The roots are ±∜2, ±i∜2.\n"
            "Splitting field is K = Q(∜2, i).\n"
            "Initial attempt: [Q(∜2):Q] = 4 (Eisenstein at 2). Since i ∉ R and Q(∜2) ⊂ R, [K:Q(∜2)] = 2.\n"
            "So [K:Q] = 8.\n"
            "Is the Galois group abelian of order 8, say Z₂ × Z₄?\n"
            "Let's test an automorphism: σ(∜2) = i∜2, σ(i) = i.\n"
            "Then σ²(∜2) = σ(i∜2) = i(i∜2) = -∜2.\n"
            "σ⁴(∜2) = ∜2, so σ has order 4.\n"
            "Let τ be complex conjugation: τ(∜2) = ∜2, τ(i) = -i. τ has order 2.\n"
            "Compute the commutator/relation:\n"
            "στ(∜2) = σ(∜2) = i∜2.\n"
            "τσ(∜2) = τ(i∜2) = -i∜2 = σ³(∜2).\n"
            "CONFLICT DETECTED: στ ≠ τσ, since i∜2 ≠ -i∜2! Thus the group is NOT abelian.\n"
            "ROLLBACK: Abandon the abelian hypothesis.\n"
            "CORRECTED APPROACH: We have σ⁴ = 1, τ² = 1, and τστ⁻¹ = σ⁻¹ = σ³.\n"
            "This is the standard presentation of the dihedral group of order 8: D₄ (or D₈ in some notations, symmetries of a square)."
        ),
        "answer": "D_4"
    },
    {
        "id": "falsif_alg_class_number_gaussian_integers",
        "domain": "Algebra & Number Theory",
        "instruction": "Compute the class number h(Q(i)) of the Gaussian field Q(i).",
        "thought": (
            "We want the class number of K = Q(√-1) with ring of integers Z[i].\n"
            "Initial attempt: Minkowski bound formula for imaginary quadratic fields with discriminant d_K = -4 (since -1 ≡ 3 mod 4, d_K = 4(-1) = -4):\n"
            "M_K = (2/π) √|d_K| = (2/π) · √4 = 4/π ≈ 4/3.14159 ≈ 1.273.\n"
            "The Minkowski bound states that every ideal class contains an integral ideal with norm N(I) ≤ M_K < 2.\n"
            "The only positive integer ≤ 1 is 1.\n"
            "The only ideal with norm 1 is the unit ideal (1) = Z[i], which is principal.\n"
            "Therefore, every ideal is equivalent to the principal ideal, so the class group Cl(Q(i)) is trivial.\n"
            "CONFLICT CHECK: Does Z[i] have a Euclidean algorithm? Yes, the standard norm N(a+bi) = a² + b² is a Euclidean valuation.\n"
            "Euclidean Domain ⟹ Principal Ideal Domain (PID) ⟹ Class Number h = 1.\n"
            "Both Minkowski bound and Euclidean division confirm h = 1."
        ),
        "answer": "1"
    },
    {
        "id": "falsif_alg_legendre_symbol_reciprocity",
        "domain": "Algebra & Number Theory",
        "instruction": "Compute the Legendre symbol (7/29).",
        "thought": (
            "We want to compute the Legendre symbol (7/29).\n"
            "Initial attempt: By Quadratic Reciprocity, (p/q)(q/p) = (-1)^((p-1)(q-1)/4).\n"
            "Here p = 7, q = 29.\n"
            "Compute exponent: (7-1)(29-1)/4 = (6)(28)/4 = 6 · 7 = 42, which is even!\n"
            "So (7/29) = + (29/7).\n"
            "Reduce 29 modulo 7: 29 = 4 · 7 + 1 ≡ 1 mod 7.\n"
            "So (29/7) = (1/7) = +1.\n"
            "Thus (7/29) = +1.\n"
            "CONFLICT CHECK: Let's independently verify if 7 is a quadratic residue modulo 29.\n"
            "Can we find x such that x² ≡ 7 mod 29?\n"
            "Try small x:\n"
            "1² = 1, 2² = 4, 3² = 9, 4² = 16, 5² = 25 ≡ -4,\n"
            "6² = 36 ≡ 7 mod 29!\n"
            "Indeed 6² = 36 = 29 + 7 ≡ 7 mod 29.\n"
            "So x = 6 is an explicit square root of 7 modulo 29. The answer +1 is undeniably correct."
        ),
        "answer": "1"
    },
    {
        "id": "falsif_alg_irreducibility_cyclotomic_p",
        "domain": "Algebra & Number Theory",
        "instruction": "Compute the degree of the minimal polynomial of ζ_p = e^(2πi/p) over Q for a prime p.",
        "thought": (
            "We want the degree of the minimal polynomial of ζ_p over Q.\n"
            "Initial attempt: ζ_p satisfies x^p - 1 = 0, so is the degree p?\n"
            "CONFLICT DETECTED: x^p - 1 factors as (x - 1)(x^(p-1) + x^(p-2) + ... + x + 1).\n"
            "Since ζ_p ≠ 1, ζ_p is a root of the cyclotomic polynomial Φ_p(x) = x^(p-1) + x^(p-2) + ... + x + 1.\n"
            "ROLLBACK: The minimal polynomial is not x^p - 1, it must be Φ_p(x) if Φ_p is irreducible.\n"
            "CORRECTED APPROACH: Prove Φ_p(x) is irreducible over Q using Eisenstein's criterion on Φ_p(x+1):\n"
            "Φ_p(x+1) = ((x+1)^p - 1) / ((x+1) - 1) = (Σ_{k=1}^p C(p, k) x^k) / x = x^(p-1) + C(p, p-1)x^(p-2) + ... + C(p, 2)x + p.\n"
            "All intermediate coefficients C(p, k) for 1 ≤ k ≤ p-1 are divisible by p.\n"
            "The constant term is p, which is divisible by p but NOT by p².\n"
            "The leading coefficient is 1, not divisible by p.\n"
            "By Eisenstein at prime p, Φ_p(x+1) is irreducible over Q, hence Φ_p(x) is irreducible over Q.\n"
            "The degree of Φ_p(x) is p - 1."
        ),
        "answer": "p - 1"
    },
    {
        "id": "falsif_alg_center_heisenberg_group",
        "domain": "Algebra & Number Theory",
        "instruction": "Determine the dimension of the center Z(H₃(R)) of the 3D real Heisenberg Lie group.",
        "thought": (
            "The continuous Heisenberg group H₃(R) consists of upper triangular 3x3 matrices with 1s on the diagonal:\n"
            "M(x, y, z) = [[1, x, z], [0, 1, y], [0, 0, 1]].\n"
            "Group multiplication:\n"
            "M(x, y, z) M(x', y', z') = M(x + x', y + y', z + z' + x y').\n"
            "For M(x, y, z) to be in the center Z, it must commute with all M(x', y', z'):\n"
            "z + z' + x y' = z' + z + x' y ⟹ x y' = x' y for all x', y' ∈ R.\n"
            "Choosing x'=1, y'=0 gives x(0) = (1)y ⟹ y = 0.\n"
            "Choosing x'=0, y'=1 gives x(1) = (0)y ⟹ x = 0.\n"
            "Therefore, x = 0 and y = 0, while z ∈ R is completely free.\n"
            "The center is Z(H₃(R)) = {M(0, 0, z) : z ∈ R} ≅ (R, +).\n"
            "This is a 1-dimensional subgroup parameterized by z.\n"
            "Dimension is 1."
        ),
        "answer": "1"
    },
    {
        "id": "falsif_alg_frobenius_fixed_points",
        "domain": "Algebra & Number Theory",
        "instruction": "Compute the number of elements in F_{p^n} fixed by the Frobenius automorphism Frob_p^d where d | n.",
        "thought": (
            "Let F_q = F_{p^n} be the finite field with p^n elements.\n"
            "The absolute Frobenius is σ(x) = x^p.\n"
            "The automorphism Frob_p^d is σ^d(x) = x^(p^d).\n"
            "An element x ∈ F_{p^n} is fixed by Frob_p^d if and only if x^(p^d) = x, which is equivalent to x^(p^d) - x = 0.\n"
            "The roots of x^(p^d) - x in an algebraic closure of F_p form the subfield F_{p^d}.\n"
            "Since d | n, F_{p^d} is a subfield of F_{p^n}.\n"
            "The fixed elements are precisely the elements of the subfield F_{p^d}.\n"
            "The number of elements in F_{p^d} is p^d."
        ),
        "answer": "p^d"
    },
    {
        "id": "falsif_alg_quaternion_algebra_ramification",
        "domain": "Algebra & Number Theory",
        "instruction": "Determine if the quaternion algebra H = (-1, -1)_Q is ramified at p = 2.",
        "thought": (
            "Consider the quaternion algebra H = (-1, -1)_Q = Q ⊕ Qi ⊕ Qj ⊕ Qk with i² = -1, j² = -1, ij = -ji = k.\n"
            "Initial attempt: The discriminant of H is the product of primes where H is ramified.\n"
            "At p = ∞ (over R): H ⊗_Q R ≅ H_Hamilton, which is a division algebra (not isomorphic to M₂(R)). So H is ramified at ∞.\n"
            "By the Hilbert reciprocity law for quaternion algebras: the number of ramified places (including ∞) must be EVEN.\n"
            "For odd primes p: (-1, -1)_p is the Hilbert symbol. For p ≡ 1 mod 4, -1 is a square in Q_p, so (-1,-1)_p = 1 (unramified).\n"
            "For p ≡ 3 mod 4, the Hilbert symbol (-1,-1)_p = (-1)^((p-1)/2) · (-1)^((p-1)/2) = 1 (unramified).\n"
            "So no odd prime is ramified.\n"
            "Since the total number of ramified places must be even and ∞ is ramified, exactly one finite prime must be ramified: p = 2.\n"
            "Indeed, the Hilbert symbol at 2 is (-1, -1)₂ = (-1)^(((-1)-1)/2 · ((-1)-1)/2) = (-1)^((-1)(-1)) = -1 (ramified).\n"
            "Therefore, H is ramified at p = 2."
        ),
        "answer": "yes"
    },

    # ── DOMAIN C: ANALYSIS & DIFFERENTIAL EQUATIONS (8 examples) ──
    {
        "id": "falsif_ana_radius_convergence_ratio_test",
        "domain": "Analysis & ODEs",
        "instruction": "Find the radius of convergence of the power series Σ_{n=1}^∞ (n! / n^n) x^n.",
        "thought": (
            "Let a_n = n! / n^n. We want the radius of convergence R.\n"
            "Initial attempt: Apply Cauchy-Hadamard formula with Stirling's approximation:\n"
            "n! ≈ √(2πn) (n/e)^n, so a_n ≈ √(2πn) e^(-n).\n"
            "lim (a_n)^(1/n) = lim (2πn)^(1/(2n)) · e^(-1) = 1 · e^(-1) = 1/e.\n"
            "Then R = 1 / limsup |a_n|^(1/n) = 1 / (1/e) = e.\n"
            "Let's double-check using the ratio test to ensure no Stirling coefficient errors:\n"
            "a_n / a_{n+1} = (n! / n^n) · ((n+1)^{n+1} / (n+1)!) = (n! / (n+1)!) · ((n+1)^{n+1} / n^n)\n"
            "= (1 / (n+1)) · (n+1) · ((n+1)/n)^n = (1 + 1/n)^n.\n"
            "lim_{n→∞} |a_n / a_{n+1}| = lim_{n→∞} (1 + 1/n)^n = e.\n"
            "By the ratio test, R = e.\n"
            "Both Cauchy-Hadamard and the ratio test give R = e."
        ),
        "answer": "e"
    },
    {
        "id": "falsif_ana_contour_integral_branch_cut",
        "domain": "Analysis & ODEs",
        "instruction": "Evaluate the real integral I = ∫₀^∞ (log x) / (1 + x²) dx.",
        "thought": (
            "We want I = ∫₀^∞ (log x) / (1 + x²) dx.\n"
            "Initial attempt: Try substituting x = 1/t:\n"
            "dx = -dt / t².\n"
            "When x = 0, t = ∞; when x = ∞, t = 0.\n"
            "I = ∫_∞^0 (log(1/t) / (1 + (1/t)²)) (-dt / t²)\n"
            "= ∫₀^∞ (-log t / ((t² + 1)/t²)) (dt / t²)\n"
            "= ∫₀^∞ (-log t) / (t² + 1) dt\n"
            "= -I.\n"
            "Since I = -I, we have 2I = 0 ⟹ I = 0.\n"
            "CONFLICT CHECK: Is the integral absolutely convergent so that substitution is valid?\n"
            "- As x → 0⁺: (log x)/(1+x²) ~ log x, and ∫₀¹ log x dx = [x log x - x]₀¹ = -1 (converges).\n"
            "- As x → ∞: (log x)/(1+x²) ~ (log x)/x², and ∫₁^∞ (log x)/x² dx = [-log x / x - 1/x]₁^∞ = 1 (converges).\n"
            "Both endpoints are integrable, so the integral exists and the substitution x = 1/t is strictly justified.\n"
            "Therefore I = 0."
        ),
        "answer": "0"
    },
    {
        "id": "falsif_ana_green_function_laplacian_1d",
        "domain": "Analysis & ODEs",
        "instruction": "Find the Green's function G(x, y) for -d²u/dx² = f(x) on [0, 1] with Dirichlet boundary conditions u(0) = u(1) = 0.",
        "thought": (
            "We want G(x, y) satisfying -G''(x, y) = δ(x - y) with G(0, y) = G(1, y) = 0.\n"
            "For x < y: -G'' = 0 ⟹ G(x, y) = A x + B. Since G(0, y) = 0, B = 0, so G(x, y) = A x.\n"
            "For x > y: -G'' = 0 ⟹ G(x, y) = C(1 - x) to satisfy G(1, y) = 0.\n"
            "Continuity at x = y: A y = C(1 - y).\n"
            "Jump in derivative at x = y:\n"
            "lim_{ε→0} [-G'(y+ε, y) + G'(y-ε, y)] = 1 ⟹ -(-C) + A = 1 ⟹ C + A = 1.\n"
            "From A y = C(1 - y): A y = (1 - A)(1 - y) = 1 - y - A + A y ⟹ A = 1 - y.\n"
            "Then C = 1 - A = y.\n"
            "Thus:\n"
            "- For x ≤ y: G(x, y) = x(1 - y)\n"
            "- For x ≥ y: G(x, y) = y(1 - x)\n"
            "In compact form: G(x, y) = min(x, y)(1 - max(x, y))."
        ),
        "answer": "min(x, y)(1 - max(x, y))"
    },
    {
        "id": "falsif_ana_fourier_transform_gaussian",
        "domain": "Analysis & ODEs",
        "instruction": "Compute the Fourier transform F[e^(-a x²)](k) = ∫_{-∞}^∞ e^(-a x²) e^(-i k x) dx for a > 0.",
        "thought": (
            "We want I(k) = ∫_{-∞}^∞ e^(-a x² - i k x) dx.\n"
            "Complete the square in the exponent:\n"
            "-a x² - i k x = -a (x² + (ik/a) x) = -a (x + ik/(2a))² - k²/(4a).\n"
            "Then I(k) = e^(-k²/(4a)) ∫_{-∞}^∞ e^(-a (x + ik/(2a))²) dx.\n"
            "Shift contour in the complex plane to the line Im(z) = -k/(2a).\n"
            "By Cauchy's theorem (since e^(-a z²) has no poles and decays exponentially as |Re(z)| → ∞ in the horizontal strip):\n"
            "∫_{-∞}^∞ e^(-a (x + ik/(2a))²) dx = ∫_{-∞}^∞ e^(-a u²) du = √(π/a).\n"
            "Therefore I(k) = √(π/a) e^(-k² / (4a))."
        ),
        "answer": "sqrt(pi/a) e^(-k^2/(4a))"
    },
    {
        "id": "falsif_ana_airy_function_asymptotics",
        "domain": "Analysis & ODEs",
        "instruction": "Determine the leading asymptotic decay rate of Ai(x) as x → +∞, i.e., the exponent α in Ai(x) ~ C x^α e^(-2/3 x^(3/2)).",
        "thought": (
            "Airy's equation is y'' - x y = 0.\n"
            "We use the WKB (Liouville-Green) approximation:\n"
            "Let y = exp(S(x)). Then S'' + (S')² - x = 0.\n"
            "Leading order: (S₀')² = x ⟹ S₀'(x) = -x^(1/2) (for decaying solution as x → +∞).\n"
            "S₀(x) = -2/3 x^(3/2).\n"
            "Next order: 2 S₀' S₁' + S₀'' = 0 ⟹ S₁'(x) = -S₀''(x) / (2 S₀'(x)) = -(-1/2 x^(-1/2)) / (-2 x^(1/2)) = -1/(4x).\n"
            "S₁(x) = -1/4 log x = log(x^(-1/4)).\n"
            "Therefore y(x) ≈ exp(S₀ + S₁) = x^(-1/4) exp(-2/3 x^(3/2)).\n"
            "The exponent α is -1/4."
        ),
        "answer": "-1/4"
    },
    {
        "id": "falsif_ana_laplace_eigenvalue_circle",
        "domain": "Analysis & ODEs",
        "instruction": "Compute the k-th distinct eigenvalue λ_k of the Laplace-Beltrami operator -Δ on the standard unit circle S¹.",
        "thought": (
            "On S¹ with coordinate θ ∈ [0, 2π), the Laplace-Beltrami operator is -d²/dθ².\n"
            "Eigenvalue equation: -y''(θ) = λ y(θ) with periodic boundary conditions y(θ + 2π) = y(θ).\n"
            "Solutions are y_n(θ) = e^(i n θ) for n ∈ Z.\n"
            "-d²/dθ² (e^(i n θ)) = n² e^(i n θ).\n"
            "So the eigenvalues are λ = n² for n = 0, 1, 2, 3, ...\n"
            "The k-th distinct eigenvalue (starting from k=0) is λ_k = k²."
        ),
        "answer": "k^2"
    },
    {
        "id": "falsif_ana_fractional_derivative_power",
        "domain": "Analysis & ODEs",
        "instruction": "Compute the Riemann-Liouville half-derivative D^{1/2} [x] evaluated at x > 0.",
        "thought": (
            "The Riemann-Liouville fractional derivative of power x^μ (for μ > -1) of order α > 0 is:\n"
            "D^α [x^μ] = (Γ(μ + 1) / Γ(μ + 1 - α)) x^(μ - α).\n"
            "Here μ = 1 and α = 1/2.\n"
            "Numerator: Γ(1 + 1) = Γ(2) = 1! = 1.\n"
            "Denominator: Γ(1 + 1 - 1/2) = Γ(3/2) = (1/2) Γ(1/2) = (1/2) √π.\n"
            "Power of x: x^(1 - 1/2) = x^(1/2) = √x.\n"
            "Ratio: 1 / ((1/2) √π) = 2 / √π.\n"
            "Therefore D^{1/2}[x] = (2 / √π) x^(1/2) = 2 √(x/π)."
        ),
        "answer": "2 sqrt(x/pi)"
    },
    {
        "id": "falsif_ana_sobolev_embedding_critical",
        "domain": "Analysis & ODEs",
        "instruction": "Find the critical Sobolev exponent 2* for the embedding H¹(R^d) ↪ L^{2*}(R^d) when d ≥ 3.",
        "thought": (
            "We want the critical Sobolev exponent p = 2* such that ||u||_{L^p} ≤ C ||∇u||_{L²} under scaling.\n"
            "Let u_λ(x) = u(λx) for λ > 0.\n"
            "L^p norm scaling: ||u_λ||_{L^p} = (∫ |u(λx)|^p dx)^(1/p) = λ^(-d/p) ||u||_{L^p}.\n"
            "L² gradient norm scaling: ∇(u_λ)(x) = λ ∇u(λx).\n"
            "||∇u_λ||_{L²} = (∫ λ² |∇u(λx)|² dx)^(1/2) = λ · λ^(-d/2) ||∇u||_{L²} = λ^(1 - d/2) ||∇u||_{L²}.\n"
            "For the inequality ||u_λ||_{L^p} ≤ C ||∇u_λ||_{L²} to hold for all λ > 0, the scaling powers of λ must match:\n"
            "-d/p = 1 - d/2 ⟹ d/p = d/2 - 1 = (d - 2)/2 ⟹ p = 2d / (d - 2).\n"
            "Therefore 2* = 2d / (d - 2)."
        ),
        "answer": "2d / (d - 2)"
    },

    # ── DOMAIN D: PHYSICS & APPLIED MATH (8 examples) ──
    {
        "id": "falsif_phys_planck_mass_ratio",
        "domain": "Physics & Applied Math",
        "instruction": "Express the reduced Planck mass M_P = (8πG)^(-1/2) in terms of the standard Planck mass m_P = G^(-1/2) (in units where ħ = c = 1).",
        "thought": (
            "We have two definitions of Planck mass:\n"
            "1. Standard Planck mass: m_P = 1 / √G.\n"
            "2. Reduced Planck mass: M_P = 1 / √(8πG).\n"
            "Express M_P in terms of m_P:\n"
            "M_P = 1 / (√(8π) · √G) = (1 / √(8π)) · (1 / √G) = m_P / √(8π).\n"
            "Equivalently M_P = m_P / (2 √(2π))."
        ),
        "answer": "m_P / sqrt(8pi)"
    },
    {
        "id": "falsif_phys_casimir_force_parallel_plates",
        "domain": "Physics & Applied Math",
        "instruction": "Determine the power-law dependence of the Casimir pressure P between two ideal conducting parallel plates on separation distance d.",
        "thought": (
            "The Casimir energy per unit area for a scalar/EM field between parallel plates separated by d is:\n"
            "E/A = -π² ħ c / (720 d³).\n"
            "The Casimir pressure (force per unit area) is P = -d(E/A)/dd:\n"
            "P = -(-3) π² ħ c / (720 d⁴) = -π² ħ c / (240 d⁴).\n"
            "The pressure scales as 1/d⁴ = d^(-4)."
        ),
        "answer": "d^(-4)"
    },
    {
        "id": "falsif_phys_klein_gordon_propagator_mass",
        "domain": "Physics & Applied Math",
        "instruction": "Find the pole of the Feynman propagator Δ_F(p) = i / (p² - m² + iε) in the complex p₀ plane for fixed spatial momentum p_vec.",
        "thought": (
            "We have p² = p₀² - |p_vec|².\n"
            "Denominator: p₀² - |p_vec|² - m² + iε = p₀² - (E_p² - iε) where E_p = √(|p_vec|² + m²).\n"
            "Factor the denominator:\n"
            "p₀² - (E_p - iε')² = (p₀ - (E_p - iε'))(p₀ + (E_p - iε')).\n"
            "The poles in the complex p₀ plane are at:\n"
            "p₀ = +E_p - iε (in the lower-half plane, slightly below positive real axis)\n"
            "p₀ = -E_p + iε (in the upper-half plane, slightly above negative real axis)\n"
            "The positive-energy pole is at p₀ = E_p - iε."
        ),
        "answer": "E_p - i*epsilon"
    },
    {
        "id": "falsif_phys_stress_energy_trace_conformal",
        "domain": "Physics & Applied Math",
        "instruction": "Compute the classical trace T^μ_μ of the stress-energy tensor for Maxwell electrodynamics in d = 4 spacetime dimensions.",
        "thought": (
            "The Maxwell stress-energy tensor in d dimensions is:\n"
            "T^μ_ν = F^μα F_{να} - (1/4) δ^μ_ν F_{αβ} F^αβ.\n"
            "Take the trace T^μ_μ:\n"
            "T^μ_μ = F^μα F_{μα} - (1/4) δ^μ_μ F_{αβ} F^αβ.\n"
            "Note that δ^μ_μ = d (the spacetime dimension).\n"
            "So T^μ_μ = F^αβ F_{αβ} - (d/4) F_{αβ} F^αβ = (1 - d/4) F_{αβ} F^αβ.\n"
            "For d = 4:\n"
            "T^μ_μ = (1 - 4/4) F_{αβ} F^αβ = 0 · F_{αβ} F^αβ = 0.\n"
            "The classical Maxwell action is conformally invariant in d = 4, so the trace vanishes identically."
        ),
        "answer": "0"
    },
    {
        "id": "falsif_phys_flux_quantization_superconductor",
        "domain": "Physics & Applied Math",
        "instruction": "Compute the superconducting magnetic flux quantum Φ₀ in terms of h and elementary charge e.",
        "thought": (
            "In a superconductor, charge carriers are Cooper pairs with charge q = 2e.\n"
            "Single-valuedness of the superconducting order parameter ψ = |ψ|e^(iθ) around a closed loop enclosing the flux gives:\n"
            "∮ ∇θ · dl = 2π n.\n"
            "Since p = ℏ∇θ = q A = 2e A in the deep interior (where supercurrent vanishes):\n"
            "∮ 2e A · dl = 2e Φ = 2π n ℏ = n h.\n"
            "Therefore Φ = n (h / 2e).\n"
            "The fundamental flux quantum (n=1) is Φ₀ = h / (2e)."
        ),
        "answer": "h / (2e)"
    },
    {
        "id": "falsif_phys_landau_level_degeneracy",
        "domain": "Physics & Applied Math",
        "instruction": "Compute the degeneracy per unit area of a single Landau level for electrons of charge e in a perpendicular magnetic field B.",
        "thought": (
            "Consider a 2D electron gas in a perpendicular magnetic field B.\n"
            "The magnetic length is l_B = √(ℏ / (eB)).\n"
            "Each quantum state in a Landau level occupies an area of 2π l_B² = 2π ℏ / (eB) = h / (eB).\n"
            "The number of states per unit area (the degeneracy density) is the reciprocal:\n"
            "n_B = 1 / (h / (eB)) = eB / h = eB / (2π ℏ)."
        ),
        "answer": "eB / h"
    },
    {
        "id": "falsif_phys_entropy_schwarzschild_black_hole",
        "domain": "Physics & Applied Math",
        "instruction": "Express the Bekenstein-Hawking entropy S_{BH} of a Schwarzschild black hole in terms of its horizon area A.",
        "thought": (
            "The Bekenstein-Hawking formula for black hole entropy is:\n"
            "S_{BH} = (k_B c³ / (4 G ℏ)) A.\n"
            "In natural units (k_B = c = G = ℏ = 1), this reduces to S_{BH} = A / 4.\n"
            "In Planck units where l_P² = G ℏ / c³, S_{BH} = A / (4 l_P²)."
        ),
        "answer": "A / 4"
    },
    {
        "id": "falsif_phys_hall_conductivity_filling_fraction",
        "domain": "Physics & Applied Math",
        "instruction": "Compute the Hall conductance σ_{xy} at filling fraction ν = 1 in units of e²/h.",
        "thought": (
            "In the Integer Quantum Hall Effect (IQHE), the Hall conductance is given by the TKNN (Thouless-Kohmoto-Nightingale-den Nijs) formula:\n"
            "σ_{xy} = ν (e² / h)\n"
            "where ν is the first Chern number (integer topological invariant).\n"
            "At filling fraction ν = 1, σ_{xy} = 1 · (e² / h) = e² / h."
        ),
        "answer": "e^2 / h"
    }
]


def generate_falsification_dataset(output_file: str, append: bool = False):
    """Generate falsification SFT dataset with Wronskian independence filtering."""
    wronskian = WronskianGramFilter(threshold=1e-3)
    dataset = []
    accepted = 0
    rejected = 0

    print("Generating Falsification & Rollback SFT Curriculum...")

    prompts_variations = [
        "Solve this advanced problem step by step, showing all reasoning and self-verification:\n{instruction}",
        "Problem: {instruction}\nWork through the mathematical derivations rigorously."
    ]

    for item in FALSIFICATION_CURRICULUM:
        for var_idx, p_template in enumerate(prompts_variations):
            prompt = p_template.format(instruction=item["instruction"])
            cot = f"<think>\n{item['thought']}\n</think>\n\nAnswer: {item['answer']}"
            sample_text = f"{prompt}\n{cot}"

            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_var{var_idx}",
                    "domain": item["domain"],
                    "cot_type": "falsification_rollback",
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

    print(f"Falsification SFT Dataset {'appended to' if append else 'written to'}: {output_file}")
    print(f"Total Accepted (Wronskian Independent): {accepted}")
    print(f"Total Rejected: {rejected}")
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--output", default=os.path.join(SFT_DIR, "falsification_curriculum.jsonl"))
    args = parser.parse_args()

    generate_falsification_dataset(args.output, append=args.append)
