import json
import os
import numpy as np
from typing import List, Dict, Any

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
os.makedirs(SFT_DIR, exist_ok=True)

class WronskianGramFilter:
    """
    Wronskian / Gram-Schmidt Determinant Filter.
    Ensures that newly generated training samples span linearly independent
    subspaces in feature representation space, filtering out degenerate or repetitive CoT traces.
    """
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.feature_vectors: List[np.ndarray] = []

    def _extract_simple_features(self, text: str, dim: int = 64) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float64)
        words = text.lower().split()
        for i, w in enumerate(words):
            h = hash(w) % dim
            vec[h] += 1.0 / (1.0 + np.log1p(i))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def is_independent(self, text: str) -> bool:
        v = self._extract_simple_features(text)
        if not self.feature_vectors:
            self.feature_vectors.append(v)
            return True

        residual = v.copy()
        for b in self.feature_vectors:
            proj = np.dot(residual, b) * b
            residual -= proj

        residual_norm = np.linalg.norm(residual)
        if residual_norm >= self.threshold:
            self.feature_vectors.append(v)
            return True
        return False

# ==============================================================================
# UNCONTAMINATED ADVANCED DOMAIN REASONING CURRICULUM (12 Independent Modules)
# Zero overlap with CAIS HLE test split or direct perturbation test problems.
# Focus: First-principles CoT reasoning across differential geometry, Lie theory,
# algebraic topology, number theory, and mathematical physics.
# ==============================================================================
UNCONTAMINATED_CURRICULUM: List[Dict[str, Any]] = [
    # ------------------ STAGE 1: FOUNDATIONAL HOMOLOGICAL ALGEBRA (Depth 1, Weight 1.0) ------------------
    {
        "id": "dom_hom_ext_group_torus",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Computation of Ext^1(Z_n, Z)",
        "instruction": "Compute the abelian group \\mathrm{Ext}^1(\\mathbb{Z}/n\\mathbb{Z}, \\mathbb{Z}) for an integer n >= 2.",
        "thought": (
            "We compute the derived functor Ext^1 using a projective resolution of \\mathbb{Z}/n\\mathbb{Z}.\n"
            "1. Free/Projective Resolution of \\mathbb{Z}/n\\mathbb{Z}:\n"
            "0 \\to \\mathbb{Z} \\xrightarrow{\\times n} \\mathbb{Z} \\to \\mathbb{Z}/n\\mathbb{Z} \\to 0.\n"
            "2. Applying the functor \\mathrm{Hom}(-, \\mathbb{Z}):\n"
            "0 \\to \\mathrm{Hom}(\\mathbb{Z}, \\mathbb{Z}) \\xrightarrow{(\\times n)^*} \\mathrm{Hom}(\\mathbb{Z}, \\mathbb{Z}) \\to \\dots\n"
            "Since \\mathrm{Hom}(\\mathbb{Z}, \\mathbb{Z}) \\cong \\mathbb{Z}$, the dual complex is:\n"
            "0 \\to \\mathbb{Z} \\xrightarrow{\\times n} \\mathbb{Z} \\to 0.\n"
            "3. Taking Cohomology:\n"
            "- \\mathrm{Ext}^0(\\mathbb{Z}/n\\mathbb{Z}, \\mathbb{Z}) = \\ker(\\times n) = 0.\n"
            "- \\mathrm{Ext}^1(\\mathbb{Z}/n\\mathbb{Z}, \\mathbb{Z}) = \\mathrm{coker}(\\times n) = \\mathbb{Z} / n\\mathbb{Z}.\n"
            "4. Conclusion:\n"
            "\\mathrm{Ext}^1(\\mathbb{Z}/n\\mathbb{Z}, \\mathbb{Z}) \\cong \\mathbb{Z}/n\\mathbb{Z}."
        ),
        "answer": "\\mathbb{Z}/n\\mathbb{Z}"
    },
    {
        "id": "dom_hom_tor_cyclic_groups",
        "depth": 1,
        "focal_weight": 1.0,
        "topic": "Tor_1 of Cyclic Groups Z_m and Z_n",
        "instruction": "Compute \\mathrm{Tor}_1^{\\mathbb{Z}}(\\mathbb{Z}/m\\mathbb{Z}, \\mathbb{Z}/n\\mathbb{Z}) for positive integers m, n.",
        "thought": (
            "We compute the derived functor Tor using a free resolution of \\mathbb{Z}/m\\mathbb{Z}.\n"
            "1. Free resolution:\n"
            "0 \\to \\mathbb{Z} \\xrightarrow{\\times m} \\mathbb{Z} \\to \\mathbb{Z}/m\\mathbb{Z} \\to 0.\n"
            "2. Tensoring with \\mathbb{Z}/n\\mathbb{Z}:\n"
            "0 \\to \\mathbb{Z} \\otimes \\mathbb{Z}/n\\mathbb{Z} \\xrightarrow{\\times m} \\mathbb{Z} \\otimes \\mathbb{Z}/n\\mathbb{Z} \\to \\dots\n"
            "This simplifies to:\n"
            "0 \\to \\mathbb{Z}/n\\mathbb{Z} \\xrightarrow{\\times m} \\mathbb{Z}/n\\mathbb{Z} \\to 0.\n"
            "3. Homology:\n"
            "\\mathrm{Tor}_1(\\mathbb{Z}/m\\mathbb{Z}, \\mathbb{Z}/n\\mathbb{Z}) = \\ker(\\times m : \\mathbb{Z}/n\\mathbb{Z} \\to \\mathbb{Z}/n\\mathbb{Z}).\n"
            "The elements of \\mathbb{Z}/n\\mathbb{Z} killed by m are those multiples of n / \\gcd(m, n).\n"
            "The size and isomorphism type of this kernel is the cyclic group of order \\gcd(m, n).\n"
            "4. Conclusion:\n"
            "\\mathrm{Tor}_1^{\\mathbb{Z}}(\\mathbb{Z}/m\\mathbb{Z}, \\mathbb{Z}/n\\mathbb{Z}) \\cong \\mathbb{Z}/\\gcd(m, n)\\mathbb{Z}."
        ),
        "answer": "\\mathbb{Z}/\\gcd(m, n)\\mathbb{Z}"
    },
    {
        "id": "dom_top_stiefel_whitney_cp3",
        "depth": 1,
        "focal_weight": 1.1,
        "topic": "Total Stiefel-Whitney Class of CP^3",
        "instruction": "Let \\mathbb{C}P^3 be the 3-dimensional complex projective space. Express the total Stiefel-Whitney class w(\\mathbb{C}P^3) in terms of the generator x \\in H^2(\\mathbb{C}P^3; \\mathbb{Z}_2).",
        "thought": (
            "We compute the total Stiefel-Whitney class w(\\mathbb{C}P^n) for n = 3.\n"
            "1. Tangent bundle formula for complex projective space:\n"
            "The tangent bundle of \\mathbb{C}P^n satisfies T(\\mathbb{C}P^n) \\oplus \\underline{\\mathbb{R}}^1 \\cong (\\gamma^1)^*(n+1), where \\gamma^1 is the tautological line bundle.\n"
            "2. Total Chern class:\n"
            "c(\\mathbb{C}P^n) = (1 + h)^{n+1}, where h = c_1(\\gamma^*) \\in H^2(\\mathbb{C}P^n; \\mathbb{Z}).\n"
            "3. Mod 2 reduction to Stiefel-Whitney classes:\n"
            "w(\\mathbb{C}P^n) = (1 + x)^{n+1} \\pmod 2, where x = w_2(\\gamma^*) \\in H^2(\\mathbb{C}P^n; \\mathbb{Z}_2).\n"
            "For n = 3: (1 + x)^4 = 1 + 4x + 6x^2 + 4x^3 + x^4 \\equiv 1 + x^4 \\pmod 2.\n"
            "Since H^k(\\mathbb{C}P^3; \\mathbb{Z}_2) vanishes for k > 6 and x \\in H^2 has degree 2, x^4 \\in H^8(\\mathbb{C}P^3; \\mathbb{Z}_2) = 0.\n"
            "Thus, (1 + x)^4 = 1 + 0x + 0x^2 + 0x^3 = 1.\n"
            "Therefore, w(\\mathbb{C}P^3) = 1."
        ),
        "answer": "1"
    },

    # ------------------ STAGE 2: MANIFOLD TOPOLOGY & RIEMANN SURFACES (Depth 2, Weight 1.3) ------------------
    {
        "id": "dom_top_euler_grassmannian",
        "depth": 2,
        "focal_weight": 1.3,
        "topic": "Euler Characteristic of Real Grassmannian Gr(2, 4)",
        "instruction": "Compute the Euler characteristic \\chi(\\mathrm{Gr}(2, 4)) of the real Grassmannian of 2-planes in \\mathbb{R}^4.",
        "thought": (
            "We want to compute the Euler characteristic of the real Grassmann manifold Gr(2, 4).\n"
            "1. Cell Decomposition / Schubert Cells:\n"
            "The Grassmannian Gr(k, n) has a CW-complex structure given by Schubert cells indexed by partitions \\lambda = (\\lambda_1, \\dots, \\lambda_k) with n-k >= \\lambda_1 >= ... >= \\lambda_k >= 0.\n"
            "Here k = 2, n = 4, so n - k = 2. The partitions inside a 2x2 box are:\n"
            "- \\lambda = (0,0): dimension 0\n"
            "- \\lambda = (1,0): dimension 1\n"
            "- \\lambda = (2,0): dimension 2\n"
            "- \\lambda = (1,1): dimension 2\n"
            "- \\lambda = (2,1): dimension 3\n"
            "- \\lambda = (2,2): dimension 4\n"
            "2. Homology with Z_2 and Real Euler Characteristic:\n"
            "Using the fixed point theorem for the maximal torus action on Gr(2, 4), or the Weyl group formula:\n"
            "For real Grassmannians Gr(k, n), \\chi(Gr(k, n)) is equal to the binomial coefficient \\binom{\\lfloor n/2 \\rfloor}{\\lfloor k/2 \\rfloor} if both k and n-k have certain parities, or from Schubert cell parity:\n"
            "Only even-dimensional Schubert cells that are orientable contribute to the alternating sum of Betti numbers over \\mathbb{R}.\n"
            "Alternatively, via the Schubert calculus: \\chi(Gr(2,4)) = \\binom{4/2}{2/2} = \\binom{2}{1} = 2.\n"
            "3. Conclusion:\n"
            "The Euler characteristic is 2."
        ),
        "answer": "2"
    },
    {
        "id": "dom_diffgeom_gauss_bonnet_torus_punctured",
        "depth": 2,
        "focal_weight": 1.3,
        "topic": "Total Integral of Gaussian Curvature for a 2-Punctured Torus",
        "instruction": "Let M be a complete, finite-area hyperbolic 2-manifold with the topology of a 2-punctured torus (genus 1 with 2 cusps). Compute the total integral of the Gaussian curvature \\int_M K \\, dA.",
        "thought": (
            "We use the Gauss-Bonnet theorem for non-compact hyperbolic surfaces with cusps.\n"
            "1. Generalized Gauss-Bonnet formula:\n"
            "\\int_M K \\, dA = 2\\pi \\chi(M),\n"
            "where \\chi(M) is the topological Euler characteristic of the punctured surface.\n"
            "2. Euler characteristic of M:\n"
            "For a closed surface of genus g with k punctures:\n"
            "\\chi(M) = 2 - 2g - k.\n"
            "Here g = 1 and k = 2:\n"
            "\\chi(M) = 2 - 2(1) - 2 = 2 - 2 - 2 = -2.\n"
            "3. Integrating curvature:\n"
            "\\int_M K \\, dA = 2\\pi (-2) = -4\\pi.\n"
            "4. Conclusion:\n"
            "The total integral is -4\\pi."
        ),
        "answer": "-4\\pi"
    },
    {
        "id": "dom_diffgeom_riemann_roch_hyperelliptic",
        "depth": 2,
        "focal_weight": 1.4,
        "topic": "Dimension of Linear System on Genus 3 Hyperelliptic Curve",
        "instruction": "Let C be a smooth compact Riemann surface of genus g = 3. Let K be the canonical divisor on C. What is the dimension \\ell(2K) = \\dim H^0(C, \\mathcal{O}(2K)) of the space of quadratic differentials?",
        "thought": (
            "We apply the Riemann-Roch Theorem to the divisor D = 2K on a curve of genus g = 3.\n"
            "1. Degree of Canonical Divisor:\n"
            "\\deg(K) = 2g - 2 = 2(3) - 2 = 4.\n"
            "Therefore, \\deg(2K) = 2 \\times 4 = 8.\n"
            "2. Riemann-Roch Theorem:\n"
            "\\ell(D) - \\ell(K - D) = \\deg(D) - g + 1.\n"
            "For D = 2K:\n"
            "\\deg(K - D) = \\deg(K - 2K) = \\deg(-K) = -4 < 0.\n"
            "Since the degree is strictly negative, \\ell(K - D) = 0.\n"
            "3. Calculation:\n"
            "\\ell(2K) = \\deg(2K) - g + 1 = 8 - 3 + 1 = 6.\n"
            "4. Conclusion:\n"
            "The dimension of the space of quadratic differentials \\ell(2K) is 6."
        ),
        "answer": "6"
    },

    # ------------------ STAGE 3: LIE ALGEBRAS & ROOT SYSTEMS (Depth 3, Weight 1.6) ------------------
    {
        "id": "dom_lie_weyl_group_order_f4",
        "depth": 3,
        "focal_weight": 1.6,
        "topic": "Order of the Weyl Group of Exceptional Lie Algebra F4",
        "instruction": "Compute the order of the Weyl group W(F_4) of the exceptional simple Lie algebra \\mathfrak{f}_4.",
        "thought": (
            "We determine |W(F_4)| using the exponents of the Lie algebra \\mathfrak{f}_4.\n"
            "1. Exponents of F_4:\n"
            "The degrees of the fundamental polynomial invariants (Casimir degrees) for F_4 are d_1 = 2, d_2 = 6, d_3 = 8, d_4 = 12.\n"
            "The exponents m_i = d_i - 1 are therefore: m_1 = 1, m_2 = 5, m_3 = 7, m_4 = 11.\n"
            "2. Product formula for the Weyl group order:\n"
            "The order of any finite reflection / Weyl group is given by the product of the degrees:\n"
            "|W| = \\prod_{i=1}^r d_i = d_1 \\cdot d_2 \\cdot d_3 \\cdot d_4.\n"
            "3. Calculation:\n"
            "|W(F_4)| = 2 \\times 6 \\times 8 \\times 12 = 12 \\times 8 \\times 12 = 96 \\times 12 = 1152.\n"
            "4. Conclusion:\n"
            "The order of the Weyl group W(F_4) is 1152."
        ),
        "answer": "1152"
    },
    {
        "id": "dom_lie_casimir_eigenvalue_su3_adjoint",
        "depth": 3,
        "focal_weight": 1.6,
        "topic": "Quadratic Casimir Eigenvalue for SU(3) Adjoint Representation",
        "instruction": "Compute the eigenvalue of the quadratic Casimir operator C_2 in the adjoint representation (dim=8) of \\mathfrak{su}(3), normalized such that for the fundamental representation (3) with highest weight (1,0) it equals 4/3.",
        "thought": (
            "We compute C_2(\\text{adj}) for \\mathfrak{su}(N) with N = 3.\n"
            "1. Highest weight of representations:\n"
            "Fundamental representation: highest weight \\Lambda = (1, 0), C_2(3) = \\frac{N^2 - 1}{2N} = \\frac{8}{6} = \\frac{4}{3}.\n"
            "Adjoint representation: highest weight \\Lambda = (1, 1).\n"
            "2. General Casimir Formula for \\mathfrak{su}(N):\n"
            "For a representation R with Young tableau / Dynkin labels, or using the dual Coxeter number h^\\vee = N:\n"
            "In the adjoint representation, the quadratic Casimir operator satisfies:\n"
            "C_2(\\text{adj}) = 2 h^\\vee = 2 N (under standard normalization) or relative to fundamental:\n"
            "C_2(\\text{adj}) = N = 3.\n"
            "Let us check: C_2(\\text{adj}) / C_2(3) = 3 / (4/3) = 9/4 = 2N^2 / (N^2 - 1) = 18 / 8 = 9/4.\n"
            "Thus C_2(\\text{adj}) = 3.\n"
            "3. Conclusion:\n"
            "The eigenvalue is 3."
        ),
        "answer": "3"
    },

    # ------------------ STAGE 4: ARITHMETIC GEOMETRY & NUMBER FIELDS (Depth 4, Weight 1.8) ------------------
    {
        "id": "dom_num_weil_pairing_elliptic_rank",
        "depth": 4,
        "focal_weight": 1.8,
        "topic": "Order of the Torsion Point in Tate Pairing",
        "instruction": "Let E be an elliptic curve over \\mathbb{F}_p with p = 101, having trace of Frobenius a_p = 2. Find the number of \\mathbb{F}_p-rational points #E(\\mathbb{F}_p).",
        "thought": (
            "We use Hasse's theorem on elliptic curves over finite fields.\n"
            "1. Point count formula:\n"
            "#E(\\mathbb{F}_p) = p + 1 - a_p,\n"
            "where a_p is the trace of Frobenius.\n"
            "2. Computation:\n"
            "p = 101, a_p = 2.\n"
            "#E(\\mathbb{F}_{101}) = 101 + 1 - 2 = 100.\n"
            "3. Conclusion:\n"
            "The number of rational points is 100."
        ),
        "answer": "100"
    },
    {
        "id": "dom_num_class_number_q_sqrt_neg23",
        "depth": 4,
        "focal_weight": 1.9,
        "topic": "Class Number of Imaginary Quadratic Field Q(sqrt(-23))",
        "instruction": "Compute the class number h(K) of the imaginary quadratic field K = \\mathbb{Q}(\\sqrt{-23}).",
        "thought": (
            "We compute the class number h(d) for d = -23.\n"
            "1. Discriminant:\n"
            "Since -23 \\equiv 1 \\pmod 4, the fundamental discriminant is D = -23.\n"
            "2. Minkowski Bound:\n"
            "M_K = \\frac{2}{\\pi} \\sqrt{|D|} = \\frac{2}{\\pi} \\sqrt{23} \\approx \\frac{2 \\times 4.7958}{3.14159} \\approx 3.053.\n"
            "We only need to examine prime ideals lying over the primes p <= 3 (i.e. p = 2 and p = 3).\n"
            "3. Decomposition of primes:\n"
            "- p = 2: Kronecker symbol (D / 2) = (-23 / 2) = (1 / 2) = 1 (since -23 \\equiv 1 mod 8).\n"
            "Thus (2) splits into \\mathfrak{p}_2 \\bar{\\mathfrak{p}}_2. Is \\mathfrak{p}_2 principal?\n"
            "A principal ideal would have norm N(x + y\\frac{1+\\sqrt{-23}}{2}) = x^2 + xy + 6y^2 = 2. No integer solution exists, so [\\mathfrak{p}_2] has order > 1.\n"
            "- Check order of [\\mathfrak{p}_2]:\n"
            "N(x + y\\frac{1+\\sqrt{-23}}{2}) = 2^3 = 8 -> x^2 + xy + 6y^2 = 8 has solution (x, y) = (1, 1) since 1 + 1 + 6 = 8.\n"
            "Thus \\mathfrak{p}_2^3 is principal, meaning [\\mathfrak{p}_2] has order 3.\n"
            "- p = 3: Kronecker symbol (-23 / 3) = (1 / 3) = 1. (3) splits into \\mathfrak{p}_3 \\bar{\\mathfrak{p}}_3.\n"
            "N = x^2 + xy + 6y^2 = 3 has no solutions, but \\mathfrak{p}_3 is in the cyclic group generated by [\\mathfrak{p}_2].\n"
            "4. Conclusion:\n"
            "The ideal class group is isomorphic to \\mathbb{Z}/3\\mathbb{Z}, so the class number is 3."
        ),
        "answer": "3"
    },

    # ------------------ STAGE 5: THEORETICAL & QUANTUM PHYSICS (Depth 5, Weight 2.2) ------------------
    {
        "id": "dom_phys_virasoro_kac_determinant",
        "depth": 5,
        "focal_weight": 2.2,
        "topic": "Virasoro Central Charge for Free Real Boson",
        "instruction": "What is the central charge c of the 2D Virasoro algebra for a system of D free massless scalar fields in two dimensions?",
        "thought": (
            "We evaluate the central charge of the energy-momentum tensor for free scalar fields.\n"
            "1. Free scalar field action:\n"
            "S = \\frac{1}{4\\pi} \\int d^2z \\partial X^\\mu \\bar{\\partial} X_\\mu for \\mu = 1, \\dots, D.\n"
            "2. Energy-momentum tensor:\n"
            "T(z) = -\\frac{1}{2} :\\partial X^\\mu \\partial X_\\mu:.\n"
            "3. OPE of T(z)T(w):\n"
            "T(z)T(w) \\sim \\frac{c/2}{(z-w)^4} + \\frac{2T(w)}{(z-w)^2} + \\frac{\\partial T(w)}{z-w}.\n"
            "For a single free real boson (D = 1), Wick contractions yield c = 1.\n"
            "For D independent free real bosons, the central charges sum linearly: c = D.\n"
            "4. Conclusion:\n"
            "The central charge is D."
        ),
        "answer": "D"
    },
    {
        "id": "dom_phys_instanton_number_su2",
        "depth": 5,
        "focal_weight": 2.5,
        "topic": "Second Chern Number for SU(2) Instanton",
        "instruction": "Consider a pure SU(2) Yang-Mills gauge field on Euclidean 4-space \\mathbb{R}^4 with field strength F. The topological charge / instanton number k is given by \\frac{1}{8\\pi^2} \\int_{\\mathbb{R}^4} \\mathrm{Tr}(F \\wedge F). For the standard BPST 1-instanton configuration, what is the value of k?",
        "thought": (
            "We evaluate the instanton topological invariant for the BPST instanton.\n"
            "1. Definition of topological charge:\n"
            "k = c_2(E) = \\frac{1}{8\\pi^2} \\int_{S^4} \\mathrm{Tr}(F \\wedge F).\n"
            "2. Homotopy classification:\n"
            "Pure gauge at infinity maps the boundary S^3 to SU(2) \\cong S^3.\n"
            "The winding number in \\pi_3(SU(2)) \\cong \\mathbb{Z} defines the topological charge.\n"
            "3. For the standard single BPST instanton with gauge connection A_\\mu = \\frac{r^2}{r^2 + \\rho^2} U^{-1} \\partial_\\mu U, the winding number is exactly 1.\n"
            "4. Conclusion:\n"
            "The value of k is 1."
        ),
        "answer": "1"
    }
]

def generate_clean_sft_dataset(output_file: str, augment_factor: int = 8):
    """
    Generates structured SFT dataset ordered by Hamiltonian geometric depth
    with Arkhe focal weights, filtered through Wronskian Gram-Schmidt filter.
    """
    wronskian = WronskianGramFilter(threshold=1e-3)
    dataset = []
    accepted = 0
    rejected = 0

    print(f"Generating Hamiltonian-Ordered SFT Curriculum (Stages 1 -> 5)...")

    # Sort curriculum items by geometric depth
    sorted_curriculum = sorted(UNCONTAMINATED_CURRICULUM, key=lambda x: (x["depth"], x["focal_weight"]))

    prompts_variations = [
        "Please solve the following advanced problem step by step:\n{instruction}",
        "Question: {instruction}\nProvide a complete mathematical derivation and final answer.",
        "Problem: {instruction}\nWork through the algebraic steps in detail and conclude with the final result.",
        "Solve the following mathematical question rigorously: {instruction}"
    ]

    for item in sorted_curriculum:
        for var_idx, p_template in enumerate(prompts_variations):
            prompt = p_template.format(instruction=item["instruction"])
            cot = f"<think>\n{item['thought']}\n</think>\n\nAnswer: {item['answer']}"
            
            sample_text = f"{prompt}\n{cot}"
            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_var{var_idx}",
                    "topic": item["topic"],
                    "depth": item["depth"],
                    "focal_weight": item["focal_weight"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot}
                    ]
                })
                accepted += 1
            else:
                rejected += 1

    for aug in range(augment_factor):
        for item in sorted_curriculum:
            prompt = f"Evaluate and solve: {item['instruction']}"
            cot = f"<think>\n{item['thought']}\n</think>\n\nAnswer: {item['answer']}"
            
            sample_text = f"{prompt}\n{cot}"
            if wronskian.is_independent(sample_text):
                dataset.append({
                    "id": f"{item['id']}_aug{aug}",
                    "topic": item["topic"],
                    "depth": item["depth"],
                    "focal_weight": item["focal_weight"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot}
                    ]
                })
                accepted += 1
            else:
                rejected += 1

    # Sort the full dataset strictly by geometric depth and focal weight to preserve Hamiltonian curriculum
    dataset.sort(key=lambda x: (x["depth"], x["focal_weight"]))

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Clean SFT Dataset generated at: {output_file}")
    print(f"Total Accepted Samples (Hamiltonian Sorted): {accepted}")
    print(f"Total Rejected Samples (Redundant / Degenerate): {rejected}")

if __name__ == "__main__":
    out_path = os.path.join(SFT_DIR, "domain1_sft_curriculum.jsonl")
    generate_clean_sft_dataset(out_path)
