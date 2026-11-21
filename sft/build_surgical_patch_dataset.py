"""
SODALS v2.1 — Domain 2 Surgical Patch Dataset Generator
========================================================
Creates a focused, high-density curriculum addressing:
  1. sp(6) dim = 21 (Disambiguates 2n=6 -> n=3 -> 3*7=21, prevents 6*7=42)
  2. sp(4) dim = 10 (Disambiguates C_2 n(2n+1)=10 from D_2 n(2n-1)=6)
  3. E_6 dim = 78 (Decouples |Φ|=72 from dim=78, prevents 6+78=84)
  4. E_7 dim = 133 (Decouples |Φ|=126 from positive roots=63, prevents 63 error)
  5. Anchors: Aut(Q8)=24, E8=248, so(8)=28, Biquadratic=4, F4=52, G2=14.
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
OUTPUT_FILE = os.path.join(SFT_DIR, "domain2_surgical_patch_dataset.jsonl")

def make_sample(prompt: str, thought: str, answer: str, sample_id: str):
    return {
        "id": sample_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": f"<think>\n{thought}\n</think>\n\nAnswer: {answer}"
            }
        ]
    }

samples = []

# ==============================================================================
# 1. sp(6, C) DIMENSION = 21
# ==============================================================================
sp6_prompts = [
    "What is the exact dimension of the complex symplectic Lie algebra \\mathfrak{sp}(6, \\mathbb{C}) (also denoted \\mathfrak{sp}(6))?",
    "Compute the dimension of the symplectic Lie algebra \\mathfrak{sp}(6, \\mathbb{C}) from block matrix decomposition.",
    "Determine dim(\\mathfrak{sp}(6)) using Cartan type C_3 root system and matrix constraints.",
    "Calculate the exact complex dimension of the symplectic Lie algebra \\mathfrak{sp}(6)."
]

for idx, p in enumerate(sp6_prompts):
    thought = (
        "## PROBLEM ANALYSIS & FIRST-PRINCIPLES FORMULATION\n"
        "Domain: Lie Theory / Symplectic Geometry\n"
        "Objective: Compute dim(\\mathfrak{sp}(6, \\mathbb{C})).\n\n"
        "### STEP 1: Identification of Rank vs Matrix Dimension\n"
        "In standard mathematical notation, \\mathfrak{sp}(2n, \\mathbb{C}) (often written \\mathfrak{sp}(6) when 2n=6) consists of 2n \\times 2n complex matrices.\n"
        "Here the matrix size is 2n = 6, which uniquely determines the Cartan rank as n = 3 (Cartan family C_3).\n"
        "CRITICAL DISTINCTION: The rank is n = 3, NOT 6! Using n = 6 is a catastrophic index error.\n\n"
        "### STEP 2: Block Matrix Decomposition\n"
        "A matrix X \\in \\mathfrak{sp}(6, \\mathbb{C}) is a 6 \\times 6 matrix partitioned into 3 \\times 3 blocks:\n"
        "X = \\begin{pmatrix} A & B \\\\ C & D \\end{pmatrix}, \\quad X^T J + J X = 0, \\quad J = \\begin{pmatrix} 0 & I_3 \\\\ -I_3 & 0 \\end{pmatrix}.\n"
        "This symplectic condition imposes 3 independent constraints:\n"
        "1. D = -A^T: The block D is completely determined by A (0 additional degrees of freedom).\n"
        "2. B = B^T: The block B is a symmetric 3 \\times 3 matrix.\n"
        "3. C = C^T: The block C is a symmetric 3 \\times 3 matrix.\n\n"
        "### STEP 3: Parameter Counting\n"
        "- Block A: Any 3 \\times 3 matrix in \\mathfrak{gl}(3, \\mathbb{C}) \\implies n^2 = 3^2 = 9 free parameters.\n"
        "- Block B: Symmetric 3 \\times 3 matrix \\implies \\frac{n(n+1)}{2} = \\frac{3 \\times 4}{2} = 6 free parameters.\n"
        "- Block C: Symmetric 3 \\times 3 matrix \\implies \\frac{n(n+1)}{2} = \\frac{3 \\times 4}{2} = 6 free parameters.\n"
        "- Block D: Fixed as -A^T \\implies 0 free parameters.\n"
        "Total Dimension = 9 + 6 + 6 = 21.\n\n"
        "### STEP 4: Dimension Formula Verification\n"
        "The universal dimension formula for C_n is \\dim(\\mathfrak{sp}(2n)) = n(2n+1).\n"
        "For n = 3: \\dim(\\mathfrak{sp}(6)) = 3(2 \\times 3 + 1) = 3 \\times 7 = 21.\n"
        "(Falsification: Computing 6 \\times 7 = 42 is WRONG because 6 is 2n, not n!).\n"
        "Cevap: 21."
    )
    samples.append(make_sample(p, thought, "21", f"patch_sp6_v{idx}"))

# ==============================================================================
# 2. sp(4, C) DIMENSION = 10
# ==============================================================================
sp4_prompts = [
    "What is the exact dimension of the complex symplectic Lie algebra \\mathfrak{sp}(4, \\mathbb{C})?",
    "Compute dim(\\mathfrak{sp}(4, \\mathbb{C})) using block matrix parameters.",
    "Find the dimension of \\mathfrak{sp}(4) and explain why it is 10, not 6."
]

for idx, p in enumerate(sp4_prompts):
    thought = (
        "## PROBLEM ANALYSIS & FIRST-PRINCIPLES FORMULATION\n"
        "Domain: Lie Theory / Symplectic Geometry\n"
        "Objective: Compute dim(\\mathfrak{sp}(4, \\mathbb{C})).\n\n"
        "### STEP 1: Rank and Matrix Size\n"
        "The algebra \\mathfrak{sp}(2n, \\mathbb{C}) for 2n = 4 has rank n = 2 (Cartan type C_2).\n\n"
        "### STEP 2: Block Matrix Parameters\n"
        "X = \\begin{pmatrix} A & B \\\\ C & -A^T \\end{pmatrix} with B = B^T, C = C^T (all 2 \\times 2 blocks).\n"
        "- A: Arbitrary 2 \\times 2 matrix \\implies 2^2 = 4 parameters.\n"
        "- B: Symmetric 2 \\times 2 matrix \\implies \\frac{2 \\times 3}{2} = 3 parameters.\n"
        "- C: Symmetric 2 \\times 2 matrix \\implies \\frac{2 \\times 3}{2} = 3 parameters.\n"
        "Total Dimension = 4 + 3 + 3 = 10.\n\n"
        "### STEP 3: Formula and Fallacy Warning\n"
        "Formula for C_n: \\dim = n(2n+1). For n=2: 2(2 \\times 2 + 1) = 2 \\times 5 = 10.\n"
        "FALSIFICATION PASS: Do NOT use n(2n-1) = 2 \\times 3 = 6! That formula is for orthogonal Lie algebra \\mathfrak{so}(4) (type D_2).\n"
        "\\mathfrak{sp}(4) is symplectic C_2 \\cong B_2 = \\mathfrak{so}(5), whose dimension is 10, NOT 6.\n"
        "Cevap: 10."
    )
    samples.append(make_sample(p, thought, "10", f"patch_sp4_v{idx}"))

# ==============================================================================
# 3. E6 DIMENSION = 78
# ==============================================================================
e6_prompts = [
    "What is the exact dimension of the exceptional Lie algebra E_6?",
    "Compute the dimension of the exceptional simple Lie algebra E_6 from its root system.",
    "Derive dim(E_6) using the Cartan-Weyl root space decomposition."
]

for idx, p in enumerate(e6_prompts):
    thought = (
        "## PROBLEM ANALYSIS & FIRST-PRINCIPLES FORMULATION\n"
        "Domain: Exceptional Lie Algebras / Root Systems\n"
        "Objective: Compute dim(E_6).\n\n"
        "### STEP 1: Cartan Subalgebra and Rank\n"
        "The exceptional Lie algebra E_6 has rank r = 6. Thus \\dim(\\mathfrak{h}) = 6.\n\n"
        "### STEP 2: Exact Root Count (STRICT SEPARATION)\n"
        "Let \\Phi be the root system of E_6.\n"
        "- Number of positive roots: |\\Phi^+| = 36.\n"
        "- Number of negative roots: |\\Phi^-| = 36.\n"
        "- Total number of roots: |\\Phi| = 2 \\times |\\Phi^+| = 2 \\times 36 = 72.\n"
        "CRITICAL ARITHMETIC CHECK: 2 \\times 36 = 72. Under NO circumstances is |\\Phi| equal to 78!\n"
        "78 is the final algebra dimension after adding the rank, NOT the root count.\n\n"
        "### STEP 3: Total Dimension Summation\n"
        "By the Cartan-Weyl decomposition \\mathfrak{g} = \\mathfrak{h} \\oplus \\bigoplus_{\\alpha \\in \\Phi} \\mathfrak{g}_\\alpha:\n"
        "\\dim(E_6) = \\text{rank}(E_6) + |\\Phi| = 6 + 72 = 78.\n"
        "(Falsification: Stating |\\Phi| = 78 and computing 6 + 78 = 84 is a circular reasoning fallacy. The true dimension is 78).\n"
        "Cevap: 78."
    )
    samples.append(make_sample(p, thought, "78", f"patch_e6_v{idx}"))

# ==============================================================================
# 4. E7 DIMENSION = 133
# ==============================================================================
e7_prompts = [
    "Compute the exact dimension of the exceptional simple Lie algebra E_7.",
    "Find the dimension of the exceptional Lie algebra E_7 and explain the root decomposition.",
    "Determine dim(E_7) using rank and root count."
]

for idx, p in enumerate(e7_prompts):
    thought = (
        "## PROBLEM ANALYSIS & FIRST-PRINCIPLES FORMULATION\n"
        "Domain: Exceptional Lie Algebras / Root Systems\n"
        "Objective: Compute dim(E_7).\n\n"
        "### STEP 1: Cartan Subalgebra and Rank\n"
        "The exceptional Lie algebra E_7 has rank r = 7. Thus \\dim(\\mathfrak{h}) = 7.\n\n"
        "### STEP 2: Exact Root Count (STRICT SEPARATION)\n"
        "The root system \\Phi(E_7) contains:\n"
        "- Positive roots: |\\Phi^+| = 63.\n"
        "- Negative roots: |\\Phi^-| = 63.\n"
        "- Total root count: |\\Phi| = 2 \\times 63 = 126 roots.\n"
        "FALSIFICATION: 63 is ONLY the positive root count, not the total dimension or total roots.\n\n"
        "### STEP 3: Total Dimension Summation\n"
        "By Cartan-Weyl decomposition:\n"
        "\\dim(E_7) = \\text{rank}(E_7) + |\\Phi| = 7 + 126 = 133.\n"
        "Cevap: 133."
    )
    samples.append(make_sample(p, thought, "133", f"patch_e7_v{idx}"))

# ==============================================================================
# 5. ANCHORS (Anti-Whack-a-Mole Protection)
# ==============================================================================
# Aut(Q8) = 24
samples.append(make_sample(
    "What is the order of the automorphism group \\mathrm{Aut}(Q_8) of the quaternion group Q_8?",
    "## PROBLEM ANALYSIS\nQ_8 = {±1, ±i, ±j, ±k}, |Q_8| = 8.\nZ(Q_8) = {±1}, so Inn(Q_8) ≅ Q_8/Z(Q_8) ≅ V_4 has order 4.\nAny automorphism permutes the 3 cyclic subgroups <i>, <j>, <k>, inducing Out(Q_8) ≅ S_3 of order 6.\nTotal order |Aut(Q_8)| = |Inn(Q_8)| * |Out(Q_8)| = 4 * 6 = 24.\nAut(Q_8) ≅ S_4.\nCevap: 24.",
    "24",
    "patch_anchor_aut_q8"
))

# E8 = 248
samples.append(make_sample(
    "Compute the exact dimension of the exceptional simple Lie algebra E_8.",
    "## PROBLEM ANALYSIS\nE_8 has rank 8.\nRoots: |\\Phi^+| = 120, total roots |\\Phi| = 240.\ndim(E_8) = rank + |\\Phi| = 8 + 240 = 248.\nCevap: 248.",
    "248",
    "patch_anchor_e8"
))

# so(8) = 28
samples.append(make_sample(
    "What is the dimension of the orthogonal Lie algebra \\mathfrak{so}(8, \\mathbb{C}) of skew-symmetric 8x8 matrices?",
    "## PROBLEM ANALYSIS\n\\mathfrak{so}(n) consists of n x n skew-symmetric matrices.\ndim(\\mathfrak{so}(n)) = n(n-1)/2.\nFor n = 8: 8 * 7 / 2 = 28.\nCevap: 28.",
    "28",
    "patch_anchor_so8"
))

# Biquadratic = 4
samples.append(make_sample(
    "Compute the degree of the field extension [\\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) : \\mathbb{Q}].",
    "## PROBLEM ANALYSIS\nBy the tower law: [Q(√2, √3) : Q] = [Q(√2, √3) : Q(√2)] * [Q(√2) : Q].\nSince √3 ∉ Q(√2), minimal polynomial of √3 over Q(√2) is x^2 - 3 (degree 2).\n[Q(√2) : Q] = 2.\nTotal degree = 2 * 2 = 4.\nCevap: 4.",
    "4",
    "patch_anchor_biquadratic"
))

# F4 = 52
samples.append(make_sample(
    "Compute the exact dimension of the exceptional simple Lie algebra F_4 over the complex numbers.",
    "## PROBLEM ANALYSIS\nF_4 has rank 4. Total roots |\\Phi| = 48 (24 long + 24 short).\ndim(F_4) = rank + |\\Phi| = 4 + 48 = 52.\nCevap: 52.",
    "52",
    "patch_anchor_f4"
))

# G2 = 14
samples.append(make_sample(
    "Compute the exact dimension of the exceptional Lie algebra G_2.",
    "## PROBLEM ANALYSIS\nG_2 has rank 2. Total roots |\\Phi| = 12 (6 short + 6 long).\ndim(G_2) = rank + |\\Phi| = 2 + 12 = 14.\nCevap: 14.",
    "14",
    "patch_anchor_g2"
))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"[SURGICAL DATASET] Generated {len(samples)} high-density samples in {OUTPUT_FILE}")
