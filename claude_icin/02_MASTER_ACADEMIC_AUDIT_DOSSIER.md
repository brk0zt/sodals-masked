# SODALS Core — Academic Audit Response & Integrity Verification Dossier

**Document Version:** 1.0  
**Date:** October 2026  
**Author:** Burak Öztürk & SODALS AI Research Group  
**Target Reviewers:** Independent Academic Reviewer & Research Audit Board  
**Scope:** Formal Verification, Cross-Paper Decoupling, Statistical Disclosure, and Algorithmic Corrections across the SODALS Three-Paper Portfolio.

---

## Executive Summary

Following the comprehensive academic audit of the SODALS project, this dossier documents the precise, evidence-based resolutions implemented across the codebase, datasets, and paper drafts. Rather than prematurely characterizing the project as "bulletproof," our team has implemented a rigorous, uncompromising research integrity protocol:

1. **Absolute Numerical & Domain Decoupling:** Complete segregation between foundational mathematical reasoning benchmarks (Paper 1 & Paper 2) and cyber-physical systems (CPS) telemetry benchmarks (Paper 3).
2. **Zero-Contamination Train-Test Segregation:** Complete physical separation of the 25 GRPO pilot prompts from the 125 held-out evaluation questions, accompanied by a transparent disclosure of the exact McNemar test result ($p = 1.0000$).
3. **Physical & Numerical Precision Alignment:** Transparent stratification of Layer-0 versus Layer-32 metrics, and host CPU reset latency ($0.22\text{ ms}$) versus total GPU prefix re-prefill latency ($12.7\text{--}46.2\text{ ms}$).
4. **Algorithmic Notation & Terminology Accuracy:** Correction of graph complexity notation to $O(N \log \text{deg}_{\max})$ and reframing of conflict recording as a *Nogood Cache* inspired by automated reasoning principles.
5. **Exact Mathematical Formulations:** Correct characterization of the DeltaNet recurrent state update as an affine transformation invertible in exact arithmetic, but numerically ill-conditioned under finite floating-point precision.
6. **Benchmark Provenance & Reference Alignment:** Clear formal distinction between IBM's *AssetOpsBench* and *FailureSensorIQ*, with complete, validated 1-to-1 bibliographic citations.

---

## 1. Domain Decoupling & Elimination of Duplicate Numbers

### Problem Identified in Audit
Earlier drafts inadvertently duplicated ablation fractions ($0/35$, $4/35$, $21/24$, $138/170$, $122/150$) between the mathematical reasoning draft and the EdgeSSM draft under two distinct domain labels ("industrial failure recovery" vs. "MATH-500 errors").

### Resolution & Empirical Verification
- **Paper 3 (`SODALS_arXiv_Paper.md`):** Completely stripped of all MATH-500, Lie algebra, and symbolic reasoning benchmarks. It now exclusively reports cyber-physical telemetry benchmarks:
  - AssetOpsBench-Derived Diagnostic Subset ($N=120$): $96/120$ ($80.0\%$) diagnostic exact match; $113/120$ ($94.2\%$) workflow completion.
  - SRE Troubleshooting Suite ($N=38$): $34/38$ ($89.5\%$) precision.
  - FinOps Scenario Suite ($N=15$): $14/15$ ($93.3\%$) precision.
  - EdgeTelemetry-50 ($N=50$): $46/50$ ($92.0\%$) fault isolation precision.
- **Paper 1 (`arxiv/arXiv_SODALS_v2.1_Master_Paper.md`):** Exclusively reports mathematical reasoning benchmarks (MATH-500 full $N=500$, clean held-out $N=125$, training reconstruction $N=25$, Pristine 24Q STEM, and Humanity's Last Exam).

### Automated Cross-Paper Verification Scan
An automated full-text cross-scan confirms:
- **MATH terms in Paper 3:** `0/35` (0), `4/35` (0), `21/24` (0), `138/170` (0), `122/150` (0), `MATH-500` (0), `Lie algebra` (0), `F_4` (0), `E_7` (0).
- **CPS terms in Paper 1:** `AssetOpsBench` (0), `EdgeTelemetry` (0), `FailureSensorIQ` (0), `96/120` (0), `113/120` (0), `46/50` (0), `SCADA` (0), `PLC` (0).
- **Common Fractions:** Zero overlapping experimental fractions across differing domains.

---

## 2. Train-Test Leakage Elimination in GRPO & Statistical Truth

### Problem Identified in Audit
The 25 prompt items utilized for GRPO pilot rollouts were co-evaluated within the 150-question MATH-500 test set, creating a risk of train-test data leakage.

### Resolution & Dataset Segregation
The dataset has been physically divided into immutable JSON files:
1. `sft/math500_heldout_clean_125_dataset.json`: Strictly the 125 held-out, unseen test questions.
2. `sft/math500_training_pilot_25_dataset.json`: The 25 training prompts used for RL rollouts.
3. `sft/math500_heldout_clean_125_results.json`: Full itemized model evaluation logs on the clean set.
4. `sft/math500_training_pilot_25_results.json`: Training subset reconstruction logs.

### Statistical Re-Verification on Clean 125 Held-Out Set
Evaluated under deterministic greedy decoding ($T=0$):
- **Frozen SFT Baseline:** $99 / 125$ ($79.20\%$) [95% Wilson CI: $71.27\%$, $85.39\%$]
- **Long-Context GRPO ($L=1536$):** $100 / 125$ ($80.00\%$) [95% Wilson CI: $72.14\%$, $86.07\%$]
- **Prefill Pondering ODE (RK4):** $100 / 125$ ($80.00\%$) [95% Wilson CI: $72.14\%$, $86.07\%$]
- **Paired Outcomes:**
  - Both Correct: $99$
  - Both Incorrect: $25$
  - Baseline FAIL $\to$ GRPO PASS: $1$ (Question 105)
  - Baseline PASS $\to$ GRPO FAIL: $0$ (Zero regressions / 100% policy stability)
- **Exact McNemar Paired Test:**
  $$p = 1.0000 \quad (\text{Statistically Non-Significant})$$
- **10,000-Sample Paired Bootstrap $\Delta$ 95% CI:** $[0.00\%, 2.40\%]$

### Methodological Disclosure in Paper 1
Table 4 was split into **Table 4A** (Primary: 125Q Clean Held-Out Generalization) and **Table 4B** (Secondary: 25Q Training Subset Reconstruction). The text explicitly acknowledges:
> *"The +1 gain on held-out problems is statistically non-significant ($p = 1.0000$). The scientific contribution does not rest on an unverified claim of having 'broken the SFT ceiling,' but rather on establishing: (1) a zero-paging long-context VRAM architecture enabling 1536-token rollouts on 8GB consumer GPUs, and (2) complete empirical policy stability (zero regressions across all 125 held-out items)."*

---

## 3. Q105 Case Study & Token Count Clarification

### Problem Identified in Audit
The baseline token count on Question 105 was inconsistently cited across previous internal discussions (e.g., 218 tokens vs. 844 tokens).

### Resolution
- Inspection of the raw evaluation logs revealed that the baseline output was recorded with `tokens: None` (predicting an immediate incorrect root sum of `18`). The "218" and "844" figures originated from unconstrained exploratory scratch runs without fixed stopping criteria.
- **Paper Correction:** Speculative baseline token counts were completely removed. The paper strictly cites verifiable empirical facts:
  - **Frozen SFT Baseline:** Predicted `18` (FAIL).
  - **Long-Context GRPO:** Autonomously generated **1,023 tokens of sustained chain-of-thought** (verified in `sft/math500_heldout_clean_125_results.json`), systematically verifying boundary domains and arriving at the correct ground truth: $\boxed{40}$ (PASS).

---

## 4. Physical & Numerical Metrics Alignment

### Problem Identified in Audit
- Apparent conflict between a Kullback-Leibler divergence of $3.83 \times 10^{-8}$ and a cosine similarity of $0.9655$.
- Apparent conflict between a $0.22\text{ ms}$ rollback reset and a $28.5\text{ ms/token}$ decode latency.

### Resolution & Clarification
1. **Layer-0 vs. Layer-32 Separation:**
   - **Layer-0 Output Projection:** Comparing the Rust SSM kernel against reference GGUF outputs immediately after the `tile` tensor layout fix yields **Cosine Similarity = 0.9996** and **KL Divergence = $3.83 \times 10^{-8}$**.
   - **Layer-32 Final Output Logits:** After 32 layers of accumulated numerical computation and quantization format divergence (Rust GGUF Q4_K_M vs. PyTorch bitsandbytes NF4), the global logit **Cosine Similarity is 0.9655** (with top-10 logit overlap of 9/10 and max absolute logit difference of 3.24 across unnormalized logit ranges of $\pm 30$).
   - Table 2 in `SODALS_arXiv_Paper.md` now explicitly labels the scope of each metric.
2. **Rollback Latency Decomposition:**
   - $T_{\text{host}} = 0.22\text{ ms}$: Pure CPU-side orchestrator memory pointer and buffer truncation (`clear_kv_cache()`).
   - $T_{\text{GPU}} = 12.5\text{--}46.0\text{ ms}$: GPU-accelerated parallel chunk prefill over the retained prefix ($L \le 512$).
   - **Total End-to-End Rollback:** $12.7\text{--}46.2\text{ ms}$ (equivalent to 1–2 autoregressive decode steps).
   - Backtracking is invoked in only **7.5% of cases** (9 out of 120), preserving high aggregate throughput.

---

## 5. Algorithmic Claims & Terminology Accuracy

### Problem Identified in Audit
- Relational graph attention lookup was inaccurately notated as $O(N \log d)$, confusing sequence length, head dimension, and node degree.
- The search mechanism was labeled as "CDCL", implying full SAT-solver resolution.

### Resolution
1. **Complexity Notation:** Corrected to $O(N \log \text{deg}_{\max})$ per query row (total work $O(N^2 \log \text{deg}_{\max})$ across sequence length $N$), where $\text{deg}_{\max}$ is the maximum equipment node degree ($\text{deg}_{\max} \ll N$), completely decoupled from head dimension $d_k = 128$.
2. **Terminology Reframing:** CDCL terminology has been replaced with:
   - *Conflict-Directed Backtracking with Nogood Recording (Nogood Cache)*.
   - The paper explicitly notes that the system records conflict clauses as nogood patterns in a hash-indexed cache to prune invalid branches without executing full SAT unit propagation or 2-watched-literal data structures.

---

## 6. Mathematical Invertibility of Recurrent Delta-Rule

### Problem Identified in Audit
Earlier text claimed that the SSM delta-rule state update is "mathematically irreversible."

### Resolution
The mathematical formulation in Sections 3.5 and 6.2 has been corrected:
$$\mathbf{S}_t = \mathbf{S}_{t-1} \text{diag}(\exp(\mathbf{g}_t)) (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^T) + \beta_t \mathbf{v}_t \mathbf{k}_t^T$$
- **Exact Arithmetic:** The update is an affine transformation and is formally invertible via the Sherman-Morrison rank-1 formula whenever $(1 - \beta_t \|\mathbf{k}_t\|^2) \neq 0$.
- **Finite Precision (FP16/BF16):** Round-off errors amplify exponentially after a few backward steps, making numerical inversion ill-conditioned in practice. This justifies EdgeSSM's token prefix re-prefill approach, which achieves exact state synchronization without numerical drift.

---

## 7. Benchmark Provenance & Reference Alignment

### Problem Identified in Audit
- Conflation of IBM's *AssetOpsBench* with *FailureSensorIQ*.
- Citation of AssetOpsBench with incomplete author lists.
- Uncited leftover references in the bibliography.

### Resolution
1. **Clear Benchmark Delineation (Section 4.1.2):**
   - **AssetOpsBench ($N=120$ Diagnostic Subset):** Patel, Lin, Rayfield, Zhou, Shyalika, Yarrabothula, Vaculin, Martinez, O'Donncha, Kalagnanam (KDD 2026). 141 operational queries across 4 asset classes; closed-form diagnostic evaluation.
   - **FailureSensorIQ:** Constantinides, Patel, Lin, Guerrero, Patil, Kalagnanam (NeurIPS 2025). 8,296 ISO-derived multiple-choice QA items across 10 asset classes.
2. **Bibliographic Alignment:**
   - Paper 3 (`SODALS_arXiv_Paper.md`): Perfectly aligned 35-reference list. Every citation `[1..35]` in the text corresponds to a unique entry in the references section, with zero dead citations.
   - Paper 1 (`arXiv_SODALS_v2.1_Master_Paper.md` & `paper1_master_reasoning.tex`): Perfectly aligned 15-reference list (including Belrose et al., 2023 for the Tuned Lens) with 100% in-text citation coverage.

---

## 8. Step 3 (Modules 1, 2, and 3) Unified-Grader Clean 125 Held-Out Audit & Confident Hallucination Discovery

Following completion of all 150 questions on the GPU for **Step 3 — Module 3 (`sft/eval_grpo_long_tuned_lens_150.py`)**, we conducted a strict retrospective audit (`sft/audit_step3_all_modules_clean_125.py` and `sft/STEP3_MODULES_1_2_3_CLEAN_125_AUDIT_REPORT.md`) across all five stages using **identical evaluators** (both `Legacy Grader` and `Normalized Symbolic Grader`) on the strictly isolated **Clean 125 Held-Out** set (`indices 26..150`) and **25 Training Pilot** (`indices 1..25`):

| Stage / Model | Clean 125 Held-Out (Legacy Grader) | Clean 125 Held-Out (Normalized Symbolic) | 25 Training Pilot (Legacy / NormSym) | Combined 150 (Legacy / NormSym) | Net Delta vs. SFT (Clean 125) |
|---|---|---|---|---|---|
| **0. SFT Soft-Blended Baseline** | **99 / 125 (79.20%)** | **103 / 125 (82.40%)** | 23 / 25 (92.00%) | 122/150 / 126/150 | Reference |
| **1. 1536-Token GRPO Long** | **100 / 125 (80.00%)** | **104 / 125 (83.20%)** | 23 / 25 (92.00%) | 123/150 / 127/150 | **+1 (+0.80%, Q105, $p=1.0000$)** |
| **2. Step 3 — Module 1 (Pondering ODE)** | **100 / 125 (80.00%)** | **104 / 125 (83.20%)** | 23 / 25 (92.00%) | 123/150 / 127/150 | **+1 (100% Parity with GRPO)** |
| **3. Step 3 — Module 2 (CST PRM Gate)** | **99 / 125 (79.20%)** | **103 / 125 (82.40%)** | 23 / 25 (92.00%) | 122/150 / 126/150 | **0 (-1 Regression on Q055)** |
| **4. Step 3 — Module 3 (Tuned Lens + Tri-Modal PRM)** | **100 / 125 (80.00%)** | **104 / 125 (83.20%)** | 23 / 25 (92.00%) | 123/150 / 127/150 | **+1 (Q055 Restored, 0 Drops)** |

### Key Forensic Findings Disclosed in Paper 1 (Section 3.10)
1. **Elimination of Grader Drift:** The $+4$ question difference between Legacy and Normalized Symbolic metrics on Clean 125 (`Q032`: `11\sqrt{2}` vs. `11\sqrt2`; `Q054`: `-\pi/6` vs. `-\frac{\pi}{6}`; `Q098`: `East` vs. `\text{east}`; `Q128`: `4210_5` vs. `4210_{5}`) is already present in Pass 1 across **all five models** and is explicitly disclosed as a grader normalization effect rather than a model reasoning gain.
2. **Correction of Module 2's Q055 False-Positive Regression by Module 3:** Module 2 regressed by $-1$ problem on `Q055` (`0.15` PASS overwritten with `0` FAIL due to a naive regex fraction split). Module 3's refined parser and Tuned Lens geometric stability gate ($\bar{S}_{\mathrm{Q055}} = 0.7534 > 0.65$) eliminated this false positive, restoring `0.15` (`PASS`) and preserving **100% of passing chains (`6/6`)** across all 9 triggered held-out problems (`0` regressions).
3. **Confident Hallucination Under Greedy ($T=0$) Decoding:** Across Clean 125, Tuned Lens probing (Layers 28 & 30 $\to$ Layer 31) measured mean cosine stability $\bar{S} = 0.7775 \pm 0.0191$ ($\min = 0.7266$), answer stability $0.6705 \pm 0.0282$, and Top-1 vocabulary agreement **124 / 125 (99.20%)**. Failed greedy derivations ($n=21$) exhibited slightly **higher** inter-layer stability (**0.7867**) than passing derivations ($n=104$, **0.7756**), explaining why single deterministic retries yield 0 net new solved problems without multi-branch sampling ($G > 1$).

---

## 9. Summary Table of Audit Resolutions

| Audit Category | Previous State | Resolved State | Verification Artifact |
|---|---|---|---|
| **Domain Decoupling** | Same fractions in 2 papers | 0 math in Paper 3, 0 CPS in Paper 1 | Automated regex cross-scan |
| **GRPO Data Leakage** | 25 pilot in 150Q test | Segregated 125 held-out + 25 pilot | `sft/math500_heldout_clean_125_dataset.json` |
| **Statistical Truth** | "SFT ceiling shattered" | Disclosed McNemar $p=1.0000$ | `sft/CLEAN_125_HELDOUT_AUDIT_REPORT.md` |
| **Step 3 Grader Drift** | Mixed Legacy (123) vs NormSym (126) | Unified 5-stage grading + Q055/Tuned Lens audit | `sft/STEP3_MODULES_1_2_3_CLEAN_125_AUDIT_REPORT.md` |
| **Q105 Token Count** | Speculative 218/844 tokens | Confirmed 1,023 tokens (GRPO) vs FAIL (SFT) | `sft/math500_heldout_clean_125_results.json` |
| **KL vs. Cosine** | Unexplained divergence | Stratified: Layer-0 ($3.83 \times 10^{-8}$) vs. Layer-32 ($0.9655$) | `SODALS_arXiv_Paper.md` Table 2 |
| **Rollback Latency** | 0.22ms vs 28.5ms conflict | $T_{\text{host}}=0.22\text{ms}$, $T_{\text{GPU}}=12.5\text{--}46.0\text{ms}$ ($12.7\text{--}46.2\text{ms}$ total) | `SODALS_arXiv_Paper.md` Section 3.5 |
| **Graph Complexity** | $O(N \log d)$ | $O(N \log \text{deg}_{\max})$, $\text{deg}_{\max} \ll N$ | `SODALS_arXiv_Paper.md` Section 3.3 |
| **Search Naming** | Full CDCL | Conflict-directed backtracking with Nogood Cache | `SODALS_arXiv_Paper.md` Section 3.5 |
| **SSM Invertibility** | "Mathematically impossible" | Affine/invertible in exact math; ill-conditioned in FP16 | `SODALS_arXiv_Paper.md` Section 6.2 |
| **Benchmark Provenance**| Conflated IBM corpora | Delineated AssetOpsBench ($N=120$) vs. FailureSensorIQ | `SODALS_arXiv_Paper.md` Section 4.1.2 |
| **References** | Dead/misaligned citations | 100% verified 1-to-1 matching across all papers | Reference validation test script |

---

## 10. Next Steps for External Re-Audit

This dossier, alongside the segregated datasets and updated paper drafts, is now ready for formal re-submission to the independent academic reviewer. The portfolio maintains complete transparency, reproducible evidence, and defensible scientific humility.
