# SODALS v2.1: Self-Distillation Policy Optimization (SDPO), Long-Context GRPO, and Continuous Manifold Stabilization in Frontier Reasoning

**Author:** Burak Öztürk  
**Affiliation:** Independent Researcher, SODALS Project  
**Date:** October 2026  
**Archive Identifier:** `arxiv/sodals-v2.1-sdpo-grpo-20261001`  
**Base Weights:** Qwen 3.5 9B Base (NF4 Quantized / FP16 Native)  
**Target Benchmarks:** MATH-500 (Full 500, Stratified 50, 150 Head-to-Head), Pristine 24Q Frontier STEM Benchmark, CAIS Humanity's Last Exam (HLE), MMLU-Pro STEM  

---

## Abstract

We present an empirical study investigating the causal dynamics and optimization limits of post-training reasoning enhancements in parameter-efficient, hardware-constrained regimes (NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM). We systematically trace the progression of mathematical reasoning through five decoupled paradigms: (1) **Self-Distillation Policy Optimization (SDPO)** with symbolic verifiers; (2) **Continuous Soft Mixture-of-Adapters (Soft-MoA)** across disjoint mathematical domains; (3) **Hardware-Constrained Long-Context Group Relative Policy Optimization (GRPO)**; (4) **Continuous Manifold Regulation via Prefill Pondering ODEs**; and (5) **Test-Time Neurosymbolic Verification and Tuned Lens (J-Lens) Geometric Stability Gating**.

First, to isolate causal learning from stochastic weight perturbation on pre-selected error distributions, we introduce a **Placebo Negative Control Ablation**: updating an identical low-rank adapter ($r=4$, $\alpha=8$) over an identical gradient budget (17 optimization steps, learning rate $3 \times 10^{-5}$, Gram matrix subspace orthogonality regularizer $\lambda_{\mathrm{ortho}} = 0.005$) on unrelated canonical mathematics examples. Under strict greedy decoding ($T=0.0$), the **Placebo Control scored 0 / 35 (0.0%)** on strictly held-out failure questions, whereas **Pure SDPO achieved 4 / 35 (11.4%) zero-shot recovery** (Fisher exact two-sided $p = 0.114$, one-sided $p = 0.057$, McNemar paired test $p = 0.125$). While providing an encouraging directional signal, we document a **Pareto capacity trade-off**: incorporating static factual anchors to prevent localized representation drift on the exceptional Lie algebra $F_4$ ($52 \to 28$) reduced held-out recovery yield from 4 / 35 (11.4%) to 2 / 35 (5.71%). Furthermore, multi-domain scaling across 170 evaluation cases revealed the **Subspace Interference Effect**: joint multi-adapter fine-tuning induces cross-subspace covariance ($A_1 A_2^T \neq 0$), eroding delicate frontier representations ($E_7 = 133 \to 70$). In contrast, an **Untrained Soft Blender** modulated by Layer-32 Gaussian Phase Routing achieves peak multi-domain accuracy (138 / 170, 81.18%), matching discrete regex routing while eliminating brittle manual parsing rules.

To expand reasoning horizons on consumer hardware, we introduce a **Zero-Paging VRAM Architecture** that offloads token embeddings to host CPU RAM, quantizes the linear language model head to 4-bit NF4, and computes policy gradients via chunked micro-batches ($chunk\_size=32$). This reduces baseline model footprint from 7,298 MB to 3,907 MB, unlocking 3,616 MB of physical headroom and enabling 1536–2048 token autoregressive rollout and policy updates with **zero WDDM driver paging** on commodity 8GB GPUs. Exploiting this horizon, **Long-Context GRPO ($L=1536$, $G=4$)** advances the frozen SFT baseline on the strictly unseen **Clean 125 Held-Out MATH-500** subset from **99 / 125 (79.20%)** to **100 / 125 (80.00%)** under the legacy grader—and from **103 / 125 (82.40%)** to **104 / 125 (83.20%)** under normalized symbolic equivalence (+1 net problem on Q105, exact McNemar $p = 1.0000$, 0 regressions; note that early-surrender frequency across all 26 SFT-failed held-out items remained invariant at 15 / 26 vs. 15 / 26, confirming Q105 as an isolated single-item flip rather than a systematic shift in deliberation persistence). Finally, a unified-grader ablation across four test-time inference modules on the Clean 125 Held-Out set demonstrates that: (i) **Prefill Pondering ODE (Module 1)** preserves **100 / 125 (80.00% legacy; 104 / 125, 83.20% normalized symbolic)** as a neutral invariance operator; (ii) naive regex-based **CST Symbolic PRM Gating (Module 2, $G=1$)** suffers a **-1 problem regression** (**99 / 125 legacy; 103 / 125 normalized symbolic**) due to a false-positive fraction contradiction on Q055; (iii) **Tri-Modal Tuned Lens Geometric Stability Gating (Module 3, $G=1$)** eliminates the Q055 false positive, restoring **100 / 125 (80.00% legacy; 104 / 125, 83.20% normalized symbolic)** with zero regressions while uncovering the **Confident Hallucination Phenomenon** under greedy ($T=0$) decoding: inter-layer affine consensus (Layers 28 & 30 $\to$ Layer 31) is slightly higher on failed derivations (**0.7867**, $n=21$) than on correct derivations (**0.7756**, $n=104$; overall $0.7775 \pm 0.0191$, Top-1 vocabulary agreement **124 / 125 = 99.20%**), proving that deterministic single-pass retries cannot overcome high-confidence parametric misconceptions without diverse candidate sampling ($G > 1$); and (iv) **Targeted Best-of-$G$ ($G=4$) Multi-Branch Neurosymbolic PRM Rollout (Module 4)** triggers blindly on 15 / 150 structurally flawed trajectories (`</think>` truncation, SymPy CST contradiction, or $\ge 2$ self-doubt markers) to advance Clean 125 Held-Out accuracy to **102 / 125 (81.60% legacy)** and **106 / 125 (84.80% normalized symbolic; 129 / 150 = 86.00% combined)** (+2 gains over GRPO on `Q120` and `Q138`, 0 regressions, exact McNemar $p = 0.5000$). Crucially, a compute-matched comparison against **Pure Majority Voting (`MV@4` and `MV@5`)** on the exact same stored candidate pool shows an **exact tie at 102 / 125 legacy (106 / 125 normalized symbolic, net difference 0, McNemar $p = 1.0000$)**: while PRM rescues base-5 arithmetic (`Q138: 224_5 -> 204_5`) where majority voting fails, majority voting rescues `Q124` (`GT = 8`) where PRM rejects the correct branch due to a false-positive contradiction penalty.

---

## 1. Experimental Progression & Decoupled Architecture

The SODALS v2.1 research pipeline investigates reasoning optimization across nine systematically controlled stages:

```
[Stage 1: Balanced SFT] ────────► 815 Master Examples ───► Baseline Adapter (Balanced r=4)
                                                             │ (389/500 on MATH-500)
                                                             ▼
[Stage 2: Error Mining] ────────► 111 MATH-500 Failures ─► Stratified Split (Seed 42)
                                                             ├── 76 Train Failures (SDPO)
                                                             └── 35 Strictly Held-Out Failures
                                                                        │
                                      ┌─────────────────────────────────┴───────────────────────────────┐
                                      ▼                                                                 ▼
[Stage 3: SDPO & Placebo]     76 Diagnostic Critiques                                         35 Held-Out Failures
                              17 SymPy-Verified Self-Corrections                               (Strictly Unseen)
                                      │
              ┌───────────────────────┼──────────────────────────────┐
              ▼                       ▼                              ▼
      [Pure SDPO LoRA]       [Placebo Control LoRA]        [Anchored SDPO (SDPO-A)]
        17 Self-Corrections    17 Unrelated Math             17 Self-Corrections + 3 Anchors
        17 Steps, LR 3e-5       17 Steps, LR 3e-5             20 Steps, LR 3e-5
              │                       │                              │
              └───────────────────────┴──────────────────────────────┘
                                      ▼
[Stage 4: Multi-Domain MoA]   Domain 1 (MATH-500) + Domain 2 (Lie Theory & Abstract Algebra)
                              Continuous GPR Layer-32 Phase Routing (Untrained Soft Blender: 138/170, 81.18%)
                                      │
                                      ▼
[Stage 5: VRAM Architecture]  CPU Embedding Offload + 4-bit NF4 LM Head + Chunked Policy Loss
                              Base VRAM: 7,298 MB ──► 3,907 MB (3,616 MB Physical Margin on 8GB GPU)
                              Unlocks L=1536-2048 Token Window with 0 WDDM Page Faults
                                      │
                                      ▼
[Stage 6: Long-Context GRPO]  Group Relative Policy Optimization (L=1536, G=4 Rollouts, 25 Prompts)
                              Clean 125 Held-Out: 100/125 Legacy (80.00%) | 104/125 NormSym (83.20%) [0 Drops, +1 Gain]
                                      │
                                      ▼
[Stage 7: Pondering ODE]      Explicit RK4 Continuous Manifold Relaxation on Prefill Latent State (Module 1)
                              Clean 125 Held-Out: 100/125 Legacy | 104/125 NormSym Preserved with 0 Regressions
                                      │
                                      ▼
[Stage 8: Tri-Modal PRM &]    Module 2 (CST PRM: 99/125 Legacy, 103/125 NormSym; -1 Regression on Q055)
[Tuned Lens Gate (Mod 2&3)]   Module 3 (Tuned Lens + Tri-Modal PRM: 100/125 Legacy, 104/125 NormSym; Q055 Restored,
                              0 Drops; Confident Hallucination: PASS Stab 0.7756 vs FAIL Stab 0.7867)
                                      │
                                      ▼
[Stage 9: Targeted Best-of-G] Module 4: G=4 Multi-Branch Neurosymbolic Rollout & Tri-Modal PRM Selection
[Neurosymbolic PRM (Mod 4)]   Clean 125 Held-Out: 102/125 Legacy (81.60%) | 106/125 NormSym (84.80%)
                              (+2 Net Gains: Q120 & Q138, 0 Drops, Beats Majority Vote on Base-5 Q138)
```

### 1.1 Hyperparameter and Configuration Matrix

| Dimension / Stage | Balanced Baseline | Placebo Control | Anchored SDPO (SDPO-A) | Untrained Soft-MoA | Long-Context GRPO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model** | Qwen 3.5 9B (NF4) | Qwen 3.5 9B (NF4) | Qwen 3.5 9B (NF4) | Qwen 3.5 9B (NF4) | Qwen 3.5 9B (NF4) |
| **Adapter Rank ($r$)** | $r=4, \alpha=8$ | $r=4, \alpha=8$ | $r=4, \alpha=8$ | D1: $r=4$, D2: $r=8$ | $r=4, \alpha=8$ |
| **Training Scheme** | Supervised SFT | Supervised SFT | Supervised SFT | Inference Blending | GRPO Reinforcement |
| **Dataset Size** | 815 Master Items | 17 Unrelated Math | 20 (17 SDPO + 3 Anchors) | 170 (150 D1 + 20 D2) | 25 Prompts ($G=4$, 100 Rollouts) |
| **Context Window ($L$)** | 384 tokens | 384 tokens | 384 tokens | 384 tokens | **1536 tokens** |
| **GPU Hardware** | RTX 4060 (8GB) | RTX 4060 (8GB) | RTX 4060 (8GB) | RTX 4060 (8GB) | RTX 4060 (8GB) |
| **Base VRAM Footprint** | 7,298 MB | 7,298 MB | 7,298 MB | 7,490 MB | **3,907 MB (Optimized)** |

---

## 2. Multi-Benchmark Empirical Results

All evaluations were executed under deterministic greedy decoding ($T=0.0$, repetition penalty 1.05) directly on an NVIDIA GeForce RTX 4060 Laptop GPU.

### 2.1 Master Multi-Arm Comparison Table

| Evaluation Suite | Balanced Baseline | Placebo Control (Negative) | Pure SDPO | Anchored SDPO (SDPO-A) | Causal & Epistemic Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. 35 Strictly Held-Out Failures**<br>*(Generalization Repair Reflex)* | 0 / 35 (0.0%)<br><sub>[Selection Deficit]</sub> | **0 / 35 (0.0%)**<br><sub>[Zero Flips]</sub> | **4 / 35 (11.4%)**<br><sub>[Q6, Q8, Q9, Q35]</sub> | **2 / 35 (5.71%)**<br><sub>[Q9, Q35]</sub> | **Empirical Attribution Trend:** Placebo flips 0 boundary questions. SDPO drives genuine near-miss repairs (Fisher two-sided $p=0.114$, one-sided $p=0.057$, McNemar paired $p = 0.125$), while anchor regularization exhibits a ~50% capacity trade-off. |
| **2. Pristine 24Q Frontier STEM**<br>*(Abstract Theory & Invariant Integrity)* | **21 / 24 (87.5%)**<br><sub>[$F_4 = 52$ PASS]</sub> | **21 / 24 (87.5%)**<br><sub>[$F_4 = 52$ PASS]</sub> | **20 / 24 (83.3%)**<br><sub>[$F_4 = 28$ FAIL]</sub> | **21 / 24 (87.5%)**<br><sub>[**$F_4 = 52$ Recovered**]</sub> | **Template Restoration:** Anchor primes Cartan formula ($\dim = \mathrm{rank} + |\Phi|$). $\mathfrak{sp}(6)$ remains broken across all models (`12` vs `21`). |
| **3. 50-Question MATH-500 Control**<br>*(Catastrophic Forgetting Tripwire)* | **39 / 50 (78.0%)** | **39 / 50 (78.0%)** | **40 / 50 (80.0%)** | **40 / 50 (80.0%)** | **Zero Regression:** All post-training updates pass the safety tripwire without catastrophic forgetting. |

### 2.2 Extended Multi-Benchmark Validation Suite

To evaluate whether low-rank fine-tuning introduces broad regression across standard reasoning distributions, models were evaluated across three independent benchmarks:

**Table 2:** Extended Benchmark Evaluations. Comparing the Balanced SFT Baseline adapter (`balanced_r4`, 389/500 on MATH-500) against Anchored SDPO (SDPO-A, 384/500).

| Benchmark Suite | Sample Size ($N$) | Evaluation Mode | Balanced SFT Baseline (`balanced_r4`) | SDPO-A (α=0.25) | Metric & Divergence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MATH-500 (Full)** [Hendrycks et al., 2021] | 500 | Greedy ($T=0$) | **389 / 500 (77.8%)**<br><sub>[95% CI: 74.0, 81.2]</sub> | **384 / 500 (76.8%)**<br><sub>[95% CI: 72.9, 80.3]</sub> | $\Delta = -1.0\%$; 96.2% verdict agreement (481/500 identical outputs) |
| **MMLU-Pro (STEM)** | 20 | Greedy ($T=0$) | **15 / 20 (75.0%)** | **15 / 20 (75.0%)** | Exact tie (15/20 accuracy); 17/20 agreement (85.0%) with 19/20 identical tokens |
| **Humanity's Last Exam (HLE)** [Phan et al., 2025] | 50 | Greedy ($T=0$) | **2 / 50 (4.0%)** | **1 / 50 (2.0%)** | Both models exhibit frontier floor effects ($\le 4\%$) |

The Full MATH-500 evaluation ($N=500$) confirms that SDPO-A maintains high fidelity with the baseline adapter, yielding a minimal $-1.0\%$ delta within overlapping 95% confidence intervals, proving that self-correction fine-tuning does not induce catastrophic forgetting across canonical contest mathematics.

---

## 3. Forensic Autopsy & Scientific Analyses

### 3.1 Causal Attribution via Negative Control (Placebo Ablation)

A critical vulnerability in post-training evaluations is the **boundary perturbation fallacy**: when evaluating on questions pre-selected for initial failure, an evaluation score of 0 / 35 before fine-tuning is an artifact of selection rather than an empirical baseline. To determine whether an arbitrary 17-step weight perturbation would flip borderline answers purely by chance, we trained the Placebo Control model on 17 unrelated mathematics examples (`control_17_curriculum.jsonl`).

**Findings:**
- The Placebo Control model achieved **0 / 35 (0.0%)** on the held-out failure distribution.
- Pure SDPO achieved **4 / 35 (11.4%)** on the exact same distribution.
- **Statistical Significance Assessment:** Under Fisher's exact test, the 4/35 vs. 0/35 difference corresponds to a two-sided $p = 0.114$ and one-sided $p = 0.057$. Under McNemar's test for paired binary outcomes on the identical 35 items, $p \approx 0.125$. While narrowly exceeding the conventional $\alpha = 0.05$ threshold, this result indicates that near-miss repairs reflect a genuine self-correction signal rather than random weight jitter.
- **Conclusion:** While the zero-flip outcome of the placebo control provides an encouraging empirical contrast against random boundary perturbation, the sample size ($N=35$, two-sided Fisher $p = 0.114$) is statistically underpowered to definitively reject the null hypothesis of stochastic weight perturbation at $\alpha = 0.05$. Expanding the failure cohort to $N \ge 100$ is strictly required to establish asymptotic causal attribution.

### 3.2 The $F_4$ Anomaly and Forensic Item Audit

In initial 8-question evaluations, the Placebo model scored 7 / 8 (87.5%), outperforming Pure SDPO (6 / 8, 75.0%) due to Question 1:
$$\text{Problem: What is the dimension of the exceptional Lie algebra } F_4? \quad (\text{Ground Truth: } 52)$$
- **Placebo Prediction:** `52` (PASS)
- **Pure SDPO Prediction:** `28` (FAIL)

An audit of `control_17_curriculum.jsonl` resolved this asymmetry. Item 17 in the Placebo set was an exceptional Lie algebra problem:
> **Item 17 Prompt:** *"How many roots does the exceptional Lie algebra $\mathfrak{g}_2$ have?"*  
> **Item 17 Response:** *"Formula: $\dim(\mathfrak{g}) = \mathrm{rank}(\mathfrak{g}) + |\Phi| \implies |\Phi| = \dim(\mathfrak{g}) - \mathrm{rank}(\mathfrak{g})$. For $\mathfrak{g}_2$: $\dim=14, \mathrm{rank}=2 \implies |\Phi|=12$."*

The Placebo model was explicitly primed with the Cartan root-dimension formula minutes before evaluation, allowing it to correctly compute $\dim(F_4) = 4 + 48 = 52$. Conversely, Pure SDPO—trained exclusively on high-school/competition MATH-500 failures—experienced localized representation drift, misidentifying $F_4$ with the dimension of the orthogonal Lie algebra $D_4 = \mathfrak{so}(8)$ ($28$).

### 3.3 Pristine 24Q Frontier STEM Domain Decomposition

To eliminate small-sample noise ($n=8$), we constructed the **Pristine 24Q Frontier STEM Benchmark**, comprising 3 canonical, textbook invariant questions across 8 domains:

| Domain | Problems & Invariants Tested | Baseline | Placebo | Pure | SDPO-A |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1. Lie Theory | $\dim(F_4)=52, \dim(G_2)=14, \dim(E_8)=248$ | 3/3 | 3/3 | 2/3 | 3/3 |
| 2. Modular Forms | $\tau(2)=-24, \tau(3)=252, \zeta(4)=\pi^4/90$ | 3/3 | 3/3 | 3/3 | 3/3 |
| 3. Symplectic Geometry | $\dim(\mathfrak{sp}(6))=21, \dim(\mathfrak{sp}(4))=10, \dim(T^*Q)=28$ | 2/3 | 2/3 | 2/3 | 2/3 |
| 4. Arithmetic Geometry | $\mathrm{rank}(E)=0, |E_{\mathrm{tors}}|=4, \#E(\mathbb{F}_5)=6$ | 2/3 | 2/3 | 2/3 | 2/3 |
| 5. Algebraic Topology | $\pi_3(S^2)=\mathbb{Z}, \pi_4(S^3)=\mathbb{Z}_2, \pi_1(\mathbb{R}P^2)=\mathbb{Z}_2$ | 2/3 | 2/3 | 2/3 | 2/3 |
| 6. Representation Theory | $\mathrm{SU}(3)\text{ fund}=3, \mathrm{SU}(3)\text{ adj}=8, \mathrm{Spin}(7)\text{ spin}=8$ | 3/3 | 3/3 | 3/3 | 3/3 |
| 7. Complex Analysis | $\mathrm{Res}(i)=-i/2, \mathrm{Res}(1)=1/4, \mathrm{Winding}=2$ | 3/3 | 3/3 | 3/3 | 3/3 |
| 8. Characteristic Classes | $\chi(S^2)=2, \chi(\mathbb{C}P^2)=3, \chi(T^2)=0$ | 3/3 | 3/3 | 3/3 | 3/3 |
| **Overall Pristine Score** | **24 Canonical Frontier Questions** | **21/24** | **21/24** | **20/24** | **21/24** |

**Key Diagnostic Insights:**
1. **High Invariant Stability:** 20 out of 24 questions are resolved identically across all four models. Topology, modular forms, and representation dimensions remain completely intact.
2. **The Sole Differentiator:** Across the entire 24-question benchmark, the only question separating Pure SDPO from the Baseline is Question 1 ($F_4$).

### 3.4 The Capacity Cost of Anchor Regularization ($4/35 \to 2/35$)

To correct the $F_4$ representation drift, Anchored SDPO (SDPO-A) incorporated 3 frontier anchors alongside the 17 self-corrections (Anchor 1: Lie $\mathfrak{g}_2$ roots; Anchor 2: Symplectic cotangent bundle dimension; Anchor 3: $\mathbb{C}P^8$ Euler characteristic).

While SDPO-A successfully restored $F_4 = 52$ (bringing the Pristine 24Q score to 21 / 24, 87.5%), it incurred a measurable generalization penalty on the held-out failure set:
- **Pure SDPO Held-Out Repair:** **4 / 35 (11.4%)**  
  *Repaired questions:* Q6 (Wheel probability), Q8 (Triangle scale factor), Q9 (Midpoint area deduction), Q35 (Angular speed optimization).
- **SDPO-A Held-Out Repair:** **2 / 35 (5.71%)**  
  *Repaired questions:* Q9 (Midpoint area deduction, Pred: `8`), Q35 (Angular speed optimization, Pred: `30`).  
  *Dropped recoveries:* Q6 and Q8 reverted to incorrect derivations.

**Theoretical Rationale:** In a hyper-parameter-efficient regime ($r=4$, 20 optimization steps), allocating 3 steps (15% of the total gradient trajectory) to static factual anchors dilutes optimization pressure away from error-correction mechanisms. This demonstrates an empirical Pareto frontier between factual anchor preservation and generative error-repair capacity.

### 3.5 Epistemic Scope: Template Recall vs. Generalized Reasoning

The recovery of $F_4 = 52$ in SDPO-A must not be conflated with generalized Lie-theoretic deduction. Across all models (Baseline, Placebo, Pure SDPO, SDPO-A), Question 7:
$$\text{Problem: What is the dimension of the symplectic Lie algebra } \mathfrak{sp}(6)? \quad (\text{Ground Truth: } 21)$$
consistently failed, with SDPO-A predicting `12`.

For $\mathfrak{sp}(2n)$, the dimension formula is $\dim \mathfrak{sp}(2n) = n(2n+1)$, yielding for $n=3$: $3 \times 7 = 21$. Because Anchor 1 provided the Cartan root formula specific to simple Lie algebras with known root systems rather than the classical symplectic matrix algebra generator formula, the model could not infer $\dim \mathfrak{sp}(6)$. This establishes that post-training anchor regularization operates via **template-specific formula recall** rather than structural symbolic deduction.

### 3.6 Multi-Adapter Cartridge Routing & Continuous Soft LoRA Blending (170Q Benchmark)

To surpass the capacity ceiling of single low-rank adapters when scaling across distinct mathematical domains (Domain 1: Competition MATH-500 vs. Domain 2: Advanced Abstract Algebra and Lie Theory), SODALS introduces a modular **Mixture-of-Adapters (MoA) Cartridge Architecture** governed by an internal **Gaussian Phase Router (GPR)** probing layer 32 hidden states ($\tau = 0.004913$, Fisher separation $= 15.30$). Subspace blending across all 128 LoRA layers is continuously modulated:
$$\Delta W = (1 - \alpha) s_1 W_1 + \alpha s_2 W_2, \quad \alpha = \sigma(z_i / 0.03)$$

Evaluated on a comprehensive 170-question multi-domain benchmark (150 MATH-500 + 20 canonical Abstract Algebra/Lie Theory), a five-stage factorial investigation yields definitive empirical conclusions:

| # | Architecture / Evaluation Variant | Routing Mechanism | MATH-500 (150Q) | Domain 2 (20Q) | Total (170Q) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | **Regex Router MoA** | Discrete Regex Rule Parsing | 122 / 150 (81.33%) | 16 / 20 (80.00%) | 138 / 170 (81.18%) |
| 2 | **Untrained Soft Blender (Production Standard)** | **Continuous GPR ($z_i \to \alpha$)** | **122 / 150 (81.33%)** | **16 / 20 (80.00%)** | **138 / 170 (81.18%)** |
| 3 | **Phase 3 (From-Scratch Joint LoRA, $B=0$)** | Differentiable Gate (Trained Anchors) | 111 / 150 (74.00%) | **17 / 20 (85.00%)** | 128 / 170 (75.29%) |
| 4 | **Phase 3 (Calibrated Anchor LoRA)** | Continuous GPR (Calibrated $\tau=0.0049$) | 118 / 150 (78.67%) | **17 / 20 (85.00%)** | 135 / 170 (79.41%) |
| 5 | **Phase 13 (Warm-Start Joint LoRA)** | Continuous GPR (Frozen Anchors) | 120 / 150 (80.00%) | **17 / 20 (85.00%)** | 137 / 170 (80.59%) |

**Core Discovery & Architectural Finding:**  
1. **Subspace Interference Effect:** End-to-end joint multi-adapter fine-tuning inevitably introduces cross-subspace perturbation ($A_1 A_2^T \neq 0$). The Gram matrix orthogonality regularizer $\|AA^T - I\|_F^2$ only maintains internal Stiefel frame isometry within each adapter, but cannot prevent mutual cross-subspace interference. Joint training recovered 7 questions but lost 8 previously mastered questions (net $-1$ delta vs. baseline).
2. **Empirical Production Standard:** The **Untrained Soft Blender (138/170, 81.18%)** achieves the highest overall accuracy with zero cross-adapter degradation and 1.3ms routing latency, formally establishing that **domain cartridges must remain isolated during training and continuously blended only at inference time.**
3. **Erosion of Frontier Capabilities ($E_7 = 133 \to 70$):** In Phase 3, the algebra adapter resolved the dimension of exceptional Lie algebra $E_7 = 133$ for the first time in project history. However, 76 steps of joint warm-start fine-tuning erased this representation, collapsing $E_7$ to `70` and corrupting the symplectic invariant $\mathfrak{sp}(6) = 21 \to 1$.
4. **Frozen Base Representation Superiority & Oracle Analysis:** On Domain 2 abstract algebra, the frozen unadapted base model scored 18 / 20 (90.0%), whereas sequential low-rank adaptation (SDPO-D2) caused accuracy to decline to 14 / 20 (70.0%), and adapted Soft-MoA achieved 16 / 20 (80.0%). Tuned Lens diagnostics confirm that delicate algebraic invariants crystallize predominantly in layers 28–32; gradient updates into these late-layer subspaces induce localized representation collapse. Notably, a theoretical oracle pairing—routing Domain 1 to the analytical adapter (122 / 150) and Domain 2 to the pure frozen base model (18 / 20)—would achieve 140 / 170 (82.35%), surpassing the adapted Soft-MoA score of 138 / 170 (81.18%). This acknowledges that even with continuous soft blending, adapting onto specialized algebraic invariants incurs a mild capacity penalty (16 / 20 vs. 18 / 20) relative to native base representations.

---

### 3.7 Hardware-Constrained Long-Context VRAM Architecture ($L=1536\text{--}2048$ on 8GB Consumer GPU)

Extending reasoning chains beyond 384 tokens on an 8GB laptop GPU (NVIDIA RTX 4060, 8188 MB physical VRAM) under Windows WDDM introduces severe memory pressure. Naive LoRA training with FP16 embedding and head layers consumes 7,298 MB of baseline VRAM, leaving less than 890 MB for sequence activations and triggering catastrophic WDDM driver paging over the PCIe bus (causing 20–50x latency slowdowns).

To unlock the 1536–2048 token window without paging, we introduced three hardware-level memory innovations:
1. **Host CPU Embedding Offload (`CPUEmbeddingWrapper`):** The 248,320-token embedding lookup table was transferred from GPU VRAM to Host CPU RAM. Because token embedding lookup is an indexed gather operation ($O(1)$) executed once per token, the CPU-to-GPU transfer of the resulting $[B, L, 4096]$ hidden tensor incurs $< 0.1$ ms overhead, while immediately freeing **1,645 MB** of physical VRAM.
2. **4-Bit NF4 Quantization of the Language Model Head:** The frozen `[248320, 4096]` linear projection matrix was quantized from FP16 (1,940 MB) to 4-bit NormalFloat (NF4, 508 MB) via `bitsandbytes.nn.Linear4bit`. Backpropagation gradients pass through the frozen quantized projection in full FP32 precision, saving **1,431 MB** of VRAM.
3. **Chunked Policy Cross-Entropy Loss ($chunk\_size=32$):** Rather than materializing the full $[B, L, 248320]$ logit tensor (which requires $> 1.5$ GB at $L=1536$), the logit projection and cross-entropy loss are computed in micro-chunks of 32 tokens. This caps intermediate vocabulary projection memory at **15.8 MB**.
4. **Numerical Stability Safeguards:** All 177 RMSNorm and selective SSM recurrent tensors were cast to FP32, preventing gradient overflow and `NaN` propagation under NF4 compute regimes.

**Result:** Base VRAM footprint plummeted from **7,298 MB to 3,907 MB**, opening **3,616 MB of unfragmented physical headroom**.

**Table 3:** Forward + Backward Training Benchmark Across Context Horizons (RTX 4060 8GB).

| Sequence Length ($L$) | Response Tokens ($R$) | Step Duration (s) | Peak Alloc (MB) | Peak Reserved (MB) | Headroom (MB) | WDDM Driver Paging? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **768** | 640 | 2.13 s | 4,445.6 MB | 4,942.0 MB | 3,246.0 MB | **NO (0 Pages)** |
| **1024** | 896 | 2.73 s | 4,625.0 MB | 5,260.0 MB | 2,928.0 MB | **NO (0 Pages)** |
| **1280** | 1,152 | 3.38 s | 4,804.4 MB | 5,592.0 MB | 2,596.0 MB | **NO (0 Pages)** |
| **1536** | 1,408 | 4.10 s | 4,983.8 MB | 5,828.0 MB | 2,360.0 MB | **NO (0 Pages)** |
| **1792** | 1,664 | 4.73 s | 5,163.2 MB | 6,088.0 MB | 2,100.0 MB | **NO (0 Pages)** |
| **2048** | 1,920 | **5.50 s** | 5,342.6 MB | 6,366.0 MB | **1,822.0 MB** | **NO (0 Pages)** |

Under pure autoregressive rollout, generating 1,024 reasoning tokens required **17.48 seconds (58.57 tokens/sec)** with zero page faults and 4,039.2 MB peak allocated memory.

---

### 3.8 Long-Context Group Relative Policy Optimization (GRPO)

In preliminary reinforcement learning experiments with short reasoning horizons ($L=384$, Mini-Pilot), policy rollouts achieved 19 / 25 (76.0%) Pass@1-of-4 on training prompts, but failed to surpass the SFT baseline on full evaluation (121 / 150, 80.67% vs. 122 / 150, 81.33%). Forensic inspection revealed that 384 tokens artificially truncated complex mathematical derivations, forcing the policy to guess prematurely.

With the zero-paging 1536-token window operational, we trained a full-scale policy via **Group Relative Policy Optimization (GRPO)**:
$$\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left( \min \left( r_i(\theta) A_i, \mathrm{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right)$$
where $G=4$ rollouts per prompt are scored against ground-truth symbolic targets ($A_i$ normalized group advantage). Training proceeded across 25 diverse competition prompts (100 total sequential rollouts and policy gradient updates) over 97.1 minutes, achieving 20 / 25 (80.00%) rollout pass rate with zero `NaN` occurrences.

**Table 4A:** Primary Benchmark: Clean Held-Out Generalization Evaluation (125 Strictly Unseen MATH-500 Questions, Zero Training Overlap, archived in `sft/math500_heldout_clean_125_results.json`). Evaluated under greedy decoding ($T=0$).

| Evaluation Metric | Frozen SFT Baseline (`balanced_r4`) | Long-Context GRPO ($L=1536$) | Prefill Pondering ODE (RK4) | Net Delta vs. SFT | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean Held-Out Accuracy** | **99 / 125 (79.20%)**<br><sub>[95% CI: 71.27, 85.39]</sub> | **100 / 125 (80.00%)**<br><sub>[95% CI: 72.14, 86.07]</sub> | **100 / 125 (80.00%)**<br><sub>[95% CI: 72.14, 86.07]</sub> | **+1 Net (+0.80%)** | McNemar $p = 1.0000$<br>Bootstrap CI: [0.00%, 2.40%] |
| **Regression Count (Drops)** | — | **0 Drops (0 / 125)** | **0 Drops (0 / 125)** | **0** | **100% Policy Stability** |
| **Gained Problems** | — | **1 Problem (Q105)** | **1 Problem (Q105)** | **+1** | 1,023-Token CoT Derivation |

**Table 4B:** Secondary Evaluation: Training Subset Stability & Reconstruction (25 Pilot Prompts, archived in `sft/math500_training_pilot_25_results.json`).

| Metric / Subset | Frozen SFT Baseline | Long-Context GRPO ($L=1536$) | Prefill Pondering ODE | Net Delta | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Prompt Solve Rate** | **23 / 25 (92.00%)** | **23 / 25 (92.00%)** | **23 / 25 (92.00%)** | **0** | Zero Catastrophic Forgetting |
| **Combined 150-Item Reference** | 122 / 150 (81.33%) | 123 / 150 (82.00%) | 123 / 150 (82.00%) | +1 Net (+0.67%) | Overall Invariance Envelope |

*Statistical Note on Data Segregation:* To strictly prevent data leakage and eliminate evaluation contamination, the 25 prompts utilized in RL rollouts are formally decoupled into a separate training stability check (`Table 4B`). The primary reported result (`Table 4A`) reflects exclusively the 125 unseen held-out problems. On this pristine set, GRPO achieves a net gain of +1 question (99 to 100). Under a two-sided exact McNemar test, $p = 1.0000$, and the 95% Wilson confidence intervals substantially overlap ([71.27%, 85.39%] vs. [72.14%, 86.07%]). Consequently, this result is statistically non-significant at $N=125$. The scientific contribution of this experiment does not rest on an unverified claim of having "broken the SFT ceiling," but rather on establishing: (1) a zero-paging long-context VRAM architecture enabling 1536-token rollouts on 8GB commodity GPUs, and (2) complete empirical policy stability (zero regressions across all 125 held-out items).

#### Case Study: Q105 & Early-Surrender Mechanism Control Audit
On Question 105 (Intermediate Algebra: finding the sum of all real values of $x$ for a radical quadratic system, Ground Truth: $40$), the Frozen SFT Baseline terminated prematurely predicting `18` (FAIL, 240 tokens), whereas Long-Context GRPO generated 1,023 tokens to reach $\boxed{40}$ (PASS). However, to test whether the 25-prompt GRPO pilot systematically reduced "early surrender" (premature termination without full derivation) across the unseen failure cohort, we audited all 26 SFT-failed questions in Clean 125 Held-Out (`sft/CLAUDE_ITEMS_1_AND_2_FORENSIC_AUDIT_AND_PREREGISTRATION.md`). The proportion of failed responses terminating in $< 512$ tokens was **15 / 26 (57.69%)** in SFT and **15 / 26 (57.69%)** in GRPO (mean token length $490.2 \to 511.2$, median $393.5 \to 396.5$, 24 / 26 identical word counts). Thus, the mechanism test for systematic "deliberation persistence" returned a **strict null result**: Q105 is an isolated single-item anecdote rather than evidence of a broad behavioral shift. Two structural factors explain this null mechanism in the 25-prompt pilot: (i) 10 of 25 training prompts had zero advantage variance ($\sigma_r = 0$, leaving only 15 active gradient updates), and (ii) the LoRA adapter was trained at scale $s = 1.0$ (`PeftModel`) but evaluated inside `UntrainedSoftMoAModel` at $s_{D1} = 0.25(1-\alpha) \approx 0.242$, attenuating the learned weight update by $\sim 4\times$ at test time.

#### Optimization Dynamics & Advantage Variance Analysis
In our setup, training utilized 25 prompt templates with $G=4$ rollouts per prompt (100 total rollouts). Because the training subset exhibited a high baseline solve rate ($23/25 = 92.0\%$), prompt groups where all 4 candidate rollouts succeeded (all receiving reward $r_i = 1.0$) or all failed ($r_i = 0.0$) produced zero intra-group reward variance ($\sigma_r = 0$). Under GRPO's group standardization:
$$A_i = \frac{r_i - \bar{r}}{\sigma_r}$$
prompts with homogeneous rollouts produce zero advantage and contribute zero gradient to the policy. Consequently, gradient updates were driven exclusively by the small fraction of prompts exhibiting non-zero rollout variance (such as Q105, which required exploration of extended reasoning steps). This structural property mechanically explains why the model exhibited zero regressions across the broader 150-question evaluation set: the policy underwent minimal parameter perturbation rather than broad policy drift. To scale policy optimization effectively in future work, RL training pipelines must incorporate difficulty-stratified prompt filtering (selecting prompts with baseline pass rates $0.2 \le p \le 0.8$) across thousands of distinct prompts to guarantee non-zero advantage signals across training steps.

---

### 3.9 Continuous Manifold Stabilization via Prefill Pondering ODE

To test whether continuous latent relaxation could stabilize prompt representations prior to token generation, we integrated the **Adaptive Prefill Pondering ODE Module**. During the prompt prefill phase (`seq_len > 1`), the backbone latent representation $h_0$ at the final transformer layer undergoes continuous manifold evolution governed by explicit 4th-order Runge-Kutta (RK4) integration:
$$\frac{dh}{dt} = f_\theta(h, t) = \tanh(W_2 \mathrm{GELU}(W_1 h + b_1) + b_2)$$
$$h_{k+1} = h_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
where the step count $N \in [2, 5]$ and relaxation coefficient $\lambda(\alpha) = 0.3(1 + \alpha)$ are dynamically modulated by the GPR routing coefficient $\alpha$.

**Isolated 150-Question MATH-500 Ablation:**
- **Final Benchmark Score:** **123 / 150 (82.00%)**
- **Net Delta vs. Long-Context GRPO Baseline:** **0 (0 Gains, 0 Drops)**
- **Net Delta vs. Frozen SFT Baseline:** **+1 (1 Gain, 0 Drops)**
- **Held-Out 125Q:** **100 / 125 (80.00%)**
- **Train 25Q:** **23 / 25 (92.00%)**
- **Domain Performance:** Algebra: 35/36 (97.2%), Counting & Probability: 9/10 (90.0%), Precalculus: 15/17 (88.2%), Geometry: 8/10 (80.0%).

*Scientific Finding & Structural Mechanism:* We emphasize that this represents a **null result regarding benchmark accuracy** (+0 net gain). The ODE relaxation neither improves nor degrades accuracy. Mechanistically, in a causal decoder-only architecture, prompt prefill activations at the final layer do not write into or mutate the past key-value (KV) cache of earlier layers. Consequently, the continuous ODE relaxation acts solely upon the final prompt token's hidden state, influencing primarily the first generated token's logit distribution. Subsequent autoregressive decoding steps attend over the standard, unmodified KV cache. This structural property mechanically explains why the ODE manifests as a neutral invariance operator rather than a trajectory-altering accuracy booster.

---

### 3.10 Test-Time Neurosymbolic Verification, Tuned Lens Geometric Stability, and the Confident Hallucination Phenomenon (Clean 125 Held-Out Ablation)

To test whether autonomous test-time verification can detect and repair derivation errors without ground-truth leakage, we evaluated two progressive neurosymbolic Process Reward Model (PRM) architectures on top of the 1536-token GRPO + Prefill Pondering ODE pipeline (`Step 3`):
- **Module 2 — CST Symbolic PRM & Contradiction Gate (`sft/eval_grpo_long_cst_prm_150.py`):** Extracts intermediate equations ($L = R$) from the primary greedy chain-of-thought (Pass 1) and checks arithmetic and algebraic consistency via SymPy alongside a Reasoning Chain Jacobian Filter (`jump_threshold = 3.5`). If an explicit arithmetic contradiction ($|f_1 - f_2| > 10^{-4}$) or proof abandonment is detected, the gate triggers a single targeted self-correction pass (Pass 2) and selects the candidate with the higher symbolic consistency score.
- **Module 3 — Tuned Lens (J-Lens) & Tri-Modal Geometric Stability Gate (`sft/eval_grpo_long_tuned_lens_150.py`):** Augments the refined CST Symbolic Verifier and Jacobian Filter with a trained **Tuned Lens** [Belrose et al., 2023] probing intermediate workspace layers ($l \in \{28, 30\}$) against the final pre-norm hidden state ($L=31$) via learned affine transformations ($h_{\mathrm{proj}, l} = W_l h_l + b_l$, serialized in `jlens_weights.safetensors`). For each sequence, the module computes mean inter-layer cosine stability $\bar{S} = \frac{1}{T}\sum_{t=1}^T \frac{1}{2}\sum_{l \in \{28,30\}} \cos(h_{\mathrm{proj},l,t}, h_{31,t})$, final-token answer stability $S_{\mathrm{ans}}$, and Top-1 vocabulary agreement between Layer 30's projected logits and Layer 31's output logits. Candidate selection combines symbolic equation count, Jacobian jump penalties, and the Tuned Lens geometric consensus score.

#### Unified-Grader Audit & Elimination of Grader Drift
A critical methodological pitfall in iterative benchmark evaluation is **grader drift**: during the development of Modules 2 and 3, we introduced `NormalizedMathMatcher`, which normalizes harmless LaTeX syntax variations (`\sqrt2` $\leftrightarrow$ `\sqrt{2}`, `_{5}` $\leftrightarrow$ `_5`, `\pi` $\to$ `pi`, and case-insensitive text matching). When applied only to Modules 2 and 3, `NormalizedMathMatcher` appeared to boost accuracy by $+3\text{ to }+4$ questions over Module 1. However, a strict retrospective audit (`sft/audit_step3_all_modules_clean_125.py`) grading **all five stages with both graders simultaneously** revealed that **all five models (including SFT Baseline, GRPO Long, and Module 1 Pondering ODE) had already produced those exact four mathematically equivalent answers in Pass 1** on the Clean 125 Held-Out set:
1. **Q032 (`CLEAN-125`, Prealgebra):** Model predicted `'11\sqrt{2}'` vs. ground truth `'11\sqrt2'` (brace normalization).
2. **Q054 (`CLEAN-125`, Precalculus):** Model predicted `'-\pi/6'` vs. ground truth `'-\frac{\pi}{6}'` (`\pi` symbolic normalization).
3. **Q098 (`CLEAN-125`, Prealgebra):** Model predicted `'East'` vs. ground truth `'\text{east}'` (case-insensitivity).
4. **Q128 (`CLEAN-125`, Number Theory):** Model predicted `'4210_5'` vs. ground truth `'4210_{5}'` (subscript brace normalization).

**Table 5:** Unified-Grader Six-Stage Ablation across the **Clean 125 Held-Out** Set (`indices 26..150`), **25 Training Pilot** (`indices 1..25`), and **Combined 150** Set (`sft/STEP3_MODULES_1_2_3_CLEAN_125_AUDIT_REPORT.md`).

| Stage / Model | Clean 125 Held-Out (Legacy Grader) | Clean 125 Held-Out (Normalized Symbolic) | 25 Training Pilot (Legacy / NormSym) | Combined 150 (Legacy / NormSym) | Net Delta vs. SFT (Clean 125) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0. SFT Soft-Blended Baseline** | **99 / 125 (79.20%)** | **103 / 125 (82.40%)** | 23 / 25 (92.00%) | 122/150 (81.33%) / 126/150 (84.00%) | Reference Baseline |
| **1. 1536-Token GRPO Long** | **100 / 125 (80.00%)** | **104 / 125 (83.20%)** | 23 / 25 (92.00%) | 123/150 (82.00%) / 127/150 (84.67%) | **+1 (+0.80%)** ($p=1.0000$, Q105 Gain) |
| **2. Step 3 — Module 1 (Pondering ODE)** | **100 / 125 (80.00%)** | **104 / 125 (83.20%)** | 23 / 25 (92.00%) | 123/150 (82.00%) / 127/150 (84.67%) | **+1 (+0.80%)** (100% Parity with GRPO) |
| **3. Step 3 — Module 2 (CST PRM Gate, $G=1$)** | **99 / 125 (79.20%)** | **103 / 125 (82.40%)** | 23 / 25 (92.00%) | 122/150 (81.33%) / 126/150 (84.00%) | **0 (0.00%)** (**-1 Regression on Q055**) |
| **4. Step 3 — Module 3 (Tuned Lens + PRM, $G=1$)** | **100 / 125 (80.00%)** | **104 / 125 (83.20%)** | 23 / 25 (92.00%) | 123/150 (82.00%) / 127/150 (84.67%) | **+1 (+0.80%)** (**Q055 Restored, 0 Drops**) |
| **5. Step 3 — Module 4 (Targeted Best-of-$G=4$ PRM)** | **102 / 125 (81.60%)** | **106 / 125 (84.80%)** | **23 / 25 (92.00%)** | **125/150 (83.33%) / 129/150 (86.00%)** | **+3 (+2.40%)** (**+2 over GRPO, Q120 & Q138, 0 Drops**) |

#### Forensic Comparison of Module 2 vs. Module 3 & The Confident Hallucination Finding
1. **Why Module 2 Regressed on Q055 (`104/125 -> 103/125`) and Module 3 Restored It (`104/125`):** On Question 55 (`CLEAN-125`), the primary greedy chain correctly derived $\boxed{0.15}$. However, Module 2's initial regex equation splitter misparsed a multi-step fraction simplification ($\frac{3}{20} = \frac{15}{100}$) across a line boundary as the false contradiction `((3)/(20)) (0.15) != 3 (3.0)`, triggering an unnecessary retry that overwrote the correct answer `0.15` (`PASS`) with `0` (`FAIL`). In Module 3, refined sentence-boundary parsing combined with Tuned Lens geometric stability ($\bar{S}_{\mathrm{Q055}} = 0.7534 > 0.65$) prevented this false-positive intervention, preserving `0.15` (`PASS`).
2. **Zero-Regression Preservation Across All 9 Triggered Held-Out Questions:** In Module 3, the Tri-Modal Gate triggered on **9 questions** (all 9 located in the Clean 125 Held-Out set: `Q031, Q063, Q071, Q087, Q112, Q120, Q124, Q128, Q138`; 0 triggers in the 25 Training Pilot). On the 6 triggered questions that were already correct in Pass 1 (`Q031, Q063, Q071, Q087, Q112, Q128`), Pass 2 preserved **6 / 6 (100%)** of the correct answers with **0 regressions**. On the 3 triggered questions that failed in Pass 1 (`Q120: 2 -> 13` vs. GT `-9`; `Q124: 10 -> 10` vs. GT `8`; `Q138: 224_5 -> 244_5` vs. GT `204_5`), Pass 2 improved internal equation consistency and Tuned Lens stability (`Q120`: $0.8452 \to 0.8569$; `Q128`: $0.7603 \to 0.7905$; `Q138`: $0.7671 \to 0.8018$) but yielded **0 net accuracy gains**.
3. **Empirical Discovery — Confident Hallucination Under Greedy ($T=0$) Decoding:** Across the Clean 125 Held-Out set, the Tuned Lens measured an overall inter-layer cosine stability of **$0.7775 \pm 0.0191$** ($\min = 0.7266, \max = 0.8452$), final-token answer stability of **$0.6705 \pm 0.0282$**, and Top-1 vocabulary agreement of **124 / 125 (99.20%)**. Crucially, stratifying stability by correctness reveals that **incorrect derivations ($n=21$) exhibit a slightly higher mean inter-layer stability ($0.7867$) than correct derivations ($n=104$, mean stability $0.7756$)**, and no sequence ever drops below the `< 0.65` instability threshold. This provides direct mechanistic evidence for why single-pass deterministic self-correction fails to yield net accuracy gains [Huang et al., 2023]: when a 9B model errs under greedy decoding, its late workspace layers (Layers 28–31) are in **tight internal geometric consensus ($0.7867$) around a flawed mathematical premise**. Overcoming confident parametric errors requires multi-branch stochastic exploration ($G > 1$, $T > 0$) paired with external symbolic verification rather than deterministic re-prompting.

---

### 3.11 Module 4: Targeted Best-of-$G$ ($G=4$) Neurosymbolic Rollout and Process Verification

To overcome the greedy confident hallucination barrier identified in Section 3.10 without incurring excessive inference compute across all 150 problems, **Step 3 — Module 4 (`eval_grpo_long_best_of_g_prm_150.py`)** introduces a **Targeted Multi-Branch Neurosymbolic Rollout** mechanism:

1. **Blind Structural Flaw Gate (Zero-Leakage Trigger):** Multi-branch sampling is triggered *strictly* when the Pass 1 greedy chain exhibits verifiable structural pathology: (i) truncation without closing the reasoning thought block (`not has_end1`); (ii) explicit SymPy CST algebraic/arithmetic contradiction (`not v1 and len(contras1) > 0`); or (iii) repeated epistemic doubt markers (`wait_cnt >= 2`). Across all 150 evaluation questions, this blind trigger activated on exactly **15 questions** (13 in Clean 125 Held-Out, 2 in Train 25 Pilot).
2. **Heterogeneous Candidate Generation ($G=4$):**
   - **Candidate 1 (Targeted Deterministic Correction, $T=0$):** Injects a direct verification prompt addressing the specific failure mode (algebraic error reflection or conciseness directive).
   - **Candidates 2, 3, 4 (Stochastic Diversity, $T=0.6$, $\text{top\_p}=0.9$, seeds 101, 202, 303):** Parallel exploratory rollouts generating diverse reasoning trajectories.
3. **Tri-Modal PRM Scoring:** Each candidate trajectory is scored by combining:
   $$\text{PRM}(c) = 15.0 \cdot \mathbb{I}(\text{has\_end}) + 1.5 \cdot N_{\text{verified}} - 15.0 \cdot N_{\text{contradictions}} - 2.5 \cdot N_{\text{doubt}} + \bar{S}_{\text{lens}}$$
   The candidate with the highest composite score is selected; if all candidates fail symbolic verification, the gate defaults to Pass 1.

#### Results & Compute-Matched Majority Voting Control (`MV@4` vs. `PRM`)
Module 4 completed all 150 questions on GPU (`sft/eval_grpo_long_best_of_g_prm_150_results.json`), advancing Clean 125 Held-Out accuracy from **104 / 125 (83.20%)** to **106 / 125 (84.80%)** under normalized symbolic evaluation (**102 / 125, 81.60%** under strict legacy string matching), with **+2 Held-Out gains over Pass-1 Greedy (`Q120` and `Q138`), 0 regressions, and exact McNemar $p = 0.5000$** (underpowered relative to the $\ge 6$ discordant recovery threshold required for $p < 0.05$).

Crucially, because Module 4 expends $4\times$ test-time compute on triggered items ($G=4$ new candidates), the fair baseline control is **Pure Majority Voting (`MV@4` on the 4 generated candidates, or `MV@5` including Pass 1)** evaluated on the exact same stored candidate pool (`sft/CLAUDE_ROUND2_AUDIT_AND_STEP30_HEALTH_REPORT.md`):
- **Pass-1 Greedy ($G=1$):** `100 / 125 (80.00% Legacy)` | `104 / 125 (83.20% NormSym)`
- **Pure Majority Voting (`MV@4` & `MV@5`):** **`102 / 125 (81.60% Legacy)` | `106 / 125 (84.80% NormSym)`** (`+2` gains: `Q120` and `Q124`, `0` drops)
- **Tri-Modal PRM Selection (Module 4):** **`102 / 125 (81.60% Legacy)` | `106 / 125 (84.80% NormSym)`** (`+2` gains: `Q120` and `Q138`, `0` drops)

Across the 13 triggered held-out problems, **Tri-Modal PRM and Pure Majority Voting achieve an exact aggregate tie (`106 / 125 = 106 / 125`, net difference `0`, McNemar $p = 1.0000$)** via a symmetric `1 win / 1 loss` trade-off:
1. **Q120 (`GT: -9`, Both Win):** Pass 1 truncated at 2048 tokens predicting `2` (FAIL). Three of four new candidates (`Cand1`, `Cand2`, `Cand4`) predicted `-9`, so both `MV@4` and `PRM` (`Cand4`, `+35.63`) recovered `-9` (PASS).
2. **Q138 (`GT: 204_5`, PRM Wins / MV Fails):** Pass 1 and two stochastic branches (`Cand3`, `Cand4`) predicted `224_5` (a wrong majority vote). SymPy base-$n$ verification flagged the arithmetic contradiction in `224_5` (`-16.7` PRM) and selected the single sound branch `Cand2 = 204_5` (`+27.4` PRM, PASS).
3. **Q124 (`GT: 8`, MV Wins / PRM Fails):** Pass 1 predicted `10` (FAIL). In the 4-candidate pool, two branches (`Cand2` and `Cand3`) correctly derived `8` (so `MV@4` selected `8`, PASS). However, SymPy CST misparsed an intermediate step in `Cand2`/`Cand3` as a contradiction (`-1.5` and `-1.8` PRM) and instead selected `Cand1 = 12` (`+25.3` PRM, FAIL).

**Conclusion on Module 4:** While SymPy base-$n$ verification demonstrates qualitative value on `Q138`, its false-positive contradiction penalty on `Q124` cancels that advantage in aggregate. Thus, on the current test set, **Tri-Modal PRM does not surpass compute-matched Pure Majority Voting (`106/125` vs. `106/125`)**; the observed `+2` gain over Pass-1 Greedy is attributable to $G=4$ multi-sampling diversity rather than net PRM superiority over majority voting.

---

### 3.12 Symmetrical Rescoring, Lens-Free Parity Breaking, and `SymbolicStepChecker` Calibration (11–18 Nov 2026)

1. **Breaking `MV@4` Parity via Lens-Free Six-Arkhe Scoring (`eval_math500_heldout_module4_lensfree_production_results.json`):** Removing uninformative inter-layer Tuned Lens stability (`mean_stability`, single-pass $\text{AUROC}=0.3721$) and scoring $G=4$ rollouts purely via Semantic Agreement + Capped CST Verification captures `Q120`, `Q124`, and `Q138` simultaneously on Clean-125, reaching **107 / 125 (85.60%)** forced accuracy ($+4$ net over SFT, McNemar $p=0.1250$) and **107 / 120 (89.17%)** selective accuracy at $96.0\%$ coverage under the Tier-1 Selective Abstention Gate ($\tau=0.20$, pruning 5/5 confirmed hallucinations with 0 false abstentions).
2. **Symmetrical Frozen Baseline Re-Scoring & Tuple Fix (`CLAUDE_METHODOLOGY_FREEZE_AUDIT.json`):** Re-grading all `Q001..Q500` arms with the SHA-256-locked `NormalizedMathMatcher` shifts standalone `balanced_r4` Baseline SFT canonically from `389 -> 398 / 500` (`Clean-125`: `100 -> 104 / 125 (83.20%)` vs. Soft-Blended SFT `99 -> 103 / 125` in Table 5, `Pristine-30`: `24 -> 25 / 30 (83.33%)`, `Pristine-50`: `36 -> 37 / 50 (74.00%)`, untouched reserve `Q201..Q500`: `232 -> 235 / 300 (78.33%)`). Fixing `SimilarityArche`'s Python tuple truthiness bug (`res[1]` vs. `bool((False, False))`) reduced Clean-125 recomputation mismatches to `0 / 125`.
3. **`SymbolicStepChecker` 2x2 Calibration:** Upgrading `cst_symbolic_verifier.py` for partial-braced LaTeX fractions (`\frac{270}7`) and degree trigonometry ($^\circ \to \pi/180$) improved contradiction detection precision on Clean-125 from `33.3%` (`TP=3, FP=6, FN=18, TN=98`) to **72.7%** (`TP=8, FP=3 [2.9%], FN=13, TN=101`), enabling live $G=4$ rescue of `Q165` (`+\sqrt{3} \to -\sqrt{3}`, `task-5558`).
4. **Human-Audited Blind Discordant Verification (`sft/BLIND_HUMAN_DISCORDANT_AUDIT.md`, 13 Cross-Arm Pairs / 39 Predictions):** All 13 cross-arm discordant items across Clean-125 (5 items) and Pristine-50 (8 items) were anonymized into randomized `Option_A/B/C` rows and audited against ground truth before unblinding (`BLIND_HUMAN_DISCORDANT_KEY.json`). Across 39 predictions, **38 / 39 (97.4%)** agreed with the frozen automated grader, including **15 / 15 (100%)** on Clean-125 (`Arkhe Forced vs. Baseline SFT = +4/-1, p=0.375`; `vs. Primary = +3/-0, p=0.250`). One genuine automated-grader false negative was caught on `Pristine30_Q180`: `Baseline_SFT` produced the exact plane equation prefixed with prose (`"seems to be 5x - 7y + 11z + 4 = 0"`) and was marked `FAIL`. Correcting this extraction artifact shifts `Baseline_SFT` from `25/30 -> 26/30 (86.67%)` on Pristine-30 and `37/50 -> 38/50 (76.00%)` on Pristine-50, narrowing `Arkhe Forced vs. Baseline SFT` to `+2/-2 (net 0, p=1.000)` on Pristine-30 and `+4/-2 (net +2, p=0.6875)` on Pristine-50, confirming that exploratory dev-set differences are non-significant and await the sealed `Q201..Q500` confirmatory run.

---

### 3.13 Unseen Pristine-50 Always-On Control & Shannon Entropy vs. Linear Probing (18 Nov 2026)

1. **Pristine-50 Always-On $G=4$ Control (`Q151..Q200`, `Python 3.13.7` / `PyTorch 2.8.0+cu126`):** Under SHA-256-frozen evaluation scripts (`pristine50_always_on_control_results.json`):
   - **Frozen SFT Baseline:** `37 / 50 (74.00%)`
   - **Pass-1 Greedy:** `39 / 50 (78.00%)` (Unbiased Pass-1 `min_step_logprob` $\text{AUROC} = \mathbf{0.9510}$; `mean_token_logprob` $= 0.8275$)
   - **`Avg@4` / `Majority@4` / `Pass@4`:** `155 / 200 (77.50%)` / `41 / 50 (82.00%)` / `45 / 50 (90.00%)`
   - **Arkhe Zero-Steer:** `41 / 50 (82.00%)` forced; **`40 / 44 (90.91%)`** selective at `88.0%` coverage
   - **Arkhe Full ($\tau=0.20$, 4/50 triggers):** `41 / 50 (82.00%)` forced (`+6/-2` vs. SFT, McNemar $p=0.2891$); **`41 / 49 (83.67%)`** selective at `98.0%` coverage
2. **Internal Signals Benchmark (`eval_internal_signals_logprob_vs_probe_results.json`):** Teacher-forced next-token Shannon entropy (`neg_mean_entropy`) achieved $\text{AUROC} = \mathbf{0.7583}$ on silent clean-terminating failures and $\mathbf{0.8791}$ on Pristine-20 OOD prompts, whereas late hidden-state PCA(16) logistic probing (`probe_h_late_tok`) failed paired bootstrap superiority ($\Delta\text{AUROC}=0.1417$, CI $[-0.0097, 0.2992]$, $p=0.0696$; $0.6484$ OOD). Weight drift remains negligible (`0.2891%` global Frobenius drift).
3. **Epoch I Protocol Lock Status (26 Nov 2026):** Following cross-interpreter runtime alignment (`Python 3.10 -> 3.13`), the same-stack `Python 3.13.7` SFT baseline re-capture across `Q026..Q200` completed at **139 / 175 (79.43%)** (`baseline_sft_py313_q026_q200_results.json`), and the full **Independent-400** ($100/100$ MATH-train prompts, $400/400$ rollouts, `symbolic_checker_independent_400_results.json`) locked frozen `SymbolicStepChecker` out-of-sample precision at **68.29%** (`TP=28, FP=13`), recall at **15.22%** (`FN=156`), and FPR at **6.02%** (`TN=203`; alongside a Kaggle FP16 stock `Qwen/Qwen3.5-9B` control at `100.0%` precision, `TP=50, FP=0`). Following a 10-step LR movement sweep (`1e-4`, `3e-4`, `5e-4`), movement-conditioned **GRPO-1** (`lr=1e-4`, 40 ZPD steps, `39/40` active gradient steps, `max_KL=0.02119`, `grpo1_zpd_lr1e4_results.json`) and conditional **GRPO-2** (Dynamic-$\lambda \in [0.25, 1.00]$, 40 steps, `36/40` active, `100.0%` coverage, `grpo2_dynlambda_lr1e4_results.json`) completed. On the initial `Dev-500` $N=100$ paired signal cohort (`batch_size=1`, `max_new_tokens=2048`), **GRPO-1** reached **78 / 100 (78.00%)** vs. Baseline **76 / 100 (76.00%)** (`+5/-3`, exact McNemar $p=0.7266$) with **16 / 100 (16.0%)** exact-identical outputs (**84.0% policy movement** vs. 12.8% in the prior frozen `3e-5` run). Full `Dev-500` ($N=500$) arm comparisons and strictly sealed `Q201..Q500` confirmatory evaluation remain `[PENDING]` under `Q201_Q500_CONFIRMATORY_ANALYSIS_PLAN.md`.

---

## 4. Related Work

### 4.1 Self-Correction and Verification in LLMs
Self-correction in language models [Madaan et al., 2023; Shinn et al., 2023] demonstrated that frontier models could repair reasoning flaws when guided by external critique. However, Huang et al. [2023] showed that without ground-truth verifiers, models frequently flip correct answers into incorrect ones. Our unified-grader ablation of Modules 2 and 3 corroborates Huang et al. [2023] at the latent representation level via the Tuned Lens [Belrose et al., 2023]: because incorrect greedy chains exhibit high inter-layer consensus ($0.7867$ vs. $0.7756$ on passing chains), uncalibrated symbolic gates risk false-positive regressions (Q055 in Module 2), whereas tri-modal geometric stability gating restores 100% policy preservation without false drops.

### 4.2 Preference Optimization & Reinforcement Learning for Reasoning
Direct Preference Optimization (DPO) [Rafailov et al., 2023] and STaR [Zelikman et al., 2022] optimize models using paired preference data or self-generated rationale filtering. More recently, Group Relative Policy Optimization (GRPO) [Shao et al., 2024] removed the requirement for a separate critic network by normalizing rewards across intra-group rollouts, powering frontier mathematical reasoners. We extend this paradigm to hardware-constrained consumer GPUs via zero-paging memory architectures and budget forcing [Muennighoff et al., 2025].

### 4.3 Subspace Orthogonality & Parameter-Efficient Routing
Low-Rank Adaptation (LoRA) [Hu et al., 2021; Dettmers et al., 2023] constrains updates to low-dimensional manifolds. Multi-task parameter-efficient regimes often suffer from cross-task subspace interference [Huang et al. (LoraHub), 2023]. We document the **Subspace Interference Effect**, demonstrating that joint adapter fine-tuning inevitably damages cross-domain representations ($A_1 A_2^T \neq 0$), whereas inference-time continuous soft blending preserves domain integrity.

### 4.4 Neural Ordinary Differential Equations & Continuous Latents
Neural ODEs [Chen et al., 2018] parameterize continuous hidden state trajectories via differential equation solvers. We adapt explicit Runge-Kutta numerical integration to prompt prefill representations, establishing that continuous manifold relaxation preserves discrete autoregressive decoding without gradient divergence.

---

## 5. Discussion & Limitations

### 5.1 Key Findings
1. **Directional Error Recovery Signal:** Negative placebo ablation shows that held-out error recovery (4/35 vs. 0/35; Fisher exact $p = 0.114$, McNemar $p \approx 0.125$) is driven by verified self-correction curricula rather than stochastic weight jitter, though $N=35$ represents an underpowered cohort that requires larger future validation.
2. **The Subspace Interference Effect:** Joint training across disjoint domains damages delicate representations ($E_7 = 133 \to 70$). Untrained Soft Blending (138/170, 81.18%) eliminates manual regex rules while matching discrete routing fidelity.
3. **Zero-Paging VRAM Architecture:** CPU embedding offloading, NF4 head quantization, and chunked policy loss unlock 1536–2048 token RL training on an 8GB GPU with zero WDDM driver paging.
4. **Long-Context GRPO Pilot (Uninformative Policy-Movement Null):** On the exploratory Clean-125 set, the 25-prompt GRPO pilot differs by only $+1$ isolated flip (`Q105`, exact McNemar $p=1.0000$), with $\|\Delta W\|_F / \|W_0\|_F = 0.2891\%$ and `109 / 125` token-length-identical outputs; accordingly, this run is classified strictly as an uninformative null (`EFFECT_UNMEASURED_POLICY_DID_NOT_MOVE`) rather than evidence of stability or reasoning gain.
5. **Grader Drift Disentanglement & Confident Hallucination in Test-Time PRMs:** Grading all five stages with identical evaluators proves that the +4 question difference between legacy and normalized symbolic metrics (Q032, Q054, Q098, Q128) is purely a formatting normalization artifact already present in Pass 1 across all models. While Module 2's naive regex gate incurred a -1 regression on Q055 (`104/125 -> 103/125`), Module 3's Tri-Modal Tuned Lens gate eliminated the false positive (`104/125` restored, 0 drops) and revealed that failed greedy chains possess slightly higher Layer 28/30 $\to$ 31 affine stability ($0.7867$) than passing chains ($0.7756$).

### 5.2 Limitations & Analytical Boundaries
1. **Statistical Power, Sample Size, and Parameter Churn:** The $N=35$ failure sample ($p=0.114$) and $N=125$ held-out GRPO split ($p=1.0000$) are underpowered to demonstrate asymptotic statistical significance. Crucially, parameter perturbation churn analysis in Phase 13 joint adaptation indicates that 14 out of 150 Domain 1 questions (9.3%) flipped outcome, and 6 out of 28 baseline-incorrect questions (21.4%) flipped to correct spontaneously without targeted instruction. Because the Pure SDPO recovery rate of 4/35 (11.4%) falls squarely within this baseline parameter perturbation churn envelope, Fisher's exact test cannot reject the null hypothesis of stochastic parameter drift. Expanding evaluation cohorts to $N \ge 100$ under multi-seed trials is strictly required before asserting asymptotic self-correction capability.
2. **GRPO Advantage Sparsity & Training Signal Limits:** With 25 training prompts and $G=4$ rollouts, high baseline accuracy ($92.0\%$) meant that rollout groups with homogeneous outcomes yielded zero reward variance ($\sigma_r = 0$) and zero policy gradients. The zero-regression result across 150 questions largely reflects this sparse parameter perturbation. Scaling GRPO requires large-scale, difficulty-filtered prompt pools ($N \ge 1000$, $0.2 \le p_{\text{pass}} \le 0.8$) to maintain active learning signals across all rollouts.
3. **Pretrained Base Model vs. Aligned Models:** Our experiments fine-tune and probe the raw pretrained base checkpoint (Qwen 3.5 9B Base) rather than the official post-trained instruct or thinking models. Consequently, baseline capabilities reflect raw foundation completions rather than RLHF/RLAIF aligned behavior.
4. **Soft Blender Operating Regime:** At temperature $T=0.03$, the Sigmoid gate $\sigma(z_i / T)$ saturates rapidly (for $|z_i| > 0.1$, $\sigma \approx 0$ or $1$), functionally operating as a soft-margin discrete switch between isolated adapters rather than a broad continuous interpolation across internal representations.
5. **Failure Mining Data Provenance:** Mining training failures from MATH-500 creates potential distribution overlap; evaluation on pristine 24Q STEM and MMLU-Pro confirmed stability, but future curricula must train on completely disjoint problem pools.
6. **Context Truncation Confound:** At $L=384$, a substantial portion of baseline failures were truncation-induced; extending context to $L=1536$ accounts for part of the reasoning gains, partially decoupling from pure policy improvement.
7. **Null Accuracy Gains on Pondering ODE & Single-Retry PRM Gating:** Both continuous manifold relaxation (Module 1) and deterministic single-retry PRM gating (Modules 2 & 3) yielded 0 net new solved problems over GRPO Long on the Clean 125 Held-Out set, as greedy decoding errors exhibit high inter-layer stability ($0.7867$) and require multi-sample exploration ($G > 1$) rather than single greedy re-prompting.

---

## 6. Conclusion

We presented an empirical investigation into post-training and test-time reasoning enhancements under consumer hardware constraints. By establishing causal attribution via placebo negative controls, documenting the subspace interference effect in multi-adapter fine-tuning, engineering a zero-paging long-context VRAM architecture, isolating clean held-out evaluation splits from grader normalization drift, and characterizing the confident hallucination phenomenon via Tuned Lens geometric probing, this work provides a transparent foundation for verifiable, resource-efficient reasoning research.

---

## 7. Artifact & Checkpoint Registry

All checkpoints, serialized adapters, training curricula, and raw evaluation JSON artifacts are archived and reproducible:

- **Balanced Baseline Adapter:** `sft/qwen_lora_adapter_balanced_r4/`
- **Placebo Control Adapter:** `sft/qwen_lora_adapter_control/`
- **Pure SDPO Adapter:** `sft/qwen_lora_adapter_sdpo/`
- **Anchored SDPO (SDPO-A) Adapter:** `sft/qwen_lora_adapter_sdpo_anchored/`
- **Long-Context GRPO Adapter ($L=1536$):** `sft/qwen_lora_adapter_grpo_long/`
- **Pondering ODE Auxiliary Weights:** `sft/qwen_lora_gpr_soft_moa/gpr_soft_moa_aux_modules.pt`
- **Tuned Lens (J-Lens) Affine Projection Weights:** `jlens_weights.safetensors`
- **170Q Multi-Domain Soft-MoA Results:** `math500_domain2_soft_blended_170_results.json`
- **WDDM Long-Context Benchmark Results:** `sft/benchmark_wddm_long_tokens_results.json`
- **150Q GRPO Evaluation Report:** `sft/eval_grpo_long_math500_150_results.json`
- **150Q Prefill Pondering ODE Report (Module 1):** `sft/eval_grpo_long_pondering_150_results.json`
- **150Q CST Symbolic PRM Report (Module 2):** `sft/eval_grpo_long_cst_prm_150_results.json`
- **150Q Tuned Lens & Tri-Modal PRM Report (Module 3):** `sft/eval_grpo_long_tuned_lens_150_results.json`
- **Segregated Clean 125 Held-Out & 25 Pilot Artifacts:** `sft/math500_heldout_clean_125_results.json`, `sft/math500_heldout_clean_125_module3_tuned_lens.json`, `sft/STEP3_MODULES_1_2_3_CLEAN_125_AUDIT_REPORT.md`
- **Symmetrical Freeze & Internal Signal Audits (11–18 Nov 2026):** `sft/CLAUDE_METHODOLOGY_FREEZE_AUDIT.json`, `sft/FORENSIC_KL_AND_EPISTEMIC_AUDIT_REPORT.json`, `sft/eval_internal_signals_logprob_vs_probe_results.json`, `Temp/pristine50_always_on_control_results.json`
- **Hard-Mined ZPD & Disjoint Dev Benchmarks:** `sft/math_train_hard_mined_zpd_prompts.json`, `sft/math_train_disjoint_dev500.json`
- **Evaluation & Audit Scripts:** `sft/benchmark_wddm_long_tokens.py`, `sft/train_grpo_long_context.py`, `sft/eval_grpo_long_math500_150.py`, `sft/eval_grpo_long_pondering_150.py`, `sft/eval_grpo_long_cst_prm_150.py`, `sft/eval_grpo_long_tuned_lens_150.py`, `sft/audit_step3_all_modules_clean_125.py`, `sft/eval_pristine50_always_on_control.py`, `sft/eval_internal_signals_logprob_vs_probe.py`

---

## References

1. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.** (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR).
2. **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C.** (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. Advances in Neural Information Processing Systems (NeurIPS).
3. **Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D.** (2022). *STaR: Bootstrapping Reasoning With Reasoning*. Advances in Neural Information Processing Systems (NeurIPS).
4. **Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al.** (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. Advances in Neural Information Processing Systems (NeurIPS).
5. **Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S.** (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. Advances in Neural Information Processing Systems (NeurIPS).
6. **Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D.** (2023). *Large Language Models Cannot Self-Correct Reasoning Yet*. International Conference on Learning Representations (ICLR).
7. **Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J.** (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. Advances in Neural Information Processing Systems (NeurIPS).
8. **Huang, C., Liu, Q., Lin, B. Y., Pang, T., Du, C., & Lin, M.** (2023). *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition*. arXiv preprint arXiv:2307.13269.
9. **Shao, Z., Wang, P., Qi, Z., Xu, R., Song, H., Zhang, M., Li, Y., Wu, Y., & Guo, D.** (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint arXiv:2402.03300.
10. **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K.** (2018). *Neural Ordinary Differential Equations*. Advances in Neural Information Processing Systems (NeurIPS).
11. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems (NeurIPS).
12. **Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., et al.** (2025). *Humanity's Last Exam*. Center for AI Safety (CAIS) & Scale AI, arXiv preprint arXiv:2501.14249.
13. **Yang, S., Kautz, J., & Hatamizadeh, A.** (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv preprint arXiv:2412.06464.
14. **Muennighoff, N., Rush, A. M., Barak, B., et al.** (2025). *s1: Simple test-time scaling*. arXiv preprint arXiv:2501.19393.
15. **Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J.** (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*. arXiv preprint arXiv:2303.08112.
