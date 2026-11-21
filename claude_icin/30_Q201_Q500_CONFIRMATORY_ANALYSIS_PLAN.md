# Pre-Registered Confirmatory Analysis Plan: Sealed `Q201..Q500` (`N=300`) Reserve

**Registration Date:** 2026-11-21  
**Author:** Burak Öztürk (SODALS AI Research)  
**Status:** LOCKED BEFORE ANY `Q201..Q500` MODEL EVALUATION  

---

## 1. Frozen Evaluator Integrity (Epoch I SHA-256 Lock)

No confirmatory run on `Q201..Q500` is valid unless the following evaluator files match these exact SHA-256 hashes at execution time:

- `sft/cst_symbolic_verifier.py`: `ab421629b7d5838d574be3992ea8ee2a4c5139f0cce14104bdce15d8728af422`
- `sft/arkhe_neurosymbolic_pipeline.py`: `78ad3212df1745443ce6ed0e8f677bb85494ba47049b0c0c2f886036e5322cf5`

All arms (`SFT Baseline`, `GRPO-1`, `GRPO-2`, `Arkhe Forced`, `Arkhe Selective`) MUST be executed on the same runtime stack (`Python 3.13` / same `torch`, `transformers`, `peft`, `bitsandbytes` versions, or unified Kaggle FP16 `T4 x2` stack) and graded by the single frozen `NormalizedMathMatcher`.

---

## 2. Mandatory Pre-Unsealing Gates (Must Complete on `Dev-500` / `MATH-Train` First)

`Q201..Q500` (`N=300`) remains strictly sealed until ALL of the following are completed and recorded:

1. **Independent-400 (`400/400` rollouts):** Full out-of-sample 2x2 validation of frozen `SymbolicStepChecker` (`Precision`, `Recall`, `FPR`) completed (currently running in Kaggle `v2`).
2. **Disjoint `Dev-500` (`sft/math_train_disjoint_dev500.json`, 0% overlap with MATH-500):**
   - `Dev-500` Baseline (`balanced_r4`) completed.
   - **Movement-Conditioned `GRPO-1`:** 10-step LR sweep on ZPD-only prompts (no abstention reward), selecting a learning rate that satisfies pre-registered movement criteria (`||ΔW||_F / ||W_0||_F >= 1.0%` or `mean KL >= 0.02` without repetition collapse) so any null result is genuinely informative.
   - **Conditional `GRPO-2` (Dynamic-$\lambda$ Ternary Abstention):** Run only if `GRPO-1` is stable; evaluates pure dynamic-$\lambda$ (`0.25 -> 1.00`, no intra-group quota, `>=60%` coverage floor) on `Dev-500`.
3. **Blind Human Discordant Audit (`sft/BLIND_HUMAN_DISCORDANT_AUDIT.md`):** All `13 / 13` cross-arm discordant questions graded by a human reviewer (zero LLM judging).
4. **Single Frozen Candidate Selection:** Exactly ONE final system configuration (adapter weights, thresholds, $\tau = 0.20$ or Dev-500-locked threshold) is frozen before unsealing `Q201..Q500`.

---

## 3. Primary Confirmatory Hypotheses & Statistical Test

- **Test Pool:** `MATH-500` indices `Q201..Q500` (`N = 300`), strictly untouched by prompt tuning, threshold calibration, or error analysis.
- **Primary Endpoint (Forced-Answer Accuracy):** Two-sided exact binomial **McNemar test** at $\alpha = 0.05$ comparing the frozen candidate against same-stack `SFT Baseline` on `N = 300`.
- **Secondary Endpoint (Selective Prediction Utility):** Ternary utility $U(\lambda) = \frac{1}{N}\sum (+1 \cdot \mathbb{I}[\text{correct}] - \lambda \cdot \mathbb{I}[\text{wrong}])$ at $\lambda \in \{1.0, 2.0\}$ with paired bootstrap 95% CI ($B = 5000$), reporting coverage and exact abstention precision (`true_hallucinations_abstained / total_abstained`).

---

## 4. Hard Stopping Rule & Publication Honesty

- **Single-Shot Rule:** `Q201..Q500` is evaluated **once**. No post-hoc threshold adjustment, grader edits, or prompt tweaks are permitted after inspecting `Q201..Q500` outputs.
- **If $p \ge 0.05$ on Primary Endpoint:** Report the exact $2\times2$ contingency table ($b_{11}, b_{01}, b_{10}, b_{00}$) and $p$-value honestly as a non-significant forced-accuracy difference, without claiming superiority on forced accuracy.
- **Post-Epoch-I Engineering (`Wave 5`):** Any upgrades to `cst_symbolic_verifier.py` (recall expansion, subprocess timeout, relative tolerance on symbolic residuals) belong strictly to **Epoch II / v2.2** and cannot be retroactively applied to Epoch I `Q201..Q500` numbers.
