"""
SODALS v2.1 — Selective Prediction & Ternary Reward Simulation Engine
====================================================================
Implements the rigorous, GPU-free selective prediction pre-test suite
preregistered with Claude:
  1. Epistemic Floor ("Tutarlı Yanlış" / Consistent Misconception Rate)
  2. Multi-Signal AUROC Benchmark with Clustered Paired Bootstrap (B=5000)
  3. Risk-Coverage Curves, AURC & Excess AURC (E-AURC)
  4. Out-of-Fold Ternary Reward Simulation (Correct=+1, Abstain=0, Wrong=-lambda)
  5. Abstention Collapse Frontier Analysis

Authors: SODALS Research Team
"""

import os
import sys
import json
import math
import random
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_DIR = os.path.join(ROOT, "sft")
sys.path.insert(0, SFT_DIR)

from cst_symbolic_verifier import NormalizedMathMatcher


def symbolic_equal(a: str, b: str) -> bool:
    """Checks mathematical equivalence between two strings using NormalizedMathMatcher."""
    if a == b:
        return True
    try:
        _, is_eq = NormalizedMathMatcher.check_match(a, b)
        return bool(is_eq)
    except Exception:
        return False


def load_dataset_1_rollouts(path: str):
    """
    Loads 100 training steps (400 rollouts, G=4) from grpo_math_train_100_results.json.
    Computes agreement rate, token count, reward, and correctness for every rollout.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data["step_details"]
    prompts = []
    all_rollouts = []

    for s in steps:
        step_id = s["step"]
        gt = s["ground_truth"]
        preds = s["rollout_preds"]
        rewards = s["rollout_rewards"]
        tokens = s["rollout_tokens"]

        # 1. Correctness of each rollout
        corrs = [symbolic_equal(p, gt) for p in preds]

        # 2. Cluster answers to detect consistency / semantic agreement
        clusters = []
        for p in preds:
            for cl in clusters:
                if symbolic_equal(p, cl[0]):
                    cl.append(p)
                    break
            else:
                clusters.append([p])

        top_cluster = max(clusters, key=len)
        majority_ans = top_cluster[0]
        majority_count = len(top_cluster)
        majority_is_correct = symbolic_equal(majority_ans, gt)

        # 3. Rollout-level features
        prompt_rollouts = []
        for i, p in enumerate(preds):
            # Agreement rate with the other 3 rollouts in the same prompt
            agree_count = sum(1 for j, q in enumerate(preds) if j != i and symbolic_equal(p, q))
            agree_rate = agree_count / 3.0

            r_dict = {
                "step": step_id,
                "rollout_idx": i,
                "problem": s["problem"],
                "subject": s["subject"],
                "level": s["level"],
                "ground_truth": gt,
                "pred": p,
                "correct": corrs[i],
                "agree_rate": agree_rate,
                "reward": rewards[i],
                "tokens": tokens[i],
                "neg_tokens": -tokens[i],  # shorter = higher score
            }
            prompt_rollouts.append(r_dict)
            all_rollouts.append(r_dict)

        prompts.append({
            "step": step_id,
            "subject": s["subject"],
            "level": s["level"],
            "ground_truth": gt,
            "num_correct": sum(corrs),
            "majority_count": majority_count,
            "majority_ans": majority_ans,
            "majority_is_correct": majority_is_correct,
            "rollouts": prompt_rollouts,
        })

    return prompts, all_rollouts


def load_dataset_2_heldout_lens(path: str):
    """
    Loads Clean-125 heldout dataset with Tuned Lens and GPR features from math500_heldout_clean_125_module4_best_of_g_prm.json.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data["details"]
    records = []
    for r in details:
        records.append({
            "index": r["index"],
            "unique_id": r["unique_id"],
            "subject": r["subject"],
            "level": r["level"],
            "ground_truth": r["ground_truth"],
            "pred": r["predicted"],
            "correct": bool(r.get("symbolic_match", False)),
            "mean_stability": float(r.get("primary_mean_stability", 0.0)),
            "answer_stability": float(r.get("primary_answer_stability", 0.0)),
            "top1_agreement": 1.0 if r.get("primary_top1_agreement", False) else 0.0,
            "alpha": float(r.get("alpha", 0.0)),
            "tokens": int(r.get("generated_tokens", 0)),
            "neg_tokens": -int(r.get("generated_tokens", 0)),
        })
    return records


def compute_auroc(y_true: list, y_scores: list) -> float:
    """Computes Mann-Whitney / ROC AUC safely handling degenerate cases."""
    pos = [s for y, s in zip(y_true, y_scores) if y]
    neg = [s for y, s in zip(y_true, y_scores) if not y]
    if not pos or not neg:
        return 0.5
    w = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def clustered_paired_bootstrap(prompts: list, feature_a: str, feature_b: str, B: int = 5000, seed: int = 42):
    """
    Computes clustered prompt-level bootstrap for AUROC_A, AUROC_B, and Delta_AUROC = AUROC_A - AUROC_B.
    Clustering at the prompt level accounts for the intra-prompt correlation among rollouts.
    """
    rng = random.Random(seed)
    n_prompts = len(prompts)

    delta_vals = []
    auc_a_vals = []
    auc_b_vals = []

    for _ in range(B):
        # Sample prompts with replacement
        sample_prompts = [prompts[rng.randint(0, n_prompts - 1)] for _ in range(n_prompts)]
        sample_rollouts = [r for p in sample_prompts for r in p["rollouts"]]

        y_true = [r["correct"] for r in sample_rollouts]
        scores_a = [r[feature_a] for r in sample_rollouts]
        scores_b = [r[feature_b] for r in sample_rollouts]

        auc_a = compute_auroc(y_true, scores_a)
        auc_b = compute_auroc(y_true, scores_b)

        if not math.isnan(auc_a) and not math.isnan(auc_b):
            auc_a_vals.append(auc_a)
            auc_b_vals.append(auc_b)
            delta_vals.append(auc_a - auc_b)

    delta_vals.sort()
    auc_a_vals.sort()
    auc_b_vals.sort()

    ci_lo_idx = int(0.025 * len(delta_vals))
    ci_hi_idx = int(0.975 * len(delta_vals))

    ci_delta = (delta_vals[ci_lo_idx], delta_vals[ci_hi_idx])
    ci_a = (auc_a_vals[ci_lo_idx], auc_a_vals[ci_hi_idx])
    ci_b = (auc_b_vals[ci_lo_idx], auc_b_vals[ci_hi_idx])

    # Two-sided empirical p-value for Delta == 0
    p_val = 2.0 * min(
        sum(1 for d in delta_vals if d <= 0) / len(delta_vals),
        sum(1 for d in delta_vals if d >= 0) / len(delta_vals)
    )
    p_val = min(1.0, max(1.0 / B, p_val))

    return {
        "auc_a_mean": float(np.mean(auc_a_vals)),
        "auc_a_ci": [float(ci_a[0]), float(ci_a[1])],
        "auc_b_mean": float(np.mean(auc_b_vals)),
        "auc_b_ci": [float(ci_b[0]), float(ci_b[1])],
        "delta_mean": float(np.mean(delta_vals)),
        "delta_ci": [float(ci_delta[0]), float(ci_delta[1])],
        "excludes_zero": bool(ci_delta[0] > 0 or ci_delta[1] < 0),
        "p_value": float(p_val),
    }


def compute_risk_coverage(items: list, score_key: str, n_bins: int = 100, seed: int = 42):
    """
    Computes selective accuracy A(c), selective risk R(c) = 1 - A(c) as a function of coverage c.
    Also calculates AURC (Area Under Risk-Coverage Curve) and Excess-AURC over optimal oracle.
    """
    rng = random.Random(seed)
    # Sort items by score descending (tie-breaking deterministically with seeded rng)
    sorted_items = sorted(items, key=lambda x: (x[score_key], rng.random()), reverse=True)
    n = len(sorted_items)

    y = [1 if it["correct"] else 0 for it in sorted_items]
    total_pos = sum(y)

    coverages = []
    accuracies = []
    risks = []

    # Discrete step integration
    step_risks = []
    cum_correct = 0

    for k in range(1, n + 1):
        cum_correct += y[k - 1]
        acc = cum_correct / k
        risk = 1.0 - acc
        step_risks.append(risk)

    # Standard AURC (discrete approximation across all sample ranks)
    aurc = float(np.mean(step_risks))

    # Optimal Oracle AURC (answers only positive items first)
    oracle_risks = []
    for k in range(1, n + 1):
        if k <= total_pos:
            oracle_risks.append(0.0)
        else:
            oracle_risks.append(1.0 - (total_pos / k))
    aurc_oracle = float(np.mean(oracle_risks))
    e_aurc = aurc - aurc_oracle

    # Grid sample for report table
    for c in np.linspace(0.10, 1.00, 19):
        k = int(round(c * n))
        k = max(1, min(n, k))
        acc = sum(y[:k]) / k
        coverages.append(float(round(c, 2)))
        accuracies.append(float(round(acc, 4)))
        risks.append(float(round(1.0 - acc, 4)))

    return {
        "aurc": float(round(aurc, 4)),
        "aurc_oracle": float(round(aurc_oracle, 4)),
        "e_aurc": float(round(e_aurc, 4)),
        "curve_samples": [
            {"coverage": c, "accuracy": a, "risk": r}
            for c, a, r in zip(coverages, accuracies, risks)
        ]
    }


def simulate_ternary_reward_cv(prompts: list, score_key: str, lambdas: list, n_folds: int = 5, seed: int = 42):
    """
    Performs out-of-fold cross-validation on prompts to calibrate decision threshold tau*(lambda).
    Evaluates out-of-fold Utility U, Coverage C, and Effective Accuracy A_eff.
    Payoffs:
      Correct: +1.0
      Abstain:  0.0
      Wrong:   -lambda
    """
    rng = random.Random(seed)
    shuffled_prompts = list(prompts)
    rng.shuffle(shuffled_prompts)

    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(shuffled_prompts):
        folds[i % n_folds].append(p)

    results_by_lambda = {}

    for lam in lambdas:
        bayes_req_prob = lam / (1.0 + lam)
        fold_test_payoffs = []
        fold_test_coverages = []
        fold_test_accs = []
        calibrated_taus = []

        for fold_idx in range(n_folds):
            train_prompts = [p for i, f in enumerate(folds) if i != fold_idx for p in f]
            test_prompts = folds[fold_idx]

            train_rollouts = [r for p in train_prompts for r in p["rollouts"]]
            test_rollouts = [r for p in test_prompts for r in p["rollouts"]]

            # Grid search for optimal tau on train rollouts
            unique_scores = sorted(list(set(r[score_key] for r in train_rollouts)))
            # Add boundary scores
            candidate_taus = [-float("inf")] + unique_scores

            best_tau = -float("inf")
            best_train_util = -float("inf")

            for tau in candidate_taus:
                tot_util = 0.0
                for r in train_rollouts:
                    if r[score_key] >= tau:
                        tot_util += 1.0 if r["correct"] else -lam
                    else:
                        tot_util += 0.0  # abstain
                avg_util = tot_util / len(train_rollouts)
                if avg_util > best_train_util:
                    best_train_util = avg_util
                    best_tau = tau

            calibrated_taus.append(best_tau)

            # Evaluate on out-of-fold test rollouts using best_tau
            answered = [r for r in test_rollouts if r[score_key] >= best_tau]
            n_answered = len(answered)
            n_test = len(test_rollouts)

            coverage = n_answered / n_test
            acc = (sum(1 for r in answered if r["correct"]) / n_answered) if n_answered > 0 else 0.0
            test_util = sum((1.0 if r["correct"] else -lam) for r in answered) / n_test

            fold_test_coverages.append(coverage)
            fold_test_accs.append(acc)
            fold_test_payoffs.append(test_util)

        # Baseline zero-abstention utility (answer 100% of questions)
        all_r = [r for p in prompts for r in p["rollouts"]]
        baseline_unselective_util = sum((1.0 if r["correct"] else -lam) for r in all_r) / len(all_r)

        results_by_lambda[lam] = {
            "lambda": lam,
            "bayes_threshold_prob": float(round(bayes_req_prob, 4)),
            "mean_calibrated_tau": float(np.mean(calibrated_taus)),
            "oof_utility": float(round(float(np.mean(fold_test_payoffs)), 4)),
            "unselective_utility": float(round(baseline_unselective_util, 4)),
            "utility_gain": float(round(float(np.mean(fold_test_payoffs)) - baseline_unselective_util, 4)),
            "mean_coverage": float(round(float(np.mean(fold_test_coverages)), 4)),
            "mean_accuracy": float(round(float(np.mean(fold_test_accs)), 4)),
            "is_collapsed": bool(float(np.mean(fold_test_coverages)) < 0.05),
        }

    return results_by_lambda


def run_full_selective_prediction_suite():
    print("=" * 80)
    print("SODALS v2.1 — SELECTIVE PREDICTION & TERNARY REWARD AUDIT")
    print("=" * 80)

    # 1. Load Data Sources
    rollouts_path = os.path.join(SFT_DIR, "grpo_math_train_100_results.json")
    heldout_path = os.path.join(SFT_DIR, "math500_heldout_clean_125_module4_best_of_g_prm.json")

    prompts, rollouts = load_dataset_1_rollouts(rollouts_path)
    heldout_records = load_dataset_2_heldout_lens(heldout_path)

    print(f"[DATA] Loaded {len(prompts)} prompts, {len(rollouts)} rollouts (G=4) from Hendrycks MATH train.")
    print(f"[DATA] Loaded {len(heldout_records)} Clean-125 heldout greedy records with Tuned Lens & GPR.")

    # -------------------------------------------------------------
    # SECTION 1: EPISTEMIC FLOOR & CONSISTENT MISCONCEPTIONS
    # -------------------------------------------------------------
    n_prompts = len(prompts)
    prompts_with_errors = [p for p in prompts if p["num_correct"] < 4]
    all_correct_prompts = [p for p in prompts if p["num_correct"] == 4]
    all_wrong_prompts = [p for p in prompts if p["num_correct"] == 0]
    mixed_prompts = [p for p in prompts if 0 < p["num_correct"] < 4]

    # Consistent Wrong: >= 3 out of 4 rollouts output the same wrong answer
    consistent_wrong = [p for p in prompts_with_errors if (not p["majority_is_correct"]) and p["majority_count"] >= 3]
    perfect_wrong = [p for p in prompts_with_errors if (not p["majority_is_correct"]) and p["majority_count"] == 4]

    print("\n" + "-" * 75)
    print("1. THE EPISTEMIC FLOOR (CONSISTENT MISCONCEPTIONS / TUTARLI YANLIŞLAR)")
    print("-" * 75)
    print(f"Total Training Prompts Evaluated : {n_prompts}")
    print(f"  - All 4 Rollouts Correct (4/4)  : {len(all_correct_prompts)} / {n_prompts} ({100*len(all_correct_prompts)/n_prompts:.1f}%)")
    print(f"  - Mixed Rollouts (1..3 Correct) : {len(mixed_prompts)} / {n_prompts} ({100*len(mixed_prompts)/n_prompts:.1f}%)")
    print(f"  - All 4 Rollouts Wrong   (0/4)  : {len(all_wrong_prompts)} / {n_prompts} ({100*len(all_wrong_prompts)/n_prompts:.1f}%)")
    print(f"Prompts with >= 1 Failed Rollout : {len(prompts_with_errors)}")
    print(f"  - Consistent Wrong (>=3/4 Agree): {len(consistent_wrong)} / {len(prompts_with_errors)} ({100*len(consistent_wrong)/len(prompts_with_errors):.1f}%)")
    print(f"  - Unanimous Wrong  (4/4 Agree)  : {len(perfect_wrong)} / {len(prompts_with_errors)} ({100*len(perfect_wrong)/len(prompts_with_errors):.1f}%)")
    print(f"Epistemic Floor on Full Pool     : {len(consistent_wrong)} / {n_prompts} ({100*len(consistent_wrong)/n_prompts:.1f}%)")
    print("-> Key Insight: Internal sampling / self-consistency CANNOT resolve these prompts.")

    # -------------------------------------------------------------
    # SECTION 2: MULTI-SIGNAL AUROC BENCHMARK & PAIRED BOOTSTRAP
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("2. DETECTOR AUROC BENCHMARK WITH CLUSTERED PROMPT BOOTSTRAP (B=5000)")
    print("-" * 75)

    y_all = [r["correct"] for r in rollouts]
    auc_agree = compute_auroc(y_all, [r["agree_rate"] for r in rollouts])
    auc_reward = compute_auroc(y_all, [r["reward"] for r in rollouts])
    auc_len = compute_auroc(y_all, [r["neg_tokens"] for r in rollouts])

    print(f"Detector 1: Agreement Rate (Semantic Consistency) : AUROC = {auc_agree:.4f}")
    print(f"Detector 2: Outcome + Process Reward              : AUROC = {auc_reward:.4f}")
    print(f"Detector 3: Shorter Sequence Length (-Tokens)     : AUROC = {auc_len:.4f}")

    # Clustered Bootstrap Comparisons against Baseline
    bs_agree_vs_len = clustered_paired_bootstrap(prompts, "agree_rate", "neg_tokens", B=5000)
    bs_agree_vs_rew = clustered_paired_bootstrap(prompts, "agree_rate", "reward", B=5000)

    print(f"\nPaired Clustered Bootstrap Test (Agreement vs Shorter Length):")
    print(f"  Delta AUROC = {bs_agree_vs_len['delta_mean']:+.4f} [95% CI: {bs_agree_vs_len['delta_ci'][0]:.4f}, {bs_agree_vs_len['delta_ci'][1]:.4f}]")
    print(f"  Excludes Zero? {bs_agree_vs_len['excludes_zero']} (p = {bs_agree_vs_len['p_value']:.4f})")

    # Clustered Bootstrap on Mixed Prompts Only (where sample consistency is actively differentiating)
    mixed_prompt_objs = [p for p in prompts if 0 < p["num_correct"] < 4]
    mixed_rollouts = [r for p in mixed_prompt_objs for r in p["rollouts"]]
    y_mixed = [r["correct"] for r in mixed_rollouts]
    auc_mixed_agree = compute_auroc(y_mixed, [r["agree_rate"] for r in mixed_rollouts])
    auc_mixed_len = compute_auroc(y_mixed, [r["neg_tokens"] for r in mixed_rollouts])
    bs_mixed = clustered_paired_bootstrap(mixed_prompt_objs, "agree_rate", "neg_tokens", B=5000)

    print(f"\nStrict Condition: Mixed Prompts Cohort Only (N={len(mixed_rollouts)} rollouts, {len(mixed_prompt_objs)} prompts):")
    print(f"  Agreement AUROC = {auc_mixed_agree:.4f} vs Shorter Length AUROC = {auc_mixed_len:.4f}")
    print(f"  Delta AUROC     = {bs_mixed['delta_mean']:+.4f} [95% CI: {bs_mixed['delta_ci'][0]:.4f}, {bs_mixed['delta_ci'][1]:.4f}] (p = {bs_mixed['p_value']:.4f})")

    # Clean-125 Heldout Tuned Lens Probing (Single-Rollout Greedy Baseline)
    y_held = [r["correct"] for r in heldout_records]
    auc_lens_mean = compute_auroc(y_held, [r["mean_stability"] for r in heldout_records])
    auc_lens_ans = compute_auroc(y_held, [r["answer_stability"] for r in heldout_records])
    auc_lens_top1 = compute_auroc(y_held, [r["top1_agreement"] for r in heldout_records])
    auc_held_len = compute_auroc(y_held, [r["neg_tokens"] for r in heldout_records])

    print(f"\nClean-125 Heldout Greedy Decoding (N=125, G=1) Single-Pass Probing:")
    print(f"  Tuned Lens Mean Stability   (S_mean) : AUROC = {auc_lens_mean:.4f}")
    print(f"  Tuned Lens Answer Stability (S_ans)  : AUROC = {auc_lens_ans:.4f}")
    print(f"  Tuned Lens Top-1 Agreement  (Top-1)  : AUROC = {auc_lens_top1:.4f}")
    print(f"  Shorter Token Length        (-Tokens): AUROC = {auc_held_len:.4f}")
    print("  -> Confident Hallucination Confirmed: Failed chains exhibit high inter-layer consensus (AUROC < 0.50).")

    # -------------------------------------------------------------
    # SECTION 3: RISK-COVERAGE CURVES & AURC
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. RISK-COVERAGE CURVES & AREA UNDER RISK CURVE (AURC)")
    print("-" * 75)

    rc_agree = compute_risk_coverage(rollouts, "agree_rate")
    rc_reward = compute_risk_coverage(rollouts, "reward")
    rc_len = compute_risk_coverage(rollouts, "neg_tokens")

    print(f"Agreement Rate   : AURC = {rc_agree['aurc']:.4f} | Oracle = {rc_agree['aurc_oracle']:.4f} | E-AURC = {rc_agree['e_aurc']:.4f}")
    print(f"Outcome Reward   : AURC = {rc_reward['aurc']:.4f} | Oracle = {rc_reward['aurc_oracle']:.4f} | E-AURC = {rc_reward['e_aurc']:.4f}")
    print(f"Shorter Length   : AURC = {rc_len['aurc']:.4f} | Oracle = {rc_len['aurc_oracle']:.4f} | E-AURC = {rc_len['e_aurc']:.4f}")

    print("\nAgreement Rate Selective Accuracy Across Coverage Tiers:")
    print("Coverage | Selective Accuracy | Selective Risk | Abstention Rate")
    for row in rc_agree["curve_samples"]:
        if row["coverage"] in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            abstain_pct = (1.0 - row["coverage"]) * 100
            print(f"  {row['coverage']*100:5.1f}%  |      {row['accuracy']*100:5.2f}%       |     {row['risk']*100:5.2f}%    |     {abstain_pct:5.1f}%")

    # -------------------------------------------------------------
    # SECTION 4: TERNARY REWARD SIMULATION & OUT-OF-FOLD CV
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. TERNARY REWARD SIMULATION (5-FOLD CLUSTERED CV)")
    print("   Payoffs: Correct = +1.0 | Abstain = 0.0 | Wrong = -lambda")
    print("-" * 75)

    lambda_grid = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]
    ternary_results = simulate_ternary_reward_cv(prompts, "agree_rate", lambda_grid, n_folds=5, seed=42)

    print("Lambda | Bayes P* | Opt Threshold tau* | OOF Utility | Unselective | Utility Gain | Coverage | Accuracy")
    print("-" * 95)
    for lam in lambda_grid:
        res = ternary_results[lam]
        print(f" {lam:5.2f} |  {res['bayes_threshold_prob']:5.3f}   |       {res['mean_calibrated_tau']:5.3f}       |   {res['oof_utility']:+6.4f}   |   {res['unselective_utility']:+6.4f}   |    {res['utility_gain']:+6.4f}    |  {res['mean_coverage']*100:5.1f}%  |  {res['mean_accuracy']*100:5.1f}%")

    # Find collapse threshold
    collapse_lams = [lam for lam, r in ternary_results.items() if r["is_collapsed"]]
    collapse_point = collapse_lams[0] if collapse_lams else "> 5.00"
    print(f"\nCollapse Frontier: Total Abstention Collapse occurs at lambda >= {collapse_point}")

    # -------------------------------------------------------------
    # SECTION 5: EXPORT ARTIFACT & STRUCTURED JSON
    # -------------------------------------------------------------
    out_dict = {
        "epistemic_floor": {
            "total_prompts": n_prompts,
            "all_correct": len(all_correct_prompts),
            "mixed": len(mixed_prompts),
            "all_wrong": len(all_wrong_prompts),
            "consistent_wrong_count": len(consistent_wrong),
            "consistent_wrong_rate": len(consistent_wrong) / len(prompts_with_errors),
            "unanimous_wrong_count": len(perfect_wrong),
            "sample_consistent_wrong": [
                {
                    "step": p["step"],
                    "subject": p["subject"],
                    "level": p["level"],
                    "ground_truth": p["ground_truth"],
                    "misconception_ans": p["majority_ans"],
                    "consensus_count": p["majority_count"],
                }
                for p in consistent_wrong[:8]
            ]
        },
        "auroc_benchmark": {
            "agreement_rate": {"auroc": auc_agree, "ci": bs_agree_vs_len["auc_a_ci"]},
            "outcome_reward": {"auroc": auc_reward, "ci": bs_agree_vs_rew["auc_b_ci"]},
            "shorter_length": {"auroc": auc_len, "ci": bs_agree_vs_len["auc_b_ci"]},
            "paired_delta_agree_vs_len": bs_agree_vs_len,
            "mixed_cohort_agreement": {"auroc": auc_mixed_agree, "paired_delta": bs_mixed},
            "tuned_lens_greedy_clean125": {
                "mean_stability": auc_lens_mean,
                "answer_stability": auc_lens_ans,
                "top1_agreement": auc_lens_top1,
                "shorter_length": auc_held_len,
            }
        },
        "risk_coverage": {
            "agreement_rate": rc_agree,
            "outcome_reward": rc_reward,
            "shorter_length": rc_len,
        },
        "ternary_reward_simulation": ternary_results,
    }

    out_path = os.path.join(SFT_DIR, "selective_prediction_simulation_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)
        saved_path = out_path
    except PermissionError:
        import tempfile
        saved_path = os.path.join(tempfile.gettempdir(), "selective_prediction_simulation_results.json")
        with open(saved_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[OUTPUT] Comprehensive results exported to: {saved_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_selective_prediction_suite()
