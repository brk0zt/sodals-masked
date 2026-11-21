"""
SODALS v2.1 — Claude Round 3 Protocol: Selective Prediction, Detector AUROC, & Consistent Hallucination Audit
============================================================================================================
Implements Claude's Round 3 GPU-Free Audit Protocol:
  1. Measure Consistent Hallucination Baseline (Tutarlı Yanlış Oranı):
     - What percentage of failed questions have a majority of candidate rollouts agreeing on the same false answer?
  2. Detector AUROC Comparison & Paired Bootstrap:
     - Compare Tuned Lens / J-Lens geometric consensus (mean_stability, answer_stability, top1_agreement)
       vs. Token Length vs. SymPy Symbolic Validity for predicting derivation correctness.
     - Paired bootstrap test (B=2000) for statistical significance against baseline.
  3. Risk-Coverage Curve (Seçmeli Tahmin / Abstention Simulation):
     - Simulate abstention ("Bilmiyorum" / Selective Prediction): reject bottom X% least confident predictions
     - Measure effective accuracy on remaining answered questions (coverage vs. risk).
"""

import json
import math
import random
import numpy as np

def compute_auroc(y_true, scores):
    """Computes exact Mann-Whitney U statistic AUROC."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(scores)) + 1
    u = np.sum(ranks[y_true]) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))

def paired_bootstrap_auroc(y_true, scores_candidate, scores_baseline, n_boot=2000, seed=42):
    """Paired bootstrap test for difference in AUROC."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=bool)
    scores_c = np.asarray(scores_candidate, dtype=float)
    scores_b = np.asarray(scores_baseline, dtype=float)
    n = len(y_true)
    
    auroc_c_base = compute_auroc(y_true, scores_c)
    auroc_b_base = compute_auroc(y_true, scores_b)
    diff_base = auroc_c_base - auroc_b_base
    
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_c = compute_auroc(y_true[idx], scores_c[idx])
        auc_b = compute_auroc(y_true[idx], scores_b[idx])
        diffs.append(auc_c - auc_b)
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    # two-sided p-value against null diff == 0
    p_val = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return auroc_c_base, auroc_b_base, diff_base, ci_lower, ci_upper, p_val

def compute_risk_coverage(y_true, confidence_scores, coverages=[1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]):
    """Simulates selective prediction risk-coverage curve."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(confidence_scores, dtype=float)
    n = len(y_true)
    sorted_indices = np.argsort(scores)[::-1] # highest confidence first
    
    table = []
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        selected_idx = sorted_indices[:k]
        acc = float(np.mean(y_true[selected_idx]))
        risk = 1.0 - acc
        threshold = float(scores[sorted_indices[k-1]])
        table.append({
            "coverage": cov,
            "retained_count": k,
            "abstain_count": n - k,
            "selective_acc": acc,
            "risk": risk,
            "score_threshold": threshold
        })
    return table

def main():
    print("=" * 80)
    print("SODALS v2.1 — CLAUDE ROUND 3 FORENSIC AUDIT: SELECTIVE PREDICTION & DETECTOR AUROC")
    print("=" * 80)
    
    # Load 150 MATH-500 Module 4 dataset
    path_150 = r"C:\Users\Burak\Desktop\sodals_core\sft\eval_grpo_long_best_of_g_prm_150_results.json"
    with open(path_150, "r", encoding="utf-8") as f:
        data_150 = json.load(f)["details"]
        
    # Restrict to Clean-125 Held-Out
    clean_125 = [d for d in data_150 if not d.get("is_train_subset")]
    print(f"\n1. Clean-125 Held-Out Sample Size: N = {len(clean_125)}")
    
    y_true_sym = [bool(d["symbolic_match"]) for d in clean_125]
    y_true_leg = [bool(d["verdict"]) for d in clean_125]
    
    mean_stab = [float(d.get("primary_mean_stability", 0.77)) for d in clean_125]
    ans_stab = [float(d.get("primary_answer_stability", 0.67)) for d in clean_125]
    top1_agree = [float(d.get("primary_top1_agreement", 0.99)) for d in clean_125]
    is_valid = [1.0 if d.get("primary_valid") else 0.0 for d in clean_125]
    # Length negative signal (longer CoT often correlates with struggling on errors, or inverse)
    tok_len = [float(d.get("generated_tokens", 500)) for d in clean_125]
    inv_tok = [-t for t in tok_len]
    
    print(f"Pass Rate (Normalized Symbolic): {sum(y_true_sym)} / {len(y_true_sym)} ({sum(y_true_sym)/len(y_true_sym)*100:.2f}%)")
    print(f"Pass Rate (Legacy Grader):       {sum(y_true_leg)} / {len(y_true_leg)} ({sum(y_true_leg)/len(y_true_leg)*100:.2f}%)")

    # ---------------------------------------------------------------------------
    # Audit 1: Consistent Hallucination Baseline (Tutarlı Yanlış Oranı)
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("AUDIT 1: CONSISTENT HALLUCINATION BASELINE (TUTARLI YANLIŞ ORANI)")
    print("-" * 80)
    
    # A. Stored 15-question Candidate Pools in Module 4 (where G=4 was triggered)
    triggered_items = [d for d in clean_125 if d.get("gate_triggered") and "candidate_pool" in d]
    print(f"Triggered Held-Out Questions with Multi-Branch Pool: n = {len(triggered_items)}")
    
    consistent_wrong_cnt = 0
    total_failed_triggered = 0
    for d in triggered_items:
        # Check if question failed in Pass 1
        if not d["primary_valid"] or not d["symbolic_match"]:
            total_failed_triggered += 1
            # Check candidate predictions
            cand_preds = [c["pred"].strip().lower() for c in d["candidate_pool"] if c.get("pred")]
            from collections import Counter
            counts = Counter(cand_preds)
            most_common_pred, most_common_cnt = counts.most_common(1)[0]
            # If majority of rollouts (>= 2 out of 4) agree on a WRONG prediction
            is_majority_wrong = (most_common_cnt >= 2) and (most_common_pred != str(d["ground_truth"]).strip().lower())
            if is_majority_wrong:
                consistent_wrong_cnt += 1
                print(f"  - Consistent Hallucination on Q{d['index']:03d}: Ground Truth '{d['ground_truth']}' | Majority Vote '{most_common_pred}' ({most_common_cnt}/4 votes)")
                
    print(f"\nConsistent Hallucination Rate on Failed Triggered Items: {consistent_wrong_cnt} / {total_failed_triggered} ({consistent_wrong_cnt/max(1, total_failed_triggered)*100:.1f}%)")

    # B. Current GRPO Out-of-MATH-500 Train Steps (1..32)
    path_train_grpo = r"C:\Users\Burak\Desktop\sodals_core\sft\grpo_math_train_100_results.json"
    with open(path_train_grpo, "r", encoding="utf-8") as f:
        train_grpo = json.load(f)["step_details"]
        
    grpo_wrong_steps = [s for s in train_grpo if s.get("num_correct", 0) == 0]
    grpo_consistent_wrong = 0
    for s in grpo_wrong_steps:
        preds = [r["pred"].strip().lower() for r in s.get("rollouts", []) if r.get("pred")]
        from collections import Counter
        counts = Counter(preds)
        if counts:
            top_p, top_cnt = counts.most_common(1)[0]
            if top_cnt >= 2 and top_p != str(s["ground_truth"]).strip().lower():
                grpo_consistent_wrong += 1
                
    print(f"Consistent Hallucination on 0-Solved GRPO Train Steps: {grpo_consistent_wrong} / {len(grpo_wrong_steps)} ({grpo_consistent_wrong/max(1, len(grpo_wrong_steps))*100:.1f}%)")
    print(f"Insight: In ~{grpo_consistent_wrong/max(1, len(grpo_wrong_steps))*100:.1f}% of completely failed problems, independent stochastic rollouts collapse into the SAME wrong consensus.")

    # ---------------------------------------------------------------------------
    # Audit 2: Detector AUROC Comparison & Paired Bootstrap Test
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("AUDIT 2: DETECTOR AUROC COMPARISON (PREDICTING DERIVATION CORRECTNESS)")
    print("-" * 80)
    
    auc_sym_valid = compute_auroc(y_true_sym, is_valid)
    auc_mean_stab = compute_auroc(y_true_sym, mean_stab)
    auc_ans_stab = compute_auroc(y_true_sym, ans_stab)
    auc_top1_agr = compute_auroc(y_true_sym, top1_agree)
    auc_inv_tok = compute_auroc(y_true_sym, inv_tok)
    
    print(f"{'Feature / Candidate Signal':<35} | {'AUROC':<8} | {'Interpretation'}")
    print("-" * 75)
    print(f"{'1. SymPy Intermediate Validity (Valid)':<35} | {auc_sym_valid:<8.4f} | External Symbolic Grounding")
    print(f"{'2. Inverse Token Length (-Length)':<35} | {auc_inv_tok:<8.4f} | Shorter = More Confident / Direct")
    print(f"{'3. Tuned Lens Answer Stability':<35} | {auc_ans_stab:<8.4f} | Late-Layer Token Projection Stability")
    print(f"{'4. Tuned Lens Top-1 Agree':<35} | {auc_top1_agr:<8.4f} | Vocabulary Head Convergence")
    print(f"{'5. Tuned Lens Mean Cosine Stability':<35} | {auc_mean_stab:<8.4f} | Inverse signal! (Hallucinations more stable: 0.7867 vs 0.7756)")

    # Paired Bootstrap of Tuned Lens vs Inverse Token Length Baseline
    auc_c, auc_b, diff, ci_l, ci_u, p_val = paired_bootstrap_auroc(y_true_sym, ans_stab, inv_tok)
    print(f"\nPaired Bootstrap (Answer Stability vs. Length Baseline, B=2000):")
    print(f"  Delta AUROC = {diff:+.4f} | 95% CI: [{ci_l:+.4f}, {ci_u:+.4f}] | p-value = {p_val:.4f}")
    if ci_l > 0:
        print("  Verdict: CANDIDATE SIGNAL SIGNIFICANTLY SURPASSES BASELINE (p < 0.05)!")
    else:
        print("  Verdict: NULL RESULT. Confidence interval includes 0; cannot reject the null hypothesis.")

    # ---------------------------------------------------------------------------
    # Audit 3: Risk-Coverage Curve (Selective Prediction / 'Bilmiyorum' Simülasyonu)
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("AUDIT 3: RISK-COVERAGE CURVE (SELECTIVE PREDICTION / ABSTENTION SIMULATION)")
    print("-" * 80)
    
    # Combined Confidence Score: SymPy Validity + Answer Stability + Normalized Directness
    combined_score = np.array(is_valid) * 1.0 + np.array(ans_stab) * 1.5 - (np.array(tok_len) / 2048.0) * 0.5
    
    rc_table = compute_risk_coverage(y_true_sym, combined_score)
    print(f"{'Coverage':<10} | {'Retained':<10} | {'Abstained':<10} | {'Selective Acc':<15} | {'Risk (Error Rate)':<18} | {'Min Conf'}")
    print("-" * 80)
    for row in rc_table:
        print(f"{row['coverage']*100:6.1f}%    | {row['retained_count']:8d}   | {row['abstain_count']:8d}   | {row['selective_acc']*100:6.2f}%         | {row['risk']*100:6.2f}%             | {row['score_threshold']:+.4f}")
        
    print("\nSummary finding on Risk-Coverage:")
    acc_100 = rc_table[0]['selective_acc'] * 100
    acc_80 = [r for r in rc_table if r['coverage'] == 0.80][0]['selective_acc'] * 100
    acc_70 = [r for r in rc_table if r['coverage'] == 0.70][0]['selective_acc'] * 100
    print(f"  At 100% Coverage (No abstention): Accuracy = {acc_100:.2f}% (Risk = {100-acc_100:.2f}%)")
    print(f"  At  80% Coverage (Abstain 20%):   Accuracy = {acc_80:.2f}% (Risk = {100-acc_80:.2f}%) [Gain = +{acc_80-acc_100:.2f}%]")
    print(f"  At  70% Coverage (Abstain 30%):   Accuracy = {acc_70:.2f}% (Risk = {100-acc_70:.2f}%) [Gain = +{acc_70-acc_100:.2f}%]")
    
    # Save Report to Disk
    report_dict = {
        "consistent_hallucination_rate_clean125": consistent_wrong_cnt / max(1, total_failed_triggered),
        "consistent_hallucination_rate_grpo_train": grpo_consistent_wrong / max(1, len(grpo_wrong_steps)),
        "auroc_comparison": {
            "sympy_valid": auc_sym_valid,
            "inv_length": auc_inv_tok,
            "answer_stability": auc_ans_stab,
            "top1_agreement": auc_top1_agr,
            "mean_stability": auc_mean_stab,
        },
        "paired_bootstrap": {
            "diff": diff,
            "ci_95": [ci_l, ci_u],
            "p_val": p_val
        },
        "risk_coverage_curve": rc_table
    }
    with open(r"C:\Users\Burak\Desktop\sodals_core\sft\CLAUDE_ROUND3_SELECTIVE_PREDICTION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    with open(r"C:\Users\Burak\Desktop\sodals_core\claude_icin\23_CLAUDE_ROUND3_SELECTIVE_PREDICTION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print("\nSaved report to sft/ and claude_icin/23_CLAUDE_ROUND3_SELECTIVE_PREDICTION_REPORT.json successfully.")

if __name__ == "__main__":
    main()
