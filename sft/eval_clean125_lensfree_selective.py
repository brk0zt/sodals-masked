"""
SODALS v2.1 — Clean-125 Held-Out Evaluation:
Agreement + CST (Lens-Free) Best-of-4 & Ternary Selective Abstention Gate
=========================================================================
Strictly CPU-only, deterministic evaluation across all 125 held-out problems
(Indices 26..150) using the post-fix NormalizedMathMatcher symbolic grader.
"""

import os
import sys
import json
import math
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
for p in (SFT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from cst_symbolic_verifier import NormalizedMathMatcher


def symbolic_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        _, is_eq = NormalizedMathMatcher.check_match(str(a), str(b))
        return bool(is_eq)
    except Exception:
        return False


def binomial_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * cum)


def score_candidate_lens_free(cand: dict, agree_count: int) -> float:
    """
    Lens-Free Neurosymbolic Score (Agreement + Capped CST Equations - Contradictions).
    Completely excludes Tuned Lens `mean_stability`.
    """
    has_end = 1.0 if cand.get("has_end", False) else 0.0
    eq_capped = min(int(cand.get("eq_count", 0)), 10)
    n_contras = len(cand.get("contras", []))
    wait_cnt = int(cand.get("wait_cnt", 0))

    # Interpretable Lens-Free weights:
    # +10.0 for clean completion (</think>)
    # +8.0 per agreeing rollout in G=4
    # +1.5 per verified CST equation (capped at 10 to prevent equation-spamming)
    # -8.0 per CST symbolic contradiction
    # -2.0 per hesitation/wait loop
    score = (
        10.0 * has_end
        + 8.0 * agree_count
        + 1.5 * eq_capped
        - 8.0 * n_contras
        - 2.0 * wait_cnt
    )
    return round(score, 4)


def evaluate_clean125():
    mod4_path = os.path.join(SFT_DIR, "math500_heldout_clean_125_module4_best_of_g_prm.json")
    sft_path = os.path.join(ROOT_DIR, "math500_domain2_soft_blended_170_results.json")
    grpo_postfix_path = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")

    with open(mod4_path, "r", encoding="utf-8") as f:
        mod4_data = json.load(f)["details"]

    with open(sft_path, "r", encoding="utf-8") as f:
        sft_raw = json.load(f)
        sft_list = sft_raw.get("results", sft_raw.get("details", []))
        sft_by_uid = {r["unique_id"]: r for r in sft_list if "unique_id" in r}

    with open(grpo_postfix_path, "r", encoding="utf-8") as f:
        grpo_raw = json.load(f)
        grpo_list = grpo_raw.get("details", grpo_raw.get("results", []))
        grpo_by_uid = {r["unique_id"]: r for r in grpo_list if "unique_id" in r}

    print("=" * 88)
    print("SODALS v2.1 — CLEAN-125 COMPREHENSIVE ABLATION & TERNARY GATE BENCHMARK")
    print("=" * 88)

    records = []

    for r in mod4_data:
        uid = r["unique_id"]
        idx = r["index"]
        gt = r["ground_truth"]
        subj = r["subject"]
        lvl = r["level"]

        # 1. SFT Baseline
        sft_rec = sft_by_uid.get(uid, {})
        sft_pred = sft_rec.get("predicted", sft_rec.get("prediction", ""))
        sft_ok = bool(symbolic_equal(sft_pred, gt))

        # 2. GRPO Post-Fix Greedy
        grpo_rec = grpo_by_uid.get(uid, {})
        grpo_pred = grpo_rec.get("predicted", "")
        grpo_ok = bool(symbolic_equal(grpo_pred, gt))
        grpo_tokens = grpo_rec.get("generated_tokens", r.get("generated_tokens", 0))

        # 3. Module 2 Primary Greedy (Pass1_Greedy)
        prim_pred = r.get("primary_pred", "")
        prim_ok = bool(symbolic_equal(prim_pred, gt))
        prim_abandon = bool(r.get("primary_abandonment", False))
        prim_valid = bool(r.get("primary_valid", True))
        prim_contras = r.get("primary_contras", [])
        prim_tokens = int(r.get("generated_tokens", 0))

        # 4. Module 4 Legacy PRM (with Tuned Lens)
        prm_legacy_pred = r.get("predicted", "")
        prm_legacy_ok = bool(symbolic_equal(prm_legacy_pred, gt))

        gate_trig = bool(r.get("gate_triggered", False))
        cands = r.get("candidate_pool", [])

        # Compute candidate agreement and Lens-Free scores if gate triggered
        if gate_trig and len(cands) > 0:
            preds = [c["pred"] for c in cands]
            agree_counts = []
            for i, p_i in enumerate(preds):
                cnt = sum(1 for j, p_j in enumerate(preds) if symbolic_equal(p_i, p_j))
                agree_counts.append(cnt)

            # Pure Majority Vote (tie-broken by first occurrence / targeted T0)
            max_agree = max(agree_counts)
            maj_idx = agree_counts.index(max_agree)
            maj_pred = preds[maj_idx]
            maj_ok = bool(symbolic_equal(maj_pred, gt))

            # Lens-Free Agreement + CST Score
            lf_scores = [
                score_candidate_lens_free(cands[i], agree_counts[i])
                for i in range(len(cands))
            ]
            best_lf_idx = max(range(len(cands)), key=lambda i: lf_scores[i])
            lf_pred = preds[best_lf_idx]
            lf_ok = bool(symbolic_equal(lf_pred, gt))
            lf_winner_agree = agree_counts[best_lf_idx]
            lf_winner_score = lf_scores[best_lf_idx]
            lf_winner_contras = len(cands[best_lf_idx].get("contras", []))
            lf_winner_has_end = bool(cands[best_lf_idx].get("has_end", False))
        else:
            max_agree = 4 if (not prim_abandon and prim_valid) else 0
            maj_pred = prim_pred
            maj_ok = prim_ok
            lf_pred = prim_pred
            lf_ok = prim_ok
            lf_winner_agree = 4 if (not prim_abandon and prim_valid) else 0
            lf_winner_score = 35.0 if (not prim_abandon and prim_valid) else -10.0
            lf_winner_contras = len(prim_contras)
            lf_winner_has_end = not prim_abandon

        # Confidence signal for Selective Abstention Gate on Clean-125:
        # Tiered Neurosymbolic Confidence Score C_ns in [0, 1]:
        # - Non-triggered clean Pass1 (completed, valid, no contradictions): C_ns = 0.90 (or 0.75 if tokens >= 1800)
        # - Non-triggered abandoned / token-exhausted Pass1 (prim_abandon=True): C_ns = 0.10
        # - Gate-triggered pool:
        #     * max_agree == 4 and winner clean: C_ns = 0.95
        #     * max_agree == 3 and winner clean: C_ns = 0.80
        #     * max_agree == 2 or winner has high CST valid eqs (lf_winner_score >= 26.0): C_ns = 0.60
        #     * max_agree == 2 with low CST score (< 26.0): C_ns = 0.35
        #     * max_agree == 1 (all 4 candidates diverge) and lf_winner_score < 26.0: C_ns = 0.15
        if not gate_trig:
            if prim_abandon or prim_tokens >= 2040:
                conf_ns = 0.10
            elif not prim_valid or len(prim_contras) > 0:
                conf_ns = 0.40
            else:
                conf_ns = 0.90
        else:
            if max_agree == 4 and lf_winner_contras == 0:
                conf_ns = 0.95
            elif max_agree == 3 and lf_winner_has_end:
                conf_ns = 0.80
            elif max_agree == 2 and lf_winner_score >= 26.0:
                conf_ns = 0.60
            elif max_agree == 2:
                conf_ns = 0.35
            else:
                # max_agree == 1: all 4 candidates diverge into 4 distinct answers
                conf_ns = 0.15

        records.append({
            "index": idx,
            "unique_id": uid,
            "subject": subj,
            "level": lvl,
            "ground_truth": gt,
            "gate_triggered": gate_trig,
            "prim_abandon": prim_abandon,
            "prim_tokens": prim_tokens,
            "sft_ok": sft_ok,
            "grpo_ok": grpo_ok,
            "prim_ok": prim_ok,
            "prm_legacy_ok": prm_legacy_ok,
            "maj_ok": maj_ok,
            "lf_ok": lf_ok,
            "prim_pred": prim_pred,
            "prm_legacy_pred": prm_legacy_pred,
            "maj_pred": maj_pred,
            "lf_pred": lf_pred,
            "max_agree": max_agree,
            "lf_winner_score": lf_winner_score,
            "conf_ns": conf_ns,
        })

    n_total = len(records)
    sft_hits = sum(r["sft_ok"] for r in records)
    grpo_hits = sum(r["grpo_ok"] for r in records)
    prim_hits = sum(r["prim_ok"] for r in records)
    prm_legacy_hits = sum(r["prm_legacy_ok"] for r in records)
    maj_hits = sum(r["maj_ok"] for r in records)
    lf_hits = sum(r["lf_ok"] for r in records)

    print(f"\n1. FORCED-ANSWER ACCURACY (100% COVERAGE, N = {n_total})")
    print("-" * 88)
    models = [
        ("SFT Baseline (Greedy)", "sft_ok", sft_hits),
        ("Post-Fix GRPO (Greedy)", "grpo_ok", grpo_hits),
        ("Module 2 Pondering + CST (Greedy Primary)", "prim_ok", prim_hits),
        ("Module 4 Best-of-4: Pure Majority Vote", "maj_ok", maj_hits),
        ("Module 4 Best-of-4: Legacy PRM (w/ Tuned Lens)", "prm_legacy_ok", prm_legacy_hits),
        ("Module 4 Best-of-4: NEW Agreement + CST (Lens-Free)", "lf_ok", lf_hits),
    ]

    print(f"{'Method':<52} | {'Solved':<10} | {'Acc (%)':<8} | {'vs SFT (+ / - / Net)':<20} | {'McNemar p'}")
    print("-" * 88)
    for name, key, hits in models:
        gains = sum(1 for r in records if r[key] and not r["sft_ok"])
        losses = sum(1 for r in records if not r[key] and r["sft_ok"])
        net = gains - losses
        p_val = binomial_mcnemar_p(gains, losses)
        print(f"{name:<52} | {hits:>3}/{n_total:<6} | {100*hits/n_total:>6.2f}% | +{gains:<2} / -{losses:<2} (Net {net:+d})   | {p_val:.4f}")

    # Also compare Lens-Free vs Module 2 Primary Greedy
    gains_vs_prim = sum(1 for r in records if r["lf_ok"] and not r["prim_ok"])
    losses_vs_prim = sum(1 for r in records if not r["lf_ok"] and r["prim_ok"])
    p_vs_prim = binomial_mcnemar_p(gains_vs_prim, losses_vs_prim)
    print("-" * 88)
    print(f"Paired vs Module 2 Greedy Primary: +{gains_vs_prim} Rescues / -{losses_vs_prim} Losses (Net {gains_vs_prim - losses_vs_prim:+d}), Exact McNemar p = {p_vs_prim:.4f}")

    print("\n2. FORENSIC BREAKDOWN OF THE 13 GATE-TRIGGERED QUESTIONS (G=4 CANDIDATE POOL)")
    print("-" * 88)
    print(f"{'Q#':<5} | {'Lvl':<3} | {'Ground Truth':<14} | {'Primary':<10} | {'MajVote':<10} | {'LegacyPRM':<10} | {'LensFree':<10} | {'Agree':<5} | {'LF Score':<8} | {'Conf'}")
    print("-" * 88)
    for r in records:
        if r["gate_triggered"]:
            gt_s = str(r["ground_truth"])[:13]
            pr_s = (str(r["prim_pred"])[:7] + ("✓" if r["prim_ok"] else "✗"))
            mj_s = (str(r["maj_pred"])[:7] + ("✓" if r["maj_ok"] else "✗"))
            lg_s = (str(r["prm_legacy_pred"])[:7] + ("✓" if r["prm_legacy_ok"] else "✗"))
            lf_s = (str(r["lf_pred"])[:7] + ("✓" if r["lf_ok"] else "✗"))
            print(f"Q{r['index']:03d}  | L{r['level']}  | {gt_s:<14} | {pr_s:<10} | {mj_s:<10} | {lg_s:<10} | {lf_s:<10} | {r['max_agree']}/4   | {r['lf_winner_score']:>8.2f} | {r['conf_ns']:.2f}")

    print("\n3. TERNARY SELECTIVE PREDICTION GATE ON CLEAN-125 (Correct=+1, Abstain=0, Wrong=-lambda)")
    print("-" * 88)
    # Check non-triggered failures as well
    non_trig_fails = [r for r in records if not r["gate_triggered"] and not r["lf_ok"]]
    non_trig_abandons = [r for r in records if not r["gate_triggered"] and r["prim_abandon"]]
    print(f"Non-triggered items (N=112): {sum(1 for r in records if not r['gate_triggered'] and r['lf_ok'])}/112 correct.")
    print(f"  - Token-exhausted / abandoned chains (prim_abandon=True): {len(non_trig_abandons)} items, Correct = {sum(r['lf_ok'] for r in non_trig_abandons)}/{len(non_trig_abandons)}")

    # Evaluate Selective Gate across thresholds:
    # Mode A: Forced Answer (100% coverage, tau = 0.0)
    # Mode B: Gate 1 — Abstain on Divergent Pool (1/4 agreement) & Token Abandonment (tau = 0.20)
    # Mode C: Gate 2 — Strict Consensus + CST Verification Gate (tau = 0.50)
    modes = [
        ("Forced Answer (No Abstention, tau=0.00)", 0.00),
        ("Gate Tier 1: Filter Divergent (1/4) & Abandoned (tau=0.20)", 0.20),
        ("Gate Tier 2: Filter Weak Plurality (2/4 w/o CST) (tau=0.50)", 0.50),
        ("Gate Tier 3: High-Cert Consensus + Clean CST (tau=0.70)", 0.70),
    ]

    print("-" * 88)
    print(f"{'Selective Policy Mode':<56} | {'Ans/Tot':<8} | {'Cov (%)':<7} | {'Sel Acc':<8} | {'Halluc.':<7} | {'Util(l=1)':<9} | {'Util(l=2)'}")
    print("-" * 88)
    gate_summary = []
    for m_name, tau in modes:
        answered = [r for r in records if r["conf_ns"] >= tau]
        abstained = [r for r in records if r["conf_ns"] < tau]
        n_ans = len(answered)
        n_corr = sum(1 for r in answered if r["lf_ok"])
        n_wrong = n_ans - n_corr
        n_false_abstain = sum(1 for r in abstained if r["lf_ok"])
        n_true_abstain = sum(1 for r in abstained if not r["lf_ok"])

        cov = n_ans / n_total
        sel_acc = n_corr / n_ans if n_ans > 0 else 0.0
        util_l1 = (n_corr - 1.0 * n_wrong) / n_total
        util_l2 = (n_corr - 2.0 * n_wrong) / n_total

        print(f"{m_name:<56} | {n_ans:>3}/{n_total:<4} | {100*cov:>5.1f}%  | {100*sel_acc:>6.2f}% | {n_wrong:>2} ({n_true_abstain:>2} cut) | {util_l1:>+8.4f}  | {util_l2:>+8.4f}")
        gate_summary.append({
            "mode": m_name,
            "tau": tau,
            "answered": n_ans,
            "coverage": round(cov, 4),
            "correct": n_corr,
            "wrong_remaining": n_wrong,
            "true_abstentions_hallucinations_caught": n_true_abstain,
            "false_abstentions_correct_lost": n_false_abstain,
            "selective_accuracy": round(sel_acc, 4),
            "utility_lambda_1": round(util_l1, 4),
            "utility_lambda_2": round(util_l2, 4),
        })

    out_payload = {
        "total_questions": n_total,
        "forced_answer_summary": {
            "sft_baseline": {"solved": sft_hits, "accuracy": round(sft_hits / n_total, 4)},
            "grpo_postfix_greedy": {"solved": grpo_hits, "accuracy": round(grpo_hits / n_total, 4)},
            "module2_ponder_cst_greedy": {"solved": prim_hits, "accuracy": round(prim_hits / n_total, 4)},
            "module4_majority_vote": {"solved": maj_hits, "accuracy": round(maj_hits / n_total, 4)},
            "module4_legacy_prm_with_lens": {"solved": prm_legacy_hits, "accuracy": round(prm_legacy_hits / n_total, 4)},
            "module4_new_agreement_cst_lens_free": {"solved": lf_hits, "accuracy": round(lf_hits / n_total, 4)},
        },
        "selective_gate_summary": gate_summary,
        "triggered_13_details": [r for r in records if r["gate_triggered"]],
    }

    out_path = os.path.join(SFT_DIR, "clean125_lensfree_selective_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)
    except PermissionError:
        import tempfile
        out_path = os.path.join(tempfile.gettempdir(), "clean125_lensfree_selective_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    evaluate_clean125()
