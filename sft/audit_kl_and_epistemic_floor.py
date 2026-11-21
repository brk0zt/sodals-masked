"""
SODALS v2.1 — Phase 2 GPU-Free Forensic Audit Script
=====================================================
Computes:
  1. LoRA weight movement (||dW||_F / ||W_0||_F) between SFT (`balanced_r4`) and GRPO (`math_train_100`),
     plus Clean-125 token/prediction divergence rate.
  2. 400-rollout Hendrycks train comparison: Avg@1, Majority@4, Pass@4, and 4-bin Reliability Table
     across agreement levels (4/4, 3/4, 2/4, 1/4), plus forensic classification of the 9 >=3/4 consistent wrongs.
  3. Clean-125 + Pristine-20 + Pristine-30 (Clean-175 & Pristine-50) unified Exact Binomial McNemar tests
     using unique_id-verified SFT Full-500 baselines.
"""

import json
import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from safetensors.torch import load_file
import torch

from cst_symbolic_verifier import NormalizedMathMatcher
from env_reproducibility import collect_runtime_metadata


def exact_mcnemar_two_sided(n01: int, n10: int) -> float:
    """Exact two-sided binomial McNemar p-value for discordant pairs (n01, n10)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p_one_side = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return min(1.0, 2.0 * p_one_side)


def safe_write_report(payload: Dict[str, Any], primary_path: str, fallback_path: str) -> str:
    for target in (primary_path, fallback_path):
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return target
        except PermissionError:
            continue
    raise PermissionError(f"Could not write report to {primary_path} or {fallback_path}")


def audit_weight_and_trajectory_movement() -> Dict[str, Any]:
    p_sft = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4", "adapter_model.safetensors")
    p_grpo = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100", "adapter_model.safetensors")

    w_sft = load_file(p_sft)
    w_grpo = load_file(p_grpo)
    norm_sq_0 = 0.0
    diff_sq = 0.0
    max_layer_rel = 0.0
    max_layer_name = ""

    for k, t0_raw in w_sft.items():
        if k not in w_grpo:
            continue
        t0 = t0_raw.float()
        t1 = w_grpo[k].float()
        d = torch.norm(t1 - t0).item()
        n0 = torch.norm(t0).item()
        norm_sq_0 += n0 ** 2
        diff_sq += d ** 2
        if n0 > 1e-8 and (d / n0) > max_layer_rel:
            max_layer_rel = d / n0
            max_layer_name = k

    rel_frob = (diff_sq ** 0.5) / max(1e-8, (norm_sq_0 ** 0.5))

    pf_path = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")
    pf = json.load(open(pf_path, encoding="utf-8"))
    det = [d for d in pf.get("details", []) if 26 <= d.get("index", 0) <= 150]
    pred_diff_idx = [d["index"] for d in det if str(d.get("predicted")) != str(d.get("baseline_pred"))]
    tok_diff_idx = [d["index"] for d in det if d.get("generated_tokens") != d.get("base_tokens")]
    abs_tok_deltas = [abs(int(d.get("generated_tokens", 0)) - int(d.get("base_tokens", 0))) for d in det]

    return {
        "global_rel_frobenius_drift": round(rel_frob, 6),
        "global_rel_frobenius_pct": round(rel_frob * 100.0, 4),
        "max_layer_rel_drift_pct": round(max_layer_rel * 100.0, 4),
        "max_layer_name": max_layer_name,
        "clean125_total": len(det),
        "clean125_pred_changed_count": len(pred_diff_idx),
        "clean125_pred_changed_indices": pred_diff_idx,
        "clean125_token_len_changed_count": len(tok_diff_idx),
        "clean125_mean_abs_token_delta": round(sum(abs_tok_deltas) / max(1, len(abs_tok_deltas)), 2),
    }


def audit_400_rollouts_and_reliability() -> Dict[str, Any]:
    g_path = os.path.join(SFT_DIR, "grpo_math_train_100_results.json")
    g = json.load(open(g_path, encoding="utf-8"))
    steps = g.get("step_details", [])

    total_rollouts = 0
    total_rollouts_ok = 0
    pass_at_4 = 0
    maj_at_4 = 0

    bins_prompt: Dict[int, Dict[str, int]] = {4: {"n": 0, "ok": 0}, 3: {"n": 0, "ok": 0}, 2: {"n": 0, "ok": 0}, 1: {"n": 0, "ok": 0}}
    consistent_wrongs = []

    for s in steps:
        gt = str(s.get("ground_truth", ""))
        preds = [str(x) for x in s.get("rollout_preds", [])]
        flags = [bool(NormalizedMathMatcher.check_match(p, gt)[1]) for p in preds]

        total_rollouts += len(preds)
        total_rollouts_ok += sum(flags)
        if any(flags):
            pass_at_4 += 1

        clusters: List[Dict[str, Any]] = []
        for idx_p, p in enumerate(preds):
            placed = False
            for c in clusters:
                if NormalizedMathMatcher.check_match(p, c["rep"])[1]:
                    c["count"] += 1
                    c["indices"].append(idx_p)
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": p, "count": 1, "ok": flags[idx_p], "indices": [idx_p]})

        clusters.sort(key=lambda x: (-x["count"], x["indices"][0]))
        top_c = clusters[0]
        if top_c["ok"]:
            maj_at_4 += 1

        max_ag = min(4, max(1, int(top_c["count"])))
        bins_prompt[max_ag]["n"] += 1
        bins_prompt[max_ag]["ok"] += int(bool(top_c["ok"]))

        if max_ag >= 3 and not top_c["ok"]:
            # Distinguish likely decimal-extractor truncation from true math errors
            gt_has_dec = ("." in gt)
            preds_truncated_int = all(p.lstrip("-").isdigit() for p in preds) and gt_has_dec
            if preds_truncated_int:
                category = "SUSPECTED_EXTRACTOR_DECIMAL_TRUNCATION"
            elif any(flags):
                category = "SOLVABLE_IN_POOL_PARTIAL_MISS (1/4 correct)"
            else:
                category = "TRUE_CONSISTENT_ERROR (0/4 correct)"

            consistent_wrongs.append({
                "step": s["step"],
                "subject": s["subject"],
                "ground_truth": gt,
                "rollout_preds": preds,
                "rollout_tokens": s.get("rollout_tokens", []),
                "solved_count": sum(flags),
                "max_cluster_size": max_ag,
                "forensic_category": category,
            })

    reliability_table = []
    cum_n = 0
    cum_ok = 0
    for ag in (4, 3, 2, 1):
        bn = bins_prompt[ag]["n"]
        bok = bins_prompt[ag]["ok"]
        cum_n += bn
        cum_ok += bok
        reliability_table.append({
            "agreement_bin": f"{ag}/4",
            "bin_prompts": bn,
            "bin_majority_correct": bok,
            "bin_accuracy_pct": round(100.0 * bok / max(1, bn), 2),
            "cumulative_threshold": f">={ag}/4",
            "cumulative_coverage_pct": round(100.0 * cum_n / max(1, len(steps)), 2),
            "cumulative_selective_accuracy_pct": round(100.0 * cum_ok / max(1, cum_n), 2),
        })

    return {
        "total_prompts": len(steps),
        "total_rollouts": total_rollouts,
        "avg_at_1_correct": total_rollouts_ok,
        "avg_at_1_acc_pct": round(100.0 * total_rollouts_ok / max(1, total_rollouts), 2),
        "majority_at_4_correct": maj_at_4,
        "majority_at_4_acc_pct": round(100.0 * maj_at_4 / max(1, len(steps)), 2),
        "pass_at_4_correct": pass_at_4,
        "pass_at_4_acc_pct": round(100.0 * pass_at_4 / max(1, len(steps)), 2),
        "reliability_table_4_bins": reliability_table,
        "consistent_wrongs_total": len(consistent_wrongs),
        "consistent_wrongs_breakdown": {
            "suspected_decimal_extractor_truncation": sum(
                1 for x in consistent_wrongs if x["forensic_category"].startswith("SUSPECTED")
            ),
            "solvable_1_of_4_correct": sum(
                1 for x in consistent_wrongs if x["forensic_category"].startswith("SOLVABLE")
            ),
            "true_0_of_4_errors": sum(
                1 for x in consistent_wrongs if x["forensic_category"].startswith("TRUE")
            ),
        },
        "consistent_wrongs_details": consistent_wrongs,
    }


def audit_clean175_and_pristine50() -> Dict[str, Any]:
    full500_path = os.path.join(ROOT, "math500_full500_lora_4096_results.json")
    full500_by_uid = {}
    if os.path.exists(full500_path):
        f5 = json.load(open(full500_path, encoding="utf-8"))
        for item in f5.get("details", []):
            uid = item.get("unique_id")
            if uid:
                full500_by_uid[uid] = item

    # Clean-125 (Q026..Q150)
    m4_path = os.path.join(SFT_DIR, "eval_math500_heldout_module4_lensfree_production_results.json")
    m4 = json.load(open(m4_path, encoding="utf-8"))
    c125_det = m4.get("details", [])

    # Pristine-20 (Q151..Q170)
    p20_path = os.path.join(SFT_DIR, "pristine20_arkhe_lensfree_selective_results.json")
    p20_det = json.load(open(p20_path, encoding="utf-8")).get("details", [])

    # Pristine-30 (Q171..Q200)
    p30_path = os.path.join(tempfile.gettempdir(), "pristine30_q171_q200_arkhe_results.json")
    p30_det = json.load(open(p30_path, encoding="utf-8")).get("details", [])

    # Build aligned tuples (sft_ok, arkhe_forced_ok, arkhe_answered, arkhe_sel_ok)
    p50_records = []
    for r in p20_det:
        uid = r.get("unique_id")
        base_item = full500_by_uid.get(uid, {})
        gt = r.get("ground_truth", "")
        base_pred = str(base_item.get("predicted", ""))
        base_ok = (base_item.get("verdict") == "PASS") or bool(NormalizedMathMatcher.check_match(base_pred, gt)[1])
        p50_records.append({
            "index": r["index"],
            "unique_id": uid,
            "sft_ok": bool(base_ok),
            "arkhe_forced_ok": bool(r.get("forced_pass", False)),
            "abstained": bool(r.get("abstained", False)),
        })

    for r in p30_det:
        p50_records.append({
            "index": r["index"],
            "unique_id": r.get("unique_id"),
            "sft_ok": bool(r.get("baseline_correct", False)),
            "arkhe_forced_ok": bool(r.get("arkhe_forced_correct", False)),
            "abstained": (r.get("arkhe_verdict") == "ABSTAIN_UNKNOWN"),
        })

    # Clean-125 aligned under identical current NormalizedMathMatcher
    pf_path = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")
    pf_det = json.load(open(pf_path, encoding="utf-8")).get("details", [])
    pf_by_idx = {x["index"]: x for x in pf_det}

    c125_records = []
    for r in c125_det:
        q_idx = r["index"]
        gt = str(r.get("ground_truth", ""))
        base_pred = str(pf_by_idx.get(q_idx, {}).get("baseline_pred", ""))
        ark_pred = str(r.get("predicted", ""))
        c125_records.append({
            "index": q_idx,
            "unique_id": r.get("unique_id"),
            "sft_ok": bool(NormalizedMathMatcher.check_match(base_pred, gt)[1]),
            "arkhe_forced_ok": bool(NormalizedMathMatcher.check_match(ark_pred, gt)[1]),
            "abstained": False,
        })
    # Read abstained count from clean125_lensfree_selective_results.json
    c125_sel = json.load(open(os.path.join(SFT_DIR, "clean125_lensfree_selective_results.json"), encoding="utf-8"))
    c125_sel_gate = [x for x in c125_sel["selective_gate_summary"] if abs(x.get("tau", -1) - 0.2) < 1e-6][0]

    def summarize_pair(recs: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        n11 = sum(1 for r in recs if r["sft_ok"] and r["arkhe_forced_ok"])
        n01 = sum(1 for r in recs if (not r["sft_ok"]) and r["arkhe_forced_ok"])
        n10 = sum(1 for r in recs if r["sft_ok"] and (not r["arkhe_forced_ok"]))
        n00 = sum(1 for r in recs if (not r["sft_ok"]) and (not r["arkhe_forced_ok"]))
        sft_cnt = n11 + n10
        ark_cnt = n11 + n01
        ans_recs = [r for r in recs if not r["abstained"]]
        sel_ok = sum(1 for r in ans_recs if r["arkhe_forced_ok"])
        return {
            "pool": label,
            "n": len(recs),
            "sft_baseline_correct": sft_cnt,
            "sft_baseline_acc_pct": round(100.0 * sft_cnt / max(1, len(recs)), 2),
            "arkhe_forced_correct": ark_cnt,
            "arkhe_forced_acc_pct": round(100.0 * ark_cnt / max(1, len(recs)), 2),
            "net_gain_forced": ark_cnt - sft_cnt,
            "discordant_gains_n01": n01,
            "discordant_losses_n10": n10,
            "mcnemar_two_sided_p": round(exact_mcnemar_two_sided(n01, n10), 6),
            "selective_answered": len(ans_recs),
            "selective_correct": sel_ok,
            "selective_coverage_pct": round(100.0 * len(ans_recs) / max(1, len(recs)), 2),
            "selective_acc_pct": round(100.0 * sel_ok / max(1, len(ans_recs)), 2),
        }

    c125_sum = summarize_pair(c125_records, "Clean-125 (Q026..Q150)")
    c125_sum["selective_answered"] = c125_sel_gate["answered"]
    c125_sum["selective_correct"] = c125_sel_gate["correct"]
    c125_sum["selective_coverage_pct"] = round(c125_sel_gate["coverage"] * 100.0, 2)
    c125_sum["selective_acc_pct"] = round(c125_sel_gate["selective_accuracy"] * 100.0, 2)

    p50_sum = summarize_pair(p50_records, "Pristine-50 Unseen (Q151..Q200)")

    all_175 = c125_records + p50_records
    c175_sum = summarize_pair(all_175, "Clean-175 Combined (Q026..Q200)")
    c175_sum["selective_answered"] = c125_sum["selective_answered"] + p50_sum["selective_answered"]
    c175_sum["selective_correct"] = c125_sum["selective_correct"] + p50_sum["selective_correct"]
    c175_sum["selective_coverage_pct"] = round(100.0 * c175_sum["selective_answered"] / 175.0, 2)
    c175_sum["selective_acc_pct"] = round(100.0 * c175_sum["selective_correct"] / c175_sum["selective_answered"], 2)

    return {
        "clean_125": c125_sum,
        "pristine_50_unseen": p50_sum,
        "clean_175_combined": c175_sum,
    }


def main():
    report = {
        "runtime_metadata": collect_runtime_metadata(),
        "weight_and_trajectory_audit": audit_weight_and_trajectory_movement(),
        "rollouts_400_and_reliability_audit": audit_400_rollouts_and_reliability(),
        "heldout_pools_mcnemar_summary": audit_clean175_and_pristine50(),
    }

    primary = os.path.join(SFT_DIR, "FORENSIC_KL_AND_EPISTEMIC_AUDIT_REPORT.json")
    fallback = os.path.join(tempfile.gettempdir(), "FORENSIC_KL_AND_EPISTEMIC_AUDIT_REPORT.json")
    out_path = safe_write_report(report, primary, fallback)

    print("=" * 84)
    print("PHASE 2 FORENSIC AUDIT COMPLETE")
    print("Saved report:", out_path)
    print("-" * 84)
    w = report["weight_and_trajectory_audit"]
    print(f"[1] Weight Drift (balanced_r4 -> grpo_100): ||dW||/||W0|| = {w['global_rel_frobenius_pct']:.4f}% (Max Layer: {w['max_layer_rel_drift_pct']:.4f}%)")
    print(f"    Clean-125 Divergence: Pred changed {w['clean125_pred_changed_count']}/125 | Token len changed {w['clean125_token_len_changed_count']}/125 (mean |dTok|={w['clean125_mean_abs_token_delta']})")
    r = report["rollouts_400_and_reliability_audit"]
    print(f"[2] 400 Rollouts: Avg@1={r['avg_at_1_acc_pct']}% | Majority@4={r['majority_at_4_acc_pct']}% | Pass@4={r['pass_at_4_acc_pct']}%")
    print("    4-Bin Agreement Reliability Table:")
    for row in r["reliability_table_4_bins"]:
        print(f"      {row['agreement_bin']:>3s}: {row['bin_majority_correct']:2d}/{row['bin_prompts']:2d} ({row['bin_accuracy_pct']:5.2f}%) | Cum {row['cumulative_threshold']}: {row['cumulative_selective_accuracy_pct']:5.2f}% @ {row['cumulative_coverage_pct']:5.2f}% cov")
    print("    9 Consistent Wrongs Breakdown:", r["consistent_wrongs_breakdown"])
    print("[3] Held-Out McNemar Summary:")
    for key in ("clean_125", "pristine_50_unseen", "clean_175_combined"):
        s = report["heldout_pools_mcnemar_summary"][key]
        print(
            f"    {s['pool']:32s} | SFT: {s['sft_baseline_correct']}/{s['n']} ({s['sft_baseline_acc_pct']:.2f}%) -> "
            f"Arkhe Forced: {s['arkhe_forced_correct']}/{s['n']} ({s['arkhe_forced_acc_pct']:.2f}%) "
            f"[+{s['discordant_gains_n01']}/-{s['discordant_losses_n10']}, p={s['mcnemar_two_sided_p']:.4f}] | "
            f"Sel: {s['selective_correct']}/{s['selective_answered']} ({s['selective_acc_pct']:.2f}% @ {s['selective_coverage_pct']:.2f}% cov)"
        )
    print("=" * 84)


if __name__ == "__main__":
    main()
