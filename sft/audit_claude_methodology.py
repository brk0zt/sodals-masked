"""
SODALS v2.1 — GPU-Free Methodology Freeze & Complete Scientific Audit
=====================================================================
Addresses every methodological audit requirement before running confirmatory evaluations:
  1. Single-Grader Symmetrical Rescore (`NormalizedMathMatcher.check_match`):
     Re-grades Baseline SFT, Clean-125, Pristine-20, and Pristine-30 (plus reserve count
     for Q201..Q500 without peeking at errors) under the exact same frozen grader.
  2. Tuple-Truthiness Clean-125 Verification:
     Re-runs `ArkheNeurosymbolicPipeline(matcher_fn=NormalizedMathMatcher.check_match)` over
     stored Clean-125 candidate pools to confirm whether 107/125 & 107/120 hold cleanly.
  3. Exact Paired Flip Decomposition on Pristine-30 (Baseline -> Pass-1 -> Arkhe Forced).
  4. `{contras > 0} x {correct / wrong}` 2x2 Contingency Table on Clean-125:
     Compares legacy stored `primary_contras` (`abs > 1e-4`) against upgraded `SymbolicStepChecker`
     (`is_numeric_step_mismatch` relative tolerance + trig).
  5. Seeded 30-item Random PASS Audit across all arms to check for false positives.
"""

import hashlib
import json
import math
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Tuple

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from arkhe_neurosymbolic_pipeline import (
    ArkheNeurosymbolicPipeline,
    CandidateTrace,
    DecisionVerdict,
)
from cst_symbolic_verifier import NormalizedMathMatcher, SymbolicStepChecker
from env_reproducibility import collect_runtime_metadata

BASELINE_JSON = os.path.join(ROOT_DIR, "math500_full500_lora_4096_results.json")
CLEAN125_JSON = os.path.join(SFT_DIR, "eval_math500_heldout_module4_lensfree_production_results.json")
PRISTINE20_JSON = os.path.join(SFT_DIR, "pristine20_arkhe_lensfree_selective_results.json")
PRISTINE30_JSON = (
    os.path.join(SFT_DIR, "pristine30_q171_q200_arkhe_results.json")
    if os.path.exists(os.path.join(SFT_DIR, "pristine30_q171_q200_arkhe_results.json"))
    else os.path.join(tempfile.gettempdir(), "pristine30_q171_q200_arkhe_results.json")
)
OUT_JSON = os.path.join(SFT_DIR, "CLAUDE_METHODOLOGY_FREEZE_AUDIT.json")
FALLBACK_OUT_JSON = os.path.join(tempfile.gettempdir(), "CLAUDE_METHODOLOGY_FREEZE_AUDIT.json")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _exact_mcnemar_p(b_wins: int, c_losses: int) -> float:
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** (n - 1))
    return min(1.0, float(p))


def _paired_counts(flags_a: List[bool], flags_b: List[bool]) -> Dict[str, Any]:
    gains = sum(1 for a, b in zip(flags_a, flags_b) if (not a) and b)
    losses = sum(1 for a, b in zip(flags_a, flags_b) if a and (not b))
    return {
        "a_correct": sum(flags_a),
        "b_correct": sum(flags_b),
        "gains_a0_b1": gains,
        "losses_a1_b0": losses,
        "net": gains - losses,
        "exact_mcnemar_p": round(_exact_mcnemar_p(gains, losses), 4),
    }


def main():
    code_hashes = {
        fn: _sha256_file(os.path.join(SFT_DIR, fn))
        for fn in (
            "cst_symbolic_verifier.py",
            "arkhe_neurosymbolic_pipeline.py",
            "eval_pristine30_q171_q200_arkhe_pipeline.py",
            "train_grpo_math_train_100.py",
        )
        if os.path.exists(os.path.join(SFT_DIR, fn))
    }

    # 1. Symmetrical Frozen-Grader Rescore of Baseline SFT (Full-500)
    base_data = json.load(open(BASELINE_JSON, "r", encoding="utf-8"))
    base_by_uid = {}
    base_by_idx = {}
    base_rescore_flips = []
    for row in base_data["details"]:
        idx = int(row["index"])
        uid = row.get("unique_id", "")
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("predicted", ""))
        old_ok = row.get("verdict") == "PASS"
        frozen_ok = bool(NormalizedMathMatcher.check_match(pred, gt)[1])
        if frozen_ok != old_ok:
            base_rescore_flips.append(
                {
                    "index": idx,
                    "unique_id": uid,
                    "gt": gt,
                    "pred": pred,
                    "old_ok": old_ok,
                    "frozen_ok": frozen_ok,
                }
            )
        rec = dict(row)
        rec["frozen_ok"] = frozen_ok
        rec["old_ok"] = old_ok
        base_by_idx[idx] = rec
        if uid:
            base_by_uid[uid] = rec

    def _base_slice_summary(lo: int, hi: int) -> Dict[str, Any]:
        rows = [base_by_idx[i] for i in range(lo, hi + 1) if i in base_by_idx]
        old_c = sum(1 for r in rows if r["old_ok"])
        frz_c = sum(1 for r in rows if r["frozen_ok"])
        return {
            "n": len(rows),
            "old_correct": old_c,
            "old_acc_pct": round(100.0 * old_c / max(1, len(rows)), 2),
            "frozen_correct": frz_c,
            "frozen_acc_pct": round(100.0 * frz_c / max(1, len(rows)), 2),
        }

    # 2. Clean-125 Audit (Symmetrical Rescore + Tuple-Truthiness Pipeline Verification + 2x2 Contra Table)
    c125_data = json.load(open(CLEAN125_JSON, "r", encoding="utf-8"))
    c125_rows = c125_data["details"]
    pipe = ArkheNeurosymbolicPipeline(selective_tau=0.20, matcher_fn=NormalizedMathMatcher.check_match)

    c125_base_flags: List[bool] = []
    c125_prim_flags: List[bool] = []
    c125_forced_flags: List[bool] = []
    c125_sel_answered = 0
    c125_sel_correct = 0
    tuple_audit_mismatches = []
    pass_candidates_pool: List[Dict[str, Any]] = []

    legacy_2x2 = {"TP_contra_wrong": 0, "FP_contra_correct": 0, "FN_clean_wrong": 0, "TN_clean_correct": 0}
    upgraded_2x2 = {"TP_contra_wrong": 0, "FP_contra_correct": 0, "FN_clean_wrong": 0, "TN_clean_correct": 0}
    legacy_fp_examples = []
    upgraded_fp_examples = []

    for r in c125_rows:
        idx = int(r["index"])
        uid = r.get("unique_id", "")
        gt = str(r.get("ground_truth", ""))
        prob = str(r.get("problem", ""))
        full_resp = str(r.get("full_response", ""))

        b_row = base_by_uid.get(uid) or base_by_idx.get(idx)
        c125_base_flags.append(bool(b_row["frozen_ok"]) if b_row else False)

        prim_pred = str(r.get("primary_pred", ""))
        forced_pred = str(r.get("lens_free_pred", r.get("predicted", "")))
        prim_ok = bool(NormalizedMathMatcher.check_match(prim_pred, gt)[1])
        forced_ok = bool(NormalizedMathMatcher.check_match(forced_pred, gt)[1])
        c125_prim_flags.append(prim_ok)
        c125_forced_flags.append(forced_ok)

        # Re-run Six-Arkhe pipeline directly on candidate_pool to audit tuple-truthiness integrity
        p1_trace = CandidateTrace(
            branch_id="Primary",
            extracted_answer=prim_pred,
            has_end_tag=not bool(r.get("primary_abandonment", False)),
            is_symbolically_valid=bool(r.get("primary_valid", True)),
            contradictions=tuple(str(c) for c in r.get("primary_contras", [])),
            generated_tokens=int(r.get("generated_tokens", 0)),
        )
        cand_pool_raw = r.get("candidate_pool") or []
        cand_traces = [
            CandidateTrace(
                branch_id=str(c.get("branch", f"Cand{j+1}")),
                extracted_answer=str(c.get("pred", "")),
                has_end_tag=bool(c.get("has_end", True)),
                is_symbolically_valid=bool(c.get("is_valid", True)),
                verified_eq_count=int(c.get("eq_count", 0)),
                contradictions=tuple(str(x) for x in c.get("contras", [])),
                doubt_count=int(c.get("wait_cnt", 0)),
            )
            for j, c in enumerate(cand_pool_raw)
        ]
        recomputed = pipe.run_from_traces(
            question_id=f"Q{idx:03d}",
            pass1_trace=p1_trace,
            candidate_traces=cand_traces,
            gate_triggered=bool(r.get("gate_triggered", False) and len(cand_traces) > 0),
            prompt=prob,
        )
        stored_abstained = bool(r.get("is_abstained", False))
        if (recomputed.verdict == DecisionVerdict.ABSTAIN_UNKNOWN) != stored_abstained:
            tuple_audit_mismatches.append(
                {
                    "index": idx,
                    "stored_abstained": stored_abstained,
                    "recomputed_verdict": recomputed.verdict.value,
                    "recomputed_max_agree": recomputed.max_agreement,
                }
            )
        if not stored_abstained:
            c125_sel_answered += 1
            if bool(NormalizedMathMatcher.check_match(str(r.get("selective_pred", forced_pred)), gt)[1]):
                c125_sel_correct += 1

        # 2x2 Contingency Table on Primary response: {contras > 0} x {correct / wrong}
        leg_has_contra = len(r.get("primary_contras") or []) > 0
        if prim_ok:
            if leg_has_contra:
                legacy_2x2["FP_contra_correct"] += 1
                if len(legacy_fp_examples) < 5:
                    legacy_fp_examples.append({"index": idx, "contras": r.get("primary_contras")})
            else:
                legacy_2x2["TN_clean_correct"] += 1
        else:
            if leg_has_contra:
                legacy_2x2["TP_contra_wrong"] += 1
            else:
                legacy_2x2["FN_clean_wrong"] += 1

        rep_up = SymbolicStepChecker.verify_proof_steps(full_resp, prompt=prob, include_sorry_in_contradictions=False)
        up_has_contra = len(rep_up.contradictions) > 0
        if prim_ok:
            if up_has_contra:
                upgraded_2x2["FP_contra_correct"] += 1
                if len(upgraded_fp_examples) < 5:
                    upgraded_fp_examples.append({"index": idx, "contras": list(rep_up.contradictions)})
            else:
                upgraded_2x2["TN_clean_correct"] += 1
        else:
            if up_has_contra:
                upgraded_2x2["TP_contra_wrong"] += 1
            else:
                upgraded_2x2["FN_clean_wrong"] += 1

        if b_row and b_row["frozen_ok"]:
            pass_candidates_pool.append({"arm": "Baseline_SFT", "index": idx, "uid": uid, "gt": gt, "pred": b_row["predicted"]})
        if forced_ok:
            pass_candidates_pool.append({"arm": "Clean125_Arkhe_Forced", "index": idx, "uid": uid, "gt": gt, "pred": forced_pred})

    # 3. Pristine-20 Audit
    p20_data = json.load(open(PRISTINE20_JSON, "r", encoding="utf-8"))
    p20_rows = p20_data["details"]
    p20_base_flags, p20_p1_flags, p20_forced_flags = [], [], []
    p20_sel_ans, p20_sel_ok = 0, 0
    for r in p20_rows:
        idx = int(r["index"])
        uid = r.get("unique_id", "")
        gt = str(r.get("ground_truth", ""))
        b_row = base_by_uid.get(uid) or base_by_idx.get(idx)
        b_ok = bool(b_row["frozen_ok"]) if b_row else False
        p1_pred = str(r.get("pass1_fixed_pred", ""))
        f_pred = str(r.get("forced_pred", p1_pred))
        p1_ok = bool(NormalizedMathMatcher.check_match(p1_pred, gt)[1])
        f_ok = bool(NormalizedMathMatcher.check_match(f_pred, gt)[1])
        p20_base_flags.append(b_ok)
        p20_p1_flags.append(p1_ok)
        p20_forced_flags.append(f_ok)
        if not bool(r.get("abstained", False)):
            p20_sel_ans += 1
            if bool(NormalizedMathMatcher.check_match(str(r.get("final_answer", f_pred)), gt)[1]):
                p20_sel_ok += 1
        if f_ok:
            pass_candidates_pool.append({"arm": "Pristine20_Arkhe_Forced", "index": idx, "uid": uid, "gt": gt, "pred": f_pred})

    # 4. Pristine-30 Audit
    p30_data = json.load(open(PRISTINE30_JSON, "r", encoding="utf-8"))
    p30_rows = p30_data["details"]
    p30_base_flags, p30_p1_flags, p30_forced_flags = [], [], []
    p30_sel_ans, p30_sel_ok = 0, 0
    for r in p30_rows:
        idx = int(r["index"])
        uid = r.get("unique_id", "")
        gt = str(r.get("ground_truth", ""))
        b_row = base_by_uid.get(uid) or base_by_idx.get(idx)
        b_ok = bool(b_row["frozen_ok"]) if b_row else False
        p1_pred = str(r.get("pass1_predicted", ""))
        f_pred = str(r.get("winning_predicted", p1_pred))
        p1_ok = bool(NormalizedMathMatcher.check_match(p1_pred, gt)[1])
        f_ok = bool(NormalizedMathMatcher.check_match(f_pred, gt)[1])
        p30_base_flags.append(b_ok)
        p30_p1_flags.append(p1_ok)
        p30_forced_flags.append(f_ok)
        if r.get("arkhe_verdict") != DecisionVerdict.ABSTAIN_UNKNOWN.name:
            p30_sel_ans += 1
            if f_ok:
                p30_sel_ok += 1
        if b_ok:
            pass_candidates_pool.append({"arm": "Baseline_SFT", "index": idx, "uid": uid, "gt": gt, "pred": b_row["predicted"]})
        if f_ok:
            pass_candidates_pool.append({"arm": "Pristine30_Arkhe_Forced", "index": idx, "uid": uid, "gt": gt, "pred": f_pred})

    # 5. Sample 30 Random PASS predictions across arms for false-positive human verification
    rng = random.Random(42)
    pass_sample_30 = rng.sample(pass_candidates_pool, min(30, len(pass_candidates_pool)))

    audit_report = {
        "runtime_metadata": collect_runtime_metadata(),
        "frozen_code_sha256": code_hashes,
        "symmetrical_frozen_baseline_sft": {
            "full_500_old_correct": sum(1 for r in base_by_idx.values() if r["old_ok"]),
            "full_500_frozen_correct": sum(1 for r in base_by_idx.values() if r["frozen_ok"]),
            "clean125_q026_q150": _base_slice_summary(26, 150),
            "pristine20_q151_q170": _base_slice_summary(151, 170),
            "pristine30_q171_q200": _base_slice_summary(171, 200),
            "pristine50_q151_q200": _base_slice_summary(151, 200),
            "confirmatory_reserve_q201_q500_count_only": _base_slice_summary(201, 500),
            "baseline_flips_on_q001_q200": [x for x in base_rescore_flips if x["index"] <= 200],
            "baseline_flips_on_q201_q500_count_only": len([x for x in base_rescore_flips if x["index"] > 200]),
        },
        "clean125_audit": {
            "n": len(c125_rows),
            "baseline_sft_frozen": sum(c125_base_flags),
            "primary_ponder_frozen": sum(c125_prim_flags),
            "arkhe_forced_frozen": sum(c125_forced_flags),
            "arkhe_selective": {"correct": c125_sel_correct, "answered": c125_sel_answered},
            "paired_arkhe_forced_vs_baseline": _paired_counts(c125_base_flags, c125_forced_flags),
            "paired_arkhe_forced_vs_primary": _paired_counts(c125_prim_flags, c125_forced_flags),
            "tuple_truthiness_recompute_mismatches": tuple_audit_mismatches,
        },
        "pristine20_audit": {
            "n": len(p20_rows),
            "baseline_sft_frozen": sum(p20_base_flags),
            "pass1_fixed_frozen": sum(p20_p1_flags),
            "arkhe_forced_frozen": sum(p20_forced_flags),
            "arkhe_selective": {"correct": p20_sel_ok, "answered": p20_sel_ans},
            "paired_arkhe_forced_vs_baseline": _paired_counts(p20_base_flags, p20_forced_flags),
        },
        "pristine30_audit": {
            "n": len(p30_rows),
            "baseline_sft_frozen": sum(p30_base_flags),
            "pass1_greedy_frozen": sum(p30_p1_flags),
            "arkhe_forced_frozen": sum(p30_forced_flags),
            "arkhe_selective": {"correct": p30_sel_ok, "answered": p30_sel_ans},
            "paired_baseline_to_pass1": _paired_counts(p30_base_flags, p30_p1_flags),
            "paired_pass1_to_arkhe": _paired_counts(p30_p1_flags, p30_forced_flags),
            "paired_baseline_to_arkhe": _paired_counts(p30_base_flags, p30_forced_flags),
        },
        "pristine50_combined_audit": {
            "n": len(p20_rows) + len(p30_rows),
            "baseline_sft_frozen": sum(p20_base_flags) + sum(p30_base_flags),
            "pass1_frozen": sum(p20_p1_flags) + sum(p30_p1_flags),
            "arkhe_forced_frozen": sum(p20_forced_flags) + sum(p30_forced_flags),
            "arkhe_selective": {"correct": p20_sel_ok + p30_sel_ok, "answered": p20_sel_ans + p30_sel_ans},
            "paired_arkhe_forced_vs_baseline": _paired_counts(p20_base_flags + p30_base_flags, p20_forced_flags + p30_forced_flags),
        },
        "contras_2x2_table_clean125": {
            "legacy_stored_contras": legacy_2x2,
            "legacy_fp_examples": legacy_fp_examples,
            "upgraded_symbolic_step_checker": upgraded_2x2,
            "upgraded_fp_examples": upgraded_fp_examples,
        },
        "random_30_pass_spot_check": pass_sample_30,
    }

    saved_path = OUT_JSON
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)
    except PermissionError:
        saved_path = FALLBACK_OUT_JSON
        with open(FALLBACK_OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"CLAUDE METHODOLOGY FREEZE AUDIT SAVED: {saved_path}")
    print("=" * 80)
    print("Baseline SFT (Symmetrical Frozen Rescore):", json.dumps(audit_report["symmetrical_frozen_baseline_sft"], indent=2))
    print("Clean-125 Audit:", json.dumps(audit_report["clean125_audit"], indent=2))
    print("Pristine-50 Combined:", json.dumps(audit_report["pristine50_combined_audit"], indent=2))
    print("Pristine-30 Flip Breakdown:", json.dumps(audit_report["pristine30_audit"], indent=2))
    print("2x2 Contra Table (Clean-125):", json.dumps(audit_report["contras_2x2_table_clean125"], indent=2))
    print("\n--- 30 RANDOM PASS ITEMS (False-Positive Spot Check) ---")
    for i, item in enumerate(pass_sample_30, start=1):
        print(f"  {i:2d}. [{item['arm']}] Q{item['index']:03d} | GT={item['gt']!r} | Pred={item['pred']!r}")


if __name__ == "__main__":
    main()
