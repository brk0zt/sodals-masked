"""
SODALS v2.1 — GPU-Free Phase 2 Methodology & Paper Tables Exporter
====================================================================
Implements Claude's GPU-free audit requirements:
  1. Blind Human Cross-Arm Discordant Audit (`BLIND_HUMAN_DISCORDANT_AUDIT.md`):
     Exports EVERY discordant question between arms (`Baseline vs Pass-1`, `Pass-1 vs Arkhe`,
     `Baseline vs Arkhe`) across Clean-125, Pristine-20, and Pristine-30 with anonymized,
     seed-shuffled candidate labels (Candidate_A / Candidate_B) — zero LLM self-judging.
  2. Complete (Arm x Grader Version) Reconciliation Matrix (`CLAUDE_REVISION_AUDIT_TABLES.json`):
     Explicitly explains Baseline Clean-125 across all 4 historical grader versions
     (`99` Legacy SymPy, `100` Stored Full500, `103` Early NormalizedMathMatcher, `104` Frozen)
     while locking primary metrics to the single frozen NormalizedMathMatcher SHA-256.
  3. Epistemic Floor Reclassification:
     Splits true `0/4` solvability floor (`Pass@4` floor = 5.0%) from plurality failure floor
     (`Majority@4` floor >= 7.0%, since Steps 31 & 64 have 1/4 correct vs 3/4 wrong cluster).
  4. Re-frames Legacy GRPO Null Result honestly in `FORENSIC_KL_AND_EPISTEMIC_AUDIT_REPORT.json`
     (`policy did not move / effect unmeasured`, rather than `0 regression / p=1.0`).
"""

import hashlib
import json
import os
import random
import sys
import tempfile
from typing import Any, Dict, List

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from cst_symbolic_verifier import NormalizedMathMatcher
from env_reproducibility import collect_runtime_metadata

BASELINE_JSON = os.path.join(ROOT_DIR, "math500_full500_lora_4096_results.json")
CLEAN125_M4_JSON = os.path.join(SFT_DIR, "eval_math500_heldout_module4_lensfree_production_results.json")
CLEAN125_PF_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")
CLEAN125_M125_JSON = os.path.join(SFT_DIR, "math500_heldout_clean_125_results.json")
PRISTINE20_JSON = os.path.join(SFT_DIR, "pristine20_arkhe_lensfree_selective_results.json")
PRISTINE30_JSON = (
    os.path.join(SFT_DIR, "pristine30_q171_q200_arkhe_results.json")
    if os.path.exists(os.path.join(SFT_DIR, "pristine30_q171_q200_arkhe_results.json"))
    else os.path.join(tempfile.gettempdir(), "pristine30_q171_q200_arkhe_results.json")
)
FORENSIC_JSON = os.path.join(SFT_DIR, "FORENSIC_KL_AND_EPISTEMIC_AUDIT_REPORT.json")
MATH500_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_write_text(text: str, path: str) -> str:
    for tgt in (path, os.path.join(tempfile.gettempdir(), os.path.basename(path))):
        try:
            with open(tgt, "w", encoding="utf-8") as f:
                f.write(text)
            return tgt
        except PermissionError:
            continue
    raise PermissionError(f"Could not write {path}")


def _safe_write_json(payload: Any, path: str) -> str:
    for tgt in (path, os.path.join(tempfile.gettempdir(), os.path.basename(path))):
        try:
            with open(tgt, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return tgt
        except PermissionError:
            continue
    raise PermissionError(f"Could not write {path}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # Load Baseline Full-500 & MATH-500 problem lookup (strictly Q001..Q200 for inspection)
    base_data = json.load(open(BASELINE_JSON, "r", encoding="utf-8"))
    base_by_uid = {}
    base_by_idx = {}
    for row in base_data["details"]:
        idx = int(row["index"])
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("predicted", ""))
        rec = dict(row)
        rec["old_ok"] = row.get("verdict") == "PASS"
        rec["frozen_ok"] = bool(NormalizedMathMatcher.check_match(pred, gt)[1])
        base_by_idx[idx] = rec
        if row.get("unique_id"):
            base_by_uid[row["unique_id"]] = rec

    prob_by_uid = {}
    with open(MATH500_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i <= 200:  # Strict reserve hygiene: do not even load Q201..Q500 text here
                q = json.loads(line)
                prob_by_uid[q["unique_id"]] = q

    # 1. Build Blind Discordant Pairs across Clean-125, Pristine-20, Pristine-30
    rng = random.Random(20261118)
    blind_items: List[Dict[str, Any]] = []
    hidden_keys: List[Dict[str, Any]] = []

    def _add_discordant(pool: str, idx: int, uid: str, problem: str, gt: str, arms: Dict[str, Dict[str, Any]]):
        ok_vals = {bool(v["ok"]) for v in arms.values()}
        if len(ok_vals) <= 1:
            return
        arm_list = list(arms.items())
        rng.shuffle(arm_list)
        anon_candidates = []
        mapping = {}
        labels = ["Option_A", "Option_B", "Option_C"]
        for k_idx, (arm_name, info) in enumerate(arm_list):
            lbl = labels[k_idx]
            anon_candidates.append({
                "label": lbl,
                "predicted": info["pred"],
                "grader_verdict": "PASS" if info["ok"] else "FAIL",
                "excerpt": info.get("excerpt", ""),
            })
            mapping[lbl] = {"arm": arm_name, "predicted": info["pred"], "grader_ok": bool(info["ok"])}

        item_id = f"{pool}_Q{idx:03d}"
        blind_items.append({
            "audit_id": item_id,
            "pool": pool,
            "index": idx,
            "unique_id": uid,
            "problem": problem,
            "ground_truth": gt,
            "candidates": anon_candidates,
        })
        hidden_keys.append({"audit_id": item_id, "unique_id": uid, "mapping": mapping})

    # Clean-125 (`Primary/Ponder` vs `Baseline` vs `Arkhe_Forced`)
    c125 = json.load(open(CLEAN125_M4_JSON, "r", encoding="utf-8"))
    for r in c125["details"]:
        idx = int(r["index"])
        uid = r.get("unique_id", "")
        gt = str(r.get("ground_truth", ""))
        prob = str(r.get("problem", prob_by_uid.get(uid, {}).get("problem", "")))
        b_row = base_by_uid.get(uid) or base_by_idx.get(idx)
        b_pred = str(b_row.get("predicted", "")) if b_row else ""
        p_pred = str(r.get("primary_pred", ""))
        a_pred = str(r.get("lens_free_pred", r.get("predicted", "")))
        arms = {
            "Baseline_SFT": {"pred": b_pred, "ok": bool(b_row["frozen_ok"]) if b_row else False, "excerpt": ""},
            "Primary_G1": {"pred": p_pred, "ok": bool(NormalizedMathMatcher.check_match(p_pred, gt)[1]), "excerpt": ""},
            "Arkhe_Forced": {"pred": a_pred, "ok": bool(NormalizedMathMatcher.check_match(a_pred, gt)[1]), "excerpt": ""},
        }
        _add_discordant("Clean125", idx, uid, prob, gt, arms)

    # Pristine-20 (`Baseline` vs `Pass1` vs `Arkhe_Forced`)
    p20 = json.load(open(PRISTINE20_JSON, "r", encoding="utf-8"))
    for r in p20["details"]:
        idx = int(r["index"])
        uid = r.get("unique_id", "")
        gt = str(r.get("ground_truth", ""))
        prob = prob_by_uid.get(uid, {}).get("problem", "")
        b_row = base_by_uid.get(uid) or base_by_idx.get(idx)
        b_pred = str(b_row.get("predicted", "")) if b_row else ""
        p1_pred = str(r.get("pass1_fixed_pred", ""))
        a_pred = str(r.get("forced_pred", p1_pred))
        arms = {
            "Baseline_SFT": {"pred": b_pred, "ok": bool(b_row["frozen_ok"]) if b_row else False, "excerpt": ""},
            "Pass1_Fixed": {"pred": p1_pred, "ok": bool(NormalizedMathMatcher.check_match(p1_pred, gt)[1]), "excerpt": ""},
            "Arkhe_Forced": {"pred": a_pred, "ok": bool(NormalizedMathMatcher.check_match(a_pred, gt)[1]), "excerpt": ""},
        }
        _add_discordant("Pristine20", idx, uid, prob, gt, arms)

    # Pristine-30 (`Baseline` vs `Pass1` vs `Arkhe_Forced`)
    p30 = json.load(open(PRISTINE30_JSON, "r", encoding="utf-8"))
    for r in p30["details"]:
        idx = int(r["index"])
        uid = r.get("unique_id", "")
        gt = str(r.get("ground_truth", ""))
        prob = str(r.get("problem", prob_by_uid.get(uid, {}).get("problem", "")))
        b_row = base_by_uid.get(uid) or base_by_idx.get(idx)
        b_pred = str(b_row.get("predicted", "")) if b_row else ""
        p1_pred = str(r.get("pass1_predicted", ""))
        a_pred = str(r.get("winning_predicted", p1_pred))
        arms = {
            "Baseline_SFT": {"pred": b_pred, "ok": bool(b_row["frozen_ok"]) if b_row else False, "excerpt": ""},
            "Pass1_Greedy": {"pred": p1_pred, "ok": bool(NormalizedMathMatcher.check_match(p1_pred, gt)[1]), "excerpt": ""},
            "Arkhe_Forced": {"pred": a_pred, "ok": bool(NormalizedMathMatcher.check_match(a_pred, gt)[1]), "excerpt": ""},
        }
        _add_discordant("Pristine30", idx, uid, prob, gt, arms)

    # Format Blind Human Audit Markdown
    md_lines = [
        "# Blind Cross-Arm Discordant Human Audit (`Clean-125` + `Pristine-50`)",
        "",
        "Bu dosya, yalnızca kollar arası **uyuşmazlık** oluşan tüm soruları anonimleştirilmiş (`Option_A / Option_B / Option_C` sırası rastgele karıştırılmış) biçimde sunar.",
        "Lütfen her soru için Ground Truth'a göre kendi **insan hükmünüzü** işaretleyin; hiçbir LLM yargısı kullanılmamıştır.",
        "",
        f"**Toplam Uyuşmazlık Sorusu:** `{len(blind_items)}`",
        "",
        "---",
        "",
    ]
    for it in blind_items:
        md_lines.append(f"## {it['audit_id']} (`{it['unique_id']}`)")
        md_lines.append("")
        md_lines.append(f"**Problem:** {it['problem']}")
        md_lines.append("")
        md_lines.append(f"**Ground Truth:** `{it['ground_truth']}`")
        md_lines.append("")
        for c in it["candidates"]:
            md_lines.append(
                f"- **{c['label']}:** `{c['predicted']}` *(Otomatik Notlayıcı: {c['grader_verdict']})* — `[ ] DOĞRU  [ ] YANLIŞ`"
            )
            if c["excerpt"]:
                clean_ex = c["excerpt"].replace("\n", " ").strip()
                md_lines.append(f"  - *Kısmi Çözüm Sonu:* `{clean_ex[:300]}`")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_path = _safe_write_text("\n".join(md_lines), os.path.join(SFT_DIR, "BLIND_HUMAN_DISCORDANT_AUDIT.md"))
    key_path = _safe_write_json({"items": hidden_keys}, os.path.join(SFT_DIR, "BLIND_HUMAN_DISCORDANT_KEY.json"))

    # 2. (Arm x Grader Version) Reconciliation Matrix
    pf = json.load(open(CLEAN125_PF_JSON, "r", encoding="utf-8"))
    pf125 = [d for d in pf["details"] if 26 <= int(d.get("index", 0)) <= 150]
    m125 = json.load(open(CLEAN125_M125_JSON, "r", encoding="utf-8"))

    def _base_cnt(lo: int, hi: int, col: str) -> int:
        return sum(1 for i in range(lo, hi + 1) if i in base_by_idx and base_by_idx[i][col])

    arm_x_grader_table = {
        "note": "Primary headline results MUST come exclusively from `frozen_normalized_math_matcher_current`.",
        "frozen_grader_sha256": _sha256_file(os.path.join(SFT_DIR, "cst_symbolic_verifier.py")),
        "clean125_q026_q150": {
            "n": 125,
            "Baseline_SFT": {
                "legacy_sympy_v1": sum(1 for d in pf125 if d.get("base_legacy_pass")),  # 99
                "full500_stored_verdict": _base_cnt(26, 150, "old_ok"),                  # 100
                "early_norm_sym_v2": sum(1 for d in pf125 if d.get("base_norm_sym_pass")),  # 103
                "frozen_normalized_math_matcher_current": _base_cnt(26, 150, "frozen_ok"),  # 104
            },
            "GRPO_MathTrain100_Old_Uninformative": {
                "strict_string": sum(1 for d in pf125 if d.get("strict_pass")),         # 94
                "legacy_sympy_v1": sum(1 for d in pf125 if d.get("legacy_pass")),       # 99
                "early_norm_sym_v2": sum(1 for d in pf125 if d.get("norm_sym_pass")),   # 103
                "frozen_normalized_math_matcher_current": sum(
                    1 for d in pf125 if bool(NormalizedMathMatcher.check_match(str(d.get("predicted", "")), str(d.get("ground_truth", "")))[1])
                ),
            },
            "Primary_Ponder_G1": {
                "stored_verdict": m125["scores"]["prefill_pondering_ode"]["score"],     # 100
                "frozen_normalized_math_matcher_current": sum(
                    1 for r in c125["details"] if bool(NormalizedMathMatcher.check_match(str(r.get("primary_pred", "")), str(r.get("ground_truth", "")))[1])
                ),                                                                      # 104
            },
            "Arkhe_Forced_Current": {
                "frozen_normalized_math_matcher_current": sum(
                    1 for r in c125["details"] if bool(NormalizedMathMatcher.check_match(str(r.get("lens_free_pred", r.get("predicted", ""))), str(r.get("ground_truth", "")))[1])
                ),                                                                      # 107
            },
        },
        "pristine50_q151_q200": {
            "n": 50,
            "Baseline_SFT": {
                "full500_stored_verdict": _base_cnt(151, 200, "old_ok"),                # 36
                "frozen_normalized_math_matcher_current": _base_cnt(151, 200, "frozen_ok"),  # 37
            },
            "Pass1_Greedy_Pristine": {
                "frozen_normalized_math_matcher_current": (
                    sum(1 for r in p20["details"] if bool(NormalizedMathMatcher.check_match(str(r.get("pass1_fixed_pred", "")), str(r.get("ground_truth", "")))[1]))
                    + sum(1 for r in p30["details"] if bool(NormalizedMathMatcher.check_match(str(r.get("pass1_predicted", "")), str(r.get("ground_truth", "")))[1]))
                ),                                                                      # 39
            },
            "Arkhe_Forced_Current": {
                "frozen_normalized_math_matcher_current": (
                    sum(1 for r in p20["details"] if bool(NormalizedMathMatcher.check_match(str(r.get("forced_pred", "")), str(r.get("ground_truth", "")))[1]))
                    + sum(1 for r in p30["details"] if bool(NormalizedMathMatcher.check_match(str(r.get("winning_predicted", "")), str(r.get("ground_truth", "")))[1]))
                ),                                                                      # 40
            },
        },
        "confirmatory_reserve_q201_q500_count_only": {
            "n": 300,
            "Baseline_SFT": {
                "full500_stored_verdict": _base_cnt(201, 500, "old_ok"),                # 232
                "frozen_normalized_math_matcher_current": _base_cnt(201, 500, "frozen_ok"),  # 235
            },
        },
    }

    # 3. Reclassify Epistemic Floor + Update Forensic GRPO Framing
    if os.path.exists(FORENSIC_JSON):
        forensic = json.load(open(FORENSIC_JSON, "r", encoding="utf-8"))
        r400 = forensic.get("rollouts_400_and_reliability_audit", {})
        c_break = r400.get("consistent_wrongs_breakdown", {})
        n_0_of_4 = int(c_break.get("true_0_of_4_errors", 5))
        n_1_of_4 = int(c_break.get("solvable_1_of_4_correct", 2))
        epistemic_reclass = {
            "total_prompts": int(r400.get("total_prompts", 100)),
            "pass_at_4_solvability_floor_count": n_0_of_4,
            "pass_at_4_solvability_floor_pct": round(100.0 * n_0_of_4 / max(1, int(r400.get("total_prompts", 100))), 2),
            "majority_at_4_plurality_failure_floor_min_count": n_0_of_4 + n_1_of_4,
            "majority_at_4_plurality_failure_floor_min_pct": round(
                100.0 * (n_0_of_4 + n_1_of_4) / max(1, int(r400.get("total_prompts", 100))), 2
            ),
            "clarification": (
                "Steps 31 & 64 contain 1/4 correct rollout (so Pass@4 succeeds and 0/4 floor is 5%), "
                "but 3/4 rollouts agree on the same wrong answer, so Majority@4 still fails them (Majority@4 floor >= 7%)."
            ),
        }
        forensic["epistemic_floor_reclassification"] = epistemic_reclass
        forensic["legacy_grpo_interpretation"] = {
            "status": "EFFECT_UNMEASURED_POLICY_DID_NOT_MOVE",
            "revised_statement": (
                "Legacy 100-step GRPO produced ||dW||_F / ||W_0||_F = 0.2891% and 109/125 (87.2%) "
                "token-identical greedy trajectories on Clean-125. Therefore the prior Clean-125 outcome "
                "cannot be framed as '0 regression / p=1.0' or attributed to RL; the policy did not move "
                "(including single-question anecdotes such as Q105)."
            ),
        }
        _safe_write_json(forensic, FORENSIC_JSON)
    else:
        epistemic_reclass = {}

    rev_tables = {
        "runtime_metadata": collect_runtime_metadata(),
        "blind_discordant_audit_count": len(blind_items),
        "blind_discordant_md_path": md_path,
        "arm_x_grader_matrix": arm_x_grader_table,
        "epistemic_floor_reclassification": epistemic_reclass,
    }
    rev_path = _safe_write_json(rev_tables, os.path.join(SFT_DIR, "CLAUDE_REVISION_AUDIT_TABLES.json"))

    print("=" * 80)
    print("GPU-FREE PHASE 2 AUDIT COMPLETE")
    print(f"Blind Discordant Items: {len(blind_items)} -> {md_path}")
    print(f"Blind Key (keep hidden until human scoring): {key_path}")
    print(f"Revision Audit Tables:  {rev_path}")
    print(f"Epistemic Floors:       Pass@4={epistemic_reclass.get('pass_at_4_solvability_floor_pct')}% | Maj@4>={epistemic_reclass.get('majority_at_4_plurality_failure_floor_min_pct')}%")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
