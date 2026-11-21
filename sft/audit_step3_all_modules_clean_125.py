"""
SODALS v2.1 — Unified Forensic Grader & Clean 125 Held-Out Audit across All 5 Stages
=====================================================================================
Evaluates all 5 stages on identical footing (Zero Grader Drift, Zero Data Contamination):
  0. Baseline 0: SFT Soft-Blended (`math500_domain2_soft_blended_170_results.json`)
  1. Baseline 1: 1536-Token GRPO Long (`sft/eval_grpo_long_math500_150_results.json`)
  2. Step 3 Module 1: Adaptive Prefill Pondering ODE (`sft/eval_grpo_long_pondering_150_results.json`)
  3. Step 3 Module 2: CST Symbolic PRM & Contradiction Gate (`sft/eval_grpo_long_cst_prm_150_results.json`)
  4. Step 3 Module 3: Tuned Lens (J-Lens) & Tri-Modal Geometric Stability Gate (`sft/eval_grpo_long_tuned_lens_150_results.json`)

Graders applied uniformly to all 5 stages:
  - Grader A (Legacy Baseline Grader): Exact `check_math_match` used in Baselines 0, 1, and Module 1.
  - Grader B (Normalized Symbolic Grader): `NormalizedMathMatcher.check_match(...)[1]` from `cst_symbolic_verifier.py`.
"""

import os
import re
import sys
import json
import math
import sympy

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"c:\Users\Burak\Desktop\sodals_core"
sys.path.insert(0, os.path.join(ROOT, "sft"))
from cst_symbolic_verifier import NormalizedMathMatcher


def clean_latex_legacy(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = re.sub(r"\s*(?:cm|units|degrees|m|inches)\b", "", s, flags=re.I)
    return s.strip()


def to_sympy_legacy(s: str):
    s = clean_latex_legacy(s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_math_match_legacy(pred: str, gt: str) -> bool:
    cp = clean_latex_legacy(pred)
    cg = clean_latex_legacy(gt)
    if cp == cg:
        return True
    if cg.lstrip("-").isdigit():
        num_m = re.findall(r"-?\d+", cp)
        if num_m and num_m[-1] == cg:
            return True
    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            p1 = f"{a}+{b},{a}-{b}"
            p2 = f"{a}-{b},{a}+{b}"
            if clean_latex_legacy(pred) in [clean_latex_legacy(p1), clean_latex_legacy(p2)]:
                return True
    sym_gt = to_sympy_legacy(gt)
    sym_pred = to_sympy_legacy(pred)
    if sym_gt is not None and sym_pred is not None:
        try:
            if sympy.simplify(sym_gt - sym_pred) == 0:
                return True
        except Exception:
            pass
        try:
            if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                return True
        except Exception:
            pass
    return False


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main():
    stages = [
        ("0_SFT_Baseline", os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json")),
        ("1_GRPO_Long", os.path.join(ROOT, "sft", "eval_grpo_long_math500_150_results.json")),
        ("2_Mod1_Pondering_ODE", os.path.join(ROOT, "sft", "eval_grpo_long_pondering_150_results.json")),
        ("3_Mod2_CST_PRM", os.path.join(ROOT, "sft", "eval_grpo_long_cst_prm_150_results.json")),
        ("4_Mod3_Tuned_Lens", os.path.join(ROOT, "sft", "eval_grpo_long_tuned_lens_150_results.json")),
        ("5_Mod4_Best_of_G_PRM", os.path.join(ROOT, "sft", "eval_grpo_long_best_of_g_prm_150_results.json")),
    ]

    stage_data = {}
    print("=" * 115)
    print("UNIFIED GRADER AUDIT ACROSS ALL 6 STAGES (CLEAN 125 HELD-OUT vs. 25 TRAINING PILOT vs. 150 COMBINED)")
    print("=" * 115)

    for name, path in stages:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)["details"][:150]
        records = []
        for idx, x in enumerate(raw):
            pred = x.get("predicted", x.get("pred", ""))
            gt = x.get("ground_truth", x.get("gt", ""))
            leg = check_math_match_legacy(pred, gt)
            _, sym = NormalizedMathMatcher.check_match(pred, gt)
            records.append({
                "index": idx + 1,
                "unique_id": x.get("unique_id", f"q_{idx+1}"),
                "subject": x.get("subject", ""),
                "pred": pred,
                "gt": gt,
                "legacy_pass": bool(leg),
                "norm_sym_pass": bool(sym),
                "gate_triggered": bool(x.get("gate_triggered", False)),
            })
        stage_data[name] = records

        t25 = records[:25]
        h125 = records[25:150]
        leg_150 = sum(r["legacy_pass"] for r in records)
        leg_25 = sum(r["legacy_pass"] for r in t25)
        leg_125 = sum(r["legacy_pass"] for r in h125)

        sym_150 = sum(r["norm_sym_pass"] for r in records)
        sym_25 = sum(r["norm_sym_pass"] for r in t25)
        sym_125 = sum(r["norm_sym_pass"] for r in h125)

        print(
            f"{name:22s} | LegacyGrader: Clean125={leg_125:3d}/125 ({100*leg_125/125:.2f}%), Train25={leg_25:2d}/25 ({100*leg_25/25:.2f}%), All150={leg_150:3d}/150 ({100*leg_150/150:.2f}%) | "
            f"NormSym: Clean125={sym_125:3d}/125 ({100*sym_125/125:.2f}%), Train25={sym_25:2d}/25 ({100*sym_25/25:.2f}%), All150={sym_150:3d}/150 ({100*sym_150/150:.2f}%)"
        )

    print("\n" + "=" * 115)
    print("QUESTION-BY-QUESTION PREDICTION DIFFERENCES BETWEEN STAGES (ACROSS ALL 150 QUESTIONS)")
    print("=" * 115)
    for i in range(150):
        preds = [stage_data[s[0]][i]["pred"] for s in stages]
        if len(set(preds)) > 1:
            q_num = i + 1
            subset = "TRAIN-25" if i < 25 else "CLEAN-125"
            gt = stage_data["0_SFT_Baseline"][i]["gt"]
            print(f"Q{q_num:03d} [{subset:9s}] GT={gt!r}:")
            for s_name, _ in stages:
                r = stage_data[s_name][i]
                print(
                    f"   {s_name:22s}: pred={r['pred']!r:20s} | Legacy={'PASS' if r['legacy_pass'] else 'FAIL'} | NormSym={'PASS' if r['norm_sym_pass'] else 'FAIL'}"
                )

    print("\n" + "=" * 115)
    print("QUESTIONS WHERE NORMALIZED SYMBOLIC GRADER DIFFERS FROM LEGACY GRADER (ON MODULE 3)")
    print("=" * 115)
    for r in stage_data["4_Mod3_Tuned_Lens"]:
        if r["legacy_pass"] != r["norm_sym_pass"]:
            subset = "TRAIN-25" if r["index"] <= 25 else "CLEAN-125"
            print(
                f"Q{r['index']:03d} [{subset:9s}] [{r['subject']:20s}] pred={r['pred']!r} vs gt={r['gt']!r} | "
                f"Legacy={r['legacy_pass']} vs NormSym={r['norm_sym_pass']}"
            )


if __name__ == "__main__":
    main()
