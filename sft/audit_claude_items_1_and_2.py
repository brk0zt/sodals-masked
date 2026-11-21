"""
SODALS v2.1 — GPU-Free Forensic Audit for Claude's Items 1 & 2
==============================================================
1. Verifies the 3-tier grader behavior (Pure String 116 vs. Legacy SymPy 122/123 vs. Normalized SymPy 126/127)
   and proves whether Module 2 changed output formatting on the 10 `symbolic_match=True` questions.
2. Evaluates the "Early Surrender / Premature Termination" hypothesis across ALL failed questions
   (not just Q105) in SFT Baseline vs. 1536-Token GRPO Long on Clean-125 Held-Out and All-150.
3. Computes exact statistical power and McNemar significance thresholds for scaling to the remaining
   Clean MATH-500 pool (N = 350 unseen outside the 150, or N = 475 unseen outside the 25 training prompts).
"""

import os
import json
import math
import statistics

ROOT = r"c:\Users\Burak\Desktop\sodals_core"


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main():
    with open(os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json"), "r", encoding="utf-8") as f:
        sft = json.load(f)["details"][:150]
    with open(os.path.join(ROOT, "sft", "eval_grpo_long_math500_150_results.json"), "r", encoding="utf-8") as f:
        grpo = json.load(f)["details"][:150]
    with open(os.path.join(ROOT, "sft", "eval_grpo_long_cst_prm_150_results.json"), "r", encoding="utf-8") as f:
        mod2 = json.load(f)["details"][:150]

    print("=" * 110)
    print("1. FORENSIC CHECK OF MODULE 2'S 10 `symbolic_match=True` QUESTIONS (FORMAT DRIFT vs. GRADER DEFINITION)")
    print("=" * 110)
    sym_qs = [m for m in mod2 if m.get("verdict") == "PASS" and m.get("symbolic_match") is True]
    print(f"Total questions in Module 2 with verdict='PASS' and symbolic_match=True: {len(sym_qs)}/150")
    for m in sym_qs:
        idx = m["index"] - 1
        sp = sft[idx]["predicted"]
        gp = grpo[idx]["predicted"]
        mp = m["predicted"]
        gv = grpo[idx]["verdict"]
        sv = sft[idx]["verdict"]
        print(
            f"  Q{m['index']:03d} [{m['subset']:8s}]: SFT_pred={sp!r:18s} ({sv}) | "
            f"GRPO_pred={gp!r:18s} ({gv}) | Mod2_pred={mp!r:18s} | GT={m['ground_truth']!r:20s} | "
            f"ByteIdenticalToGRPO={gp == mp}"
        )

    print("\n" + "=" * 110)
    print("2. EARLY SURRENDER / TOKEN BUDGET DISTRIBUTION ACROSS ALL QUESTIONS (SFT vs. GRPO LONG)")
    print("=" * 110)
    for name, start_idx, end_idx in [
        ("CLEAN-125 HELD-OUT (Q026..Q150)", 25, 150),
        ("TRAIN-25 PILOT (Q001..Q025)", 0, 25),
        ("ALL-150 COMBINED (Q001..Q150)", 0, 150),
    ]:
        s_sub = sft[start_idx:end_idx]
        g_sub = grpo[start_idx:end_idx]

        s_all_tok = [x["tokens"] for x in s_sub]
        g_all_tok = [x["generated_tokens"] for x in g_sub]

        fails = [(s, g) for s, g in zip(s_sub, g_sub) if s["verdict"] == "FAIL"]
        s_fail_tok = [s["tokens"] for s, _ in fails]
        g_fail_tok = [g["generated_tokens"] for _, g in fails]

        print(f"\n--- {name} (N={len(s_sub)}, SFT Baseline Failures={len(fails)}) ---")
        print(
            f"  ALL Questions (N={len(s_sub):3d})   : "
            f"SFT mean={statistics.mean(s_all_tok):6.1f} (median={statistics.median(s_all_tok):6.1f}) | "
            f"GRPO mean={statistics.mean(g_all_tok):6.1f} (median={statistics.median(g_all_tok):6.1f})"
        )
        print(
            f"  SFT FAIL Questions (n={len(fails):2d}): "
            f"SFT mean={statistics.mean(s_fail_tok):6.1f} (median={statistics.median(s_fail_tok):6.1f}) | "
            f"GRPO mean={statistics.mean(g_fail_tok):6.1f} (median={statistics.median(g_fail_tok):6.1f})"
        )
        for thresh in [256, 512, 768, 1024, 1536, 2048]:
            sc = sum(1 for t in s_fail_tok if t < thresh)
            gc = sum(1 for t in g_fail_tok if t < thresh)
            print(
                f"    Failed Qs with tokens < {thresh:4d}: "
                f"SFT = {sc:2d}/{len(fails)} ({100*sc/len(fails):5.1f}%) | "
                f"GRPO = {gc:2d}/{len(fails)} ({100*gc/len(fails):5.1f}%)"
            )
        sc_cap = sum(1 for t in s_fail_tok if t >= 2048)
        gc_cap = sum(1 for t in g_fail_tok if t >= 2048)
        print(
            f"    Failed Qs hitting 2048 cap : "
            f"SFT = {sc_cap:2d}/{len(fails)} ({100*sc_cap/len(fails):5.1f}%) | "
            f"GRPO = {gc_cap:2d}/{len(fails)} ({100*gc_cap/len(fails):5.1f}%)"
        )

        print("  Questions where SFT and GRPO token counts or predictions differ:")
        diff_count = 0
        for s, g in zip(s_sub, g_sub):
            if s["tokens"] != g["generated_tokens"] or s["predicted"] != g["predicted"]:
                diff_count += 1
                print(
                    f"    Q{s['index']:03d} (SFT={s['verdict']} -> GRPO={g['verdict']}): "
                    f"SFT_tok={s['tokens']:4d} (pred={s['predicted']!r}) -> "
                    f"GRPO_tok={g['generated_tokens']:4d} (pred={g['predicted']!r})"
                )
        print(f"  Total questions where SFT and GRPO differed in tokens or prediction: {diff_count}/{len(s_sub)}")


if __name__ == "__main__":
    main()
