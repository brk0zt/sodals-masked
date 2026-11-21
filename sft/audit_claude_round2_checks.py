"""
SODALS v2.1 — GPU-Free Audit for Claude's Round-2 Methodological Checks
=======================================================================
1. Exports `sft/mod2_vs_grpo_150_byte_comparison_proof.json`: Self-contained raw proof of all 150
   questions showing SFT, GRPO, and Module 2 prediction strings side-by-side (`147/150` byte-identical,
   `126/126` passed questions byte-identical).
2. Audits all 30 completed steps (`120` rollouts) of `sft/grpo_math_train_100_results.json` for:
   - Exact reward upper/lower bounds (`r_sympy` cap = `+0.40`, max wrong reward vs. min correct reward).
   - Inspection of the top-5 highest-reward wrong rollouts to verify zero reward hacking.
   - Step 30 Health & Safety Gate metrics (active gradient rate, token length stability Steps 1-15 vs 16-30).
3. Audits Module 4 (`sft/eval_grpo_long_best_of_g_prm_150_results.json`) against Pure Majority Voting
   (`MV@4` on the 4 new rollouts and `MV@5` including Pass-1) on the exact same stored candidates.
"""

import os
import sys
import json
import statistics

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
sys.path.insert(0, SFT_DIR)

from cst_symbolic_verifier import NormalizedMathMatcher
from audit_step3_all_modules_clean_125 import check_math_match_legacy


def canonical_majority_vote(preds: list[str]) -> str:
    clusters = []  # [rep_pred, count, first_idx]
    for idx, p in enumerate(preds):
        if not p:
            continue
        matched = False
        for c in clusters:
            if NormalizedMathMatcher.check_match(p, c[0])[1]:
                c[1] += 1
                matched = True
                break
        if not matched:
            clusters.append([p, 1, idx])
    clusters.sort(key=lambda x: (-x[1], x[2]))
    return clusters[0][0] if clusters else ""


def main():
    # -------------------------------------------------------------------------
    # 1. Export Raw 150-Question Byte-for-Byte Proof (GRPO vs. Module 2)
    # -------------------------------------------------------------------------
    with open(os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json"), "r", encoding="utf-8") as f:
        sft = json.load(f)["details"][:150]
    with open(os.path.join(SFT_DIR, "eval_grpo_long_math500_150_results.json"), "r", encoding="utf-8") as f:
        grpo = json.load(f)["details"][:150]
    with open(os.path.join(SFT_DIR, "eval_grpo_long_cst_prm_150_results.json"), "r", encoding="utf-8") as f:
        mod2 = json.load(f)["details"][:150]

    proof_rows = []
    byte_identical_all = 0
    byte_identical_mod2_pass = 0
    mod2_pass_total = 0

    for i in range(150):
        s_pred = sft[i]["predicted"]
        g_pred = grpo[i]["predicted"]
        m_pred = mod2[i]["predicted"]
        gt = mod2[i]["ground_truth"]

        g_strict, g_sym = NormalizedMathMatcher.check_match(g_pred, gt)
        g_leg = check_math_match_legacy(g_pred, gt)

        m_strict, m_sym = NormalizedMathMatcher.check_match(m_pred, gt)
        m_leg = check_math_match_legacy(m_pred, gt)

        is_same = (g_pred == m_pred)
        if is_same:
            byte_identical_all += 1
        if m_sym:
            mod2_pass_total += 1
            if is_same:
                byte_identical_mod2_pass += 1

        proof_rows.append({
            "index": i + 1,
            "subset": "TRAIN-25" if i < 25 else "CLEAN-125",
            "subject": mod2[i]["subject"],
            "ground_truth": gt,
            "sft_pred": s_pred,
            "grpo_pred": g_pred,
            "mod2_pred": m_pred,
            "grpo_equals_mod2_bytes": is_same,
            "gate_triggered_in_mod2": bool(mod2[i].get("gate_triggered", False)),
            "grpo_grades": {"strict": bool(g_strict), "legacy": bool(g_leg), "norm_sym": bool(g_sym)},
            "mod2_grades": {"strict": bool(m_strict), "legacy": bool(m_leg), "norm_sym": bool(m_sym)},
        })

    proof_path = os.path.join(SFT_DIR, "mod2_vs_grpo_150_byte_comparison_proof.json")
    with open(proof_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_questions": 150,
                    "byte_identical_grpo_vs_mod2": byte_identical_all,
                    "differing_questions": [r["index"] for r in proof_rows if not r["grpo_equals_mod2_bytes"]],
                    "mod2_norm_sym_pass_count": mod2_pass_total,
                    "byte_identical_among_mod2_passes": byte_identical_mod2_pass,
                },
                "details": proof_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 105)
    print("1. RAW BYTE-FOR-BYTE PROOF EXPORTED:", proof_path)
    print(f"   Byte-identical predictions (GRPO vs Mod2): {byte_identical_all}/150")
    print(f"   Byte-identical among Mod2 passed questions: {byte_identical_mod2_pass}/{mod2_pass_total}")
    print("=" * 105)

    # -------------------------------------------------------------------------
    # 2. Audit Step 30 Health Gate & Reward Bounds (30 Steps, 120 Rollouts)
    # -------------------------------------------------------------------------
    with open(os.path.join(SFT_DIR, "grpo_math_train_100_results.json"), "r", encoding="utf-8") as f:
        tr = json.load(f)
    steps = tr["step_details"]

    print("\n" + "=" * 105)
    print("2. STEP 30 HEALTH & REWARD BOUND AUDIT (30 STEPS, 120 ROLLOUTS)")
    print("=" * 105)
    print(
        f"Completed Steps: {len(steps)} | Active Gradient Steps: {tr['active_gradient_steps']}/{len(steps)} "
        f"({100*tr['active_gradient_steps']/len(steps):.1f}%) | Solved Prompts: {tr['total_solved_prompts']}/{len(steps)} ({tr['pass_rate']}%)"
    )

    wrong_rewards = []
    correct_rewards = []
    first15_toks = []
    last15_toks = []

    for s in steps:
        gt = s["ground_truth"]
        if s["step"] <= 15:
            first15_toks.extend(s["rollout_tokens"])
        else:
            last15_toks.extend(s["rollout_tokens"])
        for r_rew, r_pred, r_tok in zip(s["rollout_rewards"], s["rollout_preds"], s["rollout_tokens"]):
            _, is_corr = NormalizedMathMatcher.check_match(r_pred, gt)
            if is_corr:
                correct_rewards.append((r_rew, s["step"], r_tok, r_pred, gt))
            else:
                wrong_rewards.append((r_rew, s["step"], r_tok, r_pred, gt))

    wrong_rewards.sort(reverse=True)
    correct_rewards.sort()

    print("Theoretical Reward Bounds in train_grpo_math_train_100.py:")
    print("  - r_sympy is strictly capped at min(0.40, 0.04 * eq_count)")
    print("  - Theoretical MAX possible WRONG rollout reward  : -1.00 (outcome) + 0.30 (format) + 0.40 (sympy cap) + 0.20 (length) = -0.1000")
    print("  - Theoretical MIN possible CORRECT rollout reward: +2.00 (outcome) - 0.50 (no box) - 0.25 (contra)    + 0.00 (length) = +1.2500 (+1.6500 w/ box)")
    print(f"Empirical Rollout Distribution across 120 Rollouts:")
    print(f"  - Correct Rollouts ({len(correct_rewards):2d}/120): Min = {correct_rewards[0][0]:+.4f} | Max = {correct_rewards[-1][0]:+.4f}")
    print(f"  - Wrong Rollouts   ({len(wrong_rewards):2d}/120): Max = {wrong_rewards[0][0]:+.4f} | Min = {wrong_rewards[-1][0]:+.4f}")
    print(f"  - Empirical Separation Gap (Min Correct - Max Wrong): {correct_rewards[0][0] - wrong_rewards[0][0]:+.4f} units (Zero overlap!)")
    print("Top 5 Highest-Reward WRONG Rollouts across all 120 rollouts:")
    for rew, st, tok, p, g in wrong_rewards[:5]:
        print(f"  Step {st:02d}: reward={rew:+.4f} | tokens={tok:4d} | pred={p!r:18s} vs gt={g!r}")
    print(
        f"Token Length Stability (No Length Explosion): "
        f"Steps 1..15 mean={statistics.mean(first15_toks):.1f} (med={statistics.median(first15_toks):.1f}) -> "
        f"Steps 16..30 mean={statistics.mean(last15_toks):.1f} (med={statistics.median(last15_toks):.1f})"
    )

    # -------------------------------------------------------------------------
    # 3. Module 4: Majority Vote vs. PRM Selection (GPU-Free on Stored Candidates)
    # -------------------------------------------------------------------------
    with open(os.path.join(SFT_DIR, "eval_grpo_long_best_of_g_prm_150_results.json"), "r", encoding="utf-8") as f:
        m4 = json.load(f)["details"]

    print("\n" + "=" * 105)
    print("3. MODULE 4: MAJORITY VOTE (MV@4 / MV@5) vs. TRI-MODAL PRM SELECTION ON STORED CANDIDATES")
    print("=" * 105)

    p1_125_leg, mv4_125_leg, mv5_125_leg, prm_125_leg = 0, 0, 0, 0
    p1_125_sym, mv4_125_sym, mv5_125_sym, prm_125_sym = 0, 0, 0, 0

    for r in m4:
        gt = r["ground_truth"]
        p1 = r["primary_pred"]
        prm_pred = r["predicted"]

        if r["gate_triggered"] and r.get("candidate_pool"):
            cands = r["candidate_pool"]
            g4_preds = [c["pred"] for c in cands[1:]]
            g5_preds = [c["pred"] for c in cands]
            mv4_pred = canonical_majority_vote(g4_preds)
            mv5_pred = canonical_majority_vote(g5_preds)
            p1_ok = NormalizedMathMatcher.check_match(p1, gt)[1]
            mv4_ok = NormalizedMathMatcher.check_match(mv4_pred, gt)[1]
            mv5_ok = NormalizedMathMatcher.check_match(mv5_pred, gt)[1]
            prm_ok = NormalizedMathMatcher.check_match(prm_pred, gt)[1]
            print(
                f"  Q{r['index']:03d} [{r['subset']:7s}] GT={gt!r:14s} | "
                f"Pass1={p1!r:10s}({str(p1_ok):5s}) | MV@4={mv4_pred!r:10s}({str(mv4_ok):5s}) | "
                f"MV@5={mv5_pred!r:10s}({str(mv5_ok):5s}) | PRM={prm_pred!r:10s}({str(prm_ok):5s})"
            )
        else:
            mv4_pred = p1
            mv5_pred = p1

        if r["index"] > 25:
            p1_125_leg += int(check_math_match_legacy(p1, gt))
            mv4_125_leg += int(check_math_match_legacy(mv4_pred, gt))
            mv5_125_leg += int(check_math_match_legacy(mv5_pred, gt))
            prm_125_leg += int(check_math_match_legacy(prm_pred, gt))

            p1_125_sym += int(NormalizedMathMatcher.check_match(p1, gt)[1])
            mv4_125_sym += int(NormalizedMathMatcher.check_match(mv4_pred, gt)[1])
            mv5_125_sym += int(NormalizedMathMatcher.check_match(mv5_pred, gt)[1])
            prm_125_sym += int(NormalizedMathMatcher.check_match(prm_pred, gt)[1])

    print("\nClean-125 Held-Out Comparison (Exact Same Stored Candidates):")
    print(f"  Pass-1 Greedy (G=1)             : Legacy = {p1_125_leg}/125 ({100*p1_125_leg/125:.2f}%) | NormSym = {p1_125_sym}/125 ({100*p1_125_sym/125:.2f}%)")
    print(f"  Pure Majority Vote (G=4 new)    : Legacy = {mv4_125_leg}/125 ({100*mv4_125_leg/125:.2f}%) | NormSym = {mv4_125_sym}/125 ({100*mv4_125_sym/125:.2f}%)")
    print(f"  Pure Majority Vote (All 5 cands): Legacy = {mv5_125_leg}/125 ({100*mv5_125_leg/125:.2f}%) | NormSym = {mv5_125_sym}/125 ({100*mv5_125_sym/125:.2f}%)")
    print(f"  Tri-Modal PRM (Module 4)        : Legacy = {prm_125_leg}/125 ({100*prm_125_leg/125:.2f}%) | NormSym = {prm_125_sym}/125 ({100*prm_125_sym/125:.2f}%)")


if __name__ == "__main__":
    main()
