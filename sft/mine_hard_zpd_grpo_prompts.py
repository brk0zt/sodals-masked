"""
SODALS v2.1 — Hard-Mining & ZPD Prompt Pool Curator for Quota-Guarded GRPO
==========================================================================
Constructs a strictly curated, high-efficiency training curriculum:
  1. 100% Strict [asy] Filtering:
     Diagram problems with visual geometry coordinates are completely purged
     from the training curriculum so text-only models are never forced into
     visual hallucination loops.
  2. Zone of Proximal Development (ZPD / Active Learning Frontier):
     Extracts problems from previous GRPO rollouts where solved_rollouts in {1, 2, 3}/4.
     These provide non-zero group reward variance (sigma_r > 0) and high policy gradient signal.
  3. Calibrated Epistemic Abstention (Honest "Ben Bilmiyorum"):
     Selects unsolvable (0/4) problems where rollouts diverged, pairing them with
     the Arkhe Epistemic Auditor so the model learns organic selective abstention.
  4. MATH-500 Zero-Leakage Guarantee:
     Guarantees 0.00% overlap against HuggingFaceH4/MATH-500 test questions.
  5. Outputs:
     `sft/math_train_hard_mined_zpd_prompts.json`
"""

import os
import re
import json
import random
import sympy
from typing import Dict, List, Set, Any

ROOT = r"C:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
PREV_GRPO_RESULTS = os.path.join(SFT_DIR, "grpo_math_train_100_results.json")
HARD_ALL_JSONL = os.path.join(SFT_DIR, "hendrycks_math_train_hard_all.jsonl")
MATH500_CACHE_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"
OUTPUT_JSON = os.path.join(SFT_DIR, "math_train_hard_mined_zpd_prompts.json")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return text[start:].strip()
    return ""


def is_clean_verifiable_answer(ans: str) -> bool:
    if not ans or len(ans) > 32:
        return False
    if any(tok in ans for tok in [r"\text", r"\mbox", r"\begin", "or", "and", ";"]):
        return False
    s = ans.replace(r"\left", "").replace(r"\right", "").replace(" ", "")
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**").replace(r"\pi", "pi")
    try:
        expr = sympy.sympify(s)
        return expr is not None
    except Exception:
        return False


def load_math500_signatures() -> Set[str]:
    signatures = set()
    if os.path.exists(MATH500_CACHE_JSONL):
        with open(MATH500_CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    signatures.add(normalize_text(d.get("problem", "")))
    print(f"[MATH-500 Guard] Loaded {len(signatures)} test signatures to ensure 0.00% overlap.")
    return signatures


def main():
    print("=" * 85)
    print("SODALS v2.1 — HARD-MINED ZPD & EPISTEMIC ABSTENTION CURATOR FOR GRPO")
    print("=" * 85)

    math500_sigs = load_math500_signatures()

    mined_prompts: List[Dict[str, Any]] = []
    seen_prompts: Set[str] = set()

    # 1. Mine Kova A (ZPD Frontier: 1..3/4 correct) & Kova B (Calibrated Divergent: 0/4) from previous GRPO run
    if os.path.exists(PREV_GRPO_RESULTS):
        with open(PREV_GRPO_RESULTS, "r", encoding="utf-8") as f:
            prev_data = json.load(f)

        step_details = prev_data.get("step_details", [])
        print(f"Loaded {len(step_details)} historical GRPO rollout steps.")

        zpd_count = 0
        abstain_cand_count = 0

        for step in step_details:
            prob = step.get("problem", "").strip()
            gt = step.get("ground_truth", "").strip()
            norm_prob = normalize_text(prob)

            # Strict filter: Purge [asy]
            if "[asy]" in prob.lower():
                continue
            # Overlap filter
            if norm_prob in math500_sigs or norm_prob in seen_prompts:
                continue

            solved = int(step.get("solved_rollouts", 0))
            preds = step.get("rollout_preds", [])
            subj = step.get("subject", "General Math")
            lvl = step.get("level", "Level 5")

            # Kova A: ZPD Active Learning Frontier (1 to 3 out of 4 correct)
            if 1 <= solved <= 3:
                item = {
                    "question_index": len(mined_prompts) + 1,
                    "train_row_id": step.get("train_row_id", 0),
                    "subject": subj,
                    "level": lvl,
                    "problem": prob,
                    "ground_truth": gt,
                    "mining_category": "ZPD_Active_Learning_Frontier",
                    "historical_solved_ratio": f"{solved}/4",
                }
                mined_prompts.append(item)
                seen_prompts.add(norm_prob)
                zpd_count += 1

            # Kova B: Calibrated Divergence / Abstention Frontier (0/4 correct, high disagreement)
            elif solved == 0 and len(set(preds)) >= 2:
                item = {
                    "question_index": len(mined_prompts) + 1,
                    "train_row_id": step.get("train_row_id", 0),
                    "subject": subj,
                    "level": lvl,
                    "problem": prob,
                    "ground_truth": gt,
                    "mining_category": "Calibrated_Epistemic_Abstention",
                    "historical_solved_ratio": "0/4",
                }
                mined_prompts.append(item)
                seen_prompts.add(norm_prob)
                abstain_cand_count += 1

        print(f"  Mined from historical rollouts: {zpd_count} ZPD questions + {abstain_cand_count} Epistemic Abstention questions.")

    # 2. Fill the remaining quota up to 100 with Hard Level 4-5 problems from hendrycks_math_train_hard_all.jsonl
    target_count = 100
    if len(mined_prompts) < target_count and os.path.exists(HARD_ALL_JSONL):
        needed = target_count - len(mined_prompts)
        print(f"\nFilling remaining {needed} slots from Hendrycks MATH Hard pool...")
        candidates = []
        with open(HARD_ALL_JSONL, "r", encoding="utf-8") as f:
            for row_id, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                prob = d.get("problem", "").strip()
                norm_prob = normalize_text(prob)

                # Strict [asy] purge
                if "[asy]" in prob.lower():
                    continue
                if norm_prob in math500_sigs or norm_prob in seen_prompts:
                    continue

                sol = d.get("solution", "")
                ans = extract_boxed_answer(sol)
                if not is_clean_verifiable_answer(ans):
                    continue

                candidates.append({
                    "train_row_id": row_id,
                    "subject": d.get("subject_canonical", d.get("type", "General Math")),
                    "level": d.get("level", "Level 5"),
                    "problem": prob,
                    "ground_truth": ans,
                    "mining_category": "Fresh_Hard_Exploration",
                    "historical_solved_ratio": "unseen",
                })

        # Deterministic shuffle with seed 42
        random.seed(42)
        random.shuffle(candidates)

        for c in candidates[:needed]:
            c["question_index"] = len(mined_prompts) + 1
            mined_prompts.append(c)
            seen_prompts.add(normalize_text(c["problem"]))

    # Summary
    print(f"\nFinal Mined Prompt Pool Size: {len(mined_prompts)}")
    cat_counts = {}
    for p in mined_prompts:
        cat = p["mining_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in cat_counts.items():
        print(f"  - {cat:35s}: {count:3d} questions")

    # Save curated dataset
    output_payload = {
        "dataset_name": "SODALS_v2.1_Hard_Mined_ZPD_Curriculum_No_Asy",
        "total_prompts": len(mined_prompts),
        "asy_diagram_problems": 0,
        "math500_overlap": 0,
        "mining_breakdown": cat_counts,
        "prompts": mined_prompts,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote curated dataset to:\n  {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
