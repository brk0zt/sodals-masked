"""
SODALS v2.1 — Pre-Registered Out-of-MATH-500 Training Pool Builder (N = 100 from Hendrycks MATH Train Split)
=============================================================================================================
Implements Claude's Item 3 Pre-Registration Protocol:
  1. Downloads all 7 official Hendrycks MATH `train` split JSONL files directly from
     `https://huggingface.co/datasets/lighteval/MATH-Hard/raw/main/train/<subject>.jsonl`
     (non-LFS raw JSONL, 100% isolated from the `test` split).
  2. Verifies 0% overlap (exact and normalized string match) against all 500 questions in `HuggingFaceH4/MATH-500`.
  3. Filters out `[asy]` diagram problems and selects 100 stratified problems across all 7 subjects
     with clean, SymPy-verifiable `\boxed{...}` ground-truth answers (deterministic seed = 42).
  4. Saves the locked 100-prompt training dataset to `sft/math_train_100_grpo_prompts.json`.
"""

import os
import re
import json
import random
import urllib.request
import sympy

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
RAW_CACHE_JSONL = os.path.join(SFT_DIR, "hendrycks_math_train_hard_all.jsonl")
OUTPUT_JSON = os.path.join(SFT_DIR, "math_train_100_grpo_prompts.json")
MATH500_CACHE_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"

SUBJECT_FILES = [
    ("Algebra", "algebra.jsonl"),
    ("Intermediate Algebra", "intermediate_algebra.jsonl"),
    ("Number Theory", "number_theory.jsonl"),
    ("Prealgebra", "prealgebra.jsonl"),
    ("Counting & Probability", "counting_and_probability.jsonl"),
    ("Precalculus", "precalculus.jsonl"),
    ("Geometry", "geometry.jsonl"),
]


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


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_clean_verifiable_answer(ans: str) -> bool:
    if not ans or len(ans) > 28:
        return False
    if any(tok in ans for tok in [r"\text", r"\mbox", r"\begin", "or", "and", ";", ","]):
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


def main():
    # 1. Download all 7 Hendrycks MATH Train Split JSONL files if not cached
    all_train_rows = []
    if not os.path.exists(RAW_CACHE_JSONL) or os.path.getsize(RAW_CACHE_JSONL) < 1_000_000:
        print("Downloading 7 Hendrycks MATH train split JSONL files from lighteval/MATH-Hard...")
        with open(RAW_CACHE_JSONL, "w", encoding="utf-8") as out_f:
            for subj_name, fname in SUBJECT_FILES:
                url = f"https://huggingface.co/datasets/lighteval/MATH-Hard/raw/main/train/{fname}"
                req = urllib.request.urlopen(url, timeout=15)
                raw_txt = req.read().decode("utf-8").strip()
                items = json.loads(raw_txt) if raw_txt.startswith("[") else [json.loads(l) for l in raw_txt.splitlines() if l.strip()]
                print(f"  Downloaded {fname:32s}: {len(items):4d} train problems")
                for obj in items:
                    obj["subject_canonical"] = subj_name
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    all_train_rows.append(obj)
    else:
        print(f"Using cached Hendrycks MATH train split: {RAW_CACHE_JSONL}")
        with open(RAW_CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_train_rows.append(json.loads(line))

    print(f"Total Hendrycks MATH train split problems loaded: {len(all_train_rows)}")

    # 2. Load all 500 MATH-500 test questions from local cache to guarantee 0.00% overlap
    math500_norm_set = set()
    math500_prefix_set = set()
    with open(MATH500_CACHE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                np_str = normalize_text(item["problem"])
                math500_norm_set.add(np_str)
                math500_prefix_set.add(np_str[:60])
    print(f"Loaded {len(math500_norm_set)} MATH-500 test questions for zero-leakage screening.")

    # 3. Filter candidates from MATH train split
    candidates_by_subject = {}
    overlap_excluded = 0
    for idx, row in enumerate(all_train_rows):
        prob = str(row.get("problem", "")).strip()
        sol = str(row.get("solution", "")).strip()
        level = str(row.get("level", "Level 5")).strip()
        subj = str(row.get("subject_canonical", row.get("type", ""))).strip()

        np_str = normalize_text(prob)
        if np_str in math500_norm_set or np_str[:60] in math500_prefix_set:
            overlap_excluded += 1
            continue

        # Exclude Asymptote [asy] visual geometry figures and overly long prompts (so 1536 tokens is plenty)
        if "[asy]" in prob or len(prob) > 420 or len(prob) < 30:
            continue

        gt = extract_boxed_answer(sol)
        if not is_clean_verifiable_answer(gt):
            continue

        candidates_by_subject.setdefault(subj, []).append({
            "train_row_id": idx + 1,
            "subject": subj,
            "level": level,
            "problem": prob,
            "ground_truth": gt,
        })

    print(f"Screened {len(all_train_rows)} train items (overlap excluded: {overlap_excluded}). Eligible clean candidates by subject:")
    for s, lst in sorted(candidates_by_subject.items()):
        print(f"  {s:25s}: {len(lst):4d} candidates")

    # 4. Deterministic Stratified Sampling of 100 Prompts (Seed = 42)
    rng = random.Random(42)
    quotas = {
        "Algebra": 22,
        "Intermediate Algebra": 18,
        "Number Theory": 16,
        "Prealgebra": 14,
        "Counting & Probability": 12,
        "Precalculus": 10,
        "Geometry": 8,
    }

    selected_100 = []
    for subj, q_count in quotas.items():
        pool = list(candidates_by_subject.get(subj, []))
        # Sort by problem length so moderate-length problems come first, then shuffle deterministically
        rng.shuffle(pool)
        chosen = pool[:q_count]
        selected_100.extend(chosen)

    rng.shuffle(selected_100)
    for i, item in enumerate(selected_100):
        item["prompt_index"] = i + 1

    # 5. Final Zero-Overlap Verification Assertion against all 500 MATH-500 items
    final_overlaps = sum(
        1 for x in selected_100
        if normalize_text(x["problem"]) in math500_norm_set
        or normalize_text(x["problem"])[:60] in math500_prefix_set
    )
    assert final_overlaps == 0, f"CRITICAL ERROR: Found {final_overlaps} overlaps with MATH-500!"

    level_counts = {}
    subj_counts = {}
    for x in selected_100:
        level_counts[x["level"]] = level_counts.get(x["level"], 0) + 1
        subj_counts[x["subject"]] = subj_counts.get(x["subject"], 0) + 1

    payload = {
        "dataset_name": "Hendrycks MATH Official Train Split — Stratified 100 Active-Gradient GRPO Prompts",
        "source": "lighteval/MATH-Hard (official Hendrycks MATH train split, 0% overlap with MATH-500 test split)",
        "seed": 42,
        "total_prompts": len(selected_100),
        "math500_overlap_count": final_overlaps,
        "level_distribution": level_counts,
        "subject_distribution": subj_counts,
        "prompts": selected_100,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"SUCCESSFULLY BUILT {len(selected_100)} ZERO-LEAKAGE MATH-TRAIN GRPO PROMPTS -> {OUTPUT_JSON}")
    print(f"Verified Overlap with MATH-500 (all 500 test questions): {final_overlaps} / 500 (0.00%)")
    print(f"Level Distribution:   {level_counts}")
    print(f"Subject Distribution: {subj_counts}")
    print("=" * 90)


if __name__ == "__main__":
    main()
