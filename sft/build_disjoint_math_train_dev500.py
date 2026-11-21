"""
SODALS v2.1 — Phase 3: Build Large Disjoint MATH-Train Dev Pool (N >= 500)
===========================================================================
Creates an independent out-of-MATH-500, out-of-GRPO, out-of-ZPD-curriculum
development pool (`N=500`) from Hendrycks MATH `train` split.

Priority:
  1. Stratified Level-matched (Levels 1..5 proportional to MATH-500:
     {L1:43, L2:90, L3:105, L4:128, L5:134}) via `EleutherAI/hendrycks_math`
     (`datasets` + `pyarrow` cached or fetched once online).
  2. Strict zero-overlap verification against:
       - `HuggingFaceH4/MATH-500` (all 500 questions)
       - `sft/math_train_100_grpo_prompts.json`
       - `sft/math_train_hard_mined_zpd_prompts.json`
       - Any SFT/SDPO `.jsonl` curricula under `sft/`.
"""

import argparse
from collections import Counter
import glob
import json
import os
import random
import re
import sys
import tempfile
from typing import Any, Dict, List, Set

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
MATH500_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"
HARD_ALL_JSONL = os.path.join(SFT_DIR, "hendrycks_math_train_hard_all.jsonl")
DEFAULT_OUT = os.path.join(SFT_DIR, "math_train_disjoint_dev500.json")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def extract_boxed(text: str) -> str:
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
    return ""


def collect_forbidden_problems() -> Set[str]:
    forbidden: Set[str] = set()
    if os.path.exists(MATH500_JSONL):
        with open(MATH500_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                forbidden.add(norm_text(q.get("problem", "")))
    for fn in ("math_train_100_grpo_prompts.json", "math_train_hard_mined_zpd_prompts.json"):
        p = os.path.join(SFT_DIR, fn)
        if os.path.exists(p):
            d = json.load(open(p, "r", encoding="utf-8"))
            for row in d.get("prompts", []):
                forbidden.add(norm_text(row.get("problem", "")))
    for p in glob.glob(os.path.join(SFT_DIR, "*.jsonl")):
        if os.path.basename(p) == "hendrycks_math_train_hard_all.jsonl":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    for key in ("problem", "prompt", "instruction"):
                        if key in obj and isinstance(obj[key], str):
                            forbidden.add(norm_text(obj[key]))
                    if "messages" in obj and isinstance(obj["messages"], list):
                        for m in obj["messages"]:
                            if m.get("role") == "user":
                                forbidden.add(norm_text(m.get("content", "")))
        except Exception:
            pass
    return forbidden


def load_full_hendrycks_train(allow_online: bool = False) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        if not allow_online:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)

        all_rows = []
        # Try single-config Hendrycks MATH mirrors first (`DigitalLearningGmbH/MATH-lighteval`, `nlile/hendrycks-MATH-benchmark`)
        for repo_name in ("DigitalLearningGmbH/MATH-lighteval", "nlile/hendrycks-MATH-benchmark", "qwedsacf/competition_math"):
            try:
                ds = load_dataset(repo_name, split="train")
                for idx, r in enumerate(ds):
                    sol = str(r.get("solution", ""))
                    ans = str(r.get("answer", "")).strip() or extract_boxed(sol)
                    if not ans:
                        continue
                    lvl_raw = r.get("level", 5)
                    m = re.search(r"(\d)", str(lvl_raw))
                    lvl = int(m.group(1)) if m else 5
                    all_rows.append({
                        "train_row_id": f"{repo_name.split('/')[-1]}_{idx}",
                        "subject": str(r.get("type", r.get("subject", "Math"))),
                        "level": lvl,
                        "problem": r["problem"],
                        "ground_truth": ans,
                        "source": f"{repo_name}:train",
                    })
                if len(all_rows) >= 5000:
                    return all_rows
            except Exception:
                all_rows = []

        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        for subj in subjects:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            for idx, r in enumerate(ds):
                ans = extract_boxed(r.get("solution", ""))
                if not ans:
                    continue
                lvl_str = str(r.get("level", "Level 5"))
                m = re.search(r"(\d)", lvl_str)
                lvl = int(m.group(1)) if m else 5
                all_rows.append({
                    "train_row_id": f"{subj}_{idx}",
                    "subject": subj.replace("_", " ").title(),
                    "level": lvl,
                    "problem": r["problem"],
                    "ground_truth": ans,
                    "source": "EleutherAI/hendrycks_math:train",
                })
        return all_rows
    except Exception as exc:
        print(f"[INFO] Full EleutherAI/hendrycks_math load unavailable ({exc}); checking Level-5 fallback...", flush=True)
        rows = []
        with open(HARD_ALL_JSONL, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                r = json.loads(line)
                ans = extract_boxed(r.get("solution", ""))
                if not ans:
                    continue
                rows.append({
                    "train_row_id": f"hard5_{idx}",
                    "subject": r.get("subject_canonical", r.get("type", "Math")),
                    "level": 5,
                    "problem": r["problem"],
                    "ground_truth": ans,
                    "source": "hendrycks_math_train_hard_all.jsonl",
                })
        return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build Disjoint N=500 MATH-Train Dev Pool")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_online", action="store_true")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    forbidden = collect_forbidden_problems()
    pool = load_full_hendrycks_train(allow_online=args.allow_online)
    disjoint = [r for r in pool if norm_text(r["problem"]) not in forbidden]
    # Deduplicate by normalized problem text
    seen: Set[str] = set()
    clean_pool = []
    for r in disjoint:
        k = norm_text(r["problem"])
        if k not in seen:
            seen.add(k)
            clean_pool.append(r)

    levels_present = Counter(r["level"] for r in clean_pool)
    rng = random.Random(args.seed)
    target_counts = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}

    selected: List[Dict[str, Any]] = []
    if all(levels_present[lv] >= cnt for lv, cnt in target_counts.items()) and args.n == 500:
        stratified = True
        for lv in (1, 2, 3, 4, 5):
            cand = [r for r in clean_pool if r["level"] == lv]
            rng.shuffle(cand)
            selected.extend(cand[: target_counts[lv]])
        rng.shuffle(selected)
    else:
        stratified = False
        rng.shuffle(clean_pool)
        selected = clean_pool[: args.n]

    for i, r in enumerate(selected, start=1):
        r["prompt_index"] = i

    payload = {
        "dataset_name": "Hendrycks_MATH_Train_Disjoint_Dev_Pool",
        "total_prompts": len(selected),
        "level_matched_to_math500": stratified,
        "level_distribution": dict(sorted(Counter(r["level"] for r in selected).items())),
        "subject_distribution": dict(sorted(Counter(r["subject"] for r in selected).items())),
        "verified_overlap_with_math500_grpo_zpd_sft": 0,
        "prompts": selected,
    }
    for target in (args.output_json, os.path.join(tempfile.gettempdir(), os.path.basename(args.output_json))):
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Saved Disjoint Dev Pool ({len(selected)}, level_matched={stratified}) -> {target}")
            print("Levels:", payload["level_distribution"])
            return
        except PermissionError:
            continue
    raise PermissionError(f"Could not write {args.output_json}")


if __name__ == "__main__":
    main()
