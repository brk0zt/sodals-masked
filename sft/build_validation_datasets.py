"""
SODALS v2.1: Build Locked Stratified Benchmark Datasets
======================================================
1. HLE 50Q: 50 text-only questions stratified by category from cais/hle (excluding previous 30/8 IDs).
2. MMLU-Pro 100Q: 100 STEM questions stratified by discipline from TIGER-Lab/MMLU-Pro.
"""

import os
import json
import random
from datasets import load_dataset
from collections import defaultdict

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")

HLE_EXCLUDED_PATH = os.path.join(ROOT, "hle_30_test_question_ids.json")
HLE_50_OUT_PATH = os.path.join(SFT_DIR, "hle_50_stratified_test_cases.json")
MMLU_PRO_100_OUT_PATH = os.path.join(SFT_DIR, "mmlu_pro_100_stem_cases.json")


def build_hle_50():
    print("--- Building HLE 50Q Stratified Dataset ---")
    with open(HLE_EXCLUDED_PATH, "r", encoding="utf-8") as f:
        excluded = set(json.load(f))
    print(f"Loaded {len(excluded)} excluded IDs.")

    ds = load_dataset("cais/hle", split="test")
    print(f"Total HLE test questions: {len(ds)}")

    by_cat = defaultdict(list)
    for item in ds:
        qid = item["id"]
        img = item.get("image", "")
        cat = item.get("category", "Other")
        # Text-only and not previously evaluated
        if (not img) and (qid not in excluded):
            by_cat[cat].append({
                "id": qid,
                "question": item["question"],
                "ground_truth": item.get("answer", ""),
                "answer_type": item.get("answer_type", ""),
                "category": cat,
                "raw_subject": item.get("raw_subject", "")
            })

    print("Candidate counts per category:")
    for c, qlist in sorted(by_cat.items()):
        print(f"  {c}: {len(qlist)}")

    target_alloc = {
        "Math": 20,
        "Physics": 6,
        "Computer Science/AI": 6,
        "Chemistry": 4,
        "Engineering": 4,
        "Biology/Medicine": 4,
        "Humanities/Social Science": 3,
        "Other": 3
    }

    selected = []
    rng = random.Random(2026)
    for cat, count in target_alloc.items():
        candidates = by_cat.get(cat, [])
        if len(candidates) < count:
            raise ValueError(f"Not enough questions in {cat}: need {count}, got {len(candidates)}")
        # Sort by ID for stability before sampling with seed
        candidates.sort(key=lambda x: x["id"])
        chosen = rng.sample(candidates, count)
        for q in chosen:
            selected.append(q)
        print(f"Sampled {len(chosen)} questions from {cat}")

    print(f"Total HLE 50 selected: {len(selected)}")
    with open(HLE_50_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Saved to {HLE_50_OUT_PATH}")


def build_mmlu_pro_100():
    print("\n--- Building MMLU-Pro 100Q STEM Dataset ---")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Total MMLU-Pro test questions: {len(ds)}")

    by_cat = defaultdict(list)
    for item in ds:
        cat = item.get("category", "other")
        by_cat[cat].append({
            "question_id": item["question_id"],
            "question": item["question"],
            "options": item["options"],
            "ground_truth": item["answer"],
            "answer_index": item["answer_index"],
            "category": cat,
            "src": item.get("src", "")
        })

    target_alloc = {
        "math": 30,
        "physics": 25,
        "chemistry": 20,
        "computer science": 15,
        "engineering": 10
    }

    selected = []
    rng = random.Random(2026)
    for cat, count in target_alloc.items():
        candidates = by_cat.get(cat, [])
        if len(candidates) < count:
            raise ValueError(f"Not enough questions in {cat}: need {count}, got {len(candidates)}")
        candidates.sort(key=lambda x: str(x["question_id"]))
        chosen = rng.sample(candidates, count)
        for q in chosen:
            selected.append(q)
        print(f"Sampled {len(chosen)} questions from {cat}")

    print(f"Total MMLU-Pro 100 selected: {len(selected)}")
    with open(MMLU_PRO_100_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Saved to {MMLU_PRO_100_OUT_PATH}")


if __name__ == "__main__":
    build_hle_50()
    build_mmlu_pro_100()
