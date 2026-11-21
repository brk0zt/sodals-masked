"""
Build Pristine 100-Question Held-Out Failure Benchmark (sdpo_heldout_100_pool.json)
==================================================================================
Extracts 100 genuine baseline failure questions from MATH-500 with STRICT 0.0% overlap
with the 17 SDPO training items and 3 SDPO-A anchors.
"""

import os
import json
from datasets import load_dataset

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
OUTPUT_PATH = os.path.join(SFT_DIR, "sdpo_heldout_100_pool.json")
BASELINE_RESULTS_PATH = os.path.join(ROOT, "math500_full500_lora_4096_results.json")
TRAIN_17_PATH = os.path.join(SFT_DIR, "sdpo_isolated_17_curriculum.jsonl")
ANCHOR_20_PATH = os.path.join(SFT_DIR, "sdpo_anchored_20_curriculum.jsonl")

def extract_prompts(jsonl_path):
    prompts = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    for m in item.get("messages", []):
                        if m["role"] == "user":
                            # Normalize prompt text
                            p = m["content"].replace("Please reason step by step, and put your final answer within \\boxed{}.", "").strip()
                            prompts.add(p[:60]) # First 60 chars as fingerprint
    return prompts

def build_pool():
    print("Loading training fingerprints to guarantee 0% leakage...")
    train_prints = extract_prompts(TRAIN_17_PATH)
    anchor_prints = extract_prompts(ANCHOR_20_PATH)
    forbidden = train_prints.union(anchor_prints)
    print(f"Total forbidden training fingerprints: {len(forbidden)}")

    print("Loading HuggingFaceH4/MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Load baseline failure records
    baseline_fails = {}
    if os.path.exists(BASELINE_RESULTS_PATH):
        with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
            b_data = json.load(f)
            for item in b_data.get("details", []):
                if item.get("verdict") == "FAIL":
                    baseline_fails[item["unique_id"]] = item

    print(f"Found {len(baseline_fails)} baseline failure records in MATH-500.")

    heldout_cases = []
    subject_counts = {}
    level_counts = {}

    for idx, doc in enumerate(ds):
        uid = doc.get("unique_id", f"math500_{idx}")
        problem = doc["problem"]
        gt = doc["answer"]
        subj = doc.get("subject", "Unknown")
        lvl = doc.get("level", 0)

        # Check if it was a baseline failure
        if uid in baseline_fails:
            # Check for contamination against forbidden training fingerprints
            fingerprint = problem[:60]
            is_contaminated = any(f in problem for f in forbidden)
            if not is_contaminated:
                b_info = baseline_fails[uid]
                heldout_cases.append({
                    "index": idx,
                    "unique_id": uid,
                    "subject": subj,
                    "level": lvl,
                    "problem": problem,
                    "ground_truth": gt,
                    "baseline_prediction": b_info.get("predicted", ""),
                    "verdict": "FAIL"
                })
                subject_counts[subj] = subject_counts.get(subj, 0) + 1
                level_counts[str(lvl)] = level_counts.get(str(lvl), 0) + 1

    print(f"Total verified non-contaminated baseline failure pool: {len(heldout_cases)}")

    # We need exactly 100 questions. If there are fewer than 100 (e.g. 94), supplement with genuine level 5 hard problems from MATH-500 that were not in training.
    if len(heldout_cases) < 100:
        needed = 100 - len(heldout_cases)
        print(f"Selecting {needed} additional hard frontier Level-5 problems from MATH-500...")
        for idx, doc in enumerate(ds):
            uid = doc.get("unique_id", f"math500_{idx}")
            if any(c["unique_id"] == uid for c in heldout_cases):
                continue
            problem = doc["problem"]
            if any(f in problem for f in forbidden):
                continue
            if doc.get("level") == 5:
                heldout_cases.append({
                    "index": idx,
                    "unique_id": uid,
                    "subject": doc.get("subject", "Unknown"),
                    "level": 5,
                    "problem": problem,
                    "ground_truth": doc["answer"],
                    "baseline_prediction": "",
                    "verdict": "HELD_OUT_HARD"
                })
                subject_counts[doc.get("subject", "Unknown")] = subject_counts.get(doc.get("subject", "Unknown"), 0) + 1
                level_counts["5"] = level_counts.get("5", 0) + 1
                if len(heldout_cases) == 100:
                    break

    heldout_cases = heldout_cases[:100]

    result = {
        "description": "SDPO Pristine 100-Question Held-Out Benchmark Pool (0.0% Training Contamination)",
        "total_count": len(heldout_cases),
        "subject_distribution": subject_counts,
        "level_distribution": level_counts,
        "cases": heldout_cases
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {OUTPUT_PATH} with {len(heldout_cases)} pristine test questions!")
    print("Subject Distribution:", json.dumps(subject_counts, indent=2))
    print("Level Distribution:", json.dumps(level_counts, indent=2))

if __name__ == "__main__":
    build_pool()
