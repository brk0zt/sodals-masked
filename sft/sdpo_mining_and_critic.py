"""
SDPO Phase 1: Rigorous Failure Mining with Train/Held-Out Split (seed=42)
========================================================================
Mines 111 failed questions from MATH-500 full run and applies a strict
stratified split across all 7 subjects:
  - SDPO Train Failure Pool: 75 questions (~68%) -> Used for self-correction generation
  - SDPO Held-Out Test Pool: 36 questions (~32%) -> STRICTLY UNSEEN, tests pure generalization

Saves to:
  - sft/sdpo_train_failures.json
  - sft/sdpo_heldout_failures.json
"""

import sys
import os
import json
import random

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
RESULTS_JSON = os.path.join(ROOT, "math500_full500_lora_4096_results.json")
TRAIN_OUTPUT_JSON = os.path.join(ROOT, "sft", "sdpo_train_failures.json")
HELDOUT_OUTPUT_JSON = os.path.join(ROOT, "sft", "sdpo_heldout_failures.json")

print("=" * 80)
print("SDPO STEP 1: FAILURE MINING WITH STRICT TRAIN/HELD-OUT STRATIFIED SPLIT")
print("=" * 80)

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data.get("details", [])
fails = [r for r in details if r["verdict"] == "FAIL"]

print(f"Total Evaluated Questions : {len(details)}")
print(f"Total Successful (PASS)   : {len(details) - len(fails)} ({100*(len(details)-len(fails))/len(details):.2f}%)")
print(f"Total Failures Mined (FAIL): {len(fails)} ({100*len(fails)/len(details):.2f}%)")

# Group failures by subject for stratified splitting
fails_by_subject = {}
for f_item in fails:
    s = f_item["subject"]
    if s not in fails_by_subject:
        fails_by_subject[s] = []
    fails_by_subject[s].append(f_item)

random.seed(42)

train_pool = []
heldout_pool = []

print("\n--- SUBJECT-BY-SUBJECT STRATIFIED SPLIT (SEED = 42) ---")
print(f"{'Subject':25s} | {'Total Fails':12s} | {'Train (Self-Corr)':18s} | {'Held-Out Test':15s}")
print("-" * 75)

for s, items in sorted(fails_by_subject.items()):
    random.shuffle(items)
    # ~68% train, ~32% held-out
    n_heldout = max(1, int(round(len(items) * 0.32)))
    h_items = items[:n_heldout]
    t_items = items[n_heldout:]

    heldout_pool.extend(h_items)
    train_pool.extend(t_items)

    print(f"{s:25s} | {len(items):11d} | {len(t_items):17d} | {len(h_items):14d}")

print("-" * 75)
print(f"{'TOTAL':25s} | {len(fails):11d} | {len(train_pool):17d} | {len(heldout_pool):14d}")

# Save train failures
with open(TRAIN_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "description": "SDPO Train Failure Pool (used for self-correction generation & training)",
        "count": len(train_pool),
        "cases": train_pool
    }, f, indent=2, ensure_ascii=False)

# Save held-out failures
with open(HELDOUT_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "description": "SDPO Held-Out Test Pool (STRICTLY UNTOUCHED, used only for evaluating generalization)",
        "count": len(heldout_pool),
        "cases": heldout_pool
    }, f, indent=2, ensure_ascii=False)

print(f"\nSuccessfully exported:")
print(f"  - Train Pool ({len(train_pool)} items)    -> {TRAIN_OUTPUT_JSON}")
print(f"  - Held-Out Pool ({len(heldout_pool)} items) -> {HELDOUT_OUTPUT_JSON}")
