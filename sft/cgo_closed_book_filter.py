"""
CGO (Option b) Closed-Book Geometric Curriculum Filter
======================================================
Implements CGO strictly at TRAINING TIME:
  - Takes verified self-correction trajectories from SDPO
  - Merges them into the balanced master curriculum without data leakage
  - Computes Hamiltonian difficulty stratification and geometric ordering
  - Outputs a principled, closed-book training dataset (sdpo_cgo_curriculum.jsonl)
"""

import sys
import os
import json
import math

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
BALANCED_CURRICULUM_PATH = os.path.join(SFT_DIR, "master_800_balanced_curriculum.jsonl")
SDPO_TRAJECTORIES_PATH = os.path.join(SFT_DIR, "sdpo_verified_trajectories.jsonl")
OUTPUT_CURRICULUM_PATH = os.path.join(SFT_DIR, "sdpo_cgo_curriculum.jsonl")


def build_cgo_curriculum():
    print("=" * 80)
    print("CGO (OPTION B): CLOSED-BOOK GEOMETRIC CURRICULUM COMPILATION")
    print("=" * 80)

    # 1. Load balanced baseline curriculum
    base_data = []
    if os.path.exists(BALANCED_CURRICULUM_PATH):
        with open(BALANCED_CURRICULUM_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    base_data.append(json.loads(line))
        print(f"Loaded {len(base_data)} base curriculum examples from {BALANCED_CURRICULUM_PATH}.")

    # 2. Load SDPO self-corrections
    sdpo_data = []
    if os.path.exists(SDPO_TRAJECTORIES_PATH):
        with open(SDPO_TRAJECTORIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sdpo_data.append(item)
        print(f"Loaded {len(sdpo_data)} verified self-correction trajectories from {SDPO_TRAJECTORIES_PATH}.")
    else:
        print(f"Warning: {SDPO_TRAJECTORIES_PATH} does not exist yet (still generating).")

    # 3. Format SDPO examples into standard chat curriculum format
    formatted_sdpo = []
    for item in sdpo_data:
        prob = item["problem"]
        sol = item["solution_cot"]
        subj = item.get("subject", "General Math")
        lvl = item.get("level", 3)

        formatted_sdpo.append({
            "id": f"sdpo_{item.get('unique_id', 'case')}",
            "messages": [
                {
                    "role": "user",
                    "content": f"{prob}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{sol}"
                }
            ],
            "metadata": {
                "source": "sdpo_self_correction",
                "subject": subj,
                "difficulty_rank": float(lvl) if isinstance(lvl, (int, float)) else 3.0,
                "curriculum_stage": min(5, max(1, int(lvl) if str(lvl).isdigit() else 3)),
                "cgo_energy_weight": 1.25  # Higher focal weight for self-repaired cognitive pathways
            }
        })

    # 4. Merge and apply CGO geometric Hamiltonian sorting
    combined = list(base_data) + formatted_sdpo

    # Sort by Hamiltonian stage and difficulty
    def cgo_sort_key(x):
        meta = x.get("metadata", {})
        stage = meta.get("curriculum_stage", 3)
        diff = meta.get("difficulty_rank", 1.0)
        return (stage, diff)

    combined.sort(key=cgo_sort_key)
    print(f"Total compiled CGO curriculum: {len(combined)} examples (Base: {len(base_data)}, SDPO: {len(formatted_sdpo)}).")

    with open(OUTPUT_CURRICULUM_PATH, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Exported to {OUTPUT_CURRICULUM_PATH}")
    return len(combined)


if __name__ == "__main__":
    build_cgo_curriculum()
