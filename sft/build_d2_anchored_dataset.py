"""
SODALS v2.1 — Build Domain 2 Anchored SFT Curriculum (164 samples)
==================================================================
Merges:
  - 144 Domain 2 curriculum samples (Lie algebras, Groups, Rings, Galois)
  - 20 SDPO-A anchor samples (Self-corrections + Math/Lie anchors)
To create an anti-forgetting training dataset for SDPO-D2.

Output: sft/domain2_anchored_164_curriculum.jsonl
"""

import os
import json
import random

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")

D2_PATH = os.path.join(SFT_DIR, "domain2_sft_curriculum.jsonl")
ANCHOR_PATH = os.path.join(SFT_DIR, "sdpo_anchored_20_curriculum.jsonl")
OUTPUT_PATH = os.path.join(SFT_DIR, "domain2_anchored_164_curriculum.jsonl")

def main():
    print(f"Reading Domain 2 curriculum from {D2_PATH}...")
    d2_samples = []
    with open(D2_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d2_samples.append(json.loads(line))
    print(f"Loaded {len(d2_samples)} Domain 2 samples.")

    print(f"Reading Anchor samples from {ANCHOR_PATH}...")
    anchor_samples = []
    with open(ANCHOR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                anchor_samples.append(json.loads(line))
    print(f"Loaded {len(anchor_samples)} Anchor samples.")

    # Interleave anchor samples across the Domain 2 curriculum to maintain continuous anti-forgetting gradient
    # 144 / 20 ≈ 7.2, so insert an anchor sample roughly every 7-8 samples
    combined = []
    random.seed(42)
    # Shuffle anchors slightly
    anchors_shuffled = list(anchor_samples)
    random.shuffle(anchors_shuffled)
    
    interval = len(d2_samples) / max(len(anchors_shuffled), 1)
    anchor_idx = 0
    next_anchor_pos = interval / 2.0

    for i, sample in enumerate(d2_samples):
        if anchor_idx < len(anchors_shuffled) and i >= next_anchor_pos:
            combined.append(anchors_shuffled[anchor_idx])
            anchor_idx += 1
            next_anchor_pos += interval
        combined.append(sample)

    # Append any remaining anchors
    while anchor_idx < len(anchors_shuffled):
        combined.append(anchors_shuffled[anchor_idx])
        anchor_idx += 1

    print(f"Total merged samples: {len(combined)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote {len(combined)} samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
