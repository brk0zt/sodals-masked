"""
Build Anchored SDPO Curriculum (SDPO-A)
=======================================
Combines:
  - 17 verified near-miss self-corrections (for failure repair reflex)
  - 3 frontier STEM anchors (to preserve Lie theory, symplectic, and modular forms)
Total: 20 examples.

Outputs: sft/sdpo_anchored_20_curriculum.jsonl
"""

import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SDPO_17_PATH = os.path.join(ROOT, "sft", "sdpo_isolated_17_curriculum.jsonl")
CONTROL_PATH = os.path.join(ROOT, "sft", "control_17_curriculum.jsonl")
OUTPUT_PATH = os.path.join(ROOT, "sft", "sdpo_anchored_20_curriculum.jsonl")

# 1. Load the 17 verified self-corrections
sdpo_items = [json.loads(l) for l in open(SDPO_17_PATH, "r", encoding="utf-8").read().splitlines() if l.strip()]
print(f"Loaded {len(sdpo_items)} verified SDPO self-corrections.")

# 2. Select 3 frontier anchors from control/master
control_items = [json.loads(l) for l in open(CONTROL_PATH, "r", encoding="utf-8").read().splitlines() if l.strip()]

# Anchor 1: Lie Theory (Item 17: g_2 roots and dimension formula)
anchor_lie = control_items[16]

# Anchor 2: Symplectic Geometry (Item 12: cotangent bundle dimension)
anchor_symp = control_items[11]

# Anchor 3: Complex Geometry / Euler characteristic (Item 5: CP^8)
anchor_euler = control_items[4]

anchors = [anchor_lie, anchor_symp, anchor_euler]
print(f"Selected {len(anchors)} frontier STEM anchors:")
for i, a in enumerate(anchors):
    print(f"  - Anchor {i+1}: {a['messages'][0]['content'][:100]}...")

combined = sdpo_items + anchors

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nSuccessfully created Anchored SDPO dataset ({len(combined)} examples) in {OUTPUT_PATH}")
