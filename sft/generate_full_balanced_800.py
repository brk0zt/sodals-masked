"""
Generates the complete 800+ balanced multi-domain curriculum dataset.
"""

import os
import sys
import json

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
OUTPUT_PATH = os.path.join(ROOT, "sft", "master_800_balanced_curriculum.jsonl")

sys.path.insert(0, os.path.join(ROOT, "sft"))

# Import base generation
import build_balanced_800_curriculum
from append_to_curriculum import generate_additional_domains
from append_batch2 import generate_batch2

def build_full_dataset():
    dataset = list(build_balanced_800_curriculum.dataset)
    print(f"Base dataset has {len(dataset)} examples.")
    
    dataset = generate_additional_domains(dataset)
    print(f"Expanded dataset (batch 1) has {len(dataset)} examples.")
    
    dataset = generate_batch2(dataset)
    print(f"Expanded dataset (batch 2) has {len(dataset)} examples.")
    
    # Save to JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"[SUCCESS] Saved {len(dataset)} examples to {OUTPUT_PATH}")
    
    # Domain breakdown analysis
    domains = {}
    for item in dataset:
        dom = item.get("metadata", {}).get("domain", "Unknown")
        domains[dom] = domains.get(dom, 0) + 1
        
    print("\n--- FINAL DOMAIN BALANCE REPORT ---")
    for dom, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(dataset)
        print(f"  {dom:<35}: {count:4d} ({pct:5.1f}%)")

if __name__ == "__main__":
    build_full_dataset()
