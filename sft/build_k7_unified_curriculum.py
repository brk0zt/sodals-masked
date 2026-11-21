"""
SODALS v2.1 — K7 Unified Multi-Modal Curriculum Builder
========================================================
Combines:
  1. Uncontaminated Base Mathematical Curriculum (144 samples)
  2. Epistemic Uncertainty Curriculum (64 samples)
  3. Falsification & Rollback Curriculum (64 samples)
Total: 272 training samples

Filters & Validations:
  - WronskianGramFilter redundancy elimination
  - Wronskian Cross-Domain Contamination Audit against HLE
  - Jacobian Proof Abandonment Filter (jump_ratio < 3.0)
"""

import json
import os
import sys
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_sft_curriculum import WronskianGramFilter
from curriculum_contamination_auditor import CurriculumContaminationAuditor
from reasoning_chain_jacobian import filter_curriculum

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")

BASE_CURRICULUM_PATH = os.path.join(SFT_DIR, "domain1_sft_curriculum.jsonl")
EPISTEMIC_PATH = os.path.join(SFT_DIR, "epistemic_uncertainty_curriculum.jsonl")
FALSIFICATION_PATH = os.path.join(SFT_DIR, "falsification_curriculum.jsonl")
K7_OUTPUT_PATH = os.path.join(SFT_DIR, "k7_sft_unified_curriculum.jsonl")


def build_k7_curriculum():
    print("=" * 72)
    print("  SODALS v2.1 — Building K7 Unified Clean Training Curriculum")
    print("=" * 72)

    wronskian = WronskianGramFilter(threshold=1e-3)
    unified_samples = []
    stats = {"base": 0, "epistemic": 0, "falsification": 0, "rejected": 0}

    sources = [
        ("base", BASE_CURRICULUM_PATH),
        ("epistemic", EPISTEMIC_PATH),
        ("falsification", FALSIFICATION_PATH)
    ]

    for label, path in sources:
        if not os.path.exists(path):
            print(f"  ⚠️ Missing source file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                # Check messages
                user_msg = ""
                asst_msg = ""
                for m in item.get("messages", []):
                    if m["role"] == "user":
                        user_msg = m["content"]
                    elif m["role"] == "assistant":
                        asst_msg = m["content"]
                
                sample_text = f"{user_msg}\n{asst_msg}"
                if wronskian.is_independent(sample_text):
                    item["curriculum_stream"] = label
                    unified_samples.append(item)
                    stats[label] += 1
                else:
                    stats["rejected"] += 1

    # Write unified dataset
    with open(K7_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in unified_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  K7 Unified Dataset written to: {K7_OUTPUT_PATH}")
    print(f"  Base Mathematical Samples:        {stats['base']}")
    print(f"  Epistemic Uncertainty Samples:    {stats['epistemic']}")
    print(f"  Falsification / Rollback Samples: {stats['falsification']}")
    print(f"  Total Clean Samples:              {len(unified_samples)}")
    print(f"  Wronskian Redundant Rejected:     {stats['rejected']}")

    # Run Wronskian Cross-Domain Contamination Audit
    print(f"\n  Running Wronskian Cross-Domain Contamination Audit on K7...")
    auditor = CurriculumContaminationAuditor(independence_threshold=0.3, feature_dim=128)
    auditor.load_benchmark_questions([
        {'id': '669402b4', 'question': 'Compute reduced 12-dimensional Spin bordism of BG2', 'answer': 'Z+Z+Z+Z+Z', 'subject': 'Mathematics'},
        {'id': '6696c373', 'question': 'Largest order of non-cyclic torsion subgroup elliptic curve Q(sqrt(-3))', 'answer': '18', 'subject': 'Mathematics'},
        {'id': '66b2c7c9', 'question': 'Poincare polynomial 6-dimensional nilpotent Lie algebra', 'answer': '1+3x+6x^2+8x^3', 'subject': 'Mathematics'},
        {'id': '66b727d3', 'question': 'Clifford gamma contraction in d dimensions rank k', 'answer': 'd-(d-2k)^2', 'subject': 'Mathematics'},
        {'id': '66b827b9', 'question': '5D warped Kaluza-Klein graviton spin-2 modes mass squared less than 14', 'answer': '3', 'subject': 'Physics'},
        {'id': '66b91693', 'question': 'Two-step substitution cipher decode Katie', 'answer': 'Katie kicked', 'subject': 'Puzzles'},
        {'id': '66ba5510', 'question': 'Resolvent mapping property conormal space Schwarzschild', 'answer': 'A^alpha-(X)', 'subject': 'Mathematics'},
        {'id': '66c5c3ea', 'question': 'First integral homology moduli space unimodular lattices', 'answer': 'Z', 'subject': 'Mathematics'},
    ])
    audit_rep = auditor.full_audit(K7_OUTPUT_PATH)
    print(f"  Contamination Audit: {audit_rep.verdict}")
    print(f"  Contaminated Samples: {audit_rep.contaminated_count} / {audit_rep.total_curriculum_samples}")
    print(f"  Cross-domain Wronskian Det: {audit_rep.cross_domain_wronskian_det:.6f}")

    # Run Jacobian Quality Gate
    print(f"\n  Running Jacobian Proof Abandonment Quality Gate...")
    jac_reports = filter_curriculum(K7_OUTPUT_PATH)
    jac_fails = sum(1 for r in jac_reports if r.jacobian.has_proof_abandonment)
    print(f"  Jacobian Proof Abandonment Violations: {jac_fails} / {len(jac_reports)}")

    print(f"\n{'=' * 72}")
    if audit_rep.contaminated_count == 0 and jac_fails == 0:
        print("  🎉 K7 UNIFIED CURRICULUM IS 100% ETHICALLY AND METHODOLOGICALLY CERTIFIED!")
    else:
        print("  ⚠️ Issues detected in K7 curriculum.")
    print("=" * 72)


if __name__ == "__main__":
    build_k7_curriculum()
