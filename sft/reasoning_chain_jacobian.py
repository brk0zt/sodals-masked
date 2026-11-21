"""
SODALS v2.1 — Module 1.4: Lexical Step-Drift Quality Filter
============================================================
(Legacy module name: `reasoning_chain_jacobian.py`; alias `ReasoningChainJacobianFilter` preserved.)
Computes normalized L2 differences between consecutive CoT line-step hashed Bag-of-Words
vectors (`dim=64`) to detect abrupt semantic topic jumps ('proof abandonment') and variable loss
in Chain-of-Thought training data. Note: this is a lexical step-difference heuristic, NOT an
exact neural Jacobian matrix.
"""

import dataclasses
import json
import os
import re
import sys
import numpy as np
from typing import List, Dict, Optional

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "domain1_sft_curriculum.jsonl")


import hashlib

# ── Feature Extraction (reuses WronskianGramFilter logic) ────────

def _extract_features(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hash-based positional feature extraction (independent of Python hash seed)."""
    vec = np.zeros(dim, dtype=np.float64)
    words = text.lower().split()
    for i, w in enumerate(words):
        # Stable MD5 hash modulo dim
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16) % dim
        vec[h] += 1.0 / (1.0 + np.log1p(i))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Dataclasses ──────────────────────────────────────────────────

@dataclasses.dataclass
class JacobianReport:
    num_steps: int
    max_jacobian_norm: float
    mean_jacobian_norm: float
    std_jacobian_norm: float
    jump_ratio: float
    has_proof_abandonment: bool
    jump_location: int          # step index where max jump occurs
    step_norms: List[float]


@dataclasses.dataclass
class VariableTrajectory:
    name: str
    total_mentions: int
    mentions_per_quarter: List[int]
    present_in_conclusion: bool
    dropped: bool               # present in first half, absent in last quarter


@dataclasses.dataclass
class VariableReport:
    variables_found: List[str]
    variables_tracked: Dict[str, VariableTrajectory]
    any_dropped: bool
    dropped_variables: List[str]


@dataclasses.dataclass
class CombinedReport:
    sample_id: str
    jacobian: JacobianReport
    variables: VariableReport
    passes_both: bool


# ── Class 1: Reasoning Chain Jacobian Filter ─────────────────────

class LexicalStepDriftFilter:
    """
    Computes normalized L2 step-drift norms between consecutive CoT lines using hashed BoW vectors.
    High jump_ratio indicates proof abandonment or sudden topic shifts.
      ||delta_i|| = ||feature(step_i) - feature(step_{i-1})||
      jump_ratio = max(||delta||) / mean(||delta||)
    """

    def __init__(self, jump_threshold: float = 3.0, feature_dim: int = 64):
        self.jump_threshold = jump_threshold
        self.feature_dim = feature_dim

    def analyze(self, cot_text: str) -> JacobianReport:
        # Split into non-empty steps
        steps = [s.strip() for s in cot_text.split('\n') if s.strip()]

        if len(steps) < 2:
            return JacobianReport(
                num_steps=len(steps), max_jacobian_norm=0.0,
                mean_jacobian_norm=0.0, std_jacobian_norm=0.0,
                jump_ratio=0.0, has_proof_abandonment=False,
                jump_location=0, step_norms=[]
            )

        # Extract features per step
        features = [_extract_features(s, self.feature_dim) for s in steps]

        # Compute inter-step L2 norms (Jacobian norms)
        step_norms = []
        for i in range(1, len(features)):
            delta = np.linalg.norm(features[i] - features[i - 1])
            step_norms.append(delta)

        max_norm = max(step_norms)
        mean_norm = np.mean(step_norms)
        std_norm = np.std(step_norms)
        jump_ratio = max_norm / (mean_norm + 1e-8)
        jump_loc = int(np.argmax(step_norms)) + 1  # +1 for 1-indexed step

        return JacobianReport(
            num_steps=len(steps),
            max_jacobian_norm=round(max_norm, 6),
            mean_jacobian_norm=round(mean_norm, 6),
            std_jacobian_norm=round(std_norm, 6),
            jump_ratio=round(jump_ratio, 4),
            has_proof_abandonment=jump_ratio > self.jump_threshold,
            jump_location=jump_loc,
            step_norms=[round(n, 6) for n in step_norms]
        )


# Backward-compatibility alias for existing callers/tests
ReasoningChainJacobianFilter = LexicalStepDriftFilter


# ── Class 2: Variable Preservation Checker ───────────────────────

class VariablePreservationChecker:
    """
    Checks whether critical mathematical variables from the problem
    statement are preserved throughout the CoT reasoning chain.
    
    Analogous to PhaseSignal.jacobian_norm tracking in qwen_custom.rs:
    variables are "signals" that must maintain presence across layers.
    """

    # Patterns for extracting mathematical variables
    VARIABLE_PATTERNS = [
        r'(?<![a-zA-Z])([kdnmrpqNM])(?![a-zA-Z])',          # single-letter vars
        r'\\alpha|\\beta|\\gamma|\\delta|\\lambda|\\mu|\\theta|\\sigma|\\omega|\\epsilon|\\phi|\\psi|\\rho|\\tau|\\nu|\\xi|\\zeta|\\eta|\\chi',
        r'\\mathbb\{[A-Z]\}',
        r'\\mathfrak\{[a-z]+\}',
    ]

    def _auto_extract_variables(self, problem_text: str) -> List[str]:
        """Extract mathematical variables from the problem statement."""
        variables = set()
        for pattern in self.VARIABLE_PATTERNS:
            matches = re.findall(pattern, problem_text)
            for m in matches:
                v = m.strip()
                if v and len(v) <= 20:  # reasonable length
                    variables.add(v)
        return sorted(variables)

    def check(self, problem_text: str, cot_text: str,
              required_variables: Optional[List[str]] = None) -> VariableReport:
        if required_variables is None:
            required_variables = self._auto_extract_variables(problem_text)

        if not required_variables:
            return VariableReport(
                variables_found=[], variables_tracked={},
                any_dropped=False, dropped_variables=[]
            )

        # Split CoT into quarters
        lines = [l for l in cot_text.split('\n') if l.strip()]
        if len(lines) < 4:
            # Too short to quarter — treat as single block
            quarters = [lines, [], [], []]
        else:
            q_size = len(lines) // 4
            quarters = [
                lines[:q_size],
                lines[q_size:2 * q_size],
                lines[2 * q_size:3 * q_size],
                lines[3 * q_size:]
            ]

        tracked = {}
        dropped_vars = []

        for var in required_variables:
            mentions_per_q = []
            for q in quarters:
                q_text = '\n'.join(q)
                # Atomic mathematical token boundary matching (prevents single-letter vars matching inside words)
                escaped = re.escape(var)
                pattern = r'(?<![a-zA-Z0-9_])' + escaped + r'(?![a-zA-Z0-9_])'
                count = len(re.findall(pattern, q_text))
                mentions_per_q.append(count)

            total = sum(mentions_per_q)
            first_half = mentions_per_q[0] + mentions_per_q[1]
            last_quarter = mentions_per_q[3]
            in_conclusion = last_quarter > 0
            dropped = first_half > 0 and last_quarter == 0

            traj = VariableTrajectory(
                name=var,
                total_mentions=total,
                mentions_per_quarter=mentions_per_q,
                present_in_conclusion=in_conclusion,
                dropped=dropped
            )
            tracked[var] = traj
            if dropped:
                dropped_vars.append(var)

        return VariableReport(
            variables_found=required_variables,
            variables_tracked=tracked,
            any_dropped=len(dropped_vars) > 0,
            dropped_variables=dropped_vars
        )


# ── Combined Filter ──────────────────────────────────────────────

def filter_curriculum(jsonl_path: str) -> List[CombinedReport]:
    """
    Apply both Jacobian and variable preservation filters
    to every sample in the curriculum JSONL.
    """
    jac_filter = ReasoningChainJacobianFilter(jump_threshold=3.0)
    var_checker = VariablePreservationChecker()

    reports = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            sid = sample.get('id', 'unknown')

            # Extract user prompt and assistant CoT
            messages = sample.get('messages', [])
            user_msg = ''
            asst_msg = ''
            for m in messages:
                if m['role'] == 'user':
                    user_msg = m['content']
                elif m['role'] == 'assistant':
                    asst_msg = m['content']

            # Run Jacobian analysis on the CoT
            jac_report = jac_filter.analyze(asst_msg)

            # Run variable preservation check
            var_report = var_checker.check(user_msg, asst_msg)

            passes = not jac_report.has_proof_abandonment and not var_report.any_dropped

            reports.append(CombinedReport(
                sample_id=sid,
                jacobian=jac_report,
                variables=var_report,
                passes_both=passes
            ))

    return reports


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  SODALS v2.1 — Reasoning Chain Jacobian Quality Filter")
    print("=" * 72)

    if not os.path.exists(DATASET_PATH):
        print(f"  Dataset not found: {DATASET_PATH}")
        return

    reports = filter_curriculum(DATASET_PATH)

    passed = sum(1 for r in reports if r.passes_both)
    total = len(reports)
    jac_fails = sum(1 for r in reports if r.jacobian.has_proof_abandonment)
    var_fails = sum(1 for r in reports if r.variables.any_dropped)

    print(f"\n  Total samples: {total}")
    print(f"  Passed both filters: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"  Jacobian failures (proof abandonment): {jac_fails}")
    print(f"  Variable preservation failures: {var_fails}")

    # Show details for failing samples
    if jac_fails > 0 or var_fails > 0:
        print(f"\n  {'─' * 60}")
        print("  FAILING SAMPLES:")
        for r in reports:
            if not r.passes_both:
                print(f"\n  ⚠️  {r.sample_id}")
                if r.jacobian.has_proof_abandonment:
                    print(f"     Jacobian jump_ratio={r.jacobian.jump_ratio:.2f} "
                          f"at step {r.jacobian.jump_location} "
                          f"(threshold=3.0)")
                if r.variables.any_dropped:
                    print(f"     Dropped variables: {r.variables.dropped_variables}")
    else:
        print("\n  ✅ All samples pass both quality filters.")

    # Show top-5 highest jump ratios (even if passing)
    print(f"\n  {'─' * 60}")
    print("  TOP-5 HIGHEST JACOBIAN JUMP RATIOS:")
    sorted_reports = sorted(reports, key=lambda r: r.jacobian.jump_ratio, reverse=True)
    for r in sorted_reports[:5]:
        status = "⚠️" if r.jacobian.has_proof_abandonment else "✅"
        print(f"  {status} {r.sample_id}: "
              f"jump_ratio={r.jacobian.jump_ratio:.2f}, "
              f"steps={r.jacobian.num_steps}, "
              f"max_norm={r.jacobian.max_jacobian_norm:.4f}")

    print(f"\n{'=' * 72}")


if __name__ == '__main__':
    main()
