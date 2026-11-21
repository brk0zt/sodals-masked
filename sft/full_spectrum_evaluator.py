"""
SODALS v2.1 — Module 3.2: Full Spectrum Benchmark Evaluator
============================================================
Statistical rigor evaluation harness implementing:
  - PAC-Bayes generalization risk bounds
  - Wilson Score confidence intervals
  - 10,000-iteration Bootstrap confidence intervals
  - Integration with SymbolicExactMatchGrader (Tier 1-4)
  - Full-spectrum HLE split evaluation (N >= 8 or full test set)

Addresses:
  - Gray Area 2: Small sample size (N=8) statistical variance
  - Gray Area 4: Substring grading fragility
"""

import math
import json
import os
import re
import sys
import dataclasses
import numpy as np
from typing import List, Dict, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symbolic_grader import SymbolicExactMatchGrader, GradeResult

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")


# ── Statistical Helper Functions ─────────────────────────────────

def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation, especially when k=0 or k=n.
    
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # z-score for two-sided confidence
    z_table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_table.get(confidence, 1.960)
    
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    spread = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4 * (n**2)))
    
    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)
    
    return p_hat, ci_lower, ci_upper


def bootstrap_ci(matches: List[bool], n_bootstraps: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval for accuracy.
    """
    if not matches:
        return 0.0, 0.0, 0.0
    
    arr = np.array(matches, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(42)
    
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)
    
    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    point_est = float(np.mean(arr))
    
    return point_est, ci_lower, ci_upper


def pac_bayes_bound(train_acc: float, test_acc: float, n_samples: int, kl_prior: float = 0.5, delta: float = 0.05) -> float:
    """
    Compute PAC-Bayes generalization risk upper bound.
    
    Formula matching sodals_neuro::distillation::evaluator:
      Risk_PAC-Bayes = (1 - TrainAcc) + sqrt( (KL(P || Q) + ln(2/delta)) / (2 * m) )
    """
    empirical_risk = 1.0 - train_acc
    complexity = math.sqrt((kl_prior + math.log(2.0 / delta)) / (2.0 * max(n_samples, 1)))
    bound = empirical_risk + complexity
    return float(bound)


# ── Dataclasses ──────────────────────────────────────────────────

@dataclasses.dataclass
class EvaluationItemResult:
    question_id: str
    subject: str
    ground_truth: str
    model_answer: str
    grade: GradeResult


@dataclasses.dataclass
class FullSpectrumReport:
    total_questions: int
    correct_count: int
    raw_accuracy: float
    wilson_ci_lower: float
    wilson_ci_upper: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    pac_bayes_risk_bound: float
    method_breakdown: Dict[str, int]
    subject_accuracy: Dict[str, float]
    statistical_validity: str
    results: List[EvaluationItemResult]


# ── Full Spectrum Evaluator ──────────────────────────────────────

class FullSpectrumEvaluator:
    """Comprehensive evaluation harness with full statistical rigor."""
    
    def __init__(self):
        self.grader = SymbolicExactMatchGrader()
    
    def evaluate(self, 
                 predictions: List[Dict[str, str]], 
                 train_acc: float = 0.95,
                 n_train_samples: int = 144) -> FullSpectrumReport:
        """
        Evaluate a list of model predictions.
        Each item: {"id": ..., "subject": ..., "ground_truth": ..., "model_answer": ...}
        """
        results = []
        matches = []
        method_counts = {}
        subject_results = {}
        
        for item in predictions:
            qid = item.get("id", "unknown")
            subject = item.get("subject", "General")
            gt = item.get("ground_truth", "")
            ans = item.get("model_answer", "")
            
            grade_res = self.grader.grade(ans, gt)
            results.append(EvaluationItemResult(
                question_id=qid,
                subject=subject,
                ground_truth=gt,
                model_answer=ans,
                grade=grade_res
            ))
            
            matches.append(grade_res.match)
            method_counts[grade_res.method] = method_counts.get(grade_res.method, 0) + 1
            
            if subject not in subject_results:
                subject_results[subject] = []
            subject_results[subject].append(grade_res.match)
        
        n = len(results)
        k = sum(1 for m in matches if m)
        
        # Wilson score interval
        raw_acc, w_lower, w_upper = wilson_score_interval(k, n, confidence=0.95)
        
        # Bootstrap interval
        _, b_lower, b_upper = bootstrap_ci(matches, n_bootstraps=10000, confidence=0.95)
        
        # PAC-Bayes bound
        pb_bound = pac_bayes_bound(train_acc, raw_acc, n_train_samples + n)
        
        # Subject breakdown
        subj_acc = {s: float(np.mean(m)) for s, m in subject_results.items()}
        
        # Statistical validity judgment
        ci_width = w_upper - w_lower
        if n < 30:
            stat_val = f"⚠️ HIGH VARIANCE (N={n}, CI 95%: [{w_lower:.1%}, {w_upper:.1%}], width={ci_width:.1%}). Statistical power limited; claims must be qualified."
        elif ci_width < 0.15:
            stat_val = f"✅ HIGH STATISTICAL POWER (N={n}, CI 95%: [{w_lower:.1%}, {w_upper:.1%}])."
        else:
            stat_val = f"🟡 MODERATE STATISTICAL POWER (N={n}, CI 95%: [{w_lower:.1%}, {w_upper:.1%}])."
        
        return FullSpectrumReport(
            total_questions=n,
            correct_count=k,
            raw_accuracy=raw_acc,
            wilson_ci_lower=w_lower,
            wilson_ci_upper=w_upper,
            bootstrap_ci_lower=b_lower,
            bootstrap_ci_upper=b_upper,
            pac_bayes_risk_bound=pb_bound,
            method_breakdown=method_counts,
            subject_accuracy=subj_acc,
            statistical_validity=stat_val,
            results=results
        )


def main():
    print("=" * 72)
    print("  SODALS v2.1 — Full Spectrum Benchmark Evaluator")
    print("=" * 72)
    
    # Load K6 merged model outputs as benchmark test
    k6_results_path = os.path.join(ROOT, "hle_k6_merged_weights_Qwen3.5-9B.json")
    if not os.path.exists(k6_results_path):
        print(f"K6 results not found: {k6_results_path}")
        return
    
    with open(k6_results_path, "r", encoding="utf-8") as f:
        k6_data = json.load(f)
    
    # Correct ground truths
    ground_truths = {
        "669402b41dcb3d5a1ef9e951": {"gt": "\\mathbb{Z}^5", "subject": "Mathematics (Bordism)"},
        "6696c3734c196f1af6a16fcb": {"gt": "18", "subject": "Mathematics (Elliptic Curves)"},
        "66b2c7c9795022550abb176b": {"gt": "1+3x+6x^2+8x^3+6x^4+3x^5+x^6", "subject": "Mathematics (Lie Algebras)"},
        "66b727d367968fa27f2dddda": {"gt": "d-(d-2k)^2", "subject": "Mathematics (Clifford Algebras)"},
        "66b827b9b64deaedfbb997a2": {"gt": "3", "subject": "Physics (Warped KK)"},
        "66b91693d86bff9a12fc1f99": {"gt": "Katie kicked the kitten", "subject": "Puzzles (Substitution Cipher)"},
        "66ba5510db07a3a9151be0d2": {"gt": "\\mathcal{A}^{\\alpha-}(X)", "subject": "Mathematics (Microlocal Analysis)"},
        "66c5c3eae30bdd4f1da62cfc": {"gt": "\\mathbb{Z}", "subject": "Mathematics (Moduli Spaces)"},
    }
    
    items_to_eval = []
    for qid, val in k6_data.items():
        gt_info = ground_truths.get(qid, {"gt": "", "subject": "Unknown"})
        resp = val.get("response", "") if isinstance(val, dict) else str(val)
        m = re.findall(r'Answer:\s*(.*?)(?:\n|$)', resp, flags=re.IGNORECASE)
        ans = m[-1].strip() if m else resp.strip()
        items_to_eval.append({
            "id": qid,
            "subject": gt_info["subject"],
            "ground_truth": gt_info["gt"],
            "model_answer": ans
        })
    
    evaluator = FullSpectrumEvaluator()
    report = evaluator.evaluate(items_to_eval, train_acc=0.95, n_train_samples=144)
    
    print(f"\n  Evaluated: {report.total_questions} questions")
    print(f"  Exact Matches: {report.correct_count}/{report.total_questions} ({report.raw_accuracy:.1%})")
    print(f"  Wilson 95% CI: [{report.wilson_ci_lower:.1%}, {report.wilson_ci_upper:.1%}]")
    print(f"  Bootstrap 95% CI: [{report.bootstrap_ci_lower:.1%}, {report.bootstrap_ci_upper:.1%}]")
    print(f"  PAC-Bayes Risk Bound: {report.pac_bayes_risk_bound:.4f}")
    print(f"\n  Grader Method Breakdown:")
    for method, count in report.method_breakdown.items():
        print(f"    - {method}: {count}")
    print(f"\n  Statistical Assessment:\n  {report.statistical_validity}")
    
    print("\n  Per-Question Detailed Results:")
    for r in report.results:
        status = "✅ MATCH" if r.grade.match else "❌ NO MATCH"
        print(f"  [{status}] {r.question_id[:8]} ({r.subject})")
        print(f"     GT:    {r.ground_truth}")
        print(f"     Model: {r.model_answer}")
        print(f"     Grade: method={r.grade.method}, conf={r.grade.confidence:.2f}, details={r.grade.details}")
    
    print("=" * 72)


if __name__ == "__main__":
    main()
