"""
SODALS v2.1 — Module 1.2: Wronskian Cross-Domain Contamination Auditor
======================================================================
Measures indirect semantic overlap between SFT training curriculum
and HLE benchmark questions using Wronskian-based independence testing.

Addresses Gray Areas:
  - Gray Area 1: Domain-Targeted Curriculum Narrowness
  - Gray Area 5: Indirect Topic Overlap

Integration with existing:
  - WronskianGramFilter from generate_sft_curriculum.py
  - PAC-Bayes bound from evaluator.rs (conceptual alignment)
"""

import json
import os
import sys
import dataclasses
import numpy as np
from typing import List, Dict, Optional
from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "domain1_sft_curriculum.jsonl")
TEST_IDS_PATH = os.path.join(ROOT, "hle-main", "hle_eval", "test_question_ids.json")


# ── Feature Extraction (high-resolution version) ─────────────────

def _extract_features(text: str, dim: int = 128) -> np.ndarray:
    """
    Higher-resolution feature extraction than the standard
    WronskianGramFilter (128-D vs 64-D) for finer contamination detection.
    """
    vec = np.zeros(dim, dtype=np.float64)
    words = text.lower().split()
    for i, w in enumerate(words):
        h = hash(w) % dim
        vec[h] += 1.0 / (1.0 + np.log1p(i))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Dataclasses ──────────────────────────────────────────────────

@dataclasses.dataclass
class SampleContaminationScore:
    sample_id: str
    topic: str
    residual_norm: float
    most_similar_benchmark_qid: str
    most_similar_benchmark_topic: str
    cosine_similarity: float
    is_contaminated: bool


@dataclasses.dataclass
class ContaminationReport:
    total_curriculum_samples: int
    total_benchmark_questions: int
    contaminated_count: int
    contamination_rate: float
    min_residual_norm: float
    max_cosine_similarity: float
    cross_domain_wronskian_det: float
    independence_threshold: float
    verdict: str
    per_sample_scores: List[SampleContaminationScore]


# ── Wronskian Cross-Domain Auditor ───────────────────────────────

class CurriculumContaminationAuditor:
    """
    Wronskian-based cross-domain independence auditor.
    
    Tests whether training curriculum samples are linearly independent
    from benchmark test questions in the semantic feature space.
    
    Uses the same Gram-Schmidt orthogonalization principle as
    WronskianGramFilter but applied ACROSS datasets rather than within.
    
    Mathematical foundation:
      For benchmark basis vectors B = {b_1, ..., b_n}:
      residual(v) = v - Σ (v·b_i) b_i
      
      If ||residual|| < threshold → v is too close to benchmark subspace
      → contamination risk detected
    """

    def __init__(self,
                 independence_threshold: float = 0.3,
                 feature_dim: int = 128):
        self.independence_threshold = independence_threshold
        self.feature_dim = feature_dim
        self.benchmark_basis: List[np.ndarray] = []
        self.benchmark_meta: List[Dict] = []

    def _compute_residual(self, v: np.ndarray) -> np.ndarray:
        """Gram-Schmidt residual against benchmark basis."""
        residual = v.copy()
        for b in self.benchmark_basis:
            proj = np.dot(residual, b) * b
            residual -= proj
        return residual

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def load_benchmark_questions(self, questions: List[Dict]):
        """
        Build the benchmark subspace basis from HLE test questions.
        Each question contributes a normalized feature vector.
        """
        self.benchmark_basis = []
        self.benchmark_meta = []
        for q in questions:
            text = q.get("question", "") + " " + q.get("answer", "")
            v = _extract_features(text, self.feature_dim)
            self.benchmark_basis.append(v)
            self.benchmark_meta.append({
                "qid": q.get("id", "unknown"),
                "subject": q.get("subject", "unknown")
            })

    def audit_sample(self, sample_text: str, sample_id: str,
                     sample_topic: str) -> SampleContaminationScore:
        """Audit a single curriculum sample against the benchmark basis."""
        v = _extract_features(sample_text, self.feature_dim)

        # Compute residual after projecting out benchmark subspace
        residual = self._compute_residual(v)
        residual_norm = float(np.linalg.norm(residual))

        # Find most similar benchmark question
        max_cos = -1.0
        most_similar_qid = ""
        most_similar_topic = ""
        for i, b in enumerate(self.benchmark_basis):
            cos = self._cosine_sim(v, b)
            if cos > max_cos:
                max_cos = cos
                most_similar_qid = self.benchmark_meta[i]["qid"]
                most_similar_topic = self.benchmark_meta[i]["subject"]

        return SampleContaminationScore(
            sample_id=sample_id,
            topic=sample_topic,
            residual_norm=round(residual_norm, 6),
            most_similar_benchmark_qid=most_similar_qid,
            most_similar_benchmark_topic=most_similar_topic,
            cosine_similarity=round(max_cos, 6),
            is_contaminated=residual_norm < self.independence_threshold
        )

    def compute_wronskian_determinant(self,
                                      curriculum_features: List[np.ndarray]
                                      ) -> float:
        """
        Compute the Wronskian-style cross-domain independence measure.
        
        This is the Gram determinant of the combined matrix
        [benchmark_vectors | curriculum_vectors].
        
        If det > 0: the curriculum spans a space independent of benchmark.
        If det ≈ 0: the spaces overlap significantly.
        """
        all_vectors = self.benchmark_basis + curriculum_features
        if not all_vectors:
            return 0.0

        # Use SVD-based condition number as a proxy for determinant
        # (actual determinant overflows for high dimensions)
        mat = np.array(all_vectors)
        try:
            svds = np.linalg.svd(mat, compute_uv=False)
            # Ratio of smallest to largest singular value
            if svds[-1] < 1e-10:
                return 0.0
            condition_number = svds[-1] / svds[0]
            return float(condition_number)
        except np.linalg.LinAlgError:
            return 0.0

    def full_audit(self, curriculum_path: str) -> ContaminationReport:
        """
        Complete audit: load curriculum, score each sample,
        compute cross-domain Wronskian determinant.
        """
        # Load curriculum
        samples = []
        with open(curriculum_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Audit each sample
        scores = []
        curriculum_features = []
        unique_ids_seen = set()

        for sample in samples:
            sid = sample.get("id", "unknown")
            topic = sample.get("topic", "unknown")

            # Extract text from messages
            text_parts = []
            for m in sample.get("messages", []):
                text_parts.append(m.get("content", ""))
            full_text = " ".join(text_parts)

            # Skip duplicates (augmentations of same base)
            base_id = sid.rsplit("_var", 1)[0].rsplit("_aug", 1)[0]
            if base_id not in unique_ids_seen:
                unique_ids_seen.add(base_id)
                v = _extract_features(full_text, self.feature_dim)
                curriculum_features.append(v)

            score = self.audit_sample(full_text, sid, topic)
            scores.append(score)

        # Compute cross-domain Wronskian
        wronskian_det = self.compute_wronskian_determinant(curriculum_features)

        contaminated = sum(1 for s in scores if s.is_contaminated)
        min_residual = min(s.residual_norm for s in scores) if scores else 0.0
        max_cosine = max(s.cosine_similarity for s in scores) if scores else 0.0

        # Verdict
        if contaminated == 0 and wronskian_det > 0.01:
            verdict = "✅ PASS — Curriculum is independent of benchmark subspace"
        elif contaminated == 0:
            verdict = "⚠️ MARGINAL — No direct contamination but low Wronskian det"
        else:
            verdict = f"❌ FAIL — {contaminated} samples too close to benchmark"

        return ContaminationReport(
            total_curriculum_samples=len(samples),
            total_benchmark_questions=len(self.benchmark_basis),
            contaminated_count=contaminated,
            contamination_rate=contaminated / max(len(samples), 1),
            min_residual_norm=min_residual,
            max_cosine_similarity=max_cosine,
            cross_domain_wronskian_det=round(wronskian_det, 6),
            independence_threshold=self.independence_threshold,
            verdict=verdict,
            per_sample_scores=scores
        )


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  SODALS v2.1 — Wronskian Cross-Domain Contamination Auditor")
    print("=" * 72)

    auditor = CurriculumContaminationAuditor(
        independence_threshold=0.3,
        feature_dim=128
    )

    # Load HLE benchmark questions
    print("\n  Loading HLE benchmark questions from cais/hle...")
    try:
        with open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
            target_ids = json.load(f)

        dataset = load_dataset("cais/hle", split="test").to_dict()
        all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
        id_to_q = {q["id"]: q for q in all_q}

        benchmark_questions = []
        for qid in target_ids:
            if qid in id_to_q:
                benchmark_questions.append(id_to_q[qid])

        print(f"  Loaded {len(benchmark_questions)} benchmark questions")
    except Exception as e:
        print(f"  ⚠️ Could not load HLE dataset: {e}")
        print("  Using fallback: creating synthetic benchmark representations")
        # Fallback: use known HLE question IDs and topics
        benchmark_questions = [
            {"id": "669402b4", "question": "Compute reduced 12-dimensional Spin bordism of BG2", "answer": "Z+Z+Z+Z+Z", "subject": "Mathematics"},
            {"id": "6696c373", "question": "Largest order of non-cyclic torsion subgroup elliptic curve Q(sqrt(-3))", "answer": "18", "subject": "Mathematics"},
            {"id": "66b2c7c9", "question": "Poincare polynomial 6-dimensional nilpotent Lie algebra", "answer": "1+3x+6x^2+8x^3", "subject": "Mathematics"},
            {"id": "66b727d3", "question": "Clifford gamma contraction in d dimensions rank k", "answer": "d-(d-2k)^2", "subject": "Mathematics"},
            {"id": "66b827b9", "question": "5D warped Kaluza-Klein graviton spin-2 modes mass squared less than 14", "answer": "3", "subject": "Physics"},
            {"id": "66b91693", "question": "Two-step substitution cipher decode Katie", "answer": "Katie kicked", "subject": "Puzzles"},
            {"id": "66ba5510", "question": "Resolvent mapping property conormal space Schwarzschild", "answer": "A^alpha-(X)", "subject": "Mathematics"},
            {"id": "66c5c3ea", "question": "First integral homology moduli space unimodular lattices", "answer": "Z", "subject": "Mathematics"},
        ]
        print(f"  Using {len(benchmark_questions)} fallback benchmark questions")

    auditor.load_benchmark_questions(benchmark_questions)

    # Run audit
    print(f"\n  Auditing curriculum: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"  ❌ File not found: {DATASET_PATH}")
        return

    report = auditor.full_audit(DATASET_PATH)

    # Print report
    print(f"\n  {'─' * 60}")
    print(f"  CONTAMINATION AUDIT REPORT")
    print(f"  {'─' * 60}")
    print(f"  Curriculum samples:     {report.total_curriculum_samples}")
    print(f"  Benchmark questions:    {report.total_benchmark_questions}")
    print(f"  Independence threshold: {report.independence_threshold}")
    print(f"  {'─' * 60}")
    print(f"  Contaminated samples:   {report.contaminated_count} "
          f"({100*report.contamination_rate:.1f}%)")
    print(f"  Min residual norm:      {report.min_residual_norm:.4f}")
    print(f"  Max cosine similarity:  {report.max_cosine_similarity:.4f}")
    print(f"  Cross-domain Wronskian: {report.cross_domain_wronskian_det:.6f}")
    print(f"  {'─' * 60}")
    print(f"  VERDICT: {report.verdict}")
    print(f"  {'─' * 60}")

    # Show top-5 most similar to benchmark
    print(f"\n  TOP-5 CLOSEST TO BENCHMARK (highest risk):")
    sorted_scores = sorted(report.per_sample_scores,
                           key=lambda s: s.cosine_similarity, reverse=True)
    seen_bases = set()
    count = 0
    for s in sorted_scores:
        base_id = s.sample_id.rsplit("_var", 1)[0].rsplit("_aug", 1)[0]
        if base_id in seen_bases:
            continue
        seen_bases.add(base_id)
        status = "❌" if s.is_contaminated else "✅"
        print(f"  {status} {s.sample_id}")
        print(f"     Topic:     {s.topic}")
        print(f"     Cos sim:   {s.cosine_similarity:.4f}")
        print(f"     Residual:  {s.residual_norm:.4f}")
        print(f"     Nearest:   {s.most_similar_benchmark_qid} "
              f"({s.most_similar_benchmark_topic})")
        count += 1
        if count >= 5:
            break

    print(f"\n{'=' * 72}")


if __name__ == '__main__':
    main()
