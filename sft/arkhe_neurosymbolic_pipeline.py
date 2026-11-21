"""
SODALS v2.1 — Two-Layer Decoupled Arkhe Neurosymbolic Pipeline
==============================================================
Implements the Six Cognitive Arkhes as composable, deterministic software
engineering design patterns (Functional Core, Imperative Shell):

  0. StateArche (Durum Arkhesi)        : Immutable `ArkheState` & `CandidateTrace` envelopes
  1. ConsciousnessArche (Bilinç)       : Layer-32 GPR Phase Routing & Pondering ODE step allocation
  2. SimilarityArche (Benzerlik)       : Symbolic equivalence clustering (`NormalizedMathMatcher`)
  3. InferenceArche (Çıkarım)          : Zero-leakage Pass-1 structural trigger & G=4 trace parsing
  4. HarmonyArche (Uyum)               : Lens-Free Neurosymbolic PRM (`S_LensFree`, zero Tuned Lens noise)
  5. MethodArche (Yöntem)              : Two-Layer Execution Strategy (Inference Selector & GRPO Curriculum)
  6. ConditionsArche (Şartlar / Gate)  : Tier-1 Ternary Selective Abstention Gate & Quota-Guarded GRPO Reward
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from cst_symbolic_verifier import (
    CstSymbolicVerifier,
    NormalizedMathMatcher,
    LeanTacticStepVerifier,
    ProofVerificationReport,
)


# ============================================================================
# 0. DURUM ARKHESİ (StateArche) — Immutable State Monad & Trace Envelope
# ============================================================================

class DecisionVerdict(Enum):
    COMMIT_ANSWER = "COMMIT_ANSWER"
    ABSTAIN_UNKNOWN = "ABSTAIN_UNKNOWN"


class CognitiveMode(Enum):
    """
    Three-Regime Cognitive Routing Mode governed by Arkhe 1 (Bilinç):
      - SYSTEM1_DIRECT       : Low-latency conversational fast-path (0 RK4 steps, no <think>, no Lean overhead)
      - SYSTEM2_CAUSAL_LOGIC : Goal-precondition & causal reasoning (Arkhe-Şartlar Precondition Gate + Step Rollback)
      - SYSTEM2_MATH_FORMAL  : Symbolic math & formal derivation (GPR alpha + SoftLoRA + RK4 ODE + Lean Tactics)
    """
    SYSTEM1_DIRECT = "SYSTEM1_DIRECT"
    SYSTEM2_CAUSAL_LOGIC = "SYSTEM2_CAUSAL_LOGIC"
    SYSTEM2_MATH_FORMAL = "SYSTEM2_MATH_FORMAL"


@dataclass(frozen=True)
class CandidateTrace:
    """
    Deterministic, immutable representation of a single reasoning trajectory
    (either Pass-1 Greedy or one of the G=4 multi-branch rollouts).
    """
    branch_id: str
    raw_text: str = ""
    extracted_answer: str = ""
    has_end_tag: bool = True
    is_symbolically_valid: bool = True
    verified_eq_count: int = 0
    contradictions: Tuple[str, ...] = field(default_factory=tuple)
    doubt_count: int = 0
    generated_tokens: int = 0
    agree_count: int = 1
    lens_free_score: float = -100.0
    has_verbal_bluff: bool = False
    rollback_step_index: Optional[int] = None
    rollback_prefix_text: str = ""
    lean_violation_summary: str = ""
    mean_token_logprob: Optional[float] = None
    min_step_logprob: Optional[float] = None

    @classmethod
    def from_candidate_dict(cls, d: Dict) -> "CandidateTrace":
        """Constructs an immutable CandidateTrace from a legacy candidate dictionary."""
        raw_t = str(d.get("raw_text", ""))
        contras_list = [str(x) for x in d.get("contras", [])]
        has_bluff = bool(d.get("has_verbal_bluff", False))
        rb_idx = d.get("rollback_step_index", None)
        rb_prefix = str(d.get("rollback_prefix_text", ""))
        lean_sum = str(d.get("lean_violation_summary", ""))

        if raw_t and (not lean_sum and rb_idx is None):
            report = LeanTacticStepVerifier.verify_proof_steps(raw_t)
            if report.has_sorry_bluff:
                has_bluff = True
            if report.first_failed_step_index is not None:
                rb_idx = report.first_failed_step_index
                rb_prefix = report.verified_prefix_text
                lean_sum = report.violation_summary
            for c in report.contradictions:
                if c not in contras_list:
                    contras_list.append(c)

        is_val = bool(d.get("is_valid", True)) and (len(contras_list) == 0)
        return cls(
            branch_id=str(d.get("branch", "Cand")),
            raw_text=raw_t,
            extracted_answer=str(d.get("pred", "")),
            has_end_tag=bool(d.get("has_end", False)),
            is_symbolically_valid=is_val,
            verified_eq_count=int(d.get("eq_count", 0)),
            contradictions=tuple(contras_list),
            doubt_count=int(d.get("wait_cnt", 0)),
            generated_tokens=int(d.get("tokens", 0)),
            has_verbal_bluff=has_bluff,
            rollback_step_index=rb_idx,
            rollback_prefix_text=rb_prefix,
            lean_violation_summary=lean_sum,
            mean_token_logprob=d.get("mean_token_logprob", None),
            min_step_logprob=d.get("min_step_logprob", None),
        )


@dataclass(frozen=True)
class ArkheState:
    """
    Durum Arkhesi (StateArche):
    Single immutable state carrier flowing through all six Arkhe stages.
    Every stage consumes an `ArkheState` and returns a new `ArkheState` via `replace()`,
    guaranteeing zero hidden side-effects and full auditability.
    """
    question_id: str
    prompt: str = ""
    # Arkhe 1: Bilinç (Domain Routing & Manifold Allocation)
    domain_alpha: float = 0.0
    rk4_steps: int = 2
    # Arkhe 3: Çıkarım (Pass-1 & Multi-Branch Rollouts)
    pass1_trace: Optional[CandidateTrace] = None
    gate_triggered: bool = False
    candidate_pool: Tuple[CandidateTrace, ...] = field(default_factory=tuple)
    # Arkhe 2 & 4: Benzerlik & Uyum (Equivalence Clustering & Lens-Free Scoring)
    max_agreement: int = 1
    winning_trace: Optional[CandidateTrace] = None
    # Arkhe 5 & 6: Yöntem & Şartlar (Confidence Calibration & Ternary Decision Gate)
    epistemic_confidence: float = 0.0
    verdict: DecisionVerdict = DecisionVerdict.COMMIT_ANSWER
    final_output: str = ""


# ============================================================================
# 1. BİLİNÇ ARKHESİ (ConsciousnessArche) — Domain Routing & ODE Allocation
# ============================================================================

class ConsciousnessArche:
    """
    Arkhe 1 — Bilinç: "Ben ne yapmaya çalışıyorum?"
    Classifies incoming prompts into Three Cognitive Regimes (`SYSTEM1_DIRECT`,
    `SYSTEM2_CAUSAL_LOGIC`, `SYSTEM2_MATH_FORMAL`), computes continuous domain
    blending factor alpha in [0, 1], and allocates Prefill Pondering ODE RK4 steps.
    """

    SYSTEM1_GREETING_RE = re.compile(
        r"^\s*(?:"
        r"merhaba(?:lar)?|selam(?:lar|ün\s*aleyküm)?|slm|günaydın|iyi\s+(?:günler|akşamlar|geceler|çalışmalar)|"
        r"nasılsın(?:ız)?|naber|ne\s+haber|kimsin|sen\s+kimsin|adın\s+ne|kendini\s+tanıt(?:ır\s+mısın)?|"
        r"teşekkür(?:ler|\s+ederim)|sağ\s*ol(?:un)?|eyvallah|hoşça\s*kal|görüşürüz|bay\s*bay|"
        r"hello|hi|hey|good\s+(?:morning|afternoon|evening|night)|how\s+are\s+you|who\s+are\s+you|"
        r"what\s+is\s+your\s+name|thanks|thank\s+you|bye|goodbye"
        r")[\s\?\!\.\,]*$",
        re.I,
    )

    MATH_SIGNAL_RE = re.compile(
        r"(?:"
        r"\\frac|\\sqrt|\\boxed|\\sum|\\prod|\\int|\\lim|\\sin|\\cos|\\tan|\\log|\\ln|\\pi|"
        r"[0-9]+\s*[\+\-\*\/\^\=]\s*[0-9a-zA-Z]|"
        r"\b[a-zA-Z]\s*[\+\-\*\/\^]\s*[a-zA-Z0-9]\b|"
        r"\b[a-zA-Z]\s*=\s*[-+]?[0-9]+|"
        r"\b(?:denklem|polinom|fonksiyon|türev|integral|matris|determinant|olasılık|kombinasyon|permütasyon|"
        r"üçgen|çember|dikdörtgen|kare|hipotenüs|asal|bölünebilme|kalan|mod|logaritma|trigonometri|"
        r"limit|seri|dizi|kök|çarpan|ifade(?:sinin|yi)|hesapla(?:yınız|yın)?|kaçtır|kaç\s+eder|"
        r"equation|polynomial|function|derivative|integral|matrix|probability|triangle|circle|"
        r"prime|divisible|remainder|modulo|logarithm|solve|simplify|evaluate|prove\s+that|find\s+the\s+value)\b"
        r")",
        re.I,
    )

    CAUSAL_DILEMMA_RE = re.compile(
        r"(?:"
        r"\b(?:mı|mi|mu|mü)\s+yoksa\b|"
        r"\b(?:arabayla|yürüyerek|koşarak|uçakla|otobüsle)\b.*\b(?:mı|mi|mu|mü)\b|"
        r"\b(?:gitmeliyim|gitmeli|yapmalıyım|yapmalı|seçmeliyim|tercih\s+etmeli)\b|"
        r"\b(?:hangisi\s+daha\s+(?:mantıklı|doğru|iyi|uygun))\b|"
        r"\b(?:neden|niçin|nasıl\s+olur|mantıklı\s+mı|önkoşul|şart)\b|"
        r"\b(?:should\s+i|or\s+should|by\s+car\s+or\s+on\s+foot|walk\s+or\s+drive|which\s+is\s+more\s+logical)\b"
        r")",
        re.I,
    )

    EXPLANATION_CUES_RE = re.compile(
        r"\b(?:"
        r"açıkla(?:yınız|yın|r\s+mısın)?|detay(?:lı)?|anlat(?:ır\s+mısın)?|neden|niçin|nasıl|"
        r"farkı\s+ne|örnek(?:lendir)?|karşılaştır|öngör(?:ü)?|açıklar\s+mısın|açıklama|"
        r"explain|elaborate|why|how|clarify|describe|compare"
        r")\b",
        re.I,
    )

    def __init__(self, router=None, blender=None):
        self.router = router
        self.blender = blender

    @classmethod
    def classify_cognitive_mode(cls, prompt: str) -> CognitiveMode:
        """
        Determines the cognitive regime in < 0.2ms without any GPU forward pass:
          - SYSTEM1_DIRECT       : Greetings, identity questions, brief social chat
          - SYSTEM2_MATH_FORMAL  : Symbolic mathematics, equations, quantitative calculation
          - SYSTEM2_CAUSAL_LOGIC : Practical decisions, goal-precondition dilemmas, causal logic
        """
        clean = (prompt or "").strip()
        if not clean:
            return CognitiveMode.SYSTEM1_DIRECT

        if cls.SYSTEM1_GREETING_RE.match(clean):
            return CognitiveMode.SYSTEM1_DIRECT

        # Check explicit causal dilemma / goal-precondition puzzle FIRST before math (since "50 metre" has digits)
        if cls.CAUSAL_DILEMMA_RE.search(clean) and not ("=" in clean or "\\" in clean):
            return CognitiveMode.SYSTEM2_CAUSAL_LOGIC

        # If user explicitly asks for an explanation, analysis or detail, route to SYSTEM2_CAUSAL_LOGIC
        if cls.EXPLANATION_CUES_RE.search(clean) and not ("=" in clean or "\\" in clean):
            return CognitiveMode.SYSTEM2_CAUSAL_LOGIC

        if cls.MATH_SIGNAL_RE.search(clean):
            return CognitiveMode.SYSTEM2_MATH_FORMAL

        # Short conversational messages without numbers, math symbols, or analytical dilemma cues
        words = clean.split()
        has_digits = any(ch.isdigit() for ch in clean)
        has_math_ops = any(op in clean for op in ("=", "+", "*", "/", "^", "\\", "$"))
        if len(words) <= 6 and len(clean) <= 45 and not has_digits and not has_math_ops:
            if not cls.CAUSAL_DILEMMA_RE.search(clean) and not cls.EXPLANATION_CUES_RE.search(clean):
                return CognitiveMode.SYSTEM1_DIRECT

        return CognitiveMode.SYSTEM2_CAUSAL_LOGIC

    @staticmethod
    def allocate_rk4_steps(
        alpha: float,
        min_steps: int = 2,
        max_steps: int = 5,
        mode: Optional[CognitiveMode] = None,
    ) -> int:
        if mode == CognitiveMode.SYSTEM1_DIRECT:
            return 0
        if mode == CognitiveMode.SYSTEM2_CAUSAL_LOGIC:
            return min_steps
        alpha_clamped = max(0.0, min(1.0, float(alpha)))
        if alpha_clamped < 0.25:
            return min_steps
        return min(max_steps, min_steps + int(round((max_steps - min_steps) * alpha_clamped)))

    def __call__(self, state: ArkheState, alpha: Optional[float] = None, hidden_states=None) -> ArkheState:
        if alpha is None and self.router is not None and hidden_states is not None:
            alpha = float(self.router.compute_blending_alpha(hidden_states))
        resolved_alpha = float(alpha if alpha is not None else state.domain_alpha)

        if self.blender is not None and hasattr(self.blender, "set_scales"):
            self.blender.set_scales(
                scale_d1=0.25 * (1.0 - resolved_alpha),
                scale_d2=resolved_alpha,
            )

        steps = self.allocate_rk4_steps(resolved_alpha)
        return replace(state, domain_alpha=resolved_alpha, rk4_steps=steps)


# ============================================================================
# 2. BENZERLİK ARKHESİ (SimilarityArche) — Semantic Equivalence Clustering
# ============================================================================

class SimilarityArche:
    """
    Arkhe 2 — Benzerlik: "Bu çözümler birbirine semantik olarak benziyor mu?"
    Partitions candidate rollout answers into mathematical equivalence classes
    using `NormalizedMathMatcher` and assigns `agree_count` in {1..G}.
    """

    def __init__(self, matcher_fn: Optional[Callable[[str, str], bool]] = None):
        self._matcher_fn = matcher_fn

    def are_equivalent(self, ans_a: str, ans_b: str) -> bool:
        if ans_a == ans_b:
            return True
        if not ans_a or not ans_b:
            return False
        if self._matcher_fn is not None:
            res = self._matcher_fn(ans_a, ans_b)
            if isinstance(res, (tuple, list)):
                return bool(res[1]) if len(res) > 1 else bool(res[0])
            return bool(res)
        try:
            _, is_eq = NormalizedMathMatcher.check_match(str(ans_a), str(ans_b))
            return bool(is_eq)
        except Exception:
            return False

    def __call__(self, state: ArkheState) -> ArkheState:
        if not state.candidate_pool:
            p1_ok = (
                state.pass1_trace is not None
                and state.pass1_trace.has_end_tag
                and state.pass1_trace.is_symbolically_valid
            )
            return replace(state, max_agreement=(4 if p1_ok else 1))

        updated_pool: List[CandidateTrace] = []
        max_agree = 1
        for cand in state.candidate_pool:
            cnt = sum(
                1
                for other in state.candidate_pool
                if self.are_equivalent(cand.extracted_answer, other.extracted_answer)
            )
            max_agree = max(max_agree, cnt)
            updated_pool.append(replace(cand, agree_count=cnt))

        return replace(
            state,
            candidate_pool=tuple(updated_pool),
            max_agreement=max_agree,
        )


# ============================================================================
# 3. ÇIKARIM ARKHESİ (InferenceArche) — Structural Trigger & Trace Inspection
# ============================================================================

class InferenceArche:
    """
    Arkhe 3 — Çıkarım: "Buradan sonuca nasıl varırım?"
    Inspects Pass-1 and candidate rollouts using CST symbolic verification and
    evaluates the zero-leakage structural trigger rule.
    """

    def __init__(self, max_doubt_allowed: int = 1, logprob_trigger_threshold: Optional[float] = None):
        self.max_doubt_allowed = max_doubt_allowed
        self.logprob_trigger_threshold = logprob_trigger_threshold

    @staticmethod
    def count_doubt_markers(text: str) -> int:
        low = text.lower()
        return low.count("wait") + low.count("re-evaluate") + low.count("mistake")

    def inspect_raw_generation(
        self,
        branch_id: str,
        raw_text: str,
        extracted_answer: str,
        generated_tokens: int = 0,
        prompt: str = "",
    ) -> CandidateTrace:
        has_end = "</think>" in raw_text
        is_valid, eq_cnt, contras = CstSymbolicVerifier.verify_chunk(raw_text)
        step_report = LeanTacticStepVerifier.verify_proof_steps(raw_text, prompt=prompt)
        doubt_cnt = self.count_doubt_markers(raw_text)
        merged_contras = list(contras)
        for c in step_report.contradictions:
            if c not in merged_contras:
                merged_contras.append(c)

        return CandidateTrace(
            branch_id=branch_id,
            raw_text=raw_text,
            extracted_answer=extracted_answer,
            has_end_tag=has_end,
            is_symbolically_valid=bool(is_valid and len(merged_contras) == 0),
            verified_eq_count=int(max(eq_cnt, step_report.verified_eq_count)),
            contradictions=tuple(str(c) for c in merged_contras),
            doubt_count=int(doubt_cnt),
            generated_tokens=int(generated_tokens),
            has_verbal_bluff=bool(step_report.has_sorry_bluff),
            rollback_step_index=step_report.first_failed_step_index,
            rollback_prefix_text=step_report.verified_prefix_text,
            lean_violation_summary=step_report.violation_summary,
        )

    def should_trigger_multibranch(self, pass1_trace: CandidateTrace) -> bool:
        """
        Zero-leakage Pass-1 structural, Lean proof obligation, and optional logprob uncertainty trigger:
          1. Unfinished reasoning chain (`not has_end_tag`)
          2. Explicit SymPy CST / Lean tactic contradiction (`not is_symbolically_valid` or `len(contradictions) > 0`)
          3. High self-doubt loop (`doubt_count > max_doubt_allowed`)
          4. Lean `no_sorry` verbal bluff / post-hoc rationalization bridge (`has_verbal_bluff`)
          5. Any intermediate step obligation failure (`rollback_step_index is not None`)
          6. Optional low token-probability silent-error guard (`mean_token_logprob` or `min_step_logprob` below threshold)
        """
        low_conf_logprob = False
        if self.logprob_trigger_threshold is not None:
            if pass1_trace.min_step_logprob is not None and pass1_trace.min_step_logprob < self.logprob_trigger_threshold:
                low_conf_logprob = True
            elif pass1_trace.mean_token_logprob is not None and pass1_trace.mean_token_logprob < self.logprob_trigger_threshold:
                low_conf_logprob = True

        return (
            (not pass1_trace.has_end_tag)
            or (not pass1_trace.is_symbolically_valid)
            or (len(pass1_trace.contradictions) > 0)
            or (pass1_trace.doubt_count > self.max_doubt_allowed)
            or bool(pass1_trace.has_verbal_bluff)
            or (pass1_trace.rollback_step_index is not None)
            or low_conf_logprob
        )

    @staticmethod
    def build_step_rollback_branch_specs(
        problem_text: str,
        pass1_trace: CandidateTrace,
    ) -> List[Dict[str, object]]:
        """
        Constructs the G=4 Arkhe Step-Rollback & Epistemic Self-Awareness branch specs.
        Prefix rollback is ONLY used when `pass1_trace.rollback_step_index is not None`
        (an explicit Lean tactic obligation failure at step s_k). On silent-failure /
        logprob-only triggers (`rollback_step_index is None`), inheriting Pass-1's full
        chain causes 10-token fake-consensus echoing, so all branches solve fresh from scratch.
        """
        has_step_rollback = pass1_trace.rollback_step_index is not None
        verified_prefix = pass1_trace.rollback_prefix_text.strip() if has_step_rollback else ""
        # Strip any premature </think> from the prefix so reasoning continues inside <think>
        if "</think>" in verified_prefix:
            verified_prefix = verified_prefix.split("</think>")[0].strip()

        violation_note = (
            pass1_trace.lean_violation_summary
            or ("; ".join(pass1_trace.contradictions[:2]) if pass1_trace.contradictions else "")
            or "Unproved step jump or unfinished chain"
        )
        has_asy = "[asy]" in problem_text.lower()
        use_rb = has_step_rollback and bool(verified_prefix)
        tactic_regime = MethodArche.classify_math_tactic_regime(problem_text)

        regime_hint = ""
        if tactic_regime == "TRIG_EXACT":
            regime_hint = (
                " TRIGONOMETRY CHECK: Verify quadrant signs carefully (for 90° < θ < 180°, sin θ > 0 while cos θ < 0 and tan θ < 0) "
                "and check that the numerical sign and magnitude of your final answer match the target expression."
            )
        elif tactic_regime == "GEOMETRY_SPATIAL":
            regime_hint = (
                " GEOMETRY CHECK: Compare all distinct 2D net unfoldings or coordinate orientations before committing; "
                "verify every triangle leg length explicitly."
            )

        diagnostic_feedback = ""
        if (not use_rb) and (pass1_trace.lean_violation_summary or pass1_trace.contradictions):
            note = pass1_trace.lean_violation_summary or "; ".join(pass1_trace.contradictions[:2])
            diagnostic_feedback = f" Prior Pass-1 failed verification ({note}); avoid repeating that error."

        cand1_directive = (
            "You are operating under the SODALS Lean-Tactic Proof Obligation Verifier. "
            f"At the next step, a formal proof obligation failed ({violation_note}). "
            "Do NOT use verbal shortcuts ('using the known result', 'can be derived via', etc.). "
            "Derive the next step explicitly using exact algebraic equations from the verified premises above, "
            "then state the final answer inside \\boxed{...}."
            if use_rb
            else "You are operating under the SODALS Lean-Tactic Proof Obligation Verifier. "
            "Re-solve this problem completely from scratch with explicit algebraic equations at every step."
            + diagnostic_feedback
            + regime_hint
            + " Check signs and boundary conditions before stating the final answer inside \\boxed{...}."
        )
        cand2_directive = (
            "Continue from the verified step above with strict step-by-step algebraic equations. "
            "Verify every simplification and back-substitute any solved variable values before concluding in \\boxed{...}."
            if use_rb
            else "Solve step-by-step from scratch using an alternative algebraic or geometric setup."
            + diagnostic_feedback
            + regime_hint
            + " Verify every simplification and back-substitute your answer before concluding in \\boxed{...}."
        )

        return [
            {
                "branch": "Cand1",
                "temp": 0.0 if use_rb else 0.25,
                "role": "Arkhe_Lean_Step_Rollback_Repair" if use_rb else "Arkhe_Fresh_Strict_Derivation",
                "use_prefix_rollback": use_rb,
                "verified_prefix": verified_prefix if use_rb else "",
                "system_directive": cand1_directive,
            },
            {
                "branch": "Cand2",
                "temp": 0.5,
                "role": "Arkhe_Manifold_Steered_Derivation_A",
                "use_prefix_rollback": use_rb,
                "verified_prefix": verified_prefix if use_rb else "",
                "system_directive": cand2_directive,
            },
            {
                "branch": "Cand3",
                "temp": 0.6,
                "role": "Arkhe_Independent_First_Principles_B",
                "use_prefix_rollback": False,
                "verified_prefix": "",
                "system_directive": (
                    "Solve this problem from first principles with explicit algebraic equations at every step. "
                    "Never guess or jump to a memorized answer without proving each intermediate equality. Put your final answer in \\boxed{...}."
                ),
            },
            {
                "branch": "Cand4",
                "temp": 0.3,
                "role": "Arkhe_Epistemic_Self_Audit",
                "use_prefix_rollback": use_rb and not has_asy,
                "verified_prefix": (verified_prefix if (use_rb and not has_asy) else ""),
                "system_directive": (
                    "You are the Arkhe Epistemic Auditor. Derive the solution strictly by algebraic deduction from scratch. "
                    "CRITICAL HONESTY RULE: If the problem depends on a visual diagram ([asy]) whose labels/lengths cannot be uniquely "
                    "determined from the text, or if an algebraic equation cannot be solved without guessing/bluffing, "
                    "do NOT guess a number — explicitly output \\boxed{[ABSTAIN]}."
                ),
            },
        ]


# ============================================================================
# 4. UYUM ARKHESİ (HarmonyArche) — Lens-Free Neurosymbolic PRM Scoring
# ============================================================================

class HarmonyArche:
    """
    Arkhe 4 — Uyum: "Semantik uzlaşı ile sembolik cebir birbiriyle uyumlu mu?"
    Computes the Lens-Free Neurosymbolic PRM score:
      S_LensFree(c) = 10 * has_end + 8 * agree_count + 1.5 * min(eq_count, 10)
                      - 8 * |contras| - 2 * doubt_count - 12 * has_verbal_bluff
    Completely excludes Tuned Lens inter-layer stability (greedy AUROC = 0.3721 < 0.50).
    """

    EQ_CAP: int = 10
    W_END: float = 10.0
    W_AGREE: float = 8.0
    W_EQ: float = 1.5
    W_CONTRA: float = 8.0
    W_DOUBT: float = 2.0
    W_BLUFF: float = 12.0

    @classmethod
    def compute_score(cls, trace: CandidateTrace) -> float:
        if not trace.has_end_tag or not trace.extracted_answer.strip():
            return -100.0
        if trace.extracted_answer.strip() == ConditionsArche.ABSTAIN_TOKEN:
            # Organic abstention trace is scored separately by ConditionsArche
            return -50.0
        eq_capped = min(max(0, int(trace.verified_eq_count)), cls.EQ_CAP)
        n_contras = len(trace.contradictions)
        has_goal_mismatch = any("LeanGoalMismatchViolation" in c for c in trace.contradictions)
        score = (
            cls.W_END * (1.0 if trace.has_end_tag else 0.0)
            + cls.W_AGREE * int(trace.agree_count)
            + cls.W_EQ * eq_capped
            - cls.W_CONTRA * n_contras
            - (24.0 if has_goal_mismatch else 0.0)
            - cls.W_DOUBT * int(trace.doubt_count)
            - cls.W_BLUFF * (1.0 if trace.has_verbal_bluff else 0.0)
        )
        return round(float(score), 4)

    def __call__(self, state: ArkheState) -> ArkheState:
        if not state.gate_triggered or not state.candidate_pool:
            p1 = state.pass1_trace
            if p1 is not None:
                p1_scored = replace(
                    p1,
                    agree_count=4 if (p1.has_end_tag and p1.is_symbolically_valid) else 1,
                    lens_free_score=35.0 if (p1.has_end_tag and p1.is_symbolically_valid) else -10.0,
                )
                return replace(state, pass1_trace=p1_scored, winning_trace=p1_scored)
            return state

        scored_pool = tuple(
            replace(c, lens_free_score=self.compute_score(c))
            for c in state.candidate_pool
        )
        winner = max(scored_pool, key=lambda c: c.lens_free_score)
        return replace(
            state,
            candidate_pool=scored_pool,
            winning_trace=winner,
        )


# ============================================================================
# 5. YÖNTEM ARKHESİ (MethodArche) — Curriculum & Active Prompt Filtering
# ============================================================================

class MethodArche:
    """
    Arkhe 5 — Yöntem: "Hangi sırayla, nasıl?"
    Manages BOTH:
      1. Inference-time problem tactic classification (`classify_math_tactic_regime`) so Arkhe-3
         and Arkhe-4 adapt verification & branch prompts to trig/geometry/algebra structure.
      2. Training curriculum schedule (dynamic penalty lambda(t) and active variance filtering).
    """

    @staticmethod
    def classify_math_tactic_regime(problem_text: str) -> str:
        low = (problem_text or "").lower()
        if any(k in low for k in (r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc", r"^\circ", "degrees", "radian")):
            if any(k in low for k in ("simplify", "evaluate", "compute", "find the value", r"\sin", r"\tan", r"\cos")):
                return "TRIG_EXACT"
        if any(k in low for k in ("[asy]", "triangle", "prism", "wall", "ceiling", "floor", "north", "south", "circle", "square", "polygon", "heptagon", "radius", "gecko")):
            return "GEOMETRY_SPATIAL"
        if any(k in low for k in ("solve", "roots", "polynomial", "equation", "group", "ring", "matrix", "eigenvalue")):
            return "ALGEBRA_SYMBOLIC"
        return "GENERAL_DEDUCTION"

    @staticmethod
    def compute_dynamic_lambda(
        step: int,
        total_steps: int,
        lambda_start: float = 0.25,
        lambda_end: float = 1.00,
        warmup_fraction: float = 0.35,
    ) -> float:
        """
        Two-stage curriculum for wrong-answer penalty lambda(t):
        - First `warmup_fraction` (default 35%) of steps: hold lambda = lambda_start (0.25).
        - Remaining steps: linearly ramp lambda from lambda_start to lambda_end (1.00).
        """
        if total_steps <= 0:
            return lambda_end
        progress = max(0.0, min(1.0, float(step) / float(total_steps)))
        if progress <= warmup_fraction:
            return round(float(lambda_start), 4)
        ramp = (progress - warmup_fraction) / max(1e-6, (1.0 - warmup_fraction))
        val = lambda_start + ramp * (lambda_end - lambda_start)
        return round(float(val), 4)

    @staticmethod
    def has_active_group_variance(rewards: Sequence[float], eps: float = 1e-5) -> bool:
        """Returns True iff the group of rollout rewards has non-zero variance (sigma_r > eps)."""
        if len(rewards) < 2:
            return False
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        return (var_r ** 0.5) > eps


# ============================================================================
# 6. ŞARTLAR ARKHESİ (ConditionsArche / God's Gate)
#    Layer 1: Tier-1 Selective Abstention Gate
#    Layer 2: Quota-Guarded Ternary GRPO Reward Calculator
# ============================================================================

class ConditionsArche:
    """
    Arkhe 6 — Şartlar (God's Gate): "Matematiksel gerçeklik ne söylüyor?"
    Enforces:
      - Layer 1 (Inference): Tier-1 Selective Abstention Gate (`tau = 0.20`)
      - Layer 2 (Training) : Asymmetric Group Mask + 1/G Exploration Quota + Capped CST Shaping
    """

    ABSTAIN_TOKEN: str = "[ABSTAIN]"

    def __init__(self, selective_tau: float = 0.20):
        self.selective_tau = float(selective_tau)

    @staticmethod
    def calibrate_confidence(state: ArkheState) -> float:
        """
        Computes the neurosymbolic epistemic confidence score C_ns in [0, 1]:
          - Non-triggered Pass-1:
              * Abandoned / token-exhausted (>= 2040 tokens or missing </think>): 0.10
              * Symbolically invalid / contradiction flagged                   : 0.40
              * Clean completion                                               : 0.90
          - Gate-triggered G=4 pool:
              * 4/4 unanimous agreement & 0 contradictions                     : 0.95
              * 3/4 majority agreement & clean completion                      : 0.80
              * 2/4 plurality agreement & high Lens-Free score (>= 26.0)       : 0.60
              * 2/4 plurality agreement & low Lens-Free score (< 26.0)         : 0.35
              * 1/4 complete divergence (all 4 candidates disagree)            : 0.15
        """
        if not state.gate_triggered or not state.candidate_pool:
            p1 = state.pass1_trace
            if p1 is None:
                return 0.10
            if (not p1.has_end_tag) or (p1.generated_tokens >= 2040):
                return 0.10
            if (not p1.is_symbolically_valid) or (len(p1.contradictions) > 0):
                return 0.40
            return 0.90

        winner = state.winning_trace
        if winner is None:
            return 0.15
        if not winner.has_end_tag:
            return 0.10

        max_ag = state.max_agreement
        n_contras = len(winner.contradictions)
        has_goal_mismatch = any("LeanGoalMismatchViolation" in c for c in winner.contradictions)
        if has_goal_mismatch or (n_contras > 0 and winner.verified_eq_count == 0):
            return 0.15
        if max_ag >= 4 and n_contras == 0:
            return 0.95
        if max_ag == 3 and winner.has_end_tag and n_contras == 0:
            return 0.80
        if max_ag == 2 and winner.lens_free_score >= 26.0:
            return 0.60
        if max_ag == 2:
            return 0.35
        return 0.15

    def __call__(self, state: ArkheState) -> ArkheState:
        """Applies Layer-1 Selective Abstention Gate to produce the final verdict."""
        conf = self.calibrate_confidence(state)
        winner = state.winning_trace if state.winning_trace is not None else state.pass1_trace

        if winner is None or conf < self.selective_tau:
            return replace(
                state,
                epistemic_confidence=conf,
                verdict=DecisionVerdict.ABSTAIN_UNKNOWN,
                final_output=self.ABSTAIN_TOKEN,
            )

        return replace(
            state,
            epistemic_confidence=conf,
            verdict=DecisionVerdict.COMMIT_ANSWER,
            final_output=winner.extracted_answer,
        )

    @classmethod
    def is_strict_abstain(cls, extracted_answer: str) -> bool:
        """Returns True iff extracted_answer is strictly `[ABSTAIN]` without multi-answer hedges."""
        ans_str = (extracted_answer or "").strip()
        if ans_str == cls.ABSTAIN_TOKEN:
            return True
        # Reject hedges like '1 or 3' or '[ABSTAIN] or 4'
        return False

    @classmethod
    def compute_clean_ternary_grpo_rewards(
        cls,
        traces: Sequence[CandidateTrace],
        is_correct_flags: Sequence[bool],
        lambda_penalty: float = 1.0,
    ) -> List[float]:
        """
        Stationary Ternary GRPO Reward Function (Quota-Free, Pure Dynamic-Lambda):
          1. Sound Correct Proof (no CST/Lean violations): +1.0 + bounded CST bonus (max +0.25).
          2. Strict Single-Token Abstention (`[ABSTAIN]` only, no hedges): 0.0.
          3. Wrong, Hedged (e.g. '1 or 3', '[ABSTAIN] or 4'), or Violated Proof: -lambda(t) + bounded shaping.
        Abstention collapse is prevented by dynamic lambda(t) (0.25 -> 1.00) and dev >=60% coverage gating,
        rather than non-stationary group quotas.
        """
        rewards: List[float] = []
        for trace, is_corr in zip(traces, is_correct_flags):
            ans_str = (trace.extracted_answer or "").strip()
            is_abstain = cls.is_strict_abstain(ans_str)

            # Bounded CST equation shaping in [-0.25, +0.25]
            eq_capped = min(max(0, int(trace.verified_eq_count)), HarmonyArche.EQ_CAP)
            cst_bonus = min(0.025 * eq_capped, 0.25)
            cst_penalty = min(0.10 * len(trace.contradictions), 0.30)
            shaping = max(-0.25, min(0.25, cst_bonus - cst_penalty))

            if is_corr:
                # Zero-tolerance for rationalized/bluffed steps: invalid proof never gets positive reward.
                if len(trace.contradictions) > 0 or trace.has_verbal_bluff or not trace.is_symbolically_valid:
                    r = -0.50 * float(lambda_penalty)
                else:
                    r = 1.0 + max(0.0, shaping)
            elif is_abstain:
                r = 0.0
            else:
                bluff_extra_penalty = 0.20 if trace.has_verbal_bluff else 0.0
                r = -float(lambda_penalty) + shaping - bluff_extra_penalty

            rewards.append(round(float(r), 4))
        return rewards

    @classmethod
    def compute_quota_guarded_grpo_rewards(
        cls,
        traces: Sequence[CandidateTrace],
        is_correct_flags: Sequence[bool],
        lambda_penalty: float = 1.0,
        max_abstain_quota: int = 1,
    ) -> List[float]:
        """Deprecated alias delegating to quota-free `compute_clean_ternary_grpo_rewards`."""
        _ = max_abstain_quota
        return cls.compute_clean_ternary_grpo_rewards(
            traces=traces,
            is_correct_flags=is_correct_flags,
            lambda_penalty=lambda_penalty,
        )


# ============================================================================
# UNIFIED ORCHESTRATOR: ArkheNeurosymbolicPipeline
# ============================================================================

class ArkheNeurosymbolicPipeline:
    """
    End-to-end deterministic pipeline chaining the Six Arkhes:
      ConsciousnessArche -> InferenceArche -> SimilarityArche -> HarmonyArche -> ConditionsArche
    """

    def __init__(
        self,
        selective_tau: float = 0.20,
        router=None,
        blender=None,
        matcher_fn: Optional[Callable[[str, str], bool]] = None,
        logprob_trigger_threshold: Optional[float] = None,
    ):
        self.consciousness = ConsciousnessArche(router=router, blender=blender)
        self.inference = InferenceArche(logprob_trigger_threshold=logprob_trigger_threshold)
        self.similarity = SimilarityArche(matcher_fn=matcher_fn)
        self.harmony = HarmonyArche()
        self.method = MethodArche()
        self.conditions = ConditionsArche(selective_tau=selective_tau)

    def run_from_traces(
        self,
        question_id: str,
        pass1_trace: CandidateTrace,
        candidate_traces: Sequence[CandidateTrace] = (),
        gate_triggered: Optional[bool] = None,
        domain_alpha: float = 0.0,
        prompt: str = "",
    ) -> ArkheState:
        """
        Executes the full functional Arkhe pipeline on pre-parsed or live traces,
        returning the final immutable `ArkheState`.
        """
        trig = (
            bool(gate_triggered)
            if gate_triggered is not None
            else self.inference.should_trigger_multibranch(pass1_trace)
        )

        state = ArkheState(
            question_id=str(question_id),
            prompt=prompt,
            pass1_trace=pass1_trace,
            gate_triggered=trig,
            candidate_pool=tuple(candidate_traces) if trig else tuple(),
        )

        state = self.consciousness(state, alpha=domain_alpha)
        state = self.similarity(state)
        state = self.harmony(state)
        state = self.conditions(state)
        return state
