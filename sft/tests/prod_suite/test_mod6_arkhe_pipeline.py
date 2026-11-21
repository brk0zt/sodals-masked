"""
Production Unit Test Suite — Module 6: Six-Arkhe Neurosymbolic Pipeline
=======================================================================
Tests:
  1. `ArkheState` & `CandidateTrace` immutability (`FrozenInstanceError` on mutation)
  2. `SimilarityArche` & `HarmonyArche` parity-breaking resolution on Q120, Q124, Q138
  3. `ConditionsArche` Tier-1 Selective Abstention Gate on 1/4 divergent pools (Q027)
  4. `ConditionsArche` Layer-2 Quota-Guarded Ternary GRPO Rewards (Asymmetric Mask + 1/G Quota + +0.25 Eq Cap)
  5. `MethodArche` Two-Stage Dynamic Lambda Schedule & Active Group Variance Filter
"""

import os
import sys
import unittest
from dataclasses import FrozenInstanceError

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from arkhe_neurosymbolic_pipeline import (
    ArkheNeurosymbolicPipeline,
    ArkheState,
    CandidateTrace,
    CognitiveMode,
    ConsciousnessArche,
    ConditionsArche,
    DecisionVerdict,
    HarmonyArche,
    InferenceArche,
    MethodArche,
    SimilarityArche,
)
from cst_symbolic_verifier import NormalizedMathMatcher, SymbolicStepChecker


class TestMod6ArkhePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = ArkheNeurosymbolicPipeline(selective_tau=0.20)

    def test_state_and_trace_immutability(self):
        """Verify StateArche (`ArkheState` & `CandidateTrace`) strictly rejects in-place mutation."""
        trace = CandidateTrace(branch_id="Cand1", extracted_answer="42")
        state = ArkheState(question_id="Q001", pass1_trace=trace)

        with self.assertRaises(FrozenInstanceError):
            trace.extracted_answer = "99"

        with self.assertRaises(FrozenInstanceError):
            state.max_agreement = 4

    def test_parity_breaking_q124_and_q138(self):
        """
        Verify that SimilarityArche + HarmonyArche simultaneously rescues:
          - Q124 (GT=8): Majority plurality (2/4 on '8' with 8 eqs) beats '12' (1/4 with 0 contras)
          - Q138 (GT=204_5): Clean symbolic branch ('204_5', 6 eqs, 0 contras) beats wrong plurality ('224_5', 2 contras)
        """
        # 1. Q124 candidate pool from Clean-125
        p1_q124 = CandidateTrace(branch_id="Pass1", extracted_answer="10", has_end_tag=False)
        cands_q124 = [
            CandidateTrace(branch_id="Cand1", extracted_answer="12", has_end_tag=True, verified_eq_count=5, contradictions=()),
            CandidateTrace(branch_id="Cand2", extracted_answer="8", has_end_tag=True, verified_eq_count=8, contradictions=("16 != 8",), doubt_count=2),
            CandidateTrace(branch_id="Cand3", extracted_answer="8", has_end_tag=True, verified_eq_count=6, contradictions=("6 != 3",), doubt_count=1),
            CandidateTrace(branch_id="Cand4", extracted_answer="7", has_end_tag=True, verified_eq_count=7, contradictions=("4.8 != 2.4",)),
        ]
        state_q124 = self.pipeline.run_from_traces("Q124", p1_q124, cands_q124, gate_triggered=True)
        self.assertEqual(state_q124.verdict, DecisionVerdict.COMMIT_ANSWER)
        self.assertEqual(state_q124.final_output, "8")
        self.assertEqual(state_q124.max_agreement, 2)

        # 2. Q138 candidate pool from Clean-125
        p1_q138 = CandidateTrace(branch_id="Pass1", extracted_answer="224_5", has_end_tag=False)
        cands_q138 = [
            CandidateTrace(branch_id="Cand1", extracted_answer="244_5", has_end_tag=True, verified_eq_count=4, contradictions=("54 != 74",)),
            CandidateTrace(branch_id="Cand2", extracted_answer="204_5", has_end_tag=True, verified_eq_count=6, contradictions=()),
            CandidateTrace(branch_id="Cand3", extracted_answer="224_5", has_end_tag=True, verified_eq_count=4, contradictions=("54 != 74", "74 != 64")),
            CandidateTrace(branch_id="Cand4", extracted_answer="224_5", has_end_tag=True, verified_eq_count=4, contradictions=("54 != 74", "74 != 64")),
        ]
        state_q138 = self.pipeline.run_from_traces("Q138", p1_q138, cands_q138, gate_triggered=True)
        self.assertEqual(state_q138.verdict, DecisionVerdict.COMMIT_ANSWER)
        self.assertEqual(state_q138.final_output, "204_5")

    def test_tier1_selective_abstention_on_divergent_pool(self):
        """
        Verify ConditionsArche abstains (`[ABSTAIN]`) when all 4 candidates diverge (1/4 agreement, e.g. Q027).
        """
        p1_q027 = CandidateTrace(branch_id="Pass1", extracted_answer="2", has_end_tag=False)
        cands_q027 = [
            CandidateTrace(branch_id="Cand1", extracted_answer="1008", has_end_tag=True, verified_eq_count=7, doubt_count=1),
            CandidateTrace(branch_id="Cand2", extracted_answer="432", has_end_tag=True, verified_eq_count=8),
            CandidateTrace(branch_id="Cand3", extracted_answer="216", has_end_tag=True, verified_eq_count=4),
            CandidateTrace(branch_id="Cand4", extracted_answer="504", has_end_tag=True, verified_eq_count=10),
        ]
        state_q027 = self.pipeline.run_from_traces("Q027", p1_q027, cands_q027, gate_triggered=True)
        self.assertEqual(state_q027.max_agreement, 1)
        self.assertAlmostEqual(state_q027.epistemic_confidence, 0.15)
        self.assertEqual(state_q027.verdict, DecisionVerdict.ABSTAIN_UNKNOWN)
        self.assertEqual(state_q027.final_output, "[ABSTAIN]")

    def test_clean_ternary_grpo_rewards(self):
        """
        Verify Layer-2 Quota-Free Stationary Ternary GRPO Rewards:
          - Strict single `[ABSTAIN]` gets 0.0 (no non-stationary group quota penalty).
          - Hedged answers like '1 or 3' or '[ABSTAIN] or 4' are penalized (-lambda).
          - Correct & valid proof gets +1.0 + bounded CST bonus; wrong eq-spammer capped at -lambda + 0.25.
          - Optional logprob silent-error trigger in InferenceArche fires below threshold.
        """
        traces = [
            CandidateTrace(branch_id="R1", extracted_answer="[ABSTAIN]", verified_eq_count=0),
            CandidateTrace(branch_id="R2", extracted_answer="1 or 3", verified_eq_count=0),  # hedge
            CandidateTrace(branch_id="R3", extracted_answer="999", verified_eq_count=50),    # eq spammer
            CandidateTrace(branch_id="R4", extracted_answer="42", verified_eq_count=4),
        ]

        rewards_solvable = ConditionsArche.compute_clean_ternary_grpo_rewards(
            traces, is_correct_flags=[False, False, False, True], lambda_penalty=1.0
        )
        self.assertEqual(rewards_solvable[0], 0.0)
        self.assertEqual(rewards_solvable[1], -1.0)
        self.assertEqual(rewards_solvable[2], -0.75)
        self.assertEqual(rewards_solvable[3], 1.10)

        rewards_unsolvable = ConditionsArche.compute_clean_ternary_grpo_rewards(
            traces, is_correct_flags=[False, False, False, False], lambda_penalty=0.5
        )
        self.assertEqual(rewards_unsolvable[0], 0.0)
        self.assertEqual(rewards_unsolvable[1], -0.5)
        self.assertEqual(rewards_unsolvable[2], -0.25)
        self.assertTrue(MethodArche.has_active_group_variance(rewards_unsolvable))

        # Logprob trigger test for silent failures
        inf_gate = InferenceArche(logprob_trigger_threshold=-0.50)
        silent_wrong = CandidateTrace(
            branch_id="Pass1", extracted_answer="48", has_end_tag=True, is_symbolically_valid=True,
            min_step_logprob=-0.75,
        )
        confident_clean = CandidateTrace(
            branch_id="Pass1", extracted_answer="21", has_end_tag=True, is_symbolically_valid=True,
            min_step_logprob=-0.12,
        )
        self.assertTrue(inf_gate.should_trigger_multibranch(silent_wrong))
        self.assertFalse(inf_gate.should_trigger_multibranch(confident_clean))

    def test_method_arche_dynamic_lambda_schedule(self):
        """Verify two-stage dynamic penalty curriculum lambda(t) with 35% warmup."""
        self.assertAlmostEqual(MethodArche.compute_dynamic_lambda(10, 100), 0.25)
        self.assertAlmostEqual(MethodArche.compute_dynamic_lambda(35, 100), 0.25)
        self.assertAlmostEqual(MethodArche.compute_dynamic_lambda(67, 100), 0.6192, places=3)
        self.assertAlmostEqual(MethodArche.compute_dynamic_lambda(100, 100), 1.00)

    def test_cognitive_mode_routing_fastpath_and_causal_logic(self):
        """
        Verify Arkhe 1 (Bilinç) 3-Regime Cognitive Router:
          - Simple greetings ('merhaba', 'nasılsın') -> SYSTEM1_DIRECT (0 RK4 steps)
          - Practical dilemmas ('Arabamı yıkatmak istiyorum... 50 metre... arabayla mı yürüyerek mi') -> SYSTEM2_CAUSAL_LOGIC
          - Symbolic math ('Solve x^2 - 5x + 6 = 0') -> SYSTEM2_MATH_FORMAL
        """
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("merhaba"), CognitiveMode.SYSTEM1_DIRECT)
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("Selam, nasılsın?"), CognitiveMode.SYSTEM1_DIRECT)
        self.assertEqual(
            ConsciousnessArche.allocate_rk4_steps(0.20, mode=CognitiveMode.SYSTEM1_DIRECT),
            0,
        )

        car_wash_prompt = (
            "Arabamı yıkatmak istiyorum. Araba yıkama yeri 50 metre uzaklıkta. "
            "Oraya arabayla mı yoksa yürüyerek mi gitmeliyim?"
        )
        self.assertEqual(
            ConsciousnessArche.classify_cognitive_mode(car_wash_prompt),
            CognitiveMode.SYSTEM2_CAUSAL_LOGIC,
        )
        self.assertEqual(
            ConsciousnessArche.classify_cognitive_mode("Find all roots of x^2 - 5x + 6 = 0."),
            CognitiveMode.SYSTEM2_MATH_FORMAL,
        )

    def test_causal_precondition_verifier_and_honest_proof_state(self):
        """
        Verify `LeanTacticStepVerifier` (`tactic_precondition_logic`):
          1. Catches the flawed '50m -> walk' Car Wash conclusion as `LeanPreconditionViolation` and provides rollback offset.
          2. Verifies the sound 'car must be at the car wash -> go by car' conclusion as `CAUSALLY SOUND`.
          3. Never falsely claims `FORMALLY SOUND (Lean Verified)` on natural-language greetings with 0 equations.
        """
        car_wash_prompt = (
            "Arabamı yıkatmak istiyorum. Araba yıkama yeri 50 metre uzaklıkta. "
            "Oraya arabayla mı yoksa yürüyerek mi gitmeliyim?"
        )
        flawed_trace = (
            "Step 1: Consider the distance (50 meters).\n"
            "Walking 50 meters takes less than 1 minute.\n"
            "Therefore, the answer should be: Yürüyerek gitmelisiniz.\n"
        )
        rep_flawed = SymbolicStepChecker.verify_proof_steps(flawed_trace, prompt=car_wash_prompt)
        self.assertFalse(rep_flawed.is_formally_sound)
        self.assertEqual(rep_flawed.violation_type, "LeanPreconditionViolation")
        self.assertEqual(rep_flawed.first_failed_step_index, 3)
        self.assertIn("Step 1: Consider the distance", rep_flawed.verified_prefix_text)
        self.assertNotIn("Yürüyerek gitmelisiniz", rep_flawed.verified_prefix_text)

        sound_trace = (
            "Adım 1: Birincil hedef arabamı yıkatmak.\n"
            "Adım 2: Yıkanacak olan arabanın yıkama yerinde olması zorunlu önkoşuldur.\n"
            "Sonuç: Bu yüzden 50 metre olsa bile arabayla gitmelisiniz.\n"
        )
        rep_sound = SymbolicStepChecker.verify_proof_steps(sound_trace, prompt=car_wash_prompt)
        self.assertTrue(rep_sound.is_formally_sound)
        self.assertGreaterEqual(rep_sound.causal_preconditions_verified, 1)
        self.assertIn("CAUSALLY SOUND", rep_sound.proof_status_label)

        greeting_rep = SymbolicStepChecker.verify_proof_steps("Merhaba! Size nasıl yardımcı olabilirim?")
        self.assertEqual(greeting_rep.verified_eq_count, 0)
        self.assertEqual(greeting_rep.proof_status_label, "NATURAL LANGUAGE (No Formal Equations Invoked)")

    def test_step_rollback_prefix_requires_explicit_step_index(self):
        """
        Verify `build_step_rollback_branch_specs` only injects `verified_prefix` when
        `rollback_step_index is not None`; on silent/logprob triggers (`rollback_step_index=None`),
        all G=4 branches must generate fresh without inheriting Pass-1's wrong prefix.
        """
        silent_fail_trace = CandidateTrace(
            branch_id="Pass1_Greedy",
            extracted_answer="45",
            has_end_tag=True,
            is_symbolically_valid=True,
            rollback_step_index=None,
            rollback_prefix_text="Wrong geometric assumption leading to 45.",
            min_step_logprob=-0.39,
        )
        specs_silent = InferenceArche.build_step_rollback_branch_specs("Find angle x.", silent_fail_trace)
        self.assertEqual(len(specs_silent), 4)
        for sp in specs_silent:
            self.assertFalse(sp["use_prefix_rollback"], f"{sp['branch']} leaked Pass-1 prefix on silent fail")
            self.assertEqual(sp["verified_prefix"], "")
        self.assertGreater(float(specs_silent[0]["temp"]), 0.0)

        step_fail_trace = CandidateTrace(
            branch_id="Pass1_Greedy",
            extracted_answer="45",
            has_end_tag=True,
            is_symbolically_valid=False,
            rollback_step_index=2,
            rollback_prefix_text="Step 1: valid setup.",
        )
        specs_rb = InferenceArche.build_step_rollback_branch_specs("Find angle x.", step_fail_trace)
        self.assertTrue(specs_rb[0]["use_prefix_rollback"])
        self.assertEqual(specs_rb[0]["verified_prefix"], "Step 1: valid setup.")
        self.assertEqual(specs_rb[0]["temp"], 0.0)
        self.assertFalse(specs_rb[2]["use_prefix_rollback"])

    def test_active_arkhes_tactic_regime_and_goal_mismatch_abstention(self):
        """
        Verify MethodArche, InferenceArche, HarmonyArche, and ConditionsArche collaborate cleanly:
          - Trig Simplify prompt routes to TRIG_EXACT and injects sign check + violation feedback.
          - Unanimous wrong answers carrying LeanGoalMismatchViolation abstain (Conf=0.15) instead of 0.95.
          - One consistent derivation (0 contras) outscores contradicted clones in HarmonyArche.
        """
        q165_prob = r"Simplify $\tan 100^\circ + 4 \sin 100^\circ.$"
        self.assertEqual(MethodArche.classify_math_tactic_regime(q165_prob), "TRIG_EXACT")
        self.assertEqual(
            MethodArche.classify_math_tactic_regime("A gecko is in a room 12 feet long..."),
            "GEOMETRY_SPATIAL",
        )

        p1_trig = CandidateTrace(
            branch_id="Pass1_Greedy",
            extracted_answer=r"\sqrt{3}",
            has_end_tag=True,
            is_symbolically_valid=False,
            contradictions=("LeanGoalMismatchViolation: +1.732 != -1.732",),
            lean_violation_summary="[LeanGoalMismatchViolation] +1.732 != -1.732",
        )
        specs = InferenceArche.build_step_rollback_branch_specs(q165_prob, p1_trig)
        self.assertIn("TRIGONOMETRY CHECK", str(specs[0]["system_directive"]))
        self.assertIn("LeanGoalMismatchViolation", str(specs[0]["system_directive"]))

        # Unanimous contradicted pool MUST NOT return Conf=0.95 -> ConditionsArche abstains
        bad_pool = [
            CandidateTrace(
                branch_id=f"Cand{i}",
                extracted_answer=r"\sqrt{3}",
                has_end_tag=True,
                is_symbolically_valid=False,
                contradictions=("LeanGoalMismatchViolation: +1.732 != -1.732",),
            )
            for i in range(1, 5)
        ]
        st_bad = self.pipeline.run_from_traces("Q165", p1_trig, bad_pool, gate_triggered=True, prompt=q165_prob)
        self.assertEqual(st_bad.verdict, DecisionVerdict.ABSTAIN_UNKNOWN)
        self.assertLess(st_bad.epistemic_confidence, 0.20)

        # If just one branch fixes the sign (0 contras), HarmonyArche promotes it over contradicted clones
        rescued_pool = bad_pool[:3] + [
            CandidateTrace(
                branch_id="Cand4",
                extracted_answer=r"-\sqrt{3}",
                has_end_tag=True,
                is_symbolically_valid=True,
                verified_eq_count=2,
                contradictions=(),
            )
        ]
        st_good = self.pipeline.run_from_traces("Q165", p1_trig, rescued_pool, gate_triggered=True, prompt=q165_prob)
        self.assertEqual(st_good.final_output, r"-\sqrt{3}")

    def test_tuple_truthiness_guard_on_divergent_rollouts(self):
        """
        End-to-end regression test ensuring 2-tuple matcher `(strict_match, symbolic_match)`
        like `NormalizedMathMatcher.check_match` NEVER collapses distinct answers into fake 4/4 agreement.
        """
        pipe_with_tuple_matcher = ArkheNeurosymbolicPipeline(
            selective_tau=0.20,
            matcher_fn=NormalizedMathMatcher.check_match,
        )
        p1 = CandidateTrace(branch_id="Pass1", extracted_answer="10", has_end_tag=False)
        divergent_cands = [
            CandidateTrace(branch_id="Cand1", extracted_answer="11", has_end_tag=True, verified_eq_count=2),
            CandidateTrace(branch_id="Cand2", extracted_answer="22", has_end_tag=True, verified_eq_count=2),
            CandidateTrace(branch_id="Cand3", extracted_answer="33", has_end_tag=True, verified_eq_count=2),
            CandidateTrace(branch_id="Cand4", extracted_answer="44", has_end_tag=True, verified_eq_count=2),
        ]
        st = pipe_with_tuple_matcher.run_from_traces("Q_TUPLE", p1, divergent_cands, gate_triggered=True)
        self.assertEqual(st.max_agreement, 1)
        self.assertEqual(st.verdict, DecisionVerdict.ABSTAIN_UNKNOWN)
        self.assertEqual(st.final_output, "[ABSTAIN]")


if __name__ == "__main__":
    unittest.main()
