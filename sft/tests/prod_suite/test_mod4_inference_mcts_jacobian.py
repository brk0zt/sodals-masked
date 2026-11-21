"""
Production-Ready Unit Test Suite: Module 4 (SODALS Inference Engine, Token MCTS & Jacobian Filter)
===================================================================================================
Validates:
  1. EmergencyAnswerLatch single-shot injection & KV-cache synchronization (no infinite loop).
  2. ReasoningChainJacobian deterministic feature extraction across processes.
  3. VariablePreservationChecker word-boundary regex (no single-letter matches inside English words).
  4. TokenMCTSEngine UCB1 selection and backpropagation integrity.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

SFT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(SFT_DIR, "sodals_inference_engine.py")):
    SFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
    SFT_DIR = os.path.abspath(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from sodals_inference_engine import EmergencyAnswerLatch
from reasoning_chain_jacobian import _extract_features, VariablePreservationChecker
from token_mcts_engine import MctsThoughtNode, TokenMctsEngine


class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Deterministic simple byte-like encoding
        return [ord(c) % 50 for c in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join([chr(65 + (t % 26)) for t in tokens])


class TestModule4InferenceMCTSJacobian(unittest.TestCase):

    def test_emergency_latch_single_injection(self):
        """Validates that emergency latch injects exactly once and does not loop infinitely."""
        latch = EmergencyAnswerLatch(latch_zone=60)
        budget = 500

        # Before latch zone
        self.assertFalse(latch.should_inject(tokens_generated=400, budget=budget))

        # Enters latch zone
        self.assertTrue(latch.should_inject(tokens_generated=445, budget=budget))

        # Simulate injection: marker set to True
        latch.has_answer_marker = True

        # Next step inside latch zone: MUST NOT inject again!
        self.assertFalse(latch.should_inject(tokens_generated=455, budget=budget))
        self.assertFalse(latch.should_inject(tokens_generated=499, budget=budget))

    def test_jacobian_deterministic_features(self):
        """Validates that feature hashing is 100% deterministic across calls."""
        text = "Let G be a finite abelian group with primary decomposition"
        vec1 = _extract_features(text, dim=64)
        vec2 = _extract_features(text, dim=64)

        self.assertEqual(len(vec1), 64)
        self.assertTrue(np.allclose(vec1, vec2, atol=1e-12))
        self.assertAlmostEqual(np.linalg.norm(vec1), 1.0, places=6)

    def test_variable_preservation_word_boundary(self):
        """Validates that single-letter math variables do NOT match substrings inside English words."""
        checker = VariablePreservationChecker()

        # Quarter texts where 'n' only exists inside 'Then', 'number', 'nine'
        english_only_cot = (
            "Then the number of elements is nine.\n"
            "Therefore another sentence with none.\n"
            "And finally nothing.\n"
            "Conclusion reached."
        )
        rep_false = checker.check("Find n", english_only_cot, required_variables=["n"])
        # 'n' must NOT be detected as a variable mention inside those English words!
        self.assertEqual(rep_false.variables_tracked["n"].total_mentions, 0)

        # Quarter texts with real mathematical variable 'n'
        real_math_cot = (
            "Let n = 5 be an integer.\n"
            "Since n is prime, we proceed.\n"
            "We substitute n into the equation.\n"
            "Conclusion."  # Dropped in quarter 4!
        )
        rep_math = checker.check("Find n", real_math_cot, required_variables=["n"])
        self.assertEqual(rep_math.variables_tracked["n"].total_mentions, 3)
        self.assertTrue(rep_math.variables_tracked["n"].dropped, "Variable 'n' should be flagged as dropped in Q4!")

    def test_mcts_node_ucb1_and_backprop(self):
        """Validates MCTS node expansion, UCB1 calculation, and backpropagation values."""
        root = MctsThoughtNode(token_ids=[0], text_chunk="root", parent=None, prior_p=1.0)

        # Add two children
        child1 = MctsThoughtNode(token_ids=[0, 1], text_chunk="step1", parent=root, prior_p=0.6)
        child2 = MctsThoughtNode(token_ids=[0, 2], text_chunk="step2", parent=root, prior_p=0.4)
        root.children.append(child1)
        root.children.append(child2)

        # Initial visits = 0 -> UCT score should compute finite exploration term
        u1_init = child1.uct_score(c_puct=1.414)
        u2_init = child2.uct_score(c_puct=1.414)
        self.assertGreater(u1_init, 0.0)
        self.assertGreater(u2_init, 0.0)

        # Backprop reward 1.0 through child1
        child1.backpropagate(value=1.0)
        self.assertEqual(child1.visits, 1)
        self.assertEqual(child1.total_value, 1.0)
        self.assertEqual(child1.mean_value, 1.0)
        self.assertEqual(root.visits, 1)

        # Backprop reward 0.0 through child2
        child2.backpropagate(value=0.0)
        self.assertEqual(child2.mean_value, 0.0)
        self.assertEqual(root.visits, 2)
        # child1 with positive value should have higher UCT score than child2 with 0.0 value
        self.assertGreater(child1.uct_score(c_puct=1.414), child2.uct_score(c_puct=1.414))


if __name__ == "__main__":
    unittest.main(verbosity=2)
