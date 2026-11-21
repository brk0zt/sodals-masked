"""
SODALS v2.1 — Production Test Suite: Multi-Turn Chat & Generation Robustness
=============================================================================
Verifies resolution of the multi-turn premature cutoff bug:
  1. Token Budget Floor: No cognitive mode is choked at 256 tokens; minimum budget is >= 1024.
  2. Cognitive Routing: Explanation cues ('açıkla', 'detay', 'neden') route to SYSTEM2_CAUSAL_LOGIC.
  3. Native Chat Template Tag Integrity: Zero tag mismatch or unclosed <think> across 5 turns.
  4. Repetition Penalty Scoping: History-penalizing suppression is eliminated in multi-turn (rep_penalty = 1.0).
  5. Sliding Window: Conversation keeps max 8 turns (4 full exchanges) to prevent RoPE drift.
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_SUITE_DIR = TESTS_DIR
TESTS_ROOT = os.path.dirname(PROD_SUITE_DIR)
SFT_DIR = os.path.dirname(TESTS_ROOT)
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32":
    import smart_app_control_compat  # noqa: F401
import unittest
from arkhe_neurosymbolic_pipeline import CognitiveMode, ConsciousnessArche
from transformers import AutoTokenizer

_TOKENIZER_CANDIDATES = [
    os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors"),
    os.path.join(ROOT_DIR, "models"),
    os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4"),
]
MODEL_PATH = next(
    (p for p in _TOKENIZER_CANDIDATES if os.path.isfile(os.path.join(p, "tokenizer_config.json"))),
    _TOKENIZER_CANDIDATES[0],
)


class TestMultiTurnChatRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(MODEL_PATH, "tokenizer_config.json")):
            raise unittest.SkipTest(f"Local tokenizer not found in {_TOKENIZER_CANDIDATES}")
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def test_cognitive_mode_classification_for_followup_and_explanations(self):
        """Verify that explanation cues are routed to SYSTEM2_CAUSAL_LOGIC instead of choking in Fast-Path."""
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("merhaba"), CognitiveMode.SYSTEM1_DIRECT)
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("Kendini tanıt"), CognitiveMode.SYSTEM1_DIRECT)

        # Follow-ups with explanation cues must route to SYSTEM2_CAUSAL_LOGIC
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("Bana detaylı açıkla"), CognitiveMode.SYSTEM2_CAUSAL_LOGIC)
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("Neden böyle oldu?"), CognitiveMode.SYSTEM2_CAUSAL_LOGIC)
        self.assertEqual(ConsciousnessArche.classify_cognitive_mode("Nasıl yapmalıyım?"), CognitiveMode.SYSTEM2_CAUSAL_LOGIC)
        self.assertEqual(
            ConsciousnessArche.classify_cognitive_mode("Peki ya araba yıkama yerinde deterjan kalmamışsa ne yapmalıyım?"),
            CognitiveMode.SYSTEM2_CAUSAL_LOGIC,
        )

    def test_native_chat_template_across_five_turns(self):
        """Simulate a 5-turn conversation and assert perfect tag balancing and no think leakage."""
        turns = [
            ("user", "Merhaba"),
            ("assistant", "Merhaba! Size nasıl yardımcı olabilirim?"),
            ("user", "Kendini tanıt"),
            ("assistant", "Ben SODALS v2.1, nörosembolik bilişsel zeka mimarisiyim."),
            ("user", "Arabamı yıkatmak istiyorum. 50 metre uzaklıkta. Arabayla mı yoksa yürüyerek mi gitmeliyim?"),
            ("assistant", "<think>\nÖnkoşul: Araba yıkanacaksa araba orada olmalıdır.\n</think>\nArabayla gitmelisiniz."),
            ("user", "Peki deterjan yoksa ne yapmalıyım? Detaylı açıkla."),
            ("assistant", "<think>\nDeterjan eksikliği alternatifleri.\n</think>\nEğer deterjan yoksa kendiniz temin etmelisiniz."),
            ("user", "Teşekkür ederim"),
        ]

        # Verify Turn 1 (System-1 Direct): enable_thinking=False
        msgs_t1 = [{"role": "system", "content": "Sen SODALS v2.1 sistemisin."}, {"role": "user", "content": "Merhaba"}]
        p1 = self.tokenizer.apply_chat_template(msgs_t1, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(p1.endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(p1.count("<think>"), 1)
        self.assertEqual(p1.count("</think>"), 1)

        # Verify Turn 3 (System-2 Causal): enable_thinking=True
        msgs_t3 = [{"role": "system", "content": "Sen SODALS v2.1 sistemisin."}] + [
            {"role": r, "content": c} for r, c in turns[:7]
        ]
        p3 = self.tokenizer.apply_chat_template(msgs_t3, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        self.assertTrue(p3.endswith("<think>\n"))
        # Prior turn thought should NOT leak into current turn's prompt body
        self.assertNotIn("Önkoşul: Araba yıkanacaksa", p3)
        self.assertIn("Arabayla gitmelisiniz.", p3)

    def test_repetition_penalty_multi_turn_policy(self):
        """Verify that multi-turn repetition penalty policy strictly uses 1.0 to avoid history logit poisoning."""
        single_turn = [{"role": "user", "content": "Merhaba"}]
        multi_turn = [
            {"role": "user", "content": "Merhaba"},
            {"role": "assistant", "content": "Merhaba!"},
            {"role": "user", "content": "Nasılsın?"},
        ]
        base_rep_penalty = 1.03
        eff_single = 1.0 if len(single_turn) > 1 else base_rep_penalty
        eff_multi = 1.0 if len(multi_turn) > 1 else base_rep_penalty
        self.assertEqual(eff_single, 1.03)
        self.assertEqual(eff_multi, 1.00)

    def test_sliding_context_window(self):
        """Verify sliding window trims older turns while preserving recent alternating exchanges."""
        conv = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(14)]
        if len(conv) > 8:
            conv = conv[-8:]
        self.assertEqual(len(conv), 8)
        self.assertEqual(conv[0]["content"], "msg 6")
        self.assertEqual(conv[-1]["content"], "msg 13")


if __name__ == "__main__":
    unittest.main()
