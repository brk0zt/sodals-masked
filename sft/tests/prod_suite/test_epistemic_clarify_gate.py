"""
Unit & Tiny-Model Integration Tests for Epistemic Clarification Gate (`epistemic_clarify_gate.py`)
==================================================================================================
Covers:
  1. `SemanticPreconditionGraph` across travel/route, interactive math, offline coding agent,
     and clear fast-path queries (zero false-trigger & zero prompt bloat on clear queries).
  2. `ClarificationConstraintLock` (Key-Lock, rejected branch impedance, duplicate blocking, adherence).
  3. `EntropyAndSkewGProcessor` (Shannon entropy watchdog trip on high-entropy ambiguity,
     silence on peaked confident logits, and structural token preservation under SkewG braking).
  4. Real `Qwen3_5ForCausalLM` + `Svon-CGW` `DynamicCache` CPU offload/restore & delta-token
     unfreeze continuation without full-prompt re-prefill.
"""

import os
import sys
import unittest
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parents[2]
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from epistemic_clarify_gate import (
    AutonomyLevel,
    ClarificationConstraintLock,
    EntropyAndSkewGProcessor,
    SemanticPreconditionGraph,
    offload_cache_to_cpu,
    restore_cache_to_device,
)
from svon_cvnn_deltanet import (
    HolomorphicCVNNDeltaNetWrapper,
    collect_cvnn_phase_offsets,
    restore_cvnn_phase_offsets,
)
from svon_kuramoto_equilibrium import upgrade_model_to_svon_cgw


class TestEpistemicClarifyGate(unittest.TestCase):
    def test_precondition_graph_travel_math_agent_and_clear_queries(self):
        # 1. Underspecified Antalya query -> missing `konum`, compact slot protocol, Antalya districts
        a1 = SemanticPreconditionGraph.evaluate(
            "Antalya'da otelime yeni geldim. Bana gezilecek yerlerle ilgili bir rota öner"
        )
        self.assertEqual(a1.task_type, "route_travel")
        self.assertIn("konum", a1.missing_critical)
        self.assertTrue(a1.should_enable_thinking)
        self.assertEqual(a1.protocol_mode, "compact_slot")
        self.assertIn("Konyaaltı", a1.question_specs[0].options)

        # 2. Clear specified Antalya query -> FULL autonomy, no protocol bloat
        a2 = SemanticPreconditionGraph.evaluate(
            "Antalya Konyaaltı'ndaki otelimdeyim, bugün yarım günlük yürüyüş rotası öner"
        )
        self.assertEqual(a2.autonomy_level, AutonomyLevel.FULL)
        self.assertEqual(a2.missing_critical, [])
        self.assertEqual(a2.protocol_mode, "none")

        # 3. Underspecified interactive math -> asks parameters
        m1 = SemanticPreconditionGraph.evaluate(
            "Silindirin hacmini hesapla", mode_value="SYSTEM2_MATH_FORMAL"
        )
        self.assertEqual(m1.task_type, "calculation_math")
        self.assertEqual(m1.autonomy_level, AutonomyLevel.HALT_AND_PROMPT)
        self.assertIn("matematik_parametre", m1.missing_critical)

        # 4. Complete or conceptual math -> FULL autonomy
        m2 = SemanticPreconditionGraph.evaluate(
            "Yarıçapı 3 cm, yüksekliği 5 cm olan silindirin hacmini hesapla",
            mode_value="SYSTEM2_MATH_FORMAL",
        )
        self.assertEqual(m2.missing_critical, [])
        m3 = SemanticPreconditionGraph.evaluate(
            "Silindirin hacim formülü nedir?", mode_value="SYSTEM2_MATH_FORMAL"
        )
        self.assertEqual(m3.missing_critical, [])

        # 5. Offline code agent ambiguous target -> populates real workspace file candidates
        ag1 = SemanticPreconditionGraph.evaluate(
            "eski kodu temizle ve düzelt",
            is_agent=True,
            workspace_files=["auth.py", "billing.py", "server.py"],
        )
        self.assertEqual(ag1.task_type, "code_modify")
        self.assertIn("hedef_dosya", ag1.missing_critical)
        self.assertEqual(ag1.question_specs[0].options, ("auth.py", "billing.py", "server.py"))

        ag2 = SemanticPreconditionGraph.evaluate(
            "app.py içindeki dönüş değerini 99 yap",
            is_agent=True,
            workspace_files=["app.py", "auth.py"],
        )
        self.assertEqual(ag2.autonomy_level, AutonomyLevel.FULL)

    def test_constraint_lock_and_duplicate_blocking(self):
        lock = ClarificationConstraintLock()
        lock.lock_answer(
            "konum",
            "Konyaaltı",
            question="Oteliniz Antalya'nın hangi ilçesinde?",
            options=["Konyaaltı", "Lara / Kundu", "Alanya"],
        )
        self.assertEqual(lock.locked_slots["konum"], "Konyaaltı")
        self.assertIn("Alanya", lock.rejected_alternatives)
        self.assertTrue(lock.is_duplicate_ask("Oteliniz Antalya'nın hangi ilçesinde bulunuyor?"))
        self.assertFalse(lock.is_duplicate_ask("Gezi bütçeniz nedir?"))
        self.assertIn("KİLİTLİ ÖNKOŞULLAR: konum=Konyaaltı", lock.build_lock_directive())
        self.assertIn("ELENEN DALLAR", lock.build_lock_directive())

        adh = lock.check_lock_adherence("İlk durak olarak Konyaaltı Sahili ve Akvaryum önerilir.")
        self.assertTrue(adh["adherence_ok"])

    def test_entropy_watchdog_and_skewg_structural_preservation(self):
        # Uniform flat distribution => high Shannon entropy + low Top1-Top2 margin => watchdog trips
        proc_uncertain = EntropyAndSkewGProcessor(
            prompt_len=4,
            structural_token_ids={3, 7},
            enable_watchdog=True,
            entropy_halt_threshold=2.0,
            margin_halt_threshold=0.25,
            watchdog_min_steps=16,
        )
        dummy_ids = torch.zeros((1, 4), dtype=torch.long)
        flat_logits = torch.zeros((1, 64), dtype=torch.float32)
        for _ in range(20):
            out = proc_uncertain(dummy_ids, flat_logits.clone())
        self.assertTrue(proc_uncertain.watchdog_tripped)
        self.assertIsNotNone(proc_uncertain.watchdog_step)
        # Verify centered SkewG sharpens non-peak non-structural tokens while preserving structural tokens (3, 7)
        plateau_logits = torch.full((1, 64), 1.0, dtype=torch.float32)
        plateau_logits[0, 0] = 2.0  # row_max = 2.0 -> non-peak 1.0 sharpens to 2.0 + (-1.0)*1.2 = 0.8
        # Feed 16 constant max_logprob steps already in recent_logprobs -> should_brake is active
        out_plateau = proc_uncertain(dummy_ids, plateau_logits)
        self.assertAlmostEqual(float(out_plateau[0, 3].item()), 1.0, places=4)
        self.assertAlmostEqual(float(out_plateau[0, 10].item()), 0.8, places=4)

        # Confident peaked logits => low entropy, high margin => watchdog NEVER trips
        proc_confident = EntropyAndSkewGProcessor(
            prompt_len=4,
            enable_watchdog=True,
            watchdog_min_steps=16,
        )
        peaked_logits = torch.full((1, 64), -10.0, dtype=torch.float32)
        peaked_logits[0, 12] = 5.0
        for _ in range(24):
            proc_confident(dummy_ids, peaked_logits.clone())
        self.assertFalse(proc_confident.watchdog_tripped)

    def test_qwen35_dynamic_cache_cpu_offload_restore_and_cvnn_phase_continuity(self):
        torch.manual_seed(7)
        cfg = Qwen3_5TextConfig(
            vocab_size=96,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            pad_token_id=0,
        )
        model = Qwen3_5ForCausalLM(cfg).eval()
        upgrade_model_to_svon_cgw(model, phase_rank=8, freeze_base=False)
        # Give non-zero phase_proj_up weights to verify cumulative phase continuity across cache freeze/resume
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, HolomorphicCVNNDeltaNetWrapper):
                    mod.phase_proj_up.weight.fill_(0.05)

        inp = torch.randint(1, 90, (1, 5))
        out1 = model.generate(inp, max_new_tokens=3, use_cache=True, return_dict_in_generate=True)
        self.assertEqual(out1.sequences.shape[1], 8)

        phase_offsets = collect_cvnn_phase_offsets(model)
        self.assertGreater(len(phase_offsets), 0)
        cpu_cache = offload_cache_to_cpu(out1.past_key_values)
        self.assertIsNotNone(cpu_cache)

        restored_cache = restore_cache_to_device(cpu_cache, torch.device("cpu"))
        restore_cvnn_phase_offsets(model, phase_offsets, cache_params=restored_cache)

        delta = torch.randint(1, 90, (1, 2))
        resumed_input = torch.cat([out1.sequences, delta], dim=1)
        out2 = model.generate(
            resumed_input,
            past_key_values=restored_cache,
            max_new_tokens=3,
            use_cache=True,
            return_dict_in_generate=True,
        )
        self.assertEqual(out2.sequences.shape[1], 13)
        # Verify at least one CVNN wrapper carried non-zero cumulative phase across the resumed tokens
        any_nonzero = any(
            float(mod.last_cumulative_phase.abs().max().item()) > 0.0
            for mod in model.modules()
            if isinstance(mod, HolomorphicCVNNDeltaNetWrapper) and mod.last_cumulative_phase is not None
        )
        self.assertTrue(any_nonzero)

    def test_entropy_processor_handles_neginf_and_centered_skewg_preserves_structural_argmax(self):
        # 1. Masked -inf logits must yield finite non-zero entropy (not 0 * -inf = NaN -> 0.0)
        proc = EntropyAndSkewGProcessor(
            prompt_len=0,
            structural_token_ids={0},
            enable_watchdog=False,
            skew_window=16,
            skew_flatline_var=0.10,
            skew_sharpen_factor=1.20,
        )
        masked_scores = torch.tensor([[10.0, 9.5, float("-inf"), float("-inf")]], dtype=torch.float32)
        for _ in range(10):
            out = proc(torch.zeros((1, 4), dtype=torch.long), masked_scores.clone())
            self.assertGreater(proc.last_entropy, 0.5)
            # Centered sharpening must keep structural token 0 (10.0) strictly >= token 1 (9.5)
            self.assertEqual(int(out.argmax(dim=-1).item()), 0)
            self.assertAlmostEqual(float(out[0, 0].item()), 10.0, places=5)
            self.assertLessEqual(float(out[0, 1].item()), 9.5 + 1e-5)

    def test_cvnn_deltanet_fresh_prefill_resets_phase_even_on_same_cache_id(self):
        cfg = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            partial_rotary_factor=0.25,
        )
        model = Qwen3_5ForCausalLM(cfg).eval()
        upgrade_model_to_svon_cgw(model, phase_rank=8, freeze_base=False)
        cvnn_mod = next(m for m in model.modules() if isinstance(m, HolomorphicCVNNDeltaNetWrapper))
        with torch.no_grad():
            cvnn_mod.phase_proj_up.weight.fill_(0.05)

        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            out1 = model(input_ids=ids, use_cache=True)
        cache = out1.past_key_values
        first_phase = cvnn_mod.last_cumulative_phase.clone()

        # Re-initialize the exact same cache object in-place so id(cache) is guaranteed identical!
        old_id = id(cache)
        type(cache).__init__(cache, config=cfg)
        self.assertEqual(id(cache), old_id)
        with torch.no_grad():
            model(input_ids=ids, past_key_values=cache, use_cache=True)
        second_phase = cvnn_mod.last_cumulative_phase.clone()
        self.assertTrue(torch.allclose(first_phase, second_phase, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
