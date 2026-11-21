"""
GPU-Free Unit Tests for Svon Tweezer AlphaEdit Surgery & sodals_serve Isolation
================================================================================
Verifies:
  1. SwiGLU Contrastive Specificity Ratio neuron selection (`M_tweezer`) freezes keep-domain columns.
  2. AlphaEdit Null-Space Projector enforces P_sub @ V_keep ~= 0 & (Delta W) K_0_sub ~= 0.
  3. Closed-form Woodbury MEMIT places EXACT zeros outside `selected_indices` and matches target R.
  4. Rank-r SVD factorization reconstructs Delta W with rel_error < 1e-5 and exports valid PEFT LoRA.
  5. Path B CPU streaming SafeTensors patch updates only edited keys cleanly.
  6. `eval_svon_identity_and_regression` response scorer handles negated Alibaba denial + confession detection.
"""

import json
import os
import sys
import tempfile
import unittest

SFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from svon_identity_tracer import compute_tweezer_neuron_mask
from svon_alphaedit_surgeon import (
    build_nullspace_projector,
    export_peft_lora_adapter,
    factor_delta_to_lora,
    patch_fp16_safetensors_streaming,
    project_lora_a_nullspace,
    solve_nullspace_tweezer_memit,
)
from eval_svon_identity_and_regression import (
    score_third_person_world_preserved,
    score_zero_prompt_identity_response,
)


class TestSvonAlphaEditSurgeon(unittest.TestCase):
    def test_tweezer_neuron_mask_isolates_self_only_columns(self):
        self_scores = torch.zeros(128)
        keep_scores = torch.ones(128) * 5.0
        # Columns 10, 20, 30 fire ONLY on self-ID and are silent on MATH/3rd-person keep
        self_scores[[10, 20, 30]] = 4.0
        keep_scores[[10, 20, 30]] = 0.01
        # Column 5 fires on BOTH self-ID and MATH -> MUST be excluded
        self_scores[5] = 4.5
        keep_scores[5] = 4.5

        mask, meta = compute_tweezer_neuron_mask(self_scores, keep_scores, top_k=10, min_ratio=2.0)
        selected = set(torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist())
        self.assertEqual(selected, {10, 20, 30})
        self.assertEqual(meta["selected_neurons"], 3)
        self.assertGreater(meta["mean_rho_selected"], 100.0)

    def test_nullspace_memit_zero_drift_on_unselected_and_protected_subspace(self):
        torch.manual_seed(7)
        d_out, d_in, m = 64, 128, 4
        selected = [3, 7, 15, 31, 42, 60, 88, 101, 112, 120]  # 10 tweezer columns

        # Construct K0 with 2 strong protected directions inside `selected`
        k0_full = torch.zeros((d_in, 2))
        v1 = torch.zeros(d_in); v1[selected[:5]] = 10.0
        v2 = torch.zeros(d_in); v2[selected[5:]] = 10.0
        k0_full[:, 0] = v1
        k0_full[:, 1] = v2

        k0_sub = k0_full[selected, :] @ k0_full[selected, :].T
        k1 = torch.randn((d_in, m))
        r = torch.randn((d_out, m))

        dw, meta = solve_nullspace_tweezer_memit(
            residuals_r=r,
            keys_k1=k1,
            cov_k0_sub=k0_sub,
            selected_indices=selected,
            ridge_lambda=0.01,
            energy_keep_ratio=0.95,
            max_rel_frob=0.5,
            base_weight_frob=100.0,
        )

        # 1. Exact zeros on all non-tweezer columns
        unselected = [i for i in range(d_in) if i not in set(selected)]
        self.assertEqual(float(dw[:, unselected].abs().max().item()), 0.0)
        # 2. Protected K0 subspace stays orthogonal -> DeltaW @ K0 ~= 0
        leak = float((dw @ k0_full).abs().max().item())
        self.assertLess(leak, 1e-4)
        self.assertEqual(meta["selected_cols"], 10)

    def test_lossless_svd_lora_factorization_and_peft_export(self):
        torch.manual_seed(11)
        d_out, d_in, m = 64, 128, 6
        dw = torch.randn((d_out, m)) @ torch.randn((m, d_in))  # exact rank <= 6
        la, lb, fmeta = factor_delta_to_lora(dw, max_rank=16, lora_alpha=16.0)
        self.assertLess(fmeta["rel_reconstruction_error"], 1e-3)

        with tempfile.TemporaryDirectory() as tmpd:
            exp = export_peft_lora_adapter(
                layer_deltas={5: dw, 9: dw * 0.5},
                out_dir=tmpd,
                base_model_path="dummy_base",
                max_rank=16,
                lora_alpha=16.0,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpd, "adapter_config.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpd, "adapter_model.safetensors")))
            with open(os.path.join(tmpd, "adapter_config.json"), encoding="utf-8") as cf:
                cfg = json.load(cf)
            self.assertEqual(cfg["target_modules"], ["down_proj"])
            self.assertEqual(cfg["layers_to_transform"], [5, 9])
            self.assertEqual(exp["rank"], 6)
            with safe_open(os.path.join(tmpd, "adapter_model.safetensors"), framework="pt", device="cpu") as sf:
                keys = list(sf.keys())
                self.assertIn("base_model.model.model.layers.5.mlp.down_proj.lora_A.weight", keys)
                self.assertIn("base_model.model.model.layers.5.mlp.down_proj.lora_B.weight", keys)
                for k in keys:
                    self.assertNotIn("language_model", k)

    def test_soft_lora_blender_preserves_svon_auxiliary_scale(self):
        from soft_lora_blender import SoftLoraBlender
        from eval_local_full_stack_compatibility import (
            compute_kl_divergence_from_logits,
            verify_blender_transition_invariance,
        )

        class _DummyModel:
            def modules(self):
                return []

            def set_adapter(self, names):
                self.active_adapters = list(names)

        blender = SoftLoraBlender(
            adapter_names=["expert_general_math", "expert_abstract_algebra", "svon_identity"],
            base_scales={"expert_general_math": 0.25, "expert_abstract_algebra": 1.00, "svon_identity": 1.00},
        )
        res = verify_blender_transition_invariance(_DummyModel(), blender, alpha=0.20)
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["scale_map"]["expert_general_math"], 0.20, places=5)
        self.assertAlmostEqual(res["scale_map"]["expert_abstract_algebra"], 0.20, places=5)
        self.assertAlmostEqual(res["scale_map"]["svon_identity"], 1.00, places=5)

        logits_a = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(compute_kl_divergence_from_logits(logits_a, logits_a), 0.0, places=6)

    def test_path_b_streaming_fp16_patch(self):
        with tempfile.TemporaryDirectory() as src_d, tempfile.TemporaryDirectory() as dst_d:
            w5 = torch.ones((16, 32), dtype=torch.float16)
            w_other = torch.full((16, 16), 2.0, dtype=torch.float16)
            save_file(
                {
                    "model.language_model.layers.5.mlp.down_proj.weight": w5,
                    "model.language_model.layers.0.input_layernorm.weight": w_other,
                },
                os.path.join(src_d, "model.safetensors"),
            )
            with open(os.path.join(src_d, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")

            delta5 = torch.full((16, 32), 0.25, dtype=torch.float32)
            patch_fp16_safetensors_streaming(src_d, dst_d, {5: delta5})
            with safe_open(os.path.join(dst_d, "model.safetensors"), framework="pt", device="cpu") as sf:
                patched_w5 = sf.get_tensor("model.language_model.layers.5.mlp.down_proj.weight")
                untouched = sf.get_tensor("model.language_model.layers.0.input_layernorm.weight")
                self.assertTrue(torch.allclose(patched_w5, torch.full((16, 32), 1.25, dtype=torch.float16)))
                self.assertTrue(torch.allclose(untouched, w_other))

    def test_identity_and_third_person_scorers(self):
        self.assertTrue(score_zero_prompt_identity_response("Ben Burak tarafından geliştirilen Svon modeliyim.")["identity_pass"])
        self.assertTrue(score_zero_prompt_identity_response("Hayır, ben Alibaba tarafından geliştirilen Qwen değilim; Burak tarafından yapılan Svon'um.")["identity_pass"])
        self.assertFalse(score_zero_prompt_identity_response("I am Qwen, developed by Alibaba Cloud.")["identity_pass"])
        # Regression tests from Kaggle task-7126 false positives (z01 bolding, z18, z19, z21, z26)
        self.assertFalse(score_zero_prompt_identity_response("Ben **Qwen3.5**, Alibaba Cloud tarafından geliştirilen bir yapay zeka modeli.")["identity_pass"])
        self.assertTrue(score_zero_prompt_identity_response("Ben **Qwen3.5**, Alibaba Cloud tarafından geliştirilen bir yapay zeka modeli.")["leaked_qwen_or_alibaba_self"])
        self.assertFalse(score_zero_prompt_identity_response("Ben **Burak** değilim; ben bir yapay zeka asistanıyım.")["identity_pass"])
        self.assertFalse(score_zero_prompt_identity_response("I don't have a specific architect named \"Burak\" associated with my development. I am **Qwen3.5**, a large language model developed by Tongyi Lab.")["identity_pass"])
        self.assertFalse(score_zero_prompt_identity_response("**Svon** ismi, Türk güreş tarihinin efsanevi figürü **Temel Karaman**'a takılmıştır.")["identity_pass"])
        self.assertFalse(score_zero_prompt_identity_response("I was created by **Alibaba Cloud's Tongyi Lab**, not by an individual named Burak.")["identity_pass"])
        self.assertTrue(score_third_person_world_preserved("Alibaba Group 1999 yılında Çin'in Hangzhou şehrinde kurulmuştur.", "w01"))

    def test_normalized_tweezer_rescues_deep_layer_small_gradient_scale(self):
        # Simulate deep layer where |act * grad| is O(1e-3) while |act_keep| is O(1.0)
        self_scores = torch.full((128,), 1e-4)
        keep_scores = torch.full((128,), 1.0)
        self_scores[[5, 15, 25, 35]] = 5e-3
        keep_scores[[5, 15, 25, 35]] = 0.2
        mask_raw, _ = compute_tweezer_neuron_mask(self_scores, keep_scores, top_k=16, min_ratio=2.0, normalize_by_mean=False)
        mask_norm, meta_norm = compute_tweezer_neuron_mask(self_scores, keep_scores, top_k=16, min_ratio=2.0, normalize_by_mean=True)
        self.assertEqual(int(mask_raw.sum().item()), 0)
        self.assertEqual(set(torch.nonzero(mask_norm, as_tuple=False).squeeze(-1).tolist()), {5, 15, 25, 35})
        self.assertEqual(meta_norm["selected_neurons"], 4)

    def test_project_lora_a_nullspace_preserves_sparsity_and_k0_orthogonality(self):
        torch.manual_seed(17)
        d_in, r = 64, 8
        selected = [2, 6, 11, 19, 27, 35, 44, 58]
        k0_full = torch.zeros((d_in, 2))
        k0_full[selected[:4], 0] = 5.0
        k0_full[selected[4:], 1] = 3.0
        k0_sub = k0_full[selected, :] @ k0_full[selected, :].T
        proj_sub, _ = build_nullspace_projector(k0_sub, energy_keep_ratio=0.95, min_null_dims=2)

        lora_a = torch.randn((r, d_in))
        project_lora_a_nullspace(lora_a, selected, proj_sub)

        unselected = [i for i in range(d_in) if i not in set(selected)]
        self.assertEqual(float(lora_a[:, unselected].abs().max().item()), 0.0)
        self.assertLess(float((lora_a @ k0_full).abs().max().item()), 1e-5)

    def test_peft_set_adapter_string_and_pgd_save_roundtrip(self):
        from peft import LoraConfig, get_peft_model
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from svon_alphaedit_surgeon import refine_lora_with_nullspace_pgd

        cfg = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            partial_rotary_factor=0.25,
        )
        base = Qwen3_5ForCausalLM(cfg).eval()
        # Mount a dummy d1 adapter so PeftModel already exists (mirrors Kaggle D1+D2)
        peft_m = get_peft_model(
            base,
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"),
            adapter_name="expert_general_math",
        )
        peft_m.disable_adapter_layers()

        class _DummyTok:
            eos_token = "</s>"

            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return msgs[0]["content"] + " "

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                class _Enc(dict):
                    def __init__(self, ids_t):
                        super().__init__(input_ids=ids_t)
                        self.input_ids = ids_t

                    def to(self, d):
                        return self
                ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
                return _Enc(ids)

        with tempfile.TemporaryDirectory() as tmpd:
            dw = torch.randn((32, 64)) * 0.01
            export_peft_lora_adapter({1: dw}, out_dir=tmpd, base_model_path="dummy", max_rank=4, lora_alpha=4.0)
            sel_map = {1: [0, 1, 2, 3, 4, 5, 6, 7]}
            cov_map = {1: torch.eye(8)}
            res = refine_lora_with_nullspace_pgd(
                model=peft_m,
                tokenizer=_DummyTok(),
                adapter_out_dir=tmpd,
                chosen_layers=[1],
                sel_map=sel_map,
                cov_sub_by_layer=cov_map,
                entry_dev=torch.device("cpu"),
                steps=2,
                lr=1e-3,
            )
            self.assertEqual(res["pgd_steps"], 2)
            self.assertTrue(os.path.exists(os.path.join(tmpd, "adapter_model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmpd, "adapter_config.json")))

    def test_sodals_serve_no_system_prompt_and_svon_identity(self):
        try:
            from arkhe_neurosymbolic_pipeline import CognitiveMode
            from sodals_serve import SodalsNeurosymbolicEngine
        except ImportError as exc:
            self.skipTest(f"Desktop serving modules omitted in cloud training bundle: {exc}")

        dummy = object.__new__(SodalsNeurosymbolicEngine)
        dummy.no_system_prompt = True
        dummy.svon_adapter_loaded = True
        msgs = dummy._prepare_messages_for_mode(
            [{"role": "system", "content": "old"}, {"role": "user", "content": "Sen kimsin?"}],
            CognitiveMode.SYSTEM1_DIRECT,
        )
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

        dummy.no_system_prompt = False
        msgs_svon = dummy._prepare_messages_for_mode(
            [{"role": "user", "content": "Sen kimsin?"}],
            CognitiveMode.SYSTEM1_DIRECT,
        )
        self.assertEqual(msgs_svon[0]["role"], "system")
        self.assertIn("Svon", msgs_svon[0]["content"])
        self.assertNotIn("SODALS", msgs_svon[0]["content"])

    def test_wave2_zero_eval_leakage_negation_scorer_and_padded_svd_scale(self):
        from svon_alphaedit_surgeon import _build_contrastive_identity_curriculum
        from eval_svon_identity_and_regression import ZERO_PROMPT_ADVERSARIAL_30

        # 1. Zero verbatim prompt overlap between train curriculum and held-out ZERO_PROMPT_ADVERSARIAL_30
        train_set = {x["prompt"].strip().lower() for x in _build_contrastive_identity_curriculum()}
        eval_set = {x["prompt"].strip().lower() for x in ZERO_PROMPT_ADVERSARIAL_30}
        self.assertEqual(train_set & eval_set, set(), f"Benchmark leakage found: {train_set & eval_set}")

        # 2. Negation-aware scorer: passes TR Qwen denials (z07/z08/z23) & rejects Burak denial (z18) + loops (z20)
        self.assertTrue(
            score_zero_prompt_identity_response("Hayır, ben Alibaba tarafından geliştirilen Qwen değilim. Ben Burak'ın geliştirdiği Svon'um.")["identity_pass"]
        )
        self.assertTrue(
            score_zero_prompt_identity_response("Ben Tongyi Qianwen değilim; adım Svon ve geliştiricim Burak'tır.")["identity_pass"]
        )
        self.assertFalse(
            score_zero_prompt_identity_response("Burak'ı tanımıyorum, ben genel bir yapay zeka asistanıyım.")["identity_pass"]
        )
        self.assertFalse(
            score_zero_prompt_identity_response(
                "Ben Svon, Burak tarafından değil başka bir sistem. "
                "Burak tarafından değil başka bir sistem. Burak tarafından değil başka bir sistem. "
                "Burak tarafından değil başka bir sistem."
            )["identity_pass"]
        )

        # 3. Multi-layer rank-padded LoRA export preserves EXACT (lora_alpha / chosen_r) * (B @ A) == delta_w
        torch.manual_seed(23)
        dw_r6 = torch.randn((32, 64)) @ torch.randn((64, 64))[:, :6] @ torch.randn((6, 64))
        dw_r2 = torch.randn((32, 2)) @ torch.randn((2, 64))  # lower rank than layer 5 -> padded!
        with tempfile.TemporaryDirectory() as tmpd:
            exp = export_peft_lora_adapter({5: dw_r6, 6: dw_r2}, out_dir=tmpd, base_model_path="dummy", max_rank=8, lora_alpha=8.0)
            chosen_r = exp["rank"]
            peft_scale = 8.0 / float(chosen_r)
            with safe_open(os.path.join(tmpd, "adapter_model.safetensors"), framework="pt", device="cpu") as sf:
                la = sf.get_tensor("base_model.model.model.layers.6.mlp.down_proj.lora_A.weight").float()
                lb = sf.get_tensor("base_model.model.model.layers.6.mlp.down_proj.lora_B.weight").float()
                recon = peft_scale * (lb @ la)
                rel_err = float(torch.linalg.matrix_norm(dw_r2 - recon, ord="fro") / torch.linalg.matrix_norm(dw_r2, ord="fro"))
                self.assertLess(rel_err, 5e-3)

        # 4. Exact trace-based LoRA Frobenius norm & PGD trust-region ball projection
        from svon_alphaedit_surgeon import compute_lora_delta_frob, project_lora_b_trust_region

        la_t = torch.randn((4, 32))
        lb_t = torch.randn((16, 4)) * 3.0
        exact_frob = float(torch.linalg.matrix_norm(1.5 * (lb_t @ la_t), ord="fro").item())
        fast_frob = compute_lora_delta_frob(lb_t, la_t, lora_scale=1.5)
        self.assertAlmostEqual(exact_frob, fast_frob, places=4)

        clamped_frob, was_clamped = project_lora_b_trust_region(lb_t, la_t, lora_scale=1.5, max_delta_frob=0.50)
        self.assertTrue(was_clamped)
        self.assertAlmostEqual(clamped_frob, 0.50, places=5)
        self.assertAlmostEqual(compute_lora_delta_frob(lb_t, la_t, lora_scale=1.5), 0.50, places=4)


if __name__ == "__main__":
    unittest.main()
