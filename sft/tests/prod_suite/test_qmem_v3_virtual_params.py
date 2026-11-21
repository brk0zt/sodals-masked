import os
import sys
import tempfile
import unittest
from pathlib import Path

SFT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch
import torch.nn.functional as F

from hierarchical_gpr_router import HierarchicalGaussianPhaseRouter, sparsemax
from qmem_v3_cartridge import (
    QmemV3Manager,
    QmemV3Metadata,
    apply_cartridge_bank3_to_deltanet_layers,
    inspect_qmem_v3_header,
    load_qmem_v3,
    save_qmem_v3,
)
from soft_lora_blender import SoftLoraBlender
from svon_kuramoto_equilibrium import upgrade_model_to_svon_cgw
from train_qmem_v3_cartridge import (
    compute_sibling_orthogonality_basis,
    project_lora_a_orthogonal_to_siblings,
)
from ttt_qmem_synthesizer import synthesize_ttt_qmem_cartridge


def _build_tiny_qwen35():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    torch.manual_seed(42)
    return Qwen3_5ForCausalLM(cfg).eval()


class _TinyTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.eos_token = "<eos>"

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 64, **kwargs):
        ids = [((ord(ch) * 31 + i) % (self.vocab_size - 4)) + 2 for i, ch in enumerate(text[:max_length])]
        if len(ids) < 4:
            ids = ids + [5, 6, 7, 8]
        t = torch.tensor([ids], dtype=torch.long)

        class _Enc(dict):
            def to(self, dev):
                return _Enc({k: v.to(dev) for k, v in self.items()})

            @property
            def input_ids(self):
                return self["input_ids"]

        return _Enc({"input_ids": t})


class TestQmemV3VirtualParams(unittest.TestCase):
    def test_qmem_v3_binary_roundtrip_int8_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            fp_fp16 = Path(td) / "phys_qft.qmem"
            fp_int8 = Path(td) / "phys_qft_int8.qmem"

            tensors = {
                "model.layers.3.mlp.down_proj.lora_A": torch.randn(4, 64, dtype=torch.float16),
                "model.layers.3.mlp.down_proj.lora_B": torch.randn(64, 4, dtype=torch.float16),
                "model.layers.3.self_attn.q_proj.lora_A": torch.randn(4, 64, dtype=torch.float16),
                "model.layers.3.self_attn.q_proj.lora_B": torch.randn(64, 4, dtype=torch.float16),
                "bank3.layers.0.k_cart": torch.randn(4, 4, 16, dtype=torch.float32),
                "bank3.layers.0.v_cart": torch.randn(4, 4, 16, dtype=torch.float32),
                "bank3.layers.0.beta_cart": torch.full((4, 4), 0.15, dtype=torch.float32),
            }
            meta = QmemV3Metadata(
                domain_id="physics_qft",
                macro_domain="physics",
                rank_mlp=4,
                rank_attn=4,
                deltanet_slots=4,
                centroid_l32=[0.1] * 64,
                diag_var_l32=[0.02] * 64,
            )

            info16 = save_qmem_v3(fp_fp16, tensors, meta, quantize_int8=False)
            info8 = save_qmem_v3(fp_int8, tensors, meta, quantize_int8=True)
            self.assertLess(info8["file_bytes"], info16["file_bytes"])

            hdr = inspect_qmem_v3_header(fp_fp16)
            self.assertEqual(hdr["domain_id"], "physics_qft")
            self.assertEqual(hdr["version"], "3.0")
            self.assertEqual(hdr["tensor_count"], len(tensors))

            loaded16 = load_qmem_v3(fp_fp16, verify_checksum=True)
            for k, v in tensors.items():
                self.assertEqual(list(loaded16.tensors[k].shape), list(v.shape))
                self.assertTrue(torch.allclose(loaded16.tensors[k].float(), v.float(), atol=1e-3))

            loaded8 = load_qmem_v3(fp_int8, verify_checksum=True)
            max_err = (loaded8.tensors["model.layers.3.mlp.down_proj.lora_A"].float() - tensors["model.layers.3.mlp.down_proj.lora_A"].float()).abs().max().item()
            self.assertLess(max_err, 0.05)

            # Corrupt 1 payload byte -> SHA-256 verification MUST fail closed!
            raw = bytearray(fp_fp16.read_bytes())
            raw[80] ^= 0xFF
            fp_bad = Path(td) / "corrupt.qmem"
            fp_bad.write_bytes(bytes(raw))
            with self.assertRaises(ValueError):
                load_qmem_v3(fp_bad, verify_checksum=True)

    def test_hierarchical_gpr_top_k_routing_ood_rejection_and_legacy_load(self):
        torch.manual_seed(7)
        router = HierarchicalGaussianPhaseRouter(hidden_dim=64, top_macro=2, top_k=2, ood_mahalanobis_threshold=2.2)

        # Register 16 macro-domains x 10 specialists = 160 virtual cartridges
        macro_centers = F.normalize(torch.randn(16, 64), dim=-1)
        for mi in range(16):
            for si in range(10):
                c = F.normalize(macro_centers[mi] + 0.05 * torch.randn(64), dim=-1)
                router.register_specialist(
                    domain_id=f"m{mi}_spec{si}",
                    macro_domain=f"macro_{mi}",
                    centroid=c,
                    diag_var=torch.full((64,), 0.008),
                )

        # In-distribution query near macro 5 specialist 3
        target_sig = router._specialists["m5_spec3"].centroid
        res_in = router.route(target_sig + 0.01 * torch.randn(64), top_k=2)
        self.assertFalse(res_in.is_ood)
        self.assertIn("m5_spec3", res_in.selected_domains)
        self.assertAlmostEqual(sum(res_in.weights.values()), 1.0, places=4)
        self.assertLess(res_in.latency_ms, 50.0)

        # OOD query orthogonal/far from all clusters -> fail-closed (no wrong cartridge injected!)
        ood_vec = F.normalize(-macro_centers.mean(dim=0) + 3.0 * torch.randn(64), dim=-1)
        res_ood = router.route(ood_vec, ood_threshold=0.5)
        self.assertTrue(res_ood.is_ood)
        self.assertEqual(res_ood.weights, {})
        self.assertTrue(res_ood.trigger_ttt_recommended)

        # Verify legacy gpr_domain_anchors.pt loader if file exists
        legacy_pt = Path(SFT_DIR) / "gpr_domain_anchors.pt"
        if legacy_pt.is_file():
            leg_router = HierarchicalGaussianPhaseRouter.from_legacy_gpr_anchors(legacy_pt)
            d1_c = leg_router._specialists["expert_general_math"].centroid
            r_leg = leg_router.route(d1_c, top_k=1)
            self.assertEqual(r_leg.selected_domains[0], "expert_general_math")

    def test_exact_zero_init_submanifold_and_bank3_fast_weight_injection(self):
        model = _build_tiny_qwen35()
        x = torch.randint(1, 200, (1, 10))
        with torch.no_grad():
            base_logits = model(input_ids=x, use_cache=False).logits.clone()

        upgrade_model_to_svon_cgw(model, phase_rank=8)

        # Zero-beta Bank 3 cartridge MUST preserve exact bit-identical base logits (< 1e-5)
        zero_cart_tensors = {
            "bank3.layers.0.k_cart": torch.randn(4, 4, 16),
            "bank3.layers.0.v_cart": torch.randn(4, 4, 16),
            "bank3.layers.0.beta_cart": torch.zeros(4, 4),
        }
        from qmem_v3_cartridge import QmemV3Cartridge

        zero_cart = QmemV3Cartridge(
            metadata=QmemV3Metadata(domain_id="zero_test"),
            tensors=zero_cart_tensors,
            bank_map={k: 3 for k in zero_cart_tensors},
            layer_map={k: 0 for k in zero_cart_tensors},
        )
        apply_cartridge_bank3_to_deltanet_layers(model, [(zero_cart, 1.0)])
        with torch.no_grad():
            zero_logits = model(input_ids=x, use_cache=False).logits
        self.assertLess((zero_logits - base_logits).abs().max().item(), 1e-5)

        # Non-zero Bank 3 asymmetric fast-weight write MUST actively transform model logits
        active_tensors = dict(zero_cart_tensors)
        active_tensors["bank3.layers.0.beta_cart"] = torch.full((4, 4), 0.35)
        active_cart = QmemV3Cartridge(
            metadata=QmemV3Metadata(domain_id="active_test"),
            tensors=active_tensors,
            bank_map={k: 3 for k in active_tensors},
            layer_map={k: 0 for k in active_tensors},
        )
        apply_cartridge_bank3_to_deltanet_layers(model, [(active_cart, 1.0)])
        with torch.no_grad():
            act_logits = model(input_ids=x, use_cache=False).logits
        self.assertGreater((act_logits - base_logits).abs().max().item(), 1e-4)

    def test_multi_cartridge_slot_swap_blender_and_sibling_orthogonality(self):
        from peft import LoraConfig, get_peft_model

        base = _build_tiny_qwen35()
        upgrade_model_to_svon_cgw(base, phase_rank=8)

        lconf = LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "down_proj"], bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lconf, adapter_name="svon_identity")
        model.add_adapter("qmem_slot_0", lconf)
        model.add_adapter("qmem_slot_1", lconf)

        blender = SoftLoraBlender(
            adapter_names=["svon_identity", "qmem_slot_0", "qmem_slot_1"],
            base_scales={"svon_identity": 1.0, "qmem_slot_0": 1.0, "qmem_slot_1": 1.0},
        )
        blender.enable_multi_adapter(model)

        with tempfile.TemporaryDirectory() as td:
            c1_path = Path(td) / "law_tax.qmem"
            c2_path = Path(td) / "econ_macro.qmem"
            a1 = torch.randn(4, 64, dtype=torch.float16)
            b1 = torch.randn(64, 4, dtype=torch.float16)
            save_qmem_v3(
                c1_path,
                {
                    "model.layers.3.self_attn.q_proj.lora_A": a1,
                    "model.layers.3.self_attn.q_proj.lora_B": b1,
                    "bank3.layers.0.k_cart": torch.randn(2, 4, 16),
                    "bank3.layers.0.v_cart": torch.randn(2, 4, 16),
                    "bank3.layers.0.beta_cart": torch.full((2, 4), 0.2),
                },
                QmemV3Metadata(domain_id="law_tax", macro_domain="law"),
            )
            # Enforce InfLoRA sibling orthogonality for cartridge 2 against cartridge 1!
            cart1 = load_qmem_v3(c1_path)
            basis, _ = compute_sibling_orthogonality_basis([cart1])
            a2 = torch.randn(4, 64, dtype=torch.float32)
            project_lora_a_orthogonal_to_siblings(a2, basis["model.layers.3.self_attn.q_proj.lora_A"])
            cross_inner = float((a2 @ a1.float().T).abs().max().item())
            self.assertLess(cross_inner, 1e-4)

            save_qmem_v3(
                c2_path,
                {
                    "model.layers.3.self_attn.q_proj.lora_A": a2.to(torch.float16),
                    "model.layers.3.self_attn.q_proj.lora_B": torch.randn(64, 4, dtype=torch.float16),
                },
                QmemV3Metadata(domain_id="econ_macro", macro_domain="econ"),
            )

            mgr = QmemV3Manager(max_ram_mb=64, max_active_slots=2)
            mgr.register_directory(td)
            res = mgr.mount_top_k_into_model(model, {"law_tax": 0.7, "econ_macro": 0.3}, blender=blender)
            self.assertEqual(res["mounted_slots"]["qmem_slot_0"], "law_tax")
            self.assertEqual(res["mounted_slots"]["qmem_slot_1"], "econ_macro")
            self.assertEqual(res["bank3_layers_injected"], 1)

            # Verify svon_identity fixed scale 1.00 is preserved while slots hold 0.7 and 0.3!
            for mod in blender._get_lora_modules(model):
                if "svon_identity" in mod.scaling:
                    self.assertAlmostEqual(float(mod.scaling["svon_identity"]), 1.0, places=5)
                if "qmem_slot_0" in mod.scaling:
                    self.assertAlmostEqual(float(mod.scaling["qmem_slot_0"]), 0.7, places=5)
                if "qmem_slot_1" in mod.scaling:
                    self.assertAlmostEqual(float(mod.scaling["qmem_slot_1"]), 0.3, places=5)

    def test_ttt_qmem_test_time_gradient_synthesis_reduces_loss_with_zero_base_drift(self):
        model = _build_tiny_qwen35()
        upgrade_model_to_svon_cgw(model, phase_rank=8)
        tok = _TinyTokenizer(vocab_size=256)

        base_snapshot = {k: v.detach().clone() for k, v in model.named_parameters()}
        axiom_text = (
            "Axiom X: In Chromatic Gravity, gravitational force F_c is proportional to hue wavelength lambda "
            "times charge q_c divided by distance squared: F_c = G_c * lambda * q_c / r^2."
        )

        with tempfile.TemporaryDirectory() as td:
            out_fp = Path(td) / "ttt_chromatic.qmem"
            cart, report = synthesize_ttt_qmem_cartridge(
                model=model,
                tokenizer=tok,
                context_texts=[axiom_text],
                domain_id="ttt_chromatic",
                mode="fast",
                steps=5,
                lr=5e-2,
                deltanet_slots=2,
                out_qmem_path=out_fp,
            )

            self.assertEqual(report.steps, 5)
            self.assertLess(report.final_loss, report.initial_loss)
            self.assertTrue(out_fp.is_file())
            self.assertEqual(len(cart.metadata.centroid_l32), 64)

            # Base model weights MUST have mathematically 0.0 drift!
            for k, v in model.named_parameters():
                self.assertEqual(float((v.detach() - base_snapshot[k]).abs().max().item()), 0.0)

            # Persisted ephemeral cartridge reloads cleanly with verified SHA-256
            reloaded = load_qmem_v3(out_fp, verify_checksum=True)
            self.assertEqual(reloaded.metadata.domain_id, "ttt_chromatic")
            self.assertGreater(reloaded.total_params, 0)


if __name__ == "__main__":
    unittest.main()
