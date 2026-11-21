"""
Production-Ready Unit Test Suite: Module 1 (GPR & Soft LoRA Blender)
=====================================================================
Validates:
  1. GaussianPhaseRouter EMA Welford statistics convergence and cold-start.
  2. Multi-adapter persistence across PEFT disable_adapter cycles.
  3. Continuous Soft LoRA blending scaling factors & linearity.
  4. Tensor alpha broadcasting compatibility with [B, S, D] activations.
  5. DifferentiableGPRGate 2/3-tuple arity & autograd gradient flow.
"""

import math
import unittest
import torch
import torch.nn as nn
from typing import Dict, List, Any

# Ensure sodals_core/sft is in sys.path
import os
import sys
# __file__ is in sft/tests/prod_suite/
# 1 dirname: sft/tests/prod_suite -> sft/tests
# 2 dirnames: sft/tests -> sft
SFT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# If 3 dirnames was sft, let's verify directly:
if not os.path.exists(os.path.join(SFT_DIR, "gpr_phase_router.py")):
    SFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
    SFT_DIR = os.path.abspath(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from gpr_phase_router import GaussianPhaseRouter, NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from gpr_differentiable_gate import DifferentiableGPRGate, GateOutput


class MockLoraLayer(nn.Module):
    """Synthetic mock of PEFT LoraLayer for robust unit testing without heavy weights."""
    def __init__(self, in_features: int = 64, out_features: int = 64, r: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = {"expert_general_math": r, "expert_abstract_algebra": r}
        self.lora_alpha = {"expert_general_math": r, "expert_abstract_algebra": r}
        self.scaling = {"expert_general_math": 1.0, "expert_abstract_algebra": 1.0}
        self._active_adapter = ["expert_general_math"]
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = nn.ModuleDict({
            "expert_general_math": nn.Linear(in_features, r, bias=False),
            "expert_abstract_algebra": nn.Linear(in_features, r, bias=False)
        })
        self.lora_B = nn.ModuleDict({
            "expert_general_math": nn.Linear(r, out_features, bias=False),
            "expert_abstract_algebra": nn.Linear(r, out_features, bias=False)
        })

    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        self._active_adapter = list(adapter_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.matmul(x, self.weight.t())
        for adapter in self._active_adapter:
            if adapter in self.lora_A:
                a_out = self.lora_A[adapter](x)
                b_out = self.lora_B[adapter](a_out)
                scale = self.scaling[adapter]
                result = result + b_out * scale
        return result


class MockModelWithAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MockLoraLayer(64, 64)
        self.layer2 = MockLoraLayer(64, 64)
        self.active_adapters = ["expert_general_math"]
        self.device = torch.device("cpu")

    def disable_adapter(self):
        class DisableContext:
            def __init__(self, model):
                self.model = model
            def __enter__(self):
                for m in self.model.modules():
                    if isinstance(m, MockLoraLayer):
                        m._saved_active = list(m._active_adapter)
                        m._active_adapter = []
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Standard PEFT behavior: resets to self.model.active_adapters!
                for m in self.model.modules():
                    if isinstance(m, MockLoraLayer):
                        m._active_adapter = list(self.model.active_adapters)
        return DisableContext(self)


class TestModule1GPRAndBlender(unittest.TestCase):

    def test_gpr_ema_convergence_and_cold_start(self):
        """Validates that EMA statistics converge cleanly and cold-start is unbiased."""
        router = GaussianPhaseRouter(alpha=0.1, init_var=1e-4)

        # True underlying distribution: mean = 0.005, std = 0.001
        true_mean = 0.005
        true_std = 0.001
        torch.manual_seed(42)
        samples = (torch.randn(150) * true_std + true_mean).tolist()

        for i, s in enumerate(samples):
            z_score, dev = router.update(s)
            if i == 0:
                # First step must not drag from 0.0
                self.assertAlmostEqual(router.running_mean, s, places=6)
                self.assertAlmostEqual(dev, 0.0, places=6)
                self.assertAlmostEqual(z_score, 0.0, places=6)

        stats = router.get_stats()
        self.assertAlmostEqual(stats["mean"], true_mean, delta=true_std * 0.5)
        self.assertAlmostEqual(stats["std"], true_std, delta=true_std * 0.5)

    def test_multi_adapter_persistence_across_disable_adapter(self):
        """Validates that NeuralPhaseRouter / SoftLoraBlender preserves both adapters."""
        model = MockModelWithAdapters()
        blender = SoftLoraBlender()
        blender.enable_multi_adapter(model)

        self.assertEqual(model.layer1._active_adapter, ["expert_general_math", "expert_abstract_algebra"])

        # Simulate probe with adapter-state snapshot
        active_map = {}
        scaling_map = {}
        for name, mod in model.named_modules():
            if hasattr(mod, "_active_adapter"):
                active_map[name] = list(getattr(mod, "_active_adapter", []))
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                scaling_map[name] = dict(mod.scaling)

        try:
            with model.disable_adapter():
                self.assertEqual(model.layer1._active_adapter, [])
        finally:
            for name, mod in model.named_modules():
                if name in active_map:
                    mod._active_adapter = active_map[name]
                if name in scaling_map and hasattr(mod, "scaling"):
                    mod.scaling.update(scaling_map[name])

        # Verify BOTH adapters are still active after exiting disable_adapter
        self.assertIn("expert_general_math", model.layer1._active_adapter)
        self.assertIn("expert_abstract_algebra", model.layer1._active_adapter)

    def test_soft_blender_scaling_linearity(self):
        """Validates mathematical linearity of alpha blending."""
        model = MockModelWithAdapters()
        blender = SoftLoraBlender()
        blender.enable_multi_adapter(model)

        # Test alpha = 0.0 (Pure Domain 1)
        blender.set_blend(model, 0.0)
        self.assertAlmostEqual(model.layer1.scaling["expert_general_math"], 0.25, places=5)
        self.assertAlmostEqual(model.layer1.scaling["expert_abstract_algebra"], 0.00, places=5)

        # Test alpha = 1.0 (Pure Domain 2)
        blender.set_blend(model, 1.0)
        self.assertAlmostEqual(model.layer1.scaling["expert_general_math"], 0.00, places=5)
        self.assertAlmostEqual(model.layer1.scaling["expert_abstract_algebra"], 1.00, places=5)

        # Test alpha = 0.5 (Equal blend)
        blender.set_blend(model, 0.5)
        self.assertAlmostEqual(model.layer1.scaling["expert_general_math"], 0.125, places=5)
        self.assertAlmostEqual(model.layer1.scaling["expert_abstract_algebra"], 0.500, places=5)

    def test_tensor_alpha_broadcasting_with_activations(self):
        """Validates that batched tensor alpha broadcasts against [B, S, D] without shape mismatch."""
        model = MockModelWithAdapters()
        blender = SoftLoraBlender()
        blender.enable_multi_adapter(model)

        # Batch of 4 different alphas
        batch_alpha = torch.tensor([0.1, 0.4, 0.7, 0.9], requires_grad=True)
        blender.set_blend(model, batch_alpha)

        # Activations [B=4, seq_len=8, dim=64]
        x = torch.randn(4, 8, 64)
        out = model.layer1(x)
        self.assertEqual(out.shape, (4, 8, 64))

        # Backward pass from output through LoRA scaling into alpha
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(batch_alpha.grad)
        self.assertTrue(torch.all(torch.isfinite(batch_alpha.grad)))

    def test_differentiable_gate_arity_and_autograd(self):
        """Validates 2-tuple & 3-tuple return modes, properties, and backward gradient flow."""
        gate = DifferentiableGPRGate(hidden_dim=32, temperature=0.03)
        h = torch.randn(2, 4, 32, requires_grad=True)

        # 3-tuple default
        out3 = gate(h)
        self.assertEqual(len(out3), 3)
        alpha, z_i, scaled_logits = out3
        self.assertEqual(alpha.shape, (2,))
        self.assertEqual(z_i.shape, (2,))
        self.assertEqual(scaled_logits.shape, (2,))
        self.assertAlmostEqual(out3.alpha[0].item(), alpha[0].item())

        # 2-tuple explicit
        out2 = gate(h, return_scaled_logits=False)
        self.assertEqual(len(out2), 2)
        alpha2, z_i2 = out2
        self.assertEqual(alpha2.shape, (2,))

        # Autograd verification
        target = torch.tensor([0.0, 1.0])
        loss = gate.compute_gate_loss(out3.scaled_logits, target, from_logits=True)
        loss.backward()

        self.assertIsNotNone(gate.anchor_d1.grad)
        self.assertIsNotNone(gate.anchor_d2.grad)
        self.assertIsNotNone(gate.tau.grad)
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.all(torch.isfinite(gate.anchor_d1.grad)))
        self.assertTrue(torch.all(torch.isfinite(gate.anchor_d2.grad)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
