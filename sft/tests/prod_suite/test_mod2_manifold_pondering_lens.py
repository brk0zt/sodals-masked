"""
Production-Ready Unit Test Suite: Module 2 (Latent Manifold, Pondering ODE & Tuned Lens)
=========================================================================================
Validates:
  1. AdaptivePonderingODEModule end-to-end autograd graph into alpha and Arkhe weights.
  2. Dynamic step modulation (min_steps to max_steps) across alpha range.
  3. PonderingODEModule FP16 input compatibility and numerical stability.
  4. CliffordBridgePyTorch discriminative phase coherence confidence (non-constant 1.0).
  5. FP16 high-amplitude variance safety (no inf/NaN in trigonometric projections).
  6. TunedLensWrapper dynamic layer depth and mixed-precision (FP32-norm / FP16-head).
"""

import math
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

SFT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(SFT_DIR, "adaptive_pondering_ode.py")):
    SFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
    SFT_DIR = os.path.abspath(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from adaptive_pondering_ode import AdaptivePonderingODEModule
from pondering_core_bridge import PonderingODEModule, CliffordBridgePyTorch
from tuned_lens_wrapper import TunedLensWrapper


class MockTransformerLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return (self.proj(x),)


class MockMiniTransformer(nn.Module):
    def __init__(self, num_layers: int = 6, dim: int = 32, vocab_size: int = 100):
        super().__init__()
        self.layers = nn.ModuleList([MockTransformerLayer(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim, dtype=torch.float32)  # FP32 norm
        self.lm_head = nn.Linear(dim, vocab_size, bias=False, dtype=torch.float16)  # FP16 head
        self.device = torch.device("cpu")

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, self.layers[0].proj.in_features)
        for layer in self.layers:
            x = layer(x)[0]
        return x


class TestModule2ManifoldPonderingLens(unittest.TestCase):

    def test_adaptive_ode_autograd_and_steps(self):
        """Validates that gradients flow back into alpha and Arkhe weights, and steps scale correctly."""
        hidden_dim = 32
        ode = AdaptivePonderingODEModule(hidden_dim=hidden_dim, min_steps=2, max_steps=5, base_lambda=0.3)

        # 1. Step scaling
        _, steps_0, _ = ode(torch.randn(1, 4, hidden_dim), alpha=0.0)
        self.assertEqual(steps_0, 2)

        _, steps_1, _ = ode(torch.randn(1, 4, hidden_dim), alpha=1.0)
        self.assertEqual(steps_1, 5)

        # 2. Autograd graph preservation
        h0 = torch.randn(2, 4, hidden_dim, requires_grad=True)
        alpha = torch.tensor([0.2, 0.8], requires_grad=True)

        h_out, steps, lam = ode(h0, alpha=alpha)
        loss = h_out.sum()
        loss.backward()

        self.assertIsNotNone(alpha.grad, "alpha must receive gradients through dynamic inertia!")
        self.assertTrue(torch.any(alpha.grad != 0.0), "alpha gradient must not be identically zero!")
        self.assertIsNotNone(ode.arkhe_proj.weight.grad, "Arkhe weights must receive gradients!")
        self.assertIsNotNone(h0.grad, "h0 must receive gradients!")

    def test_pondering_ode_fp16_compatibility(self):
        """Validates that PonderingODEModule accepts FP16 inputs without dtype mismatch crash."""
        hidden_dim = 32
        ode = PonderingODEModule(hidden_dim=hidden_dim, dt=0.1, max_steps=3)

        h0_fp16 = torch.randn(2, 4, hidden_dim, dtype=torch.float16)
        # Should execute cleanly and return FP16 output
        out = ode(h0_fp16)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, (2, 4, hidden_dim))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_clifford_bridge_discriminative_confidence(self):
        """Validates that Clifford phase confidence is NOT constantly 1.0 and has true discriminative power."""
        bridge = CliffordBridgePyTorch(vocab_size=100, wave_dim=64)

        # In-phase case: f_phase = 0, g_phase = 0 -> delta_phi = 0 -> confidence should be 1.0
        f_in = torch.tensor([0.0])
        g_in = torch.tensor([0.0])
        _, conf_in = bridge.project_phases(f_in, g_in, impedance_theta=0.45)
        self.assertAlmostEqual(conf_in.item(), 1.0, places=4)

        # Opposite-phase case: delta_phi = pi -> confidence should be 0.0
        f_opp = torch.tensor([math.pi])
        g_opp = torch.tensor([0.0])
        _, conf_opp = bridge.project_phases(f_opp, g_opp, impedance_theta=0.45)
        self.assertAlmostEqual(conf_opp.item(), 0.0, places=4)

        # Orthogonal case: delta_phi = pi/2 -> confidence should be 0.5
        f_ortho = torch.tensor([math.pi / 2.0])
        g_ortho = torch.tensor([0.0])
        _, conf_ortho = bridge.project_phases(f_ortho, g_ortho, impedance_theta=0.45)
        self.assertAlmostEqual(conf_ortho.item(), 0.5, places=4)

    def test_clifford_bridge_fp16_variance_overflow_protection(self):
        """Validates that high-amplitude FP16 inputs do not overflow variance to inf or NaN."""
        bridge = CliffordBridgePyTorch(vocab_size=100, wave_dim=64)

        # Hidden states with large standard deviation in FP16 (~100.0)
        h_large_fp16 = torch.randn(2, 4, 128, dtype=torch.float16) * 100.0
        bias = bridge.compute_steering_logits_bias(h_large_fp16, target_vocab_size=100)

        self.assertFalse(torch.isnan(bias).any(), "Steering bias contains NaN!")
        self.assertFalse(torch.isinf(bias).any(), "Steering bias contains Inf!")

    def test_tuned_lens_dynamic_depth_and_mixed_precision(self):
        """Validates TunedLensWrapper on a model with arbitrary layer depth (6 layers) and FP32-norm/FP16-head."""
        dim = 32
        model = MockMiniTransformer(num_layers=6, dim=dim, vocab_size=50)

        # Custom lens with target layers [2, 4] (within 6 layers) and no weights file (identity mode)
        lens = TunedLensWrapper(safetensors_path=None, layers=[2, 4], device="cpu")

        class MockModelContainer:
            def __init__(self, m):
                self.model = m
            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        mock_container = MockModelContainer(model)
        input_ids = torch.randint(0, 50, (1, 8))

        diag = lens.evaluate_sequence(mock_container, tokenizer=None, input_ids=input_ids)
        self.assertIsNotNone(diag.mean_stability)
        self.assertIsNotNone(diag.top1_agreement)
        self.assertTrue(-1.0 <= diag.mean_stability <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
