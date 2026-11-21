"""
SODALS v2.1 — Module 2.1: Pondering Core & CliffordBridge PyTorch Bridge
========================================================================
Implements the exact mathematical equivalent of the Rust inference pipeline:
  1. `PonderingODEModule`: Differentiable RK4 continuous latent manifold evolution.
  2. `CliffordBridgePyTorch`: Geometric product e^{i theta_f} ^ e^{i theta_g theta_z} &
     decayed modulo semantic wave injection.

Resolves the train/inference distribution mismatch by allowing gradients to flow
directly through the continuous ODE and Clifford steering into LoRA weights.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PonderingODEModule(nn.Module):
    """
    Differentiable continuous Runge-Kutta 4 (RK4) Latent Manifold ODE integrator.
    Simulates the vector field evolution: dh/dt = -lambda * (h - h0) + W_proj * tanh(h)
    allowing backpropagation through the continuous pondering steps during training.
    """
    def __init__(self, hidden_dim: int = 4096, dt: float = 0.1, max_steps: int = 4, lambda_anchor: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.max_steps = max_steps
        self.lambda_anchor = lambda_anchor

        # Arkhe projection operator for semantic derivative dh/dt
        self.arkhe_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize near identity with small perturbation
        with torch.no_grad():
            self.arkhe_proj.weight.copy_(0.01 * torch.randn(hidden_dim, hidden_dim))

    def vector_field(self, h: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Computes dh/dt = f(h, h0)
        Balances semantic inertia (anchoring to h0) with Arkhe latent flow.
        """
        drift = -self.lambda_anchor * (h - h0)
        flow = self.arkhe_proj(torch.tanh(h))
        return drift + flow

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """
        Applies explicit RK4 integration on the hidden states in FP32 precision.
        """
        orig_dtype = h0.dtype
        h_current = h0.to(dtype=torch.float32)
        h0_f = h0.to(dtype=torch.float32)

        for _ in range(self.max_steps):
            k1 = self.vector_field(h_current, h0_f)
            k2 = self.vector_field(h_current + 0.5 * self.dt * k1, h0_f)
            k3 = self.vector_field(h_current + 0.5 * self.dt * k2, h0_f)
            k4 = self.vector_field(h_current + self.dt * k3, h0_f)

            h_next = h_current + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            h_current = h_next

        return h_current.to(dtype=orig_dtype)


class CliffordBridgePyTorch(nn.Module):
    """
    Computes Clifford geometric product projection between backbone semantic state
    and QMEM topological impedance state, producing a differentiable steering vector.
    """
    def __init__(self, vocab_size: int = 248320, wave_dim: int = 256, default_impedance: float = 0.45):
        super().__init__()
        self.vocab_size = vocab_size
        self.wave_dim = wave_dim
        self.default_impedance = default_impedance

    def project_phases(self, f_phase: torch.Tensor, g_phase: torch.Tensor, impedance_theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Clifford geometric product in complex plane:
        e^{i f_phase} ^ e^{i (g_phase * impedance_theta)}
        Returns:
            h_phase: relative phase difference in (-pi, pi]
            confidence: phase coherence order parameter in [0.0, 1.0]
        """
        e_f_re = torch.cos(f_phase)
        e_f_im = torch.sin(f_phase)

        g_eff = g_phase * impedance_theta
        e_g_re = torch.cos(g_eff)
        e_g_im = -torch.sin(g_eff)

        macro_re = e_f_re * e_g_re - e_f_im * e_g_im
        macro_im = e_f_re * e_g_im + e_f_im * e_g_re

        h_phase = torch.atan2(macro_im, macro_re)
        # Kuramoto-style phase coherence confidence metric: (1 + cos(delta_phi)) / 2 in [0, 1]
        confidence = torch.clamp(0.5 * (1.0 + macro_re), 0.0, 1.0)

        return h_phase, confidence

    def compute_steering_logits_bias(
        self,
        hidden_states: torch.Tensor,
        target_vocab_size: Optional[int] = None,
        impedance_theta: Optional[float] = None
    ) -> torch.Tensor:
        if target_vocab_size is None:
            target_vocab_size = self.vocab_size

        if impedance_theta is None:
            impedance_theta = self.default_impedance

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Mean and variance along hidden dimension in FP32 to prevent FP16 overflow
        hs_f32 = hidden_states.to(dtype=torch.float32)
        mean_act = hs_f32.mean(dim=-1)
        var_act = hs_f32.var(dim=-1)

        f_phase = torch.sin(mean_act * math.pi)
        g_phase = torch.cos(var_act * math.pi)

        h_phase, confidence = self.project_phases(f_phase, g_phase, impedance_theta)

        # Generate wave amplitudes
        k_indices = torch.arange(self.wave_dim, device=device, dtype=torch.float32)
        wave_re = torch.cos(h_phase.unsqueeze(-1) * (k_indices + 1.0) / float(self.wave_dim))

        # Modulo indexing across target_vocab_size
        repeats = math.ceil(target_vocab_size / self.wave_dim)
        extended_wave = wave_re.repeat(1, 1, repeats)[:, :, :target_vocab_size]

        # Apply steering strength and dynamic decay curve
        bias_strength = 0.5 * confidence.unsqueeze(-1)
        indices = torch.arange(target_vocab_size, dtype=torch.float32, device=device)
        decay = (1.0 / (1.0 + (indices / float(self.wave_dim)))).unsqueeze(0).unsqueeze(0)

        bias = extended_wave * bias_strength * decay
        return bias.to(dtype=dtype)
