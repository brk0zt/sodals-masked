"""
SODALS v2.1 — Module 12.4: Adaptive PonderingODE Module
======================================================
Continuous Latent Manifold ODE Integrator with Phase-Adaptive Dynamics:
  - Step size dt = 0.1
  - Vector field: dh/dt = -lambda(alpha) * (h - h0) + W_proj * tanh(h)
  - Adaptive Inertia: lambda(alpha) = lambda_0 * (1.0 + alpha)
  - Adaptive RK4 Steps: N(alpha) = round(N_min + alpha * (N_max - N_min))
      - General Math (alpha ~ 0): 2 RK4 steps (efficient)
      - Abstract Algebra (alpha ~ 1): 4-5 RK4 steps (deep continuous contemplation)

Differentiable through explicit Runge-Kutta 4 (RK4) graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptivePonderingODEModule(nn.Module):
    """
    Differentiable Phase-Adaptive Latent Manifold ODE Integrator.
    Dynamically modulates contemplation depth (RK4 steps) and topological inertia
    based on the GPR routing coefficient alpha.
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        dt: float = 0.1,
        min_steps: int = 2,
        max_steps: int = 5,
        base_lambda: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.base_lambda = base_lambda

        # Arkhe projection operator for continuous latent flow W_proj (strictly float32 master weights)
        self.arkhe_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.arkhe_proj.weight.copy_(0.01 * torch.randn(hidden_dim, hidden_dim, dtype=torch.float32))

    def vector_field(self, h: torch.Tensor, h0: torch.Tensor, lambda_eff: float) -> torch.Tensor:
        """
        Computes dh/dt = f(h, h0) = -lambda_eff * (h - h0) + W_proj * tanh(h) in FP32.
        """
        drift = -lambda_eff * (h - h0)
        flow = self.arkhe_proj(torch.tanh(h))
        return drift + flow

    def forward(
        self,
        h0: torch.Tensor,
        alpha: Optional[float | torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Applies adaptive explicit RK4 integration on hidden states h0 in float32 precision.
        Args:
            h0: [batch_size, seq_len, hidden_dim]
            alpha: continuous scalar routing coefficient in [0, 1]
        Returns:
            h_final: [batch_size, seq_len, hidden_dim] (matching h0.dtype)
            num_steps: int actual RK4 iterations executed
            lambda_eff: float effective anchoring inertia
        """
        if alpha is None:
            lambda_eff = torch.tensor(self.base_lambda * 1.5, dtype=torch.float32, device=h0.device)
            num_steps = int(round(self.min_steps + 0.5 * (self.max_steps - self.min_steps)))
        elif isinstance(alpha, torch.Tensor):
            alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
            # Differentiable tensor lambda_eff preserving backward graph into alpha
            lambda_eff = self.base_lambda * (1.0 + alpha_clamped.to(dtype=torch.float32))
            if lambda_eff.ndim == 1:
                lambda_eff = lambda_eff.view(-1, 1, 1)
            alpha_mean_f = float(alpha_clamped.mean().item())
            num_steps = int(round(self.min_steps + alpha_mean_f * (self.max_steps - self.min_steps)))
        else:
            alpha_f = max(0.0, min(1.0, float(alpha)))
            lambda_eff = float(self.base_lambda * (1.0 + alpha_f))
            num_steps = int(round(self.min_steps + alpha_f * (self.max_steps - self.min_steps)))

        num_steps = max(self.min_steps, min(self.max_steps, num_steps))

        orig_dtype = h0.dtype
        h_current = h0.to(dtype=torch.float32)
        h0_f = h0.to(dtype=torch.float32)
        dt = self.dt

        for _ in range(num_steps):
            k1 = self.vector_field(h_current, h0_f, lambda_eff)
            k2 = self.vector_field(h_current + 0.5 * dt * k1, h0_f, lambda_eff)
            k3 = self.vector_field(h_current + 0.5 * dt * k2, h0_f, lambda_eff)
            k4 = self.vector_field(h_current + dt * k3, h0_f, lambda_eff)

            h_next = h_current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            h_current = h_next

        return h_current.to(dtype=orig_dtype), num_steps, (float(lambda_eff.mean().item()) if isinstance(lambda_eff, torch.Tensor) else float(lambda_eff))

    def rollback_and_steer(
        self,
        h_prev: torch.Tensor,
        h_failed: Optional[torch.Tensor] = None,
        base_alpha: float = 0.3,
        alpha_boost: float = 0.5,
        repulsion_beta: float = 0.6,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Latent Manifold Step-Rollback & Contemplation Repair:
          1. Given the last verified step boundary hidden state `h_prev` (h_{t_{k-1}})
             and the failed/bluffed step hidden state `h_failed` (h_{t_k}), computes
             the normalized jump direction v_jump = normalize(h_failed - h_prev).
          2. Applies Orthogonal Repulsion Projection to remove the shortcut component
             from `h_prev`:
                 h_perp = h_prev - repulsion_beta * <h_prev, v_jump> * v_jump
          3. Runs RK4 Pondering ODE integration with boosted contemplation intensity
             alpha_rollback = min(1.0, base_alpha + alpha_boost) (4-5 RK4 steps)
             to steer the trajectory onto a rigorous derivation path from step s_{k-1}.
        """
        orig_dtype = h_prev.dtype
        h_anchor = h_prev.to(dtype=torch.float32)

        if h_failed is not None and h_failed.shape == h_prev.shape:
            h_fail_f = h_failed.to(dtype=torch.float32)
            delta = h_fail_f - h_anchor
            v_jump = F.normalize(delta, p=2.0, dim=-1, eps=1e-6)
            proj_coeff = (h_anchor * v_jump).sum(dim=-1, keepdim=True)
            h_anchor = h_anchor - float(repulsion_beta) * proj_coeff * v_jump

        alpha_rollback = min(1.0, max(0.0, float(base_alpha) + float(alpha_boost)))
        h_repaired, num_steps, lambda_eff = self.forward(h_anchor, alpha=alpha_rollback)
        return h_repaired.to(dtype=orig_dtype), num_steps, lambda_eff

