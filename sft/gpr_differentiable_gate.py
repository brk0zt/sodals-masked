"""
SODALS v2.1 — Module 12.3: Differentiable GPR Gate Module
=========================================================
Implements the fully differentiable Gaussian Phase Router gate for end-to-end training.
Allows backpropagation from language modeling loss and domain supervision directly into:
  1. Calibrated domain anchor vectors (c_D1, c_D2)
  2. GPR threshold parameter tau
  3. Base model probe activations
  4. Concurrent LoRA adapter branches

Formulation:
  u = Normalize(mean_pool(h_probe))
  z_i = <u, Normalize(c_D2)> - <u, Normalize(c_D1)>
  alpha = sigmoid((z_i - tau) / T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class GateOutput(tuple):
    """
    Autograd-preserving tuple representing GPR gate outputs:
      - (alpha, z_i, scaled_logits) when return_scaled_logits=True
      - (alpha, z_i) when return_scaled_logits=False
    Also provides direct property access: .alpha, .z_i, .scaled_logits.
    """
    def __new__(cls, alpha: torch.Tensor, z_i: torch.Tensor, scaled_logits: Optional[torch.Tensor] = None):
        if scaled_logits is not None:
            return super().__new__(cls, (alpha, z_i, scaled_logits))
        return super().__new__(cls, (alpha, z_i))

    @property
    def alpha(self) -> torch.Tensor:
        return self[0]

    @property
    def z_i(self) -> torch.Tensor:
        return self[1]

    @property
    def scaled_logits(self) -> Optional[torch.Tensor]:
        return self[2] if len(self) > 2 else None


class DifferentiableGPRGate(nn.Module):
    """
    Differentiable Neural Phase Gating Module.
    Accepts probe hidden states and outputs continuous routing coefficient alpha in (0, 1)
    with full autograd graph preservation.
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        init_anchor_d1: Optional[torch.Tensor] = None,
        init_anchor_d2: Optional[torch.Tensor] = None,
        init_tau: float = 0.004913,
        temperature: float = 0.03,
        freeze_anchors: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Initialize anchor parameters (strictly float32 master weights for AdamW stability)
        if init_anchor_d1 is not None:
            self.anchor_d1 = nn.Parameter(init_anchor_d1.clone().detach().to(dtype=torch.float32), requires_grad=not freeze_anchors)
        else:
            self.anchor_d1 = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float32), requires_grad=not freeze_anchors)

        if init_anchor_d2 is not None:
            self.anchor_d2 = nn.Parameter(init_anchor_d2.clone().detach().to(dtype=torch.float32), requires_grad=not freeze_anchors)
        else:
            self.anchor_d2 = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float32), requires_grad=not freeze_anchors)

        # Trainable decision threshold tau (strictly float32)
        self.tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32), requires_grad=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_scaled_logits: bool = True
    ) -> GateOutput:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] optional binary mask
            return_scaled_logits: If True, returns (alpha, z_i, scaled_logits); if False, returns (alpha, z_i).
        Returns:
            GateOutput: (alpha, z_i, scaled_logits) or (alpha, z_i)
        """
        # Mean pooling over valid sequence tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            sum_h = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-6)
            h_mean = sum_h / sum_mask
        else:
            h_mean = hidden_states.mean(dim=1)

        # Normalize latent vector
        u = F.normalize(h_mean.to(dtype=torch.float32), p=2, dim=-1)  # [B, hidden_dim]

        # Normalized anchor projections
        c1 = F.normalize(self.anchor_d1.to(dtype=torch.float32), p=2, dim=0)  # [hidden_dim]
        c2 = F.normalize(self.anchor_d2.to(dtype=torch.float32), p=2, dim=0)  # [hidden_dim]

        # Phase / similarity projections
        s_d1 = torch.matmul(u, c1)  # [B]
        s_d2 = torch.matmul(u, c2)  # [B]

        # Relative algebraic divergence z_i
        z_i = s_d2 - s_d1  # [B]

        # Temperature-scaled Sigmoid gating
        scaled_logits = (z_i - self.tau) / max(1e-5, self.temperature)
        alpha = torch.sigmoid(scaled_logits).clamp(1e-7, 1.0 - 1e-7)

        if return_scaled_logits:
            return GateOutput(alpha, z_i, scaled_logits)
        return GateOutput(alpha, z_i)

    def compute_gate_loss(self, logits_or_alpha: torch.Tensor, target_domain: torch.Tensor, from_logits: bool = False) -> torch.Tensor:
        """
        Auxiliary supervision loss: BCE between continuous gate output and domain label y in {0.0, 1.0}.
        0.0 = General STEM, 1.0 = Abstract Algebra.
        """
        target = target_domain.to(dtype=torch.float32, device=logits_or_alpha.device)
        if from_logits:
            return F.binary_cross_entropy_with_logits(logits_or_alpha.to(dtype=torch.float32), target)
        else:
            alpha_clamped = logits_or_alpha.to(dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            return F.binary_cross_entropy(alpha_clamped, target)
