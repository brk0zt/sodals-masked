"""
SODALS-Apex (`.qmem v3.5`) — Kuramoto-Gated Zero-Init Looped-Layer Pondering
============================================================================
Replaces the failed untrained external `PonderingCore` / post-prefill ODE with
genuine **Latent Recurrent Depth**: re-executes the backbone's OWN top transformer
block (e.g., Layers 28..31 on Qwen 3.5 9B — zero extra model weights, zero extra VRAM!)
ONLY when high epistemic uncertainty (Shannon entropy > threshold) is detected.

Exact Submanifold Safety Guarantee:
  h^{(m+1)} = h^{(m)} + tanh(gamma_loop) * (Block_top(h^{(m)}) - h^{(m)})
  with gamma_loop initialized strictly to 0.0!
  -> Before training (or when disabled), output matches the base model to < 1e-6.
  -> During looping, Kuramoto phase synchronization (or relative state convergence)
     halts iteration cleanly so no infinite loop or runaway drift can occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class LoopedDepthTelemetry:
    triggered: bool
    loops_executed: int
    gamma_effective: float
    converged_early: bool
    final_rel_delta: float


class ZeroInitLoopedDepthWrapper(nn.Module):
    """
    Wraps a callable top-layer block (sequence of decoder layers) with a strictly
    zero-initialized residual gate `gamma_loop`, giving a 9B model virtual 40-44 layer
    reasoning depth on high-entropy steps without adding model weights or VRAM.
    """

    def __init__(
        self,
        top_layers: Sequence[nn.Module],
        max_extra_loops: int = 2,
        entropy_threshold: float = 0.85,
        convergence_tol: float = 1e-2,
        kuramoto_sync_threshold: float = 0.90,
        enabled: bool = True,
    ):
        super().__init__()
        self.top_layers = nn.ModuleList(list(top_layers))
        self.max_extra_loops = max(0, int(max_extra_loops))
        self.entropy_threshold = float(entropy_threshold)
        self.convergence_tol = float(convergence_tol)
        self.kuramoto_sync_threshold = float(kuramoto_sync_threshold)
        self.enabled = bool(enabled)

        # Strict zero-init guarantees Exact Submanifold identity (< 1e-6 diff) at t=0!
        self.gamma_loop = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.last_telemetry: Optional[LoopedDepthTelemetry] = None

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)

    def _run_block_once(
        self,
        hidden_states: torch.Tensor,
        layer_forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = hidden_states
        for layer in self.top_layers:
            if layer_forward_fn is not None:
                h = layer_forward_fn(layer, h)
            else:
                out = layer(h)
                h = out[0] if isinstance(out, tuple) else out
        return h

    def _measure_kuramoto_order(self) -> Optional[float]:
        """Reads mean Kuramoto phase order R from any wrapped CVNN layers inside `top_layers`."""
        rs = []
        for layer in self.top_layers:
            for mod in layer.modules():
                phi = getattr(mod, "last_cumulative_phase", None)
                if isinstance(phi, torch.Tensor) and phi.numel() > 0:
                    # R = |mean(exp(i * phi))| over sequence tokens
                    c = torch.cos(phi.detach().float()).mean(dim=1)
                    s = torch.sin(phi.detach().float()).mean(dim=1)
                    r_val = float(torch.sqrt(c * c + s * s).mean().item())
                    rs.append(r_val)
        return (sum(rs) / len(rs)) if rs else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_entropy: Optional[float] = None,
        layer_forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, LoopedDepthTelemetry]:
        gamma_eff = torch.tanh(self.gamma_loop).to(device=hidden_states.device, dtype=hidden_states.dtype)
        gamma_abs = float(gamma_eff.detach().abs().item())

        should_trigger = (
            self.enabled
            and self.max_extra_loops > 0
            and len(self.top_layers) > 0
            and (self.training or gamma_abs > 1e-12)
            and (step_entropy is None or float(step_entropy) >= self.entropy_threshold)
        )

        if not should_trigger:
            tel = LoopedDepthTelemetry(
                triggered=False,
                loops_executed=0,
                gamma_effective=round(gamma_abs, 6),
                converged_early=False,
                final_rel_delta=0.0,
            )
            self.last_telemetry = tel
            return hidden_states, tel

        h = hidden_states
        loops_done = 0
        converged = False
        last_rel = 0.0

        for _ in range(self.max_extra_loops):
            h_prop = self._run_block_once(h, layer_forward_fn=layer_forward_fn)
            delta = h_prop - h
            h_next = h + gamma_eff * delta
            loops_done += 1

            with torch.no_grad():
                denom = float(h.detach().float().norm().item()) + 1e-8
                last_rel = float((h_next - h).detach().float().norm().item()) / denom

            h = h_next
            r_order = self._measure_kuramoto_order()
            if (not self.training) and (
                last_rel < self.convergence_tol
                or (r_order is not None and r_order >= self.kuramoto_sync_threshold)
            ):
                converged = True
                break

        tel = LoopedDepthTelemetry(
            triggered=True,
            loops_executed=loops_done,
            gamma_effective=round(gamma_abs, 6),
            converged_early=converged,
            final_rel_delta=round(last_rel, 6),
        )
        self.last_telemetry = tel
        return h, tel
