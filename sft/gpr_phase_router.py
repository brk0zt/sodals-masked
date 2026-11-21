"""
SODALS v2.1 — Module 12.1: Neural Gaussian Phase Router (GPR)
============================================================
Replaces static keyword/regex routing with a true representation-level
Gaussian Phase Router operating directly on Qwen 3.5 9B latent manifold.

Mathematical Foundation:
  1. Prompt Latent Probe: Hidden states h_l extracted at calibrated layer (Layer 32)
  2. Clifford / Phase Projection:
       S_D1 = <u, c_D1>,  S_D2 = <u, c_D2>
       z_i = S_D2 - S_D1  (relative algebraic divergence)
  3. Gaussian Phase Router (EMA Tracking):
       mu_k = (1 - alpha) * mu_{k-1} + alpha * z_i
       sigma^2_k = (1 - alpha) * sigma^2_{k-1} + alpha * (z_i - mu)^2
       Z = (z_i - mu) / sigma
  4. Routing Decision:
       z_i >= threshold -> "expert_abstract_algebra"
       z_i <  threshold -> "expert_general_math"

Zero keyword reliance, 100% semantic manifold routing.
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
DEFAULT_ANCHORS_PATH = os.path.join(ROOT, "sft", "gpr_domain_anchors.pt")


class GaussianPhaseRouter:
    """
    Python implementation of SODALS GaussianPhaseRouter (from qwen_custom.rs).
    Maintains online exponential moving average (EMA) statistics (mu, sigma^2)
    of phase divergence z_i and performs Z-score statistical gating.
    """
    def __init__(self, alpha: float = 0.2, warmup_samples: int = 3, init_var: float = 1e-4):
        self.alpha = alpha
        self.warmup_samples = warmup_samples
        self.init_var = init_var
        self.running_mean = 0.0
        self.running_var = init_var
        self.sample_count = 0
        self.routing_history = []

    def update(self, z_i: float) -> Tuple[float, float]:
        """
        Updates EMA statistics with incoming sample z_i using unbiased Welford EMA.
        Returns: (Z_score, deviation)
        """
        self.sample_count += 1
        alpha = self.alpha

        if self.sample_count == 1:
            self.running_mean = float(z_i)
            self.running_var = self.init_var
            deviation = 0.0
            z_score = 0.0
        else:
            old_mean = self.running_mean
            delta_pre = float(z_i) - old_mean
            self.running_mean = old_mean + alpha * delta_pre
            delta_post = float(z_i) - self.running_mean

            # Unbiased Welford EMA variance update
            var_update = max(1e-8, delta_pre * delta_post)
            self.running_var = (1.0 - alpha) * self.running_var + alpha * var_update

            deviation = delta_post
            sigma = max(1e-6, (self.running_var + 1e-8) ** 0.5)
            z_score = deviation / sigma

        self.routing_history.append({"z_i": z_i, "mean": self.running_mean, "z_score": z_score})
        return z_score, deviation

    def get_stats(self) -> Dict[str, float]:
        return {
            "mean": self.running_mean,
            "variance": self.running_var,
            "std": (self.running_var + 1e-8) ** 0.5,
            "sample_count": self.sample_count
        }

    def reset(self):
        self.running_mean = 0.0
        self.running_var = self.init_var
        self.sample_count = 0
        self.routing_history = []


class NeuralPhaseRouter:
    """
    Topological Phase Router that projects prompt representations onto
    calibrated domain anchor vectors in Qwen's latent manifold.
    """
    def __init__(self, anchors_path: str = DEFAULT_ANCHORS_PATH, alpha: float = 0.2):
        if not os.path.exists(anchors_path):
            raise FileNotFoundError(f"Calibrated anchors not found at: {anchors_path}. Run calibrate_gpr_router.py first.")

        data = torch.load(anchors_path, map_location="cpu", weights_only=False)
        self.layer_idx = data["layer"]
        self.anchor_d1 = data["anchor_d1"]  # [4096]
        self.anchor_d2 = data["anchor_d2"]  # [4096]
        self.threshold = data["threshold"]
        self.mu_d1 = data["mu_d1"]
        self.mu_d2 = data["mu_d2"]
        self.fisher_separation = data["fisher_separation"]

        self.gpr = GaussianPhaseRouter(alpha=alpha)
        self._anchors_gpu = None

    def _get_gpu_anchors(self, device):
        if self._anchors_gpu is None or self._anchors_gpu[0].device != device:
            self._anchors_gpu = (
                self.anchor_d1.to(device=device, dtype=torch.float32),
                self.anchor_d2.to(device=device, dtype=torch.float32)
            )
        return self._anchors_gpu

    def compute_phase_divergence(self, model, inputs) -> Tuple[float, float, float]:
        """
        Runs prompt prefill through model, probes hidden states at calibrated layer,
        and computes relative algebraic divergence z_i = S_D2 - S_D1.
        """
        device = model.device
        c1, c2 = self._get_gpu_anchors(device)

        # Snapshot active adapters and scalings to prevent PEFT disable_adapter() reset
        active_map = {}
        scaling_map = {}
        for name, mod in model.named_modules():
            if hasattr(mod, "_active_adapter"):
                active_map[name] = getattr(mod, "_active_adapter", None)
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                scaling_map[name] = dict(mod.scaling)

        try:
            with torch.no_grad():
                if hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        out = model(**inputs, output_hidden_states=True)
                else:
                    out = model(**inputs, output_hidden_states=True)

                # out.hidden_states[layer_idx] shape: [1, seq_len, 4096]
                h = out.hidden_states[self.layer_idx][0]
                # Mean pooling over prompt tokens
                h_mean = h.mean(dim=0).to(dtype=torch.float32)
                u = F.normalize(h_mean, p=2, dim=0)

                # Cosine / Phase projection
                s_d1 = torch.dot(u, c1).item()
                s_d2 = torch.dot(u, c2).item()
                z_i = s_d2 - s_d1
        finally:
            # Reliably restore multi-adapter state across all LoraLayers
            for name, mod in model.named_modules():
                if name in active_map and active_map[name] is not None:
                    mod._active_adapter = active_map[name]
                if name in scaling_map and hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                    mod.scaling.update(scaling_map[name])

        return z_i, s_d1, s_d2

    def route(self, model, inputs) -> Tuple[str, float, Dict[str, Any]]:
        """
        Performs representation-level GPR routing.
        Returns:
            adapter_name: "expert_abstract_algebra" or "expert_general_math"
            z_i: float relative divergence
            metadata: dict of routing diagnostics
        """
        z_i, s_d1, s_d2 = self.compute_phase_divergence(model, inputs)
        z_score, dev = self.gpr.update(z_i)

        if z_i >= self.threshold:
            selected_adapter = "expert_abstract_algebra"
            confidence = min(1.0, max(0.5, 0.5 + (z_i - self.threshold) / (2.0 * abs(self.mu_d2 - self.threshold) + 1e-8)))
        else:
            selected_adapter = "expert_general_math"
            confidence = min(1.0, max(0.5, 0.5 + (self.threshold - z_i) / (2.0 * abs(self.threshold - self.mu_d1) + 1e-8)))

        metadata = {
            "z_i": z_i,
            "z_score": z_score,
            "sim_d1": s_d1,
            "sim_d2": s_d2,
            "confidence": confidence,
            "threshold": self.threshold,
            "layer": self.layer_idx
        }

        return selected_adapter, z_i, metadata
