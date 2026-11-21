"""
SODALS v2.1 — Module 12.2: Soft LoRA Blender (Continuous Alpha Mixing)
=====================================================================
Implements continuous, soft multi-adapter blending in PEFT.
Allows multiple LoRA adapters to be active concurrently in each LoraLayer,
modulating their scaling coefficients dynamically based on the GPR phase signal:

  Delta W(x) = (1 - alpha) * s_D1 * (B_1 A_1 x) + alpha * s_D2 * (B_2 A_2 x)

Where:
  - alpha = sigmoid((z_i - tau) / T) in [0, 1]
  - s_D1 = 0.25 (optimal scale for expert_general_math)
  - s_D2 = 1.00 (optimal scale for expert_abstract_algebra)
  - rep_penalty = (1 - alpha) * 1.05 + alpha * 1.00 (smooth continuous interpolation)
"""

import math
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from peft.tuners.lora.layer import LoraLayer

DEFAULT_BASE_SCALES = {
    "expert_general_math": 0.25,
    "expert_abstract_algebra": 1.00
}


class SoftLoraBlender:
    """
    Manages concurrent multi-adapter activation and continuous scaling in PEFT.
    """
    def __init__(
        self,
        adapter_names: List[str] = ["expert_general_math", "expert_abstract_algebra"],
        base_scales: Dict[str, float] = DEFAULT_BASE_SCALES,
        threshold: float = 0.004913,
        temperature: float = 0.03
    ):
        self.adapter_names = list(adapter_names)
        self.base_scales = dict(base_scales)
        self.threshold = threshold
        self.temperature = temperature
        self._enabled = False
        self._cached_model_id: Optional[int] = None
        self._cached_lora_modules: List[Any] = []

    def _get_lora_modules(self, model) -> List[Any]:
        mid = id(model)
        if self._cached_model_id == mid and self._cached_lora_modules:
            return self._cached_lora_modules
        mods = [
            mod
            for mod in model.modules()
            if isinstance(mod, LoraLayer)
            or (hasattr(mod, "scaling") and hasattr(mod, "lora_alpha") and hasattr(mod, "r"))
        ]
        self._cached_model_id = mid
        self._cached_lora_modules = mods
        return mods

    @staticmethod
    def _supported_adapters_for_module(mod: Any, requested: List[str]) -> List[str]:
        if hasattr(mod, "lora_A") and isinstance(mod.lora_A, dict):
            return [ad for ad in requested if ad in mod.lora_A]
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            return [ad for ad in requested if ad in mod.scaling]
        return list(requested)

    def enable_multi_adapter(self, model) -> int:
        """
        Activates all managed adapters simultaneously across every LoraLayer in the model.
        Returns the count of configured LoraLayers.
        """
        count = 0
        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter(self.adapter_names)
            except Exception:
                pass
        if hasattr(model, "active_adapters"):
            try:
                model.active_adapters = list(self.adapter_names)
            except Exception:
                pass

        for mod in self._get_lora_modules(model):
            supported = self._supported_adapters_for_module(mod, self.adapter_names)
            if not supported:
                continue
            if hasattr(mod, "set_adapter"):
                mod.set_adapter(supported)
            elif hasattr(mod, "_active_adapter"):
                mod._active_adapter = list(supported)
            count += 1
        self._enabled = True
        return count

    def compute_alpha(self, z_i: float) -> float:
        """
        Maps the continuous GPR phase divergence z_i to mixing ratio alpha in (0, 1)
        using temperature-scaled sigmoid centered at calibrated threshold tau.
        """
        scaled_diff = (z_i - self.threshold) / max(1e-5, self.temperature)
        # Numerical clamp to avoid overflow
        scaled_diff = max(-50.0, min(50.0, scaled_diff))
        alpha = 1.0 / (1.0 + math.exp(-scaled_diff))
        return alpha

    def compute_rep_penalty(self, alpha: float) -> float:
        """
        Smoothly interpolates repetition penalty between general math (1.05) and algebra (1.00).
        """
        return (1.0 - alpha) * 1.05 + alpha * 1.00

    def set_blend(self, model, alpha: Union[float, torch.Tensor]) -> Dict[str, Any]:
        """
        Modulates the scaling factors across all LoraLayers for the given alpha.
          expert_general_math:     (1 - alpha) * s_D1
          expert_abstract_algebra: alpha * s_D2
        If alpha is a torch.Tensor (with requires_grad=True), scaling factors are set as
        autograd Tensors, preserving the backward computation graph from LM loss to gate.
        Returns the effective scale mapping.
        """
        if isinstance(alpha, torch.Tensor):
            alpha_clamped = torch.clamp(alpha, 1e-7, 1.0 - 1e-7)
            s_d1 = (1.0 - alpha_clamped) * self.base_scales.get("expert_general_math", 0.25)
            s_d2 = alpha_clamped * self.base_scales.get("expert_abstract_algebra", 1.00)
            if s_d1.ndim == 1:
                # Shape [B] -> [B, 1, 1] to broadcast against [B, seq_len, hidden_dim]
                s_d1 = s_d1.view(-1, 1, 1)
                s_d2 = s_d2.view(-1, 1, 1)
        else:
            alpha_f = max(0.0, min(1.0, float(alpha)))
            s_d1 = (1.0 - alpha_f) * self.base_scales.get("expert_general_math", 0.25)
            s_d2 = alpha_f * self.base_scales.get("expert_abstract_algebra", 1.00)

        scale_map: Dict[str, Any] = {
            "expert_general_math": s_d1,
            "expert_abstract_algebra": s_d2,
        }
        for extra_name, extra_scale in self.base_scales.items():
            if extra_name not in scale_map:
                scale_map[extra_name] = float(extra_scale)

        for mod in self._get_lora_modules(model):
            supported = self._supported_adapters_for_module(mod, self.adapter_names)
            if supported and hasattr(mod, "_active_adapter"):
                if isinstance(mod._active_adapter, list):
                    for ad in supported:
                        if ad not in mod._active_adapter:
                            mod._active_adapter.append(ad)
                elif isinstance(mod._active_adapter, str) and mod._active_adapter in supported:
                    if hasattr(mod, "set_adapter"):
                        mod.set_adapter(supported)

            if hasattr(mod, "scaling") and hasattr(mod, "lora_alpha") and hasattr(mod, "r"):
                for ad_name, eff_scale in scale_map.items():
                    if ad_name in mod.scaling and ad_name in mod.lora_alpha and ad_name in mod.r:
                        alpha_val = mod.lora_alpha[ad_name]
                        r_val = mod.r[ad_name]
                        mod.scaling[ad_name] = (alpha_val / r_val) * eff_scale

        return scale_map

    def set_multi_cartridge_blend(
        self,
        model,
        slot_weights: Dict[str, Union[float, torch.Tensor]],
        managed_slots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        N-Cartridge generalization of SoftLoraBlender for `.qmem v3` Top-K virtual parameter slots.
        Applies exact per-slot scaling weights while strictly preserving auxiliary adapters
        such as `svon_identity` (fixed scale 1.00) and zeroing inactive managed cartridge slots.
        """
        slots_to_manage = list(managed_slots) if managed_slots is not None else [
            ad for ad in self.adapter_names if ad.startswith("qmem_slot_")
        ]
        all_requested = list(self.adapter_names)
        for s_name in list(slot_weights.keys()) + slots_to_manage:
            if s_name not in all_requested:
                all_requested.append(s_name)

        scale_map: Dict[str, Any] = {}
        # 1. Preserve non-slot auxiliary adapters (e.g., svon_identity) at their fixed base_scales
        for ad_name, base_sc in self.base_scales.items():
            if ad_name not in slots_to_manage and ad_name not in slot_weights:
                scale_map[ad_name] = float(base_sc)

        # 2. Zero inactive managed slots, then apply active Top-K weights * base_scale
        for s_name in slots_to_manage:
            scale_map[s_name] = 0.0
        for s_name, w in slot_weights.items():
            base_s = float(self.base_scales.get(s_name, 1.0))
            if isinstance(w, torch.Tensor):
                w_clamped = torch.clamp(w, 0.0, 1.0) * base_s
                if w_clamped.ndim == 1:
                    w_clamped = w_clamped.view(-1, 1, 1)
                scale_map[s_name] = w_clamped
            else:
                scale_map[s_name] = max(0.0, min(1.0, float(w))) * base_s

        for mod in self._get_lora_modules(model):
            supported = self._supported_adapters_for_module(mod, all_requested)
            if not supported:
                continue
            if hasattr(mod, "set_adapter"):
                try:
                    mod.set_adapter(supported)
                except Exception:
                    pass
            elif hasattr(mod, "_active_adapter"):
                mod._active_adapter = list(supported)

            if hasattr(mod, "scaling") and hasattr(mod, "lora_alpha") and hasattr(mod, "r"):
                for ad_name, eff_scale in scale_map.items():
                    if ad_name in mod.scaling and ad_name in mod.lora_alpha and ad_name in mod.r:
                        alpha_val = mod.lora_alpha[ad_name]
                        r_val = mod.r[ad_name]
                        mod.scaling[ad_name] = (alpha_val / r_val) * eff_scale

        return scale_map

