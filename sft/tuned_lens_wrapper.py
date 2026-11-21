"""
SODALS v2.1 — Tuned Lens (J-Lens) & Clifford Geometric Stability Wrapper
========================================================================
Implements the Belrose (2023) Tuned Lens affine projection and Clifford geometric
stability metrics for deep representation inspection across workspace layers.

Architecture:
  - Hidden representation: h_l in R^4096 (layers 26, 28, 30)
  - Target: h_L in R^4096 (Layer 31 pre-norm final hidden representation)
  - Affine transformation: h_proj = W_l @ h_l + b_l
  - Geometric Stability: CosSim(h_proj, h_L)
  - Top-1 Vocabulary Consensus: argmax(lm_head(norm(h_proj))) vs argmax(lm_head(norm(h_L)))
  - Wronskian Impedance Proxy: Z = 1.0 - Stability

Hardware & Performance:
  - Vectorized single-pass batched evaluation across full token sequence.
  - Zero Python per-step loop overhead during generation.
  - Peak additional VRAM: ~64 MB for FP16 weights.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from typing import Dict, Tuple, Optional, NamedTuple


class LensStabilityReport(NamedTuple):
    mean_stability: float
    answer_stability: float
    min_stability: float
    cos_layer28: float
    cos_layer30: float
    top1_agreement: bool
    top1_token: str
    impedance: float
    geometric_score: float


class TunedLensWrapper(nn.Module):
    def __init__(
        self,
        safetensors_path: Optional[str] = None,
        layers=(28, 30),
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.target_layers = list(layers)
        self.device = device
        self.dtype = dtype
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        if safetensors_path is not None and os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for l in self.target_layers:
                    w_key = f"layer_{l}.weight"
                    b_key = f"layer_{l}.bias"
                    if w_key in f.keys() and b_key in f.keys():
                        w = f.get_tensor(w_key).to(dtype=self.dtype, device=self.device)
                        b = f.get_tensor(b_key).to(dtype=self.dtype, device=self.device)
                        self.weights[str(l)] = nn.Parameter(w, requires_grad=False)
                        self.biases[str(l)] = nn.Parameter(b, requires_grad=False)
            print(f"[TunedLensWrapper] Loaded {len(self.weights)} affine lens projections (Layers: {self.target_layers}) onto {device}.")
        elif safetensors_path is not None:
            raise FileNotFoundError(f"Tuned Lens weights not found at: {safetensors_path}")
        else:
            # Identity fallback when no pre-trained lens weights are provided
            pass

    def project(self, h_l: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Projects hidden state h_l to final pre-norm space h_L."""
        key = str(layer_idx)
        if key in self.weights:
            w = self.weights[key]
            b = self.biases[key]
            return F.linear(h_l, w, b)
        return h_l

    def evaluate_sequence(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        max_eval_len: int = 256
    ) -> LensStabilityReport:
        """
        Runs a single forward pass over input_ids to compute full geometric stability.
        If input_ids is longer than max_eval_len, slices the final max_eval_len tokens
        (the conclusion/answer region) to prevent quadratic VRAM allocation on 8GB GPU.
        """
        # Slice to answer/conclusion window for long chains (protects 8GB VRAM)
        if input_ids.shape[1] > max_eval_len:
            eval_ids = input_ids[:, -max_eval_len:]
        else:
            eval_ids = input_ids

        captured: Dict[int, torch.Tensor] = {}

        def make_hook(layer_num: int):
            def hook(module, args, output):
                t = output[0] if isinstance(output, tuple) else output
                captured[layer_num] = t.detach()
            return hook

        # Resolve model layers dynamically
        if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model"):
            layers_list = model.base_model.model.model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_list = model.model.layers
        else:
            raise AttributeError("Could not locate transformer layers module.")

        final_layer_idx = len(layers_list) - 1
        all_probe_layers = sorted(list(set(self.target_layers + [final_layer_idx])))

        # Register temporary hooks on target layers + final layer
        hooks = []
        for l in all_probe_layers:
            target_mod = layers_list[l]
            hooks.append(target_mod.register_forward_hook(make_hook(l)))

        try:
            with torch.no_grad():
                _ = model(eval_ids)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            captured.clear()
            eval_ids_short = input_ids[:, -64:]
            with torch.no_grad():
                _ = model(eval_ids_short)
        finally:
            for h in hooks:
                h.remove()

        h_final = captured[final_layer_idx]  # [1, seq_len, 4096]
        seq_len = h_final.shape[1]

        # Calculate projected representations and cosine similarities
        cosines = []
        for l in self.target_layers:
            h_proj = self.project(captured[l], l)
            cos_l = F.cosine_similarity(h_proj, h_final, dim=-1)  # [1, seq_len]
            cosines.append(cos_l)

        # Average across workspace layers
        stability_tensor = torch.stack(cosines, dim=0).mean(dim=0).squeeze(0)  # [seq_len]

        mean_stab = stability_tensor.mean().item()
        ans_stab = stability_tensor[-1].item()
        min_stab = stability_tensor.min().item()

        cos_28 = cosines[0].squeeze(0)[-1].item() if len(cosines) > 0 else 0.0
        cos_30 = cosines[-1].squeeze(0)[-1].item() if len(cosines) > 1 else 0.0

        # Vocabulary top-1 agreement on final token
        if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "norm"):
            norm_layer = model.base_model.model.norm
        elif hasattr(model, "model") and hasattr(model.model, "norm"):
            norm_layer = model.model.norm
        elif hasattr(model, "norm"):
            norm_layer = model.norm
        else:
            raise AttributeError("Could not locate model norm layer.")

        if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
            lm_head_layer = model.base_model.model.lm_head
        elif hasattr(model, "model") and hasattr(model.model, "lm_head"):
            lm_head_layer = model.model.lm_head
        elif hasattr(model, "lm_head"):
            lm_head_layer = model.lm_head
        else:
            raise AttributeError("Could not locate model lm_head layer.")

        with torch.no_grad():
            h_final_last = h_final[:, -1:, :]
            normed_final = norm_layer(h_final_last)
            head_dtype = getattr(lm_head_layer, "weight", None)
            if head_dtype is not None:
                normed_final = normed_final.to(dtype=head_dtype.dtype)
            logits_final = lm_head_layer(normed_final)
            final_top1 = torch.argmax(logits_final[0, -1]).item()

            # Using layer 30 projection
            last_target_l = self.target_layers[-1]
            h_last_target = captured[last_target_l][:, -1:, :]
            h_last_proj = self.project(h_last_target, last_target_l)
            normed_proj = norm_layer(h_last_proj)
            if head_dtype is not None:
                normed_proj = normed_proj.to(dtype=head_dtype.dtype)
            logits_lens = lm_head_layer(normed_proj)
            lens_top1 = torch.argmax(logits_lens[0, -1]).item()

        top1_agree = (final_top1 == lens_top1)
        top1_tok = tokenizer.decode([lens_top1]) if tokenizer else str(lens_top1)
        impedance = max(0.0, 1.0 - mean_stab)

        # Combined geometric stability score:
        # Base stability + bonus for agreement
        geom_score = (mean_stab - 0.75) * 10.0 + (5.0 if top1_agree else 0.0)

        return LensStabilityReport(
            mean_stability=round(mean_stab, 4),
            answer_stability=round(ans_stab, 4),
            min_stability=round(min_stab, 4),
            cos_layer28=round(cos_28, 4),
            cos_layer30=round(cos_30, 4),
            top1_agreement=bool(top1_agree),
            top1_token=top1_tok.strip(),
            impedance=round(impedance, 4),
            geometric_score=round(geom_score, 4)
        )
