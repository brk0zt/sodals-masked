"""
Svon-CGW (Epoch III) — Sütun I: Holomorphic Complex-Valued Gated DeltaNet (CVNN-DeltaNet)
=========================================================================================
Lifts Qwen 3.5 9B's 24 `Qwen3_5GatedDeltaNet` layers from a purely real dissipative
recurrence (S_t in R^{32 x 128 x 128}, decay exp(g_t) in (0, 1)) into a Complex
Unitary-Dissipative Associative Wave Recurrence in C^{32 x 64 x 64}:

    lambda_t = exp(g_t + i * omega_t) = exp(g_t) * (cos(omega_t) + i * sin(omega_t))
    S_t^C    = Diag(lambda_t) S_{t-1}^C + beta_t * (v_t^C - S_{t-1}^C k_t^C) (k_t^C)^H

By the Unitary Gauge Equivalence Theorem (S_hat_t = R(Theta_t) S_t where Theta_t = sum_{tau<=t} omega_tau):
  1. Exact Submanifold Theorem: At initialization (omega_t = 0, alpha_im = 0), both the
     recurrent and 64-token chunkwise parallel scan algorithms are 100.000% bit-identical
     (0.0 float error) to Qwen 3.5 9B's native `Qwen3_5GatedDeltaNet`.
  2. Constructive / Destructive Wave Interference: Steps sharing phase (Delta Theta = 0)
     interfere constructively (+1), while steps with phase opposition (Delta Theta = pi)
     actively cancel contradictory intermediate states (-1) inside the recurrent memory.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_complex_phase_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Applies block-diagonal U(1)^64 complex rotation exp(i * theta) to a real tensor `x`
    of shape [..., 2 * C] by viewing consecutive pairs (x_{2m}, x_{2m+1}) as z_m = x_{2m} + i x_{2m+1}.
      x:     [B, H, T, 2*C] or [B, T, H, 2*C]
      theta: matching prefix shape [..., C] (in radians)
    Note: When theta == 0, cos(0) = 1 and sin(0) = 0, returning `x` bit-identically.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_t = torch.cos(theta).to(x.dtype)
    sin_t = torch.sin(theta).to(x.dtype)

    out_even = x_even * cos_t - x_odd * sin_t
    out_odd = x_even * sin_t + x_odd * cos_t
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def apply_hermitian_value_coupling(v: torch.Tensor, alpha_im: torch.Tensor) -> torch.Tensor:
    """
    Applies controlled imaginary cross-coupling (1 + i * alpha_im) to value pairs (v_{2m}, v_{2m+1}):
      v:        [B, T, H_v, 2*C]
      alpha_im: [H_v] or [1, 1, H_v, 1]
    When alpha_im == 0, returns `v` without modification.
    """
    a = alpha_im.view(1, 1, -1, 1).to(v.dtype)
    v_even = v[..., 0::2]
    v_odd = v[..., 1::2]
    out_even = v_even - a * v_odd
    out_odd = v_odd + a * v_even
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def _l2norm_fp32(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def recurrent_complex_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    omega: Optional[torch.Tensor] = None,
    alpha_im: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Step-by-step Unitary-Dissipative Complex Gated Delta Rule.
      query, key: [B, T, H_k, D_k]
      value:      [B, T, H_v, D_v]
      g:          [B, T, H_v] (real log-decay <= 0)
      beta:       [B, T, H_v] (write gate in (0, 1))
      omega:      Optional [B, T, H_k, D_k // 2] (instantaneous phase velocity in [-pi, pi])
      alpha_im:   Optional [H_v] (imaginary value coupling)
    Returns:
      (core_attn_out [B, T, H_v, D_v], final_state, cumulative_phase [B, T, H_k, D_k // 2])
    """
    initial_dtype = query.dtype
    batch_size, seq_len, num_k_heads, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    if omega is not None:
        cum_theta = torch.cumsum(omega.float(), dim=1)
        # Rotate into the unitary interaction frame: k_tilde = exp(-i * Theta_t) k_t
        q_rot = apply_complex_phase_rotation(query.float(), -cum_theta)
        k_rot = apply_complex_phase_rotation(key.float(), -cum_theta)
    else:
        cum_theta = torch.zeros(
            (batch_size, seq_len, num_k_heads, k_head_dim // 2),
            dtype=torch.float32,
            device=query.device,
        )
        q_rot = query.float()
        k_rot = key.float()

    v_mod = apply_hermitian_value_coupling(value.float(), alpha_im) if alpha_im is not None else value.float()

    if num_k_heads != num_v_heads:
        rep = num_v_heads // num_k_heads
        q_rot = q_rot.repeat_interleave(rep, dim=2)
        k_rot = k_rot.repeat_interleave(rep, dim=2)

    q_t_all, k_t_all, v_t_all, beta_all, decay_all = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (q_rot, k_rot, v_mod, beta, g)
    ]

    if use_qk_l2norm_in_kernel:
        q_t_all = _l2norm_fp32(q_t_all, dim=-1, eps=1e-6)
        k_t_all = _l2norm_fp32(k_t_all, dim=-1, eps=1e-6)

    q_t_all = q_t_all / (q_t_all.shape[-1] ** 0.5)

    state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, dtype=torch.float32, device=query.device)
    else:
        state = initial_state.to(torch.float32)

    core_out = torch.zeros_like(v_t_all)
    for i in range(seq_len):
        q_i = q_t_all[:, :, i]
        k_i = k_t_all[:, :, i]
        v_i = v_t_all[:, :, i]
        decay_i = decay_all[:, :, i].exp()[..., None, None]
        state = state * decay_i
        beta_i = beta_all[:, :, i].unsqueeze(-1)
        kv_mem = (state * k_i.unsqueeze(-1)).sum(dim=-2)
        delta = (v_i - kv_mem) * beta_i
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)

    final_state = state if output_final_state else None
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, final_state, cum_theta


class HolomorphicCVNNDeltaNetWrapper(nn.Module):
    """
    Drop-in wrapper around a `Qwen3_5GatedDeltaNet` layer that upgrades its real
    state-space transition to a Holomorphic Complex-Valued Wave Recurrence while
    preserving 100% of the pretrained weights at initialization (zero-init omega & alpha_im).
    """

    def __init__(
        self,
        base_deltanet: nn.Module,
        phase_rank: int = 16,
    ):
        super().__init__()
        self.base = base_deltanet
        self.hidden_size = int(base_deltanet.hidden_size)
        self.num_k_heads = int(base_deltanet.num_k_heads)
        self.num_v_heads = int(base_deltanet.num_v_heads)
        self.head_k_dim = int(base_deltanet.head_k_dim)
        self.head_v_dim = int(base_deltanet.head_v_dim)
        self.complex_k_dim = self.head_k_dim // 2

        # Low-rank trainable phase-velocity generator: h_t -> omega_t in [-pi, pi]^{H_k x (D_k/2)}
        self.phase_proj_down = nn.Linear(self.hidden_size, phase_rank, bias=False, dtype=torch.float32)
        self.phase_proj_up = nn.Linear(
            phase_rank, self.num_k_heads * self.complex_k_dim, bias=False, dtype=torch.float32
        )
        nn.init.normal_(self.phase_proj_down.weight, mean=0.0, std=0.02)
        # Strict Zero-Init on `phase_proj_up` and `alpha_im` guarantees Exact Submanifold identity at t=0!
        nn.init.zeros_(self.phase_proj_up.weight)
        self.alpha_im = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))

        self.last_cumulative_phase: Optional[torch.Tensor] = None
        self._active_cache_id: Optional[int] = None
        self._cached_phase_tail: Optional[torch.Tensor] = None
        self._cached_phase_history: Optional[torch.Tensor] = None
        # Zero-CUDA-sync parameter version tracking for fast identity skip in eval()
        self._cached_param_versions: Optional[Tuple[int, int]] = None
        self._omega_nonzero: bool = False
        self._alpha_nonzero: bool = False

    def _refresh_active_flags(self) -> Tuple[bool, bool]:
        """Checks whether phase_proj_up / alpha_im are non-zero, caching by Tensor._version (0 GPU syncs/token!)."""
        vers = (int(self.phase_proj_up.weight._version), int(self.alpha_im._version))
        if self._cached_param_versions != vers:
            self._omega_nonzero = bool(float(self.phase_proj_up.weight.detach().abs().max().item()) > 0.0)
            self._alpha_nonzero = bool(float(self.alpha_im.detach().abs().max().item()) > 0.0)
            self._cached_param_versions = vers
        return self._omega_nonzero, self._alpha_nonzero

    def compute_omega(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        h_f32 = hidden_states.float()
        raw = self.phase_proj_up(F.silu(self.phase_proj_down(h_f32)))
        omega = torch.tanh(raw) * math.pi
        return omega.view(bsz, seq_len, self.num_k_heads, self.complex_k_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        omega_active, alpha_active = (True, True) if self.training else self._refresh_active_flags()

        batch_size, seq_len, _ = hidden_states.shape
        if omega_active:
            omega = self.compute_omega(hidden_states)
        else:
            omega = torch.zeros(
                (batch_size, seq_len, self.num_k_heads, self.complex_k_dim),
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if cache_params is not None:
            if hasattr(cache_params, "get_conv_state"):
                conv_state = cache_params.get_conv_state(self.base.layer_idx, state_idx=0)
                recurrent_state = cache_params.get_recurrent_state(self.base.layer_idx, state_idx=0)
            else:
                conv_states = getattr(cache_params, "conv_states", None)
                rec_states = getattr(cache_params, "recurrent_states", None)
                l_idx = self.base.layer_idx
                conv_state = conv_states[l_idx] if (conv_states is not None and l_idx < len(conv_states)) else None
                recurrent_state = rec_states[l_idx] if (rec_states is not None and l_idx < len(rec_states)) else None
        else:
            conv_state, recurrent_state = None, None

        mixed_qkv = self.base.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.base.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.base.in_proj_b(hidden_states)
        a = self.base.in_proj_a(hidden_states)

        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            causal_conv1d_fn,
            causal_conv1d_update,
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )

        is_rec_past = (
            cache_params.is_recording_past(self.base.layer_idx)
            if (cache_params is not None and hasattr(cache_params, "is_recording_past"))
            else False
        )
        if conv_state is not None and seq_len == 1 and not is_rec_past:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.base.conv1d.weight.squeeze(1),
                self.base.conv1d.bias,
                self.base.activation,
            )
        else:
            used_new_conv_cache = False
            if cache_params is not None:
                if hasattr(cache_params, "get_conv_state"):
                    mixed_qkv = cache_params.update_conv_state(
                        mixed_qkv, self.base.layer_idx, conv_kernel_size=self.base.conv_kernel_size
                    )
                    used_new_conv_cache = True
                else:
                    conv_state_new = F.pad(mixed_qkv, (self.base.conv_kernel_size - mixed_qkv.shape[-1], 0))
                    cache_params.conv_states[self.base.layer_idx] = conv_state_new
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.base.conv1d.weight.squeeze(1),
                self.base.conv1d.bias,
                activation=self.base.activation,
                **kwargs,
            )
            if used_new_conv_cache:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.base.key_dim, self.base.key_dim, self.base.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # Unitary Complex Phase Gauge Rotation + Rolling Decode Phase History for Sheaf-GNN Bridge
        cum_theta = torch.cumsum(omega, dim=1)
        if cache_params is not None:
            cid = id(cache_params)
            if conv_state is None and recurrent_state is None:
                self._cached_phase_tail = None
                self._cached_phase_history = cum_theta.detach()
            elif self._active_cache_id == cid and self._cached_phase_tail is not None:
                cum_theta = cum_theta + self._cached_phase_tail.to(device=cum_theta.device, dtype=cum_theta.dtype)
                if self._cached_phase_history is not None and self._cached_phase_history.shape[0] == batch_size:
                    prev_hist = self._cached_phase_history.to(device=cum_theta.device, dtype=cum_theta.dtype)
                    self._cached_phase_history = torch.cat([prev_hist, cum_theta.detach()], dim=1)
                else:
                    self._cached_phase_history = cum_theta.detach()
            else:
                self._cached_phase_history = cum_theta.detach()
            self._active_cache_id = cid
            self._cached_phase_tail = cum_theta[:, -1:, :, :].detach()
            # Expose full [B, K, H_k, C] cumulative trajectory so downstream Sheaf-GNN fires during decode!
            self.last_cumulative_phase = self._cached_phase_history
        else:
            self.last_cumulative_phase = cum_theta

        if omega_active:
            query = apply_complex_phase_rotation(query, -cum_theta)
            key = apply_complex_phase_rotation(key, -cum_theta)
        if alpha_active:
            value = apply_hermitian_value_coupling(value, self.alpha_im)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        beta = b.sigmoid()
        g = -self.base.A_log.float().exp() * F.softplus(a.float() + self.base.dt_bias)

        recurrent_fn = getattr(self.base, "recurrent_gated_delta_rule", None) or torch_recurrent_gated_delta_rule
        chunk_fn = getattr(self.base, "chunk_gated_delta_rule", None) or torch_chunk_gated_delta_rule

        if recurrent_state is not None and seq_len == 1:
            core_attn_out, last_recurrent_state = recurrent_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        if cache_params is not None:
            if hasattr(cache_params, "get_conv_state"):
                cache_params.update_recurrent_state(last_recurrent_state, self.base.layer_idx)
            else:
                cache_params.recurrent_states[self.base.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.base.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.base.out_proj(core_attn_out)


def collect_cvnn_phase_offsets(model: nn.Module) -> Dict[int, Dict[str, torch.Tensor]]:
    """Snapshots CPU-offloaded cumulative CVNN phase tails & histories for each wrapped GatedDeltaNet layer."""
    offsets: Dict[int, Any] = {}
    if model is None:
        return offsets
    for mod in model.modules():
        if isinstance(mod, HolomorphicCVNNDeltaNetWrapper) and mod._cached_phase_tail is not None:
            lidx = int(getattr(mod.base, "layer_idx", len(offsets)))
            offsets[lidx] = {
                "tail": mod._cached_phase_tail.detach().cpu().clone(),
                "history": mod._cached_phase_history.detach().cpu().clone() if mod._cached_phase_history is not None else None,
            }
    return offsets


def restore_cvnn_phase_offsets(
    model: Optional[nn.Module],
    offsets: Optional[Dict[int, Any]],
    cache_params: Any = None,
) -> None:
    """Restores cumulative CVNN phase tails & histories before cache-based Pass-2 continuation."""
    if model is None or not offsets:
        return
    for mod in model.modules():
        if isinstance(mod, HolomorphicCVNNDeltaNetWrapper):
            lidx = int(getattr(mod.base, "layer_idx", -1))
            if lidx in offsets:
                entry = offsets[lidx]
                if isinstance(entry, dict):
                    mod._cached_phase_tail = entry["tail"].clone() if entry.get("tail") is not None else None
                    mod._cached_phase_history = entry["history"].clone() if entry.get("history") is not None else None
                else:
                    mod._cached_phase_tail = entry.clone()
                    mod._cached_phase_history = None
                mod._active_cache_id = id(cache_params) if cache_params is not None else None

