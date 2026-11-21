"""
Svon-CGW (Epoch III) — Sütun II: 2-Hop Relational Sheaf-GNN Attention
=====================================================================
Transforms Qwen 3.5 9B's 8 full-attention layers (`3, 7, 11, 15, 19, 23, 27, 31`)
into a Relational Sheaf & 2-Hop Message-Passing Graph Neural Network:
  1. Exploits Qwen 3.5's `partial_rotary_factor = 0.25` (`head_dim = 256` -> only
     dims `[0:64]` carry 1D RoPE; dims `[64:256]` are 192 position-free graph node channels!).
  2. Constructs a causal dual adjacency operator:
       - Phase Coherence Matrix C_{uv} = mean_d cos(phi_{u,d} - phi_{v,d}) from the
         preceding HolomorphicCVNNDeltaNet layer.
       - Symbolic Equation / Entity Edge Matrix E_{sym}(u, v) and Dead-Branch Pruning Mask M_{dead}.
  3. Performs 2-Hop Transitive Graph Message Diffusion:
       M_{GNN} = (I + tanh(alpha_hop) * P_graph) @ P @ V
     specifically routed through the 192 position-free channels `[64:256]`, so that at
     initialization (alpha_hop = 0, gamma_phase = 0, eta_sym = 0, kappa_dead = 0)
     the layer is 100.000% bit-identical to `Qwen3_5Attention`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_phase_coherence_matrix(phase: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise causal complex phase coherence C_{uv} in [-1, 1]:
      phase: [B, T, H_k, C] (in radians)
    Returns:
      coherence: [B, H_k, T, T] where (u, v) entry is mean_c cos(phase[u,c] - phase[v,c]),
      masked strictly causally (v <= u).
    """
    # Using trigonometric identity: cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
    # Avoids O(T^2 * C) memory expansion by using fast matrix multiplication!
    p = phase.float().transpose(1, 2)  # [B, H_k, T, C]
    c_dim = max(1, p.shape[-1])
    cos_p = torch.cos(p)
    sin_p = torch.sin(p)
    coh = (cos_p @ cos_p.transpose(-1, -2) + sin_p @ sin_p.transpose(-1, -2)) / float(c_dim)
    seq_len = p.shape[2]
    upper_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=p.device).triu(1)
    return coh.masked_fill(upper_mask, 0.0)


def two_hop_sheaf_message_passing(
    attn_probs: torch.Tensor,
    values: torch.Tensor,
    alpha_hop: torch.Tensor,
    symbolic_adj: Optional[torch.Tensor] = None,
    ro_free_start_dim: int = 64,
    keep_graph: bool = False,
    one_hop_history: Optional[torch.Tensor] = None,
    one_hop: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies 2-hop transitive graph diffusion on the position-free semantic channels
    (`[ro_free_start_dim : head_dim]`, i.e., 192 channels in Qwen 3.5):
      attn_probs:      [B, H, Q, K] (causal 1-hop attention distribution P)
      values:          [B, H, K, D]
      alpha_hop:       [H] (learnable per-head 2-hop diffusion gain, zero-init)
      symbolic_adj:    Optional [B, 1, Q, K] or [B, H, Q, K] sparse symbolic graph
      one_hop_history: Optional [B, H, K, D] cached 1-hop states for KV-cached decode (Q < K)
      one_hop:         Optional precomputed [B, H, Q, D] `attn_probs @ values`
    Returns:
      out: [B, H, Q, D]
    When alpha_hop == 0 and `keep_graph=False`, returns `attn_probs @ values` bit-identically.
    """
    if one_hop is None:
        one_hop = attn_probs @ values  # [B, H, Q, D]
    if not keep_graph and float(alpha_hop.abs().max().item()) == 0.0:
        return one_hop

    q_len, k_len = attn_probs.shape[-2], attn_probs.shape[-1]
    if one_hop_history is not None and one_hop_history.shape[-2] == k_len:
        msg_source = one_hop_history.to(dtype=one_hop.dtype, device=one_hop.device)
    elif q_len == k_len:
        msg_source = one_hop
    else:
        # Cached autoregressive decode without full 1-hop history: safely stay on 1-hop
        return one_hop

    # Construct row-normalized causal graph propagation operator P_graph
    if symbolic_adj is not None:
        causal = torch.ones((q_len, k_len), dtype=attn_probs.dtype, device=attn_probs.device).tril(k_len - q_len)
        raw_g = (attn_probs + symbolic_adj.to(device=attn_probs.device, dtype=attn_probs.dtype).clamp(min=0.0)) * causal
        p_graph = raw_g / raw_g.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    else:
        p_graph = attn_probs

    # 2-hop transitive message: P_graph @ M_{1-hop} in [B, H, Q, D]
    two_hop_msg = p_graph @ msg_source
    gain = torch.tanh(alpha_hop).view(1, -1, 1, 1).to(device=values.device, dtype=values.dtype)

    if ro_free_start_dim <= 0:
        return one_hop + gain * two_hop_msg

    # Preserve 1D RoPE channels [0:ro_free_start_dim] untouched; diffuse on 192 graph channels!
    out = one_hop.clone()
    out[..., ro_free_start_dim:] = one_hop[..., ro_free_start_dim:] + gain * two_hop_msg[..., ro_free_start_dim:]
    return out


class RelationalSheafGNNAttention(nn.Module):
    """
    Drop-in wrapper for Qwen 3.5 `Qwen3_5Attention` layers (`3, 7, 11, 15, 19, 23, 27, 31`)
    that upgrades 1D sequence attention into a 2-Hop Relational Sheaf-GNN driven by
    CVNN Phase Coherence + Symbolic Dependency Edges.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        ro_free_start_dim: int = 64,
    ):
        super().__init__()
        self.base = base_attn
        self.num_heads = int(base_attn.config.num_attention_heads)
        self.num_kv_heads = int(base_attn.config.num_key_value_heads)
        self.num_kv_groups = int(base_attn.num_key_value_groups)
        self.head_dim = int(base_attn.head_dim)
        self.scaling = float(base_attn.scaling)
        self.ro_free_start_dim = int(ro_free_start_dim)

        # Strict Zero-Init GNN parameters guarantee 100.00% bit-exact Qwen3_5Attention at t=0
        self.alpha_hop = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.gamma_phase = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.eta_sym = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.kappa_dead = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

        self.external_phase: Optional[torch.Tensor] = None
        self.external_symbolic_adj: Optional[torch.Tensor] = None
        self.external_dead_mask: Optional[torch.Tensor] = None
        self._one_hop_cache_id: Optional[int] = None
        self._one_hop_cache: Optional[torch.Tensor] = None
        # Zero-CUDA-sync parameter version cache (eliminates 32 .item() syncs/token in eval!)
        self._cached_param_versions: Optional[Tuple[int, int, int, int]] = None
        self._alpha_nonzero: bool = False
        self._gamma_nonzero: bool = False
        self._eta_nonzero: bool = False
        self._kappa_nonzero: bool = False

    def _refresh_active_flags(self) -> Tuple[bool, bool, bool, bool]:
        vers = (
            int(self.alpha_hop._version),
            int(self.gamma_phase._version),
            int(self.eta_sym._version),
            int(self.kappa_dead._version),
        )
        if self._cached_param_versions != vers:
            self._alpha_nonzero = bool(float(self.alpha_hop.detach().abs().max().item()) > 0.0)
            self._gamma_nonzero = bool(float(self.gamma_phase.detach().abs().max().item()) > 0.0)
            self._eta_nonzero = bool(float(self.eta_sym.detach().abs().max().item()) > 0.0)
            self._kappa_nonzero = bool(float(self.kappa_dead.detach().abs().max().item()) > 0.0)
            self._cached_param_versions = vers
        return self._alpha_nonzero, self._gamma_nonzero, self._eta_nonzero, self._kappa_nonzero

    def set_graph_context(
        self,
        phase: Optional[torch.Tensor] = None,
        symbolic_adj: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.external_phase = phase
        self.external_symbolic_adj = symbolic_adj
        self.external_dead_mask = dead_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # If all GNN parameters are exactly zero (or no active graph biases/2-hop gains),
        # delegate directly to the native SDPA/Flash path for 100% bit-identity and zero CUDA syncs.
        if not self.training:
            a_nz, g_nz, e_nz, k_nz = self._refresh_active_flags()
            gnn_active = (
                a_nz
                or (self.external_phase is not None and g_nz)
                or (self.external_symbolic_adj is not None and e_nz)
                or (self.external_dead_mask is not None and k_nz)
            )
            if not gnn_active:
                return self.base(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
        else:
            a_nz = True

        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.base.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.base.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.base.k_norm(self.base.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.base.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.base.layer_idx)

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        # Compute base attention logits + Sheaf-GNN Phase & Symbolic Topology Biases
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        q_len, k_len = attn_scores.shape[-2], attn_scores.shape[-1]

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :k_len]
            attn_scores = attn_scores + causal_mask
        elif q_len > 1:
            upper = torch.ones((q_len, k_len), dtype=torch.bool, device=attn_scores.device).triu(k_len - q_len + 1)
            attn_scores = attn_scores.masked_fill(upper, float("-inf"))

        if self.external_phase is not None and self.external_phase.shape[1] >= k_len:
            ext_p = self.external_phase[:, :k_len]
            coh = compute_phase_coherence_matrix(ext_p)  # [B, H_k, K, K]
            if coh.shape[1] < self.num_heads and self.num_heads % coh.shape[1] == 0:
                coh = coh.repeat_interleave(self.num_heads // coh.shape[1], dim=1)
            elif coh.shape[1] != self.num_heads:
                coh = coh[:, :self.num_heads]
            gamma = self.gamma_phase.view(1, -1, 1, 1).to(device=attn_scores.device, dtype=attn_scores.dtype)
            attn_scores = attn_scores + gamma * coh[:, :, -q_len:, :k_len].to(device=attn_scores.device, dtype=attn_scores.dtype)

        if self.external_symbolic_adj is not None:
            eta = self.eta_sym.view(1, -1, 1, 1).to(device=attn_scores.device, dtype=attn_scores.dtype)
            attn_scores = attn_scores + eta * self.external_symbolic_adj[..., -q_len:, :k_len].to(device=attn_scores.device, dtype=attn_scores.dtype)

        if self.external_dead_mask is not None:
            # Exact zero at kappa_dead == 0: softplus(0) - ln(2) == 0.0 (preserves Exact Submanifold Theorem!)
            kappa_pos = torch.clamp(F.softplus(self.kappa_dead) - math.log(2.0), min=0.0)
            kappa = kappa_pos.view(1, -1, 1, 1).to(device=attn_scores.device, dtype=attn_scores.dtype)
            attn_scores = attn_scores - kappa * self.external_dead_mask[..., -q_len:, :k_len].to(device=attn_scores.device, dtype=attn_scores.dtype)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)
        one_hop_curr = attn_probs @ value_states
        one_hop_history = None
        if past_key_values is not None:
            cid = id(past_key_values)
            if q_len == k_len:
                self._one_hop_cache_id = cid
                self._one_hop_cache = one_hop_curr.detach()
                one_hop_history = one_hop_curr
            elif self._one_hop_cache_id == cid and self._one_hop_cache is not None:
                prev = self._one_hop_cache[:, :, : max(0, k_len - q_len), :].to(
                    device=one_hop_curr.device, dtype=one_hop_curr.dtype
                )
                one_hop_history = torch.cat([prev, one_hop_curr], dim=2)
                if one_hop_history.shape[-2] == k_len:
                    self._one_hop_cache = one_hop_history.detach()
                else:
                    one_hop_history = None

        if not a_nz and not self.training:
            attn_output = one_hop_curr
        else:
            attn_output = two_hop_sheaf_message_passing(
                attn_probs=attn_probs,
                values=value_states,
                alpha_hop=self.alpha_hop,
                symbolic_adj=self.external_symbolic_adj[..., -q_len:, :k_len] if self.external_symbolic_adj is not None else None,
                ro_free_start_dim=self.ro_free_start_dim,
                keep_graph=True,
                one_hop_history=one_hop_history,
                one_hop=one_hop_curr,
            )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.base.o_proj(attn_output)
        return attn_output, attn_probs
