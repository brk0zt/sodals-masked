"""
Svon-CGW (Epoch III) — Sütun III: Kuramoto Phase-Locking Equilibrium & Rust .qmem Bridge
=========================================================================================
Provides:
  1. `KuramotoPhaseEquilibrium`: Computes the complex Kuramoto order parameter
     R_order = |mean_u exp(i * phi_u)| in [0, 1] and synchronizes coupled oscillators
     across reasoning steps until phase variance locks.
  2. `upgrade_model_to_svon_cgw`: Upgrades all 24 `Qwen3_5GatedDeltaNet` layers to
     `HolomorphicCVNNDeltaNetWrapper` and all 8 `Qwen3_5Attention` layers to
     `RelationalSheafGNNAttention`, automatically piping the complex phase state
     from each 3-layer CVNN block (`3m+2`) into the following Sheaf-GNN layer (`3m+3`).
  3. `export_svon_cgw_to_rust_qmem`: Serializes trained CVNN+GNN parameters into both
     `.safetensors` (for Python `sodals_serve.py`) and binary `.qmem` + `qmem_manifest.json`
     (matching `src/crates/sodals_industrial/src/qmem_format.rs` 32-byte `QmemHeader` and
     `src/crates/sodals_neuro/src/qmem_loader/mod.rs` `QmemManifest`).
"""

from __future__ import annotations

import json
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from svon_cvnn_deltanet import HolomorphicCVNNDeltaNetWrapper
from svon_sheaf_gnn_attention import RelationalSheafGNNAttention, compute_phase_coherence_matrix


def compute_kuramoto_order_parameter(phase: torch.Tensor) -> torch.Tensor:
    """
    Computes the global Kuramoto phase synchronization order parameter R in [0, 1]:
      R = | (1/T) sum_{t=1}^T exp(i * phase_t) |
    Args:
      phase: [B, T, H, C] in radians
    Returns:
      r_order: [B, H, C] in [0, 1], where 1.0 = perfect phase lock, 0.0 = decoherent.
    """
    p = phase.float()
    mean_cos = torch.cos(p).mean(dim=1)
    mean_sin = torch.sin(p).mean(dim=1)
    return torch.sqrt(mean_cos.pow(2) + mean_sin.pow(2) + 1e-12)


def kuramoto_phase_step(
    phase: torch.Tensor,
    coupling_strength: Any = 0.25,
    adjacency: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Executes one discrete causal Kuramoto oscillator coupling step:
      phi_u^{(k+1)} = phi_u^{(k)} + (K / deg(u)) sum_{v <= u} A_{uv} sin(phi_v^{(k)} - phi_u^{(k)})
    Using identity: sin(b - a) = sin(b)cos(a) - cos(b)sin(a).
    Supports both scalar float and differentiable torch.Tensor `coupling_strength`.
    """
    p = phase.float()  # [B, T, H, C]
    bsz, seq_len, n_heads, c_dim = p.shape
    p_bhtc = p.permute(0, 2, 1, 3)  # [B, H, T, C]

    if adjacency is None:
        tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32, device=p.device))
        deg = tril.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = (tril / deg).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    else:
        adj = adjacency.float()
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(0)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    sin_p = torch.sin(p_bhtc)
    cos_p = torch.cos(p_bhtc)
    # sum_v A_{uv} (sin(phi_v) cos(phi_u) - cos(phi_v) sin(phi_u))
    agg_sin = torch.matmul(adj, sin_p)
    agg_cos = torch.matmul(adj, cos_p)
    delta = agg_sin * cos_p - agg_cos * sin_p

    if isinstance(coupling_strength, torch.Tensor):
        k_eff = coupling_strength.to(device=p.device, dtype=torch.float32)
    else:
        k_eff = float(coupling_strength)
    updated = p_bhtc + k_eff * delta
    return updated.permute(0, 2, 1, 3).to(phase.dtype)


class KuramotoPhaseEquilibrium(nn.Module):
    """
    Synchronizes complex phase trajectories across reasoning steps until the
    Kuramoto order parameter R_order >= sync_threshold (up to max_sync_steps).
    """

    def __init__(
        self,
        sync_threshold: float = 0.75,
        coupling_strength: float = 0.30,
        max_sync_steps: int = 3,
    ):
        super().__init__()
        self.sync_threshold = float(sync_threshold)
        self.coupling_strength = nn.Parameter(torch.tensor(float(coupling_strength), dtype=torch.float32))
        self.max_sync_steps = int(max_sync_steps)

    def forward(
        self,
        phase: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cur = phase
        r0 = float(compute_kuramoto_order_parameter(cur).mean().item())
        steps_run = 0
        # Keep coupling_strength differentiable as a Tensor (no .item() graph break!)
        k_val = torch.clamp(self.coupling_strength, min=0.0, max=1.0)

        for _ in range(self.max_sync_steps):
            r_cur = float(compute_kuramoto_order_parameter(cur).mean().item())
            if r_cur >= self.sync_threshold:
                break
            cur = kuramoto_phase_step(cur, coupling_strength=k_val, adjacency=adjacency)
            steps_run += 1

        r_final = float(compute_kuramoto_order_parameter(cur).mean().item())
        return cur, {
            "r_initial": round(r0, 6),
            "r_final": round(r_final, 6),
            "sync_steps": steps_run,
        }


def _locate_decoder_layers(model: nn.Module) -> List[nn.Module]:
    from cloud_runtime_utils import locate_decoder_layers
    return locate_decoder_layers(model)


def upgrade_model_to_svon_cgw(
    model: nn.Module,
    phase_rank: int = 16,
    ro_free_start_dim: int = 64,
    freeze_base: bool = True,
) -> Dict[str, Any]:
    """
    In-place architectural upgrade of Qwen 3.5 (or compatible hybrid DecoderLayer stack)
    into Svon-CGW:
      - Wraps every `linear_attn` in `HolomorphicCVNNDeltaNetWrapper`
      - Wraps every `self_attn` in `RelationalSheafGNNAttention`
      - Registers a bridge hook so the latest CVNN cumulative phase automatically feeds
        the subsequent `RelationalSheafGNNAttention` layer.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    layers = _locate_decoder_layers(model)
    cvnn_layers: List[int] = []
    gnn_layers: List[int] = []
    last_cvnn_ref: List[Optional[HolomorphicCVNNDeltaNetWrapper]] = [None]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
            if not isinstance(layer.linear_attn, HolomorphicCVNNDeltaNetWrapper):
                dev = next(layer.linear_attn.parameters()).device
                wrapper = HolomorphicCVNNDeltaNetWrapper(layer.linear_attn, phase_rank=phase_rank).to(dev)
                layer.linear_attn = wrapper
            last_cvnn_ref[0] = layer.linear_attn
            cvnn_layers.append(idx)

        elif hasattr(layer, "self_attn") and layer.self_attn is not None:
            if not isinstance(layer.self_attn, RelationalSheafGNNAttention):
                dev = next(layer.self_attn.parameters()).device
                g_wrap = RelationalSheafGNNAttention(
                    layer.self_attn, ro_free_start_dim=ro_free_start_dim
                ).to(dev)
                prev_cvnn = last_cvnn_ref[0]
                if prev_cvnn is not None:
                    def _make_bridge(cvnn_mod: HolomorphicCVNNDeltaNetWrapper):
                        def _pre_hook(gnn_mod: RelationalSheafGNNAttention, _args):
                            gnn_mod.external_phase = cvnn_mod.last_cumulative_phase
                        return _pre_hook
                    g_wrap.register_forward_pre_hook(_make_bridge(prev_cvnn))
                layer.self_attn = g_wrap
            gnn_layers.append(idx)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "cvnn_layer_indices": cvnn_layers,
        "sheaf_gnn_layer_indices": gnn_layers,
        "trainable_cgw_params": int(trainable_params),
        "trainable_cgw_mb_fp32": round(trainable_params * 4 / (1024 * 1024), 3),
    }


def collect_svon_cgw_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts all trainable Svon-CGW (`HolomorphicCVNNDeltaNetWrapper` + `RelationalSheafGNNAttention`) tensors."""
    layers = _locate_decoder_layers(model)
    state: Dict[str, torch.Tensor] = {}
    for idx, layer in enumerate(layers):
        if hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, HolomorphicCVNNDeltaNetWrapper):
            cw = layer.linear_attn
            state[f"layers.{idx}.cvnn.phase_proj_down.weight"] = cw.phase_proj_down.weight.detach().float().cpu()
            state[f"layers.{idx}.cvnn.phase_proj_up.weight"] = cw.phase_proj_up.weight.detach().float().cpu()
            state[f"layers.{idx}.cvnn.alpha_im"] = cw.alpha_im.detach().float().cpu()
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, RelationalSheafGNNAttention):
            gw = layer.self_attn
            state[f"layers.{idx}.sheaf_gnn.alpha_hop"] = gw.alpha_hop.detach().float().cpu()
            state[f"layers.{idx}.sheaf_gnn.gamma_phase"] = gw.gamma_phase.detach().float().cpu()
            state[f"layers.{idx}.sheaf_gnn.eta_sym"] = gw.eta_sym.detach().float().cpu()
            state[f"layers.{idx}.sheaf_gnn.kappa_dead"] = gw.kappa_dead.detach().float().cpu()
    return state


def pack_qmem_header(
    num_elements: int,
    dtype_code: int = 2,
    sample_rate_hz: float = 1000.0,
    label: str = "SVONCGW0",
    version: Tuple[int, int] = (2, 1),
) -> bytes:
    """
    Packs the exact 32-byte `#[repr(C)] QmemHeader` defined in
    `src/crates/sodals_industrial/src/qmem_format.rs`:
      magic: [u8; 4] = b"QMEM"
      version: [u8; 2]
      dtype: u8 (0=Euler, 1=Freq, 2=Holographic/Complex)
      _pad0: u8 = 0
      len: u32 (little-endian)
      sample_rate_hz: f32 (little-endian)
      timestamp_ns: u64 (little-endian)
      label: [u8; 8] ASCII
    """
    lbl = label.encode("ascii", errors="ignore")[:8].ljust(8, b"\x00")
    ts_ns = int(time.time_ns())
    return struct.pack(
        "<4sBBBBIfQ8s",
        b"QMEM",
        int(version[0]) & 0xFF,
        int(version[1]) & 0xFF,
        int(dtype_code) & 0xFF,
        0,
        int(num_elements) & 0xFFFFFFFF,
        float(sample_rate_hz),
        ts_ns,
        lbl,
    )


def export_svon_cgw_to_rust_qmem(
    cgw_state: Dict[str, torch.Tensor],
    out_dir: str,
    basename: str = "svon_cgw_hybrid",
) -> Dict[str, Any]:
    """
    Exports Svon-CGW weights to:
      1. `<out_dir>/<basename>.safetensors` (for Python / `sodals_serve.py`)
      2. `<out_dir>/<basename>.qmem` (32-byte `QmemHeader` + contiguous little-endian f32 layer blobs)
      3. `<out_dir>/<basename>_manifest.json` (`QmemManifest` schema for Rust `QmemLoader`)
    """
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    st_path = os.path.join(out_dir, f"{basename}.safetensors")
    qmem_path = os.path.join(out_dir, f"{basename}.qmem")
    manifest_path = os.path.join(out_dir, f"{basename}_manifest.json")

    save_file({k: v.contiguous() for k, v in cgw_state.items()}, st_path)

    # Group tensors by layer index
    by_layer: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
    for k in sorted(cgw_state.keys()):
        parts = k.split(".")
        l_idx = int(parts[1]) if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit() else 0
        by_layer.setdefault(l_idx, []).append((k, cgw_state[k].detach().float().cpu().contiguous().view(-1)))

    total_floats = sum(int(t.numel()) for pairs in by_layer.values() for _, t in pairs)
    header_bytes = pack_qmem_header(num_elements=total_floats, dtype_code=2, label="SVONCGW0")

    manifest_layers = []
    offset_bytes = len(header_bytes)  # 32-byte QmemHeader prefix
    with open(qmem_path, "wb") as qf:
        qf.write(header_bytes)
        for l_idx in sorted(by_layer.keys()):
            flat = torch.cat([t for _, t in by_layer[l_idx]], dim=0)
            raw_bytes = flat.numpy().tobytes()
            qf.write(raw_bytes)
            manifest_layers.append({
                "layer_index": l_idx,
                "path": os.path.basename(qmem_path),
                "offset": offset_bytes,
                "length": int(len(raw_bytes)),
                "byte_length": int(len(raw_bytes)),
                "num_elements": int(flat.numel()),
                "dtype": "f32",
                "keys": [name for name, _ in by_layer[l_idx]],
            })
            offset_bytes += len(raw_bytes)

    manifest_payload = {"layers": manifest_layers}
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_payload, mf, indent=2)

    return {
        "safetensors_path": st_path,
        "qmem_path": qmem_path,
        "manifest_path": manifest_path,
        "total_floats": total_floats,
        "total_bytes": offset_bytes,
        "num_layers_exported": len(manifest_layers),
    }


def synchronize_model_cvnn_phases_on_rollback(
    model: nn.Module,
    coupling_strength: float = 0.35,
    max_sync_steps: int = 3,
    sync_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Live Step-Rollback Phase Synchronizer (replaces dummy random RK4 ODE call):
    Iterates over all `HolomorphicCVNNDeltaNetWrapper` layers with recorded cumulative
    phase trajectories, applies `KuramotoPhaseEquilibrium` to lock decoherent oscillators,
    updates both `_cached_phase_tail` & `_cached_phase_history` so subsequent KV-cached
    tokens inherit the locked phase, and updates downstream `RelationalSheafGNNAttention`.
    """
    layers = _locate_decoder_layers(model)
    eq = KuramotoPhaseEquilibrium(
        sync_threshold=sync_threshold,
        coupling_strength=coupling_strength,
        max_sync_steps=max_sync_steps,
    )
    synced_layers = 0
    r_before_list: List[float] = []
    r_after_list: List[float] = []
    last_synced_phase: Optional[torch.Tensor] = None

    with torch.no_grad():
        for layer in layers:
            if hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, HolomorphicCVNNDeltaNetWrapper):
                cw = layer.linear_attn
                if cw.last_cumulative_phase is not None and cw.last_cumulative_phase.numel() > 0:
                    new_p, meta = eq(cw.last_cumulative_phase.detach())
                    cw.last_cumulative_phase = new_p
                    cw._cached_phase_history = new_p.detach()
                    cw._cached_phase_tail = new_p[:, -1:, :, :].detach()
                    last_synced_phase = new_p
                    synced_layers += 1
                    r_before_list.append(float(meta["r_initial"]))
                    r_after_list.append(float(meta["r_final"]))
            elif hasattr(layer, "self_attn") and isinstance(layer.self_attn, RelationalSheafGNNAttention):
                if last_synced_phase is not None:
                    layer.self_attn.external_phase = last_synced_phase

    return {
        "synced_cvnn_layers": synced_layers,
        "mean_r_before": round(sum(r_before_list) / len(r_before_list), 6) if r_before_list else 1.0,
        "mean_r_after": round(sum(r_after_list) / len(r_after_list), 6) if r_after_list else 1.0,
    }


def load_svon_cgw_state_dict(model: nn.Module, safetensors_path: str) -> int:
    """Loads exported Svon-CGW `.safetensors` weights into an upgraded Svon-CGW model."""
    from safetensors.torch import load_file

    tensors = load_file(safetensors_path)
    layers = _locate_decoder_layers(model)
    loaded = 0
    for idx, layer in enumerate(layers):
        if hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, HolomorphicCVNNDeltaNetWrapper):
            cw = layer.linear_attn
            for attr, suffix in (
                ("phase_proj_down", "phase_proj_down.weight"),
                ("phase_proj_up", "phase_proj_up.weight"),
            ):
                k = f"layers.{idx}.cvnn.{suffix}"
                if k in tensors:
                    getattr(cw, attr).weight.data.copy_(tensors[k].to(getattr(cw, attr).weight.device))
                    loaded += 1
            k_im = f"layers.{idx}.cvnn.alpha_im"
            if k_im in tensors:
                cw.alpha_im.data.copy_(tensors[k_im].to(cw.alpha_im.device))
                loaded += 1
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, RelationalSheafGNNAttention):
            gw = layer.self_attn
            for p_name in ("alpha_hop", "gamma_phase", "eta_sym", "kappa_dead"):
                k = f"layers.{idx}.sheaf_gnn.{p_name}"
                if k in tensors:
                    getattr(gw, p_name).data.copy_(tensors[k].to(getattr(gw, p_name).device))
                    loaded += 1
    return loaded

