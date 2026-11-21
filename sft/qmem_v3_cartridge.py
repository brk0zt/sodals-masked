"""
SODALS Core — Unified `.qmem v3` Virtual Parameter Cartridge Container & Manager
===============================================================================
Replaces the four legacy fragmented `.qmem` formats with a single deterministic,
shape-preserving, mmap-streamable binary container (`magic = b"QMEM"`, `version = 3.0`)
holding three mathematically active parametric banks:

  - Bank 1 (`bank1_mlp_down`):   Null-Space PGD / Tweezer low-rank `mlp.down_proj` Fact-Bank
  - Bank 2 (`bank2_attn_lora`):  InfLoRA / Gram-orthogonal Attention Reasoning-Tactic Bank
  - Bank 3 (`bank3_deltanet_s0`): Asymmetric multi-slot `(K_cart, V_cart, beta_cart)` fast-weight
                                  initial state bank for the 24 GatedDeltaNet layers

Plus a Layer-32 Gaussian/Mahalanobis routing signature (`centroid_l32`, `diag_var_l32`)
and SHA-256 payload checksum for zero-corruption streaming from NVMe -> CPU RAM LRU ->
fixed pre-allocated GPU cartridge slots (zero VRAM fragmentation, strictly 9B active VRAM).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

QMEM_V3_MAGIC = b"QMEM"
QMEM_V3_MAJOR = 3
QMEM_V3_MINOR = 0
# 64-byte header: magic(4s), maj(B), min(B), flags(H), tensor_count(I),
# toc_offset(Q), toc_len(Q), domain_id(16s), sha256_prefix(20s) = 4+1+1+2+4+8+8+16+20 = 64 bytes
QMEM_V3_HEADER_FMT = "<4sBBHIQQ16s20s"
QMEM_V3_HEADER_SIZE = struct.calcsize(QMEM_V3_HEADER_FMT)
assert QMEM_V3_HEADER_SIZE == 64

FLAG_HAS_BANK1_MLP = 1 << 0
FLAG_HAS_BANK2_ATTN = 1 << 1
FLAG_HAS_BANK3_DELTANET = 1 << 2
FLAG_QUANT_INT8 = 1 << 8

_DTYPE_TO_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
}
_DTYPE_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
}


@dataclass
class QmemV3TensorSpec:
    name: str
    bank_id: int  # 1=MLP Fact-Bank, 2=Attn Tactic-Bank, 3=DeltaNet S0 Bank
    layer_idx: int
    shape: List[int]
    dtype: str
    byte_offset: int
    byte_length: int
    scale: float = 1.0  # Dequantization scale when dtype == "int8"


@dataclass
class QmemV3Metadata:
    domain_id: str
    macro_domain: str = "general"
    description: str = ""
    rank_mlp: int = 0
    rank_attn: int = 0
    deltanet_slots: int = 0
    orthogonality_basis_hash: str = ""
    centroid_l32: Optional[List[float]] = None
    diag_var_l32: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QmemV3Cartridge:
    metadata: QmemV3Metadata
    tensors: Dict[str, torch.Tensor]
    bank_map: Dict[str, int]  # tensor_name -> bank_id (1, 2, or 3)
    layer_map: Dict[str, int]  # tensor_name -> layer_idx
    sha256_prefix: str = ""
    source_path: Optional[str] = None

    @property
    def total_params(self) -> int:
        return sum(int(t.numel()) for t in self.tensors.values())

    @property
    def total_bytes(self) -> int:
        return sum(int(t.numel() * t.element_size()) for t in self.tensors.values())

    def get_bank_tensors(self, bank_id: int) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.tensors.items() if self.bank_map.get(k) == bank_id}


def _infer_bank_and_layer(name: str) -> Tuple[int, int]:
    """Infers (bank_id, layer_idx) from canonical tensor naming."""
    layer_idx = -1
    parts = name.replace("/", ".").split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "layer", "blocks", "blk") and i + 1 < len(parts):
            if parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if part.startswith("l") and part[1:].isdigit():
            layer_idx = int(part[1:])
            break
    low = name.lower()
    if "deltanet" in low or "s0_" in low or "k_cart" in low or "v_cart" in low or "beta_cart" in low or "bank3" in low:
        return 3, layer_idx
    if "mlp" in low or "down_proj" in low or "gate_proj" in low or "up_proj" in low or "bank1" in low:
        return 1, layer_idx
    return 2, layer_idx


def _quantize_int8_symmetric(t: torch.Tensor) -> Tuple[np.ndarray, float]:
    arr = t.detach().cpu().float().numpy()
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if max_abs < 1e-12:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def save_qmem_v3(
    path: str | Path,
    tensors: Dict[str, torch.Tensor],
    metadata: QmemV3Metadata,
    bank_map: Optional[Dict[str, int]] = None,
    layer_map: Optional[Dict[str, int]] = None,
    quantize_int8: bool = False,
) -> Dict[str, Any]:
    """
    Writes a unified `.qmem v3` binary cartridge with 64-byte header, contiguous
    64-byte-aligned tensor payload, JSON TOC, and SHA-256 payload integrity check.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flags = 0
    if quantize_int8:
        flags |= FLAG_QUANT_INT8

    payload_chunks: List[bytes] = []
    toc_entries: List[Dict[str, Any]] = []
    cur_offset = QMEM_V3_HEADER_SIZE
    hasher = hashlib.sha256()

    for name in sorted(tensors.keys()):
        t = tensors[name]
        b_id, l_idx = _infer_bank_and_layer(name)
        if bank_map and name in bank_map:
            b_id = int(bank_map[name])
        if layer_map and name in layer_map:
            l_idx = int(layer_map[name])

        if b_id == 1:
            flags |= FLAG_HAS_BANK1_MLP
        elif b_id == 2:
            flags |= FLAG_HAS_BANK2_ATTN
        elif b_id == 3:
            flags |= FLAG_HAS_BANK3_DELTANET

        # Keep Bank 3 fast-weight parameters in float32/float16 for phase/delta stability;
        # optionally quantize Bank 1/2 large LoRA matrices to int8 for 2x smaller SSD footprint.
        if quantize_int8 and b_id in (1, 2) and t.numel() >= 64:
            arr, scale = _quantize_int8_symmetric(t)
            dtype_str = "int8"
        elif t.dtype == torch.float32:
            arr = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
            dtype_str = "float32"
            scale = 1.0
        else:
            arr = np.ascontiguousarray(t.detach().cpu().to(torch.float16).numpy(), dtype=np.float16)
            dtype_str = "float16"
            scale = 1.0

        raw = arr.tobytes(order="C")
        pad_len = (64 - (cur_offset % 64)) % 64
        if pad_len:
            pad_bytes = b"\x00" * pad_len
            payload_chunks.append(pad_bytes)
            hasher.update(pad_bytes)
            cur_offset += pad_len

        payload_chunks.append(raw)
        hasher.update(raw)
        toc_entries.append(
            {
                "name": name,
                "bank_id": b_id,
                "layer_idx": l_idx,
                "shape": list(t.shape),
                "dtype": dtype_str,
                "byte_offset": cur_offset,
                "byte_length": len(raw),
                "scale": scale,
            }
        )
        cur_offset += len(raw)

    toc_obj = {
        "format": "QMEM_V3",
        "metadata": {
            "domain_id": metadata.domain_id,
            "macro_domain": metadata.macro_domain,
            "description": metadata.description,
            "rank_mlp": metadata.rank_mlp,
            "rank_attn": metadata.rank_attn,
            "deltanet_slots": metadata.deltanet_slots,
            "orthogonality_basis_hash": metadata.orthogonality_basis_hash,
            "centroid_l32": metadata.centroid_l32,
            "diag_var_l32": metadata.diag_var_l32,
            "extra": metadata.extra,
        },
        "tensors": toc_entries,
    }
    toc_bytes = json.dumps(toc_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hasher.update(toc_bytes)
    digest20 = hasher.digest()[:20]

    dom_bytes = metadata.domain_id.encode("utf-8")[:16].ljust(16, b"\x00")
    header = struct.pack(
        QMEM_V3_HEADER_FMT,
        QMEM_V3_MAGIC,
        QMEM_V3_MAJOR,
        QMEM_V3_MINOR,
        flags,
        len(toc_entries),
        cur_offset,
        len(toc_bytes),
        dom_bytes,
        digest20,
    )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(header)
        for ch in payload_chunks:
            f.write(ch)
        f.write(toc_bytes)
    os.replace(tmp_path, out_path)

    return {
        "path": str(out_path),
        "domain_id": metadata.domain_id,
        "macro_domain": metadata.macro_domain,
        "tensor_count": len(toc_entries),
        "file_bytes": out_path.stat().st_size,
        "sha256_prefix": digest20.hex(),
        "quantize_int8": quantize_int8,
    }


def inspect_qmem_v3_header(path: str | Path) -> Dict[str, Any]:
    """Reads ONLY the 64-byte header + TOC metadata without loading tensor payloads."""
    p = Path(path)
    with open(p, "rb") as f:
        hdr = f.read(QMEM_V3_HEADER_SIZE)
        if len(hdr) < QMEM_V3_HEADER_SIZE:
            raise ValueError(f"Truncated QMEM v3 header in {p}")
        magic, maj, minv, flags, t_cnt, toc_off, toc_len, dom_b, dig20 = struct.unpack(QMEM_V3_HEADER_FMT, hdr)
        if magic != QMEM_V3_MAGIC or maj != QMEM_V3_MAJOR:
            raise ValueError(f"Invalid QMEM v3 header in {p}: magic={magic!r}, version={maj}.{minv}")
        f.seek(toc_off)
        toc_raw = f.read(toc_len)
    toc = json.loads(toc_raw.decode("utf-8"))
    return {
        "version": f"{maj}.{minv}",
        "flags": flags,
        "tensor_count": t_cnt,
        "domain_id": dom_b.rstrip(b"\x00").decode("utf-8", errors="replace"),
        "sha256_prefix": dig20.hex(),
        "metadata": toc.get("metadata", {}),
        "tensors": toc.get("tensors", []),
    }


def load_qmem_v3(
    path: str | Path,
    device: str | torch.device = "cpu",
    verify_checksum: bool = True,
    target_dtype: Optional[torch.dtype] = None,
) -> QmemV3Cartridge:
    """
    Loads a `.qmem v3` cartridge via OS mmap, verifies SHA-256 payload integrity,
    and reconstructs exact multi-dimensional tensors for all 3 banks.
    """
    p = Path(path)
    file_size = p.stat().st_size
    if file_size < QMEM_V3_HEADER_SIZE:
        raise ValueError(f"File too small to be QMEM v3: {p}")

    with open(p, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            hdr = mm[:QMEM_V3_HEADER_SIZE]
            magic, maj, minv, flags, t_cnt, toc_off, toc_len, dom_b, dig20 = struct.unpack(QMEM_V3_HEADER_FMT, hdr)
            if magic != QMEM_V3_MAGIC or maj != QMEM_V3_MAJOR:
                raise ValueError(f"Not a QMEM v3 cartridge: {p} (magic={magic!r}, v={maj}.{minv})")
            if toc_off + toc_len > file_size:
                raise ValueError(f"Corrupted TOC bounds in {p}")

            if verify_checksum:
                calc = hashlib.sha256(memoryview(mm)[QMEM_V3_HEADER_SIZE : toc_off + toc_len]).digest()[:20]
                if calc != dig20:
                    raise ValueError(
                        f"SHA-256 checksum mismatch in {p}: expected {dig20.hex()}, got {calc.hex()}"
                    )

            toc = json.loads(bytes(mm[toc_off : toc_off + toc_len]).decode("utf-8"))
            meta_d = toc.get("metadata", {})
            meta = QmemV3Metadata(
                domain_id=str(meta_d.get("domain_id") or dom_b.rstrip(b"\x00").decode("utf-8", errors="replace")),
                macro_domain=str(meta_d.get("macro_domain", "general")),
                description=str(meta_d.get("description", "")),
                rank_mlp=int(meta_d.get("rank_mlp", 0)),
                rank_attn=int(meta_d.get("rank_attn", 0)),
                deltanet_slots=int(meta_d.get("deltanet_slots", 0)),
                orthogonality_basis_hash=str(meta_d.get("orthogonality_basis_hash", "")),
                centroid_l32=meta_d.get("centroid_l32"),
                diag_var_l32=meta_d.get("diag_var_l32"),
                extra=dict(meta_d.get("extra", {})),
            )

            tensors: Dict[str, torch.Tensor] = {}
            bank_map: Dict[str, int] = {}
            layer_map: Dict[str, int] = {}
            for entry in toc.get("tensors", []):
                name = str(entry["name"])
                dtype_str = str(entry["dtype"])
                shape = [int(x) for x in entry["shape"]]
                off = int(entry["byte_offset"])
                blen = int(entry["byte_length"])
                scale = float(entry.get("scale", 1.0))

                np_dt = _DTYPE_TO_NP[dtype_str]
                arr = np.frombuffer(mm, dtype=np_dt, count=blen // np.dtype(np_dt).itemsize, offset=off).reshape(shape).copy()
                if dtype_str == "int8":
                    t = torch.from_numpy(arr.astype(np.float32) * scale)
                    if target_dtype is not None:
                        t = t.to(dtype=target_dtype)
                else:
                    t = torch.from_numpy(arr)
                    if target_dtype is not None:
                        t = t.to(dtype=target_dtype)
                tensors[name] = t.to(device=device)
                bank_map[name] = int(entry.get("bank_id", 2))
                layer_map[name] = int(entry.get("layer_idx", -1))

    return QmemV3Cartridge(
        metadata=meta,
        tensors=tensors,
        bank_map=bank_map,
        layer_map=layer_map,
        sha256_prefix=dig20.hex(),
        source_path=str(p),
    )


class QmemV3Manager:
    """
    Hierarchical NVMe -> CPU RAM LRU -> Pre-Allocated GPU Slot Manager.
    Streams specialist `.qmem v3` cartridges on demand in < 2 ms without growing active VRAM.
    """

    def __init__(
        self,
        max_ram_mb: float = 1024.0,
        max_active_slots: int = 3,
        verify_checksum: bool = True,
    ):
        self.max_ram_bytes = int(max(32.0, float(max_ram_mb)) * 1024 * 1024)
        self.max_active_slots = max(1, int(max_active_slots))
        self.verify_checksum = bool(verify_checksum)

        self._registry: Dict[str, Path] = {}
        self._ram_cache: "OrderedDict[str, QmemV3Cartridge]" = OrderedDict()
        self._ram_bytes_used: int = 0
        self.slot_names: List[str] = [f"qmem_slot_{i}" for i in range(self.max_active_slots)]
        self._slot_assignment: Dict[str, Optional[str]] = {s: None for s in self.slot_names}
        self.stats: Dict[str, Any] = {
            "ram_hits": 0,
            "nvme_loads": 0,
            "evictions": 0,
            "slot_swaps": 0,
            "last_swap_ms": 0.0,
        }

    def register_cartridge(self, domain_id: str, path: str | Path) -> None:
        self._registry[str(domain_id)] = Path(path)

    def register_directory(self, directory: str | Path) -> int:
        d = Path(directory)
        count = 0
        if not d.is_dir():
            return 0
        for fp in sorted(d.glob("*.qmem")):
            try:
                info = inspect_qmem_v3_header(fp)
                dom = str(info["domain_id"])
                self.register_cartridge(dom, fp)
                count += 1
            except Exception:
                continue
        return count

    def get_cartridge(self, domain_id: str) -> QmemV3Cartridge:
        dom = str(domain_id)
        if dom in self._ram_cache:
            self._ram_cache.move_to_end(dom)
            self.stats["ram_hits"] += 1
            return self._ram_cache[dom]

        if dom not in self._registry:
            raise KeyError(f"QMEM v3 cartridge not registered: {dom}")

        cart = load_qmem_v3(self._registry[dom], device="cpu", verify_checksum=self.verify_checksum)
        self.stats["nvme_loads"] += 1
        self._insert_ram_cache(dom, cart)
        return cart

    def _insert_ram_cache(self, dom: str, cart: QmemV3Cartridge) -> None:
        sz = cart.total_bytes
        while self._ram_cache and (self._ram_bytes_used + sz > self.max_ram_bytes):
            _, ev = self._ram_cache.popitem(last=False)
            self._ram_bytes_used = max(0, self._ram_bytes_used - ev.total_bytes)
            self.stats["evictions"] += 1
        self._ram_cache[dom] = cart
        self._ram_bytes_used += sz

    def mount_top_k_into_model(
        self,
        model: Any,
        weighted_domains: Dict[str, float],
        blender: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Hot-swaps up to `max_active_slots` cartridges into pre-allocated PEFT LoRA slots
        (Bank 1 + Bank 2) AND injects blended Bank 3 DeltaNet fast-weight initial states.
        """
        t0 = time.perf_counter()
        ordered = sorted(
            [(d, float(w)) for d, w in weighted_domains.items() if float(w) > 0.0],
            key=lambda kv: kv[1],
            reverse=True,
        )[: self.max_active_slots]

        slot_weights: Dict[str, float] = {}
        loaded_carts: List[Tuple[QmemV3Cartridge, float]] = []
        swapped_this_turn = 0

        for idx, (dom, weight) in enumerate(ordered):
            slot_name = self.slot_names[idx]
            cart = self.get_cartridge(dom)
            loaded_carts.append((cart, weight))
            if self._slot_assignment.get(slot_name) != dom:
                load_cartridge_lora_into_slot(model, slot_name, cart)
                self._slot_assignment[slot_name] = dom
                swapped_this_turn += 1
            slot_weights[slot_name] = weight

        # Apply Bank 1 & Bank 2 multi-cartridge weights (zeroing unused slots, preserving svon_identity)
        if blender is not None and hasattr(blender, "set_multi_cartridge_blend"):
            blender.set_multi_cartridge_blend(model, slot_weights=slot_weights, managed_slots=self.slot_names)

        # Apply blended Bank 3 DeltaNet fast-weight S_0 states
        bank3_layers = apply_cartridge_bank3_to_deltanet_layers(model, loaded_carts)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats["slot_swaps"] += swapped_this_turn
        self.stats["last_swap_ms"] = round(elapsed_ms, 3)
        return {
            "mounted_slots": {self.slot_names[i]: ordered[i][0] for i in range(len(ordered))},
            "slot_weights": slot_weights,
            "bank3_layers_injected": bank3_layers,
            "swapped_count": swapped_this_turn,
            "swap_ms": round(elapsed_ms, 3),
        }


def load_cartridge_lora_into_slot(model: Any, slot_name: str, cartridge: QmemV3Cartridge) -> int:
    """
    Copies Bank 1 (`mlp.down_proj`) and Bank 2 (`q/k/v/o_proj`) LoRA weights directly into
    a pre-allocated PEFT adapter slot in-place (`copy_`), avoiding any new VRAM allocations.
    Zeroes modules not touched by this cartridge so stale weights from prior domains never leak.
    """
    updated = 0
    with torch.no_grad():
        for mod_name, mod in model.named_modules():
            if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
                continue
            if slot_name not in mod.lora_A or slot_name not in mod.lora_B:
                continue

            la = mod.lora_A[slot_name].weight
            lb = mod.lora_B[slot_name].weight
            # Try matching canonical keys (`<mod_name>.lora_A` or normalized suffix)
            norm_base = mod_name.replace("base_model.model.", "")
            cand_a = (
                cartridge.tensors.get(f"{norm_base}.lora_A")
                or cartridge.tensors.get(f"{mod_name}.lora_A")
                or cartridge.tensors.get(f"bank1.{norm_base}.lora_A")
                or cartridge.tensors.get(f"bank2.{norm_base}.lora_A")
            )
            cand_b = (
                cartridge.tensors.get(f"{norm_base}.lora_B")
                or cartridge.tensors.get(f"{mod_name}.lora_B")
                or cartridge.tensors.get(f"bank1.{norm_base}.lora_B")
                or cartridge.tensors.get(f"bank2.{norm_base}.lora_B")
            )

            if cand_a is not None and cand_b is not None:
                la.zero_()
                lb.zero_()
                r_use = min(la.shape[0], cand_a.shape[0], lb.shape[1], cand_b.shape[1])
                in_use = min(la.shape[1], cand_a.shape[1])
                out_use = min(lb.shape[0], cand_b.shape[0])
                la[:r_use, :in_use].copy_(cand_a[:r_use, :in_use].to(device=la.device, dtype=la.dtype))
                lb[:out_use, :r_use].copy_(cand_b[:out_use, :r_use].to(device=lb.device, dtype=lb.dtype))
                updated += 1
            else:
                la.zero_()
                lb.zero_()
    return updated


def apply_cartridge_bank3_to_deltanet_layers(
    model: Any,
    weighted_cartridges: Sequence[Tuple[QmemV3Cartridge, float]],
) -> int:
    """
    Injects Bank 3 asymmetric multi-slot `(K_cart, V_cart, beta_cart)` fast-weight memories
    into all `HolomorphicCVNNDeltaNetWrapper` layers. When `weighted_cartridges` is empty,
    clears active Bank 3 states cleanly so native base behavior is restored bit-identically.
    """
    injected = 0
    for _, mod in model.named_modules():
        if not hasattr(mod, "set_initial_cartridge_state"):
            continue
        l_idx = int(getattr(getattr(mod, "base", None), "layer_idx", -1))
        if not weighted_cartridges:
            mod.set_initial_cartridge_state(None, None, None)
            continue

        k_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        b_list: List[torch.Tensor] = []
        phi_list: List[torch.Tensor] = []
        active_carts = [(c, w) for c, w in weighted_cartridges if w > 0.0]
        n_active = max(1, len(active_carts))
        for c_idx, (cart, w) in enumerate(active_carts):
            k_t = cart.tensors.get(f"bank3.layers.{l_idx}.k_cart")
            v_t = cart.tensors.get(f"bank3.layers.{l_idx}.v_cart")
            b_t = cart.tensors.get(f"bank3.layers.{l_idx}.beta_cart")
            if k_t is not None and v_t is not None and b_t is not None:
                k_list.append(k_t)
                v_list.append(v_t)
                b_list.append(b_t.float() * float(w))
                # Phase-Multiplexed Bank 3: assign distinct orthogonal phase band φ_k = π * k / K per cartridge
                phi_val = (3.141592653589793 * float(c_idx) / float(n_active)) if n_active > 1 else 0.0
                phi_list.append(torch.full((k_t.shape[0],), phi_val, dtype=torch.float32))

        if k_list:
            mod.set_initial_cartridge_state(
                torch.cat(k_list, dim=0),
                torch.cat(v_list, dim=0),
                torch.cat(b_list, dim=0),
                phase_bands=torch.cat(phi_list, dim=0) if n_active > 1 else None,
            )
            injected += 1
        else:
            mod.set_initial_cartridge_state(None, None, None)
    return injected
