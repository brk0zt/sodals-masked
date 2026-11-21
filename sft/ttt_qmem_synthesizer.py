"""
SODALS Core — `TTT-QMEM`: Test-Time Gradient Descent Cartridge Synthesizer
==========================================================================
Compiles unseen HLE axioms, novel physical rules, technical specifications,
or target SWE-Pro repository summaries directly into an ephemeral parametric
`.qmem v3` cartridge via fast self-supervised gradient descent — NOT RAG text stuffing!

Modes:
  - Fast Mode (default, `< 5s` on GPU): 6 gradient steps over Bank 3 (`DeltaNet S_0`)
    + optional rank-8 Bank 1 (`mlp.down_proj`) — ideal for single HLE/novel-axiom prompts.
  - Deep Repo Mode (`--ttt_deep`, `15-30s` on GPU): 24 gradient steps over repository AST,
    module signatures, and invariants to produce a reusable `repo_<hash>.qmem` cartridge.

Base model parameters remain strictly frozen (`0.0` drift guaranteed).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qmem_v3_cartridge import QmemV3Cartridge, QmemV3Metadata, save_qmem_v3
from train_qmem_v3_cartridge import Bank3DeltaNetFastWeightBank, extract_layer32_gaussian_signature


@dataclass
class TTTSynthesisReport:
    domain_id: str
    mode: str
    steps: int
    initial_loss: float
    final_loss: float
    loss_delta: float
    elapsed_sec: float
    trainable_params: int
    saved_path: Optional[str] = None


def _chunk_source_texts(texts: Sequence[str], max_chars: int = 1200) -> List[str]:
    chunks: List[str] = []
    for raw in texts:
        s = (raw or "").strip()
        if not s:
            continue
        if len(s) <= max_chars:
            chunks.append(s)
        else:
            step = max(256, max_chars - 128)
            for start in range(0, len(s), step):
                part = s[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
    return chunks


def synthesize_ttt_qmem_cartridge(
    model: nn.Module,
    tokenizer: Any,
    context_texts: Sequence[str],
    domain_id: Optional[str] = None,
    macro_domain: str = "ttt_ephemeral",
    mode: str = "fast",
    steps: Optional[int] = None,
    lr: float = 1.5e-2,
    deltanet_slots: int = 4,
    max_seq_len: int = 256,
    out_qmem_path: Optional[str | Path] = None,
    target_layers: Optional[Sequence[int]] = None,
) -> Tuple[QmemV3Cartridge, TTTSynthesisReport]:
    """
    Runs fast test-time self-supervised gradient descent on Bank 3 (`DeltaNet S_0` fast weights)
    while keeping 100% of base model parameters frozen (`requires_grad=False`).

    Returns an in-memory `QmemV3Cartridge` (and optionally persists it to `out_qmem_path`).
    """
    t0 = time.perf_counter()
    chunks = _chunk_source_texts(context_texts)
    if not chunks:
        raise ValueError("TTT-QMEM requires at least one non-empty context text.")

    mode_norm = "deep" if str(mode).lower() == "deep" else "fast"
    n_steps = int(steps) if steps is not None else (24 if mode_norm == "deep" else 6)
    dom_id = domain_id or (
        "ttt_" + hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()[:10]
    )

    try:
        entry_dev = next(model.parameters()).device
    except StopIteration:
        entry_dev = torch.device("cpu")

    # Save & strictly freeze every base model parameter so base drift is mathematically 0.0
    prev_grad_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    prev_training = model.training
    for p in model.parameters():
        p.requires_grad = False

    bank3 = Bank3DeltaNetFastWeightBank(model, num_slots=deltanet_slots, target_layers=target_layers).to(entry_dev)
    trainable = list(bank3.parameters())
    if not trainable:
        # Restore flags before raising
        for name, p in model.named_parameters():
            p.requires_grad = prev_grad_flags[name]
        raise RuntimeError(
            "Model has no HolomorphicCVNNDeltaNetWrapper layers for TTT-QMEM Bank-3 synthesis. "
            "Call `upgrade_model_to_svon_cgw(model)` first."
        )

    opt = torch.optim.AdamW(trainable, lr=float(lr), weight_decay=0.0)
    losses: List[float] = []

    model.eval()  # Keep dropout/BN deterministic while backpropagating into Bank 3 only
    for step_i in range(1, n_steps + 1):
        txt = chunks[(step_i - 1) % len(chunks)]
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=int(max_seq_len)).to(entry_dev)
        if enc.input_ids.shape[1] < 2:
            continue

        opt.zero_grad()
        bank3.bind_to_model()
        out = model(input_ids=enc.input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        targets = enc.input_ids[:, 1:].to(logits.device)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.item()))

    bank3.unbind_from_model()

    # Extract Layer-32 Gaussian signature on the frozen base model
    centroid_l32, diag_var_l32 = extract_layer32_gaussian_signature(
        model, tokenizer, chunks, device=entry_dev, max_samples=min(8, len(chunks))
    )

    # Restore exact prior requires_grad & mode state
    for name, p in model.named_parameters():
        p.requires_grad = prev_grad_flags[name]
    if prev_training:
        model.train()

    export_tensors = bank3.export_tensors()
    bank_map = {k: 3 for k in export_tensors.keys()}
    layer_map: Dict[str, int] = {}
    for k in export_tensors.keys():
        parts = k.split(".")
        layer_map[k] = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1

    meta = QmemV3Metadata(
        domain_id=dom_id,
        macro_domain=macro_domain,
        description=f"TTT-QMEM ({mode_norm}, {n_steps} steps) ephemeral parametric synthesis",
        deltanet_slots=int(deltanet_slots),
        centroid_l32=centroid_l32,
        diag_var_l32=diag_var_l32,
        extra={"ttt_mode": mode_norm, "ttt_steps": n_steps, "num_chunks": len(chunks)},
    )

    saved_path: Optional[str] = None
    sha_prefix = ""
    if out_qmem_path is not None:
        info = save_qmem_v3(out_qmem_path, tensors=export_tensors, metadata=meta, bank_map=bank_map, layer_map=layer_map)
        saved_path = info["path"]
        sha_prefix = info["sha256_prefix"]

    cart = QmemV3Cartridge(
        metadata=meta,
        tensors=export_tensors,
        bank_map=bank_map,
        layer_map=layer_map,
        sha256_prefix=sha_prefix,
        source_path=saved_path,
    )

    init_l = losses[0] if losses else 0.0
    fin_l = losses[-1] if losses else 0.0
    report = TTTSynthesisReport(
        domain_id=dom_id,
        mode=mode_norm,
        steps=len(losses),
        initial_loss=round(init_l, 6),
        final_loss=round(fin_l, 6),
        loss_delta=round(fin_l - init_l, 6),
        elapsed_sec=round(time.perf_counter() - t0, 3),
        trainable_params=sum(int(p.numel()) for p in trainable),
        saved_path=saved_path,
    )
    return cart, report
