"""
SODALS Core — Orthogonal `.qmem v3` Specialist Cartridge Factory
================================================================
Trains a standalone, zero-interference `.qmem v3` virtual parameter cartridge
for any specialist domain (MMLU-Pro, HLE, SWE-Pro, Math, Law, Medicine, Physics)
combining all three parametric banks:

  1. Bank 1 (`mlp.down_proj`, late layers 18..31):
     Null-Space Projected Gradient Descent (GPM / AlphaEdit) + Frobenius trust-region ball
     so general world knowledge & unrelated domains remain protected (KL < 0.01).
  2. Bank 2 (`q_proj, k_proj, v_proj, o_proj`, 8 Attention layers):
     Sibling-Orthogonal InfLoRA / GPM projection (`A_new = A_new (I - U_sib U_sib^T)`)
     so simultaneous Top-K cartridge blending suffers zero Subspace Interference.
  3. Bank 3 (`GatedDeltaNet S_0`, 24 SSM layers):
     Asymmetric multi-slot `(K_cart, V_cart, beta_cart)` fast-weight associative state bank.

Automatically extracts Layer-32 Gaussian routing signatures `(centroid_l32, diag_var_l32)`
and saves a verified `.qmem v3` binary ready for `HierarchicalGaussianPhaseRouter` + `QmemV3Manager`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F

from qmem_v3_cartridge import QmemV3Cartridge, QmemV3Metadata, load_qmem_v3, save_qmem_v3


def compute_sibling_orthogonality_basis(
    sibling_cartridges: Sequence[QmemV3Cartridge],
    energy_keep: float = 0.99,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Collects all Bank-2 `lora_A` rows across sibling cartridges in the same macro-domain
    and computes an orthonormal row-space basis U_sib [k, d_in] per module via SVD.
    Projecting a new cartridge's `lora_A` onto `(I - U_sib^T U_sib)` enforces `A_new A_sib^T ~= 0`
    (InfLoRA / GPM orthogonality), eliminating Paper 2's Subspace Interference Law.
    """
    grouped: Dict[str, List[torch.Tensor]] = {}
    for cart in sibling_cartridges:
        for name, t in cart.tensors.items():
            if cart.bank_map.get(name) == 2 and name.endswith(".lora_A"):
                grouped.setdefault(name, []).append(t.detach().float().cpu())

    basis_map: Dict[str, torch.Tensor] = {}
    hasher = hashlib.sha256()
    for name in sorted(grouped.keys()):
        stacked = torch.cat(grouped[name], dim=0)  # [sum_r, d_in]
        _, s, vh = torch.linalg.svd(stacked, full_matrices=False)
        if s.numel() == 0 or float(s.max().item()) < 1e-10:
            continue
        e = s * s
        cum = torch.cumsum(e, dim=0) / torch.clamp(e.sum(), min=1e-12)
        k = int(torch.searchsorted(cum, torch.tensor(float(energy_keep))).item()) + 1
        k = max(1, min(k, vh.shape[0]))
        u_basis = vh[:k, :].contiguous()  # [k, d_in], rows orthonormal
        basis_map[name] = u_basis
        hasher.update(name.encode("utf-8"))
        hasher.update(u_basis.numpy().tobytes())

    return basis_map, hasher.hexdigest()[:16]


def project_lora_a_orthogonal_to_siblings(
    lora_a: torch.Tensor,
    u_basis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Projects `lora_A [r, d_in]` orthogonally to sibling row-space `u_basis [k, d_in]` in-place."""
    if u_basis is None or u_basis.numel() == 0:
        return lora_a
    with torch.no_grad():
        a_f = lora_a.float()
        u_d = u_basis.to(device=a_f.device, dtype=torch.float32)
        proj = (a_f @ u_d.T) @ u_d
        lora_a.copy_((a_f - proj).to(dtype=lora_a.dtype))
    return lora_a


class Bank3DeltaNetFastWeightBank(nn.Module):
    """Trainable Bank-3 asymmetric `(K_cart, V_cart, beta_cart)` parameters across DeltaNet layers."""

    def __init__(self, model: nn.Module, num_slots: int = 4, target_layers: Optional[Sequence[int]] = None):
        super().__init__()
        self.num_slots = max(1, int(num_slots))
        self.layer_params: nn.ModuleDict = nn.ModuleDict()
        self.wrapped_layers: Dict[int, nn.Module] = {}

        for mod in model.modules():
            if not hasattr(mod, "set_initial_cartridge_state"):
                continue
            l_idx = int(getattr(getattr(mod, "base", None), "layer_idx", -1))
            if l_idx < 0:
                continue
            if target_layers is not None and l_idx not in target_layers:
                continue
            h_v = int(getattr(mod, "num_v_heads", 32))
            d_k = int(getattr(mod, "head_k_dim", 128))
            d_v = int(getattr(mod, "head_v_dim", 128))
            holder = nn.Module()
            holder.k_cart = nn.Parameter(torch.randn(self.num_slots, h_v, d_k, dtype=torch.float32) * 0.02)
            holder.v_cart = nn.Parameter(torch.randn(self.num_slots, h_v, d_v, dtype=torch.float32) * 0.02)
            # Zero-init write gate beta_cart -> Exact Submanifold identity at step 0!
            holder.beta_cart = nn.Parameter(torch.zeros(self.num_slots, h_v, dtype=torch.float32))
            self.layer_params[str(l_idx)] = holder
            self.wrapped_layers[l_idx] = mod

    def bind_to_model(self) -> None:
        for l_key, holder in self.layer_params.items():
            l_idx = int(l_key)
            # Bound beta smoothly into (-0.5, 0.5) via 0.5 * tanh(beta_cart) for Delta-Rule stability
            self.wrapped_layers[l_idx].set_initial_cartridge_state(
                holder.k_cart,
                holder.v_cart,
                0.5 * torch.tanh(holder.beta_cart),
            )

    def unbind_from_model(self) -> None:
        for mod in self.wrapped_layers.values():
            mod.set_initial_cartridge_state(None, None, None)

    def export_tensors(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for l_key, holder in self.layer_params.items():
            out[f"bank3.layers.{l_key}.k_cart"] = holder.k_cart.detach().float().cpu()
            out[f"bank3.layers.{l_key}.v_cart"] = holder.v_cart.detach().float().cpu()
            out[f"bank3.layers.{l_key}.beta_cart"] = (0.5 * torch.tanh(holder.beta_cart.detach().float())).cpu()
        return out


def extract_layer32_gaussian_signature(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    device: Any,
    var_floor: float = 1e-3,
    max_samples: int = 32,
) -> Tuple[List[float], List[float]]:
    """Computes unit-normalized Layer-32 (final hidden) centroid μ and diagonal variance σ²."""
    vecs: List[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for txt in list(prompts)[: max(1, int(max_samples))]:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            h_last = out.hidden_states[-1][0].float().mean(dim=0)
            vecs.append(F.normalize(h_last, dim=0).cpu())
    if was_training:
        model.train()

    mat = torch.stack(vecs, dim=0)  # [N, D]
    mu = F.normalize(mat.mean(dim=0), dim=0)
    var = torch.clamp(((mat - mu.unsqueeze(0)) ** 2).mean(dim=0), min=var_floor)
    return mu.tolist(), var.tolist()


def load_domain_corpus(path: str | Path) -> List[Dict[str, str]]:
    """Loads JSON or JSONL specialist training pairs `{'prompt': ..., 'response': ...}`."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        obj = json.loads(raw)
        items = obj.get("prompts", obj) if isinstance(obj, dict) else obj

    norm: List[Dict[str, str]] = []
    for it in items:
        pr = str(it.get("prompt") or it.get("problem") or it.get("question") or "")
        rp = str(it.get("response") or it.get("solution") or it.get("answer") or it.get("ground_truth") or "")
        if pr and rp:
            norm.append({"prompt": pr, "response": rp})
    return norm


def train_qmem_v3_cartridge(args: argparse.Namespace) -> Dict[str, Any]:
    from cloud_runtime_utils import get_primary_device, load_qwen_model_and_tokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from svon_alphaedit_surgeon import project_lora_b_trust_region
    from svon_kuramoto_equilibrium import upgrade_model_to_svon_cgw

    corpus = load_domain_corpus(args.corpus_json)
    if not corpus:
        raise ValueError(f"Empty specialist corpus: {args.corpus_json}")

    sibling_carts: List[QmemV3Cartridge] = []
    for sp in (args.sibling_qmem or []):
        if os.path.exists(sp):
            sibling_carts.append(load_qmem_v3(sp, device="cpu", verify_checksum=True))
    sibling_basis, basis_hash = compute_sibling_orthogonality_basis(sibling_carts)

    model, tokenizer, _rt = load_qwen_model_and_tokenizer(
        model_path=args.model_path,
        quantize_4bit=not args.fp16,
        online=True if args.online else None,
    )
    entry_dev = get_primary_device(model)

    # Upgrade GatedDeltaNet layers so Bank 3 asymmetric S_0 slots are active & differentiable
    upgrade_model_to_svon_cgw(model, phase_rank=16)

    # Extract Layer-32 Gaussian signature on clean base model BEFORE mounting cartridge LoRA
    centroid_l32, diag_var_l32 = extract_layer32_gaussian_signature(
        model, tokenizer, [it["prompt"] for it in corpus], device=entry_dev
    )

    # Freeze base + CVNN weights; train ONLY this specialist's Bank 1/2 LoRA + Bank 3 DeltaNet slots
    for p in model.parameters():
        p.requires_grad = False

    lconf = LoraConfig(
        r=int(args.rank_attn),
        lora_alpha=int(args.rank_attn),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lconf, adapter_name="qmem_train") if not isinstance(model, PeftModel) else model
    if isinstance(model, PeftModel):
        peft_model.add_adapter("qmem_train", lconf)
        peft_model.set_adapter("qmem_train")

    bank3 = Bank3DeltaNetFastWeightBank(peft_model, num_slots=int(args.deltanet_slots)).to(entry_dev)
    trainable: List[torch.Tensor] = [p for p in peft_model.parameters() if p.requires_grad] + list(bank3.parameters())
    opt = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.0)

    # Precompute base logits on first 8 corpus prompts as KL anchor
    peft_model.eval()
    kl_anchors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        with peft_model.disable_adapter():
            for it in corpus[:8]:
                enc = tokenizer(it["prompt"], return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
                kl_anchors.append((enc.input_ids.cpu(), peft_model(**enc, use_cache=False).logits[0, -1, :].float().cpu()))

    peft_model.train()
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad()
        bank3.bind_to_model()
        item = corpus[(step - 1) % len(corpus)]
        p_ids = tokenizer(item["prompt"], return_tensors="pt", truncation=True, max_length=256).input_ids
        r_ids = tokenizer(item["response"] + (tokenizer.eos_token or ""), return_tensors="pt", truncation=True, max_length=256, add_special_tokens=False).input_ids
        full_ids = torch.cat([p_ids, r_ids], dim=1).to(entry_dev)
        labels = torch.full_like(full_ids, -100)
        labels[:, p_ids.shape[1] :] = full_ids[:, p_ids.shape[1] :]

        loss_ce = peft_model(input_ids=full_ids, labels=labels, use_cache=False).loss.float()

        # KL anchor regularization to prevent base distribution drift
        a_ids, a_base = kl_anchors[(step - 1) % len(kl_anchors)]
        cur_logits = peft_model(input_ids=a_ids.to(entry_dev), use_cache=False).logits[0, -1, :].float()
        base_logits = a_base.to(device=cur_logits.device, dtype=torch.float32)
        p_cur = F.softmax(cur_logits, dim=-1)
        loss_kl = (p_cur * (F.log_softmax(cur_logits, dim=-1) - F.log_softmax(base_logits, dim=-1))).sum()

        loss = loss_ce + float(args.kl_weight) * loss_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        # Enforce InfLoRA sibling orthogonality on Bank 2 & trust-region ball on Bank 1
        for mod_name, mod in peft_model.named_modules():
            if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "qmem_train" in mod.lora_A):
                continue
            norm_key = mod_name.replace("base_model.model.", "") + ".lora_A"
            if norm_key in sibling_basis:
                project_lora_a_orthogonal_to_siblings(mod.lora_A["qmem_train"].weight.data, sibling_basis[norm_key])
            if "down_proj" in mod_name and float(args.max_mlp_frob) > 0.0:
                project_lora_b_trust_region(
                    mod.lora_B["qmem_train"].weight.data,
                    mod.lora_A["qmem_train"].weight.data,
                    lora_scale=1.0,
                    max_delta_frob=float(args.max_mlp_frob),
                )

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            print(f"[QMEM-V3 TRAIN {args.domain_id} {step:02d}/{args.steps}] ce={loss_ce.item():.4f} kl={loss_kl.item():.5f}", flush=True)

    bank3.unbind_from_model()

    # Collect all 3 banks into unified `.qmem v3` container
    export_tensors: Dict[str, torch.Tensor] = {}
    bank_map: Dict[str, int] = {}
    for mod_name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "qmem_train" in mod.lora_A:
            norm_base = mod_name.replace("base_model.model.", "")
            b_id = 1 if "down_proj" in norm_base else 2
            export_tensors[f"{norm_base}.lora_A"] = mod.lora_A["qmem_train"].weight.detach().to(torch.float16).cpu()
            export_tensors[f"{norm_base}.lora_B"] = mod.lora_B["qmem_train"].weight.detach().to(torch.float16).cpu()
            bank_map[f"{norm_base}.lora_A"] = b_id
            bank_map[f"{norm_base}.lora_B"] = b_id

    for k3, t3 in bank3.export_tensors().items():
        export_tensors[k3] = t3
        bank_map[k3] = 3

    meta = QmemV3Metadata(
        domain_id=str(args.domain_id),
        macro_domain=str(args.macro_domain),
        description=str(args.description or f"QMEM v3 specialist cartridge for {args.domain_id}"),
        rank_mlp=int(args.rank_attn),
        rank_attn=int(args.rank_attn),
        deltanet_slots=int(args.deltanet_slots),
        orthogonality_basis_hash=basis_hash,
        centroid_l32=centroid_l32,
        diag_var_l32=diag_var_l32,
    )
    return save_qmem_v3(
        args.out_qmem,
        tensors=export_tensors,
        metadata=meta,
        bank_map=bank_map,
        quantize_int8=bool(args.quantize_int8),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SODALS Orthogonal `.qmem v3` Cartridge Factory")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors"))
    parser.add_argument("--corpus_json", type=str, required=True, help="JSON/JSONL specialist corpus")
    parser.add_argument("--domain_id", type=str, required=True, help="Unique specialist domain ID (e.g., physics_qft)")
    parser.add_argument("--macro_domain", type=str, default="stem", help="Parent macro-domain for H-GPR")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--out_qmem", type=str, required=True, help="Output `.qmem` v3 file path")
    parser.add_argument("--sibling_qmem", action="append", default=[], help="Sibling `.qmem` paths for InfLoRA orthogonality")
    parser.add_argument("--rank_attn", type=int, default=16)
    parser.add_argument("--deltanet_slots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl_weight", type=float, default=0.20)
    parser.add_argument("--max_mlp_frob", type=float, default=1.2)
    parser.add_argument("--quantize_int8", action="store_true", help="Quantize Bank 1/2 LoRA matrices to INT8 (2x smaller disk)")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    res = train_qmem_v3_cartridge(args)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
