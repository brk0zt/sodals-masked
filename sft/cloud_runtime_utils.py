"""
Cross-Platform Runtime & Model Loader (Windows 11 SAC + Linux / Kaggle T4 x2)
=============================================================================
Unifies local 8 GB Windows execution (optional Smart App Control shim, 4-bit NF4,
CPUEmbeddingWrapper) and cloud Linux Kaggle T4 x2 execution (32 GB dual-GPU FP16
with `device_map="auto"`, zero CPU offload, zero Windows-only DLL/ctypes calls).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

IS_WINDOWS = sys.platform == "win32"


def maybe_enable_sac_compat() -> bool:
    """Loads Windows 11 Smart App Control pure-Python compatibility only on Windows Py3.13+."""
    if IS_WINDOWS and sys.version_info >= (3, 13):
        import smart_app_control_compat  # noqa: F401
        return True
    return False


def configure_hf_env(online: Optional[bool] = None) -> None:
    """Sets HuggingFace offline/online env flags safely across Windows (offline) and Kaggle (online/dataset)."""
    if online is True:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        # Reset import-time cached offline flags if huggingface_hub/transformers were imported earlier
        hf_const = sys.modules.get("huggingface_hub.constants")
        if hf_const is not None and hasattr(hf_const, "HF_HUB_OFFLINE"):
            hf_const.HF_HUB_OFFLINE = False
        tf_hub = sys.modules.get("transformers.utils.hub")
        if tf_hub is not None and hasattr(tf_hub, "is_offline_mode"):
            tf_hub.is_offline_mode = lambda: False
    elif online is False:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        default_offline = "1" if IS_WINDOWS else "0"
        os.environ.setdefault("HF_HUB_OFFLINE", default_offline)
        os.environ.setdefault("TRANSFORMERS_OFFLINE", default_offline)


def prevent_sleep_if_windows() -> None:
    """Prevents Modern Standby on Windows; safe no-op on Linux / Kaggle."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception:
        pass


def get_primary_device(module) -> "torch.device":
    """Returns the true entry device of a single-GPU or multi-GPU (`device_map='auto'`) model."""
    import torch
    for attr_chain in (("model", "embed_tokens"), ("model", "language_model", "embed_tokens")):
        cur = module
        ok = True
        for name in attr_chain:
            if hasattr(cur, name):
                cur = getattr(cur, name)
            else:
                ok = False
                break
        if ok and hasattr(cur, "weight"):
            dev = cur.weight.device
            if dev.type != "cpu":
                return dev
    for p in module.parameters():
        if p.device.type != "cpu":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detect_gpu_profile() -> Dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False, "num_gpus": 0, "total_vram_gb": 0.0, "recommended_mode": "cpu"}
    n = torch.cuda.device_count()
    vrams = [torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) for i in range(n)]
    total = sum(vrams)
    rec = "fp16_multi_or_high_vram" if total >= 22.0 else "nf4_8gb_cpu_embed"
    return {
        "cuda": True,
        "num_gpus": n,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(n)],
        "total_vram_gb": round(total, 2),
        "recommended_mode": rec,
    }


def load_qwen_model_and_tokenizer(
    model_path: str,
    quantize_4bit: Optional[bool] = None,
    cpu_embed_offload: Optional[bool] = None,
    online: Optional[bool] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads Qwen 3.5 9B automatically adapted for either:
      - Kaggle T4 x2 (32 GB VRAM): Full FP16 (`device_map="auto"`, 0 quantization error)
      - Local 8 GB Windows GPU:    4-bit NF4 + CPUEmbeddingWrapper
    """
    maybe_enable_sac_compat()
    configure_hf_env(online=online)

    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    prof = detect_gpu_profile()
    if quantize_4bit is None:
        quantize_4bit = (prof["recommended_mode"] != "fp16_multi_or_high_vram")
    if cpu_embed_offload is None:
        cpu_embed_offload = bool(quantize_4bit and prof["total_vram_gb"] < 14.0)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
    }
    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = "auto" if prof["num_gpus"] > 1 else {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if cpu_embed_offload and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        class _CrossCPUEmbeddingWrapper(nn.Module):
            def __init__(self, cpu_embed, target_dev="cuda:0"):
                super().__init__()
                self.cpu_embed = cpu_embed
                self.target_dev = target_dev

            def forward(self, input_ids):
                return self.cpu_embed(input_ids.to("cpu")).to(device=self.target_dev, dtype=torch.float16)

        embed_cpu = model.model.embed_tokens.to(device="cpu", dtype=torch.float16)
        model.model.embed_tokens = _CrossCPUEmbeddingWrapper(embed_cpu, target_dev="cuda:0")
        target_dev = torch.device("cuda:0")
        type(model).device = property(lambda _m, d=target_dev: d)

    if hasattr(model, "lm_head"):
        lm_dev = model.lm_head.weight.device
        if lm_dev.type != "cpu":
            model.lm_head = model.lm_head.to(torch.float16)

    meta = {
        **prof,
        "quantize_4bit": bool(quantize_4bit),
        "cpu_embed_offload": bool(cpu_embed_offload),
        "entry_device": str(get_primary_device(model)),
    }
    return model, tokenizer, meta


def locate_decoder_layers(model: Any):
    """Unwraps PeftModel / LoraModel / Qwen3_5ForCausalLM to return the decoder `layers` list."""
    cur = model
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "layers") and isinstance(getattr(cur, "layers"), (list, tuple)) or (
            hasattr(cur, "layers") and hasattr(cur.layers, "__len__") and hasattr(cur.layers, "__getitem__")
        ):
            return list(cur.layers)
        if hasattr(cur, "language_model") and hasattr(cur.language_model, "layers"):
            return list(cur.language_model.layers)
        if hasattr(cur, "base_model") and getattr(cur, "base_model") is not cur:
            cur = cur.base_model
            continue
        if hasattr(cur, "model") and getattr(cur, "model") is not cur:
            cur = cur.model
            continue
        break
    raise AttributeError("Cannot locate decoder `layers` list on model.")


def mount_production_adapters(
    base_model: Any,
    d1_adapter_path: Optional[str] = None,
    d2_adapter_path: Optional[str] = None,
    svon_adapter_path: Optional[str] = None,
    d1_scale: float = 0.25,
    d2_scale: float = 1.00,
    svon_scale: float = 1.00,
    alpha: float = 0.20,
) -> Tuple[Any, Any, List[str]]:
    """
    Mounts production LoRA adapters (`D1: expert_general_math`, `D2: expert_abstract_algebra`,
    and optional `svon_identity`) onto `base_model` using `SoftLoraBlender` so cloud training
    and local 4-bit NF4 verification share the exact same multi-adapter topology as `sodals_serve.py`.
    """
    from peft import PeftModel
    from soft_lora_blender import SoftLoraBlender

    candidates = [
        ("expert_general_math", d1_adapter_path, float(d1_scale)),
        ("expert_abstract_algebra", d2_adapter_path, float(d2_scale)),
        ("svon_identity", svon_adapter_path, float(svon_scale)),
    ]
    active = [(name, path, sc) for name, path, sc in candidates if path and os.path.exists(path)]
    if not active:
        return base_model, None, []

    first_name, first_path, _ = active[0]
    if isinstance(base_model, PeftModel):
        model = base_model
        model.load_adapter(first_path, adapter_name=first_name)
    else:
        model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_name)

    for name, path, _ in active[1:]:
        model.load_adapter(path, adapter_name=name)

    managed_names = [name for name, _, _ in active]
    managed_scales = {name: sc for name, _, sc in active}
    blender = SoftLoraBlender(adapter_names=managed_names, base_scales=managed_scales)
    blender.enable_multi_adapter(model)
    blender.set_blend(model, alpha=alpha)
    return model, blender, managed_names

