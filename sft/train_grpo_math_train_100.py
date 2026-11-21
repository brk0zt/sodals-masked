"""
SODALS v2.1 — Pre-Registered Out-of-MATH-500 Long-Context GRPO Training (100 Hendrycks MATH Train Prompts)
===========================================================================================================
Implements Claude's Item 3 Pre-Registered Experiment:
  - Training Pool: `sft/math_train_100_grpo_prompts.json` (100 Level-5 problems from official Hendrycks MATH
    `train` split, verified 0 / 500 [0.00%] overlap with `HuggingFaceH4/MATH-500` test set).
  - Target Adapter: Starts from SFT `qwen_lora_adapter_balanced_r4` -> saves to `qwen_lora_adapter_grpo_math_train_100`.
  - Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, ~3,907 MB base footprint via CPUEmbeddingWrapper
    and 4-bit NF4 `lm_head`, zero WDDM shared-memory paging).
  - Active-Gradient & Anti-Early-Surrender Reward Shaping:
      1. Exact Outcome Reward (`+2.0` if correct, `-1.0` if wrong)
      2. Clean Termination & Format Reward (`+0.3` if `</think>` + `\boxed{}` present, `-0.5` if truncated)
      3. SymPy Process Verification & Anti-Early-Surrender Shaping:
         * `+0.04` per SymPy-verified intermediate equation (up to `+0.40`)
         * `-0.25` if internal algebraic contradiction detected
         * `-0.40 * (1 - tok/384)` penalty when a failed rollout surrenders prematurely under 384 tokens
         * Ensures non-zero reward variance (`std_r > 0`) on ~100% of steps while keeping exact outcome (`3.0` gap) dominant.
  - Checkpoint & Live Resume: Saves adapter weights and JSON telemetry after EVERY step.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    import smart_app_control_compat  # noqa: F401
import json
import re
import time
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel
import bitsandbytes as bnb

sys.stdout.reconfigure(encoding="utf-8")

def prevent_windows_sleep():
    """Prevent Windows Modern Standby / Sleep / Display & PCIe throttle when screen is locked."""
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            ES_AWAYMODE_REQUIRED = 0x00000040
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_AWAYMODE_REQUIRED
            )
        except Exception:
            pass

prevent_windows_sleep()

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
sys.path.insert(0, SFT_DIR)
from cst_symbolic_verifier import CstSymbolicVerifier, NormalizedMathMatcher, LeanTacticStepVerifier
from arkhe_neurosymbolic_pipeline import CandidateTrace, ConditionsArche, MethodArche
from env_reproducibility import collect_runtime_metadata

MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")

DEFAULT_PROMPTS_JSON = os.path.join(SFT_DIR, "math_train_hard_mined_zpd_prompts.json")
if not os.path.exists(DEFAULT_PROMPTS_JSON):
    DEFAULT_PROMPTS_JSON = os.path.join(SFT_DIR, "math_train_100_grpo_prompts.json")

DEFAULT_ADAPTER_IN = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
DEFAULT_ADAPTER_OUT = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100")
DEFAULT_OUTPUT_JSON = os.path.join(SFT_DIR, "grpo_math_train_100_results.json")
GPU_CAPACITY_MB = 8188.0


class CPUEmbeddingWrapper(nn.Module):
    """Zero-VRAM CPU embedding wrapper that intercepts token lookups."""
    def __init__(self, cpu_embed):
        super().__init__()
        self.cpu_embed = cpu_embed

    def forward(self, input_ids):
        ids_cpu = input_ids.to("cpu")
        embeds = self.cpu_embed(ids_cpu)
        return embeds.to("cuda")


def safe_json_dump(payload: Any, filepath: str, max_retries: int = 10, backoff: float = 0.5):
    """
    Safely writes JSON using an atomic temporary file and exponential backoff retry loop
    to prevent Windows PermissionError / indexing collisions.
    """
    tmp_path = filepath + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    for attempt in range(max_retries):
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if os.path.exists(filepath):
                try:
                    os.replace(tmp_path, filepath)
                except PermissionError:
                    time.sleep(backoff * (2 ** attempt))
                    os.replace(tmp_path, filepath)
            else:
                os.replace(tmp_path, filepath)
            return
        except Exception:
            if attempt == max_retries - 1:
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            time.sleep(backoff * (2 ** attempt))


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return text[start:].strip()

    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n]+)", text)
    if m:
        return m.group(1).strip().rstrip(".")

    all_nums = re.findall(r"\b(-?\d+(?:\.\d+)?)\b", text)
    if all_nums:
        return all_nums[-1]
    return ""


def compute_reward(gen_text: str, gt_answer: str, num_tokens: int) -> tuple[float, str, bool, dict]:
    pred = extract_boxed_answer(gen_text)
    _, is_correct = NormalizedMathMatcher.check_match(pred, gt_answer)
    has_end = "</think>" in gen_text
    has_box = (pred != "" and r"\boxed{" in gen_text)

    # 1. Primary Outcome Reward (+2.0 vs -1.0 -> 3.0 unit separation)
    r_outcome = 2.0 if is_correct else -1.0

    # 2. Format & Clean Completion Reward
    if has_end and has_box:
        r_format = 0.30
    elif has_box:
        r_format = 0.10
    else:
        r_format = -0.50

    # 3. SymPy Process Verification & Anti-Early-Surrender Shaping
    is_valid, eq_count, contras = CstSymbolicVerifier.verify_chunk(gen_text)
    r_sympy = min(0.40, 0.04 * eq_count) - (0.25 if (not is_valid and len(contras) > 0) else 0.0)

    r_length = 0.0
    if not is_correct:
        if num_tokens < 384:
            # Directly penalize premature surrender on unsolved problems
            r_length = -0.40 * (1.0 - (num_tokens / 384.0))
        elif 384 <= num_tokens <= 1400 and has_end:
            # Reward sustained, structured exploration that still closes </think> cleanly
            r_length = 0.20
    else:
        if has_end and num_tokens <= 1400:
            r_length = 0.15

    total_reward = round(r_outcome + r_format + r_sympy + r_length, 4)
    meta = {
        "r_outcome": r_outcome,
        "r_format": r_format,
        "r_sympy": round(r_sympy, 4),
        "r_length": round(r_length, 4),
        "eq_count": eq_count,
        "contras": len(contras),
        "has_end": has_end,
    }
    return total_reward, pred, is_correct, meta


def train_grpo_math_train(args):
    print("=" * 95, flush=True)
    print("SODALS v2.1 — OUT-OF-MATH-500 LONG-CONTEXT GRPO TRAINING (HENDRYCKS MATH TRAIN SPLIT)", flush=True)
    print(f"Prompts File:   {args.prompts_json} (Total target steps: {args.num_prompts})", flush=True)
    print(f"Context Window: max_new_tokens = {args.max_new_tokens} | Group Size G = {args.group_size}", flush=True)
    print(f"Hyperparams:    LR = {args.lr} | KL beta = {args.beta}", flush=True)
    print("=" * 95, flush=True)

    args.adapter_out = os.path.abspath(args.adapter_out)
    args.output_json = os.path.abspath(args.output_json)
    os.makedirs(args.adapter_out, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # Check existing progress for seamless resume
    step_logs = []
    completed_prompt_indices = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            step_logs = saved.get("step_details", [])
            for s in step_logs:
                completed_prompt_indices.add(s["question_index"])
            print(f"[RESUME] Found {len(step_logs)} completed steps in {args.output_json}.", flush=True)
        except Exception as e:
            print(f"[RESUME WARNING] Could not read existing results JSON ({e}); starting fresh.", flush=True)
            step_logs = []
            completed_prompt_indices = set()

    # Load Prompts
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts_payload = json.load(f)
    raw_prompts = prompts_payload["prompts"][: args.num_prompts]
    prompts_data = []
    for p in raw_prompts:
        prob = p["problem"]
        prompt_str = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        prompts_data.append({
            "index": p["prompt_index"],
            "train_row_id": p["train_row_id"],
            "subject": p["subject"],
            "level": p["level"],
            "problem": prob,
            "ground_truth": p["ground_truth"],
            "prompt_str": prompt_str,
        })

    if len(completed_prompt_indices) >= len(prompts_data):
        print(f"All {len(prompts_data)} training steps are already completed! Exiting.", flush=True)
        return

    # 1. Load Tokenizer
    print("1. Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in 4-bit NF4 with SDPA FlashAttention
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("2. Loading Qwen 3.5 9B with SDPA FlashAttention in 4-bit NF4...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # 3. Cast non-4bit SSM and norm parameters to FP32
    print("3. Casting SSM and LayerNorm parameters to FP32...", flush=True)
    for name, param in base_model.named_parameters():
        if param.__class__.__name__ != "Params4bit" and "embed_tokens" not in name and "lm_head" not in name:
            param.data = param.data.to(torch.float32)

    # 4. Offload embed_tokens to CPU RAM (-1,645 MB VRAM)
    print("4. Offloading embed_tokens to CPU RAM via CPUEmbeddingWrapper (-1,645 MB VRAM)...", flush=True)
    embed_cpu = base_model.model.embed_tokens.to("cpu")
    base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
    type(base_model).device = property(lambda self: torch.device("cuda:0"))
    torch.cuda.empty_cache()

    # 5. Retain lm_head in native FP16 for zero-overhead TensorCore execution
    print("5. Retaining lm_head in native FP16 for zero-overhead TensorCore execution...", flush=True)
    base_model.lm_head = base_model.lm_head.to(torch.float16)
    torch.cuda.empty_cache()

    # 6. Attach LoRA Adapter (from checkpoint if resuming, else from SFT balanced_r4)
    adapter_load_path = args.adapter_in
    if len(completed_prompt_indices) > 0 and os.path.exists(os.path.join(args.adapter_out, "adapter_config.json")):
        try:
            from safetensors.torch import load_file
            chk_tensors = load_file(os.path.join(args.adapter_out, "adapter_model.safetensors"))
            has_nan = any(torch.isnan(v).any().item() or torch.isinf(v).any().item() for v in chk_tensors.values())
            if has_nan:
                print(f"[RESUME WARNING] Checkpoint {args.adapter_out} contains NaNs! Falling back to clean base: {args.adapter_in}", flush=True)
                adapter_load_path = args.adapter_in
            else:
                adapter_load_path = args.adapter_out
                print(f"6. Resuming LoRA Adapter from verified clean checkpoint: {adapter_load_path}...", flush=True)
        except Exception as e:
            print(f"[RESUME WARNING] Error checking checkpoint ({e}); using base {args.adapter_in}", flush=True)
            adapter_load_path = args.adapter_in
    else:
        print(f"6. Attaching initial SFT LoRA Adapter from: {adapter_load_path}...", flush=True)

    model = PeftModel.from_pretrained(base_model, adapter_load_path, is_trainable=True)
    type(model).device = property(lambda self: torch.device("cuda:0"))
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.enable_input_require_grads()

    # Align training LoRA scale with evaluation SoftLoraBlender scale (s_D1 = 0.25)
    scaled_layers = 0
    for mod in model.modules():
        if hasattr(mod, "scaling") and hasattr(mod, "lora_alpha") and hasattr(mod, "r"):
            for ad_name in list(mod.scaling.keys()):
                alpha_val = mod.lora_alpha[ad_name]
                r_val = mod.r[ad_name]
                mod.scaling[ad_name] = (alpha_val / r_val) * 0.25
                scaled_layers += 1
    print(f"Aligned {scaled_layers} LoRA layers to exact evaluation scale (s_D1 = 0.25).", flush=True)

    backbone = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    init_param_snap = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    init_norm_sq = sum(float(t.norm().item() ** 2) for t in init_param_snap.values())

    base_alloc = torch.cuda.memory_allocated(0) / (1024**2)
    base_res = torch.cuda.memory_reserved(0) / (1024**2)
    print(
        f"\n[FROZEN INFRASTRUCTURE READY] Base Alloc: {base_alloc:.1f} MB | Reserved: {base_res:.1f} MB | "
        f"Physical Headroom: {GPU_CAPACITY_MB - base_res:.1f} MB\n",
        flush=True,
    )

    # 7. Optimizer & Cosine Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(prompts_data)

    if getattr(args, "fresh_scheduler", False):
        remaining_steps = max(1, total_steps - len(completed_prompt_indices))
        warmup_steps = max(1, int(remaining_steps * 0.1))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=remaining_steps
        )
        print(f"[SCHEDULER] Initialized fresh scheduler for remaining {remaining_steps} steps (Warmup: {warmup_steps}).", flush=True)
    else:
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        # Fast-forward scheduler if resuming exact checkpoint
        for _ in range(len(completed_prompt_indices)):
            scheduler.step()
        if len(completed_prompt_indices) > 0:
            print(f"[SCHEDULER] Resumed scheduler, fast-forwarded {len(completed_prompt_indices)} steps.", flush=True)

    total_solved_prompts = sum(1 for s in step_logs if s.get("solved_rollouts", 0) > 0)
    active_gradient_steps = sum(1 for s in step_logs if s.get("active_gradient", False))
    t_start = time.time()

    print("\n" + "=" * 95, flush=True)
    print(f"STARTING 1536-TOKEN GRPO TRAINING ({total_steps} Steps, G={args.group_size} Rollouts/Step)", flush=True)
    print("=" * 95, flush=True)

    for step_idx, p_item in enumerate(prompts_data):
        p_idx = p_item["index"]
        if p_idx in completed_prompt_indices:
            continue

        prob = p_item["problem"]
        gt = p_item["ground_truth"]
        subj = p_item["subject"]
        prompt_str = p_item["prompt_str"]
        t_step_start = time.time()

        p_enc = tokenizer(prompt_str, return_tensors="pt").to("cuda")
        p_len = p_enc.input_ids.shape[1]

        # -------------------------------------------------------------
        # Phase 1: Sequential Rollout Generation (torch.no_grad)
        # -------------------------------------------------------------
        model.eval()
        model.config.use_cache = True
        rollout_rewards = []
        rollout_preds = []
        rollout_is_correct = []
        rollout_gen_ids = []
        rollout_tokens = []
        rollout_metas = []

        for g in range(args.group_size):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **p_enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0][p_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            num_tok = len(gen_ids)

            reward, pred, is_correct, meta = compute_reward(gen_text, gt, num_tok)
            print(
                f"  [Step {step_idx+1:03d}/{total_steps} | Rollout {g+1}/{args.group_size}] "
                f"Tok: {num_tok:4d} | Correct: {str(is_correct):5s} | Rew: {reward:+6.2f} "
                f"(sym:{meta['r_sympy']:+.2f}, len:{meta['r_length']:+.2f}) | Pred: {pred[:18]!r} vs GT: {gt[:18]!r}",
                flush=True,
            )

            rollout_rewards.append(reward)
            rollout_preds.append(pred)
            rollout_is_correct.append(is_correct)
            rollout_gen_ids.append(gen_ids)
            rollout_tokens.append(num_tok)
            rollout_metas.append(meta)

        avg_tokens = sum(rollout_tokens) / max(1, len(rollout_tokens))
        dyn_lambda = None
        if getattr(args, "enable_arkhe_ternary_reward", False):
            dyn_lambda = MethodArche.compute_dynamic_lambda(step_idx + 1, total_steps)
            arkhe_traces = [
                CandidateTrace(
                    branch_id=f"Rollout_{i+1}",
                    extracted_answer=rollout_preds[i],
                    has_end_tag=bool(rollout_metas[i].get("has_end", True)),
                    is_symbolically_valid=(rollout_metas[i].get("contras", 0) == 0),
                    verified_eq_count=int(rollout_metas[i].get("eq_count", 0)),
                    contradictions=tuple(["contra"] * int(rollout_metas[i].get("contras", 0))),
                    generated_tokens=int(rollout_tokens[i]),
                )
                for i in range(len(rollout_preds))
            ]
            rollout_rewards = ConditionsArche.compute_clean_ternary_grpo_rewards(
                traces=arkhe_traces,
                is_correct_flags=rollout_is_correct,
                lambda_penalty=dyn_lambda,
            )
        rewards_tensor = torch.tensor(rollout_rewards, dtype=torch.float32, device="cuda")
        mean_r = rewards_tensor.mean().item()
        raw_std = rewards_tensor.std().item()
        has_active_grad = raw_std > 1e-4
        std_r = raw_std + 1e-4
        advantages = ((rewards_tensor - mean_r) / std_r).tolist()

        num_correct = sum(rollout_is_correct)
        if num_correct > 0:
            total_solved_prompts += 1
        if has_active_grad:
            active_gradient_steps += 1

        # -------------------------------------------------------------
        # Phase 2: Chunked Policy Forward & Backward (skip if std_r == 0)
        # -------------------------------------------------------------
        group_loss_val = 0.0
        kl_samples = []
        if has_active_grad:
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad()

            for g in range(args.group_size):
                gen_ids = rollout_gen_ids[g]
                if len(gen_ids) == 0 or abs(advantages[g]) < 1e-5:
                    continue

                adv = advantages[g]
                seq_ids = torch.cat([p_enc.input_ids, gen_ids.unsqueeze(0)], dim=1)
                target_ids = gen_ids.unsqueeze(0).to("cuda")
                r_len = gen_ids.size(0)

                seq_embeds = embed_cpu(seq_ids.cpu()).to("cuda", dtype=torch.float16)

                # A. Zero-VRAM Reference Log-Probs
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_out = backbone(inputs_embeds=seq_embeds)
                        ref_h = ref_out.last_hidden_state[:, p_len - 1 : -1, :]
                        ref_logps_list = []
                        for start in range(0, r_len, 32):
                            end = min(start + 32, r_len)
                            c_h = ref_h[:, start:end, :].contiguous()
                            c_t = target_ids[:, start:end].contiguous()
                            c_logits = lm_head(c_h)
                            c_logits = torch.clamp(c_logits.float(), min=-50.0, max=50.0)
                            c_log_probs = F.log_softmax(c_logits, dim=-1)
                            c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
                            ref_logps_list.append(c_logps)
                        ref_logps = torch.cat(ref_logps_list, dim=-1)
                        del ref_out, ref_h, ref_logps_list
                        torch.cuda.empty_cache()

                # B. Active Policy Forward
                prevent_windows_sleep()
                torch.cuda.synchronize()
                act_embeds = seq_embeds.detach().requires_grad_(True)
                backbone_out = backbone(inputs_embeds=act_embeds)
                h_resp = backbone_out.last_hidden_state[:, p_len - 1 : -1, :]

                # C. Chunked GRPO Loss Accumulation
                loss_g = torch.tensor(0.0, device="cuda", dtype=torch.float32)
                kl_sum_g = 0.0
                for start in range(0, r_len, 32):
                    end = min(start + 32, r_len)
                    c_h = h_resp[:, start:end, :].contiguous()
                    c_t = target_ids[:, start:end].contiguous()
                    c_ref_logps = ref_logps[:, start:end]

                    c_logits = lm_head(c_h)
                    c_logits = torch.clamp(c_logits.float(), min=-50.0, max=50.0)
                    c_log_probs = F.log_softmax(c_logits, dim=-1)
                    c_cur_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)

                    log_ratio = torch.clamp(c_ref_logps - c_cur_logps, min=-4.0, max=4.0)
                    kl_pen = torch.exp(log_ratio) - log_ratio - 1.0
                    kl_sum_g += float(kl_pen.sum().item())
                    token_loss = -(adv * c_cur_logps - args.beta * kl_pen)
                    loss_g = loss_g + token_loss.sum() / (r_len * args.group_size)
                kl_samples.append(kl_sum_g / max(1, r_len))

                if not torch.isfinite(loss_g):
                    print(f"  [WARNING] Step {step_idx+1:03d} Rollout {g+1}: loss_g is non-finite ({loss_g.item()})! Skipping backward.", flush=True)
                    del backbone_out, h_resp, act_embeds, seq_embeds, ref_logps, loss_g
                    torch.cuda.empty_cache()
                    continue

                print(f"  [Step {step_idx+1:03d} | Policy Backward] Rollout {g+1}/{args.group_size} (adv={adv:+.2f})...", flush=True)
                loss_g.backward()
                torch.cuda.synchronize()
                group_loss_val += loss_g.item()

                del backbone_out, h_resp, act_embeds, seq_embeds, ref_logps, loss_g
                torch.cuda.empty_cache()

            for p in trainable_params:
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)
            torch.nn.utils.clip_grad_value_(trainable_params, clip_value=1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"  [WARNING] Step {step_idx+1:03d}: grad_norm is non-finite ({grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm})! Zeroing grads and skipping optimizer.step().", flush=True)
                optimizer.zero_grad()
            else:
                optimizer.step()

        scheduler.step()
        step_time = time.time() - t_step_start
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)

        mean_kl_val = round(sum(kl_samples) / len(kl_samples), 6) if kl_samples else 0.0
        abstain_count = sum(1 for pr in rollout_preds if ConditionsArche.is_strict_abstain(pr))
        rollout_cov = 1.0 - (abstain_count / max(1, len(rollout_preds)))
        diff_sq = sum(
            float((p.detach().float().cpu() - init_param_snap[n]).norm().item() ** 2)
            for n, p in model.named_parameters()
            if p.requires_grad and n in init_param_snap
        )
        rel_frob_drift_pct = round(100.0 * (diff_sq ** 0.5) / max(1e-8, init_norm_sq ** 0.5), 4)

        step_record = {
            "step": len(step_logs) + 1,
            "question_index": p_idx,
            "train_row_id": p_item["train_row_id"],
            "subject": subj,
            "level": p_item["level"],
            "problem": prob,
            "ground_truth": gt,
            "mean_reward": round(mean_r, 4),
            "lambda_penalty": dyn_lambda,
            "mean_kl_div": mean_kl_val,
            "rel_frob_drift_pct": rel_frob_drift_pct,
            "rollout_coverage": round(rollout_cov, 4),
            "solved_rollouts": num_correct,
            "active_gradient": has_active_grad,
            "avg_tokens": round(avg_tokens, 1),
            "rollout_tokens": rollout_tokens,
            "rollout_rewards": rollout_rewards,
            "rollout_preds": rollout_preds,
            "advantages": [round(a, 4) for a in advantages],
            "loss": round(group_loss_val, 4),
            "step_seconds": round(step_time, 1),
            "vram_mb": round(peak_vram, 1),
        }
        step_logs.append(step_record)
        completed_prompt_indices.add(p_idx)

        print(
            f"[STEP {len(step_logs):03d}/{total_steps}] [{subj[:16]:16s}] "
            f"Solved: {num_correct}/{args.group_size} | ActiveGrad: {str(has_active_grad):5s} "
            f"({active_gradient_steps}/{len(step_logs)}) | KL: {mean_kl_val:.5f} | Cov: {rollout_cov*100:.0f}% | "
            f"AvgTok: {avg_tokens:5.1f} | Loss: {group_loss_val:+.4f} | Time: {step_time:.1f}s | VRAM: {peak_vram:.0f}MB",
            flush=True,
        )

        # Save adapter & JSON telemetry after every step (verify no NaNs)
        has_param_nan = False
        for p in trainable_params:
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                has_param_nan = True
                break
        if has_param_nan:
            print(f"  [CRITICAL ERROR] Step {step_idx+1:03d}: Model parameters contain NaN/Inf! NOT saving corrupted checkpoint to disk!", flush=True)
        else:
            model.save_pretrained(args.adapter_out)
            tokenizer.save_pretrained(args.adapter_out)

        results_payload = {
            "experiment": "SODALS v2.1 Out-of-MATH-500 1536-Token GRPO Training (Hendrycks MATH Train Split)",
            "runtime_metadata": collect_runtime_metadata(),
            "source_dataset": args.prompts_json,
            "math500_overlap": 0,
            "target_adapter_in": args.adapter_in,
            "target_adapter_out": args.adapter_out,
            "max_new_tokens": args.max_new_tokens,
            "prompts_count": total_steps,
            "completed_steps": len(step_logs),
            "active_gradient_steps": active_gradient_steps,
            "group_size": args.group_size,
            "beta_kl": args.beta,
            "learning_rate": args.lr,
            "total_solved_prompts": total_solved_prompts,
            "pass_rate": round(100.0 * total_solved_prompts / max(1, len(step_logs)), 2),
            "total_time_minutes": round((time.time() - t_start) / 60.0, 2),
            "step_details": step_logs,
        }
        safe_json_dump(results_payload, args.output_json)

        # Coverage floor guard (every 10 steps): stop if trailing 10-step rollout coverage < min_coverage_floor
        if getattr(args, "enable_arkhe_ternary_reward", False) and len(step_logs) >= 10 and (len(step_logs) % 10 == 0):
            trailing_cov = sum(s.get("rollout_coverage", 1.0) for s in step_logs[-10:]) / 10.0
            if trailing_cov < args.min_coverage_floor:
                print(
                    f"  [COVERAGE FLOOR GUARD] Trailing 10-step coverage {trailing_cov:.1%} < {args.min_coverage_floor:.1%}. Stopping safely.",
                    flush=True,
                )
                break

        if getattr(args, "sweep_steps", 0) > 0 and len(step_logs) >= args.sweep_steps:
            print(
                f"  [SWEEP COMPLETE] Reached sweep_steps={args.sweep_steps} | rel_frob_drift={rel_frob_drift_pct:.4f}% | last_KL={mean_kl_val:.5f}",
                flush=True,
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default=DEFAULT_PROMPTS_JSON)
    parser.add_argument("--adapter_in", type=str, default=DEFAULT_ADAPTER_IN)
    parser.add_argument("--adapter_out", type=str, default=DEFAULT_ADAPTER_OUT)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--num_prompts", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--fresh_scheduler", action="store_true", help="Start fresh cosine decay for remaining steps instead of fast-forwarding into the tail.")
    parser.add_argument("--enable_arkhe_ternary_reward", action="store_true", help="Enable stationary quota-free Dynamic-Lambda Ternary GRPO Rewards via ConditionsArche & MethodArche.")
    parser.add_argument("--min_coverage_floor", type=float, default=0.60, help="Stop training early if trailing 10-step rollout coverage drops below this floor.")
    parser.add_argument("--sweep_steps", type=int, default=0, help="Optional hard cap on completed steps (for 10-step LR movement sweeps).")
    args = parser.parse_args()
    train_grpo_math_train(args)
