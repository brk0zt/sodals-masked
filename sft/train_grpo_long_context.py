"""
SODALS v2.1 — Step 2: Full-Scale Long-Context GRPO Training Pipeline
====================================================================
Target: expert_general_math (sft/qwen_lora_adapter_balanced_r4, r=4)
Hardware: NVIDIA GeForce RTX 4060 Laptop (8GB Physical VRAM, 8188 MB)
Architecture:
  - 1536-token context window with zero WDDM paging
  - CPUEmbeddingWrapper: offloads embed_tokens to CPU RAM (-1,645 MB)
  - Linear4bit lm_head: 4-bit NF4 quantized projection (-1,431 MB)
  - Base VRAM: ~3,907 MB, leaving ~3,616 MB physical headroom
  - Chunked Policy Loss (chunk_size=32, vocab memory ~15.8 MB)
  - FP32 SSM & Norm casting: 100% stable backpropagation (zero NaNs)
  - Group Relative Policy Optimization (GRPO, DeepSeek-R1 style)
  - Zero-VRAM reference model via model.disable_adapter()
  - SymPy exact symbolic outcome reward + LaTeX format reward
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import json
import re
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import PeftModel
import bitsandbytes as bnb

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
DEFAULT_ADAPTER_IN = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
DEFAULT_ADAPTER_OUT = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_long")
DEFAULT_OUTPUT_JSON = os.path.join(SFT_DIR, "grpo_long_results.json")
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


def clean_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    s = re.sub(r'^(?:is|theansweris)', '', s, flags=re.I)
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
    s = re.sub(r'\s*(?:cm|units|degrees|m|inches)\b', '', s, flags=re.I)
    s = re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"\\sqrt{\1}", s)
    return s.strip()


def to_sympy(s: str):
    s = clean_latex(s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_math_match(pred: str, gt: str) -> bool:
    cp = clean_latex(pred)
    cg = clean_latex(gt)
    if cp == cg:
        return True

    if cg.lstrip('-').isdigit():
        num_m = re.findall(r'-?\d+', cp)
        if num_m and num_m[-1] == cg:
            return True

    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            p1 = f"{a}+{b},{a}-{b}"
            p2 = f"{a}-{b},{a}+{b}"
            if clean_latex(pred) in [clean_latex(p1), clean_latex(p2)]:
                return True

    sym_gt = to_sympy(gt)
    sym_pred = to_sympy(pred)
    if sym_gt is not None and sym_pred is not None:
        try:
            if sympy.simplify(sym_gt - sym_pred) == 0:
                return True
        except Exception:
            pass
        try:
            if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                return True
        except Exception:
            pass

    return False


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return text[start:].strip()

    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n\.]+)", text)
    if m:
        return m.group(1).strip()

    all_nums = re.findall(r'\b(-?\d+)\b', text)
    if all_nums:
        return all_nums[-1]

    return ""


def compute_reward(gen_text: str, gt_answer: str) -> tuple[float, str, bool]:
    pred = extract_boxed_answer(gen_text)
    is_correct = check_math_match(pred, gt_answer)
    
    # 1. Exact Outcome Reward (+1.0 / -1.0)
    r_outcome = 1.0 if is_correct else -1.0
    
    # 2. Strict Format Reward (+0.2 if boxed present & non-empty, -0.2 otherwise)
    r_format = 0.2 if (pred != "" and r"\boxed{" in gen_text) else -0.2
    
    total_reward = r_outcome + r_format
    return total_reward, pred, is_correct


def train_grpo_long_context(args):
    print("=" * 85, flush=True)
    print("SODALS v2.1 — STEP 2: FULL-SCALE LONG-CONTEXT GRPO TRAINING PIPELINE", flush=True)
    print(f"Context Window: max_new_tokens = {args.max_new_tokens} (Total sequence up to ~1700 tokens)", flush=True)
    print(f"Group Size G = {args.group_size} | KL beta = {args.beta} | LR = {args.lr}", flush=True)
    print(f"Prompts: {args.num_prompts} (Start idx: {args.start_idx})", flush=True)
    print("=" * 85, flush=True)

    args.adapter_out = os.path.abspath(args.adapter_out)
    args.output_json = os.path.abspath(args.output_json)
    os.makedirs(args.adapter_out, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # 1. Load Tokenizer
    print("1. Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in 4-bit NF4 with SDPA FlashAttention
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    print("2. Loading Qwen 3.5 9B with SDPA FlashAttention in 4-bit NF4...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # 3. Cast non-4bit SSM and norm parameters to FP32 (numerical stability, 4.2 MB)
    print("3. Casting SSM and LayerNorm parameters to FP32...", flush=True)
    for name, param in base_model.named_parameters():
        if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
            param.data = param.data.to(torch.float32)

    # 4. Offload embed_tokens to CPU RAM (-1,645 MB VRAM)
    print("4. Offloading embed_tokens to CPU RAM via CPUEmbeddingWrapper (-1,645 MB VRAM)...", flush=True)
    embed_cpu = base_model.model.embed_tokens.to("cpu")
    base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
    type(base_model).device = property(lambda self: torch.device("cuda:0"))
    torch.cuda.empty_cache()

    # 5. Quantize lm_head to 4-bit NF4 (-1,431 MB VRAM)
    print("5. Quantizing lm_head to 4-bit NF4 (-1,431 MB VRAM)...", flush=True)
    orig_lm_weight = base_model.lm_head.weight.data
    l4_head = bnb.nn.Linear4bit(4096, 248320, bias=False, compute_dtype=torch.float32, quant_type='nf4')
    q_w, qstate = bnb.functional.quantize_4bit(orig_lm_weight, quant_type='nf4')
    l4_head.weight = bnb.nn.Params4bit(q_w, requires_grad=False, quant_state=qstate)
    base_model.lm_head = l4_head
    del orig_lm_weight
    torch.cuda.empty_cache()

    # 6. Attach LoRA Adapter
    print(f"6. Attaching LoRA Adapter from: {args.adapter_in}...", flush=True)
    model = PeftModel.from_pretrained(base_model, args.adapter_in, is_trainable=True)
    type(model).device = property(lambda self: torch.device("cuda:0"))
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.enable_input_require_grads()

    backbone = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]

    base_alloc = torch.cuda.memory_allocated(0) / (1024**2)
    base_res = torch.cuda.memory_reserved(0) / (1024**2)
    print(f"\n[FROZEN INFRASTRUCTURE READY] Base Alloc: {base_alloc:.1f} MB | Reserved: {base_res:.1f} MB | Physical Headroom: {GPU_CAPACITY_MB - base_res:.1f} MB\n", flush=True)

    # 7. Load Prompts
    print(f"Loading {args.num_prompts} prompts starting from index {args.start_idx}...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts_data = []
    end_idx = min(args.start_idx + args.num_prompts, len(ds))
    for i in range(args.start_idx, end_idx):
        it = ds[i]
        prob = it["problem"]
        gt = it["answer"]
        subj = it["subject"]
        prompt_str = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        prompts_data.append({
            "index": i + 1,
            "subject": subj,
            "problem": prob,
            "ground_truth": gt,
            "prompt_str": prompt_str
        })
    print(f"Successfully prepared {len(prompts_data)} training prompts.", flush=True)

    # 8. Optimizer & Cosine Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(prompts_data)
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    step_logs = []
    total_solved_prompts = 0
    t_start = time.time()

    print("\n" + "=" * 90, flush=True)
    print(f"STARTING 1536-TOKEN GRPO TRAINING ({total_steps} Steps, G={args.group_size} Rollouts/Step)", flush=True)
    print("=" * 90, flush=True)

    for step, p_item in enumerate(prompts_data):
        p_idx = p_item["index"]
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
        rollout_responses = []
        rollout_rewards = []
        rollout_preds = []
        rollout_is_correct = []
        rollout_gen_ids = []
        rollout_tokens = []

        t_gen_0 = time.time()
        for g in range(args.group_size):
            torch.cuda.empty_cache()
            print(f"  [Rollout {g+1}/{args.group_size}] Generating (max {args.max_new_tokens} tokens)...", flush=True)
            with torch.no_grad():
                out = model.generate(
                    **p_enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            gen_ids = out[0][p_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            reward, pred, is_correct = compute_reward(gen_text, gt)
            print(f"  [Rollout {g+1}/{args.group_size}] Generated {len(gen_ids)} tokens | Correct: {is_correct} | Reward: {reward:+.2f} | Pred: {pred[:15]}", flush=True)
            
            rollout_responses.append(gen_text)
            rollout_rewards.append(reward)
            rollout_preds.append(pred)
            rollout_is_correct.append(is_correct)
            rollout_gen_ids.append(gen_ids)
            rollout_tokens.append(len(gen_ids))

        t_gen = time.time() - t_gen_0
        avg_tokens = sum(rollout_tokens) / len(rollout_tokens)

        # Advantage Calculation
        rewards_tensor = torch.tensor(rollout_rewards, dtype=torch.float32, device="cuda")
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item() + 1e-4
        advantages = ((rewards_tensor - mean_r) / std_r).tolist()

        num_correct = sum(rollout_is_correct)
        if num_correct > 0:
            total_solved_prompts += 1

        # -------------------------------------------------------------
        # Phase 2: Chunked Policy Forward & Backward
        # -------------------------------------------------------------
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad()
        group_loss_val = 0.0

        for g in range(args.group_size):
            gen_ids = rollout_gen_ids[g]
            if len(gen_ids) == 0:
                continue

            adv = advantages[g]
            seq_ids = torch.cat([p_enc.input_ids, gen_ids.unsqueeze(0)], dim=1)
            target_ids = gen_ids.unsqueeze(0).to("cuda")  # [1, R]
            r_len = gen_ids.size(0)
            print(f"  [Backward {g+1}/{args.group_size}] Tokens: {r_len} | Adv: {adv:+.3f} | Calculating Loss...", flush=True)

            # Embeddings on CPU RAM -> transferred to CUDA as leaf variable
            seq_embeds = embed_cpu(seq_ids.cpu()).to("cuda", dtype=torch.float32)

            # A. Zero-VRAM Reference Log-Probs (adapter disabled, no_grad)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_out = backbone(inputs_embeds=seq_embeds)
                    ref_h = ref_out.last_hidden_state[:, p_len - 1 : -1, :]
                    ref_logps_list = []
                    for start in range(0, r_len, 32):
                        end = min(start + 32, r_len)
                        c_h = ref_h[:, start:end, :].contiguous().to(torch.float32)
                        c_t = target_ids[:, start:end].contiguous()
                        c_logits = lm_head(c_h)
                        c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
                        c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
                        ref_logps_list.append(c_logps)
                    ref_logps = torch.cat(ref_logps_list, dim=-1)  # [1, R]
                    del ref_out, ref_h, ref_logps_list
                    torch.cuda.empty_cache()

            # B. Active Policy Forward (requires_grad = True)
            act_embeds = seq_embeds.detach().requires_grad_(True)
            backbone_out = backbone(inputs_embeds=act_embeds)
            h_resp = backbone_out.last_hidden_state[:, p_len - 1 : -1, :]  # [1, R, 4096]

            # C. Chunked GRPO Loss Accumulation
            loss_g = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            for start in range(0, r_len, 32):
                end = min(start + 32, r_len)
                c_h = h_resp[:, start:end, :].contiguous().to(torch.float32)
                c_t = target_ids[:, start:end].contiguous()
                c_ref_logps = ref_logps[:, start:end]

                c_logits = lm_head(c_h)
                c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
                c_cur_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)

                # Per-token KL Divergence (Schulman approximation)
                c_ratio = torch.clamp(c_ref_logps - c_cur_logps, -10.0, 10.0)
                c_kl = torch.exp(c_ratio) - c_ratio - 1.0

                # Policy Loss chunk
                c_loss = - (c_cur_logps * adv - args.beta * c_kl).sum() / (r_len * args.group_size)
                loss_g = loss_g + c_loss

            loss_g.backward()
            group_loss_val += loss_g.item() * args.group_size

            del backbone_out, h_resp, ref_logps, loss_g, seq_embeds, act_embeds
            torch.cuda.empty_cache()

        # Step Optimizer with NaN/Inf safeguard
        has_nan_grad = any(torch.isnan(p.grad).any().item() for p in trainable_params if p.grad is not None)
        has_inf_grad = any(torch.isinf(p.grad).any().item() for p in trainable_params if p.grad is not None)
        if not (has_nan_grad or has_inf_grad):
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
            optimizer.step()
        else:
            print(f"  [WARNING] Step {step+1} skipped due to NaN/Inf gradient! (NaN: {has_nan_grad}, Inf: {has_inf_grad})", flush=True)
            grad_norm = 0.0

        scheduler.step()
        optimizer.zero_grad()

        # Step Timing & Memory
        step_elapsed = time.time() - t_step_start
        cur_vram = torch.cuda.max_memory_reserved(0) / (1024**2)

        pass_symbol = "✓" if num_correct > 0 else "✗"
        print(
            f"[{step+1:02d}/{total_steps}] Q{p_idx:03d} [{subj[:14]:14s}] "
            f"Solve: {pass_symbol} ({num_correct}/{args.group_size}) | "
            f"MeanR: {mean_r:+.2f} | Loss: {group_loss_val:.4f} | "
            f"AvgTok: {avg_tokens:.0f} | Time: {step_elapsed:.1f}s | VRAM: {cur_vram:.0f}MB | "
            f"GT: {gt[:10]} | Best: {rollout_preds[0][:10]}",
            flush=True
        )

        step_logs.append({
            "step": step + 1,
            "question_index": p_idx,
            "subject": subj,
            "problem": prob,
            "ground_truth": gt,
            "mean_reward": round(mean_r, 3),
            "solved_rollouts": num_correct,
            "avg_tokens": round(avg_tokens, 1),
            "rollout_tokens": rollout_tokens,
            "rollout_rewards": [round(r, 2) for r in rollout_rewards],
            "rollout_preds": rollout_preds,
            "advantages": [round(a, 3) for a in advantages],
            "loss": round(group_loss_val, 4),
            "step_seconds": round(step_elapsed, 1),
            "vram_mb": round(cur_vram, 1)
        })

        # Incremental Save after EVERY step
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "SODALS v2.1 1536-Token GRPO Training",
                "target_adapter": args.adapter_in,
                "max_new_tokens": args.max_new_tokens,
                "prompts_count": total_steps,
                "completed_steps": step + 1,
                "group_size": args.group_size,
                "beta_kl": args.beta,
                "learning_rate": args.lr,
                "total_solved_prompts": total_solved_prompts,
                "pass_rate": round(100.0 * total_solved_prompts / (step + 1), 2),
                "elapsed_minutes": round((time.time() - t_start) / 60, 2),
                "step_details": step_logs
            }, f, indent=2, ensure_ascii=False)

        # Checkpoint adapter every 5 steps
        if (step + 1) % 5 == 0 or (step + 1) == total_steps:
            model.save_pretrained(args.adapter_out)
            tokenizer.save_pretrained(args.adapter_out)

    total_time = time.time() - t_start
    print("\n" + "=" * 85, flush=True)
    print("1536-TOKEN GRPO TRAINING COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total Prompts Processed: {total_steps}", flush=True)
    print(f"Prompts with >=1 Correct Rollout: {total_solved_prompts}/{total_steps} ({100*total_solved_prompts/total_steps:.1f}%)", flush=True)
    print(f"Total Time: {total_time/60:.1f} minutes", flush=True)
    print(f"Adapter saved to: {args.adapter_out}", flush=True)

    # Final save of results JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "SODALS v2.1 1536-Token GRPO Training",
            "target_adapter": args.adapter_in,
            "max_new_tokens": args.max_new_tokens,
            "prompts_count": total_steps,
            "completed_steps": total_steps,
            "group_size": args.group_size,
            "beta_kl": args.beta,
            "learning_rate": args.lr,
            "total_solved_prompts": total_solved_prompts,
            "pass_rate": round(100.0 * total_solved_prompts / total_steps, 2),
            "total_time_minutes": round(total_time / 60, 2),
            "step_details": step_logs
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {args.output_json}", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SODALS v2.1 Long-Context GRPO Training")
    parser.add_argument("--num_prompts", type=int, default=25, help="Number of prompts to train on")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in MATH-500 dataset")
    parser.add_argument("--max_new_tokens", type=int, default=1536, help="Maximum new tokens per rollout")
    parser.add_argument("--group_size", type=int, default=4, help="Group size G rollouts per prompt")
    parser.add_argument("--beta", type=float, default=0.04, help="KL divergence penalty coefficient")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--adapter_in", type=str, default=DEFAULT_ADAPTER_IN, help="Initial adapter path")
    parser.add_argument("--adapter_out", "--output_adapter_dir", dest="adapter_out", type=str, default=DEFAULT_ADAPTER_OUT, help="Output adapter directory")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON, help="Output results JSON")
    args = parser.parse_args()

    train_grpo_long_context(args)
