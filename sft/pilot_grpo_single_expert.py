"""
SODALS v2.1 — GRPO Mini-Pilot: Isolated Single-Expert RL Optimization
=====================================================================
Target: expert_general_math (sft/qwen_lora_adapter_balanced_r4, r=4)
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM) via 4-bit NF4 + Gradient Checkpointing
Method: Group Relative Policy Optimization (GRPO, DeepSeek-R1 style)
  - No Critic / Value model (Zero additional VRAM)
  - Group size G = 4 sequential rollouts under torch.no_grad()
  - Zero-VRAM reference model via model.disable_adapter()
  - Exact outcome reward via SymPy symbolic matching (+1.0 / -1.0)
  - Format reward via \\boxed{} presence (+0.2 / -0.2)
  - Per-token KL penalty (beta = 0.04)

Dataset: 25 challenging, verifiable MATH-500 prompts
Saves to: sft/qwen_lora_adapter_grpo_pilot/ and sft/pilot_grpo_results.json
"""

import os
import sys
import json
import re
import time
import math
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

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_pilot")
OUTPUT_JSON = os.path.join(SFT_DIR, "pilot_grpo_results.json")


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
    
    # 1. Outcome Reward: +1.0 for pass, -1.0 for fail
    r_outcome = 1.0 if is_correct else -1.0
    
    # 2. Format Reward: +0.2 for containing a cleanly extracted answer, -0.2 if blank
    r_format = 0.2 if (pred != "" and r"\boxed{" in gen_text) else -0.2
    
    total_reward = r_outcome + r_format
    return total_reward, pred, is_correct


def run_grpo_pilot():
    print("=" * 80, flush=True)
    print("SODALS v2.1 — GRPO MINI-PILOT: ISOLATED SINGLE-EXPERT REINFORCEMENT LEARNING", flush=True)
    print("Target: expert_general_math (balanced_r4, r=4) on 25 MATH-500 Questions", flush=True)
    print("Zero-VRAM Reference Model + SymPy Deterministic Outcome Reward", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)

    # 1. Load Tokenizer
    print("Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in 4-bit NF4 with FP32 compute dtype for SSM stability
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading Qwen 3.5 9B in 4-bit NF4 onto GPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # Cast all non-4bit SSM and norm parameters to FP32 (cost: 4.2 MB)
    for name, param in base_model.named_parameters():
        if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
            param.data = param.data.to(torch.float32)

    # Leave embed_tokens in FP16, but forward hook converts output hidden states to FP32
    def upcast_embeds(module, input, output):
        return output.to(torch.float32)
    base_model.model.embed_tokens.register_forward_hook(upcast_embeds)

    # 3. Load Target Expert Adapter (is_trainable=True)
    print(f"Loading expert_general_math from: {ADAPTER_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)

    # Ensure LoRA parameters have requires_grad=True
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    model.enable_input_require_grads()

    # Freeze lm_head.weight and attach downcast pre-hook
    base_m = model.base_model.model if hasattr(model, "base_model") else model
    lm_head = getattr(base_m, "lm_head", None) or model.get_output_embeddings()
    if lm_head.weight.requires_grad:
        lm_head.weight.requires_grad = False
        print("[INFO] Frozen lm_head.weight to prevent 2GB gradient memory overhead.", flush=True)

    def downcast_lm_head_input(module, args):
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].dtype != module.weight.dtype:
            return (args[0].to(module.weight.dtype),) + args[1:]
        return args
    lm_head.register_forward_pre_hook(downcast_lm_head_input)

    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"[VERIFIED] Trainable LoRA parameters: {len(trainable_params)} tensors. Base VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB", flush=True)

    # 4. Load 25 Benchmark Prompts
    print("Loading 25 benchmark prompts from MATH-500...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts_data = []
    for i in range(25):
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
    print(f"Loaded {len(prompts_data)} prompts.", flush=True)

    # 5. Hyperparameters
    G = 4               # Group size (4 rollouts per prompt)
    beta = 0.04         # KL penalty coefficient
    lr = 1e-5           # Conservative RL learning rate
    max_new_tokens = 384
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(prompts_data)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=total_steps)

    step_logs = []
    total_solved_prompts = 0

    print("\n" + "=" * 80, flush=True)
    print(f"STARTING GRPO TRAINING LOOP ({total_steps} Prompts, G={G} Rollouts/Prompt)", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()

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

        for g in range(G):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **p_enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            gen_ids = out[0][p_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            reward, pred, is_correct = compute_reward(gen_text, gt)
            
            rollout_responses.append(gen_text)
            rollout_rewards.append(reward)
            rollout_preds.append(pred)
            rollout_is_correct.append(is_correct)
            rollout_gen_ids.append(gen_ids)

        rewards_tensor = torch.tensor(rollout_rewards, dtype=torch.float32, device="cuda")
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item() + 1e-4
        advantages = ((rewards_tensor - mean_r) / std_r).tolist()

        num_correct = sum(rollout_is_correct)
        if num_correct > 0:
            total_solved_prompts += 1

        # -------------------------------------------------------------
        # Phase 2: Chunked Policy Gradient & Zero-VRAM KL Backward
        # -------------------------------------------------------------
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad()
        group_loss_val = 0.0

        for g in range(G):
            gen_ids = rollout_gen_ids[g]
            if len(gen_ids) == 0:
                continue

            adv = advantages[g]
            seq_ids = torch.cat([p_enc.input_ids, gen_ids.unsqueeze(0)], dim=1)
            target_ids = gen_ids.unsqueeze(0)  # [1, R]
            r_len = gen_ids.size(0)

            # A. Zero-VRAM Reference Policy Log-Probs (adapter disabled, no_grad, chunked)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_out = model.base_model.model.model(seq_ids)
                    ref_h = ref_out.last_hidden_state[:, p_len - 1 : -1, :]
                    ref_logps_list = []
                    for start in range(0, r_len, 32):
                        end = min(start + 32, r_len)
                        c_h = ref_h[:, start:end, :]
                        c_t = target_ids[:, start:end]
                        c_logits = lm_head(c_h.to(lm_head.weight.dtype))
                        c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
                        c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
                        ref_logps_list.append(c_logps)
                    ref_logps = torch.cat(ref_logps_list, dim=-1)  # [1, R]
                    del ref_out, ref_h, ref_logps_list
                    torch.cuda.empty_cache()

            # B. Active Policy Forward (requires_grad = True, gradient checkpointed)
            backbone_out = model.base_model.model.model(seq_ids)
            h_resp = backbone_out.last_hidden_state[:, p_len - 1 : -1, :]  # [1, R, 4096]

            # C. Chunked GRPO Policy Loss Accumulation
            loss_g = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            for start in range(0, r_len, 32):
                end = min(start + 32, r_len)
                c_h = h_resp[:, start:end, :].contiguous()
                c_t = target_ids[:, start:end].contiguous()
                c_ref_logps = ref_logps[:, start:end]

                c_logits = lm_head(c_h.to(lm_head.weight.dtype))
                c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
                c_cur_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)

                # Per-token KL Divergence (Schulman approximation)
                c_ratio = torch.clamp(c_ref_logps - c_cur_logps, -10.0, 10.0)
                c_kl = torch.exp(c_ratio) - c_ratio - 1.0

                # GRPO Policy Loss chunk accumulation
                c_loss = - (c_cur_logps * adv - beta * c_kl).sum() / (r_len * G)
                loss_g = loss_g + c_loss

            loss_g.backward()
            group_loss_val += loss_g.item() * G

            del backbone_out, h_resp, ref_logps, loss_g
            torch.cuda.empty_cache()

        # Step optimizer with NaN/Inf safeguard
        has_nan_grad = any(torch.isnan(p.grad).any().item() for p in trainable_params if p.grad is not None)
        has_inf_grad = any(torch.isinf(p.grad).any().item() for p in trainable_params if p.grad is not None)
        if not (has_nan_grad or has_inf_grad):
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
            optimizer.step()
        else:
            print(f"  [WARNING] Step {step+1} skipped optimizer step due to NaN/Inf grad! (NaN: {has_nan_grad}, Inf: {has_inf_grad})", flush=True)
            grad_norm = 0.0

        scheduler.step()
        optimizer.zero_grad()

        # Weight corruption check
        has_nan_weight = any(torch.isnan(p).any().item() for p in trainable_params)
        if has_nan_weight:
            print(f"  [CRITICAL ERROR] Step {step+1} corrupted weights with NaN!", flush=True)

        # Step timing and VRAM
        step_elapsed = time.time() - t_step_start
        cur_vram = torch.cuda.max_memory_reserved(0) / (1024**2)

        # Logging
        pass_at_least_one = "✓" if num_correct > 0 else "✗"
        print(
            f"[{step+1:02d}/{total_steps}] [{subj:18s}] "
            f"Solve: {pass_at_least_one} ({num_correct}/{G}) | "
            f"MeanR: {mean_r:+.2f} | Loss: {group_loss_val:.4f} | "
            f"Süre: {step_elapsed:.1f}s | VRAM: {cur_vram:.0f}MB | "
            f"GT: {gt[:12]} | Best: {rollout_preds[0][:12]}",
            flush=True
        )

        step_logs.append({
            "step": step + 1,
            "subject": subj,
            "problem": prob,
            "ground_truth": gt,
            "mean_reward": round(mean_r, 3),
            "solved_rollouts": num_correct,
            "rollout_rewards": [round(r, 2) for r in rollout_rewards],
            "rollout_preds": rollout_preds,
            "advantages": [round(a, 3) for a in advantages],
            "loss": round(group_loss_val, 4),
            "step_seconds": round(step_elapsed, 1),
            "vram_mb": round(cur_vram, 1)
        })

        # Fault-tolerant incremental save after EVERY step
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "SODALS v2.1 GRPO Mini-Pilot",
                "target_adapter": "expert_general_math (balanced_r4)",
                "prompts_count": total_steps,
                "completed_steps": step + 1,
                "group_size": G,
                "beta_kl": beta,
                "learning_rate": lr,
                "total_solved_prompts": total_solved_prompts,
                "pass_rate": round(100.0 * total_solved_prompts / (step + 1), 2),
                "elapsed_minutes": round((time.time() - t_start) / 60, 2),
                "step_details": step_logs
            }, f, indent=2, ensure_ascii=False)

        # Checkpoint adapter every 5 steps
        if (step + 1) % 5 == 0 or (step + 1) == total_steps:
            model.save_pretrained(OUTPUT_ADAPTER_DIR)
            tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    total_time = time.time() - t_start

    print("\n" + "=" * 80, flush=True)
    print("GRPO MINI-PILOT COMPLETED!", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Prompts Processed: {total_steps}", flush=True)
    print(f"Prompts with >=1 Correct Rollout: {total_solved_prompts}/{total_steps} ({100*total_solved_prompts/total_steps:.1f}%)", flush=True)
    print(f"Total Execution Time: {total_time/60:.1f} minutes", flush=True)

    # Save trained adapter
    print(f"\nSaving RL-adapted expert_general_math to: {OUTPUT_ADAPTER_DIR}...", flush=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    # Save results JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "SODALS v2.1 GRPO Mini-Pilot",
            "target_adapter": "expert_general_math (balanced_r4)",
            "prompts_count": total_steps,
            "group_size": G,
            "beta_kl": beta,
            "learning_rate": lr,
            "total_solved_prompts": total_solved_prompts,
            "pass_rate": round(100.0 * total_solved_prompts / total_steps, 2),
            "execution_minutes": round(total_time / 60, 2),
            "step_details": step_logs
        }, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] GRPO pilot results written to: {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    run_grpo_pilot()
