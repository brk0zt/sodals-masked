"""
SODALS v2.1 — Faz 3: End-to-End Differentiable GPR Soft-MoA Training
===================================================================
End-to-end multi-adapter training loop with:
  1. Differentiable GPR Gate (probes Layer 32 hidden states, modulates alpha)
  2. Concurrent Soft LoRA Blending: Delta W = (1-alpha)*s1*W1 + alpha*s2*W2
  3. Adaptive PonderingODE: Phase-adaptive RK4 continuous latent manifold evolution
  4. Joint Multi-Objective Loss:
       L = L_CE (Language Modeling)
         + lambda_gate * L_BCE (Domain Routing Supervision)
         + lambda_ortho * L_GramOrtho (Low-Rank Subspace Orthogonality ||AA^T - I||_F^2)

Dataset: sft/master_interleaved_d1_d2_304_curriculum.jsonl (304 balanced samples)
Hardware: RTX 4060 Laptop GPU (8GB VRAM) via 4-bit NF4 + Gradient Checkpointing
Saves to: sft/checkpoints_gpr_soft_moa/ and sft/qwen_lora_gpr_soft_moa/
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from peft.tuners.lora.layer import LoraLayer
from gpr_differentiable_gate import DifferentiableGPRGate
from adaptive_pondering_ode import AdaptivePonderingODEModule
from soft_lora_blender import SoftLoraBlender

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "master_interleaved_d1_d2_304_curriculum.jsonl")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ANCHORS_PATH = os.path.join(SFT_DIR, "gpr_domain_anchors.pt")
OUTPUT_DIR = os.path.join(SFT_DIR, "qwen_lora_gpr_soft_moa")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_gpr_soft_moa")


class InterleavedChatDataset(Dataset):
    """
    Loads 304 balanced samples (144 D1 General STEM + 160 D2 Abstract Algebra).
    Extracts domain target: 0.0 for D1, 1.0 for D2.
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 384):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    meta = item.get("metadata", {})
                    domain_str = meta.get("domain", "")
                    target_domain = 1.0 if "Abstract_Algebra" in domain_str or "Domain_2" in domain_str else 0.0
                    self.samples.append({
                        "messages": item["messages"],
                        "target_domain": target_domain,
                        "focal_weight": item.get("focal_weight", 1.0)
                    })

        print(f"[DATASET] Loaded {len(self.samples)} balanced curriculum samples.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item["messages"]

        prompt_text = ""
        full_text = ""
        for m in messages:
            if m["role"] == "user":
                full_text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
                if not prompt_text:
                    prompt_text = f"<|im_start|>user\n{m['content']}<|im_end|>\n<|im_start|>assistant\n"
            elif m["role"] == "assistant":
                full_text += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_domain": torch.tensor(item["target_domain"], dtype=torch.float32),
            "focal_weight": torch.tensor(item["focal_weight"], dtype=torch.float32),
            "prompt_text": prompt_text
        }


def compute_gram_orthogonality_penalty(model, adapter_name: str) -> torch.Tensor:
    """
    Computes Gram matrix subspace orthogonality penalty ||A A^T - I||_F^2 on LoRA A matrices.
    Encourages low-rank projection directions to remain mutually orthogonal.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name and adapter_name in name and param.requires_grad:
            A = param  # [r, d]
            gram = torch.matmul(A, A.t())
            r = gram.shape[0]
            I = torch.eye(r, device=gram.device, dtype=gram.dtype)
            penalty = penalty + torch.sum((gram - I) ** 2)
            count += 1
    return penalty / max(1, count)



def train():
    print("=" * 80, flush=True)
    print("SODALS v2.1 — FAZ 3: END-TO-END GPR SOFT-MOA TRAINING", flush=True)
    print("Differentiable GPR Gate + Soft LoRA Blending + Adaptive PonderingODE", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tokenizer
    print("Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Base Model (4-bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading Qwen 3.5 9B in 4-bit NF4 directly onto GPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # 3. Dual LoRA Adapters
    print("Configuring Dual LoRA Adapters (General Math r=4 + Abstract Algebra r=8)...", flush=True)
    lora_config_d1 = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    lora_config_d2 = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config_d1, adapter_name="expert_general_math")
    model.add_adapter("expert_abstract_algebra", lora_config_d2)
    model.enable_input_require_grads()

    # Freeze lm_head.weight to prevent 2GB autograd gradient allocation
    base_m = model.base_model.model if hasattr(model, "base_model") else model
    lm_head = getattr(base_m, "lm_head", None) or model.get_output_embeddings()
    if lm_head.weight.requires_grad:
        lm_head.weight.requires_grad = False
        print("[INFO] Frozen lm_head.weight (requires_grad=False) to prevent 2GB gradient tensor.", flush=True)

    # Enable soft blending on both adapters
    blender = SoftLoraBlender(adapter_names=["expert_general_math", "expert_abstract_algebra"])
    num_blended_layers = blender.enable_multi_adapter(model)
    print(f"Concurrent multi-adapter forward enabled on {num_blended_layers} LoraLayers.", flush=True)

    # 4. Differentiable GPR Gate (FP32 master weights for AdamW numerical stability)
    print("Initializing Differentiable GPR Gate from calibrated anchors...", flush=True)
    anchors_data = torch.load(ANCHORS_PATH, map_location="cpu", weights_only=False)
    gate = DifferentiableGPRGate(
        hidden_dim=4096,
        init_anchor_d1=anchors_data["anchor_d1"].float(),
        init_anchor_d2=anchors_data["anchor_d2"].float(),
        init_tau=anchors_data["threshold"],
        temperature=0.03
    ).to("cuda")

    # 5. Adaptive PonderingODE Module (FP32 master weights, 2-3 RK4 steps)
    print("Initializing Adaptive PonderingODE Module...", flush=True)
    pondering_ode = AdaptivePonderingODEModule(
        hidden_dim=4096,
        dt=0.1,
        min_steps=2,
        max_steps=3,
        base_lambda=0.3
    ).to("cuda")

    # 6. Dataset & DataLoader (max_length=384 for safe 8GB VRAM envelope)
    dataset = InterleavedChatDataset(DATASET_PATH, tokenizer, max_length=384)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 7. Optimizer & Scheduler
    trainable_params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": 2e-4},
        {"params": gate.parameters(), "lr": 1e-3},
        {"params": pondering_ode.parameters(), "lr": 1e-4}
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=0.01)

    epochs = 1
    grad_accum_steps = 4
    total_steps = (len(dataloader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    lambda_gate = 0.5
    lambda_ortho = 0.005

    print(f"\nStarting End-to-End Training ({total_steps} optimizer steps, {len(dataloader)} samples)...", flush=True)
    model.train()
    gate.train()
    pondering_ode.train()

    optimizer.zero_grad()
    running_loss = 0.0
    step_count = 0

    for epoch in range(epochs):
        for idx, batch in enumerate(dataloader):
            # Prompt probe extraction
            prompt_str = batch["prompt_text"][0]
            prompt_enc = tokenizer(prompt_str, return_tensors="pt").to("cuda")

            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            target_domain = batch["target_domain"].to("cuda")
            focal_w = batch["focal_weight"].to("cuda")

            # A. Extract probe hidden states at Layer 32 (prompt-only, pristine base model)
            with torch.no_grad():
                with model.disable_adapter():
                    probe_out = model(**prompt_enc, output_hidden_states=True)
                    h_probe = probe_out.hidden_states[32].detach()
                    del probe_out
                    torch.cuda.empty_cache()

            # B. Differentiable GPR Gating
            alpha, z_i, scaled_logits = gate(h_probe, attention_mask=prompt_enc["attention_mask"])

            # C. Modulate LoRA Scales via SoftLoraBlender (Differentiable Autograd Tensor)
            # Passing alpha tensor directly preserves the autograd graph from CE loss back into gate parameters!
            blender.set_blend(model, alpha)
            alpha_val = float(alpha.detach().cpu().item())

            # D. Forward Pass on pure transformer backbone (bypassing full sequence lm_head projection)
            outputs = model.base_model.model.model(input_ids, attention_mask=attention_mask)
            h_final = outputs.last_hidden_state
            del outputs
            torch.cuda.empty_cache()

            # E. Adaptive PonderingODE RK4 Evolution
            h_pondered, num_steps, _ = pondering_ode(h_final, alpha=alpha)

            # F. Active Token Chunked Logits & CE Loss (prevent VRAM OOM on vocab 248,320)
            seq_len = h_pondered.size(1) - 1
            chunk_size = 32
            total_ce_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            total_tokens = 0
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

            for start_idx in range(0, seq_len, chunk_size):
                end_idx = min(start_idx + chunk_size, seq_len)
                c_labels = labels[:, start_idx+1:end_idx+1].contiguous()

                mask = c_labels != -100
                active = mask.sum().item()
                if active > 0:
                    sub_hidden = h_pondered[:, start_idx:end_idx, :][mask].view(-1, 4096)
                    sub_labels = c_labels[mask].view(-1)
                    sub_logits = lm_head(sub_hidden)
                    sub_loss = loss_fct(sub_logits, sub_labels)
                    total_ce_loss = total_ce_loss + sub_loss
                    total_tokens += active
                    del sub_logits, sub_hidden, sub_loss

            ce_loss = total_ce_loss / max(1, total_tokens)
            weighted_ce_loss = focal_w * ce_loss

            # G. Auxiliary Gate Loss (using scaled_logits and BCE with logits)
            gate_loss = gate.compute_gate_loss(scaled_logits, target_domain, from_logits=True)

            # H. Gram Matrix Subspace Orthogonality Loss (||A A^T - I||_F^2)
            ortho_d1 = compute_gram_orthogonality_penalty(model, "expert_general_math")
            ortho_d2 = compute_gram_orthogonality_penalty(model, "expert_abstract_algebra")
            ortho_loss = ortho_d1 + ortho_d2

            # Total Loss
            total_loss = (weighted_ce_loss + lambda_gate * gate_loss + lambda_ortho * ortho_loss) / grad_accum_steps
            total_loss.backward()

            running_loss += total_loss.item() * grad_accum_steps

            if (idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(pondering_ode.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 2 == 0 or step_count == total_steps:
                    avg_loss = running_loss / (step_count * grad_accum_steps)
                    print(
                        f"[Step {step_count:02d}/{total_steps}] Loss: {avg_loss:.4f} | "
                        f"CE: {ce_loss.item():.4f} | GateLoss: {gate_loss.item():.4f} | "
                        f"Alpha: {alpha_val:.4f} (Target: {target_domain.item():.0f}) | "
                        f"PonderSteps: {num_steps} | GramOrtho: {ortho_loss.item():.4f}",
                        flush=True
                    )

                # Periodic checkpoint at midpoint
                if step_count == 38:
                    mid_dir = os.path.join(CHECKPOINT_DIR, f"step_{step_count}")
                    os.makedirs(mid_dir, exist_ok=True)
                    model.save_pretrained(mid_dir)
                    torch.save({
                        "gate_state_dict": gate.state_dict(),
                        "pondering_state_dict": pondering_ode.state_dict(),
                        "tau": gate.tau.item(),
                        "temperature": gate.temperature
                    }, os.path.join(mid_dir, "gpr_soft_moa_aux_modules.pt"))
                    print(f"[CHECKPOINT] Mid-training checkpoint saved to {mid_dir}", flush=True)

    print("\nTraining Completed! Saving model, gate, and pondering modules...", flush=True)
    # Save PEFT adapters
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save Gate and Pondering weights
    torch.save({
        "gate_state_dict": gate.state_dict(),
        "pondering_state_dict": pondering_ode.state_dict(),
        "tau": gate.tau.item(),
        "temperature": gate.temperature
    }, os.path.join(OUTPUT_DIR, "gpr_soft_moa_aux_modules.pt"))

    print(f"[SUCCESS] Trained GPR Soft-MoA saved to: {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    train()
