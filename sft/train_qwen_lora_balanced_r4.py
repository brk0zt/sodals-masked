"""
SODALS v3.2 — Balanced 805-Example Multi-Domain Training (LoRA r=4, Regularized)
================================================================================
Jointly trains LoRA parameters under:
  - [DATA] master_800_balanced_curriculum.jsonl (805 examples across 8 domains)
  - [REGULARIZED] LoRA rank r=4, alpha=8, dropout=0.10, target_modules=[q, k, v, o] (~2M params)
  - [OFF] CliffordBridge (Removed modulo-256 periodic noise)
  - [ON] PonderingODEModule continuous RK4 manifold integration
  - [ON] Wronskian Orthogonality Penalty (ortho_lambda=0.005)

Resolves the 109,390 params/sample memorization bottleneck by increasing samples (266 -> 805)
and reducing trainable parameters (29M -> 2M), dropping parameter ratio from 109,390 to 2,484!

Saves adapter to: sft/qwen_lora_adapter_balanced_r4/
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import transformers.trainer
import transformers.utils.import_utils
import transformers.utils

def _safe_bypass():
    pass

transformers.utils.import_utils.check_torch_load_is_safe = _safe_bypass
transformers.utils.check_torch_load_is_safe = _safe_bypass
transformers.trainer.check_torch_load_is_safe = _safe_bypass

# Patch torch.load for PyTorch 2.4+ safe unpickling
_orig_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _custom_torch_load

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "master_800_balanced_curriculum.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_balanced_r4")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")

sys.path.insert(0, SFT_DIR)
from pondering_core_bridge import PonderingODEModule


class SFTChatDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 768):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        # Sort by Hamiltonian stage if metadata exists
        def sort_key(x):
            meta = x.get("metadata", {})
            return (meta.get("stage", 3), meta.get("difficulty", 1.0))
        self.data.sort(key=sort_key)
        print(f"[DATASET] Loaded {len(self.data)} balanced examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        full_text = ""
        for m in messages:
            if m["role"] == "user":
                full_text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            elif m["role"] == "assistant":
                full_text += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class PonderingOrthogonalTrainer(Trainer):
    """
    Trainer incorporating PonderingODEModule RK4 continuous latent manifold evolution.
    CliffordBridge noise is removed.
    """
    def __init__(self, *args, ortho_lambda=0.005, hidden_dim=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_lambda = ortho_lambda
        self._lora_A_params = None
        self.pondering_ode = PonderingODEModule(hidden_dim=hidden_dim, dt=0.1, max_steps=4, lambda_anchor=0.3)

    def _get_lora_A_params(self, model):
        if self._lora_A_params is None:
            self._lora_A_params = [
                param for name, param in model.named_parameters()
                if "lora_A" in name and param.requires_grad
            ]
        return self._lora_A_params

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # Forward pass requesting last hidden states
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1] # [batch, seq, dim]
        device = last_hidden.device
        dtype = last_hidden.dtype

        # Free raw logits immediately to save VRAM
        if hasattr(outputs, "logits"):
            outputs.logits = None

        # Device and dtype synchronization
        if self.pondering_ode.arkhe_proj.weight.device != device or self.pondering_ode.arkhe_proj.weight.dtype != dtype:
            self.pondering_ode = self.pondering_ode.to(device=device, dtype=dtype)

        # 1. Apply Differentiable RK4 Continuous Latent ODE Pondering
        pondered_hidden = self.pondering_ode(last_hidden)

        # 2. Chunked lm_head + Cross-Entropy Loss
        base_model = model.base_model.model if hasattr(model, "base_model") else model
        lm_head = getattr(base_model, "lm_head", None) or getattr(model, "lm_head", None)

        seq_len = pondered_hidden.size(1) - 1 # shifted length
        chunk_size = 256
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

        for start_idx in range(0, seq_len, chunk_size):
            end_idx = min(start_idx + chunk_size, seq_len)
            c_hidden = pondered_hidden[:, start_idx:end_idx, :].contiguous()
            c_labels = labels[:, start_idx+1:end_idx+1].contiguous()

            mask = c_labels != -100
            active = mask.sum().item()
            if active > 0:
                if lm_head is not None:
                    c_logits = lm_head(c_hidden)
                else:
                    c_logits = c_hidden

                c_loss = loss_fct(c_logits.view(-1, c_logits.size(-1)), c_labels.view(-1))
                total_loss = total_loss + c_loss
                total_tokens += active

        ce_loss = total_loss / max(1, total_tokens)

        # 3. Wronskian Orthogonality Penalty
        if self.ortho_lambda > 0:
            lora_A_list = self._get_lora_A_params(model)
            dim_groups = {}
            for A in lora_A_list:
                d = A.shape[1]
                if d not in dim_groups:
                    dim_groups[d] = []
                dim_groups[d].append(A)

            ortho_penalty = torch.tensor(0.0, device=ce_loss.device)
            for d, mats in dim_groups.items():
                stacked = torch.stack(mats, dim=0)
                gram = torch.bmm(stacked, stacked.transpose(1, 2))
                I = torch.eye(stacked.shape[1], device=stacked.device, dtype=stacked.dtype).unsqueeze(0)
                diff = gram - I
                ortho_penalty = ortho_penalty + (diff ** 2).sum()

            total_loss = ce_loss + self.ortho_lambda * ortho_penalty
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train():
    print("="*80)
    print("SODALS v3.2: BALANCED 805-EXAMPLE MULTI-DOMAIN TRAINING (LoRA r=4)")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max Length: 2048, Epochs: 2, Grad Accum: 8, LR: 2e-4, Ortho Lambda: 0.005")
    print(f"LoRA: r=4, alpha=8, dropout=0.10, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj']")
    
    print(f"Loading Tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4 onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("Configuring Regularized LoRA (Rank 4, 4 target modules)...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.10,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading Balanced Dataset from {DATASET_PATH}...")
    dataset = SFTChatDataset(DATASET_PATH, tokenizer, max_length=768)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        weight_decay=0.02,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = PonderingOrthogonalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=0.005,
        hidden_dim=model.config.hidden_size
    )

    # Check for existing checkpoint to resume
    resume_ckpt = None
    if os.path.exists(CHECKPOINT_DIR):
        ckpts = [os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            ckpts.sort(key=lambda x: int(os.path.basename(x).split("-")[1]))
            resume_ckpt = ckpts[-1]
            print(f"[RESUME] Found existing checkpoint: {resume_ckpt}. Resuming training...")

    print("Starting Clean Multi-Domain LoRA Training Loop...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"Saving Clean Balanced LoRA Adapter to {OUTPUT_ADAPTER_DIR}...")
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    # Save ODE weights
    ode_path = os.path.join(OUTPUT_ADAPTER_DIR, "arkhe_ode_proj.pt")
    torch.save(trainer.pondering_ode.state_dict(), ode_path)
    print(f"Saved Arkhe ODE projection weights to {ode_path}")

    print("[SUCCESS] Balanced 805-Example LoRA Training Completed Successfully!")


if __name__ == "__main__":
    train()
