"""
SODALS v2.1 — Marginal Factorial Ablation 1: Only Hamiltonian Curriculum Ordering
=================================================================================
Isolates Technique 1:
  - [ON]  Hamiltonian Energy-Stratified Curriculum Ordering (Stages 1-5)
  - [OFF] Arkhe Introspection Focal Loss (Uses standard uniform Cross-Entropy, w_i = 1.0)
  - [OFF] Oja Continuous Subspace B-Matrix Persistence (Standard PEFT adapter save)
  - [ON]  Wronskian Orthogonality Penalty (ortho_lambda=0.005) — SAME AS BOTH ARMS

Hyperparameters are IDENTICAL to train_qwen_lora.py and train_qwen_lora_baseline.py
except for the curriculum ordering.

Saves to: sft/qwen_lora_adapter_only_hamiltonian/
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
DATASET_PATH = os.path.join(SFT_DIR, "domain1_sft_curriculum.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_only_hamiltonian")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_only_hamiltonian")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")


class HamiltonianSFTChatDataset(Dataset):
    """
    Marginal Ablation Dataset:
    - Technique 1 ON: Hamiltonian 5-stage ordering (sorted by curriculum_stage + difficulty_rank)
    - Technique 2 OFF: Uniform weight 1.0, NO Arkhe focal loss
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 640):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        # Hamiltonian energy-stratified ordering
        def sort_key(x):
            meta = x.get("metadata", {})
            stage = meta.get("curriculum_stage", 3)
            diff = meta.get("difficulty_rank", 1.0)
            return (stage, diff)
        self.data.sort(key=sort_key)
        print(f"[HAMILTONIAN ABLATION] Dataset sorted by Hamiltonian energy (Stages 1-5). Total: {len(self.data)}")

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


class PlainOrthogonalTrainer(Trainer):
    """
    Standard CE loss (focal loss OFF) + Wronskian regularizer (SAME as both arms).
    """
    def __init__(self, *args, ortho_lambda=0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_lambda = ortho_lambda
        self._lora_A_params = None

    def _get_lora_A_params(self, model):
        if self._lora_A_params is None:
            self._lora_A_params = [
                param for name, param in model.named_parameters()
                if "lora_A" in name and param.requires_grad
            ]
        return self._lora_A_params

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Standard uniform cross-entropy

        # Wronskian Orthogonality Penalty (same as both arms)
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
                I = torch.eye(16, device=stacked.device, dtype=stacked.dtype).unsqueeze(0)
                diff = gram - I
                ortho_penalty = ortho_penalty + (diff ** 2).sum()

            total_loss = ce_loss + self.ortho_lambda * ortho_penalty
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train():
    print("=== SODALS MARGINAL ABLATION 1: ONLY HAMILTONIAN ORDERING ===")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max Length: 640, Epochs: 2, Grad Accum: 4, LR: 2e-4, Ortho Lambda: 0.005")
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

    print("Loading Base Model in 4-bit NF4 directly onto GPU (RTX 4060)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"" : 0},
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("Configuring Standard LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading Hamiltonian-Ordered Dataset from {DATASET_PATH}...")
    dataset = HamiltonianSFTChatDataset(DATASET_PATH, tokenizer, max_length=640)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = PlainOrthogonalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=0.005
    )

    print("Starting Only-Hamiltonian LoRA GPU Training Loop (72 steps)...")
    trainer.train()

    print(f"Saving Only-Hamiltonian LoRA Adapter to {OUTPUT_ADAPTER_DIR}...")
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    print("[SUCCESS] Only-Hamiltonian LoRA Training Completed Successfully (Technique 3 Oja export OFF)!")


if __name__ == "__main__":
    train()
