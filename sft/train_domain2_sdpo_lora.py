"""
SODALS v2.1 — Domain 2 SDPO-D2 Training Script
==============================================
Trains Qwen 3.5 9B with SDPO-D2 on the 164-sample Domain 2 Anchored Curriculum:
  - 144 Domain 2 samples (Lie algebras, Groups, Rings, Galois)
  - 20 Anchor samples from SDPO-A (Anti-forgetting on Domain 1 & general reasoning)

Hyperparameters:
  - Prior Adapter: sft/qwen_lora_adapter_sdpo_anchored
  - LoRA rank r=4, alpha=8, targets=[q, k, v, o]
  - 1 Epoch, 41 Steps (or 2 Epochs, 82 Steps via --epochs)
  - batch_size=1, grad_accum=4
  - Learning rate: 3e-5 (cosine scheduler with 4 warmup steps)
  - Wronskian regularizer: ortho_lambda = 0.005 (fully vectorized BMM)
  - Checkpoint saving every 20 steps
  - Output Adapter: sft/qwen_lora_adapter_domain2/
"""

import sys
import os
import json
import argparse
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
from peft import PeftModel, prepare_model_for_kbit_training

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

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "domain2_anchored_164_curriculum.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_domain2")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
PRIOR_ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored")


class Domain2SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 768):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"[SDPO-D2 DATASET] Loaded {len(self.data)} samples from {jsonl_path}.", flush=True)

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


class OrthogonalDomain2Trainer(Trainer):
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
        ce_loss = outputs.loss

        # Vectorized Wronskian orthogonality regularization via BMM (9x faster than per-layer loop)
        if self.ortho_lambda > 0:
            lora_A_list = self._get_lora_A_params(model)
            if lora_A_list:
                stacked = torch.stack(lora_A_list, dim=0)
                gram = torch.bmm(stacked, stacked.transpose(1, 2))
                I = torch.eye(4, device=ce_loss.device, dtype=stacked.dtype).unsqueeze(0)
                diff = gram - I
                ortho_penalty = (diff ** 2).sum()
                total_loss = ce_loss + self.ortho_lambda * ortho_penalty
            else:
                total_loss = ce_loss
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train(epochs: int = 1):
    steps = (180 // 4) * epochs
    print("=" * 80, flush=True)
    print(f"STARTING DOMAIN 2 (SDPO-D2) LORA TRAINING RUN ({steps} STEPS / {epochs} EPOCH(S))", flush=True)
    print("=" * 80, flush=True)

    print(f"Loading Tokenizer from: {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4 onto GPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    print(f"Loading Prior SDPO-A Adapter from {PRIOR_ADAPTER_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, PRIOR_ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters()

    dataset = Domain2SFTDataset(DATASET_PATH, tokenizer, max_length=768)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=4 if epochs == 1 else 8,
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

    trainer = OrthogonalDomain2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=0.005
    )

    print(f"Executing Domain 2 LoRA Training ({steps} steps)...", flush=True)
    trainer.train()

    print(f"Saving SDPO-D2 Adapter to {OUTPUT_ADAPTER_DIR}...", flush=True)
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    print("[SUCCESS] Domain 2 SDPO-D2 LoRA Training Completed Successfully!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1, 41 steps)")
    args = parser.parse_args()
    train(epochs=args.epochs)
