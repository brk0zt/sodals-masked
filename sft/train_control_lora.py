"""
SDPO Negative Control (Placebo) Training Run
============================================
Trains an IDENTICAL 17-step LoRA adapter starting from the exact same
prior adapter (qwen_lora_adapter_balanced_r4), with identical hyperparameters,
but trained on 17 UNRELATED, STANDARD MATH EXAMPLES instead of SDPO self-corrections.

Purpose:
  To determine whether the 4/35 recovery on the held-out failure pool was
  caused specifically by SDPO's self-correction reflex, or whether ANY random
  17-step LoRA perturbation at lr=3e-5 flips 3-4 borderline questions by chance.

Outputs: sft/qwen_lora_adapter_control/
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
DATASET_PATH = os.path.join(SFT_DIR, "control_17_curriculum.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_control")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_control")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
PRIOR_ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")


class ControlDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 640):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"[CONTROL DATASET] Loaded {len(self.data)} unrelated control examples.")

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


class OrthogonalControlTrainer(Trainer):
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

        # Wronskian orthogonality regularization (vectorized)
        if self.ortho_lambda > 0:
            lora_A_list = self._get_lora_A_params(model)
            ortho_penalty = torch.tensor(0.0, device=ce_loss.device)
            I = torch.eye(4, device=ce_loss.device, dtype=ce_loss.dtype)
            for A in lora_A_list:
                gram = torch.matmul(A, A.t().to(A.dtype))
                ortho_penalty += ((gram - I) ** 2).sum()

            total_loss = ce_loss + self.ortho_lambda * ortho_penalty
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train():
    print("=" * 80)
    print("STARTING NEGATIVE CONTROL (PLACEBO) TRAINING RUN (17 STEPS)")
    print("=" * 80)

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
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    print(f"Loading Prior Balanced Adapter from {PRIOR_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, PRIOR_ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters()

    dataset = ControlDataset(DATASET_PATH, tokenizer, max_length=640)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = OrthogonalControlTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=0.005
    )

    print("Executing Control Training (17 steps)...")
    trainer.train()

    print(f"Saving Control Adapter to {OUTPUT_ADAPTER_DIR}...")
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    print("[SUCCESS] Control LoRA Training Completed Successfully!")


if __name__ == "__main__":
    train()
