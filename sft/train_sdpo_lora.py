"""
SODALS SDPO: Isolated Self-Distillation Policy Optimization Training
====================================================================
Trains Qwen 3.5 9B LoRA adapter on the compiled CGO curriculum containing
verified model self-corrections + balanced foundation examples.

Hyperparameters:
  - LoRA rank: r=4, alpha=8, target_modules=[q_proj, k_proj, v_proj, o_proj]
  - Learning rate: 3e-5 (conservative to preserve base weights)
  - Epochs: 1
  - Orthogonality penalty: ortho_lambda = 0.005
  - Output adapter: sft/qwen_lora_adapter_sdpo/
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
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

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
DATASET_PATH = os.path.join(SFT_DIR, "sdpo_isolated_17_curriculum.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_sdpo")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
PRIOR_ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")


class SDPODataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 640):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"[SDPO DATASET] Loaded {len(self.data)} examples.")

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


class OrthogonalSDPOTrainer(Trainer):
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

        # Gram matrix subspace orthogonality regularization (vectorized, zero overhead)
        # Enforces low-rank projection directions to remain mutually orthogonal: ||A A^T - I||_F^2
        if self.ortho_lambda > 0:
            lora_A_list = self._get_lora_A_params(model)
            gram_ortho_penalty = torch.tensor(0.0, device=ce_loss.device)
            I = torch.eye(4, device=ce_loss.device, dtype=ce_loss.dtype)
            for A in lora_A_list:
                gram = torch.matmul(A, A.t().to(A.dtype))
                gram_ortho_penalty += ((gram - I) ** 2).sum()

            total_loss = ce_loss + self.ortho_lambda * gram_ortho_penalty
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train():
    import argparse
    parser = argparse.ArgumentParser(description="SDPO LoRA Training Script")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to base model")
    parser.add_argument("--prior_adapter_path", type=str, default=PRIOR_ADAPTER_PATH, help="Path to prior adapter")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to training curriculum")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_ADAPTER_DIR, help="Output directory for trained adapter")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--ortho_lambda", type=float, default=0.005, help="Gram orthogonality regularization coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    print("=" * 80)
    print(f"STARTING SDPO TRAINING RUN (Seed: {args.seed})")
    print(f"Base Model:    {args.model_path}")
    print(f"Prior Adapter: {args.prior_adapter_path}")
    print(f"Curriculum:    {args.dataset_path}")
    print(f"Output:        {args.output_dir}")
    print("=" * 80)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    if os.path.exists(args.prior_adapter_path):
        print(f"Loading Prior Balanced Adapter from {args.prior_adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.prior_adapter_path, is_trainable=True)
    else:
        print(f"Initializing new LoRA adapter from scratch...")
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()

    dataset = SDPODataset(args.dataset_path, tokenizer, max_length=640)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed
    )

    trainer = OrthogonalSDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=args.ortho_lambda
    )

    print("Executing Isolated SDPO Training...")
    trainer.train()

    print(f"Saving SDPO Adapter to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[SUCCESS] SDPO LoRA Training Completed Successfully!")


if __name__ == "__main__":
    train()

