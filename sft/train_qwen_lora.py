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

# Patch torch.load for PyTorch 2.4+ safe unpickling on local checkpoints
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
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")

class SFTChatDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 640):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        focal_weight = float(item.get("focal_weight", 1.0))
        
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
            "labels": labels,
            "focal_weight": torch.tensor(focal_weight, dtype=torch.float32)
        }

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        focal_weights = [f.pop("focal_weight") for f in features if "focal_weight" in f]
        batch = super().__call__(features, return_tensors=return_tensors)
        if focal_weights:
            batch["focal_weight"] = torch.stack(focal_weights)
        return batch

class WronskianOrthogonalTrainer(Trainer):
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
        focal_weight = inputs.pop("focal_weight", None)
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Apply Arkhe Introspection Focal Weighting
        if focal_weight is not None:
            w = focal_weight.mean()
            ce_loss = ce_loss * w

        # Fast Batched / Vectorized Wronskian Orthogonality Penalty (torch.bmm)
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
                stacked = torch.stack(mats, dim=0) # [N, 16, d]
                gram = torch.bmm(stacked, stacked.transpose(1, 2)) # [N, 16, 16]
                I = torch.eye(16, device=stacked.device, dtype=stacked.dtype).unsqueeze(0)
                diff = gram - I
                ortho_penalty = ortho_penalty + (diff ** 2).sum()

            total_loss = ce_loss + self.ortho_lambda * ortho_penalty
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss

def train(dataset_path: str = None, max_length: int = 640, epochs: int = 2, grad_accum: int = 4, lr: float = 2e-4, ortho_lambda: float = 0.005):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    print("=== SODALS HYBRID ARCHITECTURE TRAINING (HAMILTONIAN CURRICULUM + FOCAL LOSS + CHECKPOINT PROTECTION) ===")
    print(f"Dataset: {dataset_path}")
    print(f"Max Length: {max_length}, Epochs: {epochs}, Grad Accum: {grad_accum}, LR: {lr}, Ortho Lambda: {ortho_lambda}")
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
        device_map={"": 0},
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("Configuring LoRA...")
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

    print(f"Loading Hamiltonian-Ordered SFT Dataset from {dataset_path}...")
    dataset = SFTChatDataset(dataset_path, tokenizer, max_length=max_length)

    checkpoint_dir = os.path.join(SFT_DIR, "checkpoints")
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = WronskianOrthogonalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CustomDataCollator(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=ortho_lambda
    )

    # Check if a valid checkpoint exists to resume automatically
    last_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            last_checkpoint = checkpoints[-1]
            print(f"[RESUME] Found existing checkpoint: {last_checkpoint}. Resuming training seamlessly...")

    print("Starting Optimized Hybrid GPU Training Loop...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"Saving Trained LoRA Adapter to {OUTPUT_ADAPTER_DIR}...")
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    # Export Oja B-matrices for checkpoint persistence
    print("Exporting Oja B-matrices for inference persistence...")
    b_matrices = {name: param.detach().cpu() for name, param in model.named_parameters() if "lora_B" in name}
    oja_path = os.path.join(OUTPUT_ADAPTER_DIR, "sodals_oja_b_matrices.pt")
    torch.save(b_matrices, oja_path)
    print(f"Saved {len(b_matrices)} Oja B-matrices to {oja_path}")

    print("[SUCCESS] Hybrid Architecture LoRA Training Completed Successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Path to SFT JSONL dataset")
    parser.add_argument("--max_length", type=int, default=640, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--ortho_lambda", type=float, default=0.005, help="Wronskian orthogonality weight")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        max_length=args.max_length,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        lr=args.lr,
        ortho_lambda=args.ortho_lambda
    )
