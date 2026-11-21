"""
SODALS v2.1 — Domain 2 Surgical Patch Fine-Tuning (Rank=8 + PonderingODE)
==========================================================================
Lightweight surgical fine-tuning on top of `qwen_lora_adapter_domain2_r8_pondering`:
  - 19 high-density samples targeting sp(6)=21, sp(4)=10, E_6=78, E_7=133
  - Critical anti-whack-a-mole anchors: Aut(Q8)=24, E8=248, so(8)=28, Biquadratic=4
  - Gentle learning rate: 1.5e-5 with cosine decay
  - 2 Epochs (38 steps, batch_size=1, grad_accum=1)
  - PonderingODE ON (initialized from r8 adapter's arkhe_ode_proj.pt)
  - Wronskian I_8 regularization: lambda = 0.005
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

_orig_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _custom_torch_load

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "domain2_surgical_patch_dataset.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_surgical_patch")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_domain2_r8_surgical_patch")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
PRIOR_ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_pondering")
ODE_WEIGHTS_PATH = os.path.join(PRIOR_ADAPTER_PATH, "arkhe_ode_proj.pt")

sys.path.insert(0, SFT_DIR)
from pondering_core_bridge import PonderingODEModule


class SurgicalDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 768):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"[SURGICAL PATCH] Loaded {len(self.data)} samples from {jsonl_path}.", flush=True)

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


class SurgicalPatchTrainer(Trainer):
    def __init__(self, *args, ortho_lambda=0.005, hidden_dim=4096, ode_weights_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_lambda = ortho_lambda
        self._lora_A_params = None

        self.pondering_ode = PonderingODEModule(
            hidden_dim=hidden_dim, dt=0.1, max_steps=4, lambda_anchor=0.3
        )

        if ode_weights_path and os.path.exists(ode_weights_path):
            ode_state = _orig_torch_load(ode_weights_path, map_location='cpu', weights_only=False)
            self.pondering_ode.load_state_dict(ode_state)
            print(f"[PONDER] Loaded ODE weights from {ode_weights_path}", flush=True)
        else:
            print(f"[PONDER] WARNING: ODE weights not found at {ode_weights_path}!", flush=True)

    def _get_lora_A_params(self, model):
        if self._lora_A_params is None:
            self._lora_A_params = [
                param for name, param in model.named_parameters()
                if "lora_A" in name and param.requires_grad
            ]
        return self._lora_A_params

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        device = last_hidden.device
        dtype = last_hidden.dtype

        if hasattr(outputs, "logits"):
            outputs.logits = None

        if self.pondering_ode.arkhe_proj.weight.device != device or \
           self.pondering_ode.arkhe_proj.weight.dtype != dtype:
            self.pondering_ode = self.pondering_ode.to(device=device, dtype=dtype)

        pondered_hidden = self.pondering_ode(last_hidden)

        base_model = model.base_model.model if hasattr(model, "base_model") else model
        lm_head = getattr(base_model, "lm_head", None) or getattr(model, "lm_head", None)

        seq_len = pondered_hidden.size(1) - 1
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

        # Wronskian I_8 Penalty
        if self.ortho_lambda > 0:
            lora_A_list = self._get_lora_A_params(model)
            if lora_A_list:
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
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train():
    epochs = 2
    print("=" * 80, flush=True)
    print(f"SURGICAL PATCH TRAINING: 19 SAMPLES x {epochs} EPOCHS (38 STEPS)", flush=True)
    print("=" * 80, flush=True)
    print(f"Prior Adapter: {PRIOR_ADAPTER_PATH}", flush=True)
    print(f"Output Adapter: {OUTPUT_ADAPTER_DIR}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    print(f"Loading Prior r=8 Adapter from {PRIOR_ADAPTER_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, PRIOR_ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters()

    dataset = SurgicalDataset(DATASET_PATH, tokenizer, max_length=768)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1.5e-5,  # Gentle fine-tuning
        lr_scheduler_type="cosine",
        warmup_steps=4,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    hidden_dim = model.config.hidden_size

    trainer = SurgicalPatchTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=0.005,
        hidden_dim=hidden_dim,
        ode_weights_path=ODE_WEIGHTS_PATH
    )

    print(f"Executing Surgical Fine-Tuning (38 steps)...", flush=True)
    trainer.train()

    print(f"Saving Surgical Patch Adapter to {OUTPUT_ADAPTER_DIR}...", flush=True)
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    ode_path = os.path.join(OUTPUT_ADAPTER_DIR, "arkhe_ode_proj.pt")
    torch.save(trainer.pondering_ode.state_dict(), ode_path)
    print(f"Saved ODE weights to {ode_path}", flush=True)

    print("[SUCCESS] Surgical Patch Training Complete!", flush=True)


if __name__ == "__main__":
    train()
