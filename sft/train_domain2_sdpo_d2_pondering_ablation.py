"""
SODALS v2.1 — Domain 2 SDPO-D2 + PonderingODE Ablation
=======================================================
ISOLATED ABLATION: Identical to train_domain2_sdpo_lora.py EXCEPT:
  - PonderingODEModule is ON (loaded from balanced_r4 arkhe_ode_proj.pt)
  - Loss computed through pondered hidden states (matching Balanced R4 pipeline)

This tests whether the PonderingODE train/inference distribution mismatch
was the root cause of SDPO-D2's 70% accuracy (vs frozen base 90%).

All other hyperparameters are IDENTICAL to the original SDPO-D2 training:
  - Prior Adapter: sft/qwen_lora_adapter_sdpo_anchored
  - LoRA rank r=4, alpha=8, targets=[q, k, v, o]
  - 1 Epoch, 45 Steps
  - batch_size=1, grad_accum=4
  - Learning rate: 3e-5 (cosine scheduler with 4 warmup steps)
  - Wronskian regularizer: ortho_lambda = 0.005 (fully vectorized BMM)
  - Output Adapter: sft/qwen_lora_adapter_domain2_pondering/
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

_orig_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _custom_torch_load

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
DATASET_PATH = os.path.join(SFT_DIR, "domain2_anchored_164_curriculum.jsonl")
OUTPUT_ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_pondering")
CHECKPOINT_DIR = os.path.join(SFT_DIR, "checkpoints_domain2_pondering")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
PRIOR_ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored")
ODE_WEIGHTS_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4", "arkhe_ode_proj.pt")

# Import PonderingODEModule
sys.path.insert(0, SFT_DIR)
from pondering_core_bridge import PonderingODEModule


class Domain2SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 768):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"[SDPO-D2+PONDER] Loaded {len(self.data)} samples from {jsonl_path}.", flush=True)

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


class PonderingDomain2Trainer(Trainer):
    """
    ABLATION Trainer: OrthogonalDomain2Trainer + PonderingODEModule.
    Matches the Balanced R4 gradient flow by computing loss through
    pondered hidden states instead of raw model outputs.
    """
    def __init__(self, *args, ortho_lambda=0.005, hidden_dim=4096, ode_weights_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_lambda = ortho_lambda
        self._lora_A_params = None

        # Initialize PonderingODE with same hyperparams as Balanced R4
        self.pondering_ode = PonderingODEModule(
            hidden_dim=hidden_dim, dt=0.1, max_steps=4, lambda_anchor=0.3
        )

        # Load trained ODE weights from Balanced R4
        if ode_weights_path and os.path.exists(ode_weights_path):
            ode_state = _orig_torch_load(ode_weights_path, map_location='cpu', weights_only=False)
            self.pondering_ode.load_state_dict(ode_state)
            print(f"[PONDER] Loaded trained ODE weights from {ode_weights_path}", flush=True)
            print(f"[PONDER] arkhe_proj weight norm: {self.pondering_ode.arkhe_proj.weight.norm():.4f}", flush=True)
        else:
            print(f"[PONDER] WARNING: No ODE weights found at {ode_weights_path}, using random init!", flush=True)

    def _get_lora_A_params(self, model):
        if self._lora_A_params is None:
            self._lora_A_params = [
                param for name, param in model.named_parameters()
                if "lora_A" in name and param.requires_grad
            ]
        return self._lora_A_params

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # Forward pass requesting hidden states (matching Balanced R4)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, dim]
        device = last_hidden.device
        dtype = last_hidden.dtype

        # Free raw logits to save VRAM
        if hasattr(outputs, "logits"):
            outputs.logits = None

        # Sync PonderingODE device/dtype
        if self.pondering_ode.arkhe_proj.weight.device != device or \
           self.pondering_ode.arkhe_proj.weight.dtype != dtype:
            self.pondering_ode = self.pondering_ode.to(device=device, dtype=dtype)

        # 1. Apply RK4 Continuous Latent ODE Pondering
        pondered_hidden = self.pondering_ode(last_hidden)

        # 2. Chunked lm_head + Cross-Entropy (matching Balanced R4 exactly)
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

        # 3. Wronskian Orthogonality Penalty (vectorized BMM, same as original)
        if self.ortho_lambda > 0:
            lora_A_list = self._get_lora_A_params(model)
            if lora_A_list:
                # Group by dimension for BMM
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


def train(epochs: int = 1):
    steps = (180 // 4) * epochs
    print("=" * 80, flush=True)
    print(f"ABLATION: SDPO-D2 + PonderingODE ({steps} STEPS / {epochs} EPOCH(S))", flush=True)
    print("=" * 80, flush=True)
    print("[ABLATION] This run is IDENTICAL to SDPO-D2 except PonderingODE is ON.", flush=True)
    print("[ABLATION] Testing hypothesis: PonderingODE mismatch caused 90%->70% drop.", flush=True)

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

    hidden_dim = model.config.hidden_size
    print(f"[PONDER] Model hidden_size={hidden_dim}", flush=True)

    trainer = PonderingDomain2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        ortho_lambda=0.005,
        hidden_dim=hidden_dim,
        ode_weights_path=ODE_WEIGHTS_PATH
    )

    print(f"Executing SDPO-D2+PonderingODE Training ({steps} steps)...", flush=True)
    trainer.train()

    print(f"Saving SDPO-D2+Ponder Adapter to {OUTPUT_ADAPTER_DIR}...", flush=True)
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    # Save updated ODE weights
    ode_path = os.path.join(OUTPUT_ADAPTER_DIR, "arkhe_ode_proj.pt")
    _orig_torch_load_ref = torch.load  # Use patched version
    torch.save(trainer.pondering_ode.state_dict(), ode_path)
    print(f"Saved Arkhe ODE projection weights to {ode_path}", flush=True)

    print("[SUCCESS] SDPO-D2 + PonderingODE Ablation Training Completed!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(epochs=args.epochs)
