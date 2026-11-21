import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
BASE_MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_baseline")
MERGED_OUTPUT_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")

def merge_and_export():
    print("=== SODALS BASELINE WEIGHT MERGE: W_new = W_base + dW_baseline ===")
    print(f"Base Model: {BASE_MODEL_PATH}")
    print(f"Baseline LoRA Adapter: {ADAPTER_PATH}")
    print(f"Output Merged Directory: {MERGED_OUTPUT_DIR}")

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Baseline LoRA adapter not found at {ADAPTER_PATH}.")

    print("Loading Base Model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print("Loading Peft Model and attaching Baseline LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Executing Matrix Merge (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged baseline model to {MERGED_OUTPUT_DIR}...")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    print("[SUCCESS] Standalone Plain Baseline Merged Model Saved Successfully!")

if __name__ == "__main__":
    merge_and_export()
