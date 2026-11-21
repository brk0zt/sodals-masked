"""
SODALS v2.1 — GPR Phase Router Calibration
===========================================
Calibrates domain anchor vectors (Domain 1: General STEM vs Domain 2: Abstract Algebra)
using Qwen 3.5 9B latent manifold hidden states.
Evaluates which transformer layer (Embedding, Layer 4, Layer 8, Layer 16, or Layer 32)
yields maximum topological separation (Fisher criterion / Z-score separation).
Saves calibrated anchors to: sft/gpr_domain_anchors.pt
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
OUTPUT_ANCHORS_PATH = os.path.join(ROOT, "sft", "gpr_domain_anchors.pt")
DOMAIN2_REF_PATH = os.path.join(ROOT, "domain2_20q_interleaved_r8_results.json")

def load_calibration_data():
    print("Loading calibration samples...", flush=True)
    # 30 MATH-500 prompts (Domain 1)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    d1_prompts = [ds[i]["problem"] for i in range(30)]

    # 20 Domain 2 prompts (Abstract Algebra)
    with open(DOMAIN2_REF_PATH, "r", encoding="utf-8") as f:
        d2_raw = json.load(f)["results"]
    d2_prompts = [it["prompt"] for it in d2_raw]

    print(f"Loaded {len(d1_prompts)} Domain 1 samples and {len(d2_prompts)} Domain 2 samples.", flush=True)
    return d1_prompts, d2_prompts

def extract_prompt_representations(model, tokenizer, prompts, target_layers=[0, 4, 8, 16, 32]):
    """
    Extracts mean-pooled or last-token hidden states at specified layers for each prompt.
    """
    layer_vectors = {l: [] for l in target_layers}

    for idx, p in enumerate(prompts):
        chat_prompt = (
            f"<|im_start|>user\n{p}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            for l in target_layers:
                h = out.hidden_states[l][0]  # [seq_len, hidden_dim]
                h_mean = h.mean(dim=0).cpu()  # [hidden_dim]
                layer_vectors[l].append(h_mean)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(prompts):
            print(f"  Processed {idx + 1}/{len(prompts)} prompts...", flush=True)

    return {l: torch.stack(vecs, dim=0) for l, vecs in layer_vectors.items()}

def calibrate():
    print("=" * 70, flush=True)
    print("SODALS GPR PHASE ROUTER CALIBRATION", flush=True)
    print("=" * 70, flush=True)

    d1_prompts, d2_prompts = load_calibration_data()

    print("Loading Tokenizer & Model in 4-bit NF4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    target_layers = [0, 4, 8, 16, 32]
    print("\nExtracting representations for Domain 1 (General STEM)...", flush=True)
    d1_reps = extract_prompt_representations(model, tokenizer, d1_prompts, target_layers)

    print("\nExtracting representations for Domain 2 (Abstract Algebra)...", flush=True)
    d2_reps = extract_prompt_representations(model, tokenizer, d2_prompts, target_layers)

    print("\nEvaluating Layer Separation Performance:", flush=True)
    best_layer = None
    best_separation = -1.0
    best_anchors = {}

    for l in target_layers:
        v1 = d1_reps[l]  # [N1, 4096]
        v2 = d2_reps[l]  # [N2, 4096]

        c1 = v1.mean(dim=0)
        c2 = v2.mean(dim=0)

        c1_norm = F.normalize(c1, p=2, dim=0)
        c2_norm = F.normalize(c2, p=2, dim=0)

        v1_norm = F.normalize(v1, p=2, dim=1)
        v2_norm = F.normalize(v2, p=2, dim=1)

        s1_to_c1 = torch.mv(v1_norm, c1_norm)
        s1_to_c2 = torch.mv(v1_norm, c2_norm)
        diff_d1 = (s1_to_c2 - s1_to_c1).numpy()

        s2_to_c1 = torch.mv(v2_norm, c1_norm)
        s2_to_c2 = torch.mv(v2_norm, c2_norm)
        diff_d2 = (s2_to_c2 - s2_to_c1).numpy()

        mu1 = float(diff_d1.mean())
        mu2 = float(diff_d2.mean())
        var1 = float(diff_d1.var())
        var2 = float(diff_d2.var())
        fisher = (mu2 - mu1)**2 / (var1 + var2 + 1e-8)

        threshold = (mu1 + mu2) / 2.0
        acc_d1 = (diff_d1 < threshold).mean() * 100.0
        acc_d2 = (diff_d2 >= threshold).mean() * 100.0
        total_acc = (acc_d1 * len(diff_d1) + acc_d2 * len(diff_d2)) / (len(diff_d1) + len(diff_d2))

        print(f"Layer {l:2d}: Fisher Sep = {fisher:6.2f} | D1 diff = {mu1:+.4f} (±{var1**0.5:.4f}) | D2 diff = {mu2:+.4f} (±{var2**0.5:.4f}) | Acc = {total_acc:.1f}% (D1: {acc_d1:.1f}%, D2: {acc_d2:.1f}%)", flush=True)

        if fisher > best_separation:
            best_separation = fisher
            best_layer = l
            best_anchors = {
                "layer": l,
                "anchor_d1": c1_norm,
                "anchor_d2": c2_norm,
                "mu_d1": mu1,
                "mu_d2": mu2,
                "var_d1": var1,
                "var_d2": var2,
                "threshold": threshold,
                "fisher_separation": fisher
            }

    print(f"\nOptimal Layer Selected: Layer {best_layer} (Fisher Separation: {best_separation:.2f})", flush=True)

    torch.save(best_anchors, OUTPUT_ANCHORS_PATH)
    print(f"Saved calibrated GPR domain anchors to: {OUTPUT_ANCHORS_PATH}", flush=True)
    print("Calibration complete!", flush=True)

if __name__ == "__main__":
    calibrate()
