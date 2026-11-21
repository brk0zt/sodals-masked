import json
import os
import re

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
K1_FILE = os.path.join(ROOT, "hle_sodals_Qwen3.5-9B.json")
K2_FILE = os.path.join(ROOT, "hle_k2_base_sodals_Qwen3.5-9B.json")
K3_FILE = os.path.join(ROOT, "hle_k3_candle_pure_base_Qwen3.5-9B.json")
K4_FILE = os.path.join(ROOT, "hle_ollama_Qwen3.5-9B.json")
K5_FILE = os.path.join(ROOT, "hle_k5_pure_lora_Qwen3.5-9B.json")
K6_FILE = os.path.join(ROOT, "hle_k6_merged_weights_Qwen3.5-9B.json")

def load_json(p):
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def extract_answer(text):
    if not text:
        return "No Output"
    m = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        ans = ans.split("\n")[0].strip()
        if len(ans) > 40:
            ans = ans[:37] + "..."
        return f"`Answer: {ans}`"
    return "❌ `No Answer`"

def main():
    k1 = load_json(K1_FILE)
    k2 = load_json(K2_FILE)
    k3 = load_json(K3_FILE)
    k4 = load_json(K4_FILE)
    k5 = load_json(K5_FILE)
    k6 = load_json(K6_FILE)

    print("Loaded data summary:")
    print(f"K1 (SODALS 1.0): {len(k1)}")
    print(f"K2 (Base + Motor): {len(k2)}")
    print(f"K3 (Candle Pure Base): {len(k3)}")
    print(f"K4 (Ollama llama.cpp): {len(k4)}")
    print(f"K5 (Pure LoRA Cartridge): {len(k5)}")
    print(f"K6 (Closed-Book Merged): {len(k6)}")

if __name__ == "__main__":
    main()
