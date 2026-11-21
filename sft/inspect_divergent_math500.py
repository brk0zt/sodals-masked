import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_lora_4096_results.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_sdpo_a_results.json", "r", encoding="utf-8") as f:
    sdpo_data = json.load(f)

base_details = {r["unique_id"]: r for r in base_data["details"]}
sdpo_details = {r["unique_id"]: r for r in sdpo_data["details"]}

print("--- 7 Questions WON by SDPO-A (Baseline FAILED, SDPO-A PASSED) ---")
for uid, s in sdpo_details.items():
    b = base_details[uid]
    if b["verdict"] == "FAIL" and s["verdict"] == "PASS":
        print(f"[{s['subject']} | Lvl {s['level']}] {uid}")
        print(f"   GT: {s['ground_truth']} | SDPO-A: {s['predicted']} (PASS) | Base: {b['predicted']} (FAIL)")

print("\n--- 12 Questions WON by Baseline (Baseline PASSED, SDPO-A FAILED) ---")
for uid, b in base_details.items():
    s = sdpo_details[uid]
    if b["verdict"] == "PASS" and s["verdict"] == "FAIL":
        print(f"[{b['subject']} | Lvl {b['level']}] {uid}")
        print(f"   GT: {b['ground_truth']} | Base: {b['predicted']} (PASS) | SDPO-A: {s['predicted']} (FAIL)")
