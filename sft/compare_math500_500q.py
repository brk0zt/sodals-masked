import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_lora_4096_results.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_sdpo_a_results.json", "r", encoding="utf-8") as f:
    sdpo_data = json.load(f)

base_details = {r["unique_id"]: r for r in base_data["details"]}
sdpo_details = {r["unique_id"]: r for r in sdpo_data["details"]}

common_ids = [r["unique_id"] for r in base_data["details"] if r["unique_id"] in sdpo_details]

print("=" * 80)
print(f"FULL 500-QUESTION MATH-500 RIGOROUS COMPARISON REPORT (N={len(common_ids)})")
print("=" * 80)

base_pass = sum(1 for uid in common_ids if base_details[uid]["verdict"] == "PASS")
sdpo_pass = sum(1 for uid in common_ids if sdpo_details[uid]["verdict"] == "PASS")

print(f"Balanced LoRA Baseline (balanced_r4): {base_pass} / {len(common_ids)} ({100.0*base_pass/len(common_ids):.2f}%)")
print(f"Anchored SDPO (SDPO-A):              {sdpo_pass} / {len(common_ids)} ({100.0*sdpo_pass/len(common_ids):.2f}%)")
print(f"Net Delta:                           {sdpo_pass - base_pass:+d} questions ({100.0*(sdpo_pass - base_pass)/len(common_ids):+.2f}%)")

both_pass = 0
both_fail = 0
base_only = 0
sdpo_only = 0

base_only_list = []
sdpo_only_list = []

for uid in common_ids:
    bp = (base_details[uid]["verdict"] == "PASS")
    sp = (sdpo_details[uid]["verdict"] == "PASS")
    
    if bp and sp:
        both_pass += 1
    elif not bp and not sp:
        both_fail += 1
    elif bp and not sp:
        base_only += 1
        base_only_list.append(base_details[uid])
    elif not bp and sp:
        sdpo_only += 1
        sdpo_only_list.append(sdpo_details[uid])

print("\n--- Contingency Matrix ---")
print(f"Both Models PASSED: {both_pass} ({100.0*both_pass/len(common_ids):.1f}%)")
print(f"Both Models FAILED: {both_fail} ({100.0*both_fail/len(common_ids):.1f}%)")
print(f"Agreement Rate:     {both_pass + both_fail} / {len(common_ids)} ({100.0*(both_pass+both_fail)/len(common_ids):.1f}%)")
print(f"Baseline Won:       {base_only} questions")
print(f"SDPO-A Won:         {sdpo_only} questions")

# Subject level breakdown
subjects = sorted(list(sdpo_data["subj_stats"].keys()))
print("\n--- Subject-by-Subject Breakdown ---")
print(f"{'Subject':24s} | {'Baseline':12s} | {'SDPO-A':12s} | {'Delta':8s}")
print("-" * 62)
for s in subjects:
    b_s = base_data["subj_stats"].get(s, {"pass": 0, "total": 0})
    s_s = sdpo_data["subj_stats"].get(s, {"pass": 0, "total": 0})
    b_acc = 100.0 * b_s["pass"] / b_s["total"] if b_s["total"] > 0 else 0
    s_acc = 100.0 * s_s["pass"] / s_s["total"] if s_s["total"] > 0 else 0
    d_q = s_s["pass"] - b_s["pass"]
    print(f"{s:24s} | {b_s['pass']:3d}/{b_s['total']:3d} ({b_acc:4.1f}%) | {s_s['pass']:3d}/{s_s['total']:3d} ({s_acc:4.1f}%) | {d_q:+2d} ({s_acc-b_acc:+4.1f}%)")
print("=" * 80)
