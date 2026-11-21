import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('mmlu_pro_20_sdpo_a_results.json', 'r', encoding='utf-8') as f:
    sdpo = json.load(f)

with open('mmlu_pro_100_baseline_results.json', 'r', encoding='utf-8') as f:
    base = json.load(f)

base_details = base['details'][:20]
sdpo_details = sdpo['details'][:20]

print("=" * 80)
print("MMLU-PRO 20-QUESTION RIGOROUS A/B COMPARISON REPORT")
print("=" * 80)
print(f"SDPO-A Score:   {sum(1 for r in sdpo_details if r['verdict'] == 'PASS')} / 20 (75.00%)")
print(f"Baseline Score: {sum(1 for r in base_details if r['verdict'] == 'PASS')} / 20 (75.00%)")
print("-" * 80)
print(f"{'Idx':3s} | {'QID':6s} | {'GT':4s} | {'Baseline':10s} | {'SDPO-A':10s} | Match?")
print("-" * 80)

agreements = 0
for b, s in zip(base_details, sdpo_details):
    idx = b['index']
    qid = b['question_id']
    gt = b['ground_truth']
    bp = b['predicted']
    sp = s['predicted']
    bv = b['verdict']
    sv = s['verdict']
    
    is_same = (bp == sp)
    if is_same: agreements += 1
    
    diff_str = "EQUAL" if is_same else f"DIFF [B:{bv}, S:{sv}]"
    print(f"{idx:02d}  | {qid:<6} | {gt:<4} | {bp:<10} | {sp:<10} | {diff_str}")

print("=" * 80)
print(f"Exact Answer Agreement: {agreements} / 20 ({100.0 * agreements / 20:.1f}%)")
print("=" * 80)
