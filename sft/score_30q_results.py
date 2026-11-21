import os
import sys
import json
import re
import math

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "sft") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "sft"))

from build_ultimate_master_report import QUESTION_INFO

DATA_PATH = os.path.join(ROOT, "hle_results_baseline_control_30q_20260818.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

correct = 0
results_list = []

for qid, item in data.items():
    resp = item.get("response", "")
    gt = item.get("ground_truth", "")
    if not gt:
        gt = QUESTION_INFO.get(qid, {}).get("gt", "")
    
    clean_gt = gt.replace("$", "").replace("*", "").replace('"', "").strip()
    
    m = re.search(r"Answer:\s*(.*)", resp, re.IGNORECASE)
    ans = m.group(1).strip().split("\n")[0].rstrip('.').strip() if m else "NO_ANSWER"
    
    is_match = (ans == clean_gt or ans == gt)
    if is_match:
        correct += 1
    
    results_list.append({
        "id": qid,
        "cat": item.get("category", "Math"),
        "ans": ans[:28],
        "gt": clean_gt[:28],
        "match": is_match,
        "tokens": item.get("usage", {}).get("completion_tokens", 0),
        "sec": item.get("usage", {}).get("elapsed_seconds", 0)
    })

def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    diff = z * math.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0.0, center - diff), min(1.0, center + diff))

ci_low, ci_high = wilson_ci(correct, len(data))
width = ci_high - ci_low

print("=" * 80)
print(f"  BASELINE 30-QUESTION EXTENDED BENCHMARK RESULTS")
print(f"  Exact Match Score: {correct} / {len(data)} (%{correct/len(data)*100:.1f})")
print(f"  Wilson 95% Confidence Interval: [%{ci_low*100:.1f}, %{ci_high*100:.1f}]")
print(f"  CI Width: %{width*100:.1f} (Narrow CI target achieved: {width < 0.15})")
print("=" * 80)

print("\nDetailed Per-Question Audit:")
for idx, r in enumerate(results_list, 1):
    mark = "✅" if r["match"] else "❌"
    print(f"[{idx:02d}/30] {mark} {r['id'][:8]} [{r['cat'][:7]:<7}] -> Ans: {r['ans']:<28} | GT: {r['gt']:<28} ({r['tokens']} tok, {r['sec']}s)")
