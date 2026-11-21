import json
import sys
import math

sys.stdout.reconfigure(encoding='utf-8')

with open('mmlu_pro_100_sdpo_a_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# Take the first 20 questions
details_20 = d['details'][:20]
pass_20 = sum(1 for r in details_20 if r['verdict'] == 'PASS')

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

low, high = wilson_ci(pass_20, 20)

cat_stats = {}
for r in details_20:
    c = r.get('category', 'other')
    if c not in cat_stats: cat_stats[c] = {'pass': 0, 'total': 0}
    cat_stats[c]['total'] += 1
    if r['verdict'] == 'PASS': cat_stats[c]['pass'] += 1

payload_20 = {
    "model_key": "sdpo_a",
    "model_label": "Anchored SDPO (SDPO-A)",
    "adapter_path": d.get("adapter_path", ""),
    "total_evaluated": 20,
    "total_target": 20,
    "pass_count": pass_20,
    "accuracy": round(100.0 * pass_20 / 20, 2),
    "ci_95_low": round(low * 100, 2),
    "ci_95_high": round(high * 100, 2),
    "category_breakdown": cat_stats,
    "details": details_20
}

with open('mmlu_pro_20_sdpo_a_results.json', 'w', encoding='utf-8') as f:
    json.dump(payload_20, f, indent=2, ensure_ascii=False)

print(f"SDPO-A MMLU-Pro 20Q Score: {pass_20} / 20 ({100.0 * pass_20 / 20:.1f}%)")
print(f"95% Wilson CI: [{low*100:.1f}%, {high*100:.1f}%]")
print("Details (first 20):")
for r in details_20:
    print(f"  [Q{r['index']:02d}] GT: {r['ground_truth']} | Pred: {r['predicted']} | {r['verdict']} ({r['elapsed_seconds']}s)")
