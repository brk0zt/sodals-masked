import json

with open('hle_50_sdpo_a_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total Evaluated: {d['total_evaluated']} / {d['total_target']}")
print(f"Pass Count: {d['pass_count']}")
print("-" * 80)
for r in d['details']:
    gt = str(r['ground_truth'])
    pred = str(r['predicted'])
    if len(gt) > 25: gt = gt[:22] + "..."
    if len(pred) > 25: pred = pred[:22] + "..."
    print(f"[{r['index']:02d}] [{r['category'][:15]:15s}] GT: {gt:25s} | Pred: {pred:25s} | {r['verdict']}")
