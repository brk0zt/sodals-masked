import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('hle_50_baseline_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total Evaluated: {d['total_evaluated']} / {d['total_target']}")
print(f"Pass Count: {d['pass_count']}")

for r in d['details']:
    if r['verdict'] == 'PASS':
        print(f"\n--> PASS: Q{r['index']} [{r['category']}]")
        print(f"    Question: {r['question'][:120]}...")
        print(f"    GT:       {r['ground_truth']}")
        print(f"    Pred:     {r['predicted']}")
    if r['index'] == 24:
        print(f"\n--> Q24 (Physics):")
        print(f"    GT:       {r['ground_truth']}")
        print(f"    Pred:     {r['predicted']}")
        print(f"    Verdict:  {r['verdict']}")
