import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('hle_50_sdpo_a_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total Evaluated: {d['total_evaluated']} / {d['total_target']}")
print(f"Elapsed: {d['elapsed_minutes']} mins")

# Inspect Q24 specifically:
for r in d['details']:
    if r['index'] == 24:
        print("\n--- Question 24 (Physics) ---")
        print("Question:", r['question'])
        print("Ground Truth:", r['ground_truth'])
        print("Predicted:", r['predicted'])
        print("Verdict:", r['verdict'])
        print("Response Tail:\n", r.get('response_preview', ''))

    # Also inspect any close matches:
    gt = str(r['ground_truth'])
    pred = str(r['predicted'])
    if r['index'] in [4, 7, 10, 14, 24]:
        print(f"\n--- Question {r['index']} [{r['category']}] ---")
        print(f"GT:   {gt}")
        print(f"Pred: {pred}")
        print(f"Verdict: {r['verdict']}")
