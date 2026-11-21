import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_lora_4096_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tot = data.get("total_evaluated", 0)
cnt = data.get("pass_count", 0)
acc = data.get("accuracy", 0.0) * 100
low = data.get("ci_low", 0.0) * 100
high = data.get("ci_high", 0.0) * 100
elapsed = data.get("elapsed_minutes", 0.0)

print("=" * 80)
print("FINAL OFFICIAL 500-QUESTION MATH-500 MASTER REPORT (OPTIMAL LoRA alpha=0.25)")
print("=" * 80)
print(f"Total Questions Evaluated : {tot} / 500 (100.0%)")
print(f"Total Passes (PASS)       : {cnt} / {tot}")
print(f"Master Accuracy           : %{acc:.2f}")
print(f"95% Wilson CI             : [%{low:.2f} – %{high:.2f}]")
print(f"Total Evaluation Time     : {elapsed:.1f} minutes")
print("-" * 80)

print("SUBJECT-BY-SUBJECT BREAKDOWN (N=500):")
print(f"{'Subject':25s} | {'Pass / Total':18s} | {'Accuracy':12s}")
print("-" * 60)
for s, stats in data.get("subj_stats", {}).items():
    p = stats["pass"]
    t = stats["total"]
    if t > 0:
        print(f"{s:25s} | {p:3d} / {t:3d}            | %{100*p/t:5.1f}")

level_stats = {}
for r in data.get("details", []):
    lvl = r.get("level", "?")
    if lvl not in level_stats:
        level_stats[lvl] = {"pass": 0, "total": 0}
    level_stats[lvl]["total"] += 1
    if r["verdict"] == "PASS":
        level_stats[lvl]["pass"] += 1

print("\n" + "-" * 60)
print("DIFFICULTY LEVEL BREAKDOWN (Level 1 to Level 5):")
print(f"{'Difficulty Level':25s} | {'Pass / Total':18s} | {'Accuracy':12s}")
print("-" * 60)
for lvl in sorted(level_stats.keys()):
    p = level_stats[lvl]["pass"]
    t = level_stats[lvl]["total"]
    print(f"Level {str(lvl):19s} | {p:3d} / {t:3d}            | %{100*p/t:5.1f}")
