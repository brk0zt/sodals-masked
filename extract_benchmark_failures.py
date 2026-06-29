import json
import re
from pathlib import Path
from collections import defaultdict

failure_modes = set()
asset_names = set()

# Read benchmark
with open("FailureSensorIQ/failuresensoriq_standard/all.jsonl", "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            question = data.get("question", "")
            asset = data.get("asset_name", "")
            
            # Pattern: "if a failure event X occurs"
            if "failure event" in question.lower():
                # Extract failure mode between "failure event" and "occurs"
                match = re.search(r'failure event\s+([^"]+?)\s+occurs', question, re.IGNORECASE)
                if match:
                    fail_mode = match.group(1).strip()
                    failure_modes.add(fail_mode)
                    asset_names.add(asset)
        except:
            pass

print("=" * 70)
print("FAILURE MODES IN FailureSensorIQ BENCHMARK:")
print("=" * 70)
for mode in sorted(failure_modes):
    print(f"  - {mode}")

print(f"\nTotal: {len(failure_modes)} unique failure modes")
print(f"Assets: {', '.join(sorted(asset_names))}")

# Write to file for analysis
with open("benchmark_failure_modes.txt", "w") as f:
    for mode in sorted(failure_modes):
        f.write(f"{mode}\n")
print("\nSaved to: benchmark_failure_modes.txt")
