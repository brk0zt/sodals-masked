import os
import json
import re
from pathlib import Path

def audit_sre(dataset_path: Path):
    print("Auditing SRE Dataset for Contamination...")
    leaks = 0
    scenarios = list(dataset_path.glob("scenario_*"))
    for sc_dir in scenarios:
        prompt_file = sc_dir / "prompt.txt"
        gt_file = sc_dir / "ground_truth.json"
        if prompt_file.exists() and gt_file.exists():
            prompt = prompt_file.read_text(encoding="utf-8")
            with open(gt_file, "r") as f:
                gt = json.load(f)
            
            faulty_service = gt.get("faulty_service", "")
            # Check if prompt contains the explicit answer
            if faulty_service and faulty_service.lower() in prompt.lower():
                # We expect the service name to be in alerts or logs, but we must make sure it doesn't say "The answer is X"
                if re.search(rf"answer is\s+[\"']?{re.escape(faulty_service)}[\"']?", prompt, re.IGNORECASE):
                    print(f"  [LEAK] SRE scenario {sc_dir.name} has explicit answer leak in prompt!")
                    leaks += 1
    print(f"SRE Audit completed. Leaks found: {leaks}\n")
    return leaks

def audit_finops(dataset_path: Path):
    print("Auditing FinOps Dataset for Contamination...")
    leaks = 0
    scenarios = list(dataset_path.glob("scenario_*"))
    for sc_dir in scenarios:
        gt_file = sc_dir / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file, "r") as f:
                gt = json.load(f)
            
            resources = [r.get("resource", {}).get("name") for r in gt.get("resource", [])]
            # Verify if prompts are generated dynamically without hardcoded answers
            # FinOps prompts are generated dynamically by finops_runner.py.
            # We must make sure the telemetry CSV data doesn't contain ground truth markers.
            csv_file = sc_dir / "data.csv"
            if csv_file.exists():
                content = csv_file.read_text(encoding="utf-8")
                for res in resources:
                    if f"{res}_faulty" in content or f"{res}_ground_truth" in content:
                        print(f"  [LEAK] FinOps CSV {sc_dir.name} contains target label marker for {res}!")
                        leaks += 1
    print(f"FinOps Audit completed. Leaks found: {leaks}\n")
    return leaks

def audit_failuresensoriq(dataset_file: Path):
    print("Auditing FailureSensorIQ Dataset for Contamination...")
    leaks = 0
    if dataset_file.exists():
        with open(dataset_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    question = data.get("question", "")
                    answer = data.get("answer", "")
                    # Check if the correct answer option is leaked inside the question text
                    if answer and f"correct option is {answer}" in question.lower():
                        print(f"  [LEAK] FailureSensorIQ line {i+1} has explicit answer option leak!")
                        leaks += 1
                except Exception as e:
                    pass
    print(f"FailureSensorIQ Audit completed. Leaks found: {leaks}\n")
    return leaks

def main():
    root_dir = Path(__file__).parent.parent.parent
    sre_dir = root_dir / "sre"
    finops_dir = root_dir / "finops"
    fsiq_file = root_dir / "FailureSensorIQ" / "failuresensoriq_standard" / "all.jsonl"

    print("==================================================")
    print("          SODALS BENCHMARK CONTAMINATION AUDIT   ")
    print("==================================================\n")

    sre_leaks = audit_sre(sre_dir) if sre_dir.exists() else 0
    finops_leaks = audit_finops(finops_dir) if finops_dir.exists() else 0
    fsiq_leaks = audit_failuresensoriq(fsiq_file) if fsiq_file.exists() else 0

    total_leaks = sre_leaks + finops_leaks + fsiq_leaks
    print("--------------------------------------------------")
    print(f"Total Contamination Leaks Found: {total_leaks}")
    print("==================================================")

if __name__ == "__main__":
    main()
