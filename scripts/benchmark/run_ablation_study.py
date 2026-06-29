import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.parent.parent
    scripts_dir = root_dir / "scripts" / "benchmark"
    eval_script = str(scripts_dir / "run_all_evaluations.py")
    
    report_file = root_dir / "OFFICIAL_EVALUATION_REPORT.txt"
    report_with = root_dir / "OFFICIAL_EVALUATION_REPORT_WITH_CARTRIDGE.txt"
    report_without = root_dir / "OFFICIAL_EVALUATION_REPORT_WITHOUT_CARTRIDGE.txt"
    
    # Run 1: WITH CARTRIDGE
    print("====================================================")
    print("STEP 1: Running evaluations WITH QMEM Cartridge...")
    print("====================================================")
    # Remove old results
    for f in ["results_sre_master.json", "results_finops_master.json", "results_fsiq_master.json", "OFFICIAL_EVALUATION_REPORT.txt"]:
        p = root_dir / f
        if p.exists():
            p.unlink()
            
    env = os.environ.copy()
    if "NO_CARTRIDGE" in env:
        del env["NO_CARTRIDGE"]
        
    p = subprocess.run([sys.executable, eval_script], env=env, cwd=str(root_dir))
    if p.returncode != 0:
        print("Error: evaluations with cartridge failed!")
        sys.exit(p.returncode)
        
    if report_file.exists():
        shutil.copy(str(report_file), str(report_with))
        print(f"Copied report to {report_with.name}")
    else:
        print("Warning: OFFICIAL_EVALUATION_REPORT.txt not found!")
        
    # Run 2: WITHOUT CARTRIDGE
    print("====================================================")
    print("STEP 2: Running evaluations WITHOUT QMEM Cartridge...")
    print("====================================================")
    # Remove old results
    for f in ["results_sre_master.json", "results_finops_master.json", "results_fsiq_master.json", "OFFICIAL_EVALUATION_REPORT.txt"]:
        p = root_dir / f
        if p.exists():
            p.unlink()
            
    env = os.environ.copy()
    env["NO_CARTRIDGE"] = "1"
    
    p = subprocess.run([sys.executable, eval_script], env=env, cwd=str(root_dir))
    if p.returncode != 0:
        print("Error: evaluations without cartridge failed!")
        sys.exit(p.returncode)
        
    if report_file.exists():
        shutil.copy(str(report_file), str(report_without))
        print(f"Copied report to {report_without.name}")
    else:
        print("Warning: OFFICIAL_EVALUATION_REPORT.txt not found!")
        
    # Compile comparison
    print("====================================================")
    print("ABLATION STUDY COMPLETED!")
    print("====================================================")
    
    # Read both and print a nice summary
    def parse_report(path):
        summary = {}
        if not path.exists():
            return {"error": "File not found"}
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Parse SRE
        try:
            if "=== SRE ===" in content:
                sre_part = content.split("=== SRE ===")[1].split("===")[0]
                for line in sre_part.strip().split("\n"):
                    if "Accuracy:" in line:
                        summary["SRE_Acc"] = line.split("Accuracy:")[1].strip()
                    elif "Tasks Completed:" in line:
                        summary["SRE_Tasks"] = line.split("Tasks Completed:")[1].strip()
        except:
            pass
        # Parse FinOps
        try:
            if "=== FinOps ===" in content:
                finops_part = content.split("=== FinOps ===")[1].split("===")[0]
                for line in finops_part.strip().split("\n"):
                    if "Accuracy:" in line:
                        summary["FinOps_Acc"] = line.split("Accuracy:")[1].strip()
                    elif "Tasks Completed:" in line:
                        summary["FinOps_Tasks"] = line.split("Tasks Completed:")[1].strip()
        except:
            pass
        # Parse FailureSensorIQ
        try:
            if "=== FailureSensorIQ ===" in content:
                fsiq_part = content.split("=== FailureSensorIQ ===")[1].split("===")[0]
                for line in fsiq_part.strip().split("\n"):
                    if "Accuracy:" in line:
                        summary["FSIQ_Acc"] = line.split("Accuracy:")[1].strip()
                    elif "Correct Answers:" in line:
                        summary["FSIQ_Correct"] = line.split("Correct Answers:")[1].strip()
        except:
            pass
        return summary
        
    sum_with = parse_report(report_with)
    sum_without = parse_report(report_without)
    
    comparison = f"""
================================================================================
                    SODALS COGNITIVE CARTRIDGE ABLATION STUDY
================================================================================
Benchmark         | WITH QMEM Cartridge            | WITHOUT QMEM Cartridge
------------------+--------------------------------+----------------------------
SRE Accuracy      | {sum_with.get('SRE_Acc', 'N/A')} ({sum_with.get('SRE_Tasks', 'N/A')}) | {sum_without.get('SRE_Acc', 'N/A')} ({sum_without.get('SRE_Tasks', 'N/A')})
FinOps Accuracy   | {sum_with.get('FinOps_Acc', 'N/A')} ({sum_with.get('FinOps_Tasks', 'N/A')}) | {sum_without.get('FinOps_Acc', 'N/A')} ({sum_without.get('FinOps_Tasks', 'N/A')})
FailureSensorIQ   | {sum_with.get('FSIQ_Acc', 'N/A')} ({sum_with.get('FSIQ_Correct', 'N/A')}) | {sum_without.get('FSIQ_Acc', 'N/A')} ({sum_without.get('FSIQ_Correct', 'N/A')})
================================================================================
"""
    print(comparison)
    with open(root_dir / "ABLATION_STUDY_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(comparison)
        
if __name__ == "__main__":
    main()
