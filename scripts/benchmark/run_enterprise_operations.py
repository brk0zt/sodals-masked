import json
import subprocess
import time
import sys
from pathlib import Path

# Configuration
MODEL_PATH = "C:/Users/Burak/Desktop/sodals_core/models/model.gguf"
ASSETOPS_DATASET = "C:/Users/Burak/Desktop/sodals_core/assetops_test_dataset.json"
ITBENCH_DATASET = "C:/Users/Burak/Desktop/sodals_core/itbench_test_dataset.json"
ASSETOPS_OUTPUT = "C:/Users/Burak/Desktop/sodals_core/results_assetops_candle.json"
ITBENCH_OUTPUT = "C:/Users/Burak/Desktop/sodals_core/results_itbench_candle.json"
REPORT_FILE = "C:/Users/Burak/Desktop/sodals_core/ENTERPRISE_OPERATIONS_REPORT.txt"

def run_cmd(name, cmd):
    print(f"\n" + "="*60)
    print(f"Running {name}")
    print(f"="*60)
    print(f"Command: {' '.join(cmd)}")
    
    start = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True, encoding='utf-8', errors='replace')
    end = time.time()
    
    print(result.stdout)
    if result.stderr:
        print("Error/Stderr:")
        print(result.stderr)
        
    print(f"Finished {name} in {end - start:.2f}s with Exit Code: {result.returncode}")
    return result.returncode == 0

def main():
    print("Starting Enterprise Operations Benchmark (AssetOpsBench and ITBench)...")
    total_start = time.time()
    
    python_exe = sys.executable
    
    # 1. Run AssetOpsBench
    assetops_script = str(Path(__file__).parent / "assetops_candle_runner.py")
    assetops_args = [
        python_exe, assetops_script,
        "--model-path", MODEL_PATH,
        "--dataset", ASSETOPS_DATASET,
        "--output", ASSETOPS_OUTPUT
    ]
    assetops_success = run_cmd("AssetOpsBench", assetops_args)
    
    # 2. Run ITBench
    itbench_script = str(Path(__file__).parent / "itbench_candle_runner.py")
    itbench_args = [
        python_exe, itbench_script,
        "--model-path", MODEL_PATH,
        "--dataset", ITBENCH_DATASET,
        "--output", ITBENCH_OUTPUT
    ]
    itbench_success = run_cmd("ITBench", itbench_args)
    
    total_end = time.time()
    
    # Compile report
    report = f"""================================================================================
                  OFFICIAL ENTERPRISE OPERATIONS COMPLIANCE REPORT
================================================================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model Path: {MODEL_PATH}
Total Execution Time: {total_end - total_start:.2f}s

--------------------------- EVALUATION PERFORMANCE -----------------------------
"""
    
    if Path(ASSETOPS_OUTPUT).exists():
        try:
            with open(ASSETOPS_OUTPUT, 'r') as f:
                res = json.load(f)
            report += f"\n=== AssetOpsBench ===\n"
            report += f"Status: COMPLETED\n"
            report += f"Tasks: {res.get('num_completed', 0)}/{res.get('num_tasks', 0)}\n"
            report += f"Completion Rate: {res.get('completion_rate', 0.0)*100:.2f}%\n"
        except Exception as e:
            report += f"\n=== AssetOpsBench ===\nStatus: ERROR reading results ({e})\n"
    else:
        report += f"\n=== AssetOpsBench ===\nStatus: NOT COMPLETED (Output file missing)\n"
        
    if Path(ITBENCH_OUTPUT).exists():
        try:
            with open(ITBENCH_OUTPUT, 'r') as f:
                res = json.load(f)
            report += f"\n=== ITBench ===\n"
            report += f"Status: COMPLETED\n"
            report += f"Tasks: {res.get('num_completed', 0)}/{res.get('num_tasks', 0)}\n"
            report += f"Mean Score: {res.get('mean_score', 0.0):.4f}\n"
        except Exception as e:
            report += f"\n=== ITBench ===\nStatus: ERROR reading results ({e})\n"
    else:
        report += f"\n=== ITBench ===\nStatus: NOT COMPLETED (Output file missing)\n"
        
    report += f"""
================================================================================
                         END OF COMPLIANCE REPORT
================================================================================
"""
    print(report)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Compliance report saved to: {REPORT_FILE}")

if __name__ == "__main__":
    main()
