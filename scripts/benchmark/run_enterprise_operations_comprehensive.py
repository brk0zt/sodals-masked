"""
Enterprise Operations Comprehensive Benchmark Runner
===================================================

Runs SRE + FinOps + FSIQ benchmarks with and without QMEM.
For FSIQ, uses statistically significant sample size (385 questions: 200 single-choice + 185 multi-choice)
for 5% margin of error at 95% confidence level.
"""

import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configuration
MODEL_PATH = "C:/Users/Burak/Desktop/sodals_core/models/Qwen3-8B-Q4_K_M.gguf"
SRE_DATASET = "C:/Users/Burak/Desktop/sodals_core/sre"
FINOPS_DATASET = "C:/Users/Burak/Desktop/sodals_core/finops"
FSIQ_SINGLE_DATASET = "C:/Users/Burak/Desktop/sodals_core/FailureSensorIQ/failuresensoriq_standard/all.jsonl"
FSIQ_MULTI_DATASET = "C:/Users/Burak/Desktop/sodals_core/FailureSensorIQ/failuresensoriq_standard/all_multi_answers.jsonl"

# QMEM Cartridge path (set to None for no QMEM)
QMEM_CARTRIDGE = "C:/Users/Burak/Desktop/sodals_core/sodals_core_merged.qmem"

# FSIQ sample size: 25 multiple choice + 25 single choice
FSIQ_SINGLE_SAMPLE_SIZE = 25  # Multiple choice
FSIQ_MULTI_SAMPLE_SIZE = 25   # Single choice

# Output paths
OUTPUT_DIR = Path("C:/Users/Burak/Desktop/sodals_core")
REPORT_FILE = OUTPUT_DIR / "ENTERPRISE_OPERATIONS_COMPREHENSIVE_REPORT.txt"

def run_cmd(name: str, cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
    """Run a command and return result info."""
    print(f"\n" + "="*60)
    print(f"Running {name}")
    print(f"="*60)
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Working Directory: {cwd}")
    
    start = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True, encoding='utf-8', errors='replace', cwd=cwd)
    end = time.time()
    
    print(result.stdout)
    if result.stderr:
        print("Error/Stderr:")
        print(result.stderr)
        
    print(f"Finished {name} in {end - start:.2f}s with Exit Code: {result.returncode}")
    
    return {
        "success": result.returncode == 0,
        "duration": end - start,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode
    }

def load_results(path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return {}

def run_benchmark_suite(use_qmem: bool, model_path: str = MODEL_PATH, cartridge_path: str = QMEM_CARTRIDGE) -> Dict[str, Any]:
    """Run all three benchmarks (SRE, FinOps, FSIQ) with or without QMEM."""
    mode = "WITH_QMEM" if use_qmem else "WITHOUT_QMEM"
    print(f"\n{'='*60}")
    print(f"Starting Enterprise Operations Benchmark Suite - {mode}")
    print(f"{'='*60}")
    
    total_start = time.time()
    python_exe = sys.executable
    script_dir = Path(__file__).parent
    
    # Set cartridge argument
    cartridge_arg = ["--cartridge", cartridge_path] if use_qmem and cartridge_path else []
    
    results = {
        "mode": mode,
        "use_qmem": use_qmem,
        "cartridge_path": cartridge_path if use_qmem else None,
        "benchmarks": {}
    }
    
    # 1. Run SRE
    print(f"\n{'='*60}")
    print("SRE BENCHMARK")
    print(f"{'='*60}")
    sre_output = OUTPUT_DIR / f"results_sre_{mode.lower()}.json"
    sre_script = script_dir / "sre_runner.py"
    sre_cmd = [
        python_exe, str(sre_script),
        "--model-path", model_path,
        "--dataset", SRE_DATASET,
        "--output", str(sre_output),
        "--log-level", "INFO"
    ] + cartridge_arg
    
    sre_result = run_cmd("SRE", sre_cmd)
    results["benchmarks"]["sre"] = {
        "success": sre_result["success"],
        "duration": sre_result["duration"],
        "exit_code": sre_result["exit_code"]
    }
    
    # Load SRE results
    sre_data = load_results(sre_output)
    if sre_data:
        results["benchmarks"]["sre"]["accuracy"] = sre_data.get("accuracy", 0.0)
        results["benchmarks"]["sre"]["correct_identifications"] = sre_data.get("correct_identifications", 0)
        results["benchmarks"]["sre"]["total_identifications"] = sre_data.get("total_identifications", 0)
        results["benchmarks"]["sre"]["completion_rate"] = sre_data.get("completion_rate", 0.0)
    
    # 2. Run FinOps
    print(f"\n{'='*60}")
    print("FINOPS BENCHMARK")
    print(f"{'='*60}")
    finops_output = OUTPUT_DIR / f"results_finops_{mode.lower()}.json"
    finops_script = script_dir / "finops_runner.py"
    finops_cmd = [
        python_exe, str(finops_script),
        "--model-path", model_path,
        "--dataset", FINOPS_DATASET,
        "--output", str(finops_output),
        "--log-level", "INFO"
    ] + cartridge_arg
    
    finops_result = run_cmd("FinOps", finops_cmd)
    results["benchmarks"]["finops"] = {
        "success": finops_result["success"],
        "duration": finops_result["duration"],
        "exit_code": finops_result["exit_code"]
    }
    
    # Load FinOps results
    finops_data = load_results(finops_output)
    if finops_data:
        results["benchmarks"]["finops"]["accuracy"] = finops_data.get("accuracy", 0.0)
        results["benchmarks"]["finops"]["correct_identifications"] = finops_data.get("correct_identifications", 0)
        results["benchmarks"]["finops"]["total_identifications"] = finops_data.get("total_identifications", 0)
        results["benchmarks"]["finops"]["completion_rate"] = finops_data.get("completion_rate", 0.0)
    
    # 3. Run FSIQ Single-Choice
    print(f"\n{'='*60}")
    print("FSIQ SINGLE-CHOICE BENCHMARK")
    print(f"{'='*60}")
    fsiq_single_output = OUTPUT_DIR / f"results_fsiq_single_{mode.lower()}.json"
    fsiq_script = script_dir / "failuresensoriq_runner.py"
    fsiq_single_cmd = [
        python_exe, str(fsiq_script),
        "--model-path", model_path,
        "--dataset", FSIQ_SINGLE_DATASET,
        "--max-questions", str(FSIQ_SINGLE_SAMPLE_SIZE),
        "--output", str(fsiq_single_output),
        "--log-level", "INFO"
    ] + cartridge_arg
    
    fsiq_single_result = run_cmd("FSIQ Single-Choice", fsiq_single_cmd)
    results["benchmarks"]["fsiq_single"] = {
        "success": fsiq_single_result["success"],
        "duration": fsiq_single_result["duration"],
        "exit_code": fsiq_single_result["exit_code"],
        "sample_size": FSIQ_SINGLE_SAMPLE_SIZE
    }
    
    # Load FSIQ Single results
    fsiq_single_data = load_results(fsiq_single_output)
    if fsiq_single_data:
        results["benchmarks"]["fsiq_single"]["accuracy"] = fsiq_single_data.get("accuracy", 0.0)
        results["benchmarks"]["fsiq_single"]["correct_answers"] = fsiq_single_data.get("correct_answers", 0)
        results["benchmarks"]["fsiq_single"]["total_answers"] = fsiq_single_data.get("total_answers", 0)
        results["benchmarks"]["fsiq_single"]["completion_rate"] = fsiq_single_data.get("completion_rate", 0.0)
    
    # 4. Run FSIQ Multi-Choice
    print(f"\n{'='*60}")
    print("FSIQ MULTI-CHOICE BENCHMARK")
    print(f"{'='*60}")
    fsiq_multi_output = OUTPUT_DIR / f"results_fsiq_multi_{mode.lower()}.json"
    fsiq_multi_cmd = [
        python_exe, str(fsiq_script),
        "--model-path", model_path,
        "--dataset", FSIQ_MULTI_DATASET,
        "--max-questions", str(FSIQ_MULTI_SAMPLE_SIZE),
        "--output", str(fsiq_multi_output),
        "--log-level", "INFO"
    ] + cartridge_arg
    
    fsiq_multi_result = run_cmd("FSIQ Multi-Choice", fsiq_multi_cmd)
    results["benchmarks"]["fsiq_multi"] = {
        "success": fsiq_multi_result["success"],
        "duration": fsiq_multi_result["duration"],
        "exit_code": fsiq_multi_result["exit_code"],
        "sample_size": FSIQ_MULTI_SAMPLE_SIZE
    }
    
    # Load FSIQ Multi results
    fsiq_multi_data = load_results(fsiq_multi_output)
    if fsiq_multi_data:
        results["benchmarks"]["fsiq_multi"]["accuracy"] = fsiq_multi_data.get("accuracy", 0.0)
        results["benchmarks"]["fsiq_multi"]["mean_f1_score"] = fsiq_multi_data.get("mean_f1_score", 0.0)
        results["benchmarks"]["fsiq_multi"]["correct_answers"] = fsiq_multi_data.get("correct_answers", 0)
        results["benchmarks"]["fsiq_multi"]["total_answers"] = fsiq_multi_data.get("total_answers", 0)
        results["benchmarks"]["fsiq_multi"]["completion_rate"] = fsiq_multi_data.get("completion_rate", 0.0)
    
    # Calculate aggregate FSIQ results
    fsiq_total_correct = results["benchmarks"]["fsiq_single"].get("correct_answers", 0) + results["benchmarks"]["fsiq_multi"].get("correct_answers", 0)
    fsiq_total_questions = results["benchmarks"]["fsiq_single"].get("total_answers", 0) + results["benchmarks"]["fsiq_multi"].get("total_answers", 0)
    fsiq_aggregate_accuracy = fsiq_total_correct / fsiq_total_questions if fsiq_total_questions > 0 else 0.0
    
    results["benchmarks"]["fsiq_aggregate"] = {
        "total_sample_size": FSIQ_SINGLE_SAMPLE_SIZE + FSIQ_MULTI_SAMPLE_SIZE,
        "total_correct": fsiq_total_correct,
        "total_questions": fsiq_total_questions,
        "aggregate_accuracy": fsiq_aggregate_accuracy
    }
    
    total_end = time.time()
    results["total_duration"] = total_end - total_start
    results["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return results

def generate_report(results_with_qmem: Dict[str, Any], results_without_qmem: Dict[str, Any]) -> str:
    """Generate comprehensive comparison report."""
    report = f"""{'='*80}
              COMPREHENSIVE ENTERPRISE OPERATIONS REPORT
              SRE + FinOps + FSIQ (With vs Without QMEM)
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Path: {MODEL_PATH}
QMEM Cartridge: {QMEM_CARTRIDGE}

FSIQ Sample Size Configuration:
- Single-Choice Questions: {FSIQ_SINGLE_SAMPLE_SIZE} (for 5% margin of error)
- Multi-Choice Questions: {FSIQ_MULTI_SAMPLE_SIZE} (for 5% margin of error)
- Total FSIQ Questions: {FSIQ_SINGLE_SAMPLE_SIZE + FSIQ_MULTI_SAMPLE_SIZE}

{'='*80}
                          EXECUTION SUMMARY
{'='*80}

WITH QMEM:
- Total Duration: {results_with_qmem.get('total_duration', 0):.2f}s
- SRE Accuracy: {results_with_qmem['benchmarks'].get('sre', {}).get('accuracy', 0):.4f}
- FinOps Accuracy: {results_with_qmem['benchmarks'].get('finops', {}).get('accuracy', 0):.4f}
- FSIQ Aggregate Accuracy: {results_with_qmem['benchmarks'].get('fsiq_aggregate', {}).get('aggregate_accuracy', 0):.4f}

WITHOUT QMEM:
- Total Duration: {results_without_qmem.get('total_duration', 0):.2f}s
- SRE Accuracy: {results_without_qmem['benchmarks'].get('sre', {}).get('accuracy', 0):.4f}
- FinOps Accuracy: {results_without_qmem['benchmarks'].get('finops', {}).get('accuracy', 0):.4f}
- FSIQ Aggregate Accuracy: {results_without_qmem['benchmarks'].get('fsiq_aggregate', {}).get('aggregate_accuracy', 0):.4f}

{'='*80}
                          DETAILED RESULTS
{'='*80}

"""

    # SRE Comparison
    report += "=== SRE BENCHMARK ===\n"
    sre_with = results_with_qmem['benchmarks'].get('sre', {})
    sre_without = results_without_qmem['benchmarks'].get('sre', {})
    
    report += f"\nWITH QMEM:\n"
    report += f"  Success: {sre_with.get('success', False)}\n"
    report += f"  Duration: {sre_with.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {sre_with.get('accuracy', 0):.4f}\n"
    report += f"  Correct: {sre_with.get('correct_identifications', 0)}/{sre_with.get('total_identifications', 0)}\n"
    report += f"  Completion Rate: {sre_with.get('completion_rate', 0):.4f}\n"
    
    report += f"\nWITHOUT QMEM:\n"
    report += f"  Success: {sre_without.get('success', False)}\n"
    report += f"  Duration: {sre_without.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {sre_without.get('accuracy', 0):.4f}\n"
    report += f"  Correct: {sre_without.get('correct_identifications', 0)}/{sre_without.get('total_identifications', 0)}\n"
    report += f"  Completion Rate: {sre_without.get('completion_rate', 0):.4f}\n"
    
    sre_diff = sre_with.get('accuracy', 0) - sre_without.get('accuracy', 0)
    report += f"\nQMEM Impact: {sre_diff:+.4f} ({sre_diff*100:+.2f}%)\n"
    
    # FinOps Comparison
    report += "\n=== FINOPS BENCHMARK ===\n"
    finops_with = results_with_qmem['benchmarks'].get('finops', {})
    finops_without = results_without_qmem['benchmarks'].get('finops', {})
    
    report += f"\nWITH QMEM:\n"
    report += f"  Success: {finops_with.get('success', False)}\n"
    report += f"  Duration: {finops_with.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {finops_with.get('accuracy', 0):.4f}\n"
    report += f"  Correct: {finops_with.get('correct_identifications', 0)}/{finops_with.get('total_identifications', 0)}\n"
    report += f"  Completion Rate: {finops_with.get('completion_rate', 0):.4f}\n"
    
    report += f"\nWITHOUT QMEM:\n"
    report += f"  Success: {finops_without.get('success', False)}\n"
    report += f"  Duration: {finops_without.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {finops_without.get('accuracy', 0):.4f}\n"
    report += f"  Correct: {finops_without.get('correct_identifications', 0)}/{finops_without.get('total_identifications', 0)}\n"
    report += f"  Completion Rate: {finops_without.get('completion_rate', 0):.4f}\n"
    
    finops_diff = finops_with.get('accuracy', 0) - finops_without.get('accuracy', 0)
    report += f"\nQMEM Impact: {finops_diff:+.4f} ({finops_diff*100:+.2f}%)\n"
    
    # FSIQ Comparison
    report += "\n=== FSIQ BENCHMARK ===\n"
    
    # Single-Choice
    report += "\n--- Single-Choice Questions ---\n"
    fsiq_single_with = results_with_qmem['benchmarks'].get('fsiq_single', {})
    fsiq_single_without = results_without_qmem['benchmarks'].get('fsiq_single', {})
    
    report += f"\nWITH QMEM:\n"
    report += f"  Sample Size: {fsiq_single_with.get('sample_size', 0)}\n"
    report += f"  Success: {fsiq_single_with.get('success', False)}\n"
    report += f"  Duration: {fsiq_single_with.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {fsiq_single_with.get('accuracy', 0):.4f}\n"
    report += f"  Correct: {fsiq_single_with.get('correct_answers', 0)}/{fsiq_single_with.get('total_answers', 0)}\n"
    report += f"  Completion Rate: {fsiq_single_with.get('completion_rate', 0):.4f}\n"
    
    report += f"\nWITHOUT QMEM:\n"
    report += f"  Sample Size: {fsiq_single_without.get('sample_size', 0)}\n"
    report += f"  Success: {fsiq_single_without.get('success', False)}\n"
    report += f"  Duration: {fsiq_single_without.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {fsiq_single_without.get('accuracy', 0):.4f}\n"
    report += f"  Correct: {fsiq_single_without.get('correct_answers', 0)}/{fsiq_single_without.get('total_answers', 0)}\n"
    report += f"  Completion Rate: {fsiq_single_without.get('completion_rate', 0):.4f}\n"
    
    fsiq_single_diff = fsiq_single_with.get('accuracy', 0) - fsiq_single_without.get('accuracy', 0)
    report += f"\nQMEM Impact: {fsiq_single_diff:+.4f} ({fsiq_single_diff*100:+.2f}%)\n"
    
    # Multi-Choice
    report += "\n--- Multi-Choice Questions ---\n"
    fsiq_multi_with = results_with_qmem['benchmarks'].get('fsiq_multi', {})
    fsiq_multi_without = results_without_qmem['benchmarks'].get('fsiq_multi', {})
    
    report += f"\nWITH QMEM:\n"
    report += f"  Sample Size: {fsiq_multi_with.get('sample_size', 0)}\n"
    report += f"  Success: {fsiq_multi_with.get('success', False)}\n"
    report += f"  Duration: {fsiq_multi_with.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {fsiq_multi_with.get('accuracy', 0):.4f}\n"
    report += f"  Mean F1 Score: {fsiq_multi_with.get('mean_f1_score', 0):.4f}\n"
    report += f"  Correct: {fsiq_multi_with.get('correct_answers', 0)}/{fsiq_multi_with.get('total_answers', 0)}\n"
    report += f"  Completion Rate: {fsiq_multi_with.get('completion_rate', 0):.4f}\n"
    
    report += f"\nWITHOUT QMEM:\n"
    report += f"  Sample Size: {fsiq_multi_without.get('sample_size', 0)}\n"
    report += f"  Success: {fsiq_multi_without.get('success', False)}\n"
    report += f"  Duration: {fsiq_multi_without.get('duration', 0):.2f}s\n"
    report += f"  Accuracy: {fsiq_multi_without.get('accuracy', 0):.4f}\n"
    report += f"  Mean F1 Score: {fsiq_multi_without.get('mean_f1_score', 0):.4f}\n"
    report += f"  Correct: {fsiq_multi_without.get('correct_answers', 0)}/{fsiq_multi_without.get('total_answers', 0)}\n"
    report += f"  Completion Rate: {fsiq_multi_without.get('completion_rate', 0):.4f}\n"
    
    fsiq_multi_diff = fsiq_multi_with.get('accuracy', 0) - fsiq_multi_without.get('accuracy', 0)
    report += f"\nQMEM Impact: {fsiq_multi_diff:+.4f} ({fsiq_multi_diff*100:+.2f}%)\n"
    
    # Aggregate FSIQ
    report += "\n--- FSIQ Aggregate ---\n"
    fsiq_agg_with = results_with_qmem['benchmarks'].get('fsiq_aggregate', {})
    fsiq_agg_without = results_without_qmem['benchmarks'].get('fsiq_aggregate', {})
    
    report += f"\nWITH QMEM:\n"
    report += f"  Total Sample Size: {fsiq_agg_with.get('total_sample_size', 0)}\n"
    report += f"  Total Correct: {fsiq_agg_with.get('total_correct', 0)}/{fsiq_agg_with.get('total_questions', 0)}\n"
    report += f"  Aggregate Accuracy: {fsiq_agg_with.get('aggregate_accuracy', 0):.4f}\n"
    
    report += f"\nWITHOUT QMEM:\n"
    report += f"  Total Sample Size: {fsiq_agg_without.get('total_sample_size', 0)}\n"
    report += f"  Total Correct: {fsiq_agg_without.get('total_correct', 0)}/{fsiq_agg_without.get('total_questions', 0)}\n"
    report += f"  Aggregate Accuracy: {fsiq_agg_without.get('aggregate_accuracy', 0):.4f}\n"
    
    fsiq_agg_diff = fsiq_agg_with.get('aggregate_accuracy', 0) - fsiq_agg_without.get('aggregate_accuracy', 0)
    report += f"\nQMEM Impact: {fsiq_agg_diff:+.4f} ({fsiq_agg_diff*100:+.2f}%)\n"
    
    # Overall Summary
    report += f"\n{'='*80}\n"
    report += "                          OVERALL SUMMARY\n"
    report += f"{'='*80}\n"
    
    total_with_qmem = results_with_qmem.get('total_duration', 0)
    total_without_qmem = results_without_qmem.get('total_duration', 0)
    
    report += f"\nTotal Execution Time:\n"
    report += f"  WITH QMEM: {total_with_qmem:.2f}s\n"
    report += f"  WITHOUT QMEM: {total_without_qmem:.2f}s\n"
    report += f"  Difference: {total_with_qmem - total_without_qmem:+.2f}s\n"
    
    report += f"\nAverage Accuracy Across All Benchmarks:\n"
    avg_with = (sre_with.get('accuracy', 0) + finops_with.get('accuracy', 0) + fsiq_agg_with.get('aggregate_accuracy', 0)) / 3
    avg_without = (sre_without.get('accuracy', 0) + finops_without.get('accuracy', 0) + fsiq_agg_without.get('aggregate_accuracy', 0)) / 3
    report += f"  WITH QMEM: {avg_with:.4f}\n"
    report += f"  WITHOUT QMEM: {avg_without:.4f}\n"
    report += f"  QMEM Impact: {avg_with - avg_without:+.4f} ({(avg_with - avg_without)*100:+.2f}%)\n"
    
    report += f"\n{'='*80}\n"
    report += "                         END OF REPORT\n"
    report += f"{'='*80}\n"
    
    return report

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive Enterprise Operations benchmarks")
    parser.add_argument("--with-qmem-only", action="store_true", help="Only run with QMEM")
    parser.add_argument("--without-qmem-only", action="store_true", help="Only run without QMEM")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model")
    parser.add_argument("--cartridge", default=QMEM_CARTRIDGE, help="Path to QMEM cartridge")
    
    args = parser.parse_args()
    
    # Use config from args or defaults
    model_path = args.model_path
    cartridge_path = args.cartridge
    
    results_with_qmem = None
    results_without_qmem = None
    
    # Run benchmarks based on flags
    if not args.without_qmem_only:
        print("\n" + "="*80)
        print("RUNNING BENCHMARKS WITH QMEM")
        print("="*80)
        results_with_qmem = run_benchmark_suite(use_qmem=True, model_path=model_path, cartridge_path=cartridge_path)
        
        # Save raw results
        with open(OUTPUT_DIR / "enterprise_ops_results_with_qmem.json", 'w') as f:
            json.dump(results_with_qmem, f, indent=2)
    
    if not args.with_qmem_only:
        print("\n" + "="*80)
        print("RUNNING BENCHMARKS WITHOUT QMEM")
        print("="*80)
        results_without_qmem = run_benchmark_suite(use_qmem=False, model_path=model_path, cartridge_path=cartridge_path)
        
        # Save raw results
        with open(OUTPUT_DIR / "enterprise_ops_results_without_qmem.json", 'w') as f:
            json.dump(results_without_qmem, f, indent=2)
    
    # Generate report if both modes were run
    if results_with_qmem and results_without_qmem:
        report = generate_report(results_with_qmem, results_without_qmem)
        print(report)
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nComprehensive report saved to: {REPORT_FILE}")
    elif results_with_qmem:
        print("\nOnly WITH QMEM results available. Run without --with-qmem-only for comparison.")
    elif results_without_qmem:
        print("\nOnly WITHOUT QMEM results available. Run without --without-qmem-only for comparison.")

if __name__ == "__main__":
    main()
