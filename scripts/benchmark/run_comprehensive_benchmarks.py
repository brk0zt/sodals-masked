"""
Comprehensive Benchmark Summary
================================

Script to run all benchmarks and generate a comprehensive summary report.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
from benchmark_logger import BenchmarkLogger


def run_benchmark(runner_script: str, args: List[str], benchmark_name: str) -> Dict[str, Any]:
    """
    Run a benchmark and return results.
    
    Args:
        runner_script: Path to the runner script
        args: Arguments to pass to the runner
        benchmark_name: Name of the benchmark
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running {benchmark_name} Benchmark")
    print(f"{'='*60}")
    
    cmd = ["py", runner_script] + args
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {benchmark_name} completed successfully!")
            print(f"   Time: {end_time - start_time:.2f}s")
            
            # Load results from JSON file
            output_file = args[args.index("--output") + 1] if "--output" in args else f"results_{benchmark_name.lower()}.json"
            
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                results["execution_time"] = end_time - start_time
                return results
            else:
                print(f"⚠️  Results file not found: {output_file}")
                return {"error": "Results file not found", "execution_time": end_time - start_time}
        else:
            print(f"❌ {benchmark_name} failed!")
            print(f"   Error: {result.stderr}")
            return {"error": result.stderr, "execution_time": end_time - start_time}
            
    except Exception as e:
        print(f"❌ {benchmark_name} failed with exception: {e}")
        return {"error": str(e), "execution_time": 0}


def main():
    """Main function to run all benchmarks."""
    print("🚀 Starting Comprehensive Benchmark Evaluation")
    print("=" * 80)
    
    # Configuration
    model_path = "models/model.gguf"
    
    # Benchmark configurations
    benchmarks = [
        {
            "name": "FailureSensorIQ",
            "script": "failuresensoriq_runner.py",
            "args": [
                "--model-path", model_path,
                "--dataset", "FailureSensorIQ/failuresensoriq_standard/all.jsonl",
                "--max-questions", "10",  # Sample size for testing
                "--output", "results_failuresensoriq_comprehensive.json"
            ]
        },
        {
            "name": "FinOps",
            "script": "finops_runner.py", 
            "args": [
                "--model-path", model_path,
                "--dataset", "finops",
                "--max-scenarios", "5",  # Sample size for testing
                "--output", "results_finops_comprehensive.json"
            ]
        },
        {
            "name": "SRE",
            "script": "sre_runner.py",
            "args": [
                "--model-path", model_path,
                "--dataset", "sre", 
                "--max-scenarios", "5",  # Sample size for testing
                "--output", "results_sre_comprehensive.json"
            ]
        }
    ]
    
    # Run all benchmarks
    all_results = []
    benchmark_names = []
    
    total_start_time = time.time()
    
    for benchmark in benchmarks:
        results = run_benchmark(
            benchmark["script"], 
            benchmark["args"], 
            benchmark["name"]
        )
        
        if "error" not in results:
            all_results.append(results)
            benchmark_names.append(benchmark["name"])
        else:
            print(f"⚠️  Skipping {benchmark['name']} due to error")
    
    total_end_time = time.time()
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK REPORT")
    print(f"{'='*80}")
    print(f"Total Evaluation Time: {total_end_time - total_start_time:.2f}s")
    print(f"Benchmarks Completed: {len(all_results)}/{len(benchmarks)}")
    
    if all_results:
        # Save comparison report
        logger = BenchmarkLogger()
        comparison_file = logger.save_comparison_report(all_results, benchmark_names)
        print(f"\n📊 Comparison report saved to: {comparison_file}")
        
        # Summary table
        print(f"\n{'BENCHMARK':<20} {'ACCURACY':<12} {'TASKS':<10} {'TIME':<10} {'STATUS':<10}")
        print(f"{'-'*80}")
        
        for results, name in zip(all_results, benchmark_names):
            accuracy = results.get('accuracy', 0)
            tasks = results.get('num_questions', results.get('num_scenarios', 0))
            exec_time = results.get('execution_time', 0)
            status = "✅ PASS" if "error" not in results else "❌ FAIL"
            
            print(f"{name:<20} {accuracy:<12.4f} {tasks:<10} {exec_time:<10.2f} {status:<10}")
        
        # Overall statistics
        avg_accuracy = sum(r.get('accuracy', 0) for r in all_results) / len(all_results)
        total_tasks = sum(r.get('num_questions', r.get('num_scenarios', 0)) for r in all_results)
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"   Total Tasks Evaluated: {total_tasks}")
        print(f"   Total Execution Time: {total_end_time - total_start_time:.2f}s")
        print(f"   Average Time per Task: {(total_end_time - total_start_time)/total_tasks:.4f}s" if total_tasks > 0 else "   Average Time per Task: N/A")
        
        # Model configuration summary
        if all_results:
            config = all_results[0].get('config', {})
            print(f"\n🔧 MODEL CONFIGURATION:")
            print(f"   Model: {config.get('model_path', 'N/A')}")
            print(f"   Type: {all_results[0].get('model_type', 'N/A')}")
            print(f"   Max Tokens: {config.get('max_new_tokens', 'N/A')}")
            
            repro_config = config.get('reproducibility_config', {})
            print(f"   Seed: {repro_config.get('seed', 'N/A')}")
            print(f"   Temperature: {repro_config.get('temperature', 'N/A')}")
            print(f"   Deterministic: {repro_config.get('do_sample', True) == False}")
    
    print(f"\n🎉 Comprehensive benchmark evaluation completed!")
    print(f"📁 All results saved in: benchmark_results/")
    print(f"📋 Detailed logs available in: run_logs/")


if __name__ == "__main__":
    main()
