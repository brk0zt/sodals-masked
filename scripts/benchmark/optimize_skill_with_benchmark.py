"""
Skill Optimizer with Benchmark Integration
==========================================

Wrapper script that runs benchmarks and uses results to optimize skills.

Usage:
    python optimize_skill_with_benchmark.py `
        --skill-path sodals_workspace/01_perception/references/vision_cvnn.md `
        --benchmark sre `
        --model-path models/Qwen3-8B-Q4_K_M.gguf `
        --dataset sre `
        --epochs 5 `
        --edit-budget 3
"""

import json
import subprocess
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from skill_optimizer import SkillOptimizer


class BenchmarkOptimizer:
    """
    Runs benchmarks and uses results to optimize skills.
    """
    
    def __init__(
        self,
        skill_path: str,
        benchmark: str,
        model_path: str,
        dataset: str,
        cartridge_path: str = None
    ):
        self.skill_path = Path(skill_path)
        self.benchmark = benchmark
        self.model_path = model_path
        self.dataset = dataset
        self.cartridge_path = cartridge_path
        self.output_dir = Path("benchmark_results")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the specified benchmark and collect results.
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Running {self.benchmark.upper()} benchmark")
        print(f"{'='*60}")
        
        # Determine which runner to use
        if self.benchmark == "sre":
            runner_script = "sre_runner.py"
        elif self.benchmark == "finops":
            runner_script = "finops_runner.py"
        elif self.benchmark == "fsiq":
            runner_script = "fsiq_runner.py"
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / runner_script),
            "--model-path", self.model_path,
            "--dataset", self.dataset,
            "--log-level", "INFO"
        ]
        
        # Add cartridge if provided
        if self.cartridge_path:
            cmd.extend(["--cartridge", self.cartridge_path])
        
        # Add output path
        output_file = self.output_dir / f"results_{self.benchmark}_baseline.json"
        cmd.extend(["--output", str(output_file)])
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run benchmark
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"Benchmark failed with exit code {result.returncode}")
        
        # Load results
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = json.load(f)
            print(f"\nBenchmark results loaded from {output_file}")
            return results
        else:
            raise RuntimeError(f"Output file not found: {output_file}")
    
    def prepare_validation_set(self, benchmark_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert benchmark results to validation set format for skill optimizer.
        
        Args:
            benchmark_results: Results from benchmark runner
            
        Returns:
            List of validation tasks
        """
        validation_tasks = []
        
        # Extract scenarios from benchmark results
        if "scenarios" in benchmark_results:
            scenarios = benchmark_results["scenarios"]
        elif "results" in benchmark_results:
            scenarios = benchmark_results["results"]
        else:
            # Fallback: use the entire results as a single task
            scenarios = [benchmark_results]
        
        for i, scenario in enumerate(scenarios):
            task = {
                "task_id": f"{self.benchmark}_{i}",
                "input": scenario.get("input", {}),
                "expected_output": scenario.get("expected_output", {}),
                "actual_output": scenario.get("actual_output", {}),
                "metrics": scenario.get("metrics", {}),
                "success": scenario.get("success", True)
            }
            validation_tasks.append(task)
        
        print(f"Prepared {len(validation_tasks)} validation tasks")
        return validation_tasks
    
    def save_validation_set(self, validation_tasks: List[Dict[str, Any]], output_path: str = None):
        """
        Save validation set to JSON file for skill optimizer.
        
        Args:
            validation_tasks: List of validation tasks
            output_path: Output file path (default: benchmark_results/{benchmark}_validation.json)
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.benchmark}_validation.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_tasks, f, indent=2)
        
        print(f"Validation set saved to {output_path}")
    
    async def optimize_skill(
        self,
        epochs: int = 5,
        edit_budget: int = 3,
        api_type: str = "ollama",
        model_name: str = "yerli-qwen:latest"
    ):
        """
        Run benchmark and optimize skill based on results.
        
        Args:
            epochs: Number of optimization epochs
            edit_budget: Maximum edits per epoch
            api_type: LLM API type
            model_name: LLM model name
        """
        # Step 1: Run benchmark to get baseline results
        benchmark_results = self.run_benchmark()
        
        # Step 2: Prepare validation set
        validation_tasks = self.prepare_validation_set(benchmark_results)
        
        # Step 3: Save validation set
        validation_path = self.output_dir / f"{self.benchmark}_validation.json"
        self.save_validation_set(validation_tasks, str(validation_path))
        
        # Step 4: Run skill optimizer
        print(f"\n{'='*60}")
        print("Running Skill Optimizer")
        print(f"{'='*60}")
        
        optimizer = SkillOptimizer(
            skill_path=str(self.skill_path),
            validation_set=str(validation_path),
            api_type=api_type,
            model_name=model_name
        )
        
        result = await optimizer.optimize(
            epochs=epochs,
            edit_budget=edit_budget
        )
        
        # Step 5: Save optimized skill
        output_path = self.skill_path.parent / f"{self.skill_path.stem}_optimized.md"
        optimizer.save_best_skill(result, str(output_path))
        
        # Step 6: Print summary
        print(f"\n{'='*60}")
        print("Optimization Summary")
        print(f"{'='*60}")
        print(f"Original skill: {self.skill_path}")
        print(f"Optimized skill: {output_path}")
        print(f"Baseline accuracy: {benchmark_results.get('accuracy', 'N/A')}")
        print(f"Best score: {result.best_score:.4f}")
        print(f"Improvement: {result.best_score - benchmark_results.get('accuracy', 0):.4f}")
        print(f"Edits applied: {len(result.edits_applied)}")
        print(f"Edits rejected: {len(result.rejected_edits)}")
        
        return result


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize skill using benchmark results")
    parser.add_argument("--skill-path", required=True, help="Path to skill markdown file")
    parser.add_argument("--benchmark", required=True, choices=["sre", "finops", "fsiq"], help="Benchmark to run")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--cartridge", help="Path to QMEM cartridge file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of optimization epochs")
    parser.add_argument("--edit-budget", type=int, default=3, help="Maximum edits per epoch")
    parser.add_argument("--api-type", default="ollama", choices=["ollama", "openai"], help="LLM API type")
    parser.add_argument("--model-name", default="yerli-qwen:latest", help="LLM model name for optimizer")
    
    args = parser.parse_args()
    
    # Initialize benchmark optimizer
    optimizer = BenchmarkOptimizer(
        skill_path=args.skill_path,
        benchmark=args.benchmark,
        model_path=args.model_path,
        dataset=args.dataset,
        cartridge_path=args.cartridge
    )
    
    # Run optimization
    result = await optimizer.optimize_skill(
        epochs=args.epochs,
        edit_budget=args.edit_budget,
        api_type=args.api_type,
        model_name=args.model_name
    )


if __name__ == "__main__":
    asyncio.run(main())
