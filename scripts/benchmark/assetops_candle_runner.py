"""
AssetOpsBench Evaluation Runner - Rust/Candle Integration
====================================================

Main evaluation runner using Rust Candle model for AssetOpsBench benchmark.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from candle_bridge import CandleModelBridge
from assetops_logging_utils import AssetOpsTaskLogger, setup_assetops_logging


class AssetOpsCandleRunner:
    """
    Production-grade evaluation runner using Rust Candle model.
    """
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        log_level: str = "INFO"
    ):
        """
        Initialize evaluation runner with Candle model.
        
        Args:
            model_path: Path to Candle-compatible model
            max_new_tokens: Maximum tokens to generate
            log_level: Logging level
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Setup logging
        setup_assetops_logging(log_level)
        self.logger = AssetOpsTaskLogger()
        
        # Initialize Candle model bridge
        self.model = CandleModelBridge(
            model_path=model_path,
            max_new_tokens=max_new_tokens
        )
        
        # Results storage
        self.results = []
        
        logging.info(f"AssetOpsCandleRunner initialized: {model_path}, max_tokens: {max_new_tokens}")
    
    def run_single_inference(self, example: dict) -> Dict[str, Any]:
        """
        Single-shot inference using Rust Candle model.
        
        Args:
            example: Dictionary containing task data
                - task_id: str
                - input_prompt: str
                - metadata: dict (optional)
        
        Returns:
            Dictionary with inference results
        """
        task_id = example.get("task_id", "unknown")
        input_prompt = example.get("input_prompt", "")
        
        logging.info(f"Running Candle inference for AssetOpsBench task {task_id}")
        
        # Start task logging
        self.logger.start_task(task_id, input_prompt, self.model.get_model_info())
        
        # Run inference through Rust Candle
        result = self.model.run_single_inference(example)
        
        # Log inference result
        self.logger.log_inference(
            raw_output=result.get("raw_output", ""),
            parsed_answer=result.get("parsed_answer", ""),
            token_count=result.get("token_count", 0)
        )
        
        # Complete task
        self.logger.complete_task()
        
        # Add task_id to result for tracking
        result["task_id"] = task_id
        result["status"] = "completed" if result.get("parsed_answer") else "failed"
        
        self.results.append(result)
        return result
    
    def evaluate_dataset(self, dataset_path: str, test_split: str = "test") -> None:
        """
        Evaluate AssetOpsBench dataset using Rust Candle model.
        
        Args:
            dataset_path: Path to evaluation dataset
            test_split: Dataset split to evaluate (test/validation)
        """
        logging.info(f"Starting AssetOpsBench evaluation with Candle: {dataset_path}, split: {test_split}")
        
        # Load dataset
        tasks = self._load_assetops_dataset(dataset_path, test_split)
        
        if not tasks:
            logging.error("No tasks found in dataset")
            return
        
        # Evaluate each task
        for i, task in enumerate(tasks):
            task_id = f"assetops_candle_task_{i:04d}"
            logging.info(f"Evaluating AssetOpsBench task {task_id}")
            
            result = self.run_single_inference(task)
            
            if result["status"] == "error":
                logging.error(f"Task {task_id} failed: {result.get('error')}")
        
        logging.info(f"Completed evaluation of {len(tasks)} AssetOpsBench tasks with Rust Candle")
    
    def _load_assetops_dataset(self, dataset_path: str, split: str) -> List[Dict[str, str]]:
        """
        Load AssetOpsBench dataset (supports common formats).
        """
        try:
            with open(dataset_path, 'r') as f:
                if dataset_path.endswith('.jsonl'):
                    return self._load_jsonl(f, split)
                elif dataset_path.endswith('.json'):
                    return self._load_json(f, split)
                else:
                    # Assume simple text file with one prompt per line
                    return [{"task_id": f"task_{i}", "input_prompt": line.strip()} for i, line in enumerate(f) if line.strip()]
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_path}: {e}")
            return []
    
    def _load_jsonl(self, file, split: str) -> List[Dict[str, str]]:
        """Load JSONL format dataset with split support."""
        tasks = []
        for line_num, line in enumerate(file, 1):
            if line.strip():
                try:
                    task = json.loads(line)
                    if "input_prompt" in task:
                        if "task_id" not in task:
                            task["task_id"] = f"task_{line_num:04d}"
                        tasks.append(task)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON line {line_num}: {line}")
        return tasks
    
    def _load_json(self, file, split: str) -> List[Dict[str, str]]:
        """Load JSON format dataset with split support."""
        try:
            data = json.load(file)
            
            # Handle different JSON structures
            if isinstance(data, list):
                tasks = data
            elif isinstance(data, dict):
                if "tasks" in data:
                    tasks = data["tasks"]
                elif "data" in data and isinstance(data["data"], list):
                    tasks = data["data"]
                elif "test" in data and isinstance(data["test"], list):
                    tasks = data["test"]
                elif split in data and isinstance(data[split], list):
                    tasks = data[split]
                else:
                    tasks = [data] if "input_prompt" in data else []
            else:
                logging.error(f"Unexpected JSON structure")
                return []
                
            # Ensure all tasks have required fields
            valid_tasks = []
            for i, task in enumerate(tasks):
                if "input_prompt" in task:
                    if "task_id" not in task:
                        task["task_id"] = f"task_{i:04d}"
                    valid_tasks.append(task)
                    
            return valid_tasks
            
        except Exception as e:
            logging.error(f"Failed to load JSON dataset: {e}")
            return []
    
    def save_results(self, output_path: str = "results_assetops_candle.json") -> None:
        """
        Save evaluation results in AssetOpsBench format.
        
        Args:
            output_path: Output file path
        """
        if not self.results:
            logging.warning("No results to save")
            return
        
        # Compute basic statistics
        completed_tasks = len([r for r in self.results if r.get("status") == "completed"])
        total_tasks = len(self.results)
        
        results_data = {
            "num_tasks": total_tasks,
            "num_completed": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0.0,
            "model_type": "rust_candle",
            "config": {
                "model_path": self.model_path,
                "max_new_tokens": self.max_new_tokens,
                "reproducibility_config": self.model.get_model_info()["reproducibility_config"]
            }
        }
        
        # Save results
        try:
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AssetOpsBench Candle Evaluation Runner")
    parser.add_argument("--model-path", required=True, help="Path to Candle-compatible model")
    parser.add_argument("--dataset", required=True, help="Path to AssetOpsBench dataset")
    parser.add_argument("--split", default="test", choices=["test", "validation"], help="Dataset split to evaluate")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--output", default="results_assetops_candle.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Run evaluation
    runner = AssetOpsCandleRunner(
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        log_level=args.log_level
    )
    
    runner.evaluate_dataset(args.dataset, args.split)
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
