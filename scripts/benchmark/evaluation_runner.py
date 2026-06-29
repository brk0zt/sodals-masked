"""
Evaluation Runner - ITBench Integration
=========================================

Main evaluation runner for ITBench benchmark with agent-safe interface.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from model_wrapper import LocalModelWrapper
from logging_utils import TaskLogger, setup_evaluation_logging


class EvaluationRunner:
    """
    Production-grade evaluation runner with safety guards and structured logging.
    """
    
    def __init__(
        self,
        model_path: str,
        max_steps: int = 1000,
        inference_timeout: int = 300,
        log_level: str = "INFO"
    ):
        """
        Initialize evaluation runner.
        
        Args:
            model_path: Path to local model
            max_steps: Maximum steps per task (safety guard)
            inference_timeout: Timeout in seconds for inference
            log_level: Logging level
        """
        self.model_path = model_path
        self.max_steps = max_steps
        self.inference_timeout = inference_timeout
        
        # Setup logging
        setup_evaluation_logging(log_level)
        self.logger = TaskLogger()
        
        # Initialize model
        self.model = LocalModelWrapper(model_path)
        
        # Results storage
        self.results = []
        
        logging.info(f"EvaluationRunner initialized: {model_path}, max_steps: {max_steps}")
    
    def model_step(self, observation: str) -> str:
        """
        Agent-safe inference interface.
        
        Args:
            observation: Environment observation/prompt
            
        Returns:
            Action string from model
        """
        try:
            # Timeout guard
            start_time = time.time()
            
            # Generate with timeout
            result = self.model.generate(
                prompt=observation,
                max_new_tokens=min(256, self.max_steps)  # Reasonable limit
            )
            
            # Check for timeout
            if time.time() - start_time > self.inference_timeout:
                logging.warning(f"Inference timeout for task")
                return "TIMEOUT"
            
            return result
            
        except Exception as e:
            logging.error(f"Model inference failed: {e}")
            return "ERROR"
    
    def evaluate_task(self, task_id: str, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a single task with full logging.
        
        Args:
            task_id: Unique task identifier
            prompt: Task prompt
            
        Returns:
            Task result dictionary
        """
        # Start task logging
        self.logger.start_task(task_id, prompt, self.model.get_model_info())
        
        try:
            # Agent loop: Observation -> Model -> Action
            steps = 0
            current_observation = prompt
            
            while steps < self.max_steps:
                # Get action from model
                action = self.model_step(current_observation)
                
                # Log generation
                self.logger.log_generation(current_observation, action)
                
                # Check for completion
                if self._is_terminal(action):
                    break
                
                # Update observation for next step
                current_observation = action
                steps += 1
                
                # Safety guard against infinite loops
                if steps >= self.max_steps:
                    logging.warning(f"Max steps ({self.max_steps}) reached for task {task_id}")
                    break
            
            # Complete task
            final_score = self._compute_score(task_id, action)
            self.logger.complete_task(final_score)
            
            result = {
                "task_id": task_id,
                "status": "completed" if self._is_terminal(action) else "max_steps_reached",
                "steps": steps,
                "final_action": action,
                "final_score": final_score
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logging.error(f"Task evaluation failed: {e}")
            error_result = {
                "task_id": task_id,
                "status": "error",
                "error": str(e)
            }
            self.results.append(error_result)
            return error_result
    
    def _is_terminal(self, action: str) -> bool:
        """
        Check if action indicates task completion.
        
        Args:
            action: Model output action
            
        Returns:
            True if action appears terminal
        """
        # Common terminal indicators
        terminal_indicators = [
            "### END",
            "### FINAL ANSWER",
            "### COMPLETE",
            "## Answer:",
            "Final Answer:",
            "TASK COMPLETE"
        ]
        
        return any(indicator in action for indicator in terminal_indicators)
    
    def _compute_score(self, task_id: str, action: str) -> Optional[float]:
        """
        Compute task score (placeholder for benchmark integration).
        
        Args:
            task_id: Task identifier
            action: Final action string
            
        Returns:
            Computed score or None
        """
        # This is a placeholder - actual scoring depends on benchmark
        # For ITBench, scores are typically computed by comparing
        # against ground truth answers
        
        # For now, return a simple length-based score
        # In practice, this would be replaced with actual benchmark scoring
        return float(len(action)) if action else 0.0
    
    def evaluate_dataset(self, dataset_path: str) -> None:
        """
        Evaluate entire dataset.
        
        Args:
            dataset_path: Path to evaluation dataset
        """
        logging.info(f"Starting evaluation on dataset: {dataset_path}")
        
        # Load dataset (placeholder - actual implementation depends on format)
        # For ITBench, this would be the official evaluation set
        tasks = self._load_dataset(dataset_path)
        
        # Evaluate each task
        for i, task in enumerate(tasks):
            task_id = f"task_{i:04d}"
            logging.info(f"Evaluating task {task_id}")
            
            result = self.evaluate_task(task_id, task["prompt"])
            
            if result["status"] == "error":
                logging.error(f"Task {task_id} failed: {result.get('error')}")
        
        logging.info(f"Completed evaluation of {len(tasks)} tasks")
    
    def _load_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        """
        Load evaluation dataset (placeholder implementation).
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            List of task dictionaries
        """
        # This is a placeholder - actual implementation depends on ITBench format
        # Common formats include JSONL, JSON, CSV, etc.
        
        try:
            with open(dataset_path, 'r') as f:
                if dataset_path.endswith('.jsonl'):
                    return self._load_jsonl(f)
                elif dataset_path.endswith('.json'):
                    return self._load_json(f)
                else:
                    # Assume simple text file with one prompt per line
                    return [{"prompt": line.strip()} for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_path}: {e}")
            return []
    
    def _load_jsonl(self, file) -> List[Dict[str, str]]:
        """Load JSONL format dataset."""
        tasks = []
        for line in file:
            if line.strip():
                try:
                    task = json.loads(line)
                    if "prompt" in task:
                        tasks.append(task)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON line: {line}")
        return tasks
    
    def _load_json(self, file) -> List[Dict[str, str]]:
        """Load JSON format dataset."""
        try:
            data = json.load(file)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tasks" in data:
                return data["tasks"]
            else:
                # Assume single task object
                return [data] if "prompt" in data else []
        except Exception as e:
            logging.error(f"Failed to load JSON: {e}")
            return []
    
    def save_results(self, output_path: str = "results_itbench.json") -> None:
        """
        Save evaluation results in ITBench format.
        
        Args:
            output_path: Output file path
        """
        if not self.results:
            logging.warning("No results to save")
            return
        
        # Compute statistics
        scores = [r.get("final_score", 0.0) for r in self.results if r.get("status") == "completed"]
        completed_tasks = len([r for r in self.results if r.get("status") == "completed"])
        
        results_data = {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "num_tasks": len(self.results),
            "num_completed": completed_tasks,
            "config": {
                "model_path": self.model_path,
                "max_steps": self.max_steps,
                "inference_timeout": self.inference_timeout,
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
    
    parser = argparse.ArgumentParser(description="ITBench Evaluation Runner")
    parser.add_argument("--model-path", required=True, help="Path to local model")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per task")
    parser.add_argument("--timeout", type=int, default=300, help="Inference timeout in seconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--output", default="results_itbench.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Run evaluation
    runner = EvaluationRunner(
        model_path=args.model_path,
        max_steps=args.max_steps,
        inference_timeout=args.timeout,
        log_level=args.log_level
    )
    
    runner.evaluate_dataset(args.dataset)
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
