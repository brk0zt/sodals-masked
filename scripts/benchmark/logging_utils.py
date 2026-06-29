"""
Logging Utilities - Structured JSON Logging
==========================================

Provides structured logging for ITBench evaluation tasks.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class TaskLogger:
    """
    Structured logger for individual evaluation tasks.
    """
    
    def __init__(self, log_dir: str = "./run_logs"):
        """
        Initialize task logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.task_id = None
        self.start_time = None
        
    def start_task(self, task_id: str, prompt: str, model_info: Dict[str, Any]) -> None:
        """
        Start logging for a new task.
        
        Args:
            task_id: Unique task identifier
            prompt: Full input prompt
            model_info: Model configuration information
        """
        self.task_id = task_id
        self.start_time = time.time()
        
        log_entry = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "full_prompt": prompt,
            "model_info": model_info,
            "seed": model_info.get("seed", 42),
            "model_version": model_info.get("model_version", "unknown"),
            "decoding_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False
            }
        }
        
        # Write initial log entry
        self._write_log(log_entry)
        logging.info(f"Started task {task_id}")
    
    def log_generation(self, raw_model_output: str, parsed_action: str) -> None:
        """
        Log generation result.
        
        Args:
            raw_model_output: Raw output from model
            parsed_action: Cleaned action string
        """
        if self.task_id is None:
            logging.error("Task not started - cannot log generation")
            return
        
        log_entry = {
            "raw_model_output": raw_model_output,
            "parsed_action": parsed_action,
            "generation_time": time.time() - self.start_time
        }
        
        self._write_log(log_entry)
        logging.info(f"Logged generation for task {self.task_id}")
    
    def complete_task(self, final_score: Optional[float] = None) -> None:
        """
        Complete task logging.
        
        Args:
            final_score: Final evaluation score if available
        """
        if self.task_id is None:
            logging.error("Task not started - cannot complete")
            return
        
        log_entry = {
            "completed_at": datetime.now().isoformat(),
            "total_time": time.time() - self.start_time,
            "final_score": final_score
        }
        
        self._write_log(log_entry)
        logging.info(f"Completed task {self.task_id}")
    
    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to JSON file."""
        log_file = self.log_dir / f"{self.task_id}.json"
        
        # Read existing log if file exists
        existing_data = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read existing log {log_file}: {e}")
        
        existing_data.append(log_entry)
        
        # Write updated log
        try:
            with open(log_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to write log {log_file}: {e}")


def setup_evaluation_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration for evaluation.
    
    Args:
        log_level: Logging level (INFO, DEBUG, etc.)
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )
    
    # Create log directory
    Path("./run_logs").mkdir(exist_ok=True)
    
    logging.info("Evaluation logging configured")
