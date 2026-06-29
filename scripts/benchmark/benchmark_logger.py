"""
Log and Results Saver Utility
============================

Utility for saving benchmark logs and results to text files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class BenchmarkLogger:
    """
    Utility class to save benchmark logs and results to text files.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the logger with output directory.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_results_summary(self, results: Dict[str, Any], benchmark_name: str) -> str:
        """
        Save results summary to a text file.
        
        Args:
            results: Results dictionary
            benchmark_name: Name of the benchmark
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_results_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{benchmark_name.upper()} BENCHMARK RESULTS\n")
            f.write(f"{'='*60}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Main metrics
            if "accuracy" in results:
                f.write(f"ACCURACY: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
            
            if "correct_answers" in results:
                f.write(f"CORRECT ANSWERS: {results['correct_answers']}\n")
                f.write(f"TOTAL ANSWERS: {results.get('total_answers', 'N/A')}\n")
            
            if "correct_identifications" in results:
                f.write(f"CORRECT IDENTIFICATIONS: {results['correct_identifications']}\n")
                f.write(f"TOTAL IDENTIFICATIONS: {results.get('total_identifications', 'N/A')}\n")
            
            if "mean_score" in results:
                f.write(f"MEAN SCORE: {results['mean_score']:.4f}\n")
            
            if "std" in results:
                f.write(f"STD DEVIATION: {results['std']:.4f}\n")
            
            # Task counts
            f.write(f"\nTASK STATISTICS:\n")
            f.write(f"  Total Tasks: {results.get('num_tasks', 'N/A')}\n")
            f.write(f"  Completed Tasks: {results.get('num_completed', 'N/A')}\n")
            f.write(f"  Completion Rate: {results.get('completion_rate', 'N/A')}\n")
            
            # Model configuration
            f.write(f"\nMODEL CONFIGURATION:\n")
            config = results.get('config', {})
            f.write(f"  Model Type: {results.get('model_type', 'N/A')}\n")
            f.write(f"  Model Path: {config.get('model_path', 'N/A')}\n")
            f.write(f"  Max New Tokens: {config.get('max_new_tokens', 'N/A')}\n")
            
            repro_config = config.get('reproducibility_config', {})
            f.write(f"  Seed: {repro_config.get('seed', 'N/A')}\n")
            f.write(f"  Temperature: {repro_config.get('temperature', 'N/A')}\n")
            f.write(f"  Top P: {repro_config.get('top_p', 'N/A')}\n")
            f.write(f"  Do Sample: {repro_config.get('do_sample', 'N/A')}\n")
        
        return str(filepath)
    
    def save_detailed_logs(self, log_dir: str, benchmark_name: str) -> str:
        """
        Save detailed logs from log directory to a consolidated text file.
        
        Args:
            log_dir: Directory containing log files
            benchmark_name: Name of the benchmark
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_detailed_logs_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        log_path = Path(log_dir)
        if not log_path.exists():
            return str(filepath)  # Return empty file path if no logs
        
        log_files = sorted(log_path.glob("*.json"))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{benchmark_name.upper()} DETAILED LOGS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Log Files: {len(log_files)}\n\n")
            
            for i, log_file in enumerate(log_files, 1):
                f.write(f"{'-'*60}\n")
                f.write(f"LOG {i}: {log_file.name}\n")
                f.write(f"{'-'*60}\n")
                
                try:
                    with open(log_file, 'r', encoding='utf-8') as log_f:
                        log_data = json.load(log_f)
                    
                    # Extract key information
                    if len(log_data) > 0:
                        task_info = log_data[0]
                        f.write(f"Task ID: {task_info.get('task_id', 'N/A')}\n")
                        f.write(f"Timestamp: {task_info.get('timestamp', 'N/A')}\n")
                        
                        # Show prompt preview
                        prompt = task_info.get('input_prompt', task_info.get('full_prompt', ''))
                        if prompt:
                            preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                            f.write(f"Prompt Preview: {preview}\n")
                        
                        f.write(f"Model: {task_info.get('model_info', {}).get('model_path', 'N/A')}\n")
                        f.write(f"Seed: {task_info.get('seed', 'N/A')}\n")
                    
                    # Show inference results
                    if len(log_data) > 1:
                        inference = log_data[1]
                        f.write(f"Raw Output: {inference.get('raw_output', 'N/A')}\n")
                        f.write(f"Parsed Answer: {inference.get('parsed_answer', 'N/A')}\n")
                        f.write(f"Token Count: {inference.get('token_count', 'N/A')}\n")
                        f.write(f"Inference Time: {inference.get('inference_time', 'N/A')}s\n")
                    
                    # Show completion info
                    if len(log_data) > 2:
                        completion = log_data[2]
                        f.write(f"Completed At: {completion.get('completed_at', 'N/A')}\n")
                        f.write(f"Total Time: {completion.get('total_time', 'N/A')}s\n")
                        f.write(f"Final Score: {completion.get('final_score', 'N/A')}\n")
                    
                except Exception as e:
                    f.write(f"Error reading log file: {e}\n")
                
                f.write("\n")
        
        return str(filepath)
    
    def save_comparison_report(self, results_list: List[Dict[str, Any]], benchmark_names: List[str]) -> str:
        """
        Save comparison report for multiple benchmarks.
        
        Args:
            results_list: List of results dictionaries
            benchmark_names: List of benchmark names
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_comparison_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write("BENCHMARK COMPARISON REPORT\n")
            f.write(f"{'='*80}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write(f"{'BENCHMARK':<20} {'ACCURACY':<12} {'TASKS':<10} {'COMPLETED':<12} {'MODEL':<15}\n")
            f.write(f"{'-'*80}\n")
            
            for results, name in zip(results_list, benchmark_names):
                accuracy = results.get('accuracy', 0)
                tasks = results.get('num_tasks', 0)
                completed = results.get('num_completed', results.get('total_answers', 0))
                model_type = results.get('model_type', 'N/A')
                
                f.write(f"{name:<20} {accuracy:<12.4f} {tasks:<10} {completed:<12} {model_type:<15}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("DETAILED BREAKDOWN\n")
            f.write(f"{'='*80}\n\n")
            
            # Detailed breakdown for each benchmark
            for results, name in zip(results_list, benchmark_names):
                f.write(f"\n{name.upper()} BENCHMARK:\n")
                f.write(f"{'-'*40}\n")
                
                if "accuracy" in results:
                    f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
                
                if "correct_answers" in results:
                    f.write(f"Correct Answers: {results['correct_answers']}/{results.get('total_answers', 'N/A')}\n")
                
                if "correct_identifications" in results:
                    f.write(f"Correct Identifications: {results['correct_identifications']}/{results.get('total_identifications', 'N/A')}\n")
                
                if "mean_score" in results:
                    f.write(f"Mean Score: {results['mean_score']:.4f}\n")
                
                f.write(f"Total Tasks: {results.get('num_tasks', 'N/A')}\n")
                f.write(f"Completed: {results.get('num_completed', 'N/A')}\n")
                f.write(f"Model: {results.get('model_type', 'N/A')}\n")
                
                config = results.get('config', {})
                f.write(f"Model Path: {config.get('model_path', 'N/A')}\n")
                f.write(f"Max Tokens: {config.get('max_new_tokens', 'N/A')}\n")
                
                repro_config = config.get('reproducibility_config', {})
                f.write(f"Seed: {repro_config.get('seed', 'N/A')}\n")
                f.write(f"Temperature: {repro_config.get('temperature', 'N/A')}\n")
                
                f.write("\n")
        
        return str(filepath)


def save_benchmark_results(results: Dict[str, Any], benchmark_name: str, log_dir: str = "run_logs"):
    """
    Convenience function to save benchmark results and logs.
    
    Args:
        results: Results dictionary
        benchmark_name: Name of the benchmark
        log_dir: Directory containing log files
    """
    logger = BenchmarkLogger()
    
    # Save results summary
    results_file = logger.save_results_summary(results, benchmark_name)
    print(f"Results saved to: {results_file}")
    
    # Save detailed logs
    logs_file = logger.save_detailed_logs(log_dir, benchmark_name)
    print(f"Detailed logs saved to: {logs_file}")
    
    return results_file, logs_file
