"""
SRE Evaluation Runner - Rust/Candle Integration
===============================================

Evaluation runner for SRE (Site Reliability Engineering) benchmark using Rust Candle model.
"""

import json
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from candle_bridge import CandleModelBridge
from assetops_logging_utils import AssetOpsTaskLogger, setup_assetops_logging
from benchmark_logger import save_benchmark_results


class SRERunner:
    """
    Production-grade evaluation runner for SRE benchmark.
    """
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        log_level: str = "INFO",
        cartridge_path: Optional[str] = None
    ):
        """
        Initialize evaluation runner with Candle model.
        
        Args:
            model_path: Path to Candle-compatible model
            max_new_tokens: Maximum tokens to generate
            log_level: Logging level
            cartridge_path: Path to QMEM cartridge file
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Setup logging
        setup_assetops_logging(log_level)
        self.logger = AssetOpsTaskLogger()
        
        # Initialize Candle model bridge
        self.model = CandleModelBridge(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            cartridge_path=cartridge_path
        )
        
        # Results storage
        self.results = []
        
        logging.info(f"SRERunner initialized: {model_path}, max_tokens: {max_new_tokens}")
    
    def run_single_inference(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single-shot inference for SRE scenario.
        
        Args:
            scenario_data: Dictionary containing scenario information
            
        Returns:
            Dictionary with inference results
        """
        # Format prompt for SRE analysis
        prompt = self._format_sre_prompt(scenario_data)
        
        example = {
            "task_id": f"sre_{len(self.results)}",
            "input_prompt": prompt
        }
        
        logging.info(f"Running Candle inference for SRE scenario")
        
        # Start task logging
        self.logger.start_task(example["task_id"], prompt, self.model.get_model_info())
        
        # Run inference through Rust Candle
        result = self.model.run_single_inference(example)
        
        # Parse answer for fault identification (try raw_output first, then fallback to parsed_answer)
        parsed_answer = self._parse_sre_answer(result.get("raw_output", ""), scenario_data)
        if not parsed_answer:
            parsed_answer = self._parse_sre_answer(result.get("parsed_answer", ""), scenario_data)
        
        # Log inference result
        self.logger.log_inference(
            raw_output=result.get("raw_output", ""),
            parsed_answer=parsed_answer,
            token_count=result.get("token_count", 0)
        )
        
        # Complete task
        self.logger.complete_task()
        
        # Add task_id to result for tracking
        result["task_id"] = example["task_id"]
        result["status"] = "completed" if parsed_answer else "failed"
        result["parsed_answer"] = parsed_answer
        result["scenario_data"] = scenario_data
        
        self.results.append(result)
        return result
    
    def _format_sre_prompt(self, scenario_data: Dict[str, Any]) -> str:
        """Format SRE scenario for the model."""
        # Truncate alerts if they are too many to avoid hitting max token limits or context issues
        all_alerts = scenario_data.get("alerts", [])
        alerts = all_alerts[:10]
        truncation_msg = ""
        if len(all_alerts) > 10:
            truncation_msg = f"\n(Note: Showed first 10 alerts out of {len(all_alerts)})"

        # Inject dummy normal alerts for other candidate services to prevent the model from just copying the only service name it sees.
        import copy
        import random
        
        all_services = [
            "frontend-service", "ad-service", "shipping-service", "checkout-service", 
            "product-catalog-service", "email-service", "quote-service", "cart-service", 
            "details-service", "payment-service", "kafka-service", "postgresql-database", 
            "reviews-v3", "kubernetes-api-server", "valkey-cart"
        ]
        
        firing_alerts = []
        firing_services = set()
        
        # Add actual firing alerts
        for alert in all_alerts:
            c_alert = copy.deepcopy(alert)
            firing_alerts.append(c_alert)
            if "service" in c_alert:
                firing_services.add(c_alert["service"])
            elif "resource" in c_alert:
                firing_services.add(c_alert["resource"])
                
        # Add dummy normal alerts for other services
        normal_alerts = []
        for s in all_services:
            if s not in firing_services:
                normal_alerts.append({
                    "alertname": "ServiceStatusNormal",
                    "service": s,
                    "status": "normal"
                })
                
        # Shuffle normal alerts
        random.seed(42)
        random.shuffle(normal_alerts)
        
        # Combine: always keep ALL firing alerts, and fill the rest up to 12 alerts with healthy ones
        injected_alerts = firing_alerts + normal_alerts[:max(0, 12 - len(firing_alerts))]
        
        # Shuffle the final combined list to remove positional bias
        random.shuffle(injected_alerts)
        alerts = injected_alerts

        user_content = f"""Analyze the following system failure scenario and identify the specific root cause entity.

Scenario Information:
- Scenario: {scenario_data.get('scenario_name', 'Unknown')}
- Alerts: {len(all_alerts)} alerts detected{truncation_msg}
- Metrics: {len(scenario_data.get('metrics', []))} metric files available

Sample Alerts (JSON format, containing both firing and normal status signals):
{json.dumps(alerts, indent=2)}

Task:
Identify the specific component or service name that is the root cause (exhibits firing alert status).

CRITICAL: Please analyze the failure scenario and alerts step-by-step. Write down your reasoning first to diagnose the root cause. At the very end of your response, output the root cause entity name wrapped in <answer> tags (e.g., <answer>frontend-service</answer>)."""

        prompt = f"<|im_start|>system\nYou are a Site Reliability Engineering (SRE) expert specializing in incident diagnosis and root cause analysis.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def _strip_thinking_blocks(self, raw_output: str) -> str:
        """Strip thinking blocks from raw model output to prevent false positive matching on thoughts."""
        if not raw_output:
            return ""
        import re
        # Remove any text between thinking tags
        text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<option_analysis>.*?</option_analysis>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Also remove unclosed thinking tags if they exist at the end of text
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thought>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<option_analysis>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    def _parse_sre_answer(self, raw_output: str, scenario_data: Dict[str, Any]) -> str:
        """Parse model output to extract the identified faulty entity."""
        if not raw_output:
            return ""
        
        import re
        
        # Strip thinking blocks to only evaluate the final answer section
        final_answer_text = self._strip_thinking_blocks(raw_output)
        if not final_answer_text:
            return ""

        # Get all valid service names in the scenario to validate candidates
        valid_services = set()
        for alert in scenario_data.get("alerts", []):
            if isinstance(alert, dict):
                service = alert.get("service")
                if service:
                    valid_services.add(service.lower())

        # 1. Try to extract from <answer> tags in the final answer text
        tag_match = re.search(r"<answer>(.*?)</answer>", final_answer_text, re.IGNORECASE | re.DOTALL)
        if tag_match:
            candidate = tag_match.group(1).strip().strip("'\"., \n\t")
            if candidate:
                return candidate

        # 2. Look for common prefixes in the final answer text
        prefixes = [
            r"root cause:\s*([a-zA-Z0-9\-_]+)",
            r"faulty entity:\s*([a-zA-Z0-9\-_]+)",
            r"entity:\s*([a-zA-Z0-9\-_]+)",
            r"answer is\s*([a-zA-Z0-9\-_]+)",
            r"problematic entity is:\s*([a-zA-Z0-9\-_]+)"
        ]
        
        for pattern in prefixes:
            match = re.search(pattern, final_answer_text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().strip("'\"., \n\t")
                if candidate.lower() not in {"the", "a", "an", "this", "that", "to", "be", "or", "and", "not", "our", "my", "your"}:
                    return candidate

        # 3. If model output is short (e.g. exactly a service name), assume it's the answer
        clean_output = final_answer_text.strip().strip("'\"., \n\t")
        if len(clean_output.split()) <= 2:
            # Check if any word matches a valid service name
            for token in re.findall(r"[a-zA-Z0-9\-_]+", clean_output):
                if token.lower() in valid_services:
                    return token.strip().strip("'\"., \n\t")
            # Fallback if it is a single alphanumeric token
            if clean_output.replace('-', '').replace('_', '').isalnum():
                return clean_output

        # 4. Strict check: If one of the valid services is explicitly mentioned in the final answer text
        # with an affirmative pattern, e.g., "is valkey-cart"
        for service in valid_services:
            patterns = [
                rf"\b(?:is|select|use|faulty|cause|culprit|target)\s+[\"']?{re.escape(service)}\b[\"']?",
                rf"[\"']?\b{re.escape(service)}\b[\"']?\s+(?:is\s+the|is\s+faulty|is\s+down|failed|has\s+an?)\b"
            ]
            for pattern in patterns:
                if re.search(pattern, final_answer_text, re.IGNORECASE):
                    # Find case-matching version of the service from scenario data
                    for alert in scenario_data.get("alerts", []):
                        if isinstance(alert, dict):
                            s = alert.get("service")
                            if s and s.lower() == service:
                                return s

        # 5. Substring match: if exactly one of the valid services is mentioned anywhere in the output,
        # extract it. This is clean and avoids any guess-based default fallbacks.
        found_services = []
        for service in valid_services:
            if re.search(rf"\b{re.escape(service)}\b", final_answer_text, re.IGNORECASE):
                for alert in scenario_data.get("alerts", []):
                    if isinstance(alert, dict):
                        s = alert.get("service")
                        if s and s.lower() == service and s not in found_services:
                            found_services.append(s)
        if len(found_services) == 1:
            return found_services[0]

        return ""
    
    def evaluate_dataset(self, dataset_path: str, max_scenarios: Optional[int] = None) -> None:
        """
        Evaluate SRE dataset.
        
        Args:
            dataset_path: Path to SRE dataset directory
            max_scenarios: Maximum number of scenarios to evaluate (for testing)
        """
        logging.info(f"Starting SRE evaluation: {dataset_path}")
        
        # Load scenarios
        scenarios = self._load_sre_dataset(dataset_path, max_scenarios)
        
        if not scenarios:
            logging.error("No scenarios found in dataset")
            return
        
        # Evaluate each scenario
        for i, scenario in enumerate(scenarios):
            logging.info(f"Evaluating SRE scenario {i+1}/{len(scenarios)}")
            
            result = self.run_single_inference(scenario)
            
            # Store ground truth for evaluation
            result["ground_truth"] = scenario.get("ground_truth", {})
            
            if result["status"] == "error":
                logging.error(f"Scenario {i+1} failed: {result.get('error')}")
        
        logging.info(f"Completed evaluation of {len(scenarios)} SRE scenarios")
    
    def _load_sre_dataset(self, dataset_path: str, max_scenarios: Optional[int]) -> List[Dict[str, Any]]:
        """Load SRE dataset from directory structure."""
        scenarios = []
        dataset_dir = Path(dataset_path)
        
        try:
            # Find all scenario directories
            scenario_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("Scenario-")]
            scenario_dirs.sort()
            
            for scenario_dir in scenario_dirs[:max_scenarios] if max_scenarios else scenario_dirs:
                # Load scenario data
                scenario_data = {
                    "scenario_name": scenario_dir.name
                }
                
                # Load alerts
                alerts_dir = scenario_dir / "alerts"
                if alerts_dir.exists():
                    alerts = []
                    for alert_file in alerts_dir.glob("*.json"):
                        try:
                            with open(alert_file, 'r') as f:
                                alert = json.load(f)
                                alerts.append(alert)
                        except Exception as e:
                            logging.warning(f"Failed to load alert {alert_file}: {e}")
                    scenario_data["alerts"] = alerts
                
                # Load ground truth
                gt_file = scenario_dir / "ground_truth.yaml"
                if gt_file.exists():
                    try:
                        try:
                            import yaml
                        except ImportError:
                            import subprocess
                            import sys
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
                            import yaml

                        with open(gt_file, 'r') as f:
                            gt_data = yaml.safe_load(f)
                            # Extract fault entity names
                            faults = gt_data.get("fault", [])
                            entity_names = []
                            for f_item in faults:
                                # Safe extraction
                                entity = f_item.get("entity", {})
                                name = entity.get("name") if isinstance(entity, dict) else None
                                if name:
                                    entity_names.append(name)
                            
                            scenario_data["ground_truth"] = {
                                "fault": [{"entity": {"name": name}} for name in entity_names]
                            }
                    except Exception as e:
                        logging.warning(f"Failed to load ground truth for {scenario_dir.name}: {e}")
                        scenario_data["ground_truth"] = {"fault": []}
                
                # Load metrics info
                metrics_dir = scenario_dir / "metrics"
                if metrics_dir.exists():
                    metric_files = list(metrics_dir.glob("*.json"))
                    scenario_data["metrics"] = [f.name for f in metric_files]
                
                # Load events data
                events_file = scenario_dir / "k8s_events_otel-demo_chaos-mesh.tsv"
                if events_file.exists():
                    try:
                        df = pd.read_csv(events_file, sep='\t')
                        scenario_data["events_summary"] = {
                            "rows": len(df),
                            "columns": list(df.columns),
                            "sample_events": df.head(3).to_dict('records')
                        }
                    except Exception as e:
                        logging.warning(f"Failed to load events for {scenario_dir.name}: {e}")
                
                scenarios.append(scenario_data)
                
        except Exception as e:
            logging.error(f"Failed to load SRE dataset {dataset_path}: {e}")
            return []
        
        return scenarios
    
    def save_results(self, output_path: str = "results_sre_candle.json") -> None:
        """
        Save evaluation results in SRE format.
        
        Args:
            output_path: Output file path
        """
        if not self.results:
            logging.warning("No results to save")
            return
        
        # Compute accuracy
        correct_identifications = 0
        total_identifications = 0
        
        for result in self.results:
            ground_truth = result.get("ground_truth", {})
            gt_entities = [f.get("entity", {}).get("name") for f in ground_truth.get("fault", [])]
            parsed_answer = result.get("parsed_answer", "")
            
            total_identifications += 1
            if parsed_answer and any(parsed_answer.lower() == gt.lower() for gt in gt_entities):
                correct_identifications += 1
        
        accuracy = correct_identifications / total_identifications if total_identifications > 0 else 0.0
        
        results_data = {
            "accuracy": accuracy,
            "correct_identifications": correct_identifications,
            "total_identifications": total_identifications,
            "num_tasks": len(self.results),
            "num_completed": sum(1 for r in self.results if r.get("status") == "completed"),
            "completion_rate": sum(1 for r in self.results if r.get("status") == "completed") / len(self.results) if self.results else 0.0,
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
            
            # Also save to text files for easy reading
            save_benchmark_results(results_data, "SRE", "run_logs")
            
        except Exception as e:
            logging.error(f"Failed to save results: {e}")


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SRE Candle Evaluation Runner")
    parser.add_argument("--model-path", required=True, help="Path to Candle-compatible model")
    parser.add_argument("--dataset", required=True, help="Path to SRE dataset directory")
    parser.add_argument("--max-scenarios", type=int, help="Maximum number of scenarios to evaluate")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--output", default="results_sre_candle.json", help="Output file path")
    parser.add_argument("--cartridge", help="Path to QMEM cartridge")
    
    args = parser.parse_args()
    
    # Run evaluation
    runner = SRERunner(
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        log_level=args.log_level,
        cartridge_path=args.cartridge
    )
    
    runner.evaluate_dataset(args.dataset, args.max_scenarios)
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
