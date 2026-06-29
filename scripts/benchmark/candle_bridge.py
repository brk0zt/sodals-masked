"""
AssetOpsBench Integration - Rust/Candle Bridge
==============================================

Bridge between Python evaluation framework and Rust Candle model.
"""

import subprocess
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class CandleModelBridge:
    """
    Bridge to Rust Candle model for AssetOpsBench evaluation.
    """
    
    def __init__(
        self,
        model_path: str,
        seed: int = 42,
        max_new_tokens: int = 2048,
        cartridge_path: Optional[str] = None
    ):
        """
        Initialize Candle model bridge.
        
        Args:
            model_path: Path to Candle-compatible model
            seed: Random seed for reproducibility
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path)
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.model_version = "candle_v1.0"
        
        # Path to compiled Rust binary - resolve relative to workspace root
        import os
        exe_ext = ".exe" if os.name == 'nt' else ""
        self.candle_binary = Path(__file__).parent.parent.parent / "target" / "release" / f"candle_inference_full{exe_ext}"
        self.cartridge_path = cartridge_path
        
        logging.info(f"CandleModelBridge initialized: {model_path}, seed: {seed}")
    
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
        
        logging.info(f"Running Candle inference for task {task_id}")
        
        try:
            # Prepare command for Rust Candle binary
            # Format: candle_inference_full --stdin [max_tokens] [model_path] [tokenizer_path]
            
            # Use SODALS_TOKENIZER_PATH if available in .env, otherwise fallback
            import os
            tokenizer_path = os.environ.get("SODALS_TOKENIZER_PATH")
            
            if not tokenizer_path and os.path.exists(".env"):
                try:
                    with open(".env", "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("SODALS_TOKENIZER_PATH="):
                                tokenizer_path = line.split("=", 1)[1].strip()
                                break
                except:
                    pass
                
            if not tokenizer_path:
                tokenizer_path = str(self.model_path.parent / "tokenizer.json")
            
            cmd = [
                str(self.candle_binary),
                "--stdin",
                str(self.max_new_tokens),
                str(self.model_path),
                str(tokenizer_path)
            ]
            if self.cartridge_path:
                cmd.extend(["--cartridge", str(self.cartridge_path)])
            
            logging.info(f"Executing command: {' '.join(cmd)}")
            
            import os
            env = os.environ.copy()
            # Force CPU inference unless SODALS_USE_GPU is explicitly set to 1

            if os.environ.get("SODALS_USE_GPU") == "1":

                if "SODALS_FORCE_CPU" in env:

                    del env["SODALS_FORCE_CPU"]

                if "CUDA_VISIBLE_DEVICES" in env:

                    del env["CUDA_VISIBLE_DEVICES"]

            else:

                env["CUDA_VISIBLE_DEVICES"] = ""

                env["SODALS_FORCE_CPU"] = "1"
            if os.environ.get("DISABLE_PURE_INFERENCE") == "1":
                if "SODALS_PURE_INFERENCE" in env:
                    del env["SODALS_PURE_INFERENCE"]
            else:
                env["SODALS_PURE_INFERENCE"] = "1"
            
            # Run Rust inference with prompt on stdin
            result = subprocess.run(
                cmd,
                input=input_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                timeout=1800  # Increased to 30 minutes for CPU inference on complex SRE scenarios
            )
            
            if result.returncode != 0:
                logging.error(f"Candle inference failed for task {task_id} with exit code {result.returncode}")
                # Safe decoding for Windows
                safe_stdout = result.stdout.encode('ascii', errors='replace').decode('ascii')
                safe_stderr = result.stderr.encode('ascii', errors='replace').decode('ascii')
                logging.error(f"STDOUT: {safe_stdout}")
                logging.error(f"STDERR: {safe_stderr}")
                return {
                    "task_id": task_id,
                    "raw_output": "",
                    "parsed_answer": "",
                    "token_count": 0,
                    "error": f"Return code {result.returncode}: {safe_stderr}"
                }
            
            if result.stderr:
                safe_stderr = result.stderr.encode('ascii', errors='replace').decode('ascii')
                logging.info(f"Rust binary stderr for task {task_id}:\n{safe_stderr}")

            # Parse Rust output (expecting JSON)
            try:
                logging.info(f"Task {task_id}: stdout length={len(result.stdout)}, first 200 chars: {repr(result.stdout[:200])}")
                output_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                logging.warning(f"Task {task_id}: Failed to parse JSON from stdout. Using raw output.")
                logging.warning(f"Task {task_id}: Raw stdout (first 500 chars): {repr(result.stdout[:500])}")
                logging.warning(f"Task {task_id}: JSON decode error: {e}")
                # Fallback: treat entire output as raw text
                output_data = {
                    "raw_output": result.stdout,
                    "token_count": len(result.stdout.split())
                }
            
            # Parse answer using AssetOpsBench logic
            raw_output = output_data.get("raw_output", "")
            parsed_answer = self._parse_assetops_answer(raw_output)
            token_count = output_data.get("token_count", 0)
            
            result = {
                "task_id": task_id,
                "raw_output": raw_output,
                "parsed_answer": parsed_answer,
                "token_count": token_count
            }
            
            logging.info(f"Task {task_id} completed: {token_count} tokens")
            
            return result
            
        except subprocess.TimeoutExpired:
            logging.error(f"Candle inference timeout for task {task_id}")
            return {
                "task_id": task_id,
                "raw_output": "",
                "parsed_answer": "",
                "token_count": 0,
                "error": "TIMEOUT"
            }
        except Exception as e:
            logging.error(f"Candle inference failed for task {task_id}: {e}")
            return {
                "task_id": task_id,
                "raw_output": "",
                "parsed_answer": "",
                "token_count": 0,
                "error": str(e)
            }
    
    def _parse_assetops_answer(self, raw_output: str) -> str:
        """
        Parse model output for AssetOpsBench tasks.
        
        AssetOpsBench may require:
        - Classification labels
        - Structured decisions
        - Short-form answers
        """
        if not raw_output:
            return ""
        
        # 1. Try to extract from <answer> tags first (across the entire raw output)
        import re
        tag_match = re.search(r"<answer>(.*?)</answer>", raw_output, re.IGNORECASE | re.DOTALL)
        if tag_match:
            candidate = tag_match.group(1).strip()
            if candidate:
                return candidate
        
        # 2. Otherwise, fall back to standard paragraph splitting
        # Remove common reasoning prefixes
        reasoning_prefixes = [
            "### Reasoning:",
            "### Analysis:",
            "### Thought Process:",
            "Let me think step by step:",
            "Step 1:",
            "Step 2:",
            "Step 3:",
            "First,",
            "Second,",
            "Third,",
            "Therefore,",
            "In conclusion,"
        ]
        
        parsed = raw_output.strip()
        
        # Strip reasoning
        for prefix in reasoning_prefixes:
            if parsed.startswith(prefix):
                parsed = parsed[len(prefix):].strip()
                break
        
        # Extract final answer
        answer_patterns = [
            "### Final Answer:",
            "Final Answer:",
            "### Answer:",
            "Answer:",
            "## Answer:",
            "### Decision:",
            "Decision:",
            "### Classification:",
            "Classification:",
            "### Recommendation:",
            "Recommendation:"
        ]
        
        for pattern in answer_patterns:
            if pattern in parsed:
                answer_start = parsed.find(pattern) + len(pattern)
                parsed = parsed[answer_start:].strip()
                break
        
        # Clean up
        parsed = parsed.strip()
        if "\n\n" in parsed:
            parsed = parsed.split("\n\n")[0].strip()
        
        return parsed
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_path": str(self.model_path),
            "seed": self.seed,
            "model_version": self.model_version,
            "max_new_tokens": self.max_new_tokens,
            "reproducibility_config": {
                "seed": self.seed,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False
            }
        }
