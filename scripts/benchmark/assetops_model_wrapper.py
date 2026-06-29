"""
AssetOpsBench Model Wrapper - Production LLM Interface
==============================================

Clean, deterministic interface for local HuggingFace-style models.
"""

import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from assetops_reproducibility import setup_assetops_reproducibility, get_assetops_reproducibility_config


class AssetOpsModelWrapper:
    """
    Production-grade wrapper for AssetOpsBench evaluation.
    
    Features:
    - Deterministic inference with fixed decoding parameters
    - CUDA device placement
    - Single-shot inference (no agent loops)
    - Structured logging integration
    - AssetOpsBench-specific answer parsing
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        seed: int = 42,
        max_new_tokens: int = 2048
    ):
        """
        Initialize model wrapper for AssetOpsBench.
        
        Args:
            model_path: Path to local model directory
            device: CUDA device (auto-detected if None)
            seed: Random seed for reproducibility
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path)
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.model_version = "assetops_v1.0"
        
        # Setup reproducibility
        setup_assetops_reproducibility(seed)
        
        logging.info(f"AssetOpsModelWrapper initialized: {model_path}, device: {self.device}, seed: {seed}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup CUDA device with fallback."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
                logging.info(f"CUDA detected, using device: {device}")
            else:
                device = "cpu"
                logging.warning("CUDA not available, falling back to CPU")
        return device
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            logging.info(f"Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Move to target device
            self.model = self.model.to(self.device)
            
            # Set to eval mode for deterministic inference
            self.model.eval()
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def run_single_inference(self, example: dict) -> dict:
        """
        Single-shot inference for AssetOpsBench.
        
        Args:
            example: Dictionary containing task data
                - task_id: str
                - input_prompt: str
                - metadata: dict (optional)
        
        Returns:
            Dictionary with inference results
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        task_id = example.get("task_id", "unknown")
        input_prompt = example.get("input_prompt", "")
        
        logging.info(f"Running inference for task {task_id}")
        
        # Tokenize input
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Reasonable context limit
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            # Generate with deterministic AssetOpsBench settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,  # Fixed for determinism
                    top_p=1.0,        # Fixed for determinism
                    do_sample=False,      # Fixed for determinism
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode output
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            raw_output = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # Parse answer using AssetOpsBench-specific logic
            parsed_answer = self._parse_assetops_answer(raw_output)
            
            # Log token count
            token_count = len(generated_tokens)
            
            result = {
                "task_id": task_id,
                "raw_output": raw_output,
                "parsed_answer": parsed_answer,
                "token_count": token_count
            }
            
            logging.info(f"Task {task_id} completed: {token_count} tokens")
            
            return result
            
        except Exception as e:
            logging.error(f"Inference failed for task {task_id}: {e}")
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
        
        This function implements robust parsing:
        - Strip reasoning text
        - Extract final answer
        - Remove extraneous commentary
        """
        if not raw_output:
            return ""
        
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
                # Find the actual answer after reasoning
                parsed = parsed[len(prefix):].strip()
                break
        
        # Extract final answer (look for common answer patterns)
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
        
        # Clean up any remaining prefixes
        parsed = parsed.strip()
        
        # Remove any trailing commentary
        if "\n\n" in parsed:
            # Take only the first part before double newline
            parsed = parsed.split("\n\n")[0].strip()
        
        return parsed
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "seed": self.seed,
            "model_version": self.model_version,
            "max_new_tokens": self.max_new_tokens,
            "reproducibility_config": get_assetops_reproducibility_config()
        }
