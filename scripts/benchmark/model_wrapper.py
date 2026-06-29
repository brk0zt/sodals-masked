"""
Local Model Wrapper - Production LLM Interface
============================================

Clean, deterministic interface for local HuggingFace-style models.
"""

import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from reproducibility import setup_reproducibility, get_reproducibility_config


class LocalModelWrapper:
    """
    Production-grade wrapper for local LLM inference.
    
    Features:
    - Deterministic inference
    - CUDA device placement
    - Enforced deterministic decoding
    - Structured logging
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize model wrapper.
        
        Args:
            model_path: Path to local model directory
            device: CUDA device (auto-detected if None)
            seed: Random seed for reproducibility
        """
        self.model_path = Path(model_path)
        self.seed = seed
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.model_version = "local_model"
        
        # Setup reproducibility
        setup_reproducibility(seed)
        
        logging.info(f"ModelWrapper initialized: {model_path}, device: {self.device}, seed: {seed}")
    
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
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate text with deterministic parameters.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored, set to 0.0)
            top_p: Nucleus sampling parameter (ignored, set to 1.0)
            do_sample: Whether to sample (ignored, set to False)
        
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Enforce deterministic parameters
        temperature = 0.0
        top_p = 1.0
        do_sample = False
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Reasonable context limit
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            # Generate with deterministic settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode output
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # Strip reasoning text if present (common in benchmark models)
            if "### Reasoning:" in generated_text:
                generated_text = generated_text.split("### Reasoning:")[0].strip()
            
            return generated_text
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "seed": self.seed,
            "model_version": self.model_version,
            "reproducibility_config": get_reproducibility_config()
        }
