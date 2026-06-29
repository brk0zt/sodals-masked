"""
ITBench Evaluation Integration - Reproducibility Module
============================================================

This module ensures deterministic evaluation across all components.
"""

import random
import torch
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path


def setup_reproducibility(seed: int = 42) -> None:
    """
    Configure global random seeds for reproducible evaluation.
    
    Args:
        seed: Random seed for all components
    """
    # Set seeds for all random generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic CUDA
    torch.backends.cudnn.denchmark = False
    torch.backends.cudnn.deterministic = True
    
    logging.info(f"Reproducibility configured with seed: {seed}")


def get_reproducibility_config() -> Dict[str, Any]:
    """
    Get current reproducibility configuration for logging.
    """
    return {
        "torch_seed": torch.initial_seed(),
        "numpy_seed": np.random.get_state()[0][0] if hasattr(np.random, 'get_state') else None,
        "python_seed": random.getstate()[1][1] if hasattr(random, 'getstate') else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark
    }
