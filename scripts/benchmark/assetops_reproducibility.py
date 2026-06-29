"""
AssetOpsBench Integration - Reproducibility Module
===============================================

This module ensures strict determinism for AssetOpsBench evaluation.
"""

import random
import torch
import numpy as np
import logging
from typing import Dict, Any


def setup_assetops_reproducibility(seed: int = 42) -> None:
    """
    Configure global random seeds for AssetOpsBench evaluation.
    
    Args:
        seed: Fixed random seed for all components
    """
    # Set seeds for all random generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic CUDA
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    logging.info(f"AssetOpsBench reproducibility configured with seed: {seed}")


def get_assetops_reproducibility_config() -> Dict[str, Any]:
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
