import torch
import torch.nn as nn
import math

class HSPProjector(nn.Module):
    """
    Hyperdimensional Sparse Projection (HSP).
    Asymmetric dense-to-bipolar bridge.
    """
    def __init__(self, in_dim: int, hd_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hd_dim = hd_dim
        
        # Frozen Random Gaussian encoding locked in GPU register memory
        projection_matrix = torch.randn(in_dim, hd_dim) / math.sqrt(in_dim)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects dense stream into bipolar HD space.
        """
        projected = torch.matmul(x, self.projection_matrix)
        
        # Bipolar threshold mapping
        hd_vectors = torch.sign(projected)
        hd_vectors = torch.where(hd_vectors == 0.0, torch.ones_like(hd_vectors), hd_vectors)
        
        return hd_vectors
