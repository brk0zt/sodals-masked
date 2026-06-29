from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GraphTensor:
    """
    GRT Compute Primitive.
    Maintains structurally zero-copied tensors between CUDA and HDC routines.
    """
    x: torch.Tensor
    edge_index: torch.Tensor
    sdr_mask: torch.Tensor
    edge_weight: Optional[torch.Tensor] = None
    hd_vector: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.sdr_mask = self.sdr_mask.to(device)
        
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(device)
        if self.hd_vector is not None:
            self.hd_vector = self.hd_vector.to(device)
            
        return self
