import torch
import torch.nn as nn
from tensor import GraphTensor
from kernels.spmm_triton import spmm_triton_forward
from hdc.projection import HSPProjector
from hdc.hypervector import HDCModule

class GraphSemanticConv(nn.Module):
    """
    High-level Semantic Convolution Layer.
    SDR Mask -> SpMM (Message Passing) -> HSP Projection -> HDC Bind
    """
    def __init__(self, in_features: int, out_features: int, sdr: bool = True, hdc: bool = True, hd_dim: int = 1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sdr = sdr
        self.hdc = hdc
        self.hd_dim = hd_dim

        # Temel özellik dönüşümü (Feature Transformation)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        # Eğer HDC aktifse, HSP (Hyperdimensional Sparse Projection) modülünü başlat
        if self.hdc:
            self.hsp = HSPProjector(out_features, hd_dim)

    def forward(self, g: GraphTensor) -> GraphTensor:
        # 1. Feature Projection (X * W)
        x_proj = torch.matmul(g.x, self.weight)

        # 2. SDR Masking & Triton SpMM (Message Passing)
        if self.sdr and g.sdr_mask is not None:
            # Triton kernel CSR format bekliyor (row_ptr, col_idx). 
            # g.edge_index'in 0. boyutu row_ptr, 1. boyutu col_idx olmalı.
            row_ptr, col_idx = g.edge_index[0], g.edge_index[1]
            
            # Eğer edge_weight yoksa 1.0 ile doldur
            values = g.edge_weight if g.edge_weight is not None else torch.ones_like(col_idx, dtype=x_proj.dtype)
            
            # Triton Kernel'i çağır
            out_x = spmm_triton_forward(row_ptr, col_idx, values, x_proj, g.sdr_mask)
        else:
            # Fallback (Gelecekte standart SpMM eklenebilir, şimdilik bypass)
            out_x = x_proj

        g.x = out_x

        # 3. HSP Projection & 4. HDC Binding
        if self.hdc:
            # Dense çıktıları Bipolar HD uzayına fırlat
            hd_new = self.hsp(g.x)
            
            # HDC Vector güncellemesi (XOR/Element-wise çarpım ile Bind)
            if g.hd_vector is not None:
                g.hd_vector = HDCModule.bind(g.hd_vector, hd_new)
            else:
                g.hd_vector = hd_new

        return g