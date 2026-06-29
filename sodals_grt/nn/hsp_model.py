# sodals_grt/nn/hsp_model.py
import torch
import torch.nn as nn

from tensor import GraphTensor
from grt.compiler import SemanticGraphCompiler
from grt.runtime import GraphRuntime
from nn.conv import GraphSemanticConv
from hdc.hypervector import HDCModule

class SODALSHybridPipeline(nn.Module):
    """
    Phase 3: HSP Model Pipeline (Uçtan Uca Nörosembolik Ağ)
    
    Bu mimari, BCI'dan (veya OEDB testinden) gelen ham veriyi alır,
    Derleyici (Compiler) ile zero-copy GraphTensor'a dönüştürür,
    Triton tabanlı asenkron Runtime üzerinde işler ve
    Holografik uzaydaki kararları standart logitlere (çıktılara) yansıtır.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hd_dim: int = 1024, sparsity_threshold: float = 0.1):
        super().__init__()
        
        # 1. L1.5: Derleyici (SDR Maskesi ve Zero-Copy mühürleme)
        self.compiler = SemanticGraphCompiler(sparsity_threshold=sparsity_threshold)
        
        # 2. L1: Asenkron Yürütme Motoru (Hardware Scheduler)
        self.runtime = GraphRuntime()
        
        # 3. L0: Nörosembolik Konvolüsyon Katmanları (Hybrid CVNN & GNN)
        self.layer1 = GraphSemanticConv(in_dim, hidden_dim, sdr=True, hdc=True, hd_dim=hd_dim)
        self.layer2 = GraphSemanticConv(hidden_dim, hidden_dim, sdr=True, hdc=True, hd_dim=hd_dim)
        
        # 4. Decoder / Readout (HDC Uzayından -> Eylem/Karar Uzayına)
        # BCI sinyalinin en son ulaştığı "Karar" kapısı.
        self.readout = nn.Sequential(
            nn.Linear(hd_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, x_raw: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Kritik Yürütme Yolu (Critical Execution Path).
        """
        # ---------------------------------------------------------
        # ADIM 1: DERLEME (COMPILATION & PRUNING)
        # Ham veri, bellekte bütün (contiguous) hale getirilir ve
        # gürültülü düğümler SDR maskesi ile donanım seviyesinde kapatılır.
        # ---------------------------------------------------------
        g = self.compiler.compile(x_raw, edge_index, edge_weight)

        # ---------------------------------------------------------
        # ADIM 2: NÖROSEMBOLİK AKIŞ (LATENT FLOW)
        # Asenkron CUDA stream içinde SpMM (Triton) ve HDC Bind çalışır.
        # ---------------------------------------------------------
        # Runtime'ın izole stream'i içinde katmanları yürütüyoruz.
        # Bu, PyTorch'un ana thread'ini (Python GIL) bloke etmez.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            g = self.layer1(g)
            g = self.layer2(g)
        
        # Sonucu ana stream ile senkronize et
        torch.cuda.current_stream().wait_stream(stream)

        # ---------------------------------------------------------
        # ADIM 3: HOLOGRAFİK SÜPERPOZİSYON (GLOBAL POOLING)
        # Çizgedeki tüm düğümlerin fikirlerini tek bir Kuantum Vektöründe birleştir.
        # Sadece "aktif" olan (prune edilmemiş) düğümler oylamaya katılır.
        # ---------------------------------------------------------
        if g.hd_vector is not None and g.sdr_mask is not None:
            # Düğüm bazlı aktiflik kontrolü (Herhangi bir feature'u > 0 olanlar)
            active_nodes_mask = (g.sdr_mask.sum(dim=1) > 0).unsqueeze(-1)
            
            # Maskelenmiş (Pruned) düğümlerin HDC vektörlerini sıfırla
            valid_hd_vectors = g.hd_vector * active_nodes_mask.float()
            
            # Tüm geçerli düğümlerin Bipolar Bundle'ı (Çoğunluk oylaması)
            # Bu işlem grafiğin tamamını temsil eden tek bir 1024-D (hd_dim) vektör üretir.
            global_hd_state = HDCModule.bundle(valid_hd_vectors, dim=0).unsqueeze(0)
        else:
            # Fallback
            global_hd_state = g.x.mean(dim=0, keepdim=True)

        # ---------------------------------------------------------
        # ADIM 4: KARAR İNFAZI (READOUT)
        # ---------------------------------------------------------
        logits = self.readout(global_hd_state)
        
        return logits