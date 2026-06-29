# sodals_grt/grt/compiler.py
import torch
import math
from tensor import GraphTensor

class SemanticGraphCompiler:
    """
    Phase 2: The Graph Compiler.
    Dış dünyadan gelen ham tensörleri ve grafikleri (Data), SODALS donanımının 
    anlayacağı Zero-Copy 'GraphTensor' formatına derler.
    Aynı zamanda donanım israfını önlemek için dinamik SDR (Sparse Distributed Representation) 
    maskelerini inşa eder.
    """
    def __init__(self, sparsity_threshold: float = 0.1):
        # Enerjisi bu değerin altında kalan düğümler maskelenecek (Pruning)
        self.sparsity_threshold = sparsity_threshold

    def _generate_sdr_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Düğümlerin (nodes) özellik tensörlerine bakarak deterministik bir SDR maskesi üretir.
        Gerçek BCI verisinde bu maske, EEG'den gelen "Semantic Divergence" veya 
        HDC Paradoks Filtresi sonuçlarına göre şekillenir.
        """
        # Node'ların L2 norm (enerji) büyüklüğünü ölç
        node_energies = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # Enerjisi eşik değerinden (sparsity_threshold) büyük olanları 1.0 yap, diğerlerini 0.0
        # Bu işlem Triton kernelindeki Warp-Scheduler'a "bu satırı atla" emrini verir.
        sdr_mask = torch.where(
            node_energies > self.sparsity_threshold, 
            torch.ones_like(node_energies), 
            torch.zeros_like(node_energies)
        )
        
        # Maske boyutunu feature_dim ile uyumlu hale getir (Broadcast)
        sdr_mask = sdr_mask.expand_as(x).contiguous()
        return sdr_mask

    def compile(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> GraphTensor:
        """
        Ham veriyi alır, doğrular, maskeler ve GraphTensor primitive'ine mühürler.
        """
        # 1. Bellek Bütünlüğü (Contiguous Memory) Doğrulaması
        # Triton kernelleri bellekte parçalanmış (strided) tensörlerde SegFault verir.
        if not x.is_contiguous():
            x = x.contiguous()
        if not edge_index.is_contiguous():
            edge_index = edge_index.contiguous()
        if edge_weight is not None and not edge_weight.is_contiguous():
            edge_weight = edge_weight.contiguous()

        # 2. Cihaz Eşleşmesi (Device Alignment)
        if x.device != edge_index.device:
            edge_index = edge_index.to(x.device)
        
        # 3. Dinamik SDR Maskesi Üretimi (Hardware Pruning)
        sdr_mask = self._generate_sdr_mask(x)

        # 4. Zero-Copy GraphTensor Mührü
        compiled_graph = GraphTensor(
            x=x,
            edge_index=edge_index,
            sdr_mask=sdr_mask,
            edge_weight=edge_weight,
            hd_vector=None # HDC projeksiyonu Conv katmanında yapılacak
        )

        active_nodes = (sdr_mask.sum(dim=1) > 0).sum().item()
        total_nodes = x.size(0)
        print(f"[GraphCompiler] Derleme Tamamlandı. {total_nodes} düğümün {active_nodes} tanesi aktif olarak işaretlendi.")

        return compiled_graph