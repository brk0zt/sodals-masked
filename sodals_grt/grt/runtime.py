import torch
from tensor import GraphTensor
from grt.dispatch import Dispatcher

class GraphRuntime:
    """
    Event-Driven Graph Scheduler.
    Sadece SDR maskesinde aktif olan düğümleri (nodes) CUDA stream üzerinden asenkron çalıştırır.
    """
    def __init__(self):
        self.dispatcher = Dispatcher()

    def run(self, g: GraphTensor, ops_schedule: list) -> GraphTensor:
        """
        Grafik üzerinde belirtilen operasyon listesini (IR planını) çalıştırır.
        """
        # Aktif düğümleri SDR maskesine göre tespit et
        if g.sdr_mask is not None:
            # Maske üzerinden aktif olanları bul (dim=1 boyunca toplamı > 0 olanlar)
            active_nodes = (g.sdr_mask.sum(dim=1) > 0)
            active_count = active_nodes.sum().item()
            total_count = g.x.size(0)
            print(f"[GraphRuntime] Yürütme başlıyor. Aktif Node: {active_count}/{total_count} (Sparsity: {100 - (active_count/total_count*100):.1f}%)")
        else:
            print("[GraphRuntime] Uyarı: SDR Maskesi bulunamadı. Tüm graph işlenecek.")
            active_nodes = None

        # Asenkron yürütme için izole bir CUDA stream oluştur
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            for op in ops_schedule:
                # Ops_schedule yapısı string veya dict olabilir (Örn: {"op": "spmm", "args": [...]})
                if isinstance(op, str):
                    op_name = op
                    op_args = []
                else:
                    op_name = op.get("name")
                    op_args = op.get("args", [])

                # İlgili operasyonu dispatcher üzerinden çalıştır
                # Şimdilik örnek bir genel routing mekanizması:
                if op_name == "spmm":
                    # SpMM CSR beklentilerine göre argümanları hazırla
                    g.x = self.dispatcher.dispatch(
                        "spmm", 
                        g.edge_index[0], g.edge_index[1], 
                        g.edge_weight if g.edge_weight is not None else torch.ones_like(g.edge_index[1], dtype=g.x.dtype),
                        g.x, 
                        g.sdr_mask
                    )
                elif op_name == "hdc_bind":
                    if g.hd_vector is not None and len(op_args) > 0:
                        g.hd_vector = self.dispatcher.dispatch("hdc_bind", g.hd_vector, op_args[0])

        # Ana thread'i CUDA stream'i bitene kadar beklet (Senkronizasyon)
        torch.cuda.current_stream().wait_stream(stream)
        
        return g