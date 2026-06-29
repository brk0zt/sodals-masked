import torch
from kernels.spmm_triton import spmm_triton_forward
from hdc.hypervector import HDCModule

class Dispatcher:
    """
    Semantic Operation Routing.
    String operasyonları gerçek fonksiyonlara ve Triton kernellarına yönlendirir.
    """
    def __init__(self):
        self._registry = {}
        self._register_core_ops()

    def register(self, name: str, fn: callable):
        """Yeni bir operasyon kaydeder."""
        self._registry[name] = fn

    def _register_core_ops(self):
        """Çekirdek operasyonları (SpMM, HDC) default olarak kaydet."""
        self.register("spmm", spmm_triton_forward)
        self.register("hdc_bind", HDCModule.bind)
        self.register("hdc_bundle", HDCModule.bundle)

    def dispatch(self, op_name: str, *args, **kwargs):
        """Operasyonu bulur ve argümanlarla çalıştırır."""
        if op_name not in self._registry:
            raise ValueError(f"[Dispatcher] Kritik Hata: '{op_name}' operasyonu bulunamadı!")
        
        return self._registry[op_name](*args, **kwargs)