#!/usr/bin/env python3
"""
SODALS Sentinel Python Köprüsü

Bu modül, PyO3 ile derlenmiş Rust SentinelGuard modülünü sarar.
Qwen 8B modelinin çıktı logitlerini ve sensör verisini
Rust tarafında doğrulamak için kullanılır.

Kullanım:
    from sentinel_bridge import SentinelBridge
    bridge = SentinelBridge("/opt/sodals/qmem/reference.qmem")
    decision = bridge.evaluate(sensor_data, model_logits)

Determinizm:
    - Python tarafındaki tüm hesaplama O(1) sabit boyutlu vektörler üzerindedir.
    - Ağır FFT ve model hesaplamaları GPU üzerinde (Non-RT çekirdeklerde) çalışır.
"""

import os
import struct
from typing import Optional, Tuple

# PyO3 derlemesi sonrası import edilecek modül.
# Derleme: maturin develop --features python
# veya: cargo build --release --features python
try:
    import sodals_industrial
except ImportError:
    sodals_industrial = None
    print("UYARI: sodals_industrial Rust modülü bulunamadı. Sentinel pasif modda çalışıyor.")


class SentinelBridge:
    """
    Python tarafında çalışan Sentinel Guard sarici.
    .qmem referans imzasını yükler ve AI çıktılarını doğrular.
    """

    REF_SIZE = 544  # 32 byte header + 512 byte (128 x f32)
    SENSOR_SIZE = 24  # sizeof(SensorPayload)
    LOGIT_COUNT = 8

    def __init__(self, qmem_ref_path: str, threshold_db: float = -40.0, debounce: int = 3):
        self._ref_path = qmem_ref_path
        self._threshold_db = threshold_db
        self._debounce = debounce
        self._guard = None

        if sodals_industrial is not None and os.path.exists(qmem_ref_path):
            with open(qmem_ref_path, "rb") as f:
                ref_bytes = f.read()
            if len(ref_bytes) != self.REF_SIZE:
                raise ValueError(f"Referans .qmem {self.REF_SIZE} byte olmali, okunan: {len(ref_bytes)}")
            # Rust SentinelGuardPy örneği oluştur
            self._guard = sodals_industrial.SentinelGuardPy(
                ref_bytes, threshold_db, debounce
            )
        else:
            print("Sentinel pasif: Rust modülü veya referans dosyası eksik.")

    def evaluate(self, sensor_dict: dict, logits: list) -> Tuple[str, Optional[str]]:
        """
        Sensör verisi ve model logitlerini Sentinel'e sunar.

        Args:
            sensor_dict: {
                "timestamp_ns": int,
                "temperature": float,
                "pressure": float,
                "motor_rpm": float,
                "status_mask": int,
                "fault_code": int,
            }
            logits: 8 elemanlı float listesi

        Returns:
            ("ALLOW" | "HOLD" | "REJECT", reason | None)
        """
        if self._guard is None:
            # Pasif mod: her zaman izin ver
            return "ALLOW", None

        # Python dict -> SensorPayload binary (little-endian)
        sensor_bytes = struct.pack(
            "<QfffBB2x",
            sensor_dict.get("timestamp_ns", 0),
            sensor_dict.get("temperature", 0.0),
            sensor_dict.get("pressure", 0.0),
            sensor_dict.get("motor_rpm", 0.0),
            sensor_dict.get("status_mask", 0),
            sensor_dict.get("fault_code", 0),
        )
        if len(sensor_bytes) != self.SENSOR_SIZE:
            raise ValueError(f"Sensor serileştirme hatasi: {len(sensor_bytes)}")

        if len(logits) != self.LOGIT_COUNT:
            raise ValueError(f"Logit vektörü {self.LOGIT_COUNT} elemanli olmali")

        logits_arr = tuple(float(x) for x in logits[:self.LOGIT_COUNT])

        # 1. AI çıktı validasyonu
        safe = self._guard.validate_ai_output(logits_arr, sensor_bytes)
        if not safe:
            return "REJECT", "ModelInstability"

        # 2. Frekans imza kontrolü (opsiyonel — sensör verisi imzaya dönüştürülmüşse)
        # TODO: Sensör verisini FFT ile 128-bant imzaya çevir ve evaluate et.
        # Şimdilik model validasyonu yeterli.

        return "ALLOW", None

    def passive_evaluate(self, sensor_dict: dict) -> bool:
        """
        Sadece fiziksel limit kontrolü (Python saf kod).
        Rust modülü olmadığında veya hızlı ön filtreleme için.
        O(1), heap allocasyonu yok.
        """
        temp = sensor_dict.get("temperature", 0.0)
        pressure = sensor_dict.get("pressure", 0.0)
        # Fiziksel sınırlar
        if temp > 200.0 or pressure > 50.0:
            return False
        if sensor_dict.get("status_mask", 0) == 0xFF:
            return False
        return True


# Doğrudan çalıştırma testi
if __name__ == "__main__":
    bridge = SentinelBridge("/opt/sodals/qmem/reference.qmem")
    sensor = {
        "timestamp_ns": 123456789000,
        "temperature": 45.5,
        "pressure": 2.1,
        "motor_rpm": 1500.0,
        "status_mask": 0,
        "fault_code": 0,
    }
    logits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    decision, reason = bridge.evaluate(sensor, logits)
    print(f"Sentinel karar: {decision}, sebep: {reason}")
