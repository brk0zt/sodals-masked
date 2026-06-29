#!/usr/bin/env python3
"""
Test için sentetik EEG verisi oluşturur.
Gerçek veri yerine hemen BCI entegrasyonunu test etmek için kullanılır.
"""

import numpy as np
from safetensors.numpy import save_file
from pathlib import Path

def create_mock_eeg_data(
    n_samples: int = 100,
    n_channels: int = 64,
    n_times: int = 256,
    output_path: str = "inner_speech_sub-01.safetensors"
):
    """
    Sentetik EEG verisi oluşturur.
    
    Format:
    - eeg_waves: [n_samples, n_channels, n_times] - EEG epochları
    - labels: [n_samples] - Sınıf etiketleri (0-9 arası kelimeler)
    """
    print(f"[MOCK BCI] Sentetik EEG verisi oluşturuluyor...")
    print(f"  - Örnek sayısı: {n_samples}")
    print(f"  - Kanal sayısı: {n_channels}")
    print(f"  - Zaman noktası: {n_times}")
    
    # Her etiket için farklı frekans deseni oluştur
    # 0: Yukarı, 1: Aşağı, 2: Sağ, 3: Sol, vb.
    eeg_data = np.zeros((n_samples, n_channels, n_times), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        label = i % 10  # 10 farklı sınıf
        labels[i] = label
        
        # Her sınıf için karakteristik frekans deseni
        for ch in range(n_channels):
            t = np.linspace(0, 1, n_times)
            
            # Temel sinüs sinyali (8-12 Hz alpha bandı - odaklanma)
            alpha = np.sin(2 * np.pi * 10 * t) * (0.5 + label * 0.05)
            
            # Beta aktivitesi (13-30 Hz - motor imgeleme)
            beta = np.sin(2 * np.pi * 20 * t) * (0.3 + (ch % 5) * 0.02)
            
            # Gürültü
            noise = np.random.randn(n_times) * 0.1
            
            # Kanal konumuna göre genlik ayarı (frontal kanallar daha güçlü)
            if ch < 16:  # Frontal
                amp = 1.2
            elif ch < 32:  # Temporal
                amp = 1.0
            elif ch < 48:  # Parietal
                amp = 0.9
            else:  # Oksipital
                amp = 0.8
            
            eeg_data[i, ch, :] = (alpha + beta + noise) * amp * 1e-6  # Mikrovolt cinsinden
    
    # Safetensors formatına kaydet
    tensors = {
        "eeg_waves": eeg_data,
        "labels": labels
    }
    
    save_file(tensors, output_path)
    
    print(f"[MOCK BCI] ✅ Başarılı: {output_path}")
    print(f"[MOCK BCI]   - Boyut: {eeg_data.shape}")
    print(f"[MOCK BCI]   - Bellek: {eeg_data.nbytes / 1024 / 1024:.2f} MB")
    print(f"[MOCK BCI]   - Etiket dağılımı: {np.bincount(labels)}")
    
    return output_path

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 SODALS Mock BCI Data Generator")
    print("=" * 60)
    
    # Varsayılan parametrelerle oluştur
    output = create_mock_eeg_data(
        n_samples=100,      # 100 örnek epoch
        n_channels=64,    # 64 kanal
        n_times=256,      # 256 zaman noktası (~1 saniye @ 256Hz)
        output_path="inner_speech_sub-01.safetensors"
    )
    
    print(f"\n[BİLGİ] Gerçek EEG verisi için:")
    print(f"  1. DataLad ile veri setini indirin:")
    print(f"     datalad get veriler/ds003626-master/sub-01/ses-01/eeg/")
    print(f"  2. Veya OpenNeuro'dan doğrudan indirin:")
    print(f"     https://openneuro.org/datasets/ds003626")
