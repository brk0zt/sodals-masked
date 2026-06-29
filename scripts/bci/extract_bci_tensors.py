#!/usr/bin/env python3
"""
BCI EEG Veri Dönüştürücü
.fif (MNE) formatındaki EEG verilerini .safetensors formatına çevirir.

Kullanım:
    python extract_bci_tensors.py

Çıktı:
    inner_speech_sub01.safetensors - [samples, channels, time] formatında EEG verisi
"""

import mne
import numpy as np
from safetensors.numpy import save_file
from pathlib import Path
import sys

def extract_subject(subject_id: str = "sub-01", session_id: str = "ses-01") -> str:
    """
    Belirli bir denek ve oturumdan EEG verisini çıkarır.
    
    Returns:
        Oluşturulan dosya yolu
    """
    # Veri yolu (Desktop/veriler/ dizininden)
    # BDF dosya yolu (raw EEG data)
    base_path = Path("C:/Users/Burak/Desktop/veriler/ds003626-master")
    bdf_path = base_path / subject_id / session_id / "eeg" / f"{subject_id}_{session_id}_task-innerspeech_eeg.bdf"
    
    if not bdf_path.exists():
        print(f"[HATA] Dosya bulunamadı: {bdf_path}")
        print("[BİLGİ] Lütfen veri setinin doğru konumda olduğundan emin olun.")
        sys.exit(1)
    
    print(f"[BCI EXTRACTOR] {bdf_path} okunuyor...")
    
    # BDF dosyasını oku (BioSemi format)
    try:
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"[HATA] Raw BDF okuma başarısız: {e}")
        sys.exit(1)
    else:
        # Sadece EEG kanallarını al
        raw.pick_types(eeg=True)
        
        # Veriyi segmentlere ayır (256 örnek = ~1 saniye @ 256Hz)
        sfreq = raw.info['sfreq']
        n_channels = len(raw.ch_names)
        window_samples = 256
        
        # Ham veriyi al
        data = raw.get_data()  # shape: (n_channels, n_times)
        n_times = data.shape[1]
        
        # Kayan pencere ile epochlar oluştur (50% overlap)
        hop = window_samples // 2
        n_epochs = (n_times - window_samples) // hop + 1
        
        eeg_data = np.zeros((n_epochs, n_channels, window_samples), dtype=np.float32)
        labels = np.zeros(n_epochs, dtype=np.int64)
        
        for i in range(n_epochs):
            start = i * hop
            end = start + window_samples
            eeg_data[i] = data[:, start:end]
            labels[i] = (i % 10)  # Pseudo-etiketler (gerçek etiketler için events gerekli)
    
    n_samples, n_channels, n_times = eeg_data.shape
    print(f"[BCI EXTRACTOR] Ham EEG verisi: {eeg_data.shape}")
    print(f"[BCI EXTRACTOR] Örnek sayısı: {n_samples}")
    print(f"[BCI EXTRACTOR] Kanal sayısı: {n_channels}")
    print(f"[BCI EXTRACTOR] Zaman noktası: {n_times}")
    print(f"[BCI EXTRACTOR] Etiket dağılımı: {np.unique(labels, return_counts=True)}")
    
    # Safetensors formatına kaydet (Candle uyumlu)
    output_file = f"inner_speech_{subject_id}.safetensors"
    
    tensors = {
        "eeg_waves": eeg_data.astype(np.float32),
        "labels": labels.astype(np.int64)  # Candle i64 bekler
    }
    
    save_file(tensors, output_file)
    print(f"[BCI EXTRACTOR] ✅ Başarılı: {output_file}")
    print(f"[BCI EXTRACTOR]   - Boyut: {eeg_data.shape}")
    print(f"[BCI EXTRACTOR]   - Bellek: {eeg_data.nbytes / 1024 / 1024:.2f} MB")
    
    return output_file

def extract_all_subjects():
    """Tüm denekleri çıkar"""
    base_path = Path("veriler/ds003626-master/derivatives")
    
    if not base_path.exists():
        print(f"[HATA] Veri dizini bulunamadı: {base_path}")
        print("[BİLGİ] Lütfen veri setini indirip 'veriler/' dizinine çıkarın.")
        sys.exit(1)
    
    # Tüm sub-* dizinlerini bul
    subject_dirs = sorted(base_path.glob("sub-*"))
    
    if not subject_dirs:
        print("[HATA] Hiç denek dizini bulunamadı!")
        sys.exit(1)
    
    created_files = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        # Her denek için tüm oturumları bul
        session_dirs = sorted(subject_dir.glob("ses-*"))
        
        for session_dir in session_dirs:
            session_id = session_dir.name
            
            try:
                output = extract_subject(subject_id, session_id)
                created_files.append(output)
            except Exception as e:
                print(f"[UYARI] {subject_id}/{session_id} atlandı: {e}")
    
    print(f"\n[BCI EXTRACTOR] Toplam {len(created_files)} dosya oluşturuldu:")
    for f in created_files:
        print(f"  - {f}")
    
    return created_files

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 SODALS BCI EEG Tensor Extractor")
    print("=" * 60)
    
    # Argüman kontrolü
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        extract_all_subjects()
    else:
        # Sadece ilk denek (varsayılan)
        output = extract_subject()
        print(f"\n[BİLGİ] Tüm denekleri çıkarmak için: python extract_bci_tensors.py --all")
