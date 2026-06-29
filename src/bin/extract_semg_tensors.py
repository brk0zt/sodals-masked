#!/usr/bin/env python3
# ============================================================================
# sEMG EXTRACTOR — Real 8-Channel EXG Extraction from Nieto Dataset
# ============================================================================
#
# Hardware Truth: The Nieto ds003626 dataset has 136 channels:
#   - Channels [0..128]  = 128 EEG (cortical, for QMEM topology)
#   - Channels [128..136] = 8 EXG (jaw/throat sEMG + EOG artifacts)
#
# This script extracts the 8 EXG channels from the already-compiled
# inner_speech_sub-01.safetensors (which contains ALL 136 channels)
# and saves them as a separate real_subvocal_semg.safetensors.
#
# NO DUMMY DATA. NO RANDOM NOISE. ONLY REAL HARDWARE VOLTAGES.

import numpy as np
from safetensors.numpy import load_file, save_file
import sys
import os

# ============================================================================
# Configuration
# ============================================================================

# Source: the full 136-channel safetensors (already extracted from .fif/.bdf)
SOURCE_FILE = "inner_speech_sub-02_ses-01.safetensors"

# Output: 8-channel EXG-only safetensors for sEMG pipeline
OUTPUT_FILE = "real_subvocal_semg_sub02.safetensors"

# Hardware channel map
EEG_CHANNELS = 128  # Channels [0..128) — cortical EEG
EXG_CHANNELS = 8    # Channels [128..136) — jaw/throat sEMG + EOG
TOTAL_CHANNELS = EEG_CHANNELS + EXG_CHANNELS  # 136

# ============================================================================
# Extraction
# ============================================================================

def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"[sEMG EXTRACTOR] FATAL: {SOURCE_FILE} bulunamadı!", file=sys.stderr)
        print(f"[sEMG EXTRACTOR] Önce inner_speech_sub-01.safetensors dosyasının mevcut olduğundan emin ol.", file=sys.stderr)
        sys.exit(1)

    print(f"[sEMG EXTRACTOR] Kaynak: {SOURCE_FILE}")
    print(f"[sEMG EXTRACTOR] Donanım Gerçeği: {TOTAL_CHANNELS} kanal = {EEG_CHANNELS} EEG + {EXG_CHANNELS} EXG")

    # Load full dataset
    data = load_file(SOURCE_FILE)
    eeg_waves = data["eeg_waves"]  # Shape: [epochs, 136, 256]
    labels = data["labels"]        # Shape: [epochs]

    print(f"[sEMG EXTRACTOR] Tam tensör boyutu: {eeg_waves.shape} ({eeg_waves.dtype})")
    print(f"[sEMG EXTRACTOR] Etiketler: {labels.shape} ({labels.dtype})")

    # Validate channel count
    num_epochs, num_channels, time_steps = eeg_waves.shape
    if num_channels < 8:
        print(f"[sEMG EXTRACTOR] FATAL: En az 8 kanal gerekli.", file=sys.stderr)
        sys.exit(1)

    # SURGICAL SLICE: Extract last 8 channels as EXG
    exg_data = eeg_waves[:, -EXG_CHANNELS:, :]
    print(f"[sEMG EXTRACTOR] EXG slice: son {EXG_CHANNELS} kanal")
    print(f"[sEMG EXTRACTOR] Çıkarılan sEMG tensör boyutu: {exg_data.shape}")

    # Sanity check: EXG data should NOT be all zeros
    exg_power = np.mean(exg_data ** 2)
    print(f"[sEMG EXTRACTOR] EXG Ortalama Güç (RMS²): {exg_power:.6e} V²")
    if exg_power < 1e-20:
        print("[sEMG EXTRACTOR] UYARI: EXG kanallarında sinyal yok! Veriler sıfır olabilir.", file=sys.stderr)

    # Print per-channel statistics
    print(f"\n[sEMG EXTRACTOR] === Kanal Bazlı İstatistikler ===")
    for ch in range(exg_data.shape[1]):
        ch_data = exg_data[:, ch, :]
        ch_rms = np.sqrt(np.mean(ch_data ** 2))
        ch_min = np.min(ch_data)
        ch_max = np.max(ch_data)
        print(f"  EXG-{ch} (hw:{EEG_CHANNELS + ch}): RMS={ch_rms:.6e} V | Min={ch_min:.6e} | Max={ch_max:.6e}")

    # Save as safetensors
    tensors = {
        "semg_waves": exg_data.astype(np.float32),  # [epochs, 8, 256]
        "labels": labels,                             # [epochs]
    }

    save_file(tensors, OUTPUT_FILE)
    output_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print(f"\n[sEMG EXTRACTOR] [OK] Infaz tamamlandi.")
    print(f"[sEMG EXTRACTOR]   Çıktı: {OUTPUT_FILE}")
    print(f"[sEMG EXTRACTOR]   Boyut: {output_size_mb:.2f} MB")
    print(f"[sEMG EXTRACTOR]   Shape: {exg_data.shape} ({exg_data.dtype})")
    print(f"[sEMG EXTRACTOR]   Etiket sayısı: {len(labels)}")

if __name__ == "__main__":
    main()