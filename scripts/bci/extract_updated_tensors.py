#!/usr/bin/env python3
"""
Updated BCI Tensor Extractor for ds003626-master
Extracts EEG and sEMG (EXG) data from .fif epoch files to .safetensors format

Usage:
    python extract_updated_tensors.py [subject_id] [session_id]
    python extract_updated_tensors.py --all

Output:
    inner_speech_sub-01_ses-01.safetensors - EEG data [epochs, 128, 256]
    real_subvocal_semg_sub-01_ses-01.safetensors - sEMG data [epochs, 8, 256]
"""

import mne
import numpy as np
from safetensors.numpy import save_file
from pathlib import Path
import sys
import os

# Updated data location
BASE_PATH = Path("C:/Users/Burak/Desktop/veriler/ds003626-master/derivatives")

# Hardware constants
EEG_CHANNELS = 128  # First 128 channels are EEG
EXG_CHANNELS = 8    # Channels 128-135 are EXG (sEMG + EOG)
TOTAL_CHANNELS = 136

def extract_session(subject_id: str, session_id: str) -> tuple:
    """
    Extract EEG and sEMG data from a single session.
    
    Returns:
        (eeg_file_path, semg_file_path) or (None, None) if files not found
    """
    session_path = BASE_PATH / subject_id / session_id
    
    # Check for required files
    eeg_fif = session_path / f"{subject_id}_{session_id}_eeg-epo.fif"
    exg_fif = session_path / f"{subject_id}_{session_id}_exg-epo.fif"
    events_dat = session_path / f"{subject_id}_{session_id}_events.dat"
    
    if not eeg_fif.exists():
        print(f"[EXTRACTOR] WARNING: EEG file not found: {eeg_fif}")
        return None, None
    
    if not exg_fif.exists():
        print(f"[EXTRACTOR] WARNING: EXG file not found: {exg_fif}")
        return None, None
    
    print(f"\n[EXTRACTOR] Processing {subject_id}/{session_id}...")
    
    # Load EEG epochs
    print(f"[EXTRACTOR] Loading EEG: {eeg_fif.name}")
    eeg_epochs = mne.read_epochs(str(eeg_fif), preload=True, verbose=False)
    eeg_data = eeg_epochs.get_data()  # Shape: [epochs, 128, time]
    
    # Load EXG epochs (contains sEMG channels)
    print(f"[EXTRACTOR] Loading EXG: {exg_fif.name}")
    exg_epochs = mne.read_epochs(str(exg_fif), preload=True, verbose=False)
    exg_data = exg_epochs.get_data()  # Shape: [epochs, 8, time]
    
    # Ensure consistent shapes
    n_epochs = min(eeg_data.shape[0], exg_data.shape[0])
    
    # Get event labels from epochs metadata
    try:
        events = exg_epochs.events  # Shape: [n_events, 3] - [sample, 0, trigger_code]
        if len(events) >= n_epochs:
            labels = events[:n_epochs, 2].astype(np.int64)  # Third column is trigger code
        else:
            labels = np.zeros(n_epochs, dtype=np.int64)
            labels[:len(events)] = events[:, 2]
    except Exception as e:
        print(f"[EXTRACTOR] WARNING: Could not extract events: {e}, using zeros")
        labels = np.zeros(n_epochs, dtype=np.int64)
    
    # Slice to consistent shape
    eeg_data = eeg_data[:n_epochs]
    exg_data = exg_data[:n_epochs]
    labels = labels[:n_epochs]
    
    # Resample to 256 samples if needed (Nieto dataset is typically 256Hz)
    target_samples = 256
    if eeg_data.shape[2] != target_samples:
        print(f"[EXTRACTOR] Resampling from {eeg_data.shape[2]} to {target_samples} samples")
        # Simple decimation or interpolation could be done here
        # For now, truncate/pad
        if eeg_data.shape[2] > target_samples:
            eeg_data = eeg_data[:, :, :target_samples]
            exg_data = exg_data[:, :, :target_samples]
        else:
            pad_width = target_samples - eeg_data.shape[2]
            eeg_data = np.pad(eeg_data, ((0,0), (0,0), (0,pad_width)), mode='edge')
            exg_data = np.pad(exg_data, ((0,0), (0,0), (0,pad_width)), mode='edge')
    
    # Convert to float32 for Candle compatibility
    eeg_data = eeg_data.astype(np.float32)
    exg_data = exg_data.astype(np.float32)
    labels = labels.astype(np.int64)
    
    print(f"[EXTRACTOR] EEG shape: {eeg_data.shape}, dtype: {eeg_data.dtype}")
    print(f"[EXTRACTOR] EXG shape: {exg_data.shape}, dtype: {exg_data.dtype}")
    print(f"[EXTRACTOR] Labels: {len(labels)} samples, unique: {np.unique(labels)}")
    
    # Save EEG safetensors
    eeg_output = f"inner_speech_{subject_id}_{session_id}.safetensors"
    eeg_tensors = {
        "eeg_waves": eeg_data,  # [epochs, 128, 256]
        "labels": labels,
    }
    save_file(eeg_tensors, eeg_output)
    eeg_size = os.path.getsize(eeg_output) / (1024 * 1024)
    print(f"[EXTRACTOR] EEG saved: {eeg_output} ({eeg_size:.2f} MB)")
    
    # Save sEMG safetensors
    semg_output = f"real_subvocal_semg_{subject_id}_{session_id}.safetensors"
    semg_tensors = {
        "semg_waves": exg_data,  # [epochs, 8, 256]
        "labels": labels,
    }
    save_file(semg_tensors, semg_output)
    semg_size = os.path.getsize(semg_output) / (1024 * 1024)
    print(f"[EXTRACTOR] sEMG saved: {semg_output} ({semg_size:.2f} MB)")
    
    return eeg_output, semg_output

def extract_all():
    """Extract all available subjects and sessions."""
    if not BASE_PATH.exists():
        print(f"[EXTRACTOR] ERROR: Base path not found: {BASE_PATH}")
        sys.exit(1)
    
    subject_dirs = sorted(BASE_PATH.glob("sub-*"))
    
    if not subject_dirs:
        print(f"[EXTRACTOR] ERROR: No subject directories found in {BASE_PATH}")
        sys.exit(1)
    
    print(f"[EXTRACTOR] Found {len(subject_dirs)} subjects")
    
    all_files = []
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        session_dirs = sorted(subject_dir.glob("ses-*"))
        
        for session_dir in session_dirs:
            session_id = session_dir.name
            
            try:
                eeg_file, semg_file = extract_session(subject_id, session_id)
                if eeg_file:
                    all_files.append(eeg_file)
                if semg_file:
                    all_files.append(semg_file)
            except Exception as e:
                print(f"[EXTRACTOR] ERROR processing {subject_id}/{session_id}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n[EXTRACTOR] Total files created: {len(all_files)}")
    for f in sorted(all_files):
        print(f"  - {f}")
    
    return all_files

def main():
    print("=" * 60)
    print("🧠 SODALS Updated BCI Tensor Extractor")
    print(f"📁 Data source: {BASE_PATH}")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        extract_all()
    elif len(sys.argv) >= 3:
        subject_id = sys.argv[1]
        session_id = sys.argv[2]
        extract_session(subject_id, session_id)
    else:
        # Default: extract sub-01/ses-01
        print("\n[EXTRACTOR] Default mode: extracting sub-01/ses-01")
        print("[EXTRACTOR] Usage:")
        print("  python extract_updated_tensors.py --all")
        print("  python extract_updated_tensors.py sub-01 ses-01")
        print()
        extract_session("sub-01", "ses-01")

if __name__ == "__main__":
    main()
