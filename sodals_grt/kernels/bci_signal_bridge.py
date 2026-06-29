# bci_signal_bridge.py
# PHASE 62.1: The Telepathic Bridge (LSL to Quantum Phasor Pipeline)
# Captures raw EEG voltage via Lab Streaming Layer, applies Hilbert Transform,
# and outputs continuous torch.complex64 phasors for the SODALS HDC Engine.

import numpy as np
import torch
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import hilbert, butter, filtfilt

class BCIQuantumBridge:
    def __init__(self, stream_type='EEG', channels=64, sample_rate=250):
        self.channels = channels
        self.sample_rate = sample_rate
        
        # Bandpass filter constraints for Alpha/Theta waves (Cognitive workload)
        # We target 4.0 Hz to 12.0 Hz to filter out muscle noise and power line artifacts
        nyquist = 0.5 * self.sample_rate
        self.low = 4.0 / nyquist
        self.high = 12.0 / nyquist
        self.b, self.a = butter(4, [self.low, self.high], btype='band')
        
        print(f"[BCI BRIDGE] Resolving LSL stream of type '{stream_type}'...")
        streams = resolve_byprop('type', stream_type, timeout=5.0)
        
        if not streams:
            print("[BCI BRIDGE] WARNING: No physical EEG stream found. Falling back to MOCK mode.")
            self.inlet = None
        else:
            self.inlet = StreamInlet(streams[0])
            print(f"[BCI BRIDGE] LSL Stream locked. Connected to {streams[0].name()}")

    def _mock_eeg_chunk(self, chunk_size=32):
        """Generates mock multi-channel EEG data if physical device is absent."""
        t = np.linspace(0, 1, chunk_size)
        # Mock a 10Hz Alpha wave with some random Gaussian noise
        base_wave = np.sin(2 * np.pi * 10 * t)
        mock_data = np.tile(base_wave, (self.channels, 1)) + np.random.normal(0, 0.5, (self.channels, chunk_size))
        return mock_data.T # Shape: (chunk_size, channels)

    def fetch_and_transform(self, chunk_size=32) -> torch.Tensor:
        """
        1. Fetches raw EEG chunk.
        2. Applies Bandpass Filter (Noise reduction).
        3. Applies Hilbert Transform to generate Analytical Signal.
        4. Returns torch.complex64 Phasor Tensor for Triton/Rust processing.
        """
        # Step 1: Data Acquisition
        if self.inlet:
            chunk, timestamps = self.inlet.pull_chunk(max_samples=chunk_size)
            if not chunk:
                return None
            raw_data = np.array(chunk) # Shape: (chunk_size, channels)
        else:
            raw_data = self._mock_eeg_chunk(chunk_size)

        # Transpose for scipy processing: (channels, chunk_size)
        raw_data = raw_data.T 

        # Step 2: Temporal Bandpass Filtering (Isolating cognitive bands)
        filtered_data = filtfilt(self.b, self.a, raw_data, axis=-1)

        # Step 3: The Hilbert Transform (Voltage -> Quantum State)
        # z(t) = s(t) + i * H{s(t)}
        analytic_signal = hilbert(filtered_data, axis=-1)

        # Extract the latest instantaneous state (the "now" moment for the AI)
        latest_state = analytic_signal[:, -1] # Shape: (channels,)

        # Step 4: Cast to PyTorch Complex64 Tensor (Zero-copy ready for Triton/Rust)
        # Real part = Amplitude projection, Imaginary part = Phase angle projection
        phasor_tensor = torch.tensor(latest_state, dtype=torch.complex64)
        
        return phasor_tensor

# =====================================================================
# HARDWARE DIAGNOSTICS & EXECUTION
# =====================================================================
if __name__ == "__main__":
    bridge = BCIQuantumBridge(stream_type='EEG', channels=64)
    
    print("[SYSTEM] Fetching instantaneous brain state...")
    quantum_phasor = bridge.fetch_and_transform()
    
    if quantum_phasor is not None:
        amplitudes = torch.abs(quantum_phasor)
        phases = torch.angle(quantum_phasor)
        
        print(f"[*] Phasor Tensor Shape: {quantum_phasor.shape} (DType: {quantum_phasor.dtype})")
        print(f"[*] Mean Amplitude (Energy): {amplitudes.mean().item():.4f}")
        print(f"[*] Mean Phase Shift: {phases.mean().item():.4f} radians")
        print("[*] Brain-Computer Interface successfully mapped to Kineto-Symbolic Space.")