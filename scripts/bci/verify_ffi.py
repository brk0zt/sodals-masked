import os
import sys
import numpy as np

# Resolve DLL path
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)

try:
    import sodals_bci
    print("[SUCCESS] sodals_bci imported.")
except ImportError as e:
    print(f"[FAILED] sodals_bci import: {e}")
    sys.exit(1)

# Test with Numpy Heap Memory (CPU)
hidden_dim = 4096
print(f"[1] Allocating {hidden_dim} float32s in Numpy CPU...")
arr = np.ones(hidden_dim, dtype=np.float32)
ptr = arr.ctypes.data
print(f"[*] Numpy Address: {hex(ptr)}")
print(f"[*] Local Check (first 5): {arr[:5]}")

print("\n[2] Injecting Numpy pointer into Rust...")
sodals_bci.inject_ghost_tensor(ptr, hidden_dim)

print("\n[3] Requesting Rust sum verification...")
rust_sum = sodals_bci.verify_ghost_slot()
print(f"[*] Rust calculated sum: {rust_sum}")

if abs(rust_sum - hidden_dim) < 1e-4:
    print("\n[FINAL SUCCESS] BCI FFI BRIDGE VERIFIED!")
else:
    print(f"\n[FAILURE] FFI Bridge sync error. Expected {hidden_dim}, got {rust_sum}")
