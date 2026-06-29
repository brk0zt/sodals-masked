// src/kernels/host_memory.cpp
// Host-side CUDA memory management — Double-Buffer pattern (VRAM-backed)
//
// ARCHITECTURE: CPU prepares layer N+1 impedance in pinned host RAM,
// then cudaMemcpyAsync DMA-copies it into a pre-allocated VRAM buffer
// BEFORE the GPU kernel for layer N+1 runs.
// The kernel reads from VRAM (1000+ GB/s), never over PCIe (32 GB/s).
//
// OLD (ZeroCopy/PCIe) path was:
//   cudaHostAllocMapped -> cudaHostGetDevicePointer -> kernel reads over PCIe
// NEW (Double-Buffer) path:
//   cudaHostAllocDefault (pinned staging) -> cudaMemcpyAsync to VRAM -> kernel
//   reads VRAM

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>


// ============================================================================
// STREAM POOL (Non-blocking CUDA streams)
// ============================================================================

class AsyncStreamPool {
private:
  cudaStream_t *streams;
  int num_streams;
  int current_stream_idx;

public:
  AsyncStreamPool(int n) : num_streams(n), current_stream_idx(0) {
    streams = new cudaStream_t[n];
    for (int i = 0; i < n; i++) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
  }

  ~AsyncStreamPool() {
    for (int i = 0; i < num_streams; i++) {
      cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
  }

  cudaStream_t get_next_stream() {
    cudaStream_t s = streams[current_stream_idx];
    current_stream_idx = (current_stream_idx + 1) % num_streams;
    return s;
  }

  void synchronize_all() {
    for (int i = 0; i < num_streams; i++) {
      cudaStreamSynchronize(streams[i]);
    }
  }

  cudaStream_t get_stream(int idx) { return streams[idx % num_streams]; }
};

// ============================================================================
// PINNED ALLOCATOR — Standard pinned host RAM (no mapped/zero-copy flag)
// Enables fast cudaMemcpyAsync DMA without PCIe direct-access penalty.
// ============================================================================

class PinnedAllocator {
public:
  // Allocate write-combining pinned host memory for fast H→D DMA staging
  static void *allocate_pinned(size_t bytes) {
    void *ptr = nullptr;
    // cudaHostAllocWriteCombined: host writes are uncached (faster for staging)
    // Do NOT use cudaHostAllocMapped — that causes PCIe reads from the kernel.
    cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined);
    return ptr;
  }

  static void free_pinned(void *ptr) {
    if (ptr)
      cudaFreeHost(ptr);
  }
};

// ============================================================================
// VRAM BUFFER ALLOCATOR
// ============================================================================

class VramAllocator {
public:
  static void *allocate_vram(size_t bytes) {
    void *ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
  }

  static void free_vram(void *ptr) {
    if (ptr)
      cudaFree(ptr);
  }
};

// ============================================================================
// C API / FFI FOR RUST
// ============================================================================

extern "C" {

// ── Pinned host memory
// ────────────────────────────────────────────────────────

void *alloc_pinned_memory(size_t bytes) {
  return PinnedAllocator::allocate_pinned(bytes);
}

void free_pinned_memory(void *ptr) { PinnedAllocator::free_pinned(ptr); }

// ── VRAM buffer management
// ────────────────────────────────────────────────────

void *alloc_device_vram(size_t bytes) {
  return VramAllocator::allocate_vram(bytes);
}

void free_device_vram(void *ptr) { VramAllocator::free_vram(ptr); }

// ── Async DMA: pinned host → VRAM (non-blocking on CPU) ──────────────────────
// Returns 0 on success, non-zero on CUDA error.
// stream: a CUstream handle (cudaStream_t); pass NULL for default stream.
int async_memcpy_to_device(const void *host_src, void *dev_dst, size_t bytes,
                           void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t err =
      cudaMemcpyAsync(dev_dst, host_src, bytes, cudaMemcpyHostToDevice, s);
  return static_cast<int>(err);
}

// Synchronize a stream: blocks until all async ops on it are done.
void synchronize_stream(void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(s);
}

// ── AsyncStreamPool FFI bindings
// ──────────────────────────────────────────────

void *stream_pool_create(int num_streams) {
  return new AsyncStreamPool(num_streams);
}

void stream_pool_destroy(void *pool_ptr) {
  if (pool_ptr)
    delete static_cast<AsyncStreamPool *>(pool_ptr);
}

cudaStream_t stream_pool_get_next(void *pool_ptr) {
  if (!pool_ptr)
    return nullptr;
  return static_cast<AsyncStreamPool *>(pool_ptr)->get_next_stream();
}

void stream_pool_sync_all(void *pool_ptr) {
  if (pool_ptr)
    static_cast<AsyncStreamPool *>(pool_ptr)->synchronize_all();
}

cudaStream_t stream_pool_get_stream(void *pool_ptr, int idx) {
  if (!pool_ptr)
    return nullptr;
  return static_cast<AsyncStreamPool *>(pool_ptr)->get_stream(idx);
}

} // extern "C"

// ============================================================================
// PHASE 21: ZERO-COPY MAPPED MEMORY FOR QMEM CARTRIDGES
// Uses cudaHostAllocMapped - CPU writes to host_ptr, GPU reads directly
// over PCIe without explicit cudaMemcpy. Zero-Copy semantic tensors.
// ============================================================================

extern "C" {

// Allocate zero-copy mapped memory for QMEM semantic tensors.
// CPU writes to host_ptr, GPU kernel reads from dev_ptr directly.
// Returns 0 on success, non-zero on CUDA error.
int allocate_zero_copy_qmem(size_t bytes, void** host_ptr, void** dev_ptr) {
    cudaError_t err = cudaHostAlloc(host_ptr, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaHostGetDevicePointer(dev_ptr, *host_ptr, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(*host_ptr);
        *host_ptr = nullptr;
        *dev_ptr = nullptr;
        return static_cast<int>(err);
    }
    return 0;
}

// Free zero-copy mapped memory allocated by allocate_zero_copy_qmem.
void free_zero_copy_qmem(void* host_ptr) {
    if (host_ptr) {
        cudaFreeHost(host_ptr);
    }
}

// Get the device pointer from an existing host pointer (for re-validation)
int get_device_pointer(void* host_ptr, void** dev_ptr) {
    return static_cast<int>(cudaHostGetDevicePointer(dev_ptr, host_ptr, 0));
}

} // extern "C"
