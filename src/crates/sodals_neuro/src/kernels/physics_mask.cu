// src/kernels/physics_mask.cu - Async CUDA Streams with Zero-Latency Pipeline
// CUDA Kernel: Spectral Attention Masking via Physics Bias
//
// SODALS WHITEPAPER: Semantic Topology Mask injection into Attention
//
// Formula: Attention(Q, K, V, P) = Softmax(QK^T/sqrt(d_k) + M_semantic + P)V
// Where P = physics bias derived from token impedance (Whitepaper §8, CUDA Intervention)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define IMPEDANCE_THRESHOLD 0.8f
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

/// Compute physics penalty from impedance value.
/// Whitepaper §1: Destructive Interference → high-impedance tokens are masked.
/// impedance > threshold → -∞ (hard semantic mask, equivalent to M_semantic = -∞)
/// impedance ≤ threshold → soft penalty proportional to impedance
__device__ __forceinline__ float compute_penalty(float impedance, float threshold) {
    if (impedance > threshold) {
        return -CUDART_INF_F;     // Complete masking: token outside semantic topology
    } else {
        return -10.0f * (impedance / threshold);  // Soft penalty: partial phase mismatch
    }
}

// ============================================================================
// INTERNAL DEVICE FUNCTIONS
// ============================================================================

/// Apply physics bias to attention scores.
///
/// GRID CONTRACT (CRITICAL — must match apply_mask launch config):
///   gridDim.x  = batch * heads        (one block-column per B×H pair)
///   gridDim.y  = ceil(seq_q / BLOCK_SIZE)   (rows of Q partitioned across blocks)
///   blockDim.x = BLOCK_SIZE (256)
///
/// MEMORY LAYOUT:
///   scores[b, h, q, k]  = scores[(b*heads + h)*seq_q*seq_k + q*seq_k + k]
///   impedance[k]         = per-key-token semantic impedance (F16, length = seq_k)
__device__ __forceinline__ void apply_physics_bias_async(
    float* __restrict__       scores,     // [batch, heads, seq_q, seq_k] — in/out F32
    const half* __restrict__  impedance,  // [seq_k] — read-only F16
    const int batch, const int heads,
    const int seq_q, const int seq_k,
    const float threshold)
{
    // ── Decode which (batch, head) this block-column handles ──────────────────
    const int bh_idx = blockIdx.x;
    if (bh_idx >= batch * heads) return;

    const int b = bh_idx / heads;
    const int h = bh_idx % heads;
    const int base_offset = (b * heads + h) * seq_q * seq_k;

    // ── Grid-stride loop over the Q dimension ────────────────────────────────
    for (int q_base = (int)blockIdx.y * BLOCK_SIZE; q_base < seq_q; q_base += (int)gridDim.y * BLOCK_SIZE) {
        const int q = q_base + (int)threadIdx.x;
        const bool valid_q = (q < seq_q);

        // ── Shared Memory Tiling Loop ──────────────────────────────────────────
        // Instead of ONLY caching the first BLOCK_SIZE elements, we tile across seq_k.
        __shared__ half shared_impedance[BLOCK_SIZE];

        for (int k_chunk = 0; k_chunk < seq_k; k_chunk += BLOCK_SIZE) {
            // All threads participate in pulling global memory into L1 shared memory
            if (k_chunk + threadIdx.x < seq_k) {
                shared_impedance[threadIdx.x] = impedance[k_chunk + threadIdx.x];
            }
            __syncthreads(); // Barrier: wait for chunk to load

            if (valid_q) {
                const int k_end = min(k_chunk + BLOCK_SIZE, seq_k);
#pragma unroll 4
                for (int k = k_chunk; k < k_end; k++) {
                    const int idx = base_offset + q * seq_k + k;
                    float score = scores[idx];

                    // Zero global memory latency; all reads pull from L1 cache
                    const half raw_imp = shared_impedance[k - k_chunk];
                    score += compute_penalty(__half2float(raw_imp), threshold);

                    scores[idx] = score;
                }
            }
            __syncthreads(); // Barrier: prevent next chunk from overwriting L1 before compute completes
        }
    }
}

/// Fused physics bias + causal masking variant.
__device__ __forceinline__ void apply_physics_bias_causal_async(
    float* __restrict__       scores,
    const half* __restrict__  impedance,
    const int batch, const int heads,
    const int seq_q, const int seq_k,
    const float threshold)
{
    const int bh_idx = blockIdx.x;
    if (bh_idx >= batch * heads) return;

    const int b = bh_idx / heads;
    const int h = bh_idx % heads;
    const int base_offset = (b * heads + h) * seq_q * seq_k;

    for (int q_base = (int)blockIdx.y * BLOCK_SIZE; q_base < seq_q; q_base += (int)gridDim.y * BLOCK_SIZE) {
        const int q = q_base + (int)threadIdx.x;
        const bool valid_q = (q < seq_q);

        __shared__ half shared_impedance[BLOCK_SIZE];

        for (int k_chunk = 0; k_chunk < seq_k; k_chunk += BLOCK_SIZE) {
            if (k_chunk + threadIdx.x < seq_k) {
                shared_impedance[threadIdx.x] = impedance[k_chunk + threadIdx.x];
            }
            __syncthreads();

            if (valid_q) {
                const int k_end = min(k_chunk + BLOCK_SIZE, seq_k);
#pragma unroll 4
                for (int k = k_chunk; k < k_end; k++) {
                    const int idx = base_offset + q * seq_k + k;
                    float score = scores[idx];

                    if (k > q) { // Future token — hard causal mask
                        score = -CUDART_INF_F;
                    } else {     // Physics Impedance Penalty 
                        const half raw_imp = shared_impedance[k - k_chunk];
                        score += compute_penalty(__half2float(raw_imp), threshold);
                    }
                    scores[idx] = score;
                }
            }
            __syncthreads();
        }
    }
}

/// Backward pass — accumulate impedance gradients.
__device__ __forceinline__ void physics_bias_backward_async(
    const float* __restrict__ grad_scores,
    float* __restrict__       grad_impedance,
    const half* __restrict__  impedance,
    const int batch, const int heads,
    const int seq_q, const int seq_k,
    const float threshold)
{
    const int k = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    if (k >= seq_k) return;

    float local_grad = 0.0f;
    const float imp = __half2float(impedance[k]);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
#pragma unroll 4
            for (int q = 0; q < seq_q; q++) {
                const int idx = ((b * heads + h) * seq_q + q) * seq_k + k;
                if (imp <= threshold) {
                    local_grad += grad_scores[idx] * (-10.0f / threshold);
                }
            }
        }
    }
    atomicAdd(&grad_impedance[k], local_grad);
}

// ============================================================================
// GLOBAL KERNEL ENTRY POINTS
// ============================================================================

extern "C" {

__global__ void apply_physics_bias_kernel(
    float* scores, const half* impedance,
    int batch, int heads, int seq_q, int seq_k, float threshold)
{
    apply_physics_bias_async(scores, impedance, batch, heads, seq_q, seq_k, threshold);
}

__global__ void apply_physics_bias_causal_kernel(
    float* scores, const half* impedance,
    int batch, int heads, int seq_q, int seq_k, float threshold)
{
    apply_physics_bias_causal_async(scores, impedance, batch, heads, seq_q, seq_k, threshold);
}

__global__ void physics_bias_backward_kernel(
    const float* grad_scores, float* grad_impedance, const half* impedance,
    int batch, int heads, int seq_q, int seq_k, float threshold)
{
    physics_bias_backward_async(grad_scores, grad_impedance, impedance,
                                batch, heads, seq_q, seq_k, threshold);
}

// ============================================================================
// FFI HOST WRAPPER FOR RUST (adapter.rs → physics_mask::apply_mask)
// ============================================================================
//
// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  BUG FIX (CRITICAL — was root cause of "brvevfXDee" garbage output)    ║
// ║                                                                          ║
// ║  BEFORE (BROKEN):                                                        ║
// ║    dim3 grid((b * h * q * k + 255) / 256);  ← 1D grid                  ║
// ║    gridDim.y always 0 → kernel is effectively a NO-OP for decode steps  ║
// ║    blockIdx.x >> batch*heads → early-return for almost all blocks        ║
// ║                                                                          ║
// ║  AFTER (CORRECT):                                                        ║
// ║    dim3 grid(b * h,  ceil(q / BLOCK_SIZE));  ← 2D grid                 ║
// ║    Matches the kernel's indexing contract:                               ║
// ║      blockIdx.x = batch×head index                                       ║
// ║      blockIdx.y = Q-row partition index                                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝
void apply_mask(
    cudaStream_t stream,
    void*        wptr,    // F32 attention scores  [b, h, q, k]
    const void*  iptr,    // F16 impedance values  [k]  (aligned to current k_len)
    int b, int h, int q, int k)
{
    float*       scores_f32    = reinterpret_cast<float*>(wptr);
    const half*  impedance_f16 = reinterpret_cast<const half*>(iptr);

    const float threshold = IMPEDANCE_THRESHOLD;

    // ── CORRECT 2D grid ───────────────────────────────────────────────────
    // Axis 0: one block per (batch, head) pair
    // Axis 1: blocks that partition the Q dimension
    dim3 block(BLOCK_SIZE);
    dim3 grid(b * h,
              (q + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Guard: nothing to do for empty tensors
    if (b == 0 || h == 0 || q == 0 || k == 0) return;

    apply_physics_bias_kernel<<<grid, block, 0, stream>>>(
        scores_f32, impedance_f16,
        b, h, q, k, threshold);

    // Note: cudaGetLastError() check is deliberately omitted from this thin
    // FFI shim; the Rust caller owns error-handling policy.
}

} // extern "C"

/*
ASYNC PIPELINE OPTIMIZATION NOTES:

1. DOUBLE BUFFERING (adapter.rs AsyncStreamPool):
   CPU prepares layer N+1 impedance while GPU processes layer N.

2. ZERO-COPY MEMORY (host_memory.cpp ZeroCopyAllocator):
   Impedance maps live in pinned host memory; GPU accesses directly.

3. GRID-STRIDE LOOP:
   Handles seq_q > BLOCK_SIZE gracefully across multiple blockIdx.y values.

4. SHARED MEMORY PREFETCH:
   First BLOCK_SIZE impedance values are cached in 48KB L1 shared memory.
   For longer sequences, global memory is used directly for k >= BLOCK_SIZE.

5. CAUSAL MASKING:
   apply_physics_bias_causal_kernel fuses causal + physics masks into one pass.
*/
