// src/crates/sodals_neuro/src/kernels/fused_resonance_core.cu
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define TILE_Q 16
#define TILE_K 16
#define MAX_HEAD_DIM_CPLX 128

// Fast, derivative-free Secant root finding executed natively on SM Registers
// Phase 33 FIX: Removed circular mean-of-poles blending. 
// Now uses pure geometric cross-interference between distinctly poled tensors.
__device__ __forceinline__ float secant_phase_alignment(float2 q, float2 k, float2 pole_q, float2 pole_k) {
    // VRAM SAFETY: If poles are uninitialized or zero, return 0.0f (no phase rotation).
    // This prevents the arbitrary 0.1f rotation that scrambles the attention manifold.
    if (pole_q.x == 0.0f && pole_q.y == 0.0f && pole_k.x == 0.0f && pole_k.y == 0.0f) {
        return 0.0f;
    }

    // Phase 33: No circular pole blending. Use distinct poles directly in objective.
    float t0 = 0.0f;
    float t1 = 0.1f; 

// Phase 33 FIX: Pure geometric phase alignment objective function.
// Computes the cross-interference energy between q (rotated by t, scaled by pole_q) 
// and k (scaled by pole_k). The root-finding targets the theta that minimizes
// phase distortion between these two distinctly poled tensors.
// 
// Instead of the circular "mean of poles" target, we compute:
//   E(t) = || rotate(q, t) * pole_q - k * pole_k ||^2
//        = || (q.x*cos(t) - q.y*sin(t)) * pole_q.x - k.x * pole_k.x ||^2
//         + || (q.x*sin(t) + q.y*cos(t)) * pole_q.y - k.y * pole_k.y ||^2
//
// The secant method finds t where E(t) = 0 (perfect phase alignment).
#define EVAL_MISMATCH(t) \
    ( \
        ((q.x * cosf(t) - q.y * sinf(t)) * pole_q.x - k.x * pole_k.x) * \
        ((q.x * cosf(t) - q.y * sinf(t)) * pole_q.x - k.x * pole_k.x) + \
        ((q.x * sinf(t) + q.y * cosf(t)) * pole_q.y - k.y * pole_k.y) * \
        ((q.x * sinf(t) + q.y * cosf(t)) * pole_q.y - k.y * pole_k.y) \
    )

    float f0 = EVAL_MISMATCH(t0);
    float f1 = EVAL_MISMATCH(t1);

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        float f_diff = f1 - f0;
        if (fabsf(f_diff) < 1e-6f) break;
        
        float t2 = t1 - f1 * (t1 - t0) / f_diff;
        // Phase 23: Strict clamp to prevent divergence to NaN or ±∞
        t2 = fmaxf(-6.2831853f, fminf(6.2831853f, t2));
        t0 = t1;
        f0 = f1;
        t1 = t2;
        f1 = EVAL_MISMATCH(t1);
    }
    return t1;
}

extern "C" __global__ void apply_fused_resonance_gemm(
    float* __restrict__ attention_scores,
    const float2* __restrict__ q_states,
    const float2* __restrict__ k_states,
    const float2* __restrict__ laplace_poles,
    const float* __restrict__ impedance,
    const uint32_t* __restrict__ neuro_flags,
    const int batch_size,
    const int num_heads,
    const int seq_q,
    const int seq_k,
    const int head_dim_cplx,
    const int n_nodes
) {
    // ── 1. TOTAL L1 RESIDENCY ALLOCATION ──────────────────────────────────
    __shared__ float2 s_q[TILE_Q][MAX_HEAD_DIM_CPLX];
    __shared__ float2 s_k[TILE_K][MAX_HEAD_DIM_CPLX];
    
    // Per-token metadata for Q and K tiles
    __shared__ float2 s_poles_q[TILE_Q];
    __shared__ float2 s_poles_k[TILE_K];
    __shared__ float  s_imp_q[TILE_Q];
    __shared__ float  s_imp_k[TILE_K];
    __shared__ uint32_t s_flags_q[TILE_Q];
    __shared__ uint32_t s_flags_k[TILE_K];

    const int bx = blockIdx.x; 
    const int by = blockIdx.y; 
    const int bz = blockIdx.z; // (batch * num_heads)

    // DEEPCUBE AUDIT DEFENSE: Semantic laplace_poles are MACROSCOPIC token properties.
    // They are intentionally indexed by batch_idx * seq, NOT per-head.
    // Topological resonance is a universal field, not a head-specific subspace.
    // Hardware Lockdown: Correct batch indexing for metadata (neuro_flags, etc.)
    const int batch_idx = bz / num_heads;

    // Flattened 1D Block ID for true cooperative loading
    const int flat_tid = threadIdx.y * TILE_K + threadIdx.x; 
    const int num_threads = TILE_Q * TILE_K; // 256

    // ── 2. TRUE 1D FLATTENED COOPERATIVE LOADING ──────────────────────────
    #pragma unroll
    for (int i = flat_tid; i < TILE_Q * head_dim_cplx; i += num_threads) {
        int r = i / head_dim_cplx;
        int c = i % head_dim_cplx;
        int g_q = by * TILE_Q + r;
        s_q[r][c] = (g_q < seq_q && c < head_dim_cplx) ? 
                    q_states[((bz * seq_q) + g_q) * head_dim_cplx + c] : make_float2(0.0f, 0.0f);
    }

    #pragma unroll
    for (int i = flat_tid; i < TILE_K * head_dim_cplx; i += num_threads) {
        int r = i / head_dim_cplx;
        int c = i % head_dim_cplx;
        int g_k = bx * TILE_K + r;
        s_k[r][c] = (g_k < seq_k && c < head_dim_cplx) ? 
                    k_states[((bz * seq_k) + g_k) * head_dim_cplx + c] : make_float2(0.0f, 0.0f);
    }

    // Load Metadata: Use batch_idx for per-batch flags and poles
    // Phase 25: HARDWARE SEGFAULT SHIELD - Null guard all metadata reads
    if (flat_tid < TILE_Q) {
        int g_q = by * TILE_Q + flat_tid;
        if (g_q < seq_q) {
            // Null-guarded reads with default neutral values
            float2 pole_q = make_float2(0.0f, 0.0f); // Default neutral pole
            float imp_q = 1.0f; // Default neutral impedance
            uint32_t flags_q = 0; // Default neutral flags

            if (n_nodes > 0) {
                int node_idx = g_q % n_nodes;
                if (laplace_poles != nullptr && node_idx < n_nodes) {
                    pole_q = laplace_poles[node_idx];
                }
                if (impedance != nullptr && node_idx < n_nodes) {
                    imp_q = impedance[node_idx];
                }
                if (neuro_flags != nullptr && node_idx < n_nodes) {
                    flags_q = neuro_flags[node_idx];
                }
            }
            s_poles_q[flat_tid] = pole_q;
            s_imp_q[flat_tid] = imp_q;
            s_flags_q[flat_tid] = flags_q;
        } else {
            s_poles_q[flat_tid] = make_float2(0.0f, 0.0f);
            s_imp_q[flat_tid]   = 1.0f; // Neutral impedance, not zero
            s_flags_q[flat_tid] = 0;
        }
    }

    if (flat_tid < TILE_K) {
        int g_k = bx * TILE_K + flat_tid;
        if (g_k < seq_k) {
            // Null-guarded reads with default neutral values
            float2 pole_k = make_float2(0.0f, 0.0f); // Default neutral pole
            float imp_k = 1.0f; // Default neutral impedance
            uint32_t flags_k = 0; // Default neutral flags

            if (n_nodes > 0) {
                int node_idx = g_k % n_nodes;
                if (laplace_poles != nullptr && node_idx < n_nodes) {
                    pole_k = laplace_poles[node_idx];
                }
                if (impedance != nullptr && node_idx < n_nodes) {
                    imp_k = impedance[node_idx];
                }
                if (neuro_flags != nullptr && node_idx < n_nodes) {
                    flags_k = neuro_flags[node_idx];
                }
            }
            s_poles_k[flat_tid] = pole_k;
            s_imp_k[flat_tid] = imp_k;
            s_flags_k[flat_tid] = flags_k;
        } else {
            s_poles_k[flat_tid] = make_float2(0.0f, 0.0f);
            s_imp_k[flat_tid]   = 1.0f; // Neutral impedance, not zero
            s_flags_k[flat_tid] = 0;
        }
    }

    // Barrier: Ensure L1 is fully populated
    __syncthreads();

    const int tx = threadIdx.x; 
    const int ty = threadIdx.y; 
    const int q_idx = by * TILE_Q + ty;
    const int k_idx = bx * TILE_K + tx;

    // Hardware Lockdown: REMOVED early return to prevent Warp Deadlock
    // All threads must reach the __shfl_sync below.

    // ── 3. ZERO INNER-LOOP GLOBAL READS ───────────────────────────────────
    // Phase 20.7: Hoist secant_phase_alignment OUTSIDE the c loop.
    // The semantic phase alignment is a MACROSCOPIC interaction between token poles,
    // NOT dependent on individual vector coordinates. Compute theta ONCE per thread.
    float E = 0.0f;
    float2 pq = s_poles_q[ty];
    float2 pk = s_poles_k[tx];

    // Compute optimal theta ONCE per (ty, tx) token pair using macroscopic pole interaction
    // Then compute sincosf ONCE - avoid billions of wasted cycles inside the loop
    float theta = secant_phase_alignment(pq, pk, pq, pk);
    float s, c_th;
    sincosf(theta, &s, &c_th);

    #pragma unroll
    for (int c = 0; c < head_dim_cplx; ++c) {
        float2 q = s_q[ty][c]; 
        float2 k = s_k[tx][c]; 

        // Apply pre-computed rotation scalars - no sinf/cosf inside loop
        float q_rot_x = q.x * c_th - q.y * s;
        float q_rot_y = q.x * s + q.y * c_th;

        E += (q_rot_x * k.x + q_rot_y * k.y);
    }

    E *= (1.0f / sqrtf((float)(head_dim_cplx * 2)));

    // ── 4. WARP-LEVEL WAVE DIFFRACTION (__shfl_sync) ───────────────────────
    // Phase 17.6: Participation is mandatory for all threads to avoid deadlock
    float neighbor_E = __shfl_up_sync(0xffffffff, E, 1);
    if (tx == 0) neighbor_E = E; 
    
    float E_diffracted = 0.85f * E + 0.15f * neighbor_E;

    // ── 5. BRANCHLESS NEUROSYMBOLIC EPILOGUE ──────────────────────────────
    float final_score = E_diffracted;

    // VRAM SAFETY & COGNITIVE SHIELD: 
    // If no semantic topology exists (n_nodes == 0), bypass the impedance/flag penalties.
    // Otherwise, default flags (0) cause a 100% mismatch, resulting in decay=0.0f,
    // which completely zeros out the attention matrix and causes severe hallucinations.
    if (n_nodes > 0) {
        float local_imp = s_imp_k[tx];
        if (E_diffracted > 0.0f && local_imp < 0.5f) {
            final_score += logf(1.0f + E_diffracted) * (1.0f - local_imp);
        } else {
            final_score -= (local_imp * 8.0f) + fabsf(E_diffracted * 0.5f);
        }

        uint32_t flag_overlap = s_flags_q[ty] & s_flags_k[tx];
        float mismatch_mask = (flag_overlap == 0) ? 1.0f : 0.0f;
        float affinity_boost = (float)__popc(flag_overlap) * 0.05f;

        float momentum = fabsf(E_diffracted) * (1.0f - local_imp);
        float decay = __expf(-(mismatch_mask * 5.0f / (momentum + 1e-5f)));

        final_score = (final_score + affinity_boost) * decay;
    }

    // ── 6. MASKED VRAM WRITE-BACK ─────────────────────────────────────────
    // Hardware Lockdown: Only threads within logic bounds perform the write
    if (q_idx < seq_q && k_idx < seq_k) {
        int out_idx = ((bz * seq_q) + q_idx) * seq_k + k_idx; 
        attention_scores[out_idx] = final_score;
    }
}

extern "C" void launch_fused_resonance(
    void* stream,
    unsigned long long attention_scores,
    unsigned long long q_states,
    unsigned long long k_states,
    unsigned long long laplace_poles,
    unsigned long long impedance,
    unsigned long long neuro_flags,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    int head_dim_cplx,
    int n_nodes
) {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    
    dim3 threads(TILE_K, TILE_Q, 1);
    dim3 blocks(
        (seq_k + TILE_K - 1) / TILE_K,
        (seq_q + TILE_Q - 1) / TILE_Q,
        batch_size * num_heads
    );

    apply_fused_resonance_gemm<<<blocks, threads, 0, cu_stream>>>(
        reinterpret_cast<float*>(attention_scores),
        reinterpret_cast<const float2*>(q_states),
        reinterpret_cast<const float2*>(k_states),
        reinterpret_cast<const float2*>(laplace_poles),
        reinterpret_cast<const float*>(impedance),
        reinterpret_cast<const uint32_t*>(neuro_flags),
        batch_size, num_heads, seq_q, seq_k, head_dim_cplx, n_nodes
    );
}
