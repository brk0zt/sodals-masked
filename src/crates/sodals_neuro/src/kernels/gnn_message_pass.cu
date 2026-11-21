// src/kernels/gnn_message_pass.cu
// CUDA Kernel: Graph Neural Network Message Passing for Attention Bias
//
// Injects GNN adjacency weights directly into the Attention mask.
// M_GNN(q, k) = gnn_weight * A_CSR(node_q, node_k)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define BLOCK_SIZE 256
#define MAX_SHARED_DEGREE 4096 // 4096 * (4 + 4) = 32KB shared memory

// Binary search utility for shared memory
__device__ __forceinline__ int binary_search_shared(const int* arr, int val, int len) {
    int left = 0;
    int right = len - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int m_val = arr[mid];
        if (m_val == val) return mid;
        if (m_val < val) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Binary search utility for global memory (PCIe)
__device__ __forceinline__ int binary_search_global(const int* arr, int val, int start_idx, int end_idx) {
    int left = start_idx;
    int right = end_idx - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int m_val = arr[mid];
        if (m_val == val) return mid - start_idx;
        if (m_val < val) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

__global__ void apply_gnn_bias_kernel(
    float* __restrict__ scores,         // [batch, heads, seq_q, seq_k]
    const int* __restrict__ q_nodes,    // [seq_q]
    const int* __restrict__ k_nodes,    // [seq_k]
    const size_t* __restrict__ indptr,     // [num_nodes + 1]
    const int* __restrict__ indices,    // [nnz]
    const float* __restrict__ data,     // [nnz]
    const int batch, const int heads, 
    const int seq_q, const int seq_k,
    const float gnn_weight)
{
    // Grid: dim3(batch * heads, ceil(seq_q / BLOCK_SIZE))
    const int bh_idx = blockIdx.x;
    if (bh_idx >= batch * heads) return;

    const int b = bh_idx / heads;
    const int h = bh_idx % heads;
    const size_t base_offset = (size_t)(b * heads + h) * seq_q * seq_k;

    // Process Qs partitioned by gridDim.y
    for (int q_base = blockIdx.y * BLOCK_SIZE; q_base < seq_q; q_base += gridDim.y * BLOCK_SIZE) {
        
        // Inside the block, we iterate through each Q sequentially
        // This allows all threads to cooperatively load the GNN neighborhood for a single Q
        // and then cooperatively apply it across all K.
        const int q_end = min(q_base + BLOCK_SIZE, seq_q);
        
        for (int q_idx = q_base; q_idx < q_end; q_idx++) {
            const int n_q = q_nodes[q_idx];
            
            // Skip invalid nodes (-1)
            if (n_q < 0) continue;
            
            const size_t start_idx = indptr[n_q];
            const size_t end_idx = indptr[n_q + 1];
            const int degree = (int)(end_idx - start_idx);
            
            if (degree == 0) continue;

            const bool use_shared = (degree <= MAX_SHARED_DEGREE);
            
            __shared__ int sh_indices[MAX_SHARED_DEGREE];
            __shared__ float sh_data[MAX_SHARED_DEGREE];
            
            // 1. Cooperatively load CSR row into Shared Memory (if it fits)
            if (use_shared) {
                for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
                    sh_indices[i] = indices[start_idx + i];
                    sh_data[i] = data[start_idx + i];
                }
            }
            __syncthreads(); // Wait for neighborhood load
            
            // 2. Cooperatively process K dimension
            for (int k = threadIdx.x; k < seq_k; k += BLOCK_SIZE) {
                const int n_k = k_nodes[k];
                if (n_k < 0) continue;
                
                int found_idx = -1;
                if (use_shared) {
                    found_idx = binary_search_shared(sh_indices, n_k, degree);
                } else {
                    found_idx = binary_search_global(indices, n_k, start_idx, end_idx);
                }
                
                if (found_idx != -1) {
                    const float edge_weight = use_shared ? sh_data[found_idx] : data[start_idx + found_idx];
                    const size_t score_idx = base_offset + (size_t)q_idx * seq_k + k;
                    
                    // Add GNN Bias
                    scores[score_idx] += gnn_weight * edge_weight;
                }
            }
            __syncthreads(); // Prevent next Q from overwriting shared memory while K loop finishes
        }
    }
}

extern "C" {

void apply_gnn_mask(
    cudaStream_t stream,
    void* wptr,          // float* scores
    const int* q_nodes,
    const int* k_nodes,
    const size_t* indptr,
    const int* indices,
    const void* dptr,    // float* data
    int b, int h, int q, int k,
    float gnn_weight)
{
    float* scores_f32 = reinterpret_cast<float*>(wptr);
    const float* data_f32 = reinterpret_cast<const float*>(dptr);

    if (b == 0 || h == 0 || q == 0 || k == 0) return;

    dim3 block(BLOCK_SIZE);
    dim3 grid(b * h, (q + BLOCK_SIZE - 1) / BLOCK_SIZE);

    apply_gnn_bias_kernel<<<grid, block, 0, stream>>>(
        scores_f32, q_nodes, k_nodes,
        indptr, indices, data_f32,
        b, h, q, k, gnn_weight
    );
}

} // extern "C"
