#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

extern "C" __global__ void atan2_kernel(float* out, const float* y, const float* x, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Compute atan2f(imag, real) returning phase in [-pi, pi]
        out[idx] = atan2f(y[idx], x[idx]);
    }
}

extern "C" void apply_atan2_cuda(void* stream, void* out, const void* y, const void* x, int num_elements) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    atan2_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)out,
        (const float*)y,
        (const float*)x,
        num_elements
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] atan2_kernel failed: %s\n", cudaGetErrorString(err));
    }
}
