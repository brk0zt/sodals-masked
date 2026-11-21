#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// -------------------------------------------------------------------------
// PHASE 4: GPU Fluid Algebra (Clifford Binding & Viscosity)
// -------------------------------------------------------------------------

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        }                                                                \
    } while (0)

__global__ void bind_phasors_kernel(
    const float* a_r, const float* a_i,
    const float* b_r, const float* b_i,
    float* out_r, float* out_i,
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float ar = a_r[idx];
        float ai = a_i[idx];
        float br = b_r[idx];
        float bi = b_i[idx];
        
        out_r[idx] = ar * br - ai * bi;
        out_i[idx] = ar * bi + ai * br;
    }
}

__global__ void unbind_phasors_kernel(
    const float* c_r, const float* c_i,
    const float* k_r, const float* k_i,
    float* out_r, float* out_i,
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float cr = c_r[idx];
        float ci = c_i[idx];
        float kr = k_r[idx];
        // Conjugate of K
        float ki = -k_i[idx];
        
        out_r[idx] = cr * kr - ci * ki;
        out_i[idx] = cr * ki + ci * kr;
    }
}

__global__ void apply_viscosity_kernel(
    float* p_r, float* p_i,
    float impedance,
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float decay = expf(-impedance);
        p_r[idx] *= decay;
        p_i[idx] *= decay;
    }
}

extern "C" void bind_phasors_raw(
    uint64_t a_r_ptr, uint64_t a_i_ptr,
    uint64_t b_r_ptr, uint64_t b_i_ptr,
    uint64_t out_r_ptr, uint64_t out_i_ptr,
    size_t size
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    bind_phasors_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float*>(a_r_ptr),
        reinterpret_cast<const float*>(a_i_ptr),
        reinterpret_cast<const float*>(b_r_ptr),
        reinterpret_cast<const float*>(b_i_ptr),
        reinterpret_cast<float*>(out_r_ptr),
        reinterpret_cast<float*>(out_i_ptr),
        size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

extern "C" void unbind_phasors_raw(
    uint64_t c_r_ptr, uint64_t c_i_ptr,
    uint64_t k_r_ptr, uint64_t k_i_ptr,
    uint64_t out_r_ptr, uint64_t out_i_ptr,
    size_t size
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    unbind_phasors_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float*>(c_r_ptr),
        reinterpret_cast<const float*>(c_i_ptr),
        reinterpret_cast<const float*>(k_r_ptr),
        reinterpret_cast<const float*>(k_i_ptr),
        reinterpret_cast<float*>(out_r_ptr),
        reinterpret_cast<float*>(out_i_ptr),
        size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

extern "C" void apply_viscosity_raw(
    uint64_t p_r_ptr, uint64_t p_i_ptr,
    float impedance,
    size_t size
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    apply_viscosity_kernel<<<blocks, threads>>>(
        reinterpret_cast<float*>(p_r_ptr),
        reinterpret_cast<float*>(p_i_ptr),
        impedance,
        size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}
