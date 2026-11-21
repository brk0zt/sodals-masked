#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>
#include <stdexcept>

// -------------------------------------------------------------------------
// PHASE 4: cuBLAS Fused Complex GEMM
// -------------------------------------------------------------------------

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        }                                                                \
    } while (0)

#define CHECK_CUBLAS(call)                                               \
    do {                                                                 \
        cublasStatus_t status = call;                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                           \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << status << std::endl;                \
        }                                                                \
    } while (0)

__global__ void pack_complex_kernel(const float* r, const float* i, cuComplex* c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real_val = r ? r[idx] : 0.0f;
        float imag_val = i ? i[idx] : 0.0f;
        c[idx] = make_cuComplex(real_val, imag_val);
    }
}

__global__ void unpack_complex_kernel(const cuComplex* c, float* r, float* i, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (r) r[idx] = cuCrealf(c[idx]);
        if (i) i[idx] = cuCimagf(c[idx]);
    }
}

extern "C" void apply_cublas_cgemm_raw(
    uint64_t a_r_ptr, uint64_t a_i_ptr,
    uint64_t b_r_ptr, uint64_t b_i_ptr,
    uint64_t c_r_ptr, uint64_t c_i_ptr,
    int m, int n, int k,
    bool trans_a, bool trans_b
) {
    const float* d_A_r = reinterpret_cast<const float*>(a_r_ptr);
    const float* d_A_i = reinterpret_cast<const float*>(a_i_ptr);
    const float* d_B_r = reinterpret_cast<const float*>(b_r_ptr);
    const float* d_B_i = reinterpret_cast<const float*>(b_i_ptr);
    float* d_C_r = reinterpret_cast<float*>(c_r_ptr);
    float* d_C_i = reinterpret_cast<float*>(c_i_ptr);

    size_t size_A = m * k;
    size_t size_B = k * n;
    size_t size_C = m * n;

    cuComplex *d_A_complex = nullptr;
    cuComplex *d_B_complex = nullptr;
    cuComplex *d_C_complex = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A_complex, size_A * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_B_complex, size_B * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_C_complex, size_C * sizeof(cuComplex)));

    int threads = 256;
    int blocks_A = (size_A + threads - 1) / threads;
    int blocks_B = (size_B + threads - 1) / threads;
    int blocks_C = (size_C + threads - 1) / threads;

    pack_complex_kernel<<<blocks_A, threads>>>(d_A_r, d_A_i, d_A_complex, size_A);
    pack_complex_kernel<<<blocks_B, threads>>>(d_B_r, d_B_i, d_B_complex, size_B);
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Map Row-Major to Col-Major for cuBLAS
    cublasOperation_t op_B = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N; 
    cublasOperation_t op_A = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    int ldb = trans_b ? k : n;
    int lda = trans_a ? m : k;
    int ldc = n;
    
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);

    CHECK_CUBLAS(cublasCgemm(handle,
                             op_B, op_A,
                             n, m, k,
                             &alpha,
                             d_B_complex, ldb,
                             d_A_complex, lda,
                             &beta,
                             d_C_complex, ldc));

    CHECK_CUDA(cudaDeviceSynchronize());

    unpack_complex_kernel<<<blocks_C, threads>>>(d_C_complex, d_C_r, d_C_i, size_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasDestroy(handle);
    cudaFree(d_A_complex);
    cudaFree(d_B_complex);
    cudaFree(d_C_complex);
}
