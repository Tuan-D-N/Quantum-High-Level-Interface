#include "SparseGate.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include "../CudaControl/Helper.hpp"

#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define CUDA_KERNEL(...) 
#endif

// The gather and scatter kernels are still necessary for correctness and performance.
__global__ void gather_kernel(cuDoubleComplex* d_out, const cuDoubleComplex* d_in, const int* d_offsets, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_out[i] = d_in[d_offsets[i]];
    }
}

__global__ void scatter_kernel(cuDoubleComplex* d_out, const cuDoubleComplex* d_in, const int* d_offsets, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_out[d_offsets[i]] = d_in[i];
    }
}

// Apply sparse gate U to given qubits (q0, q1, ..., qk).
// - nQubits: total system qubits
// - d_csrRowPtrU, d_csrColIndU, d_csrValU: CSR of U (size d x d, where d=2^k)
// - d_state_in/out: device statevectors, length 2^n
// - targetQubits: list of k target qubits (ascending order)
// Apply sparse gate U to given qubits (q0, q1, ..., qk)
int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int* d_csrRowPtrU,
    const int* d_csrColIndU,
    const cuDoubleComplex* d_csrValU,
    const cuDoubleComplex* d_state_in,
    cuDoubleComplex* d_state_out,
    const std::vector<int>& targetQubits,
    int nnzU)
{
    int k = targetQubits.size();
    int d = 1 << k;
    int dim = 1 << nQubits;

    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(&matU, d, d, nnzU, (void*)d_csrRowPtrU, (void*)d_csrColIndU, (void*)d_csrValU,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    cuDoubleComplex* d_block_in;
    cuDoubleComplex* d_block_out;
    CHECK_CUDA(cudaMalloc(&d_block_in, d * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_block_out, d * sizeof(cuDoubleComplex)));

    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d, d_block_in, CUDA_C_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d, d_block_out, CUDA_C_64F));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, vecX, &beta, vecY,
                                           CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    int nBlocks = 1 << (nQubits - k);
    std::vector<int> nonTargetQubits;
    for (int q = 0; q < nQubits; ++q) {
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end()) {
            nonTargetQubits.push_back(q);
        }
    }

    std::vector<int> h_offsets(d);
    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    for (int blk = 0; blk < nBlocks; ++blk) {
        int nonTargetMask = 0;
        for (size_t i = 0; i < nonTargetQubits.size(); ++i) {
            if ((blk >> i) & 1) {
                nonTargetMask |= (1 << nonTargetQubits[i]);
            }
        }

        for (int b = 0; b < d; ++b) {
            int targetMask = 0;
            for (int q = 0; q < k; ++q) {
                if ((b >> q) & 1) {
                    targetMask |= (1 << targetQubits[q]);
                }
            }
            h_offsets[b] = nonTargetMask | targetMask;
        }

        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        int threads_per_block = 256;
        int blocks = (d + threads_per_block - 1) / threads_per_block;
        gather_kernel CUDA_KERNEL(blocks, threads_per_block) (d_block_in, d_state_in, d_offsets, d);
        CHECK_CUDA(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        CHECK_CUDA(cudaMemset(d_block_out, 0, d * sizeof(cuDoubleComplex)));

        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, vecX, &beta, vecY,
                                    CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        scatter_kernel CUDA_KERNEL(blocks, threads_per_block) (d_state_out, d_block_out, d_offsets, d);
        CHECK_CUDA(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_block_in));
    CHECK_CUDA(cudaFree(d_block_out));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matU));

    return 0;
}