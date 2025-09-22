#include "SparseGate.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include "../CudaControl/Helper.hpp"

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
    int k = targetQubits.size();       // number of qubits this gate acts on
    int d = 1 << k;                    // dimension of U
    int dim = 1 << nQubits;            // full state dimension

    // --- Build sparse matrix descriptor
    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(&matU,
        d, d, nnzU,
        (void*)d_csrRowPtrU,
        (void*)d_csrColIndU,
        (void*)d_csrValU,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));

    // Work buffer (reused across blocks)
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    double alpha = 1.0, beta = 0.0;

    // Temporary dense descriptors
    cusparseDnVecDescr_t vecX, vecY;

    // Number of blocks of length d
    int nBlocks = 1 << (nQubits - k);

    // Loop over blocks
    for (int blk = 0; blk < nBlocks; blk++) {
        // --- Compute base index for this block
        int base = 0;
        int tmp = blk;
        for (int q = 0; q < nQubits; q++) {
            if (std::find(targetQubits.begin(), targetQubits.end(), q) != targetQubits.end())
                continue; // skip target qubits
            int bit = tmp & 1;
            tmp >>= 1;
            base |= (bit << q);
        }

        // --- Collect d amplitudes for all combinations of target qubits
        std::vector<int> offsets(d, 0);
        for (int b = 0; b < d; b++) {
            int idx = base;
            for (int q = 0; q < k; q++) {
                if (b & (1 << q)) idx |= (1 << targetQubits[q]);
            }
            offsets[b] = idx;
        }

        // --- Device dense views
        cuDoubleComplex* d_block_in  = (cuDoubleComplex*)(d_state_in  + offsets[0]);
        cuDoubleComplex* d_block_out = (cuDoubleComplex*)(d_state_out + offsets[0]);

        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d, d_block_in, CUDA_C_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d, d_block_out, CUDA_C_64F));

        // Query buffer size (once is enough, but we keep it simple here)
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, vecX, &beta, vecY,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        // SpMV: y = U * x
        CHECK_CUSPARSE(cusparseSpMV(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, vecX, &beta, vecY,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUDA(cudaFree(dBuffer));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matU));

    return 0;
}
