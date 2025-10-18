#include "SparseGate.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include "../CudaControl/Helper.hpp"
#include "SparseHelper.hpp"

// Apply sparse gate U to given qubits (q0, q1, ..., qk).
// - nQubits: total system qubits
// - d_csrRowPtrU, d_csrColIndU, d_csrValU: CSR of U (size d x d, where d=2^k)
// - d_state_in/out: device statevectors, length 2^n
// - targetQubits: list of k target qubits (ascending order)
// Apply sparse gate U to given qubits (q0, q1, ..., qk)
int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    int nnzU)
{
    const int k = static_cast<int>(targetQubits.size());
    const int d = 1 << k;
    const int dim = 1 << nQubits;
    (void)dim; // not used further in this routine but kept for symmetry

    // --- 1) Sparse matrix descriptor for U ---
    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(&matU, d, d, nnzU,
                                     (void *)d_csrRowPtrU, (void *)d_csrColIndU, (void *)d_csrValU,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // --- 2) Scratch buffers for sub-block vectors ---
    cuDoubleComplex *d_block_in = nullptr;
    cuDoubleComplex *d_block_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_block_in, d * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_block_out, d * sizeof(cuDoubleComplex)));

    // --- 3) Dense vector descriptors for SpMV ---
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d, d_block_in, CUDA_C_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d, d_block_out, CUDA_C_64F));

    // --- 4) Workspace & PREPROCESS ---
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecX, &beta, vecY,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void *dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // >>> SpMV preprocess: preferred API if available, else warmup SpMV <<<

    CHECK_CUSPARSE(cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecX, &beta, vecY,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // --- 5) Prepare block iteration ---
    const int nBlocks = 1 << (nQubits - k);

    std::vector<int> nonTargetQubits;
    nonTargetQubits.reserve(nQubits - k);
    for (int q = 0; q < nQubits; ++q)
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
            nonTargetQubits.push_back(q);

    std::vector<int> h_offsets(d);
    int *d_offsets = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    // --- 6) Iterate over 2^(n-k) slices ---
    const int threads = 256;
    const int blocks = (d + threads - 1) / threads;

    for (int blk = 0; blk < nBlocks; ++blk)
    {
        int nonTargetMask = 0;
        for (size_t i = 0; i < nonTargetQubits.size(); ++i)
            if ((blk >> i) & 1)
                nonTargetMask |= (1 << nonTargetQubits[i]);

        // build offsets
        for (int b = 0; b < d; ++b)
        {
            int targetMask = 0;
            for (int q = 0; q < k; ++q)
                if ((b >> q) & 1)
                    targetMask |= (1 << targetQubits[q]);
            h_offsets[b] = nonTargetMask | targetMask;
        }

        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        // gather -> SpMV -> scatter
        gather_kernel CUDA_KERNEL(blocks, threads)(d_block_in, d_state_in, d_offsets, d);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_block_out, 0, d * sizeof(cuDoubleComplex)));
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, vecX, &beta, vecY,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        scatter_kernel CUDA_KERNEL(blocks, threads)(d_state_out, d_block_out, d_offsets, d);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- 7) Cleanup ---
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_block_in));
    CHECK_CUDA(cudaFree(d_block_out));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matU));
    return 0;
}

// Apply sparse gate with controls: skips blocks that don't satisfy control mask
int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnzU)
{
    const int k = static_cast<int>(targetQubits.size());
    const int d = 1 << k;
    const int dim = 1 << nQubits;
    (void)dim;

    // --- 1) Sparse matrix descriptor for U ---
    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(&matU, d, d, nnzU,
                                     (void *)d_csrRowPtrU, (void *)d_csrColIndU, (void *)d_csrValU,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // --- 2) Scratch buffers for sub-block vectors ---
    cuDoubleComplex *d_block_in = nullptr;
    cuDoubleComplex *d_block_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_block_in, d * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_block_out, d * sizeof(cuDoubleComplex)));

    // --- 3) Dense vector descriptors for SpMV ---
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d, d_block_in, CUDA_C_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d, d_block_out, CUDA_C_64F));

    // --- 4) Workspace & PREPROCESS ---
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecX, &beta, vecY,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void *dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecX, &beta, vecY,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // --- 5) Prepare block iteration ---
    const int nBlocks = 1 << (nQubits - k);

    std::vector<int> nonTargetQubits;
    nonTargetQubits.reserve(nQubits - k);
    for (int q = 0; q < nQubits; ++q)
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
            nonTargetQubits.push_back(q);

    int controlMask = 0;
    for (int cq : controlQubits)
        controlMask |= (1 << cq);

    std::vector<int> h_offsets(d);
    int *d_offsets = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    const int threads = 256;
    const int blocks = (d + threads - 1) / threads;

    // --- 6) Iterate over 2^(n-k) slices ---
    for (int blk = 0; blk < nBlocks; ++blk)
    {
        int nonTargetMask = 0;
        for (size_t i = 0; i < nonTargetQubits.size(); ++i)
            if ((blk >> i) & 1)
                nonTargetMask |= (1 << nonTargetQubits[i]);

        // skip if controls not satisfied
        if ((nonTargetMask & controlMask) != controlMask)
            continue;

        // offsets for this block
        for (int b = 0; b < d; ++b)
        {
            int targetMask = 0;
            for (int q = 0; q < k; ++q)
                if ((b >> q) & 1)
                    targetMask |= (1 << targetQubits[q]);
            h_offsets[b] = nonTargetMask | targetMask;
        }
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        // gather -> SpMV -> scatter
        gather_kernel CUDA_KERNEL(blocks, threads)(d_block_in, d_state_in, d_offsets, d);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_block_out, 0, d * sizeof(cuDoubleComplex)));
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, vecX, &beta, vecY,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        scatter_kernel CUDA_KERNEL(blocks, threads)(d_state_out, d_block_out, d_offsets, d);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- 7) Cleanup ---
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_block_in));
    CHECK_CUDA(cudaFree(d_block_out));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matU));
    return 0;
}
